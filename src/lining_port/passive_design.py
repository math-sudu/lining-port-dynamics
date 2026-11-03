"""Regional connection design within wire and gauge-dissipation budgets."""
from dataclasses import replace
from itertools import combinations

import numpy as np
from scipy.optimize import brentq, minimize

from .passive_acquisition_study import refine_candidate, routed_candidate
from .passive_bridge import ARM_SIGN, TrimmedBridge, response_spaces
from .passive_hardware import route_bridge
from .passive_network_study import attenuation_cap
from .passive_resources import design_ratio, matched_power_network, minimum_arm_units


def design_gauge_peaks(hardware, healthy, changed):
    """One peak per channel, over all its gauges and prescribed design states."""
    return np.array([max(h.operating_point(healthy)["gauge_power_w"].max(),
                         h.operating_point(changed)["gauge_power_w"].max())
                     for h in hardware])


def power_candidate(network, units, circuit, routes, config, budget):
    scaled, supply, hardware = matched_power_network(network, units, circuit, routes,
        config["differential_input_resistance_ohm"], budget["source_power_w"])
    return {"network": scaled, "circuit": supply, "routes": routes, "hardware": hardware}


def enforce_gauge_limit(network, units, circuit, routes, healthy, changed, config, budget):
    """Uniformly increase arm scales when needed, preserving nominal source power."""
    def excess(factor):
        current = power_candidate(network, units * factor, circuit, routes, config, budget)
        return design_gauge_peaks(current["hardware"], healthy, changed).max() / budget["gauge_peak_power_w"] - 1

    if excess(1.) <= 0:
        return units
    upper = 2.
    while excess(upper) > 0:
        upper *= 2
    factor = brentq(excess, 1., upper, xtol=1e-12)
    return units * factor * (1 + 1e-11)


def constrained_scales(network, circuit, routes, healthy, changed, config,
                       acquisition, budget):
    """Fit positive arm scales at fixed total source power and bounded gauge power.

    The common supply follows the exact nominal source-power equality. All
    gauges and healthy/changed design loads enter the dissipation inequalities.
    """
    length = sum(r["total_conductor_m"] for r in routes)
    if length > budget["conductor_length_m"] + 1e-10:
        return None
    lower = np.array([minimum_arm_units(b, circuit, r) for b, r in zip(network, routes)])
    limit = budget["gauge_peak_power_w"]

    def candidate(units):
        return power_candidate(network, units, circuit, routes, config, budget)

    def feasible_units(units):
        return enforce_gauge_limit(network, units, circuit, routes, healthy, changed, config, budget)

    initial = feasible_units(np.maximum(lower, [b.arm_units for b in network]))
    cache = {}

    def values(x):
        key = tuple(x)
        if key not in cache:
            current = candidate(lower * np.exp(x))
            value = -design_ratio(current["hardware"], healthy, changed, config, acquisition)[0]
            peaks = design_gauge_peaks(current["hardware"], healthy, changed) / limit
            cache[key] = np.r_[value, 1 - peaks]
        return cache[key]

    def derivative(x):
        # Microvolt differences make machine-size voltage finite differences
        # unsuitable; a centered log-scale perturbation resolves these slopes.
        step = 1e-3
        columns = []
        for i in range(len(x)):
            plus, minus = x.copy(), x.copy()
            plus[i] += step
            minus[i] = max(0., x[i] - step)
            columns.append((values(plus) - values(minus)) / (plus[i] - minus[i]))
        return np.stack(columns, axis=-1)

    start = np.log(initial / lower)
    fit = minimize(lambda x: values(x)[0], start, jac=lambda x: derivative(x)[0],
                   method="SLSQP", bounds=[(0., None)] * len(network),
                   constraints={"type": "ineq", "fun": lambda x: values(x)[1:],
                                "jac": lambda x: derivative(x)[1:]},
                   options={"ftol": 1e-7, "maxiter": 150})
    if not fit.success:
        raise RuntimeError(f"Resource-constrained scale fit failed: {fit.message}")
    fitted = feasible_units(lower * np.exp(fit.x))
    choices = [candidate(initial), candidate(fitted)]
    selected = max(choices, key=lambda c: design_ratio(c["hardware"], healthy, changed, config, acquisition)[0])
    selected["scale_fit"] = {
        "iterations": int(fit.nit), "minimum_arm_units": lower.tolist(),
        "initial_arm_units": initial.tolist(),
        "selected_arm_units": [b.arm_units for b in selected["network"]],
    }
    return selected


def arm_neighbors(bridge):
    """Move one active cell or exchange two active cells between equal-sign arms."""
    active = np.flatnonzero(bridge.attenuation > 0)
    for i in active:
        arms = bridge.arm.copy()
        arms[i] = 3 - arms[i]
        yield replace(bridge, arm=arms, arm_units=float(len(arms)))
    for i, j in combinations(active, 2):
        if bridge.arm[i] != bridge.arm[j] and ARM_SIGN[bridge.arm[i]] == ARM_SIGN[bridge.arm[j]]:
            arms = bridge.arm.copy()
            arms[i], arms[j] = arms[j], arms[i]
            yield replace(bridge, arm=arms, arm_units=float(len(arms)))


def route(network, circuit, geometry, depth, config):
    cell, instrument, center = geometry
    return [route_bridge(b, cell, center, instrument, depth,
                         config["wire_resistance_ohm_per_m"]) for b in network]


def allocate_arms(network, circuit, geometry, depth, budget, config, criterion):
    """Greedy physical reallocation with fixed coefficients and group membership.

    A length pass shortens over-budget starts. A capacity pass reduces the
    physical lower bound using only assignments inside the total wire budget.
    Neither proxy selects the final network; full-circuit design does that.
    """
    network = list(network)
    routes = route(network, circuit, geometry, depth, config)
    moves = []
    for q in range(len(network)):
        while True:
            old = network[q]
            old_route = routes[q]
            old_lower = minimum_arm_units(old, circuit, old_route)
            other_length = sum(r["total_conductor_m"] for k, r in enumerate(routes) if k != q)
            best, best_route, best_lower = old, old_route, old_lower
            for trial in arm_neighbors(old):
                trial_route = route([trial], circuit, geometry, depth, config)[0]
                trial_lower = minimum_arm_units(trial, circuit, trial_route)
                if criterion == "capacity":
                    if other_length + trial_route["total_conductor_m"] > budget["conductor_length_m"] + 1e-10:
                        continue
                    improve = trial_lower < best_lower - 1e-10
                elif criterion == "length":
                    improve = trial_route["total_conductor_m"] < best_route["total_conductor_m"] - 1e-10
                else:
                    raise ValueError("Arm allocation criterion must be capacity or length")
                if improve:
                    best, best_route, best_lower = trial, trial_route, trial_lower
            if best is old:
                break
            network[q] = replace(best, arm_units=max(old.arm_units, best_lower))
            routes[q] = best_route
            moves.append({"channel": q, "gauges": old.gauge_indices[old.arm != best.arm].tolist(),
                          "old_arms": old.arm[old.arm != best.arm].tolist(),
                          "new_arms": best.arm[old.arm != best.arm].tolist()})
    return network, routes, moves


def physical_candidates(network, circuit, geometry, depth, healthy, changed, config, acquisition, budget):
    """Compare original arms, shorter routes and lower arm occupancy."""
    routes = route(network, circuit, geometry, depth, config)
    variants = [("retained_arms", network, routes, [])]
    shorter, short_routes, short_moves = allocate_arms(
        network, circuit, geometry, depth, budget, config, "length")
    variants.append(("shorter_arms", shorter, short_routes, short_moves))
    for label, start, _, trace in list(variants):
        balanced, balance_routes, moves = allocate_arms(
            start, circuit, geometry, depth, budget, config, "capacity")
        variants.append((label + "_balanced", balanced, balance_routes, trace + moves))
    candidates, seen = [], set()
    for label, current, current_routes, moves in variants:
        signature = tuple(np.concatenate([b.arm for b in current]))
        if signature in seen:
            continue
        seen.add(signature)
        fitted = constrained_scales(current, circuit, current_routes, healthy, changed, config, acquisition, budget)
        if fitted is not None:
            fitted["arm_method"] = label
            fitted["arm_moves"] = moves
            candidates.append(fitted)
    return candidates


def refine_arms(candidate, geometry, depth, healthy, changed, config, acquisition, budget):
    """One channel sweep of exact-objective same-sign moves and exchanges."""
    score = lambda c: design_ratio(c["hardware"], healthy, changed, config, acquisition)[0]
    current = candidate
    moves = list(candidate.get("arm_moves", []))
    for q in range(len(candidate["network"])):
        best, best_score = current, score(current)
        circuit, routes = current["circuit"], current["routes"]
        original_units = np.array([b.arm_units for b in current["network"]])
        for bridge in arm_neighbors(current["network"][q]):
            trial_routes = list(routes)
            trial_routes[q] = route([bridge], circuit, geometry, depth, config)[0]
            if sum(r["total_conductor_m"] for r in trial_routes) > budget["conductor_length_m"] + 1e-10:
                continue
            network = list(current["network"])
            network[q] = bridge
            lower = minimum_arm_units(bridge, circuit, trial_routes[q])
            for value in set((max(original_units[q], lower), lower)):
                units = original_units.copy()
                units[q] = value
                units = enforce_gauge_limit(network, units, circuit, trial_routes, healthy, changed, config, budget)
                trial = power_candidate(network, units, circuit, trial_routes, config, budget)
                trial_score = score(trial)
                if trial_score > best_score:
                    best, best_score = trial, trial_score
        if best is not current:
            old, new = current["network"][q], best["network"][q]
            mask = old.arm != new.arm
            moves.append({"channel": q, "gauges": old.gauge_indices[mask].tolist(),
                          "old_arms": old.arm[mask].tolist(), "new_arms": new.arm[mask].tolist()})
        current = best
    fitted = constrained_scales(current["network"], current["circuit"], current["routes"],
                                healthy, changed, config, acquisition, budget)
    best = max([candidate, current, fitted], key=score)
    if best is candidate:
        return candidate
    best.update({"arm_method": candidate.get("arm_method", "retained_arms") + "_exact_sweep",
                 "arm_moves": moves, "origin": candidate["origin"],
                 "proposal": candidate.get("proposal", {})})
    return best


def projection_swaps(network, response, increments, spatial_order):
    """Best mechanically retained-energy exchange for each pair of channels.

    Each proposal is screened in the same commissioned coordinates. The
    geometric mean of the total grouped projection favors poorly retained
    regions; it is a proposal score, not a voltage prediction.
    """
    rank = np.argsort(spatial_order)
    groups = [b.gauge_indices for b in network]

    def energy(group):
        null = response_spaces(response[group])[1]
        return np.sum((increments[..., group] @ null)**2, axis=(1, 2))

    energies = np.stack([energy(g) for g in groups])
    total = energies.sum(0)
    score = float(np.mean(np.log(total)))
    proposals = []
    for a, b in combinations(range(len(groups)), 2):
        best_score, best = score, None
        for i, ga in enumerate(groups[a]):
            for j, gb in enumerate(groups[b]):
                left, right = groups[a].copy(), groups[b].copy()
                left[i], right[j] = gb, ga
                candidate_score = float(np.mean(np.log(total - energies[a] - energies[b] + energy(left) + energy(right))))
                if candidate_score > best_score:
                    best_score, best = candidate_score, (int(ga), int(gb), left, right)
        if best is not None:
            ga, gb, left, right = best
            new_groups = list(groups)
            new_groups[a], new_groups[b] = left[np.argsort(rank[left])], right[np.argsort(rank[right])]
            proposals.append({"channels": [a, b], "gauges": [ga, gb], "groups": new_groups,
                              "projection_log_gain": best_score - score})
    return proposals


def regroup_candidate(network, proposal, response, healthy, changed, ratio,
                      circuit, geometry, depth, config, acquisition, budget):
    """Project the retained weights into changed group null spaces and refine."""
    full_weight = np.zeros(healthy.shape[-1])
    for bridge in network:
        full_weight[bridge.gauge_indices] = bridge.weight
    initial = []
    for q, group in enumerate(proposal["groups"]):
        if q not in proposal["channels"]:
            initial.append(replace(network[q], arm_units=float(np.bincount(network[q].arm, minlength=4).max())))
            continue
        null = response_spaces(response[group])[1]
        weight = null @ (null.T @ full_weight[group])
        bridge = TrimmedBridge.from_weight(group, weight)
        cap = attenuation_cap(ratio[group], circuit, config["max_shunt_ohm"])
        initial.append(replace(bridge, attenuation=cap * bridge.attenuation, nominal_ratio=ratio[group]))
    candidate = routed_candidate(initial, circuit, geometry, depth, config, budget["source_power_w"])
    fitted, info = refine_candidate(candidate, response, healthy, changed, config, acquisition)
    return fitted, info
