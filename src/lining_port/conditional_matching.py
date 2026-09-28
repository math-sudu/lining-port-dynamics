"""Separate load and structural nuisance uncertainty using real forward solves."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
import logging
from pathlib import Path
import shutil
import subprocess
import time
import traceback

import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import yaml

from .fe import RingParameters
from .matching import (MatchingObjective, ParameterCoordinates, forward,
                       save_state, verify_saved_states)
from .network import make_layout
from .nuisance import PARAMETERS
from .pipeline import dump


class ConditionalCoordinates:
    """Optimize selected normalized coordinates, preserving all others exactly."""

    def __init__(self, base, bounds, free_parameters):
        self.base = base
        self.full = ParameterCoordinates(base, bounds)
        self.parameter_names = tuple(free_parameters)
        self.indices = np.array([PARAMETERS.index(p) for p in self.parameter_names])
        self.bounds = tuple(a[self.indices] for a in self.full.bounds)
        self.scales = np.r_[self.full.span, np.pi][self.indices]
        self.fixed_parameters = tuple(p for p in PARAMETERS if p not in self.parameter_names)

    def encode(self, params):
        return self.full.encode(params)[self.indices]

    def decode(self, coordinates):
        full_coordinates = self.full.encode(self.base)
        full_coordinates[self.indices] = coordinates
        candidate = self.full.decode(full_coordinates)
        # Retain exact original values (including phase) for the fixed group.
        return replace(self.base, **{p: getattr(candidate, p) for p in self.parameter_names},
                       reduction=0.)

    def clip(self, coordinates):
        return np.clip(coordinates, *self.bounds)


def initial_points(coords, tangent, seed):
    """Use the reduced Jacobian, not a slice of the six-parameter solution."""
    healthy = coords.encode(coords.base)
    jacobian = tangent['nuisance_jacobian'][:, coords.indices] * coords.scales
    delta = tangent['softened'] - tangent['healthy']
    increment, _, rank, singular_values = np.linalg.lstsq(jacobian, delta, rcond=1e-6)
    points = {'healthy': healthy, 'clipped_linear': coords.clip(healthy + increment)}
    rng = np.random.default_rng(seed)
    for j in range(2):
        point = rng.uniform(size=len(coords.parameter_names))
        if 'phase' in coords.parameter_names:
            point[coords.parameter_names.index('phase')] -= .5
        points[f'random_{j}'] = point
    return points, {'normalized_linear_increment': increment.tolist(), 'rank': int(rank),
                    'singular_values': singular_values.tolist(), 'rcond': 1e-6}


def verify_conditions(output):
    """Check fixed parameters and observation access from the saved raw states."""
    output = Path(output)
    plan = json.loads((output / 'plan.json').read_text())
    records = json.loads((output / 'results.json').read_text())
    checks = []
    for record in records:
        path = output / record['candidate_file']
        with np.load(path, allow_pickle=False) as candidate, np.load(
                path.parent / 'softened_target.npz', allow_pickle=False) as target:
            params = json.loads(str(candidate['params_json']))
            target_params = json.loads(str(target['params_json']))
            free = plan['conditions'][record['condition']]
            fixed = [p for p in PARAMETERS if p not in free]
            checks.append({
                'scenario_family_id': record['scenario_family_id'], 'model': record['model'],
                'condition': record['condition'], 'start': record['start'],
                'fixed_parameter_differences': {p: params[p] - target_params[p] for p in fixed},
                'fixed_parameters_exact': all(params[p] == target_params[p] for p in fixed),
                'same_interrogation': bool(np.array_equal(candidate['q'], target['q'])),
                'same_times': bool(np.array_equal(candidate['time'], target['time'])),
                'external_sequence_shape': list(candidate['z'].shape),
                'training_group_only': record['split'] == 'train',
                'layout_id': record['layout_id'],
            })
    passed = bool(checks) and all(
        r['fixed_parameters_exact'] and r['same_interrogation'] and r['same_times']
        and r['external_sequence_shape'] == [400, 4] and r['training_group_only']
        and r['layout_id'] == plan['layout_id'] for r in checks)
    report = {'pass': passed, 'candidate_count': len(checks), 'checks': checks}
    dump(output / 'condition_checks.json', report)
    if not passed:
        raise AssertionError('Saved conditional fit changed fixed parameters or observations')
    return report


def run(config_path, output):
    config_path = Path(config_path).resolve()
    root = config_path.parent
    config = yaml.safe_load(config_path.read_text())
    source = root / config['source']
    previous_plan = json.loads((source / 'plan.json').read_text())
    previous_verification = json.loads((source / 'verification.json').read_text())
    manifest = json.loads((root / previous_plan['source'] / 'split_manifest.json').read_text())
    rows = [r for r in manifest if r['scenario_family_id'] in config['families']]
    if {r['scenario_family_id'] for r in rows} != set(config['families']):
        raise ValueError('Requested training family is missing')
    if any(r['split'] != 'train' for r in rows):
        raise ValueError('Conditional matching uses training families only')
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s', handlers=[
        logging.FileHandler(output / 'run.log', encoding='utf-8'), logging.StreamHandler()])
    plan = {**config, **{k: previous_plan[k] for k in
                        ('bounds', 'layout_seed', 'layout_id', 'sigma', 'elements',
                         'verification_elements', 'tangent_source')},
            'training_groups': rows,
            'initialization': 'Reduced normalized Jacobian; two uniform starts per group, shared across models',
            'phase_rule': 'Optimize periodic phase only when free; retain exact original phase when fixed',
            'optimization_scope': 'Local multistart search; no global minimum certificate'}
    dump(output / 'plan.json', plan)
    dump(output / 'provenance.json', {
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'scipy': importlib.metadata.version('scipy'), 'numpy': np.__version__,
        'source_git_commit': json.loads((source / 'provenance.json').read_text())['git_commit']})
    started = time.perf_counter()
    layout = make_layout(plan['layout_seed'])
    records, verifications, failures = [], [], []
    total_evaluations = 0
    for row in rows:
        family = row['scenario_family_id']
        base = RingParameters(**row['params'], reduction=0.)
        for model in config['models']:
            source_folder = source / f'{family}_{model}'
            with np.load(source_folder / 'softened_target.npz', allow_pickle=False) as saved:
                target_params = RingParameters(**json.loads(str(saved['params_json'])))
                target_sequence = saved['z'].ravel().copy()
            target = forward(target_params, layout, model, plan['elements'])
            total_evaluations += 1
            source_error = float(np.max(np.abs(target['z'] - target_sequence)))
            if source_error > 1e-10:
                raise AssertionError(f'Third-round target replay differs: {family}/{model}: {source_error}')
            with np.load(root / plan['tangent_source'] / f'{family}_{model}_step0.5.npz') as data:
                tangent = {k: data[k] for k in ('nuisance_jacobian', 'softened', 'healthy')}
            previous = next(r for r in previous_verification
                            if r['scenario_family_id'] == family and r['model'] == model)
            for condition_index, (condition, free) in enumerate(config['conditions'].items()):
                coords = ConditionalCoordinates(base, plan['bounds'], free)
                folder = output / f'{family}_{model}_{condition}'
                folder.mkdir()
                save_state(folder / 'softened_target.npz', target)
                seed = config['random_seed'] + int(family[1:]) + 1000 * condition_index
                starts, linear = initial_points(coords, tangent, seed)
                dump(folder / 'initialization.json', dict(linear, seed=seed,
                     free_parameters=free, coordinates={k: v.tolist() for k, v in starts.items()}))
                group_records = []
                for name in config['starts']:
                    timer = time.perf_counter()
                    with (folder / f'{name}_trace.jsonl').open('w', encoding='utf-8') as trace:
                        obj = MatchingObjective(coords, layout, model, target['z'], plan['elements'],
                                                plan['sigma'], trace, config['optimizer']['finite_difference_step'])
                        try:
                            initial = obj.evaluate(starts[name])
                            save_state(folder / f'{name}_initial.npz', initial)
                            settings = {k: v for k, v in config['optimizer'].items()
                                        if k != 'finite_difference_step'}
                            fit = least_squares(obj.fun, starts[name], jac=obj.jac, bounds=coords.bounds,
                                                x_scale=1., **settings)
                            candidate = obj.evaluate(fit.x)
                            save_state(folder / f'{name}_candidate.npz', candidate)
                            distance = float(np.linalg.norm(candidate['residual']))
                            record = {
                                'scenario_family_id': family, 'split': row['split'], 'layout_id': plan['layout_id'],
                                'model': model, 'condition': condition, 'start': name, 'free_parameters': free,
                                'initial_parameters': asdict(initial['params']), 'parameters': asdict(candidate['params']),
                                'initial_distance': float(np.linalg.norm(initial['residual'])), 'distance': distance,
                                'known_pair_gaussian_BA': float(ndtr(distance / 2)),
                                'optimizer_success': bool(fit.success), 'optimizer_status': int(fit.status),
                                'stop_reason': fit.message, 'optimality': float(fit.optimality),
                                'optimizer_nfev': int(fit.nfev), 'optimizer_njev': int(fit.njev),
                                'FE_network_evaluations': obj.evaluations,
                                'active_bounds': {p: int(v) for p, v in zip(free, fit.active_mask)},
                                'topologies_visited': len(obj.catalog),
                                'topology_changed_from_target': candidate['topology'] != target['topology'],
                                'one_sided_derivatives': obj.one_sided_derivatives,
                                'topology_derivative_adjustments': obj.topology_derivative_adjustments,
                                'wall_seconds': time.perf_counter() - timer,
                                'candidate_file': (folder / f'{name}_candidate.npz').relative_to(output).as_posix()}
                            records.append(record)
                            group_records.append(record)
                            logging.info('%s %s %s %s: d=%.6g, status=%d, %d FE calls, %.1f s',
                                         family, model, condition, name, distance, fit.status,
                                         obj.evaluations, record['wall_seconds'])
                        except Exception as exc:
                            failures.append({'scenario_family_id': family, 'model': model, 'condition': condition,
                                             'start': name, 'error': repr(exc), 'traceback': traceback.format_exc()})
                            logging.exception('Conditional fit failed: %s/%s/%s/%s', family, model, condition, name)
                        finally:
                            total_evaluations += obj.evaluations
                            dump(folder / f'{name}_topologies.json', obj.catalog)
                            dump(output / 'results.json', records)
                            dump(output / 'failures.json', failures)
                if not group_records:
                    continue
                best = min(group_records, key=lambda r: r['distance'])
                params = RingParameters(**best['parameters'])
                replay = forward(params, layout, model, plan['elements'])
                fine = forward(params, layout, model, plan['verification_elements'])
                adaptive = forward(params, layout, model, plan['elements'], 'adaptive')
                total_evaluations += 3
                save_state(folder / 'best_fine_candidate.npz', fine)
                save_state(folder / 'best_adaptive_candidate.npz', adaptive)
                for name in ('best_fine_target.npz', 'best_adaptive_target.npz'):
                    shutil.copyfile(source_folder / name, folder / name)
                with np.load(folder / 'best_fine_target.npz') as fine_target, np.load(
                        folder / 'best_adaptive_target.npz') as adaptive_target, np.load(
                        output / best['candidate_file']) as retained:
                    residual_error = ((adaptive_target['z'].ravel() - adaptive['z'])
                                      - (target['z'] - replay['z']))
                    verification = {
                        'scenario_family_id': family, 'model': model, 'condition': condition,
                        'best_start': best['start'], 'best_distance': best['distance'],
                        'all_start_distances': {r['start']: r['distance'] for r in group_records},
                        'fine_mesh_distance': float(np.linalg.norm(fine['z'] - fine_target['z'].ravel()) / plan['sigma']),
                        'adaptive_distance': float(np.linalg.norm(adaptive['z'] - adaptive_target['z'].ravel()) / plan['sigma']),
                        'integration_residual_difference_noise_units': float(np.linalg.norm(residual_error) / plan['sigma']),
                        'candidate_replay_max_error': float(np.max(np.abs(replay['z'] - retained['z'].ravel()))),
                        'target_replay_max_error': source_error,
                        'all_free_distance_round3': previous['best_distance'],
                        'conditional_minus_all_free_distance': best['distance'] - previous['best_distance'],
                        'all_free_fine_mesh_distance_round3': previous['fine_mesh_distance'],
                        'retained_optimizer_success': best['optimizer_success']}
                verifications.append(verification)
                dump(output / 'verification.json', verifications)
    verify_saved_states(output)
    verify_conditions(output)
    nonconverged = sum(not r['optimizer_success'] for r in records)
    dump(output / 'completion.json', {
        'status': 'completed' if not failures and not nonconverged else 'completed_with_failures',
        'candidate_count': len(records), 'failed_fits': len(failures), 'optimizer_nonconverged_count': nonconverged,
        'FE_solves': total_evaluations, 'network_solves': total_evaluations,
        'wall_seconds': time.perf_counter() - started, 'all_families_from_training': True})
    return records, verifications


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='conditional_matching_config.yaml')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.config, args.output)


if __name__ == '__main__':
    main()
