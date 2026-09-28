"""Compile fixed rail wirings using exact RC response columns and nuisance fields."""
from __future__ import annotations

import numpy as np


def record_scores(contrasts, nuisance, objective):
    """Stable scores for batches: contrasts (..., time, cases), J (..., time, p).

    Explicit residual records avoid subtracting nearly equal Gram energies
    when damage and nuisance have almost parallel receiver signatures.
    """
    if objective == "residual":
        norm = np.linalg.norm(nuisance, axis=-2, keepdims=True)
        scaled = np.divide(nuisance, norm, out=np.zeros_like(nuisance), where=norm > 0)
        left, singular, _ = np.linalg.svd(scaled, full_matrices=False)
        retain = singular > np.finfo(float).eps * max(scaled.shape[-2:]) * singular[..., :1]
        coefficients = np.swapaxes(left, -2, -1) @ contrasts
        contrasts = contrasts - left @ (retain[..., :, None] * coefficients)
    with np.errstate(divide="ignore"):
        return np.mean(np.log(np.sum(contrasts**2, axis=-2)), axis=-1)


def synthesize(contrasts, nuisance, initial_codes, objective="residual"):
    """Best improving one-bit exchanges, with the nuisance space recomputed.

    Every accepted wiring improves the exact finite-template objective. Search
    terminates at a one-bit local optimum; no claim of a global optimum is made.
    Restarts are the supplied physical comparator wirings, not fitted test data.
    """
    if objective not in {"residual", "raw"}:
        raise ValueError("objective must be residual or raw")
    damage_columns = np.transpose(contrasts, (2, 1, 0))
    nuisance_columns = np.transpose(nuisance, (2, 1, 0))
    solutions = []
    for start in initial_codes:
        code = np.asarray(start, dtype=float).copy()
        damage = np.einsum("i,itm->tm", code, damage_columns)
        jacobian = np.einsum("i,itp->tp", code, nuisance_columns)
        current = float(record_scores(damage, jacobian, objective))
        history = [current]
        while True:
            trials = damage[None] - 2 * code[:, None, None] * damage_columns
            nuisance_trials = jacobian[None] - 2 * code[:, None, None] * nuisance_columns
            trial_scores = record_scores(trials, nuisance_trials, objective)
            best = int(np.argmax(trial_scores))
            if trial_scores[best] <= current + 32 * np.finfo(float).eps * max(1, abs(current)):
                break
            code[best] *= -1
            damage = np.einsum("i,itm->tm", code, damage_columns)
            jacobian = np.einsum("i,itp->tp", code, nuisance_columns)
            current = float(record_scores(damage, jacobian, objective))
            history.append(current)
        solutions.append({"code": code, "score": current, "history": history})
    chosen = max(solutions, key=lambda item: item["score"])
    return chosen["code"], {"objective": objective, "score": chosen["score"],
                            "restarts": [{"score": s["score"], "history": s["history"]}
                                         for s in solutions]}
