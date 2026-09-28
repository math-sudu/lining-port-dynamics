"""Finite gain/time attribution at an identical healthy transfer function."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from .fe import RingParameters, solve_ring, recover_gauge_strain
from .network import make_layout, geometry, simulate, node_parameters
from .mechanism_study import signature_metrics, dump


def decompose_pair(strain_healthy, strain_soft, geom, layout, kappa=.3, alpha=.5):
    """Freeze tau at its healthy P4 value while leaving strain-gain active.

    This is a truth-conditioned mechanism diagnostic. The matched node times
    use the healthy simulated strain; they are not learned from external ports.
    """
    healthy_times = node_parameters(strain_healthy, layout, 'P4', alpha)['tau']
    matched = dict(layout, tau0=healthy_times)
    h = simulate(strain_healthy, geom, layout, family='P4', kappa=kappa,
                 alpha=alpha, method='expm', noise_sigma=0)['z_clean']
    soft = simulate(strain_soft, geom, layout, family='P4', kappa=kappa,
                    alpha=alpha, method='expm', noise_sigma=0)['z_clean']
    frozen = simulate(strain_soft, geom, matched, family='P3', kappa=kappa,
                      method='expm', noise_sigma=0)['z_clean']
    return {'healthy': h, 'soft': soft, 'frozen_time_soft': frozen,
            'gain': frozen-h, 'time': soft-frozen, 'total': soft-h}


def run(config_path, manifest_path, output_path):
    output = Path(output_path); output.mkdir(parents=True, exist_ok=False)
    started = datetime.now(timezone.utc).isoformat()
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    layouts = [make_layout(seed, config['nodes']) for seed in config['layout_seeds']]
    records, examples = [], {}
    first = next(row['scenario_family_id'] for row in manifest if row['split'] == 'test')
    for row in manifest:
        if row['split'] != 'test': continue
        pair = [solve_ring(RingParameters(**row['params'], reduction=y*row['positive_reduction']),
                           config['beam_elements']) for y in (0, 1)]
        for li, layout in enumerate(layouts):
            geom = geometry(layout, row['params']['radius'], row['params']['thickness'])
            strains = [recover_gauge_strain(fe, geom['theta'], geom['z'], config['gauge_length_m']) for fe in pair]
            for kappa in (0., .3):
                result = decompose_pair(*strains, geom, layout, kappa=kappa)
                gain_norm, time_norm = np.linalg.norm(result['gain']), np.linalg.norm(result['time'])
                cosine = float(np.vdot(result['gain'], result['time'])/(gain_norm*time_norm)) if gain_norm*time_norm else None
                sigma = config['receiver_noise_sigma']
                records.append({'family': row['scenario_family_id'], 'layout': li, 'kappa': kappa,
                                'gain_distance': float(gain_norm/sigma), 'time_distance': float(time_norm/sigma),
                                'gain_time_cosine': cosine,
                                'full': signature_metrics(result['healthy'], result['soft'], sigma),
                                'matched_fixed_time': signature_metrics(result['healthy'], result['frozen_time_soft'], sigma)})
                if row['scenario_family_id'] == first:
                    for key, value in result.items(): examples[f'L{li:02d}_k{kappa}_{key}'] = value
        print(f'Attributed {row["scenario_family_id"]}', flush=True)
    dump(output/'results.json', records)
    np.savez_compressed(output/'response_examples.npz', **examples)
    dump(output/'execution.json', {'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
                                   'exit_code': 0, 'config_hash': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                                   'data_fingerprint': hashlib.sha256(Path(manifest_path).read_bytes()+Path(config['source']).read_bytes()).hexdigest(),
                                   'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                                   'purpose': 'Separate finite gain/time changes without a mismatch in the healthy transfer function',
                                   'information_permission': 'private truth-conditioned mechanism control',
                                   'pairs_per_coupling': len(records)//2})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1): run(args.config, args.manifest, args.output)


if __name__ == '__main__':
    main()
