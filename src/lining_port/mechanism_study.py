"""Controlled finite-port study; every invocation creates a new result directory."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import time

import numpy as np
import yaml
from threadpoolctl import threadpool_limits

from .fe import RingParameters, solve_ring, recover_gauge_strain, recover_displacement
from .network import (make_layout, geometry, coverage, simulate,
                      strain_directional_response)
from .readout import run_readout, metrics, grouped_interval
from .continuum import solve_continuum, point_fields, recover_continuum_gauge


def dump(path, value):
    def convert(item):
        return item.tolist() if isinstance(item, np.ndarray) else item.item()
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')


def signature_metrics(healthy, softened, sigma):
    delta = softened-healthy
    denom = np.sum(healthy**2, axis=0)
    gain = np.divide(np.sum(healthy*softened, axis=0), denom,
                     out=np.ones(healthy.shape[1]), where=denom > 0)
    shape_residual = softened-healthy*gain
    return {'distance': float(np.linalg.norm(delta)/sigma),
            'terminal_distance': float(np.linalg.norm(delta[-1])/sigma),
            'shape_distance': float(np.linalg.norm(shape_residual)/sigma),
            'port_gain': gain.tolist()}


def nuisance_projection(delta, jacobian, sigma):
    norms = np.linalg.norm(jacobian, axis=0)
    scaled = np.divide(jacobian, norms, out=np.zeros_like(jacobian), where=norms > 0)
    coefficients, _, rank, singular = np.linalg.lstsq(scaled, delta, rcond=None)
    residual = delta-scaled@coefficients
    return {'distance': float(np.linalg.norm(delta)/sigma),
            'residual_distance': float(np.linalg.norm(residual)/sigma),
            'rank': int(rank), 'singular_values': singular.tolist(),
            'parameter_coefficients': np.divide(coefficients, norms, out=np.zeros_like(norms), where=norms > 0).tolist()}


def subset(data, take):
    count = len(data['y'])
    return {k: (v[take] if k not in {'q', 'time'} and len(v) == count else v)
            for k, v in data.items()}


def generate(config, manifest, output):
    layouts = [make_layout(seed, config['nodes']) for seed in config['layout_seeds']]
    count, nl = len(manifest)*len(layouts)*2, len(layouts)
    observations = {name: np.empty((count, 400, 4), dtype=np.float32) for name in config['physical_models']}
    masks = np.empty((count, 4), dtype=bool)
    records, coverage_rows, signatures, examples, structural = [], [], [], {}, []
    # Only test templates are needed by the subsequent truth-conditioned analysis.
    clean = {}
    for fi, row in enumerate(manifest):
        pair = [solve_ring(RingParameters(**row['params'], reduction=label*row['positive_reduction']),
                           config['beam_elements']) for label in (0, 1)]
        for label, fe in enumerate(pair):
            structural.append({'family': row['scenario_family_id'], 'label': label,
                               'params': asdict(fe.params), 'diagnostics': fe.diagnostics})
        for li, layout in enumerate(layouts):
            geo = geometry(layout, row['params']['radius'], row['params']['thickness'])
            strains = [recover_gauge_strain(fe, geo['theta'], geo['z'], config['gauge_length_m']) for fe in pair]
            coverage_rows.append({'family': row['scenario_family_id'], 'split': row['split'], 'layout': li,
                                  **geo['diagnostics'], **coverage(geo, row['params']['sector_center'], row['params']['sector_width'])})
            for label in (0, 1):
                index = (fi*nl+li)*2+label
                masks[index] = geo['mask']
                records.append({'sample_id': f'{row["scenario_family_id"]}_L{li:02d}_y{label}',
                                'scenario_family_id': row['scenario_family_id'], 'layout_id': f'L{li:02d}',
                                'layout_index': li, 'split': row['split'], 'y': label,
                                'noise_seed': row['acquisition_noise_seeds'][li][label]})
            for name, settings in config['physical_models'].items():
                responses = []
                for label in (0, 1):
                    sim = simulate(strains[label], geo, layout, **settings, method='expm',
                                   noise_seed=row['acquisition_noise_seeds'][li][label],
                                   noise_sigma=config['receiver_noise_sigma'])
                    observations[name][(fi*nl+li)*2+label] = sim['z']
                    responses.append(sim['z_clean'])
                if row['split'] == 'test':
                    clean[(fi, li, name)] = responses
                    signatures.append({'family': row['scenario_family_id'], 'layout': li, 'model': name,
                                       **signature_metrics(*responses, config['receiver_noise_sigma'])})
                    if fi == next(i for i, r in enumerate(manifest) if r['split'] == 'test'):
                        examples[f'{name}_L{li:02d}'] = np.stack(responses)
            if row['split'] == 'test' and fi == next(i for i, r in enumerate(manifest) if r['split'] == 'test'):
                components = strain_directional_response(strains[0], strains[1]-strains[0], geo, layout)
                examples[f'strain_L{li:02d}'] = np.stack(strains)
                for key, value in components.items(): examples[f'tangent_{key}_L{li:02d}'] = value
        if (fi+1)%16 == 0: logging.info('Generated %d/%d families on %d layouts', fi+1, len(manifest), nl)
    dump(output/'observation_manifest.json', records)
    dump(output/'coverage.json', coverage_rows)
    dump(output/'structural_diagnostics.json', structural)
    dump(output/'signatures.json', signatures)
    np.savez_compressed(output/'response_examples.npz', **examples)
    common = {k: np.asarray([r[k] for r in records]) for k in
              ('sample_id', 'scenario_family_id', 'layout_id', 'layout_index', 'split', 'y', 'noise_seed')}
    common.update(mask=masks, time=sim['time'], q=sim['q'])
    return layouts, common, observations, clean


def readouts(config, common, observations, output):
    results = {}
    domains = config['layout_domains']
    transfer_take = np.zeros(len(common['y']), dtype=bool)
    for split, domain in [('train', 'source_train'), ('validation', 'source_validation'), ('test', 'target_test')]:
        transfer_take |= (common['split'] == split)&np.isin(common['layout_index'], domains[domain])
    intervals = {}
    for feature in config['readouts']['features']:
        source_scores, source_thresholds = {}, {}
        for name, z in observations.items():
            data = dict(common, z=z if feature == 'sequence' else z[:, -1:, :])
            source = subset(data, transfer_take)
            folder = output/'readouts'/'source_to_unseen_layouts'/feature/name
            report, score, threshold = run_readout(source, folder, config['readouts']['ridge_alphas'], permutations=0)
            source_scores[name], source_thresholds[name] = score, threshold
            results[f'source_to_unseen_layouts/{feature}/{name}'] = report
            for li in range(len(config['layout_seeds'])):
                local = subset(data, data['layout_index'] == li)
                key = f'within_each_layout/{feature}/{name}/L{li:02d}'
                if not local['mask'].any():
                    results[key] = {'status': 'rejected_no_ports', 'coverage': 0., 'layout': li}
                    continue
                report, _, _ = run_readout(local, output/'readouts'/key,
                                           config['readouts']['ridge_alphas'], permutations=0, layout_policy='seen')
                results[key] = report
            logging.info('Readouts complete: %s / %s', name, feature)
        intervals[feature] = grouped_interval(source, source_scores, source_thresholds,
                                              repetitions=1000, seed=config['family_seed']+54)
    dump(output/'readout_results.json', results)
    dump(output/'readout_intervals.json', intervals)
    return results


def nuisance_analysis(config, manifest, layouts, clean, output):
    rows = []
    parameters = config['mechanism']['nuisance_parameters']
    sigma = config['receiver_noise_sigma']
    target_layouts = config['layout_domains']['target_test']
    for fi, row in enumerate(manifest):
        if row['split'] != 'test': continue
        p = RingParameters(**row['params'], reduction=0.)
        for factor in config['mechanism']['finite_difference_step_factors']:
            perturbed = {}
            for key in parameters:
                step_spec = config['mechanism']['finite_difference_steps']
                step = factor*(step_spec[key] if key in ('harmonic', 'phase') else step_spec['relative']*getattr(p, key))
                # A centered harmonic perturbation can cross zero. Negative harmonic
                # is the same physical load family with phase shifted by pi/2.
                perturbed[key] = (step, [solve_ring(replace(p, **{key: getattr(p, key)+sign*step}),
                                                    config['beam_elements']) for sign in (-1, 1)])
            for li in target_layouts:
                layout = layouts[li]
                base_geo = geometry(layout, p.radius, p.thickness)
                by_model = {name: [] for name in config['mechanism']['nuisance_models']}
                changed = []
                for key in parameters:
                    step, pair = perturbed[key]
                    pairs = {name: [] for name in by_model}
                    for fe in pair:
                        geo = geometry(layout, fe.params.radius, fe.params.thickness)
                        topology_changed = (not np.array_equal(geo['W'] > 0, base_geo['W'] > 0)
                                            or not np.array_equal(geo['port_nodes'], base_geo['port_nodes']))
                        if topology_changed: changed.append(key)
                        strain = recover_gauge_strain(fe, geo['theta'], geo['z'], config['gauge_length_m'])
                        for name in by_model:
                            pairs[name].append(simulate(strain, geo, layout, **config['physical_models'][name],
                                                       method='expm', noise_sigma=0)['z_clean'].ravel())
                    for name in by_model: by_model[name].append((pairs[name][1]-pairs[name][0])/(2*step))
                for name, columns in by_model.items():
                    jac = np.column_stack(columns)
                    ip, ik, it, ih, ia = [parameters.index(k) for k in ('pressure', 'radial_modulus', 'thickness', 'harmonic', 'phase')]
                    # The E direction follows exactly from the mechanical scaling identity.
                    j_e = -(p.pressure*jac[:, ip]+p.radial_modulus*jac[:, ik])/p.youngs
                    variants = {'all_unknown': jac, 'structure_known': jac[:, [ip, ih, ia]],
                                'load_known': np.column_stack((j_e, jac[:, ik], jac[:, it]))}
                    pair = clean[(fi, li, name)]
                    delta = (pair[1]-pair[0]).ravel()
                    for condition, basis in variants.items():
                        rows.append({'family': row['scenario_family_id'], 'layout': li, 'model': name,
                                     'condition': condition, 'step_factor': factor,
                                     'topology_changed_parameters': sorted(set(changed)),
                                     **nuisance_projection(delta, basis, sigma)})
        logging.info('Nuisance directions complete: %s', row['scenario_family_id'])
    dump(output/'nuisance_results.json', rows)


def functional_analysis(config, manifest, layouts, output):
    reports = []
    variants = {'algebraic': {'strain_law': 'algebraic'}, 'transfer_0p7': {'strain_transfer': .7}}
    for variant, options in variants.items():
        z_by_model = {name: [] for name in ('P3_coupled', 'P4_coupled')}
        metadata, signatures = [], []
        for row in manifest:
            if row['split'] != 'test': continue
            pair = [solve_ring(RingParameters(**row['params'], reduction=y*row['positive_reduction']),
                               config['beam_elements']) for y in (0, 1)]
            for li in config['layout_domains']['target_test']:
                layout = layouts[li]; geo = geometry(layout, row['params']['radius'], row['params']['thickness'])
                strains = [recover_gauge_strain(fe, geo['theta'], geo['z'], config['gauge_length_m']) for fe in pair]
                for y in (0, 1): metadata.append({'y': y, 'family': row['scenario_family_id'], 'layout': li, 'mask': geo['mask'].tolist()})
                for name in z_by_model:
                    sims = [simulate(strains[y], geo, layout, **config['physical_models'][name], **options,
                                     method='expm', noise_sigma=config['receiver_noise_sigma'],
                                     noise_seed=row['acquisition_noise_seeds'][li][y]) for y in (0, 1)]
                    z_by_model[name].extend(s['z'].astype(np.float32) for s in sims)
                    signatures.append({'family': row['scenario_family_id'], 'layout': li, 'model': name,
                                       **signature_metrics(sims[0]['z_clean'], sims[1]['z_clean'], config['receiver_noise_sigma'])})
        mask = np.array([r['mask'] for r in metadata]); y = np.array([r['y'] for r in metadata])
        eligible = mask.any(axis=1)
        results = {}
        for name, sequences in z_by_model.items():
            z = np.asarray(sequences)
            for feature in config['readouts']['features']:
                frozen = np.load(output/'readouts'/'source_to_unseen_layouts'/feature/name/'model.npz')
                x = np.concatenate(((z if feature == 'sequence' else z[:, -1:, :]).reshape(len(z), -1), mask), axis=1)
                score = ((x-frozen['mean'])/frozen['scale'])@frozen['coef']+frozen['intercept']
                results[f'{feature}/{name}'] = metrics(y[eligible], score[eligible], float(frozen['threshold']))
        reports.append({'variant': variant, 'options': options, 'frozen_source_readouts': results,
                        'coverage': float(eligible.mean()), 'signatures': signatures})
        logging.info('Functional control complete: %s', variant)
    dump(output/'functional_results.json', reports)


def field_validation(config, layouts, output):
    cases = {
        'nominal_port_aligned': RingParameters(reduction=.45, sector_center=0.),
        'nominal_between_ports': RingParameters(reduction=.45, sector_center=np.pi/4),
        'low_load_thin_ring': RingParameters(pressure=1e5, thickness=.27, youngs=33e9, radial_modulus=60e6,
                                             harmonic=0., reduction=.3, sector_center=np.pi/4),
        'high_load_thick_ring': RingParameters(pressure=3e5, thickness=.33, youngs=27e9, radial_modulus=40e6,
                                              harmonic=.3, phase=np.pi/6, reduction=.6, sector_center=0.)}
    folder = output/'fields'; folder.mkdir()
    records = []
    for case, p in cases.items():
        theta = (np.arange(512)+.5)*2*np.pi/512
        depths = np.linspace(-.45, .45, 9)*p.thickness
        grids = np.meshgrid(theta, depths)
        arrays = {'theta': theta, 'depth': depths}
        solvers = {'beam_128': ('beam', 128, None), 'beam_256': ('beam', 256, None),
                   'continuum_256x4': ('continuum', 256, 4), 'continuum_512x8': ('continuum', 512, 8)}
        results = {}
        for name, (kind, n, nr) in solvers.items():
            pair = []
            for y in (0, 1):
                params = replace(p, reduction=y*p.reduction)
                fe = solve_ring(params, n) if kind == 'beam' else solve_continuum(params, n, nr, config['validation']['continuum']['poisson_ratio'])
                pair.append(fe)
                if kind == 'beam':
                    strain = recover_gauge_strain(fe, *grids, config['gauge_length_m'])
                    u = recover_displacement(fe, theta)
                else:
                    strain = recover_continuum_gauge(fe, *grids, config['gauge_length_m'])
                    u = point_fields(fe, theta, 0.)['displacement']
                arrays[f'{name}_strain_y{y}'] = strain
                arrays[f'{name}_u_y{y}'] = u
            results[name] = (kind, pair)
        port_rows = []
        for li in (config['layout_domains']['target_test'][0], config['layout_domains']['target_test'][-1]):
            layout = layouts[li]; geo = geometry(layout, p.radius, p.thickness)
            for solver, (kind, pair) in results.items():
                strains = [(recover_gauge_strain if kind == 'beam' else recover_continuum_gauge)
                           (fe, geo['theta'], geo['z'], config['gauge_length_m']) for fe in pair]
                for model, settings in config['physical_models'].items():
                    responses = [simulate(eps, geo, layout, **settings, method='expm', noise_sigma=0)['z_clean'] for eps in strains]
                    port_rows.append({'layout': li, 'solver': solver, 'model': model,
                                      **signature_metrics(*responses, config['receiver_noise_sigma'])})
                    arrays[f'{solver}_{model}_L{li}_delta'] = responses[1]-responses[0]
        field_comparisons = []
        for tested, reference in [('beam_128', 'beam_256'), ('continuum_256x4', 'continuum_512x8'), ('beam_256', 'continuum_512x8')]:
            for y in (0, 1):
                for quantity in ('strain', 'u'):
                    a, b = arrays[f'{tested}_{quantity}_y{y}'], arrays[f'{reference}_{quantity}_y{y}']
                    field_comparisons.append({'tested': tested, 'reference': reference, 'quantity': quantity, 'label': y,
                                              'relative_l2_difference': float(np.linalg.norm(a-b)/np.linalg.norm(b)),
                                              'maximum_absolute_difference': float(np.max(abs(a-b)))})
        np.savez_compressed(folder/f'{case}.npz', **arrays)
        records.append({'case': case, 'params': asdict(p), 'field_comparisons': field_comparisons,
                        'port_comparisons': port_rows, 'artifact_ref': f'fields/{case}.npz'})
        logging.info('Independent fields complete: %s', case)
    dump(output/'field_validation.json', records)


def run(config_path, manifest_path, output_path):
    output = Path(output_path); output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output/'run.log', encoding='utf-8'), logging.StreamHandler()])
    config = yaml.safe_load(Path(config_path).read_text(encoding='utf-8'))
    manifest = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    started = datetime.now(timezone.utc).isoformat(); start = time.perf_counter()
    provenance = {'started_at': started, 'config_hash': hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
                  'data_fingerprint': hashlib.sha256(Path(manifest_path).read_bytes()+Path(config['source']).read_bytes()).hexdigest(),
                  'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
                  'config': config, 'observation_storage': 'float32 in memory; regenerated from manifest for full acquisition arrays',
                  'source_arrays_retained': 'double-precision signatures, fixed response examples, fields, fitted readouts and individual predictions'}
    dump(output/'provenance.json', provenance)
    try:
        layouts, common, observations, clean = generate(config, manifest, output)
        readouts(config, common, observations, output)
        del observations
        nuisance_analysis(config, manifest, layouts, clean, output)
        functional_analysis(config, manifest, layouts, output)
        field_validation(config, layouts, output)
    except Exception:
        logging.exception('Mechanism study failed')
        raise
    dump(output/'completion.json', {'started_at': started, 'finished_at': datetime.now(timezone.utc).isoformat(),
                                    'exit_code': 0, 'wall_seconds': time.perf_counter()-start,
                                    'families': len(manifest), 'layouts': len(layouts),
                                    'observations_per_model': len(common['y']), 'models': len(config['physical_models'])})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True); parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    with threadpool_limits(limits=1): run(args.config, args.manifest, args.output)


if __name__ == '__main__':
    main()
