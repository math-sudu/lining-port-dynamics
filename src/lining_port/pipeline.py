"""First-round numerical experiment. Each run writes a new output directory."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import importlib.metadata as metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import numpy as np
import yaml
from .readout import run_readout, grouped_interval, external_features


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray)
                               else x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def load_config(path):
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8'))
    source = yaml.safe_load((path.parent / config['source']).read_text(encoding='utf-8'))
    return config, source


def freeze_manifest(config, source):
    rng = np.random.default_rng(config['seed'])
    rows = []
    index = 0
    st = source['structure']
    def draw(spec):
        return float(rng.uniform(spec['low'], spec['high']))
    for split, count in config['family_splits'].items():
        for _ in range(count):
            params = dict(radius=st['radius_m'], width=st['width_m'], thickness=draw(st['thickness_m']),
                          youngs=draw(st['youngs_modulus_Pa']), radial_modulus=draw(st['foundation']['radial_modulus_N_m3']),
                          tangential_ratio=st['foundation']['tangential_radial_ratio'],
                          pressure=draw(st['pressure']['p0_Pa']), harmonic=draw(st['pressure']['second_harmonic_amplitude']),
                          phase=draw(st['pressure']['phase_rad']),
                          sector_center=float(rng.uniform(0, 2*np.pi)),
                          sector_width=float(np.deg2rad(st['target']['sector_width_deg'])))
            reduction = draw(st['target']['positive_peak_reduction'])
            rows.append({'scenario_family_id': f'F{index:03d}', 'split': split, 'params': params,
                         'positive_reduction': reduction, 'layout_indices': config['layout_assignment'][split],
                         'noise_seed_base': config['seed'] + 10000 + index*10})
            index += 1
    return rows


def generate(config, source, manifest, output, elements):
    from .fe import RingParameters, solve_ring, recover_gauge_strain
    from .network import make_layout, geometry, simulate, coverage
    from .readout import audit_groups
    data_dir = output/'data'; data_dir.mkdir()
    raw_dir = data_dir/'raw'; raw_dir.mkdir()
    layouts = [make_layout(seed, source['layout']['mvp_nodes']) for seed in config['layout_seeds']]
    for j, layout in enumerate(layouts):
        np.savez_compressed(raw_dir/f'layout_L{j}.npz', **layout)
    arrays = {name: [] for name in config['physical_models']}
    rows, failed, structural, coverage_rows = [], [], [], []
    for index, row in enumerate(manifest):
        family = row['scenario_family_id']
        pair = []
        for label in (0, 1):
            params = RingParameters(**row['params'], reduction=label*row['positive_reduction'])
            try:
                result = solve_ring(params, elements)
                if result.diagnostics.get('converged') is False:
                    raise RuntimeError('FE diagnostics returned unconverged state')
                pair.append(result)
                np.savez_compressed(raw_dir/f'{family}_y{label}_fe.npz', theta=result.theta, xy=result.xy,
                                    displacements=result.displacements, reactions=result.reactions,
                                    local_forces=result.local_forces)
                structural.append({'scenario_family_id': family, 'label': label, 'params': asdict(params),
                                   'diagnostics': result.diagnostics})
            except Exception as exc:
                failed.append({'scenario_family_id': family, 'label': label, 'stage': 'FE',
                               'error': repr(exc), 'traceback': traceback.format_exc(),
                               'action': 'Exclude entire paired family from classification'})
                logging.exception('FE failed %s y%d', family, label)
                break
        if len(pair) != 2:
            continue
        for layout_index in row['layout_indices']:
            layout = layouts[layout_index]
            geo = geometry(layout, row['params']['radius'], row['params']['thickness'])
            np.savez_compressed(raw_dir/f'{family}_L{layout_index}_geometry.npz',
                                **{k: v for k, v in geo.items() if isinstance(v, np.ndarray)})
            coverage_rows.append({'scenario_family_id': family, 'layout_id': f'L{layout_index}',
                                  **geo['diagnostics'], **coverage(geo, row['params']['sector_center'], row['params']['sector_width'])})
            noise_seed = row['noise_seed_base'] + layout_index
            for label, result in enumerate(pair):
                strain = recover_gauge_strain(result, geo['theta'], geo['z'], source['layout']['gauge_length_m'])
                sample_id = f'{family}_y{label}_L{layout_index}'
                np.savez_compressed(raw_dir/f'{sample_id}_strain.npz', strain=strain)
                record = dict(sample_id=sample_id, scenario_family_id=family, split=row['split'],
                              layout_id=f'L{layout_index}', y=label, noise_seed=noise_seed)
                completed = {}
                for name, parameters in config['physical_models'].items():
                    sim = simulate(strain, geo, layout, **parameters, method=config['network_method'],
                                   noise_seed=noise_seed, noise_sigma=config['receiver_noise_sigma'])
                    completed[name] = sim
                rows.append(record)
                for name, sim in completed.items():
                    arrays[name].append(sim)
        if (index+1) % 10 == 0:
            logging.info('Completed %d/%d paired FE families; %d external observations per configuration',
                         index+1, len(manifest), len(rows))
    dump(output/'failures.json', failed)
    dump(data_dir/'structural_diagnostics.json', structural)
    dump(data_dir/'coverage.json', coverage_rows)
    dump(data_dir/'observation_manifest.json', rows)
    common = {key: np.asarray([r[key] for r in rows]) for key in ('sample_id','scenario_family_id','layout_id','split','y')}
    for name, sims in arrays.items():
        dataset = dict(common, z=np.stack([s['z'] for s in sims]), mask=np.stack([s['mask'] for s in sims]),
                       q=sims[0]['q'], time=sims[0]['time'])
        audit_groups(dataset)
        np.savez_compressed(data_dir/f'{name}.npz', **dataset)
        dump(data_dir/f'{name}_integration.json', [dict(sample_id=r['sample_id'], **s['diagnostics']) for r,s in zip(rows,sims)])
    return layouts, rows


def end_to_end_checks(config, source, manifest, output, elements, layouts, datasets, scores, thresholds):
    from .fe import RingParameters, solve_ring, recover_gauge_strain
    from .network import geometry, simulate
    from .readout import metrics
    checks = {}
    successful = set(datasets['P3_coupled']['scenario_family_id'])
    manifest = [row for row in manifest if row['scenario_family_id'] in successful]
    row = manifest[0]; layout_index = row['layout_indices'][0]; layout = layouts[layout_index]
    geo = geometry(layout, row['params']['radius'], row['params']['thickness'])
    params = RingParameters(**row['params'], reduction=0.0)
    fe_a, fe_b = solve_ring(params, elements), solve_ring(params, elements)
    strain = recover_gauge_strain(fe_a, geo['theta'], geo['z'])
    checks['fixed_seed_FE_exact'] = bool(np.array_equal(fe_a.displacements, fe_b.displacements))
    null_checks, repeat_checks, mesh_checks = {}, {}, {}
    fine = solve_ring(params, elements*2)
    fine_strain = recover_gauge_strain(fine, geo['theta'], geo['z'])
    for name, settings in config['physical_models'].items():
        sim = simulate(strain, geo, layout, **settings, method=config['network_method'],
                       noise_seed=row['noise_seed_base']+layout_index, noise_sigma=config['receiver_noise_sigma'])
        repeat_checks[name] = bool(np.array_equal(sim['z'], datasets[name]['z'][0]))
        null0 = simulate(np.zeros_like(strain), geo, layout, **settings, method=config['network_method'], noise_seed=301)
        null1 = simulate(np.zeros_like(strain), geo, layout, **settings, method=config['network_method'], noise_seed=301)
        null_checks[name] = float(np.max(np.abs(null0['z']-null1['z'])))
        coarse_sim = simulate(strain, geo, layout, **settings, method=config['network_method'], noise_sigma=0)
        fine_sim = simulate(fine_strain, geo, layout, **settings, method=config['network_method'], noise_sigma=0)
        mesh_checks[name] = {'absolute_max_difference': float(np.max(np.abs(coarse_sim['z']-fine_sim['z']))),
                            'relative_l2_difference': float(np.linalg.norm(coarse_sim['z']-fine_sim['z'])/max(np.linalg.norm(fine_sim['z']), 1e-12))}
    checks['fixed_seed_observation_exact'] = repeat_checks
    checks['zero_strain_same_pair_noise_max_difference'] = null_checks
    null_results = {}
    for name, settings in config['physical_models'].items():
        data = datasets[name]
        null_z = []
        for r in manifest:
            for li in r['layout_indices']:
                g = geometry(layouts[li], r['params']['radius'], r['params']['thickness'])
                null = simulate(np.zeros(len(layouts[li]['theta'])),g,layouts[li],**settings,
                                method=config['network_method'],noise_seed=r['noise_seed_base']+li,
                                noise_sigma=config['receiver_noise_sigma'])
                null_z.extend([null['z'],null['z']])
        null_data = dict(data,z=np.stack(null_z))
        null_metrics,_,_=run_readout(null_data,output/'negative_controls'/name,config['ridge_alphas'],permutations=0)
        null_results[name] = null_metrics['metrics']['test']
    checks['zero_strain_full_pipeline_test_metrics'] = null_results
    checks['FE_mesh_to_port_example'] = mesh_checks
    softened = solve_ring(RingParameters(**row['params'], reduction=row['positive_reduction']), elements)
    softened_fine = solve_ring(RingParameters(**row['params'], reduction=row['positive_reduction']), elements*2)
    for name, settings in config['physical_models'].items():
        pair = [simulate(recover_gauge_strain(fe, geo['theta'], geo['z']),geo,layout,
                         **settings,method=config['network_method'],noise_sigma=0)['z']
                for fe in (softened, softened_fine)]
        mesh_checks[name+'_softened'] = {'absolute_max_difference':float(np.abs(pair[0]-pair[1]).max()),
                                        'relative_l2_difference':float(np.linalg.norm(pair[0]-pair[1])/np.linalg.norm(pair[1]))}
    # Validation predictions only: compare exact propagation with tighter adaptive ODE,
    # and explicitly halve adaptive max_step for the complete validation families.
    refinement = {}
    val_manifest = [r for r in manifest if r['split']=='validation']
    for name, settings in config['physical_models'].items():
        baseline = datasets[name]; indices = np.flatnonzero(baseline['split']=='validation')
        refined, step_halved = [], []
        for r in val_manifest:
            li = r['layout_indices'][0]; geo = geometry(layouts[li], r['params']['radius'], r['params']['thickness'])
            for label in (0,1):
                fe = solve_ring(RingParameters(**r['params'], reduction=label*r['positive_reduction']), elements)
                strain = recover_gauge_strain(fe,geo['theta'],geo['z'])
                kwargs = dict(**settings, method='adaptive',noise_seed=r['noise_seed_base']+li,
                              noise_sigma=config['receiver_noise_sigma'])
                refined.append(simulate(strain,geo,layouts[li],**kwargs,rtol=1e-7,atol=1e-9,max_step=.005)['z'])
                step_halved.append(simulate(strain,geo,layouts[li],**kwargs,rtol=1e-8,atol=1e-10,max_step=.0025)['z'])
        refined, step_halved = np.stack(refined), np.stack(step_halved)
        model = np.load(output/'physical'/name/'model.npz')
        features = np.concatenate([step_halved.reshape(len(indices),-1),baseline['mask'][indices]],axis=1)
        new_score = ((features-model['mean'])/model['scale'])@model['coef']+model['intercept']
        old_score = scores[name][indices]; threshold = thresholds[name]
        old_ba = metrics(baseline['y'][indices],old_score,threshold)['balanced_accuracy']
        new_ba = metrics(baseline['y'][indices],new_score,threshold)['balanced_accuracy']
        delta = step_halved - refined
        exact_delta = step_halved - baseline['z'][indices]
        refinement[name] = {'max_step_initial_s':.005,'max_step_refined_s':.0025,
                            'step_halving_max_absolute_error':float(np.abs(delta).max()),
                            'exact_vs_refined_max_absolute_error':float(np.abs(exact_delta).max()),
                            'relative_sequence_error':float(np.linalg.norm(delta)/np.linalg.norm(step_halved)),
                            'validation_BA_change':float(new_ba-old_ba),
                            'maximum_score_change':float(np.max(np.abs(new_score-old_score))),
                            'prediction_flip_fraction':float(np.mean((new_score>=threshold)!=(old_score>=threshold))),
                            'nearest_threshold_distance':float(np.min(np.abs(old_score-threshold)))}
        np.savez_compressed(output/'validation'/f'{name}_refinement.npz',sample_id=baseline['sample_id'][indices],
                            z_adaptive=refined,z_half_step=step_halved,score=new_score)
        logging.info('Validation integration refinement complete: %s',name)
    checks['validation_step_refinement'] = refinement
    checks['pass'] = bool(checks['fixed_seed_FE_exact'] and all(repeat_checks.values())
                          and max(null_checks.values())==0
                          and all(v['balanced_accuracy'] in (.5,None) for v in null_results.values())
                          and all(v['relative_l2_difference']<.001 for v in mesh_checks.values())
                          and all(abs(v['validation_BA_change'])<=.002 and v['step_halving_max_absolute_error']<1e-4
                                  and v['exact_vs_refined_max_absolute_error']<1e-4 and v['relative_sequence_error']<.001
                                  for v in refinement.values()))
    dump(output/'validation'/'end_to_end_checks.json',checks)
    return checks


def run(config_path, output, digital=True):
    output=Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    (output/'validation').mkdir()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(output/'run.log',encoding='utf-8'),logging.StreamHandler(sys.stdout)])
    config,source=load_config(config_path)
    dump(output/'executed_config.json',{'execution':config,'research_spec':source})
    import openseespy.opensees as ops
    packages={d.metadata['Name']:d.version for d in metadata.distributions()}
    dump(output/'environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                                  'processor':platform.processor(),'logical_cpus':os.cpu_count(),'packages':packages,
                                  'OpenSees_core_version':ops.version(),
                                  'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                  'git_status':subprocess.check_output(['git','status','--short'],text=True).strip(),
                                  'seeds':{'experiment':config['seed'],'layouts':config['layout_seeds'],'probe':4100}})
    manifest=freeze_manifest(config,source)
    dump(output/'split_manifest.json',manifest)  # Before any FE data exist.
    start=time.perf_counter()
    try:
        from .fe import run_fe_validation
        from .network import run_network_validation
        fe=run_fe_validation(output/'validation')
        logging.info('FE validation: %s; selected mesh %s',fe.get('pass'),fe.get('selected_elements'))
        if not fe['pass']:
            raise RuntimeError('FE validation failed; dataset generation not started')
        network=run_network_validation(output/'validation')
        logging.info('Network validation: %s',network.get('pass'))
        if not network['pass']:
            raise RuntimeError('Network validation failed; dataset generation not started')
        layouts,rows=generate(config,source,manifest,output,fe['selected_elements'])
        datasets={name:dict(np.load(output/'data'/f'{name}.npz')) for name in config['physical_models']}
        results,scores,thresholds={},{},{}
        for name,data in datasets.items():
            results[name],scores[name],thresholds[name]=run_readout(data,output/'physical'/name,config['ridge_alphas'],
                                                                  config['permutation_replicates'],config['seed'])
        dump(output/'physical_results.json',results)
        dump(output/'grouped_intervals.json',grouped_interval(datasets['P3_coupled'],scores,thresholds,
                                                             config['bootstrap_replicates']))
        checks=end_to_end_checks(config,source,manifest,output,fe['selected_elements'],layouts,datasets,scores,thresholds)
        if not checks['pass']:
            raise RuntimeError('End-to-end numerical checks failed; digital comparison not started')
        if digital:
            from .digital import run_digital_comparison
            digital_result = run_digital_comparison(
                output/'data'/f"{config['digital']['locked_physical_source']}.npz",output/'digital',
                seed=config['digital']['seed'],epochs=config['digital']['epochs'])
            if digital_result['failures']:
                details = '; '.join(f"{item['model']}: {item['error']}" for item in digital_result['failures'])
                raise RuntimeError(f'Digital comparison failed: {details}')
        dump(output/'completion.json',{'status':'complete','wall_seconds':time.perf_counter()-start,
                                       'FE_states':2*len(set(r['scenario_family_id'] for r in rows)),
                                       'observation_sequences_per_physical_config':len(rows),
                                       'physical_configs':len(config['physical_models']),'digital_run':digital,
                                       'claim':'First-round implementation and numerical checks only'})
    except Exception as exc:
        logging.exception('Run stopped')
        dump(output/'run_error.json',{'error':repr(exc),'traceback':traceback.format_exc()})
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default='config.yaml')
    parser.add_argument('--output',required=True)
    parser.add_argument('--skip-digital',action='store_true')
    args=parser.parse_args()
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run(args.config,args.output,not args.skip_digital)


if __name__=='__main__':
    main()
