"""Recompute a saved test case from FE through its frozen four-port classifier."""
import argparse
import json
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from .fe import RingParameters, solve_ring, recover_gauge_strain
from .network import make_layout, geometry, simulate


def reproduce(run_dir, sample_id='F080_y1_L3', model_name='P3_coupled'):
    run_dir=Path(run_dir)
    config=json.loads((run_dir/'executed_config.json').read_text())['execution']
    rows=json.loads((run_dir/'data/observation_manifest.json').read_text())
    row=next(r for r in rows if r['sample_id']==sample_id)
    family=next(r for r in json.loads((run_dir/'split_manifest.json').read_text())
                if r['scenario_family_id']==row['scenario_family_id'])
    elements=json.loads((run_dir/'validation/fe_validation.json').read_text())['selected_elements']
    params=RingParameters(**family['params'],reduction=row['y']*family['positive_reduction'])
    result=solve_ring(params,elements)
    layout_index=int(row['layout_id'][1:])
    layout=make_layout(config['layout_seeds'][layout_index])
    geom=geometry(layout,params.radius,params.thickness)
    strain=recover_gauge_strain(result,geom['theta'],geom['z'])
    observed=simulate(strain,geom,layout,**config['physical_models'][model_name],
                      method=config['network_method'],noise_seed=row['noise_seed'],
                      noise_sigma=config['receiver_noise_sigma'])
    predictor=np.load(run_dir/'physical'/model_name/'model.npz')
    feature=np.r_[observed['z'].ravel(),observed['mask']]
    score=float(((feature-predictor['mean'])/predictor['scale'])@predictor['coef']+predictor['intercept'])
    reference=np.load(run_dir/'physical'/model_name/'predictions.npz')
    index=np.flatnonzero(reference['sample_id']==sample_id)[0]
    original=np.load(run_dir/'data'/f'{model_name}.npz')
    sequence_index=np.flatnonzero(original['sample_id']==sample_id)[0]
    answer={'sample_id':sample_id,'model':model_name,'split':row['split'],
            'FE_elements':elements,'FE_max_displacement_m':result.diagnostics['max_displacement_m'],
            'node_strain_range': [float(strain.min()),float(strain.max())],
            'external_sequence_shape':list(observed['z'].shape),'valid_ports':int(observed['mask'].sum()),
            'true_label':row['y'],'score':score,'threshold':float(predictor['threshold']),
            'prediction':int(score>=predictor['threshold']) if observed['mask'].any() else -1,
            'sequence_exactly_reproduced':bool(np.array_equal(observed['z'],original['z'][sequence_index])),
            'score_absolute_reproduction_error':float(abs(score-reference['score'][index]))}
    if not answer['sequence_exactly_reproduced'] or answer['score_absolute_reproduction_error']>1e-10:
        raise AssertionError(f'End-to-end reproducibility check failed: {answer}')
    return answer


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='results/round1')
    parser.add_argument('--sample-id',default='F080_y1_L3')
    parser.add_argument('--model',default='P3_coupled')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(reproduce(args.run,args.sample_id,args.model),indent=2))
