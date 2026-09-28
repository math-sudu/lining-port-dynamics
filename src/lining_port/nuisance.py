"""Local four-port nuisance tangent diagnostic, separate from deployment ranks."""
from dataclasses import asdict, replace
import argparse
import json
import logging
from pathlib import Path
import time
import numpy as np
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
from .fe import RingParameters, solve_ring, recover_gauge_strain
from .network import make_layout, geometry, simulate
from .pipeline import dump


PARAMETERS=('youngs','thickness','radial_modulus','pressure','harmonic','phase')
MODELS={'P0':dict(family='P0',kappa=0.,alpha=0.),
        'P3_coupled':dict(family='P3',kappa=.3,alpha=0.),
        'P4_coupled':dict(family='P4',kappa=.3,alpha=.5)}


def evaluate(params,layout,elements=128,raw_path=None):
    result=solve_ring(params,elements)
    geom=geometry(layout,params.radius,params.thickness)
    strain=recover_gauge_strain(result,geom['theta'],geom['z'])
    outputs={name:simulate(strain,geom,layout,**settings,noise_sigma=0.,method='expm')['z'].ravel()
             for name,settings in MODELS.items()}
    if raw_path is not None:
        np.savez_compressed(raw_path,params_json=json.dumps(asdict(params)),
                            displacements=result.displacements,local_forces=result.local_forces,
                            node_strain=strain,coordinates=geom['xy'],W=geom['W'],H=geom['H'],
                            **{f'z_{name}':value.reshape(400,4) for name,value in outputs.items()})
    return outputs,geom


def projected_evidence(delta,jacobian,sigma=.01,svd_relative_cutoff=1e-6):
    norms=np.linalg.norm(jacobian,axis=0)
    if np.any(norms==0):
        raise ValueError('Zero nuisance derivative requires explicit handling')
    directions=jacobian/norms
    u,s,vh=np.linalg.svd(directions,full_matrices=False)
    keep=s>s[0]*svd_relative_cutoff
    residual=delta-u[:,keep]@(u[:,keep].T@delta)
    # Coefficients in the original physical parameter units.
    coefficients=(vh[keep].T@((u[:,keep].T@delta)/s[keep]))/norms
    return residual,coefficients,s,int(keep.sum())


def run(source,output):
    source=Path(source);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    start=time.perf_counter();layout=make_layout(5100)
    raw_dir=output/'raw';raw_dir.mkdir()
    raw_calls=[]
    def recorded_evaluate(params,case):
        path=raw_dir/f'{len(raw_calls):03d}_{case}.npz'
        values,geom=evaluate(params,layout,raw_path=path)
        raw_calls.append({'case':case,'path':path.relative_to(output).as_posix(),'params':asdict(params)})
        return values,geom
    manifest=json.loads((source/'split_manifest.json').read_text())
    # Fixed positions in the training-family list, selected before this run.
    chosen=[manifest[i] for i in (0,15,30,45)]
    plan={'families':[r['scenario_family_id'] for r in chosen],'layout':'L0',
          'parameters':PARAMETERS,'steps':'1% of E,t,k,p; 0.01 harmonic and 0.01 rad phase; repeat at half step',
          'sigma':.01,'svd_relative_cutoff':1e-6,
          'interpretation':'Offline local nuisance tangent at the healthy state; paired finite softening difference, not a classifier',
          'bounds':{'youngs':[27e9,33e9],'thickness':[.27,.33],'radial_modulus':[40e6,60e6],
                    'pressure':[1e5,3e5],'harmonic':[0,.3],'phase':[0,2*np.pi]}}
    dump(output/'plan.json',plan)
    records=[];evaluations=0
    for row in chosen:
        base=RingParameters(**row['params'],reduction=0.)
        family=row['scenario_family_id']
        normal,geo=recorded_evaluate(base,f'{family}_healthy')
        soft,_=recorded_evaluate(replace(base,reduction=row['positive_reduction']),f'{family}_softened')
        evaluations+=2
        steps={k:.01*getattr(base,k) for k in PARAMETERS[:4]};steps.update(harmonic=.01,phase=.01)
        previous={}
        for factor in (1.,.5):
            jacobians={name:[] for name in MODELS};topology_changes=[]
            for name in PARAMETERS:
                h=steps[name]*factor
                plus,gp=recorded_evaluate(replace(base,**{name:getattr(base,name)+h}),f'{family}_{name}_plus_{factor}')
                minus,gm=recorded_evaluate(replace(base,**{name:getattr(base,name)-h}),f'{family}_{name}_minus_{factor}')
                evaluations+=2
                topology_changes.append(bool(np.any((gp['W']>0)!=(geo['W']>0)) or np.any((gm['W']>0)!=(geo['W']>0))
                                               or np.any(gp['H']!=geo['H']) or np.any(gm['H']!=geo['H'])))
                for model in MODELS:jacobians[model].append((plus[model]-minus[model])/(2*h))
            for model in MODELS:
                delta=soft[model]-normal[model];jacobian=np.column_stack(jacobians[model])
                residual,coef,s,rank=projected_evidence(delta,jacobian)
                d=np.linalg.norm(delta)/.01;dr=np.linalg.norm(residual)/.01
                proposed={name:float(getattr(base,name)+coefficient) for name,coefficient in zip(PARAMETERS,coef)}
                proposed['phase']%=2*np.pi
                in_bounds=all(plan['bounds'][k][0]<=v<=plan['bounds'][k][1] for k,v in proposed.items())
                record={'scenario_family_id':row['scenario_family_id'],'model':model,'step_factor':factor,
                        'nuisance_rank':rank,'normalized_singular_values':s.tolist(),
                        'topology_or_port_changes':dict(zip(PARAMETERS,topology_changes)),
                        'raw_sequence_distance':float(d),'distance_after_nuisance_projection':float(dr),
                        'remaining_squared_signal_fraction':float((dr/d)**2),
                        'known_nuisance_BA':float(ndtr(d/2)),
                        'local_projected_BA_diagnostic':float(ndtr(dr/2)),
                        'linearized_healthy_match_parameters':proposed,'match_within_source_ranges':in_bounds}
                if model in previous:
                    record['relative_jacobian_change_on_step_halving']=float(np.linalg.norm(jacobian-previous[model])/np.linalg.norm(jacobian))
                previous[model]=jacobian
                # Test the local explanation against an actual healthy FE state,
                # only when its estimated parameters remain within source ranges.
                if factor==.5 and in_bounds and not any(topology_changes):
                    healthy_match,_=recorded_evaluate(replace(base,**proposed),f'{family}_{model}_healthy_match');evaluations+=1
                    record['actual_healthy_match_distance']=float(np.linalg.norm(soft[model]-healthy_match[model])/.01)
                np.savez_compressed(output/f"{row['scenario_family_id']}_{model}_step{factor}.npz",
                                    healthy=normal[model],softened=soft[model],nuisance_jacobian=jacobian,
                                    projected_residual=residual,parameter_order=np.array(PARAMETERS),linear_match_increment=coef)
                records.append(record)
        logging.info('Nuisance diagnosis complete: %s',row['scenario_family_id'])
    dump(output/'results.json',records)
    dump(output/'raw_manifest.json',raw_calls)
    dump(output/'completion.json',{'status':'completed','FE_solves':evaluations,'network_solves':3*evaluations,
                                   'wall_seconds':time.perf_counter()-start,'all_families_from_training':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source',default='results/round1')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s')
    with threadpool_limits(limits=1):run(args.source,args.output)
