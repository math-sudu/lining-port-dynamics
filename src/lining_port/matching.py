"""Bounded nonlinear matching of measured sequences, offline diagnostics only."""
from __future__ import annotations
import argparse
from collections import OrderedDict
from dataclasses import asdict, replace
import importlib.metadata
import json
import logging
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from scipy.optimize import least_squares
from scipy.special import ndtr
from threadpoolctl import threadpool_limits
import yaml
from .fe import RingParameters, solve_ring, recover_gauge_strain
from .network import make_layout, geometry, simulate
from .nuisance import PARAMETERS, MODELS
from .pipeline import dump


class ParameterCoordinates:
    """Five normalized bounded parameters plus an unwrapped pi-periodic phase."""
    parameter_names = PARAMETERS

    def __init__(self, base, bounds):
        self.base = base
        self.low = np.array([bounds[k][0] for k in PARAMETERS[:5]])
        self.span = np.array([bounds[k][1]-bounds[k][0] for k in PARAMETERS[:5]])
        self.bounds = (np.r_[np.zeros(5), -np.inf], np.r_[np.ones(5), np.inf])

    def encode(self, params):
        first = (np.array([getattr(params,k) for k in PARAMETERS[:5]])-self.low)/self.span
        # Nearest equivalent phase to the original state; the second harmonic
        # has physical period pi even though the specification samples 0..2pi.
        phase = ((params.phase-self.base.phase+np.pi/2) % np.pi - np.pi/2)/np.pi
        return np.r_[first, phase]

    def decode(self, coordinates):
        first = self.low+np.asarray(coordinates[:5])*self.span
        phase = (self.base.phase+np.pi*coordinates[5]) % np.pi
        return replace(self.base, **dict(zip(PARAMETERS, [*first,phase])), reduction=0.)

    def clip(self, coordinates):
        result=np.array(coordinates,dtype=float,copy=True)
        result[:5]=np.clip(result[:5],0,1)
        return result


def topology_key(geom):
    return (np.packbits(geom['W']>0).tobytes(), geom['port_nodes'].tobytes())


def forward(params, layout, model, elements, method='expm'):
    fe=solve_ring(params,elements)
    geom=geometry(layout,params.radius,params.thickness)
    strain=recover_gauge_strain(fe,geom['theta'],geom['z'])
    sim=simulate(strain,geom,layout,**MODELS[model],noise_sigma=0.,method=method,
                 rtol=1e-8,atol=1e-10,max_step=.0025)
    return {'params':params,'fe':fe,'geom':geom,'strain':strain,'sim':sim,
            'z':sim['z'].ravel(),'topology':topology_key(geom)}


def save_state(path, state):
    fe,geom=state['fe'],state['geom']
    np.savez_compressed(path,params_json=json.dumps(asdict(state['params'])),
                        elements=fe.n_elements,coordinates=geom['xy'],theta=geom['theta'],depth=geom['z'],
                        displacements=fe.displacements,reactions=fe.reactions,loads=fe.loads,
                        local_forces=fe.local_forces,radial_stiffness=fe.radial_stiffness,
                        tangential_stiffness=fe.tangential_stiffness,node_strain=state['strain'],
                        W=geom['W'],H=geom['H'],port_nodes=geom['port_nodes'],mask=geom['mask'],
                        z=state['sim']['z'],q=state['sim']['q'],time=state['sim']['time'],
                        fe_diagnostics_json=json.dumps(fe.diagnostics),
                        network_diagnostics_json=json.dumps(state['sim']['diagnostics']))


class MatchingObjective:
    def __init__(self, coordinates, layout, model, target, elements, sigma, trace, step=.001):
        self.coordinates=coordinates; self.layout=layout; self.model=model
        self.target=target; self.elements=elements; self.sigma=sigma; self.trace=trace; self.step=step
        self.cache=OrderedDict(); self.evaluations=0; self.topologies={}; self.catalog=[]
        self.one_sided_derivatives=0; self.topology_derivative_adjustments=0

    def evaluate(self, x, kind='residual'):
        key=np.asarray(x,dtype=np.float64).tobytes()
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        params=self.coordinates.decode(x)
        state=forward(params,self.layout,self.model,self.elements)
        state['residual']=(state['z']-self.target)/self.sigma
        topo=state['topology']
        if topo not in self.topologies:
            self.topologies[topo]=len(self.catalog)
            geom=state['geom']
            self.catalog.append({'topology_id':self.topologies[topo],
                                 'edges':np.argwhere(np.triu(geom['W']>0,1)).tolist(),
                                 'port_nodes':geom['port_nodes'].tolist()})
        self.evaluations+=1
        if self.trace is not None:
            self.trace.write(json.dumps({'evaluation':self.evaluations,'kind':kind,
                                         'parameters':asdict(params),'coordinates':np.asarray(x).tolist(),
                                         'sequence_distance':float(np.linalg.norm(state['residual'])),
                                         'topology_id':self.topologies[topo]})+'\n')
            self.trace.flush()
        self.cache[key]=state
        if len(self.cache)>32:self.cache.popitem(last=False)
        return state

    def fun(self,x):
        return self.evaluate(x)['residual']

    def jac(self,x):
        """Differences stay inside parameter bounds and the center's topology.

        Trial optimizer points may cross an edge/port event; their objective is
        always recomputed by FE. No derivative interpolates across that jump.
        """
        center=self.evaluate(x); columns=[]
        low,high=self.coordinates.bounds
        for j, parameter in enumerate(self.coordinates.parameter_names):
            h=self.step
            for shrink in range(14):
                candidates={}
                for sign in (-1,1):
                    delta=sign*h
                    if low[j]<=x[j]+delta<=high[j]:
                        point=np.array(x,copy=True);point[j]+=delta
                        value=self.evaluate(point,kind=f'derivative_{parameter}')
                        if value['topology']==center['topology']:
                            candidates[sign]=value['residual']
                        else:
                            self.topology_derivative_adjustments+=1
                if len(candidates)==2:
                    column=(candidates[1]-candidates[-1])/(2*h);break
                if len(candidates)==1:
                    sign=next(iter(candidates))
                    column=(candidates[sign]-center['residual'])/(sign*h)
                    self.one_sided_derivatives+=1;break
                h/=2
            else:
                raise RuntimeError(f'No same-topology finite difference for {parameter}')
            columns.append(column)
        return np.column_stack(columns)


def initial_points(coords, tangent, seed):
    healthy=coords.encode(coords.base)
    increment=np.asarray(tangent['linear_match_increment'])
    predicted=replace(coords.base,**{k:getattr(coords.base,k)+v for k,v in zip(PARAMETERS,increment)})
    rng=np.random.default_rng(seed)
    points={'healthy':healthy,'clipped_linear':coords.clip(coords.encode(predicted))}
    for j in range(2):
        point=rng.uniform(size=6);point[5]-=.5
        points[f'random_{j}']=point
    return points


def verify_saved_states(output):
    """Recompute reported distances and domain checks from retained raw states."""
    output=Path(output)
    plan=json.loads((output/'plan.json').read_text())
    records=json.loads((output/'results.json').read_text())
    checks=[]
    for record in records:
        path=output/record['candidate_file']
        candidate=np.load(path,allow_pickle=False)
        target=np.load(path.parent/'softened_target.npz',allow_pickle=False)
        params=json.loads(str(candidate['params_json']))
        distance=float(np.linalg.norm((candidate['z']-target['z']).ravel())/plan['sigma'])
        fe=json.loads(str(candidate['fe_diagnostics_json']))
        check={'scenario_family_id':record['scenario_family_id'],'model':record['model'],'start':record['start'],
               'distance_recalculated':distance,'distance_record_error':abs(distance-record['distance']),
               'healthy_state':params['reduction']==0.,
               'parameters_in_range':all(plan['bounds'][k][0]<=params[k]<=plan['bounds'][k][1] for k in PARAMETERS),
               'same_observed_channel_mask':bool(np.array_equal(candidate['mask'],target['mask'])),
               'normalized_bound_margins':{
                   k:[(params[k]-plan['bounds'][k][0])/(plan['bounds'][k][1]-plan['bounds'][k][0]),
                      (plan['bounds'][k][1]-params[k])/(plan['bounds'][k][1]-plan['bounds'][k][0])]
                   for k in PARAMETERS[:5]},
               'FE_equilibrium_residual':fe['relative_equilibrium_residual'],
               'FE_energy_error':fe['relative_energy_error'],'FE_warnings':fe['warnings']}
        if 'condition' in record:
            check['condition']=record['condition']
        checks.append(check)
    report={'candidate_count':len(checks),'checks':checks,
            'pass':bool(checks) and all(r['healthy_state'] and r['parameters_in_range']
                 and r['same_observed_channel_mask'] and r['distance_record_error']<1e-10 for r in checks)}
    dump(output/'saved_state_checks.json',report)
    if not report['pass']:raise AssertionError('Saved matching states disagree with reported distances or admissible inputs')
    return report


def run(config_path, output, family_filter=None):
    config_path=Path(config_path).resolve();root=config_path.parent
    config=yaml.safe_load(config_path.read_text())
    source=root/config['source'];tangents=root/config['tangent_source']
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(output/'run.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter();layout=make_layout(config['layout_seed'])
    plan=json.loads((tangents/'plan.json').read_text())
    manifest=json.loads((source/'split_manifest.json').read_text())
    families=config['families'] if family_filter is None else [family_filter]
    if any(f not in config['families'] for f in families):raise ValueError('Family not in frozen matching plan')
    rows=[r for r in manifest if r['scenario_family_id'] in families]
    if any(r['split']!='train' for r in rows):raise ValueError('Matching uses training families only')
    dump(output/'plan.json',dict(config,bounds=plan['bounds'],executed_families=families,
                               phase_rule='unwrapped optimizer coordinate; physical phase modulo pi',
                               derivative_rule='normalized central/one-sided finite differences within topology',
                               optimization_scope='Local multistart search; no global minimum certificate'))
    dump(output/'provenance.json',{'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                 'scipy':importlib.metadata.version('scipy'),'numpy':np.__version__})
    records=[];verifications=[];failures=[];total_evaluations=0
    for row in rows:
        family=row['scenario_family_id'];base=RingParameters(**row['params'],reduction=0.)
        coords=ParameterCoordinates(base,plan['bounds'])
        for model in config['models']:
            target_params=replace(base,reduction=row['positive_reduction'])
            target=forward(target_params,layout,model,config['elements']);total_evaluations+=1
            folder=output/f'{family}_{model}';folder.mkdir()
            save_state(folder/'softened_target.npz',target)
            tangent=np.load(tangents/f'{family}_{model}_step0.5.npz')
            discrepancy=float(np.max(np.abs(target['z']-tangent['softened'])))
            if discrepancy>1e-10:raise RuntimeError(f'Cannot reproduce saved target: {family}/{model}: {discrepancy}')
            starts=initial_points(coords,tangent,config['random_seed']+int(family[1:]))
            model_records=[]
            for name in config['starts']:
                x0=starts[name];timer=time.perf_counter()
                with (folder/f'{name}_trace.jsonl').open('w',encoding='utf-8') as trace:
                    obj=MatchingObjective(coords,layout,model,target['z'],config['elements'],config['sigma'],trace,
                                          config['optimizer']['finite_difference_step'])
                    try:
                        initial=obj.evaluate(x0);save_state(folder/f'{name}_initial.npz',initial)
                        settings={k:v for k,v in config['optimizer'].items() if k!='finite_difference_step'}
                        fit=least_squares(obj.fun,x0,jac=obj.jac,bounds=coords.bounds,x_scale=1.,**settings)
                        result=obj.evaluate(fit.x);save_state(folder/f'{name}_candidate.npz',result)
                        distance=float(np.linalg.norm(result['residual']))
                        record={'scenario_family_id':family,'model':model,'start':name,
                                'initial_parameters':asdict(initial['params']),'parameters':asdict(result['params']),
                                'initial_distance':float(np.linalg.norm(initial['residual'])),
                                'distance':distance,'known_pair_gaussian_BA':float(ndtr(distance/2)),
                                'optimizer_success':bool(fit.success),'optimizer_status':int(fit.status),
                                'stop_reason':fit.message,'optimality':float(fit.optimality),
                                'optimizer_nfev':int(fit.nfev),'optimizer_njev':int(fit.njev),
                                'FE_network_evaluations':obj.evaluations,
                                'active_bounds':{p:int(v) for p,v in zip(PARAMETERS,fit.active_mask)},
                                'topologies_visited':len(obj.catalog),
                                'topology_changed_from_target':result['topology']!=target['topology'],
                                'one_sided_derivatives':obj.one_sided_derivatives,
                                'topology_derivative_adjustments':obj.topology_derivative_adjustments,
                                'within_original_bounds':all(plan['bounds'][k][0]<=getattr(result['params'],k)<=plan['bounds'][k][1] for k in PARAMETERS),
                                'wall_seconds':time.perf_counter()-timer,
                                'candidate_file':(folder/f'{name}_candidate.npz').relative_to(output).as_posix()}
                        records.append(record);model_records.append(record)
                        logging.info('%s %s %s: d=%.6g, status=%d, %d FE calls, %.1f s',
                                     family,model,name,distance,fit.status,obj.evaluations,record['wall_seconds'])
                    except Exception as exc:
                        failure={'scenario_family_id':family,'model':model,'start':name,
                                 'error':repr(exc),'traceback':traceback.format_exc()}
                        failures.append(failure);logging.exception('Matching failed: %s/%s/%s',family,model,name)
                    finally:
                        total_evaluations+=obj.evaluations
                        dump(folder/f'{name}_topologies.json',obj.catalog)
                        dump(output/'results.json',records);dump(output/'failures.json',failures)
            if not model_records:continue
            # This is the best observed candidate, not a certified global optimum.
            best=min(model_records,key=lambda r:r['distance']);best_params=RingParameters(**best['parameters'])
            replay=forward(best_params,layout,model,config['elements']);total_evaluations+=1
            saved=np.load(output/best['candidate_file'])
            fine_target=forward(target_params,layout,model,config['verification_elements'])
            fine_match=forward(best_params,layout,model,config['verification_elements']);total_evaluations+=2
            save_state(folder/'best_fine_target.npz',fine_target);save_state(folder/'best_fine_candidate.npz',fine_match)
            adaptive_target=forward(target_params,layout,model,config['elements'],'adaptive')
            adaptive_match=forward(best_params,layout,model,config['elements'],'adaptive');total_evaluations+=2
            save_state(folder/'best_adaptive_target.npz',adaptive_target);save_state(folder/'best_adaptive_candidate.npz',adaptive_match)
            normalized_error=np.linalg.norm((adaptive_target['z']-adaptive_match['z'])-(target['z']-replay['z']))/config['sigma']
            item={'scenario_family_id':family,'model':model,'best_start':best['start'],
                  'best_distance':best['distance'],
                  'all_start_distances':{r['start']:r['distance'] for r in model_records},
                  'fine_mesh_distance':float(np.linalg.norm(fine_match['z']-fine_target['z'])/config['sigma']),
                  'adaptive_distance':float(np.linalg.norm(adaptive_match['z']-adaptive_target['z'])/config['sigma']),
                  'integration_residual_difference_noise_units':float(normalized_error),
                  'candidate_replay_max_error':float(np.max(np.abs(saved['z'].ravel()-replay['z']))),
                  'bounds_valid':best['within_original_bounds'],
                  'retained_optimizer_success':best['optimizer_success']}
            verifications.append(item);dump(output/'verification.json',verifications)
    dump(output/'completion.json',{'status':'completed' if not failures else 'completed_with_failures',
                                  'candidate_count':len(records),'failed_fits':len(failures),
                                  'optimizer_nonconverged_count':sum(not r['optimizer_success'] for r in records),
                                  'FE_solves':total_evaluations,'network_solves':total_evaluations,
                                  'wall_seconds':time.perf_counter()-start,'all_families_from_training':True})
    verify_saved_states(output)
    return records,verifications


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='matching_config.yaml')
    parser.add_argument('--output',required=True);parser.add_argument('--family')
    args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.config,args.output,args.family)


if __name__=='__main__':main()
