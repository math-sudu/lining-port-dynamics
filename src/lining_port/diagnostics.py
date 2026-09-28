"""Bounded, exploratory diagnosis of the first round's failed classification."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace
import numpy as np
from scipy.special import ndtr
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from threadpoolctl import threadpool_limits
import yaml
from .fe import RingParameters, recover_gauge_strain
from .network import make_layout, geometry, simulate
from .pipeline import dump
from .readout import audit_groups, external_features, run_readout, fit_model


def noise_for_acquisition(seed, family_index, layout_index, acquisition_index, shape=(400,4)):
    """Independent SeedSequence streams; class labels are not an input."""
    rng = np.random.default_rng(np.random.SeedSequence([seed, family_index, layout_index, acquisition_index]))
    return rng.normal(size=shape)


def load_sections(source, family, label):
    params = RingParameters(**family['params'], reduction=label*family['positive_reduction'])
    with np.load(source/'data/raw'/f"{family['scenario_family_id']}_y{label}_fe.npz") as saved:
        # Only these saved quantities are consumed by section/gauge recovery.
        return SimpleNamespace(params=params, local_forces=saved['local_forces'],
                               n_elements=len(saved['theta']), mesh_rotation=0.0)


def original_score_diagnostics(source, output):
    result = {}
    for model_path in sorted((source/'physical').iterdir()):
        predictor=np.load(model_path/'model.npz')
        pred=np.load(model_path/'predictions.npz')
        threshold=float(predictor['threshold'])
        result[model_path.name]={'coefficient_l2_norm':float(np.linalg.norm(predictor['coef'])),
                                 'threshold':threshold,'splits':{}}
        for split in ('train','validation','test'):
            take=pred['split']==split; y=pred['y'][take]; score=pred['score'][take]
            result[model_path.name]['splits'][split]={
                'score_range':[float(score.min()),float(score.max())],
                'score_mean':float(score.mean()),'score_std':float(score.std()),
                'signed_label_mse':float(np.mean((score-(2*y-1))**2)),
                'BA_at_signed_origin':float(balanced_accuracy_score(y,score>=0)),
                'BA_at_frozen_validation_threshold':float(balanced_accuracy_score(y,score>=threshold)),
                'AUROC':float(roc_auc_score(y,score))}
    dump(output/'round1_score_diagnostics.json',result)
    return result


def generate_observations(config, source, target):
    original=json.loads((source/'executed_config.json').read_text())['execution']
    manifest=json.loads((source/'split_manifest.json').read_text())
    models={k:original['physical_models'][k] for k in config['models']}
    layouts={i:make_layout(original['layout_seeds'][i]) for i in config['layouts']}
    rows=[]; clean={k:[] for k in models}; masks=[]; raw_strain=[]; coverage=[]; integrations=[]
    count_new=0; count_reused=0; reproduction_errors=[]
    existing={k:dict(np.load(source/'data'/f'{k}.npz')) for k in models}
    original_rows=json.loads((source/'data/observation_manifest.json').read_text())
    original_index={r['sample_id']:i for i,r in enumerate(original_rows)}
    for index, family in enumerate(manifest):
        sections=[load_sections(source,family,label) for label in (0,1)]
        for li,layout in layouts.items():
            geom=geometry(layout,family['params']['radius'],family['params']['thickness'])
            coverage.append(dict(scenario_family_id=family['scenario_family_id'],layout_id=f'L{li}',**geom['diagnostics']))
            for acquisition,section in enumerate(sections):
                sample_id=f"{family['scenario_family_id']}_y{acquisition}_L{li}"
                strain=recover_gauge_strain(section,geom['theta'],geom['z'])
                rows.append(dict(sample_id=sample_id,scenario_family_id=family['scenario_family_id'],
                                 family_index=index,layout_id=f'L{li}',layout_index=li,split=family['split'],
                                 y=acquisition,acquisition_index=acquisition))
                raw_strain.append(strain);masks.append(geom['mask'])
                if sample_id in original_index:
                    old_index=original_index[sample_id]
                    old=original_rows[old_index]
                    noise=np.random.default_rng(old['noise_seed']).normal(0,original['receiver_noise_sigma'],(400,4))*geom['mask']
                    count_reused+=1
                else:
                    count_new+=1
                for name,settings in models.items():
                    sim=simulate(strain,geom,layout,**settings,noise_sigma=0,method=original['network_method'])
                    clean[name].append(sim['z_clean'])
                    integrations.append(dict(sample_id=sample_id,model=name,**sim['diagnostics']))
                    if sample_id in original_index:
                        reproduction_errors.append(float(np.max(np.abs(sim['z_clean']+noise-existing[name]['z'][old_index]))))
        if (index+1)%20==0:
            logging.info('Recovered %d/%d saved FE families on L0 and L3',index+1,len(manifest))
    common={k:np.asarray([r[k] for r in rows]) for k in ('sample_id','scenario_family_id','layout_id','split','y')}
    common.update(mask=np.stack(masks),q=existing['P0']['q'],time=existing['P0']['time'])
    dump(target/'observation_manifest.json',rows)
    dump(target/'coverage.json',coverage)
    dump(target/'integration.json',integrations)
    dump(target/'round1_reproduction.json',{'max_absolute_observation_difference':max(reproduction_errors),
                                          'comparisons':len(reproduction_errors),
                                          'clean_sequence_method':'Direct network propagation from saved FE sections; no subtraction'})
    np.savez_compressed(target/'recovered_strain.npz',strain=np.stack(raw_strain),**{k:common[k] for k in ('sample_id','scenario_family_id','layout_id','y')})
    datasets={}
    for name,values in clean.items():
        datasets[name]=dict(common,z=np.stack(values))
        np.savez_compressed(target/f'{name}_clean.npz',**datasets[name])
    return datasets,rows,{'existing_FE_states_reused':2*len(manifest),'new_FE_solves':0,
                          'recovered_existing_acquisitions':count_reused,'new_layout_acquisitions':count_new,
                          'new_network_solves':(count_new+count_reused)*len(models)}


def conditional_gaussian_evidence(clean, sigma=.01):
    """Known-nuisance two-template oracle: BA=Phi(||mu1-mu0||/(2*sigma)).

    Samples have iid post-receiver Gaussian noise. This uses both labelled
    clean templates and is never supplied to a deployed classifier.
    """
    result=[]
    for family in np.unique(clean['scenario_family_id']):
        for layout in np.unique(clean['layout_id']):
            take=(clean['scenario_family_id']==family)&(clean['layout_id']==layout)
            if take.sum()!=2:
                raise ValueError('Conditional oracle requires complete paired acquisitions')
            mu0=clean['z'][take&(clean['y']==0)][0]
            mu1=clean['z'][take&(clean['y']==1)][0]
            delta=(mu1-mu0).ravel()
            d=float(np.linalg.norm(delta)/sigma)
            result.append({'scenario_family_id':str(family),'layout_id':str(layout),
                           'sequence_difference_RMS':float(np.sqrt(np.mean(delta**2))),
                           'full_sequence_mahalanobis_distance':d,
                           'known_nuisance_optimal_BA':float(ndtr(d/2))})
    return result


def select_protocol(data, protocol):
    take=np.array([li==f"L{protocol[split]}" for li,split in zip(data['layout_id'],data['split'])])
    result={k:(v[take] if k not in ('q','time') else v) for k,v in data.items()}
    audit_groups(result,protocol['layout_policy'])
    return result


def run(config_path, output):
    config=yaml.safe_load(Path(config_path).read_text())
    source=Path(config['source_run']).resolve(); output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    data_dir=output/'data';data_dir.mkdir()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(output/'run.log',encoding='utf-8'),logging.StreamHandler()])
    start=time.perf_counter()
    dump(output/'executed_config.json',config)
    dump(output/'provenance.json',{'source_run':str(source),'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                                    'git_status':subprocess.check_output(['git','status','--short'],text=True).strip(),
                                    'scientific_status':'exploratory; first-round test labels previously inspected'})
    original_score_diagnostics(source,output)
    clean,rows,counts=generate_observations(config,source,data_dir)
    summary={};oracle={};fit_checks={};all_predictions={}
    for name,data in clean.items():
        oracle[name]=conditional_gaussian_evidence(data)
        for condition,noise_config in config['noise_conditions'].items():
            noises=np.stack([noise_for_acquisition(config['noise_seed'],r['family_index'],r['layout_index'],
                              r['acquisition_index'] if noise_config['pairing']=='independent' else 0)
                              for r in rows])
            observed=dict(data,z=data['z']+noise_config['sigma']*noises*data['mask'][:,None,:])
            # Preserve the actually used realizations and split metadata.
            np.savez_compressed(data_dir/f'{name}_{condition}.npz',**observed)
            for protocol_name,protocol in config['protocols'].items():
                selected=select_protocol(observed,protocol)
                key=f'{name}/{condition}/{protocol_name}'
                report,score,threshold=run_readout(selected,output/'readouts'/name/condition/protocol_name,
                    config['ridge_alphas'],permutations=0,layout_policy=protocol['layout_policy'])
                scaler,model,ref_threshold,reproduced=fit_model(selected,config['ridge_alphas'])
                train=selected['split']=='train'
                x=scaler.transform(external_features(selected))[train];y=2*selected['y'][train]-1
                residual=x.T@(x@model.coef_+model.intercept_-y)+model.alpha*model.coef_
                optimality=float(np.linalg.norm(residual)/max(np.linalg.norm(x.T@y),1e-12))
                fit_checks[key]={'ridge_gradient_relative_residual':optimality,
                                 'repeat_score_max_error':float(np.max(np.abs(score-reproduced))),
                                 'repeat_threshold_equal':threshold==ref_threshold}
                summary[key]={'alpha':report['alpha'],'threshold':threshold,'metrics':report['metrics']}
                for split in ('train','validation','test'):
                    take=selected['split']==split;s=score[take]
                    summary[key]['metrics'][split].update(score_range=[float(s.min()),float(s.max())],
                        BA_at_signed_origin=float(balanced_accuracy_score(selected['y'][take],s>=0)))
                all_predictions[key]=(selected['sample_id'],score)
            logging.info('Completed %s, %s: seen/transfer/refit',name,condition)
    # In the seen and transfer protocols training and validation are identical.
    # Test changes must therefore leave the fitted predictor unchanged.
    for name in clean:
        for condition in config['noise_conditions']:
            paths=[output/'readouts'/name/condition/p/'model.npz' for p in ('seen_L0','transfer_L0_L3')]
            a,b=map(np.load,paths)
            if not all(np.array_equal(a[k],b[k]) for k in a.files):
                raise AssertionError('Test layout contaminated fit or model selection')
    dump(output/'conditional_oracle.json',{'assumptions':'Known nuisance/softening templates; iid post-receiver Gaussian noise sigma=.01; equal class priors; offline upper bound only',
                                         'formula':'Phi(||mu1-mu0||_2/(2*sigma))','records':oracle})
    dump(output/'fit_checks.json',fit_checks)
    dump(output/'classification_diagnostics.json',summary)
    if max(v['ridge_gradient_relative_residual'] for v in fit_checks.values())>1e-5:
        raise AssertionError('Ridge optimality residual requires investigation')
    dump(output/'completion.json',{'status':'completed','counts':counts,'readout_fits':len(summary),
                                   'wall_seconds':time.perf_counter()-start,'first_round_unchanged':True,
                                   'test_layout_did_not_affect_fit':True})


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',default='diagnostic_config.yaml')
    parser.add_argument('--output',required=True);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.config,args.output)
