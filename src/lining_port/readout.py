"""Fixed linear readout with family-disjoint fitting and persisted fit IDs."""
from pathlib import Path
import json
import time
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, confusion_matrix


def external_features(data):
    # Deliberately enumerate observable fields; unknown/internal fields are ignored.
    # Long, correlated time series require double precision in the ridge solve,
    # including when acquisition arrays use float32 storage.
    z = np.asarray(data['z'], dtype=np.float64)
    return np.concatenate([z.reshape(len(z), -1), data['mask']], axis=1)


def audit_groups(data, layout_policy='disjoint'):
    split = data['split']
    groups = {s: set(data['scenario_family_id'][split == s].tolist())
              for s in ('train', 'validation', 'test')}
    for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
        if groups[a] & groups[b]:
            raise ValueError(f'Scenario family leakage: {a}/{b}')
    layouts = {s: set(data['layout_id'][split == s].tolist()) for s in groups}
    if layout_policy == 'disjoint':
        for a, b in (('train', 'validation'), ('train', 'test'), ('validation', 'test')):
            if layouts[a] & layouts[b]:
                raise ValueError(f'Unexpected shared layout in T2-only pilot: {a}/{b}')
        protocol = 'T2 zero-adjustment; unseen validation layout'
    elif layout_policy == 'seen':
        if not (layouts['validation'] <= layouts['train'] and layouts['test'] <= layouts['train']):
            raise ValueError('Seen-layout protocol requires validation/test layouts in training')
        protocol = 'T1 seen layout; new scenario families'
    elif layout_policy == 'unseen_test':
        if not layouts['validation'] <= layouts['train'] or layouts['test'] & (layouts['train'] | layouts['validation']):
            raise ValueError('Unseen-test protocol requires seen validation and disjoint test layouts')
        protocol = 'T2 zero-adjustment; seen validation layout'
    else:
        raise ValueError(f'Unknown layout policy: {layout_policy}')
    for g in np.unique(data['scenario_family_id']):
        for layout in np.unique(data['layout_id'][data['scenario_family_id'] == g]):
            index = (data['scenario_family_id'] == g) & (data['layout_id'] == layout)
            if sorted(data['y'][index].tolist()) != [0, 1]:
                raise ValueError(f'Incomplete healthy/softened pair: {g}/{layout}')
    return {'pass': True, 'family_counts': {s: len(v) for s, v in groups.items()},
            'layouts': {s: sorted(set(data['layout_id'][split == s].tolist())) for s in groups},
            'split_unit': 'scenario_family_id', 'test_protocol': protocol}


def best_threshold(y, score):
    values = np.unique(score)
    candidates = np.r_[values[0] - 1e-9, (values[:-1] + values[1:]) / 2, values[-1] + 1e-9]
    # On ties prefer the threshold closest to the signed-label origin.
    return float(max(candidates, key=lambda t: (balanced_accuracy_score(y, score >= t), -abs(t))))


def metrics(y, score, threshold):
    if len(y) == 0:
        return {'status': 'rejected_no_observable_samples', 'balanced_accuracy': None,
                'auroc': None, 'fpr': None, 'tpr': None, 'confusion_matrix': [[0,0],[0,0]]}
    pred = score >= threshold
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
            'auroc': float(roc_auc_score(y, score)),
            'fpr': float(fp / (tn + fp)), 'tpr': float(tp / (tp + fn)),
            'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]}


def fit_model(data, alphas, labels=None):
    x = external_features(data)
    y = data['y'] if labels is None else labels
    train, val = data['split'] == 'train', data['split'] == 'validation'
    train &= data['mask'].sum(axis=1) > 0
    val &= data['mask'].sum(axis=1) > 0
    if not train.any() or not val.any():
        raise ValueError('No observed ports in train or validation; classification rejected')
    scaler = StandardScaler().fit(x[train])
    xs = scaler.transform(x)
    candidates = []
    for alpha in alphas:
        model = Ridge(alpha=alpha).fit(xs[train], 2 * y[train] - 1)
        score = model.predict(xs[val])
        threshold = best_threshold(y[val], score)
        candidates.append((balanced_accuracy_score(y[val], score >= threshold), -alpha,
                           model, threshold))
    _, _, model, threshold = max(candidates, key=lambda row: row[:2])
    return scaler, model, threshold, model.predict(xs)


def grouped_interval(data, scores, thresholds, repetitions=1000, seed=7081):
    """Shared crossed layout/family weights; one-layout limits explicitly retained."""
    test = (data['split'] == 'test') & (data['mask'].sum(axis=1) > 0)
    y, family, layout = (data[k][test] for k in ('y', 'scenario_family_id', 'layout_id'))
    families, fi = np.unique(family, return_inverse=True)
    layouts, li = np.unique(layout, return_inverse=True)
    if not len(layouts):
        return {'status': 'rejected_no_observable_test_samples', 'coverage': 0.0}
    rng = np.random.default_rng(seed)
    draws = {key: [] for key in scores}
    for _ in range(repetitions):
        wf = rng.multinomial(len(families), np.ones(len(families))/len(families))
        wl = rng.multinomial(len(layouts), np.ones(len(layouts))/len(layouts))
        for key, score in scores.items():
            pred = score[test] >= thresholds[key]
            bas, weights = [], []
            for j in range(len(layouts)):
                if wl[j] == 0:
                    continue
                take = li == j
                if any(wf[fi[take & (y == c)]].sum() == 0 for c in (0, 1)):
                    continue
                recalls = [np.average(pred[take & (y == c)] == c,
                                      weights=wf[fi[take & (y == c)]]) for c in (0, 1)]
                bas.append(np.mean(recalls)); weights.append(wl[j])
            if bas:
                draws[key].append(float(np.average(bas, weights=weights)))
    out = {'method': 'shared crossed multinomial layout and family weights',
           'replicates': repetitions, 'test_layout_count': len(layouts),
           'limitation': 'Conditional on the sole test layout; no population layout CI' if len(layouts) == 1 else None,
           'models': {k: np.quantile(v, [.025, .975]).tolist() for k, v in draws.items()}}
    if 'P4_coupled' in draws and 'P3_coupled' in draws:
        delta = np.array(draws['P4_coupled']) - draws['P3_coupled']
        out['P4_minus_P3_fixed_config_95_interval'] = np.quantile(delta, [.025, .975]).tolist()
    return out


def run_readout(data, output_dir, alphas, permutations=19, seed=20260926, layout_policy='disjoint'):
    output_dir = Path(output_dir); output_dir.mkdir(parents=True, exist_ok=True)
    audit = audit_groups(data, layout_policy)
    start = time.perf_counter()
    scaler, model, threshold, score = fit_model(data, alphas)
    elapsed = time.perf_counter() - start
    eligible = data['mask'].sum(axis=1) > 0
    score[~eligible] = np.nan
    out = {'group_audit': audit, 'alpha': float(model.alpha), 'threshold': threshold,
           'input_dimension': int(model.n_features_in_), 'trainable_coefficients': int(model.coef_.size + 1),
           'fit_wall_seconds': elapsed, 'zero_port_rejections': int((~eligible).sum()),
           'coverage': float(eligible.mean()), 'metrics': {}, 'layout_metrics': {}}
    for split in ('train', 'validation', 'test'):
        take = (data['split'] == split) & eligible
        out['metrics'][split] = metrics(data['y'][take], score[take], threshold)
        out['layout_metrics'][split] = {}
        for layout in np.unique(data['layout_id'][take]):
            index = take & (data['layout_id'] == layout)
            out['layout_metrics'][split][str(layout)] = metrics(data['y'][index], score[index], threshold)
        layout_bas = [m['balanced_accuracy'] for m in out['layout_metrics'][split].values()]
        out['metrics'][split]['layout_macro_balanced_accuracy'] = float(np.mean(layout_bas)) if layout_bas else None
    out['fit_ids'] = {purpose: data['sample_id'][(data['split'] == split) & eligible].tolist()
                      for purpose, split in [('normalization', 'train'), ('ridge_fit', 'train'),
                                             ('regularization_selection', 'validation'), ('threshold_selection', 'validation')]}
    np.savez_compressed(output_dir/'model.npz', mean=scaler.mean_, scale=scaler.scale_,
                        coef=model.coef_, intercept=model.intercept_, threshold=threshold,
                        alpha=model.alpha)
    np.savez_compressed(output_dir/'predictions.npz', score=score, prediction=np.where(eligible,(score>=threshold).astype(int),-1),
                        y=data['y'], sample_id=data['sample_id'], split=data['split'],
                        scenario_family_id=data['scenario_family_id'], layout_id=data['layout_id'])
    # Permute paired label orientation at family level, preserving layout replicas.
    rng = np.random.default_rng(seed + 778)
    permutation_ba = []
    for _ in range(permutations):
        labels = data['y'].copy()
        for group in np.unique(data['scenario_family_id'][data['split'] != 'test']):
            if rng.integers(2):
                take = data['scenario_family_id'] == group
                labels[take] = 1 - labels[take]
        _, _, shuffled_threshold, shuffled_score = fit_model(data, alphas, labels)
        test = (data['split'] == 'test') & eligible
        permutation_ba.append(float(balanced_accuracy_score(data['y'][test], shuffled_score[test] >= shuffled_threshold)) if test.any() else None)
    out['label_shuffle'] = {'replicates': permutations, 'test_BA': permutation_ba,
                            'unit': 'paired-label orientation per family; layouts share permutation',
                            'interpretation': 'Diagnostic null distribution; no accuracy acceptance threshold'}
    # No hidden-field additions can change observable features.
    poisoned = dict(data, strain=np.full((len(score), 64), np.nan), internal_state=data['y'][:, None])
    out['hidden_field_pollution_pass'] = bool(np.array_equal(external_features(data), external_features(poisoned)))
    (output_dir/'metrics.json').write_text(json.dumps(out, indent=2), encoding='utf-8')
    return out, score, threshold
