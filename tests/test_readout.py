import numpy as np
import pytest
from lining_port.readout import audit_groups, external_features, fit_model, run_readout


def fixture():
    rng=np.random.default_rng(10)
    return dict(z=rng.normal(size=(12,8,4)),mask=np.ones((12,4)),y=np.tile([0,1],6),
                split=np.repeat(['train','validation','test'],4),
                scenario_family_id=np.repeat(['a','b','c','d','e','f'],2),
                layout_id=np.repeat(['L0','L1','L2'],4),sample_id=np.arange(12).astype(str))


def test_family_and_layout_leakage_rejected():
    data=fixture(); assert audit_groups(data)['pass']
    data['scenario_family_id'][4]='a'
    with pytest.raises(ValueError,match='leakage'): audit_groups(data)


def test_hidden_fields_and_test_data_cannot_affect_fit():
    data=fixture(); a=fit_model(data,[1,10])
    original=external_features(data)
    data['internal_state']=data['y'][:,None]
    assert np.array_equal(original,external_features(data))
    data['z'][8:]=1e12
    data['y'][8:]=1-data['y'][8:]
    b=fit_model(data,[1,10])
    np.testing.assert_array_equal(a[0].mean_,b[0].mean_)
    np.testing.assert_array_equal(a[1].coef_,b[1].coef_)
    assert a[2]==b[2]


def test_zero_port_test_samples_are_rejected(tmp_path):
    data=fixture(); data['mask'][8:]=0
    report,score,_=run_readout(data,tmp_path,[1],permutations=1)
    assert report['zero_port_rejections']==4
    assert report['metrics']['test']['balanced_accuracy'] is None
    assert np.isnan(score[8:]).all()
    predictions=np.load(tmp_path/'predictions.npz')
    assert (predictions['prediction'][8:]==-1).all()


def test_correlated_float32_features_match_an_independent_svd_ridge():
    rng = np.random.default_rng(319)
    n, width = 32, 400
    latent = np.linspace(-1., 1., n)
    z = (latent[:, None, None] + rng.normal(0., .002, (n, width, 4))).astype(np.float32)
    data = dict(z=z, mask=np.ones((n, 4), dtype=bool),
                y=(latent > 0).astype(int),
                split=np.array(['train' if i%4 else 'validation' for i in range(n)]))
    alpha = .01
    scaler, model, _, _ = fit_model(data, [alpha])
    x = scaler.transform(external_features(data))
    train = data['split'] == 'train'
    center = x[train].mean(axis=0)
    target = 2*data['y'][train]-1
    u, s, vt = np.linalg.svd(x[train]-center, full_matrices=False)
    coef = vt.T@((s/(s*s+alpha))*(u.T@(target-target.mean())))
    expected = (x-center)@coef+target.mean()
    np.testing.assert_allclose(model.predict(x), expected, rtol=1e-7, atol=1e-8)
