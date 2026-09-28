import numpy as np
import pytest
from lining_port.readout import audit_groups
from lining_port.diagnostics import noise_for_acquisition, conditional_gaussian_evidence
from lining_port.nuisance import projected_evidence


def test_declared_layout_protocol_preserves_family_disjointness():
    data=dict(split=np.repeat(['train','validation','test'],2),
              scenario_family_id=np.repeat(['a','b','c'],2),layout_id=np.array(['L0']*6),y=np.tile([0,1],3))
    with pytest.raises(ValueError,match='shared layout'):audit_groups(data)
    assert audit_groups(data,'seen')['pass']
    data['layout_id'][4:]='L3'
    assert audit_groups(data,'unseen_test')['pass']
    with pytest.raises(ValueError,match='Seen-layout'):audit_groups(data,'seen')
    data['scenario_family_id'][4:]='a'
    with pytest.raises(ValueError,match='leakage'):audit_groups(data,'unseen_test')


def test_acquisition_noise_is_reproducible_and_independent():
    a=noise_for_acquisition(7,2,0,0)
    b=noise_for_acquisition(7,2,0,1)
    np.testing.assert_array_equal(a,noise_for_acquisition(7,2,0,0))
    assert not np.array_equal(a,b)
    assert abs(np.corrcoef(a.ravel(),b.ravel())[0,1])<.1


def test_conditional_gaussian_oracle_uses_sequence_distance():
    data=dict(z=np.array([[[0.]],[[2.]]]),scenario_family_id=np.array(['a','a']),
              layout_id=np.array(['L0','L0']),y=np.array([0,1]))
    row=conditional_gaussian_evidence(data,sigma=1)[0]
    assert row['full_sequence_mahalanobis_distance']==2
    assert row['known_nuisance_optimal_BA']==pytest.approx(.8413447460685429)


def test_nuisance_projection_is_invariant_to_parameter_units():
    j=np.array([[1.,0.],[0.,2.],[0.,0.]])
    delta=np.array([3.,4.,5.])
    residual,coef,_,rank=projected_evidence(delta,j)
    np.testing.assert_allclose(residual,[0.,0.,5.])
    np.testing.assert_allclose(coef,[3.,2.])
    changed=projected_evidence(delta,j*np.array([1e9,1e-3]))
    np.testing.assert_allclose(changed[0],residual)
    assert rank==2
