from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from lining_port.fe import RingParameters
from lining_port.matching import ParameterCoordinates, MatchingObjective, forward, topology_key
from lining_port.network import make_layout


BOUNDS={'youngs':[27e9,33e9],'thickness':[.27,.33],'radial_modulus':[40e6,60e6],
        'pressure':[1e5,3e5],'harmonic':[0,.3],'phase':[0,2*np.pi]}


def test_periodic_phase_and_bounded_round_trip():
    base=RingParameters(phase=2*np.pi-.01)
    coords=ParameterCoordinates(base,BOUNDS)
    x=coords.encode(base)
    a=coords.decode(x);b=coords.decode(x+np.array([0,0,0,0,0,1]))
    assert 0<=a.phase<np.pi
    assert abs(a.phase-b.phase)<1e-14
    for field in ('youngs','thickness','radial_modulus','pressure','harmonic'):
        assert np.isclose(getattr(a,field),getattr(base,field))


def test_actual_phase_periodicity_and_elastic_scale_invariance():
    base=RingParameters();layout=make_layout(5100)
    first=forward(base,layout,'P4_coupled',128)
    periodic=forward(replace(base,phase=base.phase+np.pi),layout,'P4_coupled',128)
    scaled=forward(replace(base,youngs=base.youngs*1.01,radial_modulus=base.radial_modulus*1.01,
                           pressure=base.pressure*1.01),layout,'P4_coupled',128)
    np.testing.assert_allclose(first['z'],periodic['z'],atol=2e-10,rtol=1e-9)
    np.testing.assert_allclose(first['z'],scaled['z'],atol=2e-10,rtol=1e-9)


def test_nonlinear_fit_recovers_known_healthy_sequence():
    base=RingParameters();layout=make_layout(5100);coords=ParameterCoordinates(base,BOUNDS)
    target=forward(base,layout,'P0',64)
    obj=MatchingObjective(coords,layout,'P0',target['z'],64,.01,None)
    x=coords.encode(base);x[3]+=.04;x[4]-=.03;x[5]+=.01
    initial=np.linalg.norm(obj.fun(x))
    result=least_squares(obj.fun,x,jac=obj.jac,bounds=coords.bounds,max_nfev=20)
    assert np.linalg.norm(result.fun)<initial*1e-3
    assert np.linalg.norm(result.fun)<.01


def test_finite_differences_do_not_cross_topology_jump():
    # Exact artificial branch change tests a behavior that a smooth case misses.
    class Objective(MatchingObjective):
        def evaluate(self,x,kind='residual'):
            side=x[1]>=.5
            return {'residual':np.array([x[0]+x[1]+10*side]),'topology':side}
    coords=ParameterCoordinates(RingParameters(),BOUNDS)
    obj=Objective(coords,None,None,None,None,None,None)
    x=np.array([.4,.5,.4,.4,.4,0.])
    jac=obj.jac(x)
    np.testing.assert_allclose(jac[0,:2],[1,1],atol=1e-10)
    assert obj.topology_derivative_adjustments>0
