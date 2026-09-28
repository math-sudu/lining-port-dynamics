"""Bilateral supported elastic ring, SI units and CCW beam connectivity.

The only local material response exposed is tangential fiber strain.  Positive
depth is outward; curvature is defined so epsilon = epsilon0 - depth * kappa.
OpenSees is process-global: calls to this module must not run in parallel threads.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import json
from pathlib import Path

import numpy as np
import openseespy.opensees as ops
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve


@dataclass(frozen=True)
class RingParameters:
    radius: float = 3.0
    width: float = 1.0
    thickness: float = 0.3
    youngs: float = 30e9
    radial_modulus: float = 50e6
    tangential_ratio: float = 0.05
    pressure: float = 2e5
    harmonic: float = 0.15
    phase: float = 0.2
    reduction: float = 0.0
    sector_center: float = 0.0
    sector_width: float = np.pi / 6

    @property
    def area(self) -> float:
        return self.width * self.thickness

    @property
    def inertia(self) -> float:
        return self.width * self.thickness**3 / 12


@dataclass
class FEResult:
    params: RingParameters
    n_elements: int
    theta: np.ndarray
    xy: np.ndarray
    displacements: np.ndarray
    reactions: np.ndarray
    local_forces: np.ndarray
    loads: np.ndarray
    radial_stiffness: np.ndarray
    tangential_stiffness: np.ndarray
    diagnostics: dict
    mesh_rotation: float = 0.0


class FEConvergenceError(RuntimeError):
    """An unsuccessful or nonfinite solve is never a classification sample."""


def _geometry(p: RingParameters, n: int, rotation: float = 0.0):
    theta = np.arange(n) * (2 * np.pi / n) + rotation
    radial = np.column_stack((np.cos(theta), np.sin(theta)))
    tangent = np.column_stack((-np.sin(theta), np.cos(theta)))
    return theta, p.radius * radial, radial, tangent


def _foundation(p: RingParameters, theta: np.ndarray):
    delta = (theta - p.sector_center + np.pi) % (2 * np.pi) - np.pi
    weight = np.where(abs(delta) <= p.sector_width / 2,
                      0.5 * (1 + np.cos(2 * np.pi * delta / p.sector_width)), 0)
    tributary = 2 * np.pi * p.radius / len(theta)
    kr = p.radial_modulus * p.width * tributary * (1 - p.reduction * weight)
    kt = np.full(len(theta), p.radial_modulus * p.tangential_ratio * p.width * tributary)
    return kr, kt


def _element_matrices(p: RingParameters, point_i, point_j):
    delta = point_j - point_i
    length = np.linalg.norm(delta)
    c, s = delta / length
    transform = np.zeros((6, 6))
    rotation = np.array([[c, s, 0], [-s, c, 0], [0, 0, 1]])
    transform[:3, :3] = rotation
    transform[3:, 3:] = rotation
    axial = p.youngs * p.area / length
    bending = p.youngs * p.inertia / length**3
    matrix = np.array([
        [axial, 0, 0, -axial, 0, 0],
        [0, 12*bending, 6*length*bending, 0, -12*bending, 6*length*bending],
        [0, 6*length*bending, 4*length**2*bending, 0, -6*length*bending, 2*length**2*bending],
        [-axial, 0, 0, axial, 0, 0],
        [0, -12*bending, -6*length*bending, 0, 12*bending, -6*length*bending],
        [0, 6*length*bending, 2*length**2*bending, 0, -6*length*bending, 4*length**2*bending],
    ])
    return length, transform, matrix


def assemble_frame(p: RingParameters, n: int, rotation: float = 0.0):
    """Independent textbook 2-D Euler-Bernoulli matrix assembly (no ops data)."""
    theta, xy, radial, tangent = _geometry(p, n, rotation)
    rows, cols, values = [], [], []
    for i in range(n):
        j = (i + 1) % n
        _, transform, local = _element_matrices(p, xy[i], xy[j])
        dofs = np.r_[np.arange(3*i, 3*i+3), np.arange(3*j, 3*j+3)]
        rows.extend(np.repeat(dofs, 6))
        cols.extend(np.tile(dofs, 6))
        values.extend((transform.T @ local @ transform).ravel())
    ring = coo_matrix((values, (rows, cols)), shape=(3*n, 3*n)).tocsr()
    kr, kt = _foundation(p, theta)
    support = np.einsum('i,ij,ik->ijk', kr, radial, radial)
    support += np.einsum('i,ij,ik->ijk', kt, tangent, tangent)
    node_dofs = 3 * np.arange(n)[:, None] + np.arange(2)
    support_matrix = coo_matrix((support.ravel(), (
        np.repeat(node_dofs, 2, axis=1).ravel(),
        np.tile(node_dofs, (1, 2)).ravel())), shape=(3*n, 3*n)).tocsr()
    loads = np.zeros((n, 3))
    tributary = 2 * np.pi * p.radius / n
    pressure = p.pressure * (1 + p.harmonic * np.cos(2 * (theta - p.phase)))
    loads[:, :2] = -pressure[:, None] * p.width * tributary * radial
    return ring, support_matrix, loads


def _static_analysis():
    ops.constraints('Plain')
    ops.numberer('RCM')
    ops.system('BandSPD')
    ops.algorithm('Linear')
    ops.integrator('LoadControl', 1.0)
    ops.analysis('Static')
    code = ops.analyze(1)
    if code != 0:
        raise FEConvergenceError(f'OpenSees static analysis returned {code}')


def solve_ring(params: RingParameters, n_elements: int = 128,
               mesh_rotation: float = 0.0) -> FEResult:
    p, n = params, n_elements
    positive = (p.radius, p.width, p.thickness, p.youngs, p.radial_modulus,
                p.tangential_ratio, p.sector_width)
    if not all(np.isfinite(positive)) or min(positive) <= 0:
        raise ValueError('Geometry, modulus, both foundation directions and sector width must be positive')
    if n < 8 or int(n) != n or not 0 <= p.reduction < 1 or not abs(p.harmonic) < 1:
        raise ValueError('Invalid mesh, softening fraction, or pressure harmonic')
    theta, xy, radial, tangent = _geometry(p, n, mesh_rotation)
    kr, kt = _foundation(p, theta)
    ring, support, loads = assemble_frame(p, n, mesh_rotation)
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.geomTransf('Linear', 1)
    for i in range(n):
        ops.node(i+1, *xy[i])
        ops.node(n+i+1, *xy[i])
        ops.fix(n+i+1, 1, 1, 1)
        ops.uniaxialMaterial('Elastic', 2*i+1, float(kr[i]))
        ops.uniaxialMaterial('Elastic', 2*i+2, float(kt[i]))
        ops.element('zeroLength', n+i+1, n+i+1, i+1,
                    '-mat', 2*i+1, 2*i+2, '-dir', 1, 2,
                    '-orient', *radial[i], 0.0, *tangent[i], 0.0)
    for i in range(n):
        ops.element('elasticBeamColumn', i+1, i+1, (i+1) % n+1,
                    p.area, p.youngs, p.inertia, 1)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    for i in range(n):
        ops.load(i+1, *loads[i])
    _static_analysis()
    displacement = np.array([ops.nodeDisp(i+1) for i in range(n)])
    if not np.isfinite(displacement).all():
        raise FEConvergenceError('OpenSees produced nonfinite displacement')
    ops.reactions()
    reactions = np.array([ops.nodeReaction(n+i+1) for i in range(n)])
    local = np.array([ops.eleResponse(i+1, 'localForce') for i in range(n)])
    u, f = displacement.ravel(), loads.ravel()
    residual = (ring + support) @ u - f
    load_scale = np.linalg.norm(f)
    relative_residual = np.linalg.norm(residual) / max(load_scale, 1.0)
    work = float(u @ f)
    energy = float(0.5 * u @ (ring + support) @ u)
    resultant = (loads + reactions).sum(axis=0)
    moment = np.sum(xy[:, 0] * (loads[:, 1]+reactions[:, 1])
                    - xy[:, 1] * (loads[:, 0]+reactions[:, 0])
                    + loads[:, 2] + reactions[:, 2])
    diagnostics = dict(
        converged=True, solver_return_code=0, solver_version=ops.version(),
        units='m,N,Pa; rotation rad; strain dimensionless; curvature 1/m',
        relative_equilibrium_residual=float(relative_residual),
        relative_energy_error=abs(2*energy-work)/max(abs(work), 1e-30),
        strain_energy_J=energy, external_work_N_m=work,
        force_resultant_N=resultant[:2].tolist(), moment_resultant_N_m=float(moment),
        relative_force_balance=float(np.linalg.norm(resultant[:2])/max(np.sum(np.linalg.norm(loads[:, :2], axis=1)), 1.0)),
        relative_moment_balance=float(abs(moment)/max(p.radius*np.sum(np.linalg.norm(loads[:, :2], axis=1)), 1.0)),
        max_displacement_m=float(np.linalg.norm(displacement[:, :2], axis=1).max()),
        total_radial_stiffness_N_m=float(kr.sum()),
        total_tangential_stiffness_N_m=float(kt.sum()),
        warnings=[],
    )
    result = FEResult(p, n, theta, xy, displacement, reactions, local, loads,
                      kr, kt, diagnostics, mesh_rotation)
    # End sections bound the linear curvature and fiber strain in every element.
    eps0 = local[:, 3] / (p.youngs * p.area)
    kappa = np.column_stack((local[:, 2], -local[:, 5])) / (p.youngs*p.inertia)
    fibers = np.concatenate([(eps0[:, None]-z*kappa).ravel() for z in [-p.thickness/2, p.thickness/2]])
    diagnostics.update(max_abs_strain=float(abs(fibers).max()),
                       max_abs_fiber_stress_Pa=float(abs(fibers).max()*p.youngs),
                       max_tensile_fiber_stress_Pa=float(max(fibers.max(), 0)*p.youngs))
    if diagnostics['max_displacement_m']/p.radius > 1e-3:
        diagnostics['warnings'].append('displacement_over_radius_exceeds_1e-3')
    if diagnostics['max_abs_strain'] > 5e-4:
        diagnostics['warnings'].append('strain_exceeds_5e-4')
    if fibers.max() > 0:
        diagnostics['warnings'].append('tensile_fiber_stress_elastic_idealization_only')
    if relative_residual >= 1e-8:
        raise FEConvergenceError(f'Independent force residual {relative_residual:.3e} exceeds 1e-8')
    return result


def _locations(result: FEResult, theta):
    fractional = np.mod(np.asarray(theta)-result.mesh_rotation, 2*np.pi) * result.n_elements/(2*np.pi)
    element = np.floor(fractional).astype(int) % result.n_elements
    return element, fractional - np.floor(fractional)


def recover_section(result: FEResult, theta):
    """Recover constant axial strain and linearly varying outward curvature."""
    element, fraction = _locations(result, theta)
    force = result.local_forces[element]
    p = result.params
    eps0 = force[..., 3] / (p.youngs*p.area)
    kappa = ((1-fraction)*force[..., 2] - fraction*force[..., 5])/(p.youngs*p.inertia)
    return eps0, kappa


def recover_gauge_strain(result: FEResult, theta, z, gauge_length: float = 0.01):
    """Exact arc-average of piecewise beam strain over a centered finite gauge.

    Fixed outward depth is used along the gauge.  Subintervals are split at
    every element boundary including the periodic seam; midpoint integration
    is exact because each unloaded straight beam has linear curvature.
    """
    theta, z = np.broadcast_arrays(np.asarray(theta, dtype=float), np.asarray(z, dtype=float))
    if not np.isfinite(theta).all() or not np.isfinite(z).all() or np.any(abs(z) > result.params.thickness/2 + 1e-12):
        raise ValueError('Gauge coordinates must be finite and inside the section')
    if not 0 <= gauge_length < 2*np.pi*result.params.radius:
        raise ValueError('Gauge length must be nonnegative and shorter than circumference')
    if gauge_length == 0:
        eps0, kappa = recover_section(result, theta)
        return eps0-z*kappa
    values = np.empty(theta.size)
    angular_length = gauge_length/result.params.radius
    step = 2*np.pi/result.n_elements
    for i, (angle, depth) in enumerate(zip(theta.ravel(), z.ravel())):
        left, right = angle-angular_length/2, angle+angular_length/2
        first = int(np.floor((left-result.mesh_rotation)/step))+1
        last = int(np.ceil((right-result.mesh_rotation)/step))
        inner = result.mesh_rotation + np.arange(first, last)*step
        cuts = np.r_[left, inner, right]
        eps0, curvature = recover_section(result, (cuts[:-1]+cuts[1:])/2)
        values[i] = np.sum((eps0-depth*curvature)*np.diff(cuts))/angular_length
    return values.reshape(theta.shape)


def recover_displacement(result: FEResult, theta):
    """Axial-linear / transverse-Hermite beam interpolation, in global axes."""
    angles = np.asarray(theta)
    elements, fractions = _locations(result, angles)
    output = np.empty((angles.size, 2))
    for row, (i, s) in enumerate(zip(elements.ravel(), fractions.ravel())):
        j = (i+1) % result.n_elements
        length, transform, _ = _element_matrices(result.params, result.xy[i], result.xy[j])
        dofs = transform @ np.r_[result.displacements[i], result.displacements[j]]
        axial = (1-s)*dofs[0]+s*dofs[3]
        transverse = ((1-3*s*s+2*s**3)*dofs[1] + length*(s-2*s*s+s**3)*dofs[2]
                      +(3*s*s-2*s**3)*dofs[4] + length*(-s*s+s**3)*dofs[5])
        output[row] = transform[:2, :2].T @ [axial, transverse]
    return output.reshape(angles.shape+(2,))


def _beam_check():
    p, length, axial_force, transverse_force = RingParameters(), 2.0, 1200.0, 1000.0
    ops.wipe()
    ops.model('basic', '-ndm', 2, '-ndf', 3)
    ops.node(1, 0, 0)
    ops.node(2, length, 0)
    ops.fix(1, 1, 1, 1)
    ops.geomTransf('Linear', 1)
    ops.element('elasticBeamColumn', 1, 1, 2, p.area, p.youngs, p.inertia, 1)
    ops.timeSeries('Linear', 1)
    ops.pattern('Plain', 1, 1)
    ops.load(2, axial_force, transverse_force, 0)
    _static_analysis()
    disp = np.array(ops.nodeDisp(2))
    exact = np.array([axial_force*length/(p.youngs*p.area),
                      transverse_force*length**3/(3*p.youngs*p.inertia),
                      transverse_force*length**2/(2*p.youngs*p.inertia)])
    forces = np.array(ops.eleResponse(1, 'localForce'))
    fractions = np.linspace(0, 1, 7)
    curvature_force = ((1-fractions)*forces[2]-fractions*forces[5])/(p.youngs*p.inertia)
    curvature_exact = -transverse_force*length*(1-fractions)/(p.youngs*p.inertia)
    # v'' from Hermite shape functions; positive outward z is negative local y.
    curvature_dof = -((-6+12*fractions)*0/length**2 + (-4+6*fractions)*0/length
                      + (6-12*fractions)*disp[1]/length**2 + (-2+6*fractions)*disp[2]/length)
    strain_error = 0.0
    for z in [-p.thickness/2, p.thickness/2]:
        actual = forces[3]/(p.youngs*p.area)-z*curvature_force
        expected = axial_force/(p.youngs*p.area)-z*curvature_exact
        strain_error = max(strain_error, float(np.max(abs(actual-expected))))
    error = float(np.max(abs(disp-exact)/abs(exact)))
    recovery_error = float(np.max(abs(curvature_force-curvature_dof)))
    return dict(pass_=error < 1e-10 and strain_error < 1e-12 and recovery_error < 1e-12,
                displacement_relative_error=error, fiber_strain_absolute_error=strain_error,
                force_vs_shape_curvature_error_1_m=recovery_error,
                tip_displacement=disp.tolist(), analytic_tip_displacement=exact.tolist())


def _support_direction_check():
    angle, kr, kt = 0.73, 7e6, 3e5
    radial = np.array([np.cos(angle), np.sin(angle)])
    tangent = np.array([-np.sin(angle), np.cos(angle)])
    errors = []
    for direction, stiffness in [(radial, kr), (tangent, kt)]:
        ops.wipe()
        ops.model('basic', '-ndm', 2, '-ndf', 3)
        ops.node(1, 0, 0)
        ops.node(2, 0, 0)
        ops.fix(1, 1, 1, 1)
        ops.fix(2, 0, 0, 1)
        ops.uniaxialMaterial('Elastic', 1, kr)
        ops.uniaxialMaterial('Elastic', 2, kt)
        ops.element('zeroLength', 1, 1, 2, '-mat', 1, 2, '-dir', 1, 2,
                    '-orient', *radial, 0.0, *tangent, 0.0)
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)
        ops.load(2, *direction, 0.0)
        _static_analysis()
        actual = np.array(ops.nodeDisp(2))[:2]
        errors.append(float(np.linalg.norm(actual-direction/stiffness)*stiffness))
    return dict(pass_=max(errors) < 1e-10, radial_and_tangential_relative_errors=errors,
                stiffness_units='(N/m^3) * width_m * tributary_arc_m = N/m')


def save_fe_result(result: FEResult, path, gauge_theta=None, gauge_z=None):
    arrays = dict(theta_rad=result.theta, node_xy_m=result.xy,
                  displacement_m_rad=result.displacements, support_reaction_N_Nm=result.reactions,
                  element_local_force_N_Nm=result.local_forces, applied_load_N_Nm=result.loads,
                  radial_stiffness_N_m=result.radial_stiffness,
                  tangential_stiffness_N_m=result.tangential_stiffness,
                  params_json=np.array(json.dumps(asdict(result.params))),
                  diagnostics_json=np.array(json.dumps(result.diagnostics)))
    if gauge_theta is not None:
        arrays.update(gauge_theta_rad=gauge_theta, gauge_depth_m=gauge_z,
                      gauge_strain=recover_gauge_strain(result, gauge_theta, gauge_z),
                      gauge_length_m=np.array(0.01))
    np.savez_compressed(path, **arrays)


def run_fe_validation(output_dir) -> dict:
    """Run the specified small validation matrix; write measured values only."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checks = {'single_beam': _beam_check(), 'support_direction': _support_direction_check()}
    ring, support, _ = assemble_frame(RingParameters(), 64)
    eigen_free = np.linalg.eigvalsh(ring.toarray())
    eigen_supported = np.linalg.eigvalsh((ring+support).toarray())
    _, xy, _, _ = _geometry(RingParameters(), 64)
    rigid = np.zeros((192, 3))
    rigid[0::3, 0] = 1
    rigid[1::3, 1] = 1
    rigid[0::3, 2], rigid[1::3, 2], rigid[2::3, 2] = -xy[:, 1], xy[:, 0], 1
    rigid_residual = float(np.linalg.norm(ring@rigid)/(np.linalg.norm(ring.data)*np.linalg.norm(rigid)))
    free_count = int(np.count_nonzero(abs(eigen_free)/max(abs(eigen_free)) < 1e-10))
    checks['rigid_modes'] = dict(pass_=free_count == 3 and eigen_supported.min() > 0 and rigid_residual < 1e-12,
                                free_rigid_mode_count=free_count, rigid_mode_relative_residual=rigid_residual,
                                free_smallest_eigenvalues=eigen_free[:5].tolist(),
                                supported_min_eigenvalue=float(eigen_supported.min()))
    rng = np.random.default_rng(20260926)
    gauge_theta = rng.uniform(0, 2*np.pi, 64)
    quantiles = rng.uniform(size=64)
    cases = {
        'normal': RingParameters(),
        'softened': RingParameters(reduction=0.6, sector_center=0.37),
        'low_boundary': RingParameters(thickness=.27, youngs=27e9, radial_modulus=40e6,
                                       pressure=1e5, harmonic=.3, phase=1.31, reduction=.3, sector_center=5.91),
        'high_boundary': RingParameters(thickness=.33, youngs=33e9, radial_modulus=60e6,
                                        pressure=3e5, harmonic=.3, phase=.73, reduction=.6, sector_center=2.77),
    }
    all_mesh = {}
    independent = {}
    snapshots = {}
    for name, params in cases.items():
        depth = np.sqrt((params.radius-params.thickness/2)**2 + quantiles*(2*params.radius*params.thickness))-params.radius
        records = []
        previous = None
        for n in [64, 128, 256, 512]:
            result = solve_ring(params, n)
            strain = recover_gauge_strain(result, gauge_theta, depth)
            displacement = recover_displacement(result, gauge_theta)
            row = dict(elements=n, **result.diagnostics)
            if previous is not None:
                row['displacement_change'] = float(np.max(np.linalg.norm(displacement-previous[0], axis=1))/max(np.max(np.linalg.norm(displacement, axis=1)), 1e-12))
                row['gauge_strain_change'] = float(np.max(abs(strain-previous[1])/np.maximum(abs(strain), 1e-4)))
                row['mesh_pass'] = row['displacement_change'] < .01 and row['gauge_strain_change'] < .02
            records.append(row)
            previous = displacement, strain
            snapshots[name, n] = result
        all_mesh[name] = records
        candidate = snapshots[name, 128]
        stiffness, foundation, force = assemble_frame(params, 128)
        independent_u = spsolve(stiffness+foundation, force.ravel()).reshape(-1, 3)
        error = float(np.linalg.norm(candidate.displacements-independent_u)/np.linalg.norm(independent_u))
        independent[name] = dict(pass_=error < 1e-8, relative_displacement_error=error)
    # Choose a mesh only if the following refinement confirms it for all cases.
    selected = None
    for coarse, fine in [(64, 128), (128, 256), (256, 512)]:
        if coarse < 128:
            continue
        if all(next(row for row in rows if row['elements'] == fine)['mesh_pass'] for rows in all_mesh.values()):
            selected = coarse
            break
    checks['independent_frame'] = dict(pass_=all(x['pass_'] for x in independent.values()), cases=independent)
    checks['mesh_convergence'] = dict(pass_=selected is not None, cases=all_mesh,
                                    strain_reference_floor=1e-4,
                                    displacement_limit=.01, gauge_strain_limit=.02)
    equilibrium_ok = all(row['relative_equilibrium_residual'] < 1e-8 and row['relative_energy_error'] < .001
                         and row['relative_force_balance'] < 1e-8 and row['relative_moment_balance'] < 1e-8
                         for rows in all_mesh.values() for row in rows)
    checks['equilibrium_energy'] = dict(pass_=equilibrium_ok)
    p = RingParameters()
    total_stiffness = [snapshots['normal', n].radial_stiffness.sum() for n in [64, 128, 256, 512]]
    exact_total = p.radial_modulus*p.width*2*np.pi*p.radius
    support_error = float(np.max(abs(np.array(total_stiffness)/exact_total-1)))
    checks['support_length_scaling'] = dict(pass_=support_error < 1e-12,
                                           total_radial_stiffness_N_m=total_stiffness,
                                           exact_total_N_m=exact_total)
    angle = .431
    base = snapshots['softened', 128]
    rotated = solve_ring(replace(base.params, phase=base.params.phase+angle,
                                 sector_center=base.params.sector_center+angle), 128, angle)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotation_error = float(np.linalg.norm(rotated.displacements[:, :2]-base.displacements[:, :2]@rotation.T)/np.linalg.norm(base.displacements[:, :2]))
    checks['rotation_covariance'] = dict(pass_=rotation_error < 1e-8, relative_displacement_error=rotation_error)
    thin_rows = []
    for thickness in [.3, .15, .075]:
        p = RingParameters(thickness=thickness, harmonic=0)
        result = solve_ring(p, 512)
        radial_u = np.sum(result.displacements[:, :2]*result.xy/p.radius, axis=1).mean()
        membrane = -p.pressure*p.width/(p.youngs*p.area/p.radius**2+p.radial_modulus*p.width)
        thin_rows.append(dict(thickness_m=thickness, radius_ratio=thickness/p.radius,
                              computed_radial_displacement_m=float(radial_u),
                              membrane_approximation_m=membrane,
                              relative_error=float(abs(radial_u/membrane-1))))
    checks['thin_ring_membrane'] = dict(pass_=all(row['relative_error'] < .01 for row in thin_rows), cases=thin_rows)
    if selected is not None:
        example = snapshots['softened', selected]
        depth = np.sqrt((example.params.radius-example.params.thickness/2)**2 + quantiles*(2*example.params.radius*example.params.thickness))-example.params.radius
        save_fe_result(example, output/'structure_example.npz', gauge_theta, depth)
        save_fe_result(snapshots['normal', selected], output/'structure_normal.npz', gauge_theta, depth)
    # Public JSON uses the ordinary pass key, while avoiding Python keyword syntax.
    def normalize(value):
        if isinstance(value, dict):
            return {('pass' if key == 'pass_' else key): normalize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item) for item in value]
        return value
    checks = normalize(checks)
    report = dict(selected_elements=selected, pass_=all(item['pass'] for item in checks.values()),
                  checks=checks, solver_version=ops.version(), fixed_seed=20260926,
                  port_mesh_check='performed by the network integration layer on the same gauge coordinates')
    report = normalize(report)
    (output/'fe_validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
