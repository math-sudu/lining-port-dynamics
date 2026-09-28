"""Independent plane-stress annulus with bilinear quadrilateral elements.

The elastic support and pressure act on the outer boundary. Multiplication by
R / R_outer preserves the beam model's nominal force/stiffness per angle in
the circular limit. Unlike the beam's centerline springs, tangential boundary
traction also acts at the outer radius; this modelling distinction is explicit.
The inner boundary is traction free. No OpenSees matrices enter this solver.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

from .fe import RingParameters


def plane_stress_matrix(youngs: float, poisson: float) -> np.ndarray:
    """Voigt strain is (epsilon_xx, epsilon_yy, 2 epsilon_xy)."""
    return youngs / (1-poisson**2) * np.array(
        [[1., poisson, 0.], [poisson, 1., 0.], [0., 0., (1-poisson)/2]])


def shape(xi, eta):
    xi, eta = np.broadcast_arrays(xi, eta)
    n = np.stack(((1-xi)*(1-eta), (1+xi)*(1-eta),
                  (1+xi)*(1+eta), (1-xi)*(1+eta)), axis=-1) / 4
    dx = np.stack((-(1-eta), 1-eta, 1+eta, -(1+eta)), axis=-1) / 4
    dy = np.stack((-(1-xi), -(1+xi), 1+xi, 1-xi), axis=-1) / 4
    return n, np.stack((dx, dy), axis=-1)


def strain_matrix(coords: np.ndarray, dshape: np.ndarray):
    jac = np.einsum('...ni,...nj->...ij', coords, dshape)
    determinant = np.linalg.det(jac)
    if np.any(determinant <= 0):
        raise ValueError('The continuum mesh contains an inverted element')
    gradient = dshape @ np.linalg.inv(jac)
    b = np.zeros(gradient.shape[:-2] + (3, 8))
    b[..., 0, 0::2] = gradient[..., :, 0]
    b[..., 1, 1::2] = gradient[..., :, 1]
    b[..., 2, 0::2] = gradient[..., :, 1]
    b[..., 2, 1::2] = gradient[..., :, 0]
    return b, determinant


@dataclass
class ContinuumResult:
    params: RingParameters
    n_theta: int
    n_radial: int
    poisson: float
    xy: np.ndarray
    cells: np.ndarray
    displacements: np.ndarray
    diagnostics: dict


def solve_continuum(params: RingParameters, n_theta: int = 256,
                    n_radial: int = 4, poisson: float = .2) -> ContinuumResult:
    p = params
    if n_theta < 16 or n_radial < 1 or not -1 < poisson < .5:
        raise ValueError('Invalid continuum mesh or Poisson ratio')
    if min(p.youngs, p.thickness, p.width, p.radial_modulus, p.tangential_ratio) <= 0:
        raise ValueError('Positive material, geometry and both support directions are required')
    if not 0 < p.thickness < 2*p.radius or not 0 <= p.reduction < 1:
        raise ValueError('Invalid annulus or support reduction')
    angles = np.arange(n_theta) * 2*np.pi/n_theta
    radii = np.linspace(p.radius-p.thickness/2, p.radius+p.thickness/2, n_radial+1)
    xy = (radii[:, None, None] * np.stack((np.cos(angles), np.sin(angles)), axis=-1)).reshape(-1, 2)
    i = np.tile(np.arange(n_theta), n_radial)
    j = np.repeat(np.arange(n_radial), n_theta)
    cells = np.column_stack((j*n_theta+i, (j+1)*n_theta+i,
                             (j+1)*n_theta+(i+1)%n_theta, j*n_theta+(i+1)%n_theta))
    coords = xy[cells]
    constitutive = plane_stress_matrix(p.youngs, poisson)
    stiffness = np.zeros((len(cells), 8, 8))
    gauss = (-1/np.sqrt(3), 1/np.sqrt(3))
    for xi in gauss:
        for eta in gauss:
            _, dn = shape(xi, eta)
            b, det = strain_matrix(coords, dn)
            stiffness += np.einsum('eia,ij,ejb->eab', b, constitutive, b) * (det*p.width)[:, None, None]
    dofs = (2*cells[..., None]+np.arange(2)).reshape(-1, 8)
    rows = [np.repeat(dofs, 8, axis=1).ravel()]
    cols = [np.tile(dofs, (1, 8)).ravel()]
    vals = [stiffness.ravel()]
    load = np.zeros(2*len(xy))
    boundary = np.column_stack((n_radial*n_theta+np.arange(n_theta),
                                n_radial*n_theta+(np.arange(n_theta)+1)%n_theta))
    edge_xy = xy[boundary]
    lengths = np.linalg.norm(edge_xy[:, 1]-edge_xy[:, 0], axis=1)
    edge_dofs = (2*boundary[..., None]+np.arange(2)).reshape(-1, 4)
    kb = np.zeros((n_theta, 4, 4)); fb = np.zeros((n_theta, 4))
    boundary_scale = p.radius/radii[-1]
    for point in gauss:
        n = np.array([(1-point)/2, (1+point)/2])
        xq = np.einsum('n,eni->ei', n, edge_xy)
        radial = xq/np.linalg.norm(xq, axis=1)[:, None]
        tangent = np.column_stack((-radial[:, 1], radial[:, 0]))
        theta = np.arctan2(radial[:, 1], radial[:, 0])
        delta = (theta-p.sector_center+np.pi)%(2*np.pi)-np.pi
        weight = np.where(abs(delta) <= p.sector_width/2,
                          .5*(1+np.cos(2*np.pi*delta/p.sector_width)), 0.)
        kr = p.radial_modulus*(1-p.reduction*weight)*boundary_scale
        kt = p.radial_modulus*p.tangential_ratio*boundary_scale
        support = kr[:, None, None]*np.einsum('ei,ej->eij', radial, radial)
        support += kt*np.einsum('ei,ej->eij', tangent, tangent)
        measure = lengths*p.width/2
        kb += np.einsum('n,m,eij,e->enimj', n, n, support, measure).reshape(-1, 4, 4)
        pressure = p.pressure*(1+p.harmonic*np.cos(2*(theta-p.phase)))*boundary_scale
        fb += np.einsum('n,ei,e,e->eni', n, -radial, pressure, measure).reshape(-1, 4)
    rows.append(np.repeat(edge_dofs, 4, axis=1).ravel())
    cols.append(np.tile(edge_dofs, (1, 4)).ravel()); vals.append(kb.ravel())
    np.add.at(load, edge_dofs.ravel(), fb.ravel())
    matrix = coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(len(load), len(load))).tocsr()
    u = spsolve(matrix, load)
    if not np.isfinite(u).all():
        raise RuntimeError('The continuum solution is nonfinite')
    residual = matrix @ u-load
    return ContinuumResult(p, n_theta, n_radial, poisson, xy, cells, u.reshape(-1, 2), {
        'model': 'Q4 plane stress; 2x2 integration; outer bilateral support; free inner boundary',
        'dofs': len(u), 'boundary_scale_R_over_Router': boundary_scale,
        'relative_equilibrium_residual': float(np.linalg.norm(residual)/np.linalg.norm(load)),
        'strain_energy_J': float(.5*u@(matrix@u)),
        'max_displacement_m': float(np.linalg.norm(u.reshape(-1, 2), axis=1).max()),
        'field_coordinates': 'Radial fraction and polar angle map the circular annulus to its polygonal Q4 approximation.'})


def point_fields(result: ContinuumResult, theta, depth):
    """Interpolate displacement and physical strain on the polygonal annulus.

    Matching polar angle/radial fraction supplies corresponding material points.
    Their Cartesian positions converge to the circular points with mesh refinement.
    """
    theta, depth = np.broadcast_arrays(np.asarray(theta, float), np.asarray(depth, float))
    if not np.isfinite(theta).all() or not np.isfinite(depth).all() or np.any(abs(depth) > result.params.thickness/2+1e-12):
        raise ValueError('Field coordinates must be finite and inside the annulus')
    nt, nr, thickness = result.n_theta, result.n_radial, result.params.thickness
    angular_step = 2*np.pi/nt
    reduced = np.mod(theta.ravel(), 2*np.pi)
    angular_cell = np.floor(reduced/angular_step).astype(int)%nt
    local_angle = reduced-angular_cell*angular_step
    fraction = np.sin(local_angle)/(np.sin(local_angle)+np.sin(angular_step-local_angle))
    radial_fraction = (depth.ravel()/thickness+.5)*nr
    radial_cell = np.clip(np.floor(radial_fraction).astype(int), 0, nr-1)
    xi, eta = 2*(radial_fraction-radial_cell)-1, 2*fraction-1
    n, dn = shape(xi, eta)
    cells = result.cells[radial_cell*nt+angular_cell]
    b, _ = strain_matrix(result.xy[cells], dn)
    values = result.displacements[cells]
    strain = np.einsum('eij,ej->ei', b, values.reshape(-1, 8))
    tangent = np.column_stack((-np.sin(theta.ravel()), np.cos(theta.ravel())))
    hoop = strain[:, 0]*tangent[:, 0]**2+strain[:, 1]*tangent[:, 1]**2+strain[:, 2]*tangent[:, 0]*tangent[:, 1]
    return {'xy': np.einsum('en,eni->ei', n, result.xy[cells]).reshape(theta.shape+(2,)),
            'displacement': np.einsum('en,eni->ei', n, values).reshape(theta.shape+(2,)),
            'strain_voigt': strain.reshape(theta.shape+(3,)),
            'hoop_strain': hoop.reshape(theta.shape)}


def recover_continuum_gauge(result: ContinuumResult, theta, depth, gauge_length: float = .01):
    """Arc-average circumferential strain, splitting at element boundaries."""
    theta, depth = np.broadcast_arrays(np.asarray(theta, float), np.asarray(depth, float))
    if not 0 <= gauge_length < 2*np.pi*result.params.radius:
        raise ValueError('Gauge length must be nonnegative and shorter than circumference')
    if gauge_length == 0:
        return point_fields(result, theta, depth)['hoop_strain']
    step = 2*np.pi/result.n_theta
    length = gauge_length/result.params.radius
    points, weights = np.polynomial.legendre.leggauss(3)
    values = np.empty(theta.size)
    for idx, (angle, z) in enumerate(zip(theta.ravel(), depth.ravel())):
        left, right = angle-length/2, angle+length/2
        cuts = np.r_[left, np.arange(int(np.floor(left/step))+1, int(np.ceil(right/step)))*step, right]
        center, half = (cuts[:-1]+cuts[1:])/2, np.diff(cuts)/2
        query = center[:, None]+half[:, None]*points
        eps = point_fields(result, query, z)['hoop_strain']
        values[idx] = np.sum((eps@weights)*half)/length
    return values.reshape(theta.shape)
