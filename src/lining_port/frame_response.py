"""Static response of an asymmetric plane frame with piecewise elastic members.

Members carry joint loads only. Splitting at every stiffness discontinuity makes
the Euler--Bernoulli displacement and section-force interpolation exact within
each prismatic segment. Coordinates, forces and moduli use SI units.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import solve
from scipy.sparse.csgraph import shortest_path

from .fe import _element_matrices


@dataclass(frozen=True)
class BeamSection:
    youngs: float = 30e9
    width: float = .25
    thickness: float = .35

    @property
    def area(self):
        return self.width * self.thickness

    @property
    def inertia(self):
        return self.width * self.thickness**3 / 12


class PlaneFrame:
    def __init__(self, points, members, fixed_nodes, section=BeamSection(), patches=()):
        self.joints = np.asarray(points, dtype=float)
        self.members = np.asarray(members, dtype=int)
        self.section = section
        self.lengths = np.linalg.norm(self.joints[self.members[:, 1]] - self.joints[self.members[:, 0]], axis=1)
        points = self.joints.tolist()
        self.elements = []
        for member, (start, end) in enumerate(self.members):
            selected = [p for p in patches if p[0] == member]
            boundaries = sorted({0., 1., *(b for _, a, z, _ in selected for b in (a, z))})
            nodes = [int(start)]
            for x in boundaries[1:-1]:
                nodes.append(len(points))
                points.append(((1 - x) * self.joints[start] + x * self.joints[end]).tolist())
            nodes.append(int(end))
            for a, b, first, last in zip(boundaries[:-1], boundaries[1:], nodes[:-1], nodes[1:]):
                factor = np.prod([1 - loss for _, lo, hi, loss in selected if lo <= (a + b) / 2 <= hi])
                local_section = replace(section, youngs=section.youngs * factor)
                length, transform, stiffness = _element_matrices(local_section, np.array(points[first]), np.array(points[last]))
                dofs = np.r_[np.arange(3 * first, 3 * first + 3), np.arange(3 * last, 3 * last + 3)]
                self.elements.append((member, a, b, length, dofs, transform, stiffness))
        self.points = np.asarray(points)
        self.stiffness = np.zeros((3 * len(points), 3 * len(points)))
        for _, _, _, _, dofs, transform, stiffness in self.elements:
            self.stiffness[np.ix_(dofs, dofs)] += transform.T @ stiffness @ transform
        fixed = (3 * np.asarray(fixed_nodes)[:, None] + np.arange(3)).ravel()
        self.free = np.setdiff1d(np.arange(3 * len(points)), fixed)

    def displacement(self, joint_loads):
        loads = np.zeros((len(self.stiffness), np.shape(joint_loads)[-1]))
        loads[:3 * len(self.joints)] = np.asarray(joint_loads).reshape(3 * len(self.joints), -1)
        displacement = np.zeros_like(loads)
        displacement[self.free] = solve(self.stiffness[np.ix_(self.free, self.free)], loads[self.free], assume_a="pos")
        return displacement

    def strain(self, displacement, gauge_member, fraction, depth, gauge_length=.0002):
        """Average the member-axis fiber strain over the actual gauge length."""
        answer = np.zeros((len(fraction), displacement.shape[-1]))
        for i, (member, center, z) in enumerate(zip(gauge_member, fraction, depth)):
            half = gauge_length / (2 * self.lengths[member])
            lo, hi = center - half, center + half
            if lo < 0 or hi > 1 or gauge_length <= 0:
                raise ValueError("The complete finite gauge must lie inside one member")
            for m, a, b, length, dofs, transform, _ in self.elements:
                left, right = max(lo, a), min(hi, b)
                if m != member or right <= left:
                    continue
                s = ((left + right) / 2 - a) / (b - a)
                curvature = np.array([0., (-6 + 12 * s) / length**2, (-4 + 6 * s) / length,
                                      0., (6 - 12 * s) / length**2, (-2 + 6 * s) / length])
                axial = np.array([-1., 0., 0., 1., 0., 0.]) / length
                answer[i] += (right - left) / (hi - lo) * ((axial - z * curvature) @ transform @ displacement[dofs])
        return answer

    def response(self, joint_loads, installation):
        return self.strain(self.displacement(joint_loads), installation["member"],
                           installation["fraction"], installation["depth_m"])

    def path_distances(self, gauge_member, fraction, depth):
        """Conductor routes along member centerlines, with a depth lead at each end."""
        count = len(fraction)
        graph = np.full((len(self.joints) + count, len(self.joints) + count), np.inf)
        np.fill_diagonal(graph, 0.)
        for m, (start, end) in enumerate(self.members):
            gauges = np.flatnonzero(gauge_member == m)
            gauges = gauges[np.argsort(fraction[gauges])]
            nodes = np.r_[start, len(self.joints) + gauges, end]
            position = np.r_[0., fraction[gauges], 1.]
            length = np.diff(position) * self.lengths[m]
            graph[nodes[:-1], nodes[1:]] = length
            graph[nodes[1:], nodes[:-1]] = length
        distance = shortest_path(graph, directed=False)
        cells = distance[len(self.joints):, len(self.joints):] + np.abs(depth[:, None]) + np.abs(depth[None, :])
        np.fill_diagonal(cells, 0.)
        # Instrument at joint 0; module heads lie at a chosen gauge's centerline.
        return cells, distance[0, len(self.joints):], distance[len(self.joints):, len(self.joints):]


def asymmetric_frame(patches=(), modulus_scale=1.):
    points = [[0., 0.], [4., 0.], [9., 0.], [0., 3.], [4., 3.6], [9., 2.8]]
    members = [[0, 3], [1, 4], [2, 5], [3, 4], [4, 5]]
    return PlaneFrame(points, members, [0, 1, 2], BeamSection(youngs=30e9 * modulus_scale), patches)


def joint_load_basis(rank):
    loads = np.zeros((6, 3, 6))
    for j in range(3):
        loads[j + 3, 0, 2 * j] = 10_000.
        loads[j + 3, 1, 2 * j + 1] = -30_000.
    if rank == 6:
        return loads
    if rank == 2:
        return np.einsum("ncr,rk->nck", loads, np.array([
            [0., 1.], [1., 0.], [0., .6], [1.5, 0.], [0., 1.1], [1.2, 0.]]))
    raise ValueError("The frame study defines two or six normal load modes")


def random_installation(seed, gauges_per_member=12):
    rng = np.random.default_rng(seed)
    count = 5 * gauges_per_member
    member = np.repeat(np.arange(5), gauges_per_member)
    fraction = rng.uniform(.01, .99, count)
    depth = rng.uniform(-.175, .175, count)
    return {"member": member, "fraction": fraction, "depth_m": depth}
