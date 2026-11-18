"""Analytic shapes: Sphere, Cuboid, Ellipsoid and Slab, and factories for many of them.

See :mod:`.base` for the conventions (signed distance, occupancy, softness in metres).
"""
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .base import EDGE_WIDTH_PER_SOFTNESS, Shape, _param, _rotation_rows, _safe_norm, default_softness


class Sphere(Shape):
    """Sphere of ``radius`` (metres, parameter). SDF ``|p| - radius``. See :class:`Shape`."""

    def __init__(self, center, radius, RI, softness: Optional[float] = None,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__(center, RI, softness, rotation_axis, rotation_angle)
        self.radius = _param(radius)

    def _sdf(self, p):
        return _safe_norm(p) - self.radius


class Cuboid(Shape):
    """Cuboid with edge ``length`` (metres, parameter): a scalar for a cube, or ``(3,)`` sides.

    Exact box SDF. See :class:`Shape`.

    Notes
    -----
    Inside the box the SDF is the largest of the three face distances, so its gradient jumps
    on the planes equidistant from two faces. Both sides are valid gradients; for a grid point
    on such a plane, rounding decides which face it belongs to (e.g. float32 vs float64).
    """

    def __init__(self, center, length, RI, softness: Optional[float] = None,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__(center, RI, softness, rotation_axis, rotation_angle)
        self.length = _param(length)

    def _sdf(self, p):
        q = p.abs() - self.length / 2
        outside = _safe_norm(q.clamp_min(0))
        inside = q.max(dim=-1).values.clamp_max(0)
        return outside + inside


class Ellipsoid(Shape):
    """Ellipsoid with semi-axes ``radii`` ``(3,)`` (metres, parameter). See :class:`Shape`.

    Notes
    -----
    An ellipsoid has no closed-form SDF. This uses the first-order estimate
    ``k0 (k0 - 1) / k1`` with ``k0 = |p / r|`` and ``k1 = |p / r^2|``: exactly 0 on the surface,
    with the right gradient there, so occupancy is 0.5 on the surface and the edge has the
    stated width. Far from the surface it is only approximate, which doesn't matter for the soft
    step. At the centre (0/0) it takes its limit, ``-min(r)`` to ``-max(r)``; ``-min(r)`` is used.
    """

    def __init__(self, center, radii, RI, softness: Optional[float] = None,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__(center, RI, softness, rotation_axis, rotation_angle)
        self.radii = _param(radii)

    def _sdf(self, p):
        k0 = _safe_norm(p / self.radii)
        k1 = _safe_norm(p / self.radii ** 2)
        tiny = 1e-12 / self.radii.min()
        ratio = torch.where(k1 > tiny, k0 / k1.clamp_min(tiny), self.radii.min())
        return (k0 - 1) * ratio


def make_spheres(centers: Tensor, radii: Tensor, RIs: Tensor, softness: Optional[float] = None):
    """Build one :class:`Sphere` per row of ``centers (N, 3)``, ``radii (N,)``, ``RIs (N,)``."""
    assert centers.dim() == 2, "Each row must contain sphere attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.size()[0], "Each row must contain sphere attributes"
    return [Sphere(centers[i], radii[i], RIs[i], softness=softness) for i in range(centers.shape[0])]


def make_cuboids(centers: Tensor, lengths: Tensor, RIs: Tensor, rotation_axes: Tensor = None,
               rotation_angles: Tensor = None, softness: Optional[float] = None):
    """Build one :class:`Cuboid` per row.

    Parameters
    ----------
    centers : Tensor
        ``(N, 3)`` centres in metres.
    lengths : Tensor
        ``(N,)`` edge lengths or ``(N, 3)`` side lengths, in metres.
    RIs : Tensor
        ``(N,)`` refractive indices.
    rotation_axes : Tensor, optional
        ``(N, 3)`` rotation axes; default ``(0, 0, 1)``.
    rotation_angles : Tensor, optional
        ``(N,)`` rotation angles in degrees; default 0.
    softness : float, optional
        Edge softness in metres; default: set by the volume (about one voxel).
    """
    assert centers.dim() == 2, "Each row must contain cube attributes"
    assert RIs.size()[0] == centers.shape[0] == lengths.size()[0], "Each row must contain cube attributes"
    rotations = _rotation_rows(centers.shape[0], rotation_axes, rotation_angles)
    return [Cuboid(centers[i], lengths[i], RIs[i], softness=softness, rotation_axis=axis, rotation_angle=angle)
            for i, (axis, angle) in enumerate(rotations)]


def make_ellipsoids(centers: Tensor, radii: Tensor, RIs: Tensor, rotation_axes: Tensor = None,
                    rotation_angles: Tensor = None, softness: Optional[float] = None):
    """Build one :class:`Ellipsoid` per row.

    Parameters
    ----------
    centers : Tensor
        ``(N, 3)`` centres in metres.
    radii : Tensor
        ``(N, 3)`` semi-axes in metres.
    RIs : Tensor
        ``(N,)`` refractive indices.
    rotation_axes : Tensor, optional
        ``(N, 3)`` rotation axes; default ``(0, 0, 1)``.
    rotation_angles : Tensor, optional
        ``(N,)`` rotation angles in degrees; default 0.
    softness : float, optional
        Edge softness in metres; default: set by the volume (about one voxel).
    """
    assert centers.dim() == radii.dim() == 2, "Each row must contain ellipsoid attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.shape[0], "Each row must contain ellipsoid attributes"
    rotations = _rotation_rows(centers.shape[0], rotation_axes, rotation_angles)
    return [Ellipsoid(centers[i], radii[i], RIs[i], softness=softness, rotation_axis=axis, rotation_angle=angle)
            for i, (axis, angle) in enumerate(rotations)]

class Slab(Shape):
    """Infinite slab of ``thickness`` centred on ``point``, perpendicular to ``normal``.

    Parameters
    ----------
    point : Tensor
        ``(3,)`` point on the slab's mid-plane, in metres (parameter).
    normal : Tensor
        ``(3,)`` slab normal (parameter, any length; normalised when evaluated).
    thickness : Tensor
        Thickness in metres (parameter); the slab spans ``-t/2 <= n.(x - point) <= t/2``.
    RI : float, Tensor, nn.Module or callable
        Refractive index inside the slab, as for :class:`Shape`.
    softness : float, optional
        Edge softness in metres; default: set by the volume (about one voxel).

    Notes
    -----
    SDF ``|n.(x - point)| - t/2``. The slab's own frame has its z axis along the unit normal
    (x and y in the mid-plane), so an RI map varies across the slab along its own z. The
    orientation comes from ``normal``; a slab has no ``quat``.
    """

    def __init__(self, point, normal, thickness, RI, softness: Optional[float] = None):
        nn.Module.__init__(self)
        self.point = _param(point)
        self.normal = _param(normal)
        self.thickness = _param(thickness)
        self._set_ri(RI)
        self.softness = None if softness is None else float(softness)

    def _frame(self) -> Tensor:
        """Rotation whose columns are (u, v, n): two in-plane directions and the unit normal."""
        n = self.normal / torch.linalg.vector_norm(self.normal)
        helper = torch.tensor([1., 0., 0.], dtype=n.dtype, device=n.device)
        if n[0].abs() > 0.9:
            helper = torch.tensor([0., 1., 0.], dtype=n.dtype, device=n.device)
        u = helper - (helper @ n) * n
        u = u / torch.linalg.vector_norm(u)
        return torch.stack([u, torch.linalg.cross(n, u), n], dim=-1)

    def local(self, points: Tensor) -> Tensor:
        return (points - self.point) @ self._frame()

    def _sdf(self, p):
        return p[..., 2].abs() - self.thickness / 2

    def set_quat(self, rotation_axis, rotation_angle):
        raise AttributeError("a Slab is oriented by its normal; set slab.normal instead")
