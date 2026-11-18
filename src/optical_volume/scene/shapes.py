"""Analytic shapes defined by signed distance functions (SDFs).

Every shape is placed by a centre and a rotation (see :func:`.utils.world_to_object`; the
object is rotated by R, right-handed) and described by its signed distance ``sdf(points)``:
negative inside, positive outside, in metres. Its **occupancy** is a soft step of the SDF,

    occupancy = sigmoid(-sdf / softness),

with ``softness`` in metres: the 10-90 % edge width is ``2 ln 9 * softness ~ 4.39 * softness``
for every shape and size. A shape's refractive index is a separate feature (``RI``): a scalar,
or a map that varies inside the shape (:mod:`.refractive_index`).
:class:`~optical_volume.scene.volume.CollectionVolume` combines occupancy and refractive index.
"""
import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor, nn

from .refractive_index import FunctionRI
from .utils import from_axis_angle, world_to_object

#: 10-90 % width of sigmoid(-d / s) in units of s: 2 ln 9.
EDGE_WIDTH_PER_SOFTNESS = 2 * 2.1972245773362196


def default_softness(spacing) -> float:
    """Softness (metres) that gives a 10-90 % edge about one voxel wide on a grid with ``spacing``."""
    return min(float(d) for d in spacing) / EDGE_WIDTH_PER_SOFTNESS


def _axis_angle_quat(rotation_axis, rotation_angle) -> Tensor:
    """Unit quaternion for a rotation of ``rotation_angle`` degrees about ``rotation_axis``."""
    axis = torch.as_tensor(rotation_axis, dtype=torch.float32)
    angle = torch.as_tensor(rotation_angle, dtype=torch.float32)
    return from_axis_angle(axis, torch.deg2rad(angle))


def _rotation_rows(n: int, rotation_axes, rotation_angles):
    """Per-shape (axis, angle) pairs for the make_* factories; defaults to no rotation."""
    axes = torch.tensor([[0., 0., 1.]]).expand(n, 3) if rotation_axes is None else torch.as_tensor(rotation_axes)
    angles = torch.zeros(n) if rotation_angles is None else torch.as_tensor(rotation_angles)
    if axes.shape != (n, 3) or angles.shape != (n,):
        raise ValueError(f"rotation_axes must be ({n}, 3) and rotation_angles ({n},), "
                         f"got {tuple(axes.shape)} and {tuple(angles.shape)}")
    return list(zip(axes, angles))


def _safe_norm(v: Tensor) -> Tensor:
    """Euclidean norm over the last axis with a finite gradient at 0 (where it is 0)."""
    sq = (v * v).sum(-1)
    return torch.sqrt(sq.clamp_min(1e-30)) * (sq > 0)


def _param(value) -> nn.Parameter:
    value = value if isinstance(value, Tensor) else torch.as_tensor(value, dtype=torch.float32)
    return nn.Parameter(value if value.is_floating_point() else value.float())


class Shape(nn.Module):
    """Base class: a posed signed distance function with a refractive index.

    Parameters
    ----------
    center : Tensor
        Centre ``(3,)`` in metres (parameter).
    RI : float, Tensor, nn.Module or callable
        Refractive index inside the shape. A number or scalar tensor is a parameter ``RI``. A
        map (:class:`~.refractive_index.VoxelRI`, :class:`~.refractive_index.FunctionRI`, any
        ``nn.Module`` or callable of object-frame points ``(..., 3)``) makes it vary inside the
        shape; it is stored as ``ri_map`` and ``RI`` is ``None``.
    softness : float, optional
        Edge softness in metres (10-90 % edge width 4.39 * softness). ``None`` (default): set
        by the volume the shape is placed in, about one voxel (:func:`default_softness`).
    rotation_axis : tuple of float
        Rotation axis ``(3,)``.
    rotation_angle : float
        Rotation angle in degrees; the shape is rotated by it about the axis (right-handed).
    """

    def __init__(self, center, RI, softness: Optional[float] = None,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__()
        self.center = _param(center)
        self._set_ri(RI)
        self.softness = None if softness is None else float(softness)
        self.quat = nn.Parameter(_axis_angle_quat(rotation_axis, rotation_angle))

    def _set_ri(self, RI):
        if isinstance(RI, nn.Module) or (callable(RI) and not isinstance(RI, Tensor)):
            self.ri_map = RI if isinstance(RI, nn.Module) else FunctionRI(RI)
            self.register_parameter("RI", None)
        else:
            self.ri_map = None
            self.RI = _param(RI)

    def local(self, points: Tensor) -> Tensor:
        """Object-frame coordinates ``(..., 3)`` of world points (backward map)."""
        return world_to_object(points, self.center, self.quat)

    def sdf(self, points: Tensor) -> Tensor:
        """Signed distance (metres) at world points ``(..., 3)``: < 0 inside, > 0 outside."""
        return self._sdf(self.local(points))

    def _sdf(self, p: Tensor) -> Tensor:
        raise NotImplementedError

    def occupancy(self, points: Tensor, softness: Optional[float] = None) -> Tensor:
        """Occupancy in [0, 1] at world points ``(..., 3)``.

        ``softness`` (metres) is used only if the shape has none of its own.
        """
        s = self.softness if self.softness is not None else softness
        if s is None:
            raise ValueError("this shape has softness=None: pass softness= (metres), or evaluate it "
                             "through a volume, which sets it from the voxel spacing")
        return torch.sigmoid(-self.sdf(points) / s)

    def forward(self, points: Tensor, softness: Optional[float] = None) -> Tensor:
        """Occupancy at world points; see :meth:`occupancy`."""
        return self.occupancy(points, softness)

    def refractive_index(self, points: Tensor) -> Tensor:
        """Refractive index at world points ``(..., 3)``: the scalar ``RI`` broadcast, or the
        map evaluated at the object-frame coordinates (so it moves and rotates with the shape)."""
        if self.ri_map is not None:
            return self.ri_map(self.local(points))
        return self.RI.expand(points.shape[:-1])

    def set_quat(self, rotation_axis, rotation_angle):
        """Set the orientation in place: axis ``(3,)`` and angle in degrees."""
        # Copy in place: replacing the Parameter would detach it from existing optimizers.
        with torch.no_grad():
            self.quat.copy_(_axis_angle_quat(rotation_axis, rotation_angle))
        return self


class Sphere(Shape):
    """Sphere of ``radius`` (metres, parameter). SDF ``|p| - radius``. See :class:`Shape`."""

    def __init__(self, center, radius, RI, softness: Optional[float] = None,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__(center, RI, softness, rotation_axis, rotation_angle)
        self.radius = _param(radius)

    def _sdf(self, p):
        return _safe_norm(p) - self.radius


class Cube(Shape):
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


def make_cubes(centers: Tensor, lengths: Tensor, RIs: Tensor, rotation_axes: Tensor = None,
               rotation_angles: Tensor = None, softness: Optional[float] = None):
    """Build one :class:`Cube` per row.

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
    return [Cube(centers[i], lengths[i], RIs[i], softness=softness, rotation_axis=axis, rotation_angle=angle)
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
