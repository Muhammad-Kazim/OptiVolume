import torch
from torch import Tensor, nn

from typing import Tuple

from .utils import from_axis_angle, world_to_object

# all shapes are voxel volumes of analytical objects


def _axis_angle_quat(rotation_axis, rotation_angle) -> Tensor:
    """Unit quaternion for a rotation of ``rotation_angle`` degrees about ``rotation_axis``."""
    axis = torch.as_tensor(rotation_axis, dtype=torch.float32)
    angle = torch.as_tensor(rotation_angle, dtype=torch.float32)
    return from_axis_angle(axis, torch.deg2rad(angle))


def _set_quat_(module: nn.Module, rotation_axis, rotation_angle) -> nn.Module:
    # Copy in place: replacing the Parameter would detach it from existing optimizers.
    with torch.no_grad():
        module.quat.copy_(_axis_angle_quat(rotation_axis, rotation_angle))
    return module


def _rotation_rows(n: int, rotation_axes, rotation_angles):
    """Per-shape (axis, angle) pairs for the make_* factories; defaults to no rotation."""
    axes = torch.tensor([[0., 0., 1.]]).expand(n, 3) if rotation_axes is None else torch.as_tensor(rotation_axes)
    angles = torch.zeros(n) if rotation_angles is None else torch.as_tensor(rotation_angles)
    if axes.shape != (n, 3) or angles.shape != (n,):
        raise ValueError(f"rotation_axes must be ({n}, 3) and rotation_angles ({n},), "
                         f"got {tuple(axes.shape)} and {tuple(angles.shape)}")
    return list(zip(axes, angles))

class Sphere(nn.Module):
    def __init__(self, center: Tensor, radius: Tensor, RI: Tensor, softness=1e-12,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__()

        self.center = nn.Parameter(center)
        self.radius = nn.Parameter(radius)
        self.RI = nn.Parameter(RI)
        self.softness = softness

        self.quat = nn.Parameter(_axis_angle_quat(rotation_axis, rotation_angle))  

    def forward(self, grid: Tensor):

        pts = world_to_object(grid, self.center, self.quat)

        return self.RI * (1 - _soft_step(pts[..., 0]**2 + pts[..., 1]**2 + pts[..., 2]**2 - self.radius**2, self.softness))

    def set_quat(self, rotation_axis, rotation_angle):
        """Set the orientation in place: axis ``(3,)`` and angle in degrees."""
        return _set_quat_(self, rotation_axis, rotation_angle)


def make_spheres(centers: Tensor, radii: Tensor, RIs: Tensor, softness: float = 1e-12):

    assert centers.dim() == 2, "Each row must contain sphere attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.size()[0], "Each row must contain sphere attributes"

    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Sphere(centers[i], radii[i], RIs[i], softness=softness))

    return shapes


class Cube(nn.Module):
    def __init__(self, center: Tensor, length: Tensor, RI: Tensor, softness=1e-9, 
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__()

        self.center = nn.Parameter(center)
        self.length = nn.Parameter(length)
        self.RI = nn.Parameter(RI)

        self.quat = nn.Parameter(_axis_angle_quat(rotation_axis, rotation_angle))  

        self.softness = softness

    def forward(self, grid: Tensor):

        pts = world_to_object(grid, self.center, self.quat)

        if self.length.dim() > 0 and self.length.size()[0] > 1:
            sx = self.length[0]/2
            sy = self.length[1]/2
            sz = self.length[2]/2
        else:
            sx = sy = sz = self.length/2

        mask_x = _soft_step(pts[..., 0] + sx, softness=self.softness) * (1 - _soft_step(pts[..., 0] - sx, softness=self.softness))
        mask_y = _soft_step(pts[..., 1] + sy, softness=self.softness) * (1 - _soft_step(pts[..., 1] - sy, softness=self.softness))
        mask_z = _soft_step(pts[..., 2] + sz, softness=self.softness) * (1 - _soft_step(pts[..., 2] - sz, softness=self.softness))

        return self.RI * (mask_x * mask_y * mask_z)  # smooth transition between 0 and 1

    def set_quat(self, rotation_axis, rotation_angle):
        """Set the orientation in place: axis ``(3,)`` and angle in degrees."""
        return _set_quat_(self, rotation_axis, rotation_angle)


def make_cubes(centers: Tensor, lengths: Tensor, RIs: Tensor, rotation_axes: Tensor = None,
               rotation_angles: Tensor = None, softness: float = 1e-9):
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
    softness : float
        Edge softness of the sigmoid step, in metres.
    """
    assert centers.dim()  == 2, "Each row must contain cube attributes"
    assert RIs.size()[0] == centers.shape[0] == lengths.size()[0], "Each row must contain cube attributes"

    rotations = _rotation_rows(centers.shape[0], rotation_axes, rotation_angles)
    shapes = []
    for i, (axis, angle) in enumerate(rotations):
        shapes.append(Cube(centers[i], lengths[i], RIs[i], softness=softness,
                           rotation_axis=axis, rotation_angle=angle))

    return shapes


class Ellipsoid(nn.Module):
    def __init__(self, center: Tensor, radii: Tensor, RI: Tensor, softness: float = 1e-12,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):

        super().__init__()

        self.center = nn.Parameter(center)
        self.radii = nn.Parameter(radii)
        self.RI = nn.Parameter(RI)

        self.quat = nn.Parameter(_axis_angle_quat(rotation_axis, rotation_angle))

        self.softness = softness

    def forward(self, grid: Tensor):

        pts = world_to_object(grid, self.center, self.quat)

        rx, ry, rz = self.radii

        # Compute normalized squared distance
        mask_x = (pts[..., 0] / rx) ** 2
        mask_y = (pts[..., 1] / ry) ** 2
        mask_z = (pts[..., 2] / rz) ** 2

        return self.RI * (1 - _soft_step(mask_x + mask_y + mask_z - 1, softness=self.softness))

    def set_quat(self, rotation_axis, rotation_angle):
        """Set the orientation in place: axis ``(3,)`` and angle in degrees."""
        return _set_quat_(self, rotation_axis, rotation_angle)


def make_ellipsoids(centers: Tensor, radii: Tensor, RIs: Tensor, rotation_axes: Tensor = None,
                    rotation_angles: Tensor = None, softness: float = 1e-12):
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
    softness : float
        Softness of the sigmoid step on the normalised radius (dimensionless).
    """
    assert centers.dim() == radii.dim() == 2, "Each row must contain ellipsoid attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.shape[0], "Each row must contain ellipsoid attributes"

    rotations = _rotation_rows(centers.shape[0], rotation_axes, rotation_angles)
    shapes = []
    for i, (axis, angle) in enumerate(rotations):
        shapes.append(Ellipsoid(centers[i], radii[i], RIs[i], softness=softness,
                                rotation_axis=axis, rotation_angle=angle))

    return shapes

class Plane(nn.Module):
    """
        Add a thick plane to the grid. Physical coordinates.

        Args:
            point (float): point that lies on the plane.
            normal (float): normal to the planes.
            RI (float): RI of plane.
            thickness (float, optional): Thickness/2 on either halfspace.
    """
    def __init__(self, point: Tensor, normal: Tensor, thickness: Tensor, RI: float, softness: float = 1e-12):
        super().__init__()

        self.point = nn.Parameter(point)
        self.normal = nn.Parameter(normal)/torch.linalg.vector_norm(normal)
        self.thickness = nn.Parameter(thickness)
        self.RI = nn.Parameter(RI)

        self.softness = softness

    def forward(self, grid: Tensor):
        px, py, pz = self.point
        nx, ny, nz = self.normal

        X = grid.X - px
        Y = grid.Y - py
        Z = grid.Z - pz

        # Plane equation: n . (x - p) = 0
        mask_p = self._soft_step(nx * X + ny * Y + nz * Z, softness=self.softness)
        mask_n = 1 - self._soft_step(nx * X + ny * Y + nz * Z - self.thickness, softness=self.softness)

        return self.RI * (1 - mask_p * mask_n)


def _soft_step(x: Tensor, softness=1e-6): 
    return torch.sigmoid(x / softness)
