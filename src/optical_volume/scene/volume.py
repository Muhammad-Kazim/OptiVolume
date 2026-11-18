import warnings

import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, Tuple, List

from .base import default_softness
from .rotations import from_axis_angle, world_to_object

class Grid(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float]):
        super().__init__()

        self.nx, self.ny, self.nz = shape
        self.dx, self.dy, self.dz = spacing

        x = torch.arange(self.nx) * self.dx
        y = torch.arange(self.ny) * self.dy
        z = torch.arange(self.nz) * self.dz

        grid = torch.meshgrid(x, y, z, indexing="ij")

        self.register_buffer('grid', torch.stack(grid, dim=-1))


class CollectionVolume(Grid):
    """Refractive-index volume built from analytic shapes on a regular grid.

    Parameters
    ----------
    shape : tuple of int
        ``(nx, ny, nz)`` samples.
    spacing : tuple of float
        Sample spacing ``(dx, dy, dz)`` in metres; sample ``(i, j, k)`` sits at
        ``(i*dx, j*dy, k*dz)``.
    n_medium : float
        Background refractive index.

    Notes
    -----
    Shapes are combined in the order they were added, each painting over what is there:
    ``n <- n * (1 - occupancy) + n_shape * occupancy``, with each shape's index ``n_shape``. Where shapes overlap, the later one wins.
    Shapes with ``softness=None`` get :func:`~optical_volume.scene.shapes.default_softness`
    of this grid, an edge about one voxel wide.
    """

    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], n_medium: float):
        super().__init__(shape, spacing)

        self.n_medium = n_medium
        self.shapes = nn.ModuleList()

    def add(self, shape: List):
        self.shapes.extend(shape)

    @property
    def default_softness(self) -> float:
        """Softness (metres) given to shapes without their own: 10-90 % edge about one voxel."""
        return default_softness((self.dx, self.dy, self.dz))

    def forward(self):
        spacing = min(self.dx, self.dy, self.dz)
        field = torch.full_like(self.grid[..., 0], self.n_medium)
        for shape in self.shapes:
            if shape.softness is not None and shape.softness < 0.01 * spacing:
                warnings.warn(f"{type(shape).__name__} softness {shape.softness:.3g} m is below 1 % of the "
                              f"voxel spacing ({spacing:.3g} m): its edge is a hard step and gradients with "
                              "respect to its geometry vanish. Use softness=None for about one voxel.",
                              stacklevel=2)
            occ = shape.occupancy(self.grid, softness=self.default_softness)
            # Occupancy and refractive index stay separate, so nothing divides by RI (A18).
            field = field * (1 - occ) + shape.refractive_index(self.grid) * occ

        return field


class VoxelVolume(nn.Module):
    """Refractive-index voxel array, rotatable and sampled onto any grid.

    Parameters
    ----------
    shape : tuple of int
        ``(nx, ny, nz)`` voxels of the object box.
    spacing : tuple of float
        Voxel size ``(dx, dy, dz)`` in metres. Voxel ``(i, j, k)`` sits at
        ``(i*dx, j*dy, k*dz)``.
    volume : Tensor, optional
        Refractive indices of shape ``shape``. Default: filled with ``n_medium``.
    n_medium : float
        Background refractive index of the medium around the object (keyword only,
        required). Fills an empty volume.
    rotation_axis : tuple of float
        Rotation axis ``(3,)``.
    rotation_angle : float
        Rotation angle in degrees. The object is rotated by this angle about the axis
        (right-handed), the same convention as the shapes; see
        :func:`optical_volume.scene.rotations.world_to_object`.
    object_center : tuple of float, optional
        Rotation centre in metres. Default: the centre of the voxel box,
        ``((nx-1)*dx/2, (ny-1)*dy/2, (nz-1)*dz/2)``.

    Notes
    -----
    ``forward(grid)`` samples the rotated volume at the points ``grid (..., 3)`` (metres)
    with trilinear interpolation (a backward map, so there are no holes). Points that map
    outside the voxel box are the background ``n_medium``: the difference ``volume - n_medium`` is
    interpolated with zero padding and ``n_medium`` added back.
    """

    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], volume: Tensor = None,
                 *, n_medium: float, rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.,
                 object_center: Tuple = None) -> None:
        super().__init__()

        self.shape = tuple(shape)
        self.spacing = tuple(spacing)
        self.n_medium = n_medium

        self.volume = torch.nn.Parameter(torch.ones(self.shape).float()*self.n_medium) if volume is None else self.set_volume(volume)
        self.quat = torch.nn.Parameter(from_axis_angle(torch.tensor(rotation_axis).float(), 
                                                       torch.deg2rad(torch.tensor(rotation_angle))))
        if object_center is None:
            # Samples sit at 0..(n-1)*d, so this is the middle of the box that forward normalises to.
            object_center = [(n - 1) * d / 2 for n, d in zip(self.shape, self.spacing)]
        self.register_buffer('center', torch.as_tensor(object_center, dtype=torch.float32))

    def set_volume(self, volume: Tensor):

        assert tuple(volume.shape) == tuple(self.shape)
        volume = torch.nn.Parameter(volume.float())

        return volume

    def forward(self, grid: Tensor):

        # Backward map: each grid point reads the voxels at R^T (g - c) + c, so the object is
        # rotated by R, the same convention as the analytic shapes.
        pts = world_to_object(grid, self.center, self.quat) + self.center

        # normalize the points to the range [0, 1] for grid_sample
        pts_normalized = pts.clone()
        pts_normalized[..., 0] = (pts[..., 0] / ((self.shape[0] - 1) * self.spacing[0])) * 2 - 1
        pts_normalized[..., 1] = (pts[..., 1] / ((self.shape[1] - 1) * self.spacing[1])) * 2 - 1
        pts_normalized[..., 2] = (pts[..., 2] / ((self.shape[2] - 1) * self.spacing[2])) * 2 - 1

        # grid_sample expects the grid in the shape (N, D, H, W, 3), so we need to add a batch dimension
        pts_normalized = pts_normalized.unsqueeze(0)

        # Interpolate the contrast to the background with zero padding, so points outside the
        # box are exactly n_medium. Repeating the edge voxels ('border') would smear an object that
        # touches the box edge out to the grid boundary (A9).
        contrast = (self.volume - self.n_medium).permute(2, 1, 0)

        sampled = F.grid_sample(
            contrast.unsqueeze(0).unsqueeze(0),  # (N, C, D, H, W)
            pts_normalized,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        return sampled.squeeze() + self.n_medium
