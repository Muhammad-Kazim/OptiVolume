import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, Tuple, List

from .utils import from_axis_angle, to_axis_angle, to_matrix

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
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], n_bg: float):
        super().__init__(shape, spacing)

        self.n_bg = n_bg
        self.shapes = nn.ModuleList()

    def add(self, shape: List):
        self.shapes.extend(shape)

    def forward(self):
        field = torch.ones_like(self.grid[..., 0]) * self.n_bg
        for shape in self.shapes:
            mask = shape(self.grid)
            field = field * (1 - mask/shape.RI) + mask

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
        Refractive indices of shape ``shape``. Default: filled with ``n0``.
    n0 : float
        Background refractive index of the medium around the object (keyword only,
        required). Fills an empty volume.
    rotation_axis : tuple of float
        Rotation axis ``(3,)``.
    rotation_angle : float
        Rotation angle in degrees.
    object_center : tuple of float, optional
        Rotation centre in metres. Default: the centre of the voxel box,
        ``((nx-1)*dx/2, (ny-1)*dy/2, (nz-1)*dz/2)``.

    Notes
    -----
    ``forward(grid)`` samples the rotated volume at the points ``grid (..., 3)`` (metres)
    with trilinear interpolation. Points outside the box currently take the value of the
    nearest edge voxel (``padding_mode='border'``).
    """

    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], volume: Tensor = None,
                 *, n0: float, rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.,
                 object_center: Tuple = None) -> None:
        super().__init__()

        self.shape = tuple(shape)
        self.spacing = tuple(spacing)
        self.n0 = n0

        self.volume = torch.nn.Parameter(torch.ones(self.shape).float()*self.n0) if volume is None else self.set_volume(volume)
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

        # Rotation matrix
        R = to_matrix(self.quat).squeeze()

        pts = grid - self.center
        pts = pts @ R.T
        pts = pts + self.center

        # normalize the points to the range [0, 1] for grid_sample
        pts_normalized = pts.clone()
        pts_normalized[..., 0] = (pts[..., 0] / ((self.shape[0] - 1) * self.spacing[0])) * 2 - 1
        pts_normalized[..., 1] = (pts[..., 1] / ((self.shape[1] - 1) * self.spacing[1])) * 2 - 1
        pts_normalized[..., 2] = (pts[..., 2] / ((self.shape[2] - 1) * self.spacing[2])) * 2 - 1

        # grid_sample expects the grid in the shape (N, D, H, W, 3), so we need to add a batch dimension
        pts_normalized = pts_normalized.unsqueeze(0)

        ri = self.volume.permute(2, 1, 0)

        # Sample the RI distribution using grid_sample
        sampled_ri = F.grid_sample(
            ri.unsqueeze(0).unsqueeze(0),  # (N, C, D, H, W)
            pts_normalized,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return sampled_ri.squeeze()
