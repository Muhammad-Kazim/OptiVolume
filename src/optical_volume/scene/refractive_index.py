"""Refractive-index maps: the refractive index inside a shape as a function of position.

A shape's geometry (its occupancy) and its refractive index are separate features. The
refractive index is a scalar by default; a map makes it vary inside the shape. Maps are
evaluated at **object-frame** coordinates, metres relative to the shape's centre and
rotated with it, so the map moves and turns with the shape. Any ``nn.Module`` (or callable)
that maps points ``(..., 3)`` to values ``(...)`` can serve as a map.
"""
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def _triple(value):
    if isinstance(value, (int, float)):
        return (float(value),) * 3
    value = tuple(float(v) for v in value)
    if len(value) != 3:
        raise ValueError(f"expected 3 values, got {len(value)}")
    return value


class VoxelRI(nn.Module):
    """Refractive index on a voxel grid in the shape's frame, interpolated trilinearly.

    Parameters
    ----------
    values : Tensor
        ``(nx, ny, nz)`` refractive indices (parameter).
    spacing : float or tuple of float
        Voxel size ``(dx, dy, dz)`` in metres.

    Notes
    -----
    The grid is centred on the shape's centre: voxel ``(i, j, k)`` sits at object coordinates
    ``((i - (nx-1)/2) dx, (j - (ny-1)/2) dy, (k - (nz-1)/2) dz)``. Points beyond the grid take
    the nearest edge value, so make the grid cover the shape; outside the shape the occupancy
    hides the map anyway.
    """

    def __init__(self, values: Tensor, spacing):
        super().__init__()
        values = torch.as_tensor(values)
        if values.dim() != 3:
            raise ValueError(f"values must be (nx, ny, nz), got shape {tuple(values.shape)}")
        self.values = nn.Parameter(values.float())
        self.spacing = _triple(spacing)

    def forward(self, points: Tensor) -> Tensor:
        n = self.values.shape
        coords = []
        for axis in range(3):
            half = (n[axis] - 1) * self.spacing[axis] / 2
            # grid_sample's normalised coordinates: -1 and 1 are the first and last voxel centres.
            coords.append(points[..., axis] / half if half > 0 else torch.zeros_like(points[..., axis]))
        grid = torch.stack(coords, dim=-1).reshape(1, -1, 1, 1, 3).to(self.values.dtype)
        # grid_sample reads (x, y, z) as (W, H, D): store the values as (nz, ny, nx).
        sampled = F.grid_sample(self.values.permute(2, 1, 0)[None, None], grid, mode="bilinear",
                                padding_mode="border", align_corners=True)
        return sampled.reshape(points.shape[:-1])


class FunctionRI(nn.Module):
    """Refractive index given by a function of object-frame coordinates.

    Parameters
    ----------
    fn : callable
        Maps points ``(..., 3)`` (metres, relative to the shape's centre, rotated with it) to
        refractive indices ``(...)``. For trainable parameters, pass an ``nn.Module`` as the
        shape's ``RI`` directly instead.

    Examples
    --------
    A gradient index along the shape's own z axis, 1.40 at the centre, +0.01 per micrometre:

    >>> ri = FunctionRI(lambda p: 1.40 + 1e4 * p[..., 2])
    """

    def __init__(self, fn: Callable[[Tensor], Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, points: Tensor) -> Tensor:
        return self.fn(points)
