"""VoxelVolume construction defaults (B6) and sampling outside the box (A9)."""
import pytest
import torch

from optical_volume import scene

SHAPE = (8, 6, 4)
SPACING = (100e-9, 120e-9, 150e-9)


def _grid():
    return scene.volume.Grid(SHAPE, SPACING).grid


def test_voxel_volume_defaults():
    volume = 1.4 + 0.1 * torch.rand(SHAPE, generator=torch.Generator().manual_seed(0))
    grid = _grid()

    still = scene.VoxelVolume(SHAPE, SPACING, volume, n0=1.33)
    torch.testing.assert_close(still(grid), volume)

    # 180 deg about z maps the grid onto itself only if the default centre is the box centre.
    turned = scene.VoxelVolume(SHAPE, SPACING, volume, n0=1.33, rotation_angle=180.)
    torch.testing.assert_close(turned(grid), volume.flip(0, 1), atol=1e-5, rtol=1e-5)

    empty = scene.VoxelVolume(SHAPE, SPACING, n0=1.33)
    torch.testing.assert_close(empty(grid), torch.full(SHAPE, 1.33))


def test_voxel_volume_requires_n0_keyword():
    with pytest.raises(TypeError):
        scene.VoxelVolume(SHAPE, SPACING)


def test_voxel_outside_is_background():
    # An object filling its whole box, rotated by 30 degrees: grid points that map outside the
    # box are exactly the background, inside it the object (A9).
    n, d, n0 = 32, 1e-7, 1.33
    shape, spacing = (n, n, 3), (d, d, d)
    vol = scene.VoxelVolume(shape, spacing, torch.full(shape, 1.38), n0=n0, rotation_angle=30.)
    grid = scene.volume.Grid(shape, spacing).grid
    with torch.no_grad():
        ri = vol(grid)
    # Where does each grid point land in the box (in voxel units)?
    c = torch.tensor([(n - 1) * d / 2, (n - 1) * d / 2, d])
    th = torch.deg2rad(torch.tensor(30.))
    R = torch.tensor([[torch.cos(th), -torch.sin(th), 0.], [torch.sin(th), torch.cos(th), 0.], [0., 0., 1.]])
    p = (((grid - c) @ R + c) / d)[..., :2]                   # rotation about z: z stays on its voxels
    outside = ((p < -1) | (p > n)).any(dim=-1)                # more than one voxel outside the box
    inside = ((p > 1) & (p < n - 2)).all(dim=-1)
    assert outside.any() and inside.any()
    assert torch.equal(ri[outside], torch.full_like(ri[outside], n0))
    torch.testing.assert_close(ri[inside], torch.full_like(ri[inside], 1.38))
