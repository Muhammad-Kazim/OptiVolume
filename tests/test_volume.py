"""VoxelVolume construction defaults (B6)."""
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
