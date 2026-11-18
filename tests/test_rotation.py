"""Rotation convention: shapes and voxel objects turn the same way, right-handed (A10)."""
import torch

from optical_volume import scene

N, D = 48, 1e-7                       # 48 x 48 x 3 grid of 100 nm voxels
SHAPE, SPACING = (N, N, 3), (D, D, D)
CENTER = torch.tensor([(N - 1) * D / 2, (N - 1) * D / 2, D])


def _grid():
    return scene.volume.Grid(SHAPE, SPACING).grid


def _bar(rotation_angle, softness=1e-7):
    """A 3 µm x 1 µm bar along x, rotated about z."""
    return scene.shapes.Cuboid(CENTER.clone(), torch.tensor([3e-6, 1e-6, 1e-6]), torch.tensor(1.0),
                             softness=softness, rotation_axis=(0, 0, 1), rotation_angle=rotation_angle)


def _centroid(w, grid):
    w = w.clamp_min(0)
    return (w[..., None] * grid).sum(dim=(0, 1, 2)) / w.sum()


def test_rotation_right_handed():
    # A marker at +x from the centre ends up at +y after +90 degrees about z, for both kinds of object.
    grid = _grid()
    offset = torch.tensor([1.5e-6, 0., 0.])
    marker = scene.shapes.Cuboid(CENTER + offset, torch.tensor(0.8e-6), torch.tensor(1.0), softness=1e-7)
    with torch.no_grad():
        still = marker(grid)
        vox = scene.VoxelVolume(SHAPE, SPACING, still + 1.0, n0=1.0, rotation_angle=90., object_center=CENTER.tolist())
        moved_vox = _centroid(vox(grid) - 1.0, grid) - CENTER
    expected = torch.tensor([0., 1.5e-6, 0.])
    torch.testing.assert_close(moved_vox, expected, atol=0.05e-6, rtol=0)

    bar = _bar(90.)
    with torch.no_grad():
        occ = bar(grid)[..., 1]
    # The bar's long axis (x) now points along y: it spans y, not x.
    span_x = (occ > 0.5).any(dim=1).sum().item() * D
    span_y = (occ > 0.5).any(dim=0).sum().item() * D
    assert span_y > 2.5e-6 and span_x < 1.5e-6


def test_rotation_convention():
    # An analytic bar rotated by 30 degrees matches a voxelized copy of the unrotated bar rotated
    # by the same 30 degrees in VoxelVolume (up to interpolation at the edges).
    grid = _grid()
    with torch.no_grad():
        analytic = _bar(30.)(grid)
        voxels = _bar(0.)(grid)
        sampled = scene.VoxelVolume(SHAPE, SPACING, voxels + 1.0, n0=1.0, rotation_axis=(0, 0, 1),
                                    rotation_angle=30., object_center=CENTER.tolist())(grid) - 1.0
    err = (sampled - analytic)[..., 1]
    assert err.abs().mean() < 0.01
    assert err.abs().max() < 0.3
    # The opposite rotation is far off: the test can tell the two directions apart.
    with torch.no_grad():
        wrong = _bar(-30.)(grid)
    assert (sampled - wrong)[..., 1].abs().mean() > 0.1
