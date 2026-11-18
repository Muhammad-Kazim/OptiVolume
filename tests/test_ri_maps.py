"""Refractive-index maps: RI that varies inside a shape, in the shape's own frame."""
import pytest
import torch

from optical_volume import scene

N, D, N_BG = 32, 0.1e-6, 1.33
C = torch.full((3,), (N - 1) * D / 2)        # the grid centre


def _volume(*shapes):
    vol = scene.CollectionVolume((N, N, N), (D, D, D), N_BG)
    vol.add(list(shapes))
    return vol


def test_ri_map_constant_equals_scalar():
    # A uniform voxel map gives the same volume as the scalar refractive index.
    scalar = scene.shapes.Sphere(C.clone(), torch.tensor(1e-6), torch.tensor(1.5))
    mapped = scene.shapes.Sphere(C.clone(), torch.tensor(1e-6), scene.VoxelRI(torch.full((5, 5, 5), 1.5), 0.5e-6))
    assert mapped.RI is None and isinstance(mapped.ri_map, scene.VoxelRI)
    with torch.no_grad():
        torch.testing.assert_close(_volume(mapped)(), _volume(scalar)())


def test_ri_map_rotates_with_shape():
    # A map rising along the cube's own x axis rises along world y after +90 degrees about z.
    values = torch.linspace(1.40, 1.50, 9)[:, None, None].expand(9, 9, 9).clone()
    cube = scene.shapes.Cube(C.clone(), torch.tensor(2.4e-6), scene.VoxelRI(values, 0.3e-6),
                             rotation_axis=(0, 0, 1), rotation_angle=90.)
    with torch.no_grad():
        ri = _volume(cube)()
    c, k = N // 2, 6                                   # 0.6 µm from the centre, well inside the cube
    assert ri[c, c + k, c] - ri[c, c - k, c] == pytest.approx(0.1 * 1.2 / 2.4, abs=2e-3)   # rises along +y
    assert ri[c + k, c, c] - ri[c - k, c, c] == pytest.approx(0.0, abs=1e-4)               # flat along x


def test_ri_map_gradients():
    # Map values and geometry are both trainable; the map's gradient is non-zero only near the shape.
    values = torch.full((16, 16, 16), 1.45)
    sphere = scene.shapes.Sphere(C.clone(), torch.tensor(0.8e-6), scene.VoxelRI(values, 0.2e-6))
    w = torch.rand(N, N, N, generator=torch.Generator().manual_seed(0)) - 0.5
    (_volume(sphere)() * w).sum().backward()
    g = sphere.ri_map.values.grad
    assert torch.isfinite(g).all() and g.abs().sum() > 0
    corner = g[0, 0, 0].abs().item()                    # 2.6 µm from the centre, far outside the sphere
    assert corner < 1e-6 * g.abs().max().item()
    for p in (sphere.center, sphere.radius):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0


def test_function_ri():
    # A plain function of object-frame coordinates: a gradient index along the shape's z.
    sphere = scene.shapes.Sphere(C.clone(), torch.tensor(1.2e-6), lambda p: 1.45 + 1e4 * p[..., 2])
    assert isinstance(sphere.ri_map, scene.FunctionRI)
    with torch.no_grad():
        ri = _volume(sphere)()
    c = N // 2
    # +0.01 per µm along z; the grid centre sits half a voxel below sample c.
    assert ri[c, c, c + 5] - ri[c, c, c - 5] == pytest.approx(0.01, abs=1e-4)
