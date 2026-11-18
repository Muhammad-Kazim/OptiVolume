"""Shapes: factories and setters (B4, C4); SDF occupancy, softness in metres (A13); composition (A18)."""
import math
import warnings

import pytest
import torch

from optical_volume import scene
from optical_volume.scene.utils import from_axis_angle

AXES = torch.tensor([[0., 0., 1.], [1., 0., 0.], [0., 1., 1.]])
ANGLES = torch.tensor([0., 30., 90.])


def _expected_quat(axis, angle_deg):
    return from_axis_angle(axis.float(), torch.deg2rad(angle_deg.float()))


def test_make_shapes():
    centers = torch.rand(3, 3) * 1e-5
    ris = torch.tensor([1.4, 1.45, 1.5])
    built = [
        scene.shapes.make_cubes(centers, torch.full((3,), 2e-6), ris,
                                rotation_axes=AXES, rotation_angles=ANGLES),
        scene.shapes.make_ellipsoids(centers, torch.full((3, 3), 2e-6), ris,
                                     rotation_axes=AXES, rotation_angles=ANGLES),
    ]
    for shapes in built:
        assert len(shapes) == 3
        for shape, axis, angle in zip(shapes, AXES, ANGLES):
            torch.testing.assert_close(shape.quat.detach(), _expected_quat(axis, angle))
    # Defaults: no rotation.
    cube = scene.shapes.make_cubes(centers, torch.full((3,), 2e-6), ris)[0]
    torch.testing.assert_close(cube.quat.detach(), _expected_quat(AXES[0], ANGLES[0]))


def test_setter_keeps_parameter_identity():
    # Unequal semi-axes and a soft edge, so the output depends smoothly on orientation.
    shape = scene.shapes.Ellipsoid(torch.zeros(3), torch.tensor([2e-6, 1e-6, 0.5e-6]),
                                   torch.tensor(1.5), softness=0.5e-6)
    quat = shape.quat
    optimizer = torch.optim.Adam(shape.parameters(), lr=1e-2)

    shape.set_quat((1., 0., 0.), 45.)
    assert shape.quat is quat
    torch.testing.assert_close(quat.detach(), _expected_quat(torch.tensor([1., 0., 0.]), torch.tensor(45.)))

    before = quat.detach().clone()
    grid = torch.rand(6, 6, 6, 3, generator=torch.Generator().manual_seed(0)) * 4e-6 - 2e-6
    optimizer.zero_grad()
    (shape(grid) * grid[..., 0]).sum().backward()
    optimizer.step()
    assert not torch.equal(quat.detach(), before)


C = torch.tensor([2e-6, 2e-6, 2e-6])


def _shapes(softness=None):
    """One of each shape at C, rotated so that the object frame differs from the world frame."""
    kw = dict(softness=softness, rotation_axis=(1., 1., 0.), rotation_angle=25.)
    return {
        "sphere": scene.shapes.Sphere(C.clone(), torch.tensor(1e-6), torch.tensor(1.5), **kw),
        "cube": scene.shapes.Cube(C.clone(), torch.tensor([1.6e-6, 1.0e-6, 0.8e-6]), torch.tensor(1.5), **kw),
        "ellipsoid": scene.shapes.Ellipsoid(C.clone(), torch.tensor([1.2e-6, 0.8e-6, 0.5e-6]), torch.tensor(1.5), **kw),
    }


def _surface_points(name, shape):
    """World points on the analytic surface (and the object-frame directions they lie in)."""
    with torch.no_grad():
        R = scene.utils.to_matrix(shape.quat).reshape(3, 3)
        if name == "sphere":
            local = torch.tensor([[1., 0, 0], [0, 1., 0], [0.6, 0, 0.8]]) * shape.radius
        elif name == "cube":
            h = shape.length / 2
            local = torch.stack([torch.tensor([h[0], 0., 0.]), torch.tensor([0.3 * h[0], h[1], -0.2 * h[2]]),
                                 torch.tensor([0., 0., -h[2]])])
        else:
            r = shape.radii
            u = torch.tensor([[1., 0, 0], [0, 0, 1.], [0.6, 0.8, 0]])
            local = u / torch.linalg.norm(u / r, dim=-1, keepdim=True)     # on the ellipsoid along u
        return local @ R.T + shape.center


@pytest.mark.parametrize("name", ["sphere", "cube", "ellipsoid"])
def test_shape_surface_half_occupancy(name):
    shape = _shapes(softness=0.05e-6)[name]
    surface = _surface_points(name, shape)
    with torch.no_grad():
        on = shape(surface)
        inside = shape(shape.center + 0.25 * (surface - shape.center))
        outside = shape(shape.center + 2.0 * (surface - shape.center))
    torch.testing.assert_close(on, torch.full_like(on, 0.5), atol=1e-4, rtol=0)
    assert (inside > 0.99).all() and (outside < 0.01).all()
    # Without its own softness, a shape evaluated outside a volume needs one.
    with pytest.raises(ValueError):
        _shapes()[name](surface)


@pytest.mark.parametrize("radius", [1e-6, 10e-6])
def test_softness_in_metres(radius):
    # The 10-90 % edge width is 2 ln 9 x softness, whatever the size (the old m^2 softness of
    # Sphere made it depend on the radius).
    s = 50e-9
    sphere = scene.shapes.Sphere(torch.zeros(3), torch.tensor(radius), torch.tensor(1.5), softness=s)
    r = radius + torch.linspace(-0.5e-6, 0.5e-6, 20001)
    pts = torch.stack([r, torch.zeros_like(r), torch.zeros_like(r)], dim=-1)
    with torch.no_grad():
        occ = sphere(pts)
    r90 = r[torch.argmin((occ - 0.9).abs())]
    r10 = r[torch.argmin((occ - 0.1).abs())]
    assert (r10 - r90).item() == pytest.approx(2 * math.log(9) * s, rel=1e-3)


@pytest.mark.parametrize("name", ["sphere", "cube", "ellipsoid"])
def test_shape_gradients(name):
    # At the default softness (set by the volume), geometry and pose get finite, non-zero gradients.
    shape = _shapes()[name]
    vol = scene.CollectionVolume((40, 40, 40), (0.1e-6,) * 3, 1.33)
    vol.add([shape])
    w = torch.rand(40, 40, 40, generator=torch.Generator().manual_seed(0)) - 0.5
    (vol() * w).sum().backward()
    size = {"sphere": "radius", "cube": "length", "ellipsoid": "radii"}[name]
    for pname in ("center", size, "quat", "RI"):
        g = getattr(shape, pname).grad
        assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0, f"{name}.{pname}"


def test_small_softness_warns():
    vol = scene.CollectionVolume((8, 8, 8), (0.1e-6,) * 3, 1.33)
    vol.add([scene.shapes.Sphere(torch.full((3,), 0.4e-6), torch.tensor(0.2e-6), torch.tensor(1.5), softness=1e-12)])
    with pytest.warns(UserWarning, match="softness"):
        vol()
    vol = scene.CollectionVolume((8, 8, 8), (0.1e-6,) * 3, 1.33)
    vol.add([scene.shapes.Sphere(torch.full((3,), 0.4e-6), torch.tensor(0.2e-6), torch.tensor(1.5))])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        vol()


def test_collection_ri_to_zero():
    # Composition no longer divides by RI (A18): RI = 0 gives a finite volume and finite gradients,
    # and the occupancy doesn't depend on RI.
    vol = scene.CollectionVolume((24, 24, 24), (0.1e-6,) * 3, 1.33)
    sphere = scene.shapes.Sphere(torch.full((3,), 1.2e-6), torch.tensor(0.6e-6), torch.tensor(0.0))
    vol.add([sphere])
    ri = vol()
    assert torch.isfinite(ri).all() and ri.min().item() == pytest.approx(0.0, abs=1e-3)
    (ri * torch.linspace(0, 1, 24)[:, None, None]).sum().backward()
    for p in (sphere.center, sphere.radius, sphere.RI):
        assert torch.isfinite(p.grad).all() and p.grad.abs().sum() > 0
    with torch.no_grad():
        occ0 = sphere.occupancy(vol.grid, vol.default_softness)
        sphere.RI.fill_(1.5)
        torch.testing.assert_close(sphere.occupancy(vol.grid, vol.default_softness), occ0)


def test_collection_overlap_order():
    # Where shapes overlap, the one added later sets the refractive index.
    vol = scene.CollectionVolume((24, 24, 24), (0.1e-6,) * 3, 1.33)
    big = scene.shapes.Sphere(torch.full((3,), 1.2e-6), torch.tensor(0.8e-6), torch.tensor(1.40))
    small = scene.shapes.Sphere(torch.full((3,), 1.2e-6), torch.tensor(0.4e-6), torch.tensor(1.50))
    vol.add([big, small])
    with torch.no_grad():
        assert vol()[12, 12, 12].item() == pytest.approx(1.50, abs=1e-4)
        vol.shapes = torch.nn.ModuleList([small, big])
        assert vol()[12, 12, 12].item() == pytest.approx(1.40, abs=1e-4)
