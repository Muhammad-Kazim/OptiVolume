"""Shape factories and in-place orientation setters (B4, C4)."""
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
                                   torch.tensor(1.5), softness=0.5)
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
