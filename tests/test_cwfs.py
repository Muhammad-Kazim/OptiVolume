"""CodedPhaseCamV3: mask plane and shared distance (A1)."""
import torch

from optical_volume import optics, pipelines

WL = 500e-9
SPACING = (1.5e-6, 1.5e-6)
DIST = 2.5e-3  # DIST_PM_IM in CodedPhaseCamV3


def _camera():
    # No padding, so propagation is periodic and back-and-forth is exact.
    return pipelines.CodedPhaseCamV3(wavelength=WL, spacing=SPACING, shape=(64, 64), padding=None)


def _field(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.exp(1j * 0.5 * torch.rand(64, 64, generator=g)).to(torch.complex64)


def test_cwfs_mask_plane():
    cam = _camera()
    mask = cam.pcam_FM.elements[1]
    seen = []
    mask.register_forward_hook(lambda module, inp, out: seen.append(inp[0].detach()))
    field = _field()
    with torch.no_grad():
        cam(field)
        expected = optics.FreeSpacePropagator(WL, SPACING, -DIST)(field)
    torch.testing.assert_close(seen[0], expected, rtol=1e-4, atol=1e-4)

    # A flat mask: the sensor sees the image-plane field itself.
    with torch.no_grad():
        mask.height.zero_()
        img = cam(field)
        direct = cam.pcam_FM.elements[3](field).abs() ** 2
    torch.testing.assert_close(img, direct, rtol=1e-4, atol=1e-4 * direct.max().item())


def test_cwfs_distance_shared():
    cam = _camera()
    to_mask, to_sensor = cam.pcam_FM.elements[0], cam.pcam_FM.elements[2]
    assert to_mask.distance is to_sensor.distance
    optimizer = torch.optim.SGD([to_mask.distance], lr=1e-12)
    cam(_field()).sum().backward()
    optimizer.step()
    assert to_mask.distance.item() != DIST
    assert to_sensor.distance.item() == to_mask.distance.item()
