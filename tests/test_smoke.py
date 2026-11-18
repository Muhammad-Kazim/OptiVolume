"""Fast smoke tests: the package imports and the core blocks run and are differentiable."""
import pytest
import torch

import optical_volume
from optical_volume import optics, scene, pipelines

WL = 500e-9
SPACING = (1e-6, 1e-6, 1e-6)
SHAPE = (32, 32, 16)
N_BG = 1.33


def _sphere_volume():
    sphere = scene.shapes.Sphere(
        torch.tensor([16e-6, 16e-6, 8e-6]), torch.tensor(5e-6), torch.tensor(1.46)
    )
    vol = scene.CollectionVolume(SHAPE, SPACING, N_BG)
    vol.add([sphere])
    return vol, sphere


def test_version():
    assert isinstance(optical_volume.__version__, str)


def test_collection_volume_values():
    vol, _ = _sphere_volume()
    ri = vol()
    assert ri.shape == SHAPE
    assert ri.min().item() == pytest.approx(N_BG, abs=1e-4)
    assert ri.max().item() == pytest.approx(1.46, abs=1e-4)


def test_volume_propagation_is_differentiable():
    vol, sphere = _sphere_volume()
    field = optics.VolumePropagator(WL, SPACING, N_BG)(vol())
    assert field.shape == SHAPE[:2]
    assert field.is_complex()
    field.real.sum().backward()
    assert sphere.radius.grad is not None
    assert sphere.RI.grad is not None


def test_free_space_plane_wave_keeps_unit_amplitude():
    plane_wave = torch.ones(SHAPE[:2], dtype=torch.complex64)
    out = optics.FreeSpacePropagator(WL, SPACING[:2], 20e-6)(plane_wave)
    assert out.shape == plane_wave.shape
    assert torch.allclose(out.abs(), torch.ones_like(out.abs()), atol=1e-4)


def test_lens_passes_dc():
    plane_wave = torch.ones(SHAPE[:2], dtype=torch.complex64)
    out = optics.F4ImagingSystem(WL, SPACING[:2], SHAPE[:2], NA=0.5)(plane_wave)
    assert torch.allclose(out.abs(), torch.ones_like(out.abs()), atol=1e-4)


def test_binary_phase_mask_is_phase_only():
    mask = optics.BinaryPhaseMask(WL, 4e-6, SPACING, SHAPE, height=500e-9, sigma=1.0, RI_PM=1.46)
    plane_wave = torch.ones(SHAPE[:2], dtype=torch.complex64)
    out = mask(plane_wave)
    assert out.shape == SHAPE[:2]
    assert torch.allclose(out.abs(), torch.ones_like(out.abs()), atol=1e-4)


def test_optical_system_chain():
    system = optics.OpticalSystem((
        optics.FreeSpacePropagator(WL, SPACING[:2], 10e-6),
        optics.F4ImagingSystem(WL, SPACING[:2], SHAPE[:2], NA=0.5),
        optics.PixelIntegrator(2),
    ), record=True)
    out = system(torch.ones(SHAPE[:2], dtype=torch.complex64))
    assert out.shape == (SHAPE[0] // 2, SHAPE[1] // 2)
    assert len(system.fields) == 3


def test_coded_phase_camera_runs():
    cam = pipelines.CodedPhaseCamV3(wavelength=WL, spacing=(1.5e-6, 1.5e-6), shape=(64, 64), padding=16)
    img = cam(torch.ones(64, 64, dtype=torch.complex64))
    assert img.dim() == 2
    assert not img.is_complex()
    assert (img >= 0).all()
