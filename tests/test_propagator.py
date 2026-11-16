"""Propagators: padding, batch dimensions and the BPM source field (B1-B3, C2)."""
import math

import pytest
import torch

from optical_volume import optics

WL = 500e-9
N0 = 1.33
SPACING = (200e-9, 200e-9, 200e-9)


def _random_volume(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return N0 + 0.05 * torch.rand(*shape, generator=g)


def _random_field(*shape, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.exp(1j * torch.rand(*shape, generator=g)).to(torch.complex64)


def test_volume_propagator_padding():
    # In a uniform medium (n = n0) every slice only adds the phase n0*k0*dz.
    nz = 8
    ri = torch.full((32, 32, nz), N0)
    expected = torch.exp(torch.tensor(1j * N0 * 2 * math.pi / WL * SPACING[2] * nz)).to(torch.complex64)
    for padding in (None, 8):
        out = optics.VolumePropagator(WL, SPACING, N0, padding=padding)(ri)
        assert out.shape == (32, 32)
        torch.testing.assert_close(out, torch.full_like(out, expected), rtol=1e-4, atol=1e-4)


def test_padding_zero():
    field = _random_field(16, 16)
    free = [optics.FreeSpacePropagator(WL, SPACING[:2], 2e-6, padding=p) for p in (None, 0)]
    torch.testing.assert_close(free[1](field), free[0](field))

    ri = _random_volume(16, 16, 4)
    bpm = [optics.VolumePropagator(WL, SPACING, N0, padding=p) for p in (None, 0)]
    torch.testing.assert_close(bpm[1](ri), bpm[0](ri))


def test_batched_input():
    fields = _random_field(2, 16, 16)
    free = optics.FreeSpacePropagator(WL, SPACING[:2], 2e-6, padding=4)
    out = free(fields)
    assert out.shape == (2, 16, 16)
    torch.testing.assert_close(out, torch.stack([free(f) for f in fields]))

    vols = _random_volume(2, 16, 16, 4)
    bpm = optics.VolumePropagator(WL, SPACING, N0, padding=4)
    out = bpm(vols)
    assert out.shape == (2, 16, 16)
    torch.testing.assert_close(out, torch.stack([bpm(v) for v in vols]))


def test_volume_propagator_source_argument():
    bpm = optics.VolumePropagator(WL, SPACING, N0)
    # No source set: any lateral size works, nothing is cached.
    assert bpm(_random_volume(16, 16, 4)).shape == (16, 16)
    assert bpm(_random_volume(8, 8, 4)).shape == (8, 8)
    assert bpm.source_field is None

    # A source passed to forward is differentiable.
    source = _random_field(8, 8).requires_grad_(True)
    bpm(_random_volume(8, 8, 4), source=source).abs().sum().backward()
    assert source.grad is not None

    # A stored source of the wrong size is an error, not silently replaced.
    bpm.set_source_field(_random_field(16, 16))
    with pytest.raises(ValueError, match="lateral shape"):
        bpm(_random_volume(8, 8, 4))


def _grating(period, n=64, dx=100e-9):
    x = torch.arange(n) * dx
    return torch.exp(1j * 2 * math.pi * x / period)[:, None].expand(n, n).to(torch.complex64)


def test_evanescent_suppressed():
    # lambda = 500 nm, dx = 100 nm: a 200 nm grating is evanescent, a 1.6 um grating propagates.
    fine, coarse = _grating(200e-9), _grating(1.6e-6)
    free = optics.FreeSpacePropagator(WL, (100e-9, 100e-9), 1e-6)
    assert free(fine).abs().max() < 1e-4  # was 1 before A3; float32 rounding leaves ~1e-5
    torch.testing.assert_close(free(coarse).abs(), coarse.abs(), rtol=1e-4, atol=1e-4)

    # 'decay': exp(-kappa*d) forward; removed (not amplified) for d < 0.
    decay = optics.FreeSpacePropagator(WL, (100e-9, 100e-9), 20e-9, evanescent="decay")
    kappa = 2 * math.pi * math.sqrt(1 / 200e-9**2 - 1 / WL**2)
    assert decay(fine).abs().mean().item() == pytest.approx(math.exp(-kappa * 20e-9), rel=1e-3)
    decay.distance.data.fill_(-20e-9)
    decay.set_direction("forward")
    assert decay(fine).abs().max() < 1e-4

    with pytest.raises(ValueError):
        optics.FreeSpacePropagator(WL, (100e-9, 100e-9), 1e-6, evanescent="keep")
