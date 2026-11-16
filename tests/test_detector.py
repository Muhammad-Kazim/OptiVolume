"""Detector chain: tile/pixel rounding, PixelIntegrator, Intensity (A2, A8)."""
import warnings

import pytest
import torch

from optical_volume import optics

WL = 500e-9


def test_tile_rounding():
    mask = optics.BinaryPhaseMask(WL, 12e-6, (100e-9, 100e-9), (240, 240), 500e-9, 1.0, 1.46, seed=0)
    assert mask.tile_size_px == (120, 120)
    with pytest.warns(UserWarning, match="tile size"):
        optics.BinaryPhaseMask(WL, 12e-6, (0.7e-6, 0.7e-6), (64, 64), 500e-9, 1.0, 1.46, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert optics.utils.samples_per(6e-6, 2e-6) == 3


def _block_sum(x, f):
    H, W = x.shape[-2:]
    return x[..., : H // f * f, : W // f * f].reshape(*x.shape[:-2], H // f, f, W // f, f).sum((-3, -1))


def test_pixel_integrator_integer_matches_block_sum():
    g = torch.Generator().manual_seed(0)
    real = torch.rand(2, 3, 13, 12, generator=g)
    field = torch.complex(torch.rand(13, 12, generator=g), torch.rand(13, 12, generator=g))
    for x in (real, field):
        out = optics.utils.PixelIntegrator(2)(x)
        torch.testing.assert_close(out, _block_sum(x, 2))
    # pixel_size / spacing gives the same factor
    assert optics.utils.PixelIntegrator(pixel_size=6e-6, spacing=2e-6).factor == (3.0, 3.0)


def test_pixel_integrator_fractional_conserves_sum():
    x = torch.rand(10, 15, generator=torch.Generator().manual_seed(1), dtype=torch.float64)
    out = optics.utils.PixelIntegrator(2.5)(x)
    assert out.shape == (4, 6)
    assert out.sum().item() == pytest.approx(x.sum().item(), rel=1e-6)


def test_pixel_integrator_fractional_matches_upsampled():
    x = torch.rand(10, 15, generator=torch.Generator().manual_seed(2), dtype=torch.float64)
    up = x.repeat_interleave(2, 0).repeat_interleave(2, 1)  # each sample -> 2x2 half-samples
    reference = _block_sum(up, 5) / 4
    torch.testing.assert_close(optics.utils.PixelIntegrator(2.5)(x), reference)


def test_pixel_integrator_offset_gradient():
    x = torch.rand(12, 12, generator=torch.Generator().manual_seed(3), dtype=torch.float64)
    w = torch.rand(4, 4, generator=torch.Generator().manual_seed(4), dtype=torch.float64)
    pix = optics.utils.PixelIntegrator(pixel_size=5e-6, spacing=2e-6, offset=(0.7e-6, -0.3e-6)).double()
    (pix(x) * w).sum().backward()
    h = 1e-9
    for axis in range(2):
        with torch.no_grad():
            pix.offset[axis] += h
            up = (pix(x) * w).sum()
            pix.offset[axis] -= 2 * h
            down = (pix(x) * w).sum()
            pix.offset[axis] += h
        assert pix.offset.grad[axis].item() == pytest.approx(((up - down) / (2 * h)).item(), rel=1e-6)


def test_intensity_then_integrate_vs_field():
    i, j = torch.meshgrid(torch.arange(8), torch.arange(8), indexing="ij")
    checker = (1 - 2 * ((i + j) % 2)).to(torch.complex64)  # +-1 checkerboard
    intensity_first = optics.utils.PixelIntegrator(2)(optics.utils.Intensity()(checker))
    field_first = optics.utils.Intensity()(optics.utils.PixelIntegrator(2)(checker))
    torch.testing.assert_close(intensity_first, torch.full((4, 4), 4.0))
    torch.testing.assert_close(field_first, torch.zeros(4, 4))
