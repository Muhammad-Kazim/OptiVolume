"""Magnification, Resample and output_spacing (A5)."""
import math

import pytest
import torch

from optical_volume import optics


def _bandlimited(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    spec = torch.randn(n, n, generator=g, dtype=torch.float64) + 1j * torch.randn(n, n, generator=g, dtype=torch.float64)
    f = torch.fft.fftfreq(n)
    return torch.fft.ifft2(spec * ((f[:, None] ** 2 + f[None, :] ** 2) < 0.2 ** 2))


def _energy(field, spacing):
    return (field.abs() ** 2).sum().item() * spacing[0] * spacing[1]


def test_magnification_energy():
    field, s0 = _bandlimited(), (100e-9, 100e-9)
    mag = optics.utils.Magnification(40)
    assert _energy(mag(field), mag.output_spacing(s0)) == pytest.approx(_energy(field, s0), rel=1e-9)
    for method, tol in (("fourier", 1e-9), ("bilinear", 2e-2)):
        system = optics.OpticalSystem((mag, optics.utils.Resample(2, method=method)))
        out = system(field)
        assert out.shape == (64, 64)
        assert _energy(out, system.output_spacing(s0)) == pytest.approx(_energy(field, s0), rel=tol)


def test_resample_fourier_exact():
    n, period = 16, 8
    x = torch.arange(n, dtype=torch.float64)
    field = torch.exp(1j * 2 * math.pi * (x[:, None] + 2 * x[None, :]) / period)
    up = optics.utils.Resample(2.5, method="fourier")(field)
    xo = torch.arange(40, dtype=torch.float64) / 2.5
    expected = torch.exp(1j * 2 * math.pi * (xo[:, None] + 2 * xo[None, :]) / period)
    torch.testing.assert_close(up, expected)
    # Downsampling a band-limited field back gives the original samples.
    torch.testing.assert_close(optics.utils.Resample(0.4, method="fourier")(up), field)


def test_output_spacing_chain():
    system = optics.OpticalSystem((
        optics.utils.Magnification(40),
        optics.utils.Resample(2),
        optics.utils.PixelIntegrator(2.5),
    ))
    assert system.output_spacing(100e-9) == pytest.approx((5e-6, 5e-6))
