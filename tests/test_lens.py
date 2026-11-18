"""F4ImagingSystem pupil: soft edge at the cutoff in frequency samples, NA trainable (A14)."""
import math

import pytest
import torch

from optical_volume import optics

WL, DX = 500e-9, 100e-9


@pytest.mark.parametrize("n", [64, 96])
def test_pupil_edge_at_cutoff(n):
    # Choose NA so that the cutoff NA/wavelength falls on a frequency sample (10 samples out).
    df = 1 / (n * DX)
    lens = optics.F4ImagingSystem(WL, (DX, DX), (n, n), NA=10 * df * WL)
    with torch.no_grad():
        mask = lens.cutoff_mask()
    c = n // 2                                           # zero frequency after fftshift
    assert mask[c + 10, c].item() == pytest.approx(0.5, abs=1e-5)
    # One sample from the cutoff the step is at sigmoid(+-1/softness): with the default softness,
    # 1/(2 ln 9), that is 0.988 and 0.012, a 10-90 % edge about one frequency sample wide.
    edge = 1 / lens.softness
    assert edge == pytest.approx(2 * math.log(9))
    assert mask[c + 9, c].item() == pytest.approx(torch.sigmoid(torch.tensor(edge)).item(), abs=1e-5)
    assert mask[c + 11, c].item() == pytest.approx(torch.sigmoid(torch.tensor(-edge)).item(), abs=1e-5)
    assert mask[c, c].item() > 0.999


def test_na_gradient():
    # d(loss)/dNA is non-zero and matches a central finite difference.
    n = 64
    g = torch.Generator().manual_seed(0)
    field = torch.polar(torch.ones(n, n), 2 * math.pi * torch.rand(n, n, generator=g))
    w = torch.rand(n, n, generator=g) - 0.5
    lens = optics.F4ImagingSystem(WL, (DX, DX), (n, n), NA=0.6)

    def loss(na):
        with torch.no_grad():
            lens.NA.fill_(na)
        return ((lens(field).abs() ** 2) * w).sum()

    lens.zero_grad()
    loss(0.6).backward()
    grad = lens.NA.grad.item()
    h = 1e-3
    fd = (loss(0.6 + h).item() - loss(0.6 - h).item()) / (2 * h)
    assert grad != 0
    assert grad == pytest.approx(fd, rel=2e-2)
