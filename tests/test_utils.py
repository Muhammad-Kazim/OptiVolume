"""Utilities: quadratic fit (A15), normalization (A16)."""
import pytest
import torch

from optical_volume import utils

COEFFS = torch.tensor([3e-3, -2e-3, 1.5e-3, 0.2, -0.1, 5.0], dtype=torch.float64)


def _surface(nx, ny, coeffs, dx=1.0, dy=1.0):
    X, Y = torch.meshgrid(torch.arange(nx, dtype=torch.float64) * dx, torch.arange(ny, dtype=torch.float64) * dy,
                          indexing="ij")
    a, b, c, d, e, f = coeffs
    return a * X**2 + b * Y**2 + c * X * Y + d * X + e * Y + f


def test_fit_quadratic():
    # Six known coefficients plus a little noise are recovered, and the returned surface is the model
    # with each term's own coefficient (A15).
    clean = _surface(40, 30, COEFFS)
    noisy = clean + 1e-3 * torch.randn(40, 30, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    coeffs, surface = utils.fit_quadratic(noisy)
    torch.testing.assert_close(coeffs, COEFFS, rtol=2e-2, atol=1e-4)
    torch.testing.assert_close(surface, clean, rtol=0, atol=2e-3)
    coeffs, surface = utils.fit_quadratic(clean)
    torch.testing.assert_close(coeffs, COEFFS, rtol=1e-8, atol=1e-10)
    torch.testing.assert_close(surface, clean)


def test_fit_quadratic_spacing():
    # With spacing (metres) the coefficients come out per metre: the same surface sampled at 2 µm.
    d = 2e-6
    per_metre = COEFFS / torch.tensor([d**2, d**2, d**2, d, d, 1.], dtype=torch.float64)
    img = _surface(40, 30, per_metre, d, d)
    coeffs, surface = utils.fit_quadratic(img, spacing=d)
    torch.testing.assert_close(coeffs, per_metre, rtol=1e-6, atol=0)
    torch.testing.assert_close(surface, img)


@pytest.mark.parametrize("totype, top", [("uint8", 255), ("uint16", 65535)])
def test_normalization_range(totype, top):
    # Minimum -> 0, maximum -> 2^b - 1, order kept, right dtype: nothing wraps around (A16).
    field = torch.linspace(-3.0, 7.0, 1000).reshape(20, 50).numpy()
    out = utils.normalization(field, totype)
    assert out.dtype.name == totype
    assert out.min() == 0 and out.max() == top
    assert out[0, 0] == 0 and out[-1, -1] == top
    assert (out.ravel()[1:].astype(int) >= out.ravel()[:-1].astype(int)).all()


def test_normalization_edge_cases():
    # A constant field gives zeros; a wrong type raises, including the old misleading names.
    assert (utils.normalization(torch.full((4, 4), 2.5).numpy(), "uint8") == 0).all()
    for bad in ("int16", "int8", "float"):
        with pytest.raises(ValueError):
            utils.normalization(torch.rand(4, 4).numpy(), bad)
