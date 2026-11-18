"""Plots follow the [x, y] convention: x to the right, y up, origin at the bottom left (A17)."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")    # optional dependency (extra "plot")
matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from optical_volume import visualization  # noqa: E402

NX, NY = 40, 30


def _spot_field():
    """A bright spot at large x (i = 35) and small y (j = 4)."""
    field = np.full((NX, NY), 0.1 + 0j)
    field[35, 4] = 1.0
    return field


def _check(ax, unit="µm"):
    im = ax.images[0]
    data = np.asarray(im.get_array())
    row, col = np.unravel_index(np.argmax(data), data.shape)
    # Transposed with origin='lower': row = y index (small -> bottom), column = x index (large -> right).
    assert (row, col) == (4, 35) and im.origin == "lower"
    assert ax.get_xlabel() == f"x ({unit})" and ax.get_ylabel() == f"y ({unit})"
    assert im.get_extent() == pytest.approx([0, NX * 0.5, 0, NY * 0.5])


def test_visualize_field_orientation():
    ax = visualization.visualize_field(_spot_field(), (NX * 0.5e-6, NY * 0.5e-6))
    _check(ax)
    plt.close("all")


def test_visualize_complex_field_labels():
    axs = visualization.visualize_complex_field(_spot_field() * np.exp(0.3j), (NX * 0.5e-3, NY * 0.5e-3), units=1e-3)
    assert len(axs) == 2
    for ax in axs:
        _check(ax, unit="mm")
    plt.close("all")
