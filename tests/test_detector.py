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
