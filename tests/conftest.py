"""Shared pytest fixtures."""
import os
from pathlib import Path

import numpy as np
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "characterization"

# Set OPTIVOLUME_UPDATE_FIXTURES=1 to (re)write the golden files instead of comparing.
# Only do this in a commit that changes physics on purpose, and show a before/after comparison.
UPDATE_ENV = "OPTIVOLUME_UPDATE_FIXTURES"


@pytest.fixture
def golden():
    """Compare named arrays with a stored ``.npz`` golden file, or write it in update mode.

    Tolerances: rtol=1e-5 and atol=1e-6 * max|reference|. This is loose enough for float32
    FFT rounding differences between machines and torch builds, and tight enough to catch
    any real numerical change.
    """

    def check(name, **arrays):
        path = FIXTURE_DIR / f"{name}.npz"
        arrays = {k: np.asarray(v) for k, v in arrays.items()}
        if os.environ.get(UPDATE_ENV) == "1":
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **arrays)
            return
        if not path.exists():
            pytest.fail(f"missing golden file {path}; generate it with {UPDATE_ENV}=1")
        with np.load(path, allow_pickle=False) as stored:
            assert sorted(stored.files) == sorted(arrays), f"{name}: stored arrays differ"
            for key, actual in arrays.items():
                expected = stored[key]
                assert actual.shape == expected.shape, f"{name}/{key}: shape changed"
                np.testing.assert_allclose(
                    actual, expected, rtol=1e-5, atol=1e-6 * np.abs(expected).max(),
                    err_msg=f"{name}/{key} differs from the golden file",
                )

    return check
