"""utils.io: normalization to unsigned integer images (A16)."""
import pytest
import torch

from optical_volume import utils


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
