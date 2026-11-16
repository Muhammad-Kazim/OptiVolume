"""Phase-mask tile maps: seeding and state_dict round trip (C7)."""
import pytest
import torch

from optical_volume import optics

WL = 500e-9
ARGS = (WL, 4e-6, (1e-6, 1e-6), (32, 32), 500e-9, 1.0, 1.46)


def _mask(**kwargs):
    return optics.BinaryPhaseMask(*ARGS, **kwargs)


def test_phase_mask_seed_reproducible():
    assert torch.equal(_mask(seed=1).tile_map, _mask(seed=1).tile_map)
    assert not torch.equal(_mask(seed=1).tile_map, _mask(seed=2).tile_map)

    gen = torch.Generator().manual_seed(0)
    first, second = _mask(generator=gen), _mask(generator=gen)
    assert not torch.equal(first.tile_map, second.tile_map)
    for mask in (first, second, _mask()):
        assert torch.equal(_mask(seed=mask.seed).tile_map, mask.tile_map)

    # Unseeded masks still follow torch.manual_seed: it fixes the drawn seed.
    torch.manual_seed(5)
    a = _mask()
    torch.manual_seed(5)
    assert torch.equal(_mask().tile_map, a.tile_map)


def test_phase_mask_state_dict_restores_map():
    source = _mask()
    target = _mask()
    target.load_state_dict(source.state_dict())
    assert torch.equal(target.tile_map, source.tile_map)
    assert target.seed == source.seed
    field = torch.ones(32, 32, dtype=torch.complex64)
    torch.testing.assert_close(target(field), source(field))


def test_phase_mask_arguments():
    tiles = torch.ones(8, 8)
    assert _mask(tile_map=tiles).seed is None
    with pytest.raises(ValueError):
        _mask(tile_map=torch.ones(3, 3))
    with pytest.raises(ValueError):
        _mask(tile_map=tiles, seed=1)
    with pytest.raises(ValueError):
        _mask(seed=1, generator=torch.Generator())
