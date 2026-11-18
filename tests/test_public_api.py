"""Public API (0.2.0): what each subpackage exports, the removed 0.1 names, state_dict keys."""
import importlib

import pytest
import torch

import optical_volume as ov
from optical_volume import optics, pipelines, scene

SUBPACKAGES = ["optical_volume", "optical_volume.optics", "optical_volume.scene", "optical_volume.pipelines",
               "optical_volume.utils"]


@pytest.mark.parametrize("name", SUBPACKAGES + ["optical_volume.visualization"])
def test_public_api(name):
    # Every name in __all__ exists, appears once, and the list is sorted (classes before functions).
    if name.endswith("visualization"):
        pytest.importorskip("matplotlib")
    module = importlib.import_module(name)
    names = module.__all__
    assert len(names) == len(set(names)), f"{name}: duplicates in __all__"
    for n in names:
        assert hasattr(module, n), f"{name}.{n} is listed in __all__ but missing"
    if name != "optical_volume":
        assert names == sorted(names), f"{name}: __all__ is not sorted"


def test_top_level_reexports():
    # The main classes are the same objects at the top level and in their subpackage.
    for sub in (optics, scene, pipelines):
        for n in sub.__all__:
            assert getattr(ov, n) is getattr(sub, n)


@pytest.mark.parametrize("module, old", [
    (optics, "FreeSpacePropagator"), (optics, "VolumePropagator"), (optics, "F4ImagingSystem"),
    (optics, "Modulator"), (optics, "CustomPhaseMask"), (optics, "utils"),
    (pipelines, "CodedPhaseCamV3"), (scene.shapes, "Cube"), (scene.shapes, "make_cubes"),
    (scene.shapes, "Plane"), (ov.utils, "utils"), (ov.utils, "torch_TV"), (ov.utils, "torch_L2_grad"),
    (ov.utils, "torch_grad_optr"), (ov.utils, "auto_corr_fn"), (ov.utils, "HiddenPrints"),
])
def test_old_names_removed(module, old):
    # 0.2.0 breaks the 0.1 API on purpose (no aliases); a leftover would mean a half-done rename.
    assert not hasattr(module, old), f"{module.__name__}.{old} should be gone"


def test_state_dict_keys():
    # Saved models depend on these names (n, n_map, n_medium, ...); changing them breaks loading.
    keys = lambda m: sorted(m.state_dict())
    assert keys(scene.Sphere(torch.zeros(3), torch.tensor(1e-6), torch.tensor(1.5))) == ["center", "n", "quat", "radius"]
    assert keys(scene.Sphere(torch.zeros(3), torch.tensor(1e-6), scene.VoxelRI(torch.ones(2, 2, 2), 1e-7))) == \
        ["center", "n_map.values", "quat", "radius"]
    assert keys(scene.Slab(torch.zeros(3), torch.tensor([0., 0., 1.]), torch.tensor(1e-6), 1.5)) == \
        ["n", "normal", "point", "thickness"]
    assert keys(scene.VoxelVolume((4, 4, 4), (1e-7,) * 3, n_medium=1.33)) == ["center", "quat", "volume"]
    assert keys(optics.AngularSpectrum(5e-7, (1e-7, 1e-7), 1e-6)) == ["distance"]
    assert keys(optics.FourFSystem(5e-7, (1e-7, 1e-7), (8, 8), 0.5)) == ["NA", "pupil_amp", "pupil_opd"]
    assert keys(optics.BinaryPhaseMask(5e-7, 1e-6, (1e-7, 1e-7), (16, 16), 1e-7, 0.6, 1.46, seed=0)) == \
        ["_extra_state", "height", "log_sigma", "tile_map"]
    assert keys(optics.Transmission(torch.ones(4, 4))) == ["operator"]
    # Keyword names of the refractive indices.
    assert scene.CollectionVolume((4, 4, 4), (1e-7,) * 3, n_medium=1.33).n_medium == 1.33
    assert optics.BeamPropagation(5e-7, (1e-7,) * 3, n_medium=1.33).n_medium == 1.33
    assert optics.BinaryPhaseMask(5e-7, 1e-6, (1e-7, 1e-7), (16, 16), 1e-7, 0.6, n_mask=1.46, n_medium=1.0).n_mask == 1.46
