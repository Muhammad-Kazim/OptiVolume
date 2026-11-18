"""Device moves: every tensor a module owns is registered, so .to() moves it (C2, C3).

CUDA results are compared with the CPU in ``test_gpu.py``.
"""
import pytest
import torch
from torch import Tensor

from optical_volume import optics, pipelines, scene

WL = 500e-9
SPACING = (1e-6, 1e-6, 1e-6)
SHAPE = (16, 16, 4)


def _modules():
    bpm = optics.VolumePropagator(WL, SPACING, 1.33, padding=4)
    bpm.set_source_field(torch.ones(SHAPE[:2], dtype=torch.complex64))
    collection = scene.CollectionVolume(SHAPE, SPACING, 1.33)
    collection.add(scene.shapes.make_ellipsoids(torch.full((2, 3), 8e-6), torch.full((2, 3), 3e-6),
                                                torch.tensor([1.4, 1.5])))
    return {
        "free_space": optics.FreeSpacePropagator(WL, SPACING[:2], 5e-6, padding=4),
        "bpm": bpm,
        "modulator": optics.Modulator(torch.ones(SHAPE[:2], dtype=torch.complex64)),
        "binary_mask": optics.BinaryPhaseMask(WL, 4e-6, SPACING, SHAPE, 500e-9, 1.0, 1.46),
        "custom_mask": optics.CustomPhaseMask(WL, 4e-6, SPACING, SHAPE, torch.zeros(4, 4), 1.0, 1.46),
        "lens": optics.F4ImagingSystem(WL, SPACING[:2], SHAPE[:2], NA=0.5),
        "collection": collection,
        "voxels": scene.VoxelVolume(SHAPE, SPACING, n0=1.33),
        "camera": pipelines.CodedPhaseCamV3(WL, (1.5e-6, 1.5e-6), (64, 64), padding=16),
        "detector": optics.OpticalSystem((
            optics.Magnification(40), optics.Resample(2), optics.Intensity(),
            optics.PixelIntegrator(pixel_size=5e-6, spacing=2e-6, offset=(1e-7, 0.)))),
        "blur": optics.PartialCoherenceBlur(SPACING[:2], sigma0=1e-7, angular_spread=0.1, defocus=1e-6),
    }


@pytest.mark.parametrize("name", list(_modules()))
def test_to_device_moves_state(name):
    # The meta device stands in for a GPU: .to() moves registered tensors only, and the
    # dtype is kept (.to(torch.float64) would drop the imaginary part of complex buffers).
    module = _modules()[name].to("meta")
    for sub_name, sub in module.named_modules():
        stray = [k for k, v in vars(sub).items() if isinstance(v, Tensor)]
        assert not stray, f"{name}.{sub_name}: unregistered tensor attributes {stray}"
    for key, tensor in list(module.named_parameters()) + list(module.named_buffers()):
        assert tensor.device.type == "meta", f"{name}.{key} stayed on {tensor.device}"
