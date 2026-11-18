"""GPU tests: CUDA results match the CPU, gradients work on the device, example chains run.

Every test here is marked ``gpu`` and skipped without a CUDA device, so a CPU-only run skips
them all. On a GPU machine, ``pytest -m gpu`` runs only these. ``OPTIVOLUME_TEST_DEVICE``
picks the device (default ``cuda``, e.g. ``cuda:1``).

The example chains (04, 06, 07) are rebuilt here at the examples' settings, without notebooks
or data files. Each one checks GPU against CPU and the absolute brightness of the reference
image: a unit plane wave has intensity 1/M² after ``Magnification(M)``, and a camera pixel
sums the intensity of the samples it covers.
"""
import os

import pytest
import torch

from optical_volume import optics, pipelines, scene

DEVICE = os.environ.get("OPTIVOLUME_TEST_DEVICE", "cuda")
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(DEVICE.startswith("cuda") and not torch.cuda.is_available(), reason="needs a CUDA device"),
]

WL = 500e-9


def assert_matches_cpu(gpu, cpu, rtol=1e-4):
    """GPU and CPU agree up to float32 FFT rounding, relative to the largest value."""
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=rtol, atol=rtol * cpu.abs().max().item())


def run_both(module, *inputs):
    """Run a module on the CPU, then move it (and the inputs) to the device and run again."""
    with torch.no_grad():
        cpu = module(*inputs)
        gpu = module.to(DEVICE)(*(x.to(DEVICE) for x in inputs))
    return gpu, cpu


def centre_mean(img):
    H, W = img.shape[-2:]
    return img[..., H // 4: 3 * H // 4, W // 4: 3 * W // 4].mean().item()


# ---------------------------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------------------------

def _elements():
    return {
        "pixel_integer": (optics.utils.PixelIntegrator(3), "real"),
        "pixel_fractional_offset": (optics.utils.PixelIntegrator(pixel_size=6e-6, spacing=2.5e-6, offset=(3e-7, -2e-7)), "real"),
        "intensity": (optics.utils.Intensity(), "complex"),
        "magnification": (optics.utils.Magnification(40), "complex"),
        "resample_fourier": (optics.utils.Resample(2), "complex"),
        "resample_bilinear": (optics.utils.Resample(2, method="bilinear"), "complex"),
        "coherence_blur": (optics.utils.PartialCoherenceBlur((1e-7, 1e-7), sigma0=5e-8, angular_spread=0.05, defocus=5e-6), "complex"),
        "binary_mask": (optics.BinaryPhaseMask(WL, 12e-6, (2e-6, 2e-6), (96, 96), 100e-9, 0.6, 1.46, seed=0), "complex"),
        "free_space": (optics.FreeSpacePropagator(WL, (2e-6, 2e-6), 1.43e-3, padding=32), "complex"),
        "lens": (optics.F4ImagingSystem(WL, (1e-7, 1e-7), (96, 96), 0.7), "complex"),
    }


@pytest.mark.parametrize("name", list(_elements()))
def test_element_matches_cpu(name):
    module, kind = _elements()[name]
    g = torch.Generator().manual_seed(0)
    x = torch.rand(96, 96, generator=g)
    if kind == "complex":
        x = torch.polar(x + 0.5, 2 * torch.pi * torch.rand(96, 96, generator=g))
    gpu, cpu = run_both(module, x)
    assert gpu.device.type == torch.device(DEVICE).type
    assert_matches_cpu(gpu, cpu)


def _grad_cases():
    """(module, input, parameter names) for comparing gradients between devices.

    The comparison is only meaningful if float32 computes each gradient accurately, so the
    cases avoid three traps (each measured against float64 on the CPU):
    - A phase mask alone doesn't change the intensity (|t| = 1): its height gradient is
      exactly zero and only rounding noise is left. The mask is followed by a propagator.
    - The loss weights are zero-mean (see ``_grads``).
    - The pixel offset is not 0 along either axis: with pixel edges on sample edges the box
      overlap has a kink, and rounding decides which side's slope is returned.
    With these, every gradient here is accurate to <= 5e-6 in float32, and the test uses 1e-4.
    """
    g = torch.Generator().manual_seed(1)
    field = torch.polar(torch.rand(64, 64, generator=g) + 0.5, torch.rand(64, 64, generator=g))
    mask = optics.BinaryPhaseMask(WL, 12e-6, (2e-6, 2e-6), (64, 64), 100e-9, 0.6, 1.46, seed=0)
    return {
        "pixel_offset": (optics.utils.PixelIntegrator(pixel_size=6e-6, spacing=2.5e-6, offset=(3e-7, -2e-7)),
                         torch.rand(64, 64, generator=g), ["offset"]),
        "coherence_blur": (optics.utils.PartialCoherenceBlur((1e-7, 1e-7), sigma0=5e-8, angular_spread=0.05, defocus=5e-6),
                           field, ["sigma0", "angular_spread", "defocus"]),
        "mask_height": (optics.OpticalSystem((mask, optics.FreeSpacePropagator(WL, (2e-6, 2e-6), 1.43e-3, padding=16))),
                        field, ["elements.0.height"]),
        "propagation_distance": (optics.FreeSpacePropagator(WL, (2e-6, 2e-6), 1.43e-3, padding=16),
                                 field, ["distance"]),
        "lens_na": (optics.F4ImagingSystem(WL, (1e-7, 1e-7), (64, 64), 0.6), field, ["NA"]),
    }


def _grads(module, x, names):
    module.zero_grad()
    y = module(x)
    y = y.abs() ** 2 if y.is_complex() else y
    # Fixed random, zero-mean weights. A constant part of the weights would multiply the total
    # intensity, which the optics conserve: a large term with no gradient, whose float32
    # rounding (~1e-3 relative here) would swamp the small, real sensitivity.
    w = torch.rand(y.shape, generator=torch.Generator().manual_seed(7))
    w = (w - w.mean()).to(y.device, y.dtype)
    (y * w).sum().backward()
    params = dict(module.named_parameters())
    return [params[n].grad.detach().cpu().clone() for n in names]


@pytest.mark.parametrize("name", list(_grad_cases()))
def test_gradients_match_cpu(name):
    module, x, names = _grad_cases()[name]
    cpu = _grads(module, x, names)
    gpu = _grads(module.to(DEVICE), x.to(DEVICE), names)
    for n, c, d in zip(names, cpu, gpu):
        assert torch.isfinite(d).all() and d.abs().sum() > 0, f"{n}: no gradient on {DEVICE}"
        torch.testing.assert_close(d, c, rtol=1e-4, atol=1e-4 * c.abs().max().item(),
                                   msg=lambda m, n=n: f"{n} gradient: {m}")


def test_mask_border_flat_on_gpu():
    # A7 on the device: a uniform mask applies the same phase everywhere, also at its border.
    mask = optics.BinaryPhaseMask(WL, 12e-6, (2e-6, 2e-6), (64, 64), 100e-9, 2.0, 1.46,
                                  tile_map=torch.ones(11, 11)).to(DEVICE)
    with torch.no_grad():
        phase = mask(torch.ones(64, 64, dtype=torch.complex64, device=DEVICE)).angle()
    assert (phase.max() - phase.min()).item() < 1e-5


def _scenes():
    """A CollectionVolume with every shape (one with an RI map) and a rotated VoxelVolume."""
    n, d = 32, 0.1e-6
    c = torch.full((3,), (n - 1) * d / 2)
    coll = scene.CollectionVolume((n, n, n), (d, d, d), 1.33)
    coll.add([scene.shapes.Slab(c.clone(), torch.tensor([0.2, 0.1, 1.0]), torch.tensor(1.0e-6), 1.36),
              scene.shapes.Cube(c.clone(), torch.tensor([1.6e-6, 1.0e-6, 0.8e-6]), torch.tensor(1.40),
                                rotation_axis=(1., 1., 0.), rotation_angle=25.),
              scene.shapes.Ellipsoid(c + 0.6e-6, torch.tensor([0.9e-6, 0.6e-6, 0.4e-6]), torch.tensor(1.45),
                                     rotation_axis=(0., 1., 1.), rotation_angle=40.),
              scene.shapes.Sphere(c - 0.6e-6, torch.tensor(0.5e-6), scene.VoxelRI(
                  1.42 + 0.04 * torch.rand(8, 8, 8, generator=torch.Generator().manual_seed(3)), 0.15e-6))])
    vox = scene.VoxelVolume((n, n, n), (d, d, d),
                            1.33 + 0.05 * torch.rand(n, n, n, generator=torch.Generator().manual_seed(4)),
                            n0=1.33, rotation_axis=(1., 2., 3.), rotation_angle=35.)
    grid = scene.volume.Grid((n, n, n), (d, d, d))
    return {"collection": (coll, lambda m: m()),
            "voxel_volume": (vox, lambda m: m(grid.grid.to(m.volume.device)))}


# The exact box SDF takes the max over the faces; at a point equidistant from two faces the
# gradient jumps, and float rounding picks the face. Measured float32 vs float64 on the CPU:
# 2e-2 for these three, <= 5e-6 for every other scene gradient.
_KINKED = {"shapes.1.center", "shapes.1.length", "shapes.1.quat"}


@pytest.mark.parametrize("name", ["collection", "voxel_volume"])
def test_scene_matches_cpu(name):
    # Volumes and all their parameter gradients agree between CUDA and the CPU.
    def run(module, f):
        module.zero_grad()
        y = f(module)
        w = torch.rand(y.shape, generator=torch.Generator().manual_seed(7))
        (y * (w - w.mean()).to(y.device)).sum().backward()
        return y.detach().cpu(), {k: p.grad.detach().cpu().clone() for k, p in module.named_parameters()}
    module, f = _scenes()[name]
    y_cpu, g_cpu = run(module, f)
    y_gpu, g_gpu = run(module.to(DEVICE), f)
    assert_matches_cpu(y_gpu, y_cpu)
    for k, c in g_cpu.items():
        rtol = 5e-2 if k in _KINKED else 1e-4
        torch.testing.assert_close(g_gpu[k], c, rtol=rtol, atol=rtol * c.abs().max().item(),
                                   msg=lambda m, k=k: f"{k} gradient: {m}")


def test_camera_matches_cpu():
    torch.manual_seed(0)
    cam = pipelines.CodedPhaseCamV3(WL, (1.5e-6, 1.5e-6), (64, 64), padding=16)
    field = torch.exp(1j * torch.rand(64, 64)).to(torch.complex64)
    gpu, cpu = run_both(cam, field)
    assert_matches_cpu(gpu, cpu)


# ---------------------------------------------------------------------------------------------
# Example chains at the examples' settings
# ---------------------------------------------------------------------------------------------

def _cwfs_chain(obj_spacing, shape, NA, mag, eff_mag, dist, tile, height, sigma, pixel, n_bg=1.51, pad=512):
    """The coded-wavefront-sensing chain of examples 04 and 06 (object side to camera pixels)."""
    img_spacing = [obj_spacing[0] * mag / eff_mag] * 2
    to_mask = optics.FreeSpacePropagator(WL, img_spacing, dist, direction="backward", padding=pad)
    to_sensor = optics.FreeSpacePropagator(WL, img_spacing, dist, direction="forward", padding=pad)
    to_sensor.distance = to_mask.distance
    mask = optics.BinaryPhaseMask(WL, tile, img_spacing, [x * eff_mag for x in shape], height, sigma, 1.46, seed=0)
    system = optics.OpticalSystem((
        optics.FreeSpacePropagator(WL / n_bg, obj_spacing[:2], shape[2] * obj_spacing[2] / 2 - 1e-9,
                                   direction="backward", padding=pad),
        optics.F4ImagingSystem(WL, obj_spacing, shape, NA),
        optics.utils.Magnification(mag), optics.utils.Resample(eff_mag),
        to_mask, mask, to_sensor,
        optics.utils.Intensity(), optics.utils.PixelIntegrator(pixel_size=pixel, spacing=img_spacing)))
    return system, mask, to_mask


def _ex04():
    spacing, shape = [100e-9] * 3, [600, 600, 100]
    vol = scene.CollectionVolume(shape, spacing, 1.51)
    vol.add([scene.shapes.Sphere(torch.tensor((20e-6, 45e-6, 3e-6)), torch.tensor(5e-6), torch.tensor(1.46)),
             scene.shapes.Sphere(torch.tensor((30e-6, 20e-6, 5e-6)), torch.tensor(3e-6), torch.tensor(1.46))])
    bpm = optics.VolumePropagator(WL, spacing, 1.51)
    system, mask, to_mask = _cwfs_chain(spacing, shape, NA=0.7, mag=40, eff_mag=2, dist=1.43e-3,
                                        tile=12e-6, height=100e-9, sigma=0.6, pixel=6e-6)
    return vol, bpm, system, mask, to_mask, spacing, shape


def test_example04_chain():
    vol, bpm, system, mask, to_mask, spacing, shape = _ex04()
    with torch.no_grad():
        field_cpu = bpm(vol()).to(torch.complex64)
        ref_cpu, img_cpu = system(torch.ones(shape[:2], dtype=torch.complex64)), system(field_cpu)
        vol.to(DEVICE), bpm.to(DEVICE), system.to(DEVICE)
        field = bpm(vol()).to(torch.complex64)
        ref, img = system(torch.ones(shape[:2], dtype=torch.complex64, device=DEVICE)), system(field)
    assert_matches_cpu(field, field_cpu, rtol=1e-3)            # 100 BPM slices accumulate rounding
    assert_matches_cpu(ref, ref_cpu)
    assert_matches_cpu(img, img_cpu, rtol=1e-3)
    assert system.output_spacing(spacing[:2]) == pytest.approx((6e-6, 6e-6))
    # 1/40² after Magnification(40), 3 × 3 samples per 6 µm pixel.
    assert centre_mean(ref) == pytest.approx(9 / 1600, rel=0.02)


def test_example04_training_step():
    # The optimisation set up in example 04: spheres, mask distance and mask height on the device.
    vol, bpm, system, mask, to_mask, spacing, shape = _ex04()
    vol.to(DEVICE), bpm.to(DEVICE), system.to(DEVICE)
    with torch.no_grad():
        target = system(bpm(vol()).to(torch.complex64))
    for p in vol.parameters():
        p.data += 0.01 * p.data.abs().mean()                  # start away from the target
    opt = torch.optim.Adam([{"params": vol.parameters(), "lr": 1e-2},
                            {"params": [to_mask.distance, mask.height], "lr": 1e-3}])
    before = [p.detach().clone() for p in (to_mask.distance, mask.height)]
    loss = ((system(bpm(vol()).to(torch.complex64)) - target) ** 2).mean()
    loss.backward()
    grads = [p.grad for p in list(vol.parameters()) + [to_mask.distance, mask.height]]
    assert torch.isfinite(loss) and all(g is not None and torch.isfinite(g).all() for g in grads)
    opt.step()
    assert all(not torch.equal(b, p) for b, p in zip(before, (to_mask.distance, mask.height)))


def test_example06_voxel_chain():
    # Example 06 at reduced size: a voxel object rotated by 45° about y, without the mesh file.
    n, d = 128, 1.5e-6
    shape, spacing = [n] * 3, [d] * 3
    voxels = torch.full(shape, 1.51)
    voxels[40:88, 30:98, 34:94] += 0.03
    vol = scene.VoxelVolume(shape, spacing, voxels, n0=1.51, rotation_axis=[0, 1, 0], rotation_angle=45,
                            object_center=[d * n / 2] * 3)
    grid = scene.volume.Grid(shape, spacing)
    bpm = optics.VolumePropagator(WL, spacing, 1.51)
    system, _, _ = _cwfs_chain(spacing, shape, NA=0.4, mag=4, eff_mag=2, dist=2.5e-3,
                               tile=12e-6, height=570e-9, sigma=0.9, pixel=3e-6, pad=128)
    with torch.no_grad():
        ri_cpu = vol(grid.grid)
        img_cpu = system(bpm(ri_cpu).to(torch.complex64))
        vol.to(DEVICE), grid.to(DEVICE), bpm.to(DEVICE), system.to(DEVICE)
        ri = vol(grid.grid)
        img = system(bpm(ri).to(torch.complex64))
        ref = system(torch.ones(shape[:2], dtype=torch.complex64, device=DEVICE))
    assert_matches_cpu(ri, ri_cpu)
    assert_matches_cpu(img, img_cpu, rtol=1e-3)
    # 1/4² after Magnification(4); a 3 µm pixel on the 3 µm grid is one sample.
    assert centre_mean(ref) == pytest.approx(1 / 16, rel=0.02)


def test_example07_camera_chain():
    # Example 07's image formation (one time step): object-side resampling, lens, Magnification(40),
    # CodedPhaseCamV3 with 6 µm pixels on the 3 µm image grid.
    spacing, shape, eff = [150e-9, 150e-9, 100e-9], [600, 600, 100], 2
    img_spacing, img_shape = [s * 20 for s in spacing[:2]], [x * eff for x in shape]
    vol = scene.CollectionVolume(shape, spacing, 1.51)
    vol.add([scene.shapes.Sphere(torch.tensor((60e-6, 20e-6, 5e-6)), torch.tensor(7e-6), torch.tensor(1.46)),
             scene.shapes.Sphere(torch.tensor((40e-6, 33e-6, 5e-6)), torch.tensor(9e-6), torch.tensor(1.46))])
    bpm = optics.VolumePropagator(WL, spacing, 1.51)
    torch.manual_seed(0)
    system = optics.OpticalSystem((
        optics.FreeSpacePropagator(WL / 1.51, spacing[:2], shape[2] * spacing[2] / 2 - 1e-9, direction="backward", padding=512),
        optics.utils.PartialCoherenceBlur(spacing[:2]),
        optics.utils.Resample(eff),
        optics.F4ImagingSystem(WL, [x / eff for x in spacing], img_shape, 0.75),
        optics.utils.Magnification(40),
        pipelines.CodedPhaseCamV3(wavelength=WL, spacing=img_spacing, shape=img_shape, padding=512, pad_mode="edge")))
    with torch.no_grad():
        img_cpu = system(bpm(vol()).to(torch.complex64))
        vol.to(DEVICE), bpm.to(DEVICE), system.to(DEVICE)
        img = system(bpm(vol()).to(torch.complex64))
        ref = system(torch.ones(shape[:2], dtype=torch.complex64, device=DEVICE))
    assert_matches_cpu(img, img_cpu, rtol=1e-3)
    # 1/40² after Magnification(40), 2 × 2 samples per 6 µm pixel.
    assert centre_mean(ref) == pytest.approx(4 / 1600, rel=0.02)
