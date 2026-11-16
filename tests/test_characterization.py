"""Characterization (golden-master) tests: pin what the code computes today.

They don't claim the numbers are physically right. Known errors (A1, A2, ...) are baked in.
Their job is to make every refactor reproduce today's outputs exactly, unless a commit changes
physics on purpose; that commit regenerates the affected file and shows a before/after figure:

    OPTIVOLUME_UPDATE_FIXTURES=1 pytest tests/test_characterization.py

No random numbers are used. The phase-mask tile maps come from a fixed formula, so the
results don't depend on torch's RNG stream.
"""
import torch

from optical_volume import optics, pipelines, scene

WL = 500e-9
N_BG = 1.51
OBJ_SPACING = (100e-9, 100e-9, 100e-9)
OBJ_SHAPE = (64, 64, 16)
PAD = 64


def _tile_map(num_x, num_y):
    """Deterministic, irregular 0/1 tile pattern (stands in for the random map)."""
    i, j = torch.meshgrid(torch.arange(int(num_x)), torch.arange(int(num_y)), indexing="ij")
    return (((i * 37 + j * 91 + i * j) % 11) < 5).float()


def _sphere_exit_field():
    sphere = scene.shapes.Sphere(
        torch.tensor((3.2e-6, 3.2e-6, 0.8e-6)), torch.tensor(1.5e-6), torch.tensor(1.46), softness=1e-12
    )
    vol = scene.CollectionVolume(OBJ_SHAPE, OBJ_SPACING, N_BG)
    vol.add([sphere])
    with torch.no_grad():
        return optics.VolumePropagator(WL, OBJ_SPACING, N_BG)(vol())


def _v5_cwfs_chain():
    """The chain from examples/04_coded_wavefront_sensing.ipynb, scaled down to 64²."""
    obj_img_factor, mag = 20, 40
    eff_mag = mag // obj_img_factor
    img_spacing = [s * obj_img_factor for s in OBJ_SPACING[:2]]  # 2 µm
    img_shape = [n * eff_mag for n in OBJ_SHAPE]  # 128 x 128 (x 32)
    mask_args = (WL, 12e-6, img_spacing, img_shape, 100e-9, 0.6, 1.46)

    probe = optics.BinaryPhaseMask(*mask_args)  # only to read the tile count
    mask = optics.BinaryPhaseMask(*mask_args, tile_map=_tile_map(*probe.get_num_tiles()))

    # The same instance twice, as in v5 (this is bug A1, pinned on purpose).
    prop_img_space = optics.FreeSpacePropagator(WL, img_spacing, 1.43e-3, direction="backward", padding=PAD)
    return optics.OpticalSystem((
        optics.FreeSpacePropagator(
            WL / N_BG, OBJ_SPACING[:2], OBJ_SHAPE[2] * OBJ_SPACING[2] / 2 - 1e-9,
            direction="backward", padding=PAD,
        ),
        optics.F4ImagingSystem(WL, OBJ_SPACING, OBJ_SHAPE, NA=0.7),
        optics.utils.Magnification(torch.tensor(mag / obj_img_factor)),
        prop_img_space,
        mask,
        prop_img_space.set_direction("forward"),
        optics.utils.PixelIntegrator(int(6e-6 / img_spacing[0])),
    ))


def test_char_cwfs_reference(golden):
    system = _v5_cwfs_chain()
    exit_field = _sphere_exit_field().to(torch.complex64)
    with torch.no_grad():
        img_ref = system(torch.ones(OBJ_SHAPE[:2], dtype=torch.complex64)).abs() ** 2
        img_spn = system(exit_field).abs() ** 2
    assert img_ref.shape == (42, 42)
    golden("cwfs_reference", img_ref=img_ref.numpy(), img_spn=img_spn.numpy())


def test_char_bpm_sphere(golden):
    exit_field = _sphere_exit_field()
    assert exit_field.shape == OBJ_SHAPE[:2]
    golden("bpm_sphere", exit_field=exit_field.numpy())


def test_char_coded_phase_cam(golden):
    cam = pipelines.CodedPhaseCamV3(wavelength=WL, spacing=(1.5e-6, 1.5e-6), shape=(64, 64), padding=16)
    pm = cam.pcam_FM.elements[1]
    # Replace the randomly drawn map with the deterministic pattern.
    pm.tile_map.copy_(_tile_map(*pm.get_num_tiles()))
    with torch.no_grad():
        img = cam(torch.ones(64, 64, dtype=torch.complex64))
    assert img.shape == (16, 16)
    golden("coded_phase_cam", img=img.numpy())
