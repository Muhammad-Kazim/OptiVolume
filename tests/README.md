# tests

Run with `pytest -q` (in the `optical_volume` conda env).

- `conftest.py`: shared fixtures; `golden` compares arrays with the stored characterization files.
- `test_smoke.py`: fast checks that the package imports and the core blocks run and are differentiable.
- `test_characterization.py`: golden-master tests that pin today's numerical outputs (BPM, v5 CWFS chain, `CodedPhaseCamV3`).
- `test_propagator.py`: propagator padding, batch dimensions and the BPM source field.
- `test_system.py`: `OpticalSystem` recording and hooks, `Modulator` registration.
- `test_shapes.py`: shape factories with rotations, in-place `set_quat`; SDF occupancy (0.5 on the surface), softness in metres, gradients, composition in `CollectionVolume`, `Slab`.
- `test_utils.py`: `fit_quadratic` (each term's coefficient, optional spacing in metres) and `normalization`.
- `test_visualization.py`: 2D plots put x to the right and y up, labelled (skipped without matplotlib).
- `test_volume.py`: `VoxelVolume` defaults (centre, required `n0`); points outside the box are the background.
- `test_lens.py`: `F4ImagingSystem` soft pupil edge at the cutoff on any grid; NA gradient matches finite differences.
- `test_ri_maps.py`: refractive-index maps inside shapes (`VoxelRI`, `FunctionRI`): equal to a scalar when uniform, rotate with the shape, trainable.
- `test_rotation.py`: one rotation convention for shapes and `VoxelVolume` (right-handed, object rotated by R, backward map).
- `test_phase_mask.py`: seeded phase-mask tile maps and their `state_dict` round trip.
- `test_cwfs.py`: `CodedPhaseCamV3` mask plane and shared propagation distance.
- `test_detector.py`: tile/pixel rounding, `PixelIntegrator` (integer, fractional, offset) and `Intensity`.
- `test_sampling.py`: `Magnification`, `Resample` and `output_spacing`.
- `test_device.py`: `.to()` moves every tensor a module owns (checked on the `meta` device).
- `test_gpu.py`: GPU only (marker `gpu`, skipped without CUDA): elements, gradients, scenes (all shapes, RI maps, rotated `VoxelVolume`) and the example 04/06/07 chains on CUDA match the CPU; reference brightness of the example chains. `pytest -m gpu` runs only these; `OPTIVOLUME_TEST_DEVICE=cuda:1` picks the device.
- `fixtures/`: stored reference data for the tests (see `fixtures/README.md`).
