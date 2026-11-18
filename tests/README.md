# tests

Run with `pytest -q` (in the `optical_volume` conda env). The folders mirror the package
(`tests/optics/` tests `optical_volume/optics/`, and so on); tests that span the package sit at the top.

- `conftest.py`: shared fixtures; `golden` compares arrays with the stored characterization files.
- `test_smoke.py`: fast checks that the package imports and the core blocks run and are differentiable.
- `test_characterization.py`: golden-master tests that pin the numerical outputs (BPM sphere, the example-04 CWFS chain, `CodedWavefrontSensor`).
- `test_public_api.py`: every name in each `__all__` exists, the 0.1 names are gone, `state_dict` keys are as documented.
- `test_device.py`: `.to()` moves every tensor a module owns (checked on the `meta` device).
- `test_gpu.py`: GPU only (marker `gpu`, skipped without CUDA): elements, gradients, scenes (all shapes, RI maps, rotated `VoxelVolume`) and the example 04/06/07 chains on CUDA match the CPU; reference brightness of the example chains. `pytest -m gpu` runs only these; `OPTIVOLUME_TEST_DEVICE=cuda:1` picks the device.
- `optics/`
  - `test_propagation.py`: `AngularSpectrum` and `BeamPropagation`: padding, batch dimensions, evanescent waves, signed distance, the BPM source field.
  - `test_lenses.py`: `FourFSystem` soft pupil edge at the cutoff on any grid; NA gradient matches finite differences.
  - `test_masks.py`: seeded phase-mask tile maps, their `state_dict` round trip, flat border, constructor arguments.
  - `test_sampling.py`: `Magnification`, `Resample`, `output_spacing`, `PartialCoherenceBlur`.
  - `test_detectors.py`: tile/pixel rounding, `PixelIntegrator` (integer, fractional, offset) and `Intensity`.
  - `test_system.py`: `OpticalSystem` recording and hooks, `Transmission` registration.
- `scene/`
  - `test_shapes.py`: factories with rotations, in-place `set_quat`; SDF occupancy (0.5 on the surface), softness in metres, gradients, composition in `CollectionVolume`, `Slab`.
  - `test_refractive_index.py`: refractive-index maps inside shapes (`VoxelRI`, `FunctionRI`): equal to a scalar when uniform, rotate with the shape, trainable.
  - `test_rotation.py`: one rotation convention for shapes and `VoxelVolume` (right-handed, object rotated by R, backward map).
  - `test_volume.py`: `VoxelVolume` defaults (centre, required `n_medium`); points outside the box are the background.
- `pipelines/test_coded_wavefront_sensing.py`: `CodedWavefrontSensor` mask plane, shared propagation distance, integration option.
- `utils/test_fitting.py`: `fit_quadratic` (each term's coefficient, optional spacing in metres).
- `utils/test_io.py`: `normalization` to the full uint8/uint16 range.
- `visualization/test_plots.py`: 2D plots put x to the right and y up, labelled (skipped without matplotlib).
- `fixtures/`: stored reference data for the tests (see `fixtures/README.md`).
