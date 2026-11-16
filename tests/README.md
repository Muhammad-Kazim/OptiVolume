# tests

Run with `pytest -q` (in the `optical_volume` conda env).

- `conftest.py`: shared fixtures; `golden` compares arrays with the stored characterization files.
- `test_smoke.py`: fast checks that the package imports and the core blocks run and are differentiable.
- `test_characterization.py`: golden-master tests that pin today's numerical outputs (BPM, v5 CWFS chain, `CodedPhaseCamV3`).
- `test_propagator.py`: propagator padding, batch dimensions and the BPM source field.
- `test_system.py`: `OpticalSystem` recording and hooks, `Modulator` registration.
- `test_shapes.py`: shape factories with rotations, in-place `set_quat`.
- `test_volume.py`: `VoxelVolume` defaults (centre, required `n0`).
- `test_phase_mask.py`: seeded phase-mask tile maps and their `state_dict` round trip.
- `test_device.py`: `.to()` moves every tensor a module owns; CUDA vs CPU when a GPU is present.
- `fixtures/`: stored reference data for the tests (see `fixtures/README.md`).
