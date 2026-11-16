# tests

Run with `pytest -q` (in the `optical_volume` conda env).

- `conftest.py`: shared fixtures; `golden` compares arrays with the stored characterization files.
- `test_smoke.py`: fast checks that the package imports and the core blocks run and are differentiable.
- `test_characterization.py`: golden-master tests that pin today's numerical outputs (BPM, v5 CWFS chain, `CodedPhaseCamV3`).
- `fixtures/`: stored reference data for the tests (see `fixtures/README.md`).
