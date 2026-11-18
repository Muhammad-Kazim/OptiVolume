# tests/fixtures

Reference data used by the tests. Files are NumPy `.npz` archives (no pickle).

## characterization/

Golden outputs of `tests/test_characterization.py`, generated from the code at the start of CS2.
They pin current behaviour, including known physics errors (A1, A2, …), so refactors can't change
numbers silently.

- `bpm_sphere.npz`: `exit_field`, complex 64×64 BPM exit field of one sphere (`test_char_bpm_sphere`).
- `cwfs_reference.npz`: `img_ref`, `img_spn`, 42×42 intensity images of the v5 CWFS chain without and with the sphere (`test_char_cwfs_reference`).
- `coded_phase_cam.npz`: `img`, 16×16 `CodedWavefrontSensor` (was `CodedPhaseCamV3`) image of a phase bump (`test_char_coded_phase_cam`).

Regenerate only in a commit that changes physics on purpose, and include a before/after comparison:

```bash
OPTIVOLUME_UPDATE_FIXTURES=1 pytest tests/test_characterization.py
```
