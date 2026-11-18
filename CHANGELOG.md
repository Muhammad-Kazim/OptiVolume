# Changelog

All notable changes to OptiVolume. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [semantic versioning](https://semver.org/) (before 1.0, a minor version may break the API).

## [0.2.0] — unreleased

A breaking release. The package layout and names change (no aliases for the old ones), and a
series of physics fixes **change simulation results**. The `v0.1` tag keeps the old API and
results reproducible.

### Changed: layout and names (PR for CS5)
- One concept per module:
  - optics: `propagation`, `lenses`, `masks`, `sampling`, `detectors`, `coherence`, `system`;
  - scene: `base`, `shapes`, `rotations`, `volume`, `refractive_index`;
  - `utils` is flat: `fitting`, `gradients`, `filters`, `io`;
  - `visualization` is a package: `plots`, plus `render3d` for mayavi.
- Every subpackage defines `__all__`. The optics, scene and pipelines classes are also
  available at the top level (`optical_volume.Sphere`).
- Removed: `AngularSpectrum.set_direction` (use `direction=` or a signed distance),
  `utils.HiddenPrints`, the unused `softness` argument of the coded wavefront sensor, the
  `optics.utils` and `utils.utils` namespaces.
- `state_dict` keys follow the new names, so models saved with 0.1 don't load.
- New optional extra `mesh` (`plyfile`, `trimesh`) for example 06.

| 0.1 | 0.2 |
|---|---|
| `optics.FreeSpacePropagator` | `optics.AngularSpectrum` |
| `optics.VolumePropagator(wl, spacing, RI_background)`, `forward(RI_distribution)` | `optics.BeamPropagation(wl, spacing, n_medium)`, `forward(n)` |
| `optics.F4ImagingSystem` | `optics.FourFSystem` |
| `optics.utils.Modulator` | `optics.Transmission` |
| `optics.CustomPhaseMask` | `optics.HeightMapPhaseMask` |
| `BinaryPhaseMask(..., RI_PM, RI_bg=)` | `BinaryPhaseMask(..., n_mask, n_medium=)` |
| `optics.utils.Magnification`, `Resample`, `Intensity`, `PixelIntegrator`, `PartialCoherenceBlur`, `samples_per` | `optics.Magnification`, … (same names, one level up) |
| `pipelines.CodedPhaseCamV3` | `pipelines.CodedWavefrontSensor` |
| `scene.shapes.Cube`, `make_cubes` | `scene.Cuboid`, `make_cuboids` |
| `scene.shapes.Plane` | `scene.Slab(point, normal, thickness, n)` |
| shape `RI` (argument and attribute), `ri_map` | `n`, `n_map` |
| `make_spheres(centers, radii, RIs)` (and cubes, ellipsoids) | `make_spheres(centers, radii, n)` |
| `CollectionVolume(shape, spacing, n_bg)` | `CollectionVolume(shape, spacing, n_medium)` |
| `VoxelVolume(..., n0=)` | `VoxelVolume(..., n_medium=)` |
| `scene.utils` | `scene.rotations` |
| `utils.utils.<f>` | `utils.<f>` |
| `utils.torch_grad_optr`, `torch_TV`, `torch_L2_grad`, `auto_corr_fn` | `utils.finite_differences`, `total_variation`, `gradient_l2`, `autocorrelation` |
| `utils.plot` | `visualization.plot` |
| `visualize_grid*(RI_distribution, n_background=)` | `visualize_grid*(n, n_medium=)` |

### Fixed: physics (results change; CS4a–CS4d)
- **Propagation:** evanescent waves are removed in free space (A3); the propagation distance
  is signed, so focus searches pass through 0 (A4); the BPM pad is background medium (A11); the
  coded wavefront sensor propagates back to the mask and forward to the sensor (A1).
- **Detector and sampling:** pixels integrate intensity, Σ|E|² (A2); `Magnification(M)` is
  physical (amplitude ÷ M), separate from `Resample(f)` (Fourier by default) (A5);
  `PartialCoherenceBlur` replaces a blur that did nothing (A6); the mask's edge blur pads by
  replication (A7); tile and pixel sizes round to whole samples, with a warning (A8).
- **Scene and lens:**
  - `VoxelVolume` is the background outside its box (A9);
  - one rotation convention (active, right-handed) for shapes and voxel objects (A10);
  - `Slab` replaces the broken `Plane` (A12);
  - shapes are signed distances with softness in metres and a default of one voxel (A13);
  - occupancy and refractive index are composed without dividing by RI (A18);
  - the pupil edge is soft in frequency samples, so NA is trainable (A14).
- **Utilities:** `fit_quadratic` uses each term's coefficient (A15); `normalization` maps to the
  full uint range (A16); plots show x to the right and y up (A17).

### Added
- `scene.VoxelRI` and `scene.FunctionRI`: refractive index varying inside a shape.
- `optics.Intensity`; `PixelIntegrator` with non-integer factors and a trainable offset;
  `OpticalSystem.output_spacing`.
- `BinaryPhaseMask(seed=, generator=)`: reproducible tile maps, saved in the `state_dict`.
- `examples/01_scene_basics.ipynb`: conventions and usage.
- Tests: characterization fixtures, GPU tests (`pytest -m gpu`), public-API tests.

### Fixed: crashes and state (CS3)
- Padded BPM, `padding=0` and batched inputs work. The `make_*` factories run. `VoxelVolume`
  has proper defaults. `.to(device)` moves every tensor a module owns. Setters update
  parameters in place, so optimizers keep working.

## [0.1.0] — 2026-04-13

The state used for the arXiv paper (tag `v0.1`, environment in `arxiv/environment.yml`).
