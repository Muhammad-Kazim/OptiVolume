# examples

Install the extras first: `pip install -e ".[examples]"`. Notebooks are committed without outputs.
Run notebooks from this folder; data paths are relative to it (`../data/`, which is not in git).

Planned replacements for `05`–`07`: fitting one parameter (`03`) and several (`04`), data
generation with the CWFS pipeline (`05`), and working with voxel volumes (`06`).

- `01_scene_basics.ipynb`: scene conventions (units, axes, rotations, soft edges) and usage: shapes, refractive-index maps, composition, `VoxelVolume`, `Slab`, gradients.
- `02_build_cwfs_sensor.ipynb`: a coded wavefront sensor assembled from components behind a microscope; reference and object speckle, amplitude and OPD at the phase mask, OPD gradients and the speckle shift d·∇W.
- `05_object_rotation.ipynb`: rotating ellipsoids about different axes with `set_quat`.
- `06_voxel_volume_from_mesh.ipynb`: voxelizing a 3D mesh (`../data/benchy.ply`) into a `VoxelVolume` (needs the `mesh` extra; the 3D view needs mayavi; to be ported to `scene.io`).
- `07_microbeads_time_series.py`: CWFS images of moving microbeads, one volume per time step.
