# examples

Install the extras first: `pip install -e ".[examples]"`. Notebooks are committed without outputs.
Run notebooks from this folder; data paths are relative to it (`../data/`, which is not in git).

Numbers follow the planned example set; `02_brightfield_beads` and `03_fit_object_parameters` are
still to be written.

- `01_scene_basics.ipynb`: scene conventions (units, axes, rotations, soft edges) and usage: shapes, refractive-index maps, composition, `VoxelVolume`, `Slab`, gradients.
- `04_coded_wavefront_sensing.ipynb`: coded wavefront sensing chain (BPM → lens → magnification → phase mask → pixels) on two spheres.
- `05_object_rotation.ipynb`: rotating ellipsoids about different axes with `set_quat`.
- `06_voxel_volume_from_mesh.ipynb`: voxelizing a 3D mesh (`../data/benchy.ply`) into a `VoxelVolume` (uses trimesh/open3d/plyfile; to be ported to `scene.io`).
- `07_microbeads_time_series.py`: CWFS images of moving microbeads, one volume per time step.
