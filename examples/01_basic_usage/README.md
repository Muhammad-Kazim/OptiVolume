# examples/01_basic_usage

Install the extras first: `pip install -e ".[examples]"`. Notebooks are committed without outputs.

- `v5_cwfs_system.ipynb`: coded wavefront sensing chain (BPM → lens → magnification → phase mask → pixels) on two spheres.
- `v6_object_rotation.ipynb`: rotating ellipsoids about different axes with `set_quat`.
- `v7_projection_3dloaded.ipynb`: voxelizing a 3D mesh into a `VoxelVolume` (uses trimesh/open3d/plyfile and local data paths; to be ported to `scene.io`).
- `v8_microbeads_time_series.py`: CWFS images of moving microbeads, one volume per time step.
