# OptiVolume

`OptiVolume` (`optical_volume`) is a differentiable, PyTorch-based wave-optics toolkit. It builds 3D refractive-index (RI) volumes from simple shapes (spheres, cubes, ellipsoids) or voxel data, and provides optical components such as free-space and beam propagators, a 4f imaging system, phase masks and pixel integration. These chain into imaging pipelines like brightfield and coded wavefront sensing. Every block is a `torch.nn.Module`, so scene and system parameters can be optimised with autograd.

## Installation

Requires Python ≥ 3.9 and PyTorch ≥ 2.4.

```bash
pip install git+https://github.com/Muhammad-Kazim/OptiVolume.git
```

Pin a branch, tag or commit with `@`:

```bash
pip install "git+https://github.com/Muhammad-Kazim/OptiVolume.git@main"
```

**GPU / CUDA builds of PyTorch.** pip installs the default PyTorch wheel from PyPI. For a specific CUDA build, install PyTorch first with the command from [pytorch.org](https://pytorch.org/get-started/locally/), then OptiVolume:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install git+https://github.com/Muhammad-Kazim/OptiVolume.git
```

**Optional extras**

| extra      | adds                                  | for                                   |
|------------|---------------------------------------|---------------------------------------|
| `plot`     | matplotlib                            | 2D plotting helpers                   |
| `mayavi`   | mayavi                                | 3D volume rendering in `visualization`|
| `examples` | matplotlib, tifffile, jupyter         | running the scripts in `examples/`    |
| `dev`      | pytest, build, matplotlib, tifffile   | development                           |

```bash
pip install "optical-volume[examples] @ git+https://github.com/Muhammad-Kazim/OptiVolume.git"
```

Mayavi is often easier to install with conda: `conda install -c conda-forge mayavi`.

### Development install

```bash
git clone https://github.com/Muhammad-Kazim/OptiVolume.git
cd OptiVolume
conda env create -f environment.yml   # optional: full dev environment (edit the torch/CUDA line if needed)
pip install -e ".[dev]"
pytest
```

## Quick start

```python
import torch
from optical_volume import scene, optics

WL = 500e-9                       # wavelength in vacuum [m]
SPACING = (150e-9, 150e-9, 100e-9)
SHAPE = (256, 256, 100)
N_BG = 1.33

# 1. Scene: a bead in water
bead = scene.shapes.Sphere(torch.tensor([19e-6, 19e-6, 5e-6]), torch.tensor(3e-6), torch.tensor(1.46))
vol = scene.CollectionVolume(SHAPE, SPACING, N_BG)
vol.add([bead])

# 2. Propagate a plane wave through the volume (beam propagation method)
exit_field = optics.VolumePropagator(WL, SPACING, N_BG)(vol())

# 3. Refocus to the volume centre and image with a 4f system
imaging = optics.OpticalSystem((
    optics.FreeSpacePropagator(WL / N_BG, SPACING[:2], SHAPE[2] * SPACING[2] / 2,
                               direction="backward", padding=128),
    optics.F4ImagingSystem(WL, SPACING[:2], SHAPE[:2], NA=0.75),
))
intensity = imaging(exit_field).abs() ** 2

# Everything is differentiable, e.g. w.r.t. the bead radius
intensity.mean().backward()
print(bead.radius.grad)
```

See `examples/` for coded wavefront sensing, object rotation, loading 3D models and optimisation.

## Package layout

```
optical_volume/
├── scene/        RI volumes: shapes, CollectionVolume, VoxelVolume, quaternion utils
├── optics/       FreeSpacePropagator, VolumePropagator, F4ImagingSystem,
│                 BinaryPhaseMask, CustomPhaseMask, OpticalSystem, utils
├── pipelines/    CodedPhaseCamV3 (coded wavefront sensing camera)
├── utils/        gradients, TV/L2 regularisers, fitting, NCC, I/O
└── visualization.py   matplotlib / mayavi plotting (optional deps)
```

Units are SI (metres) throughout.

## License and citation

OptiVolume is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/): you may use, share and adapt it for any purpose, provided you give appropriate credit. **If you use OptiVolume in research, please cite it:**

```
Syed Kazim. OptiVolume: differentiable PyTorch wave optics. https://github.com/Muhammad-Kazim/OptiVolume
```

See [LICENSE](LICENSE) for details.
