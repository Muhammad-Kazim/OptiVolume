"""OptiVolume: differentiable PyTorch wave optics.

Subpackages
-----------
scene      refractive-index volumes built from analytic shapes or voxel data
optics     propagators, lenses, phase masks and the OpticalSystem container
pipelines  ready-made imaging models (e.g. coded wavefront sensing)
utils      gradients, regularisers, fitting and I/O helpers
visualization  plots (needs matplotlib; not imported by default)

The main classes are also available at the top level, e.g. ``optical_volume.Sphere``.
"""
from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("optical-volume")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

from . import optics, pipelines, scene, utils
from .optics import *  # noqa: F401,F403  (the elements, listed in optics.__all__)
from .pipelines import *  # noqa: F401,F403
from .scene import *  # noqa: F401,F403

# visualization needs matplotlib (extra "plot"); import it explicitly: from optical_volume import visualization
__all__ = ["optics", "pipelines", "scene", "utils", "__version__",
           *optics.__all__, *pipelines.__all__, *scene.__all__]
