"""OptiVolume: differentiable PyTorch wave optics.

Subpackages
-----------
scene      refractive-index volumes built from analytic shapes or voxel data
optics     propagators, lenses, phase masks and the OpticalSystem container
pipelines  ready-made imaging models (e.g. coded wavefront sensing)
utils      gradients, regularisers, fitting and I/O helpers
"""
from importlib.metadata import version as _version, PackageNotFoundError

try:
    __version__ = _version("optical-volume")
except PackageNotFoundError:  # running from a source tree without installing
    __version__ = "0.0.0+unknown"

from . import scene
from . import optics
from . import utils
from . import pipelines

__all__ = ["scene", "optics", "utils", "pipelines", "__version__"]
