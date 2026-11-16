import math

import torch

from typing import Optional, Tuple
from torch import nn, Tensor
import torch.nn.functional as F

from .utils import samples_per


def _tile_layout(side_length: float, spacing, shape):
    """Tile size in pixels and number of tiles, per axis.

    Returns ``((tile_x, tile_y), (num_x, num_y))``. Tile sizes are rounded to the nearest
    whole number of samples, with a warning if the tile doesn't fit the grid.
    """
    tile = tuple(samples_per(side_length, d, "tile size") for d in spacing[:2])
    num = tuple(math.ceil(n / t) for n, t in zip(shape[:2], tile))
    return tile, num


def _expand_tiles(tiles: Tensor, tile_size_px, out_shape) -> Tensor:
    """Repeat each tile ``tile_size_px`` times per axis and crop to ``out_shape``."""
    tx, ty = tile_size_px
    return tiles.repeat_interleave(tx, 0).repeat_interleave(ty, 1)[:out_shape[0], :out_shape[1]]


class BinaryPhaseMask(nn.Module):
    """Binary-height phase mask made of square tiles.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength, in metres.
    side_length : float
        Tile edge length, in metres.
    spacing : tuple of float
        Field sampling ``(dx, dy, ...)`` in metres.
    shape : tuple of int
        Field size ``(Nx, Ny, ...)`` in pixels.
    height : float
        Step height of a raised tile, in metres (parameter).
    sigma : float
        Width of the Gaussian edge blur, in pixels (parameter, stored as its log).
    RI_PM : float
        Refractive index of the mask material.
    prob : float
        A tile is raised when its uniform draw exceeds ``prob``.
    RI_bg : float
        Refractive index of the surrounding medium.
    tile_map : Tensor, optional
        ``(num_x, num_y)`` 0/1 map of raised tiles. If given, nothing is drawn.
    seed : int, optional
        Seed for drawing the map. Default: a seed drawn from ``generator``, or from the
        global RNG if no generator is given.
    generator : torch.Generator, optional
        Source of the seed when ``seed`` is not given.

    Notes
    -----
    The map is drawn on the CPU from a private generator, so the global RNG is advanced by
    one draw only and the map doesn't depend on the device. The seed used is stored as
    ``self.seed`` (``None`` for a given map) and saved in the ``state_dict`` with the map
    (buffer ``tile_map``), so any mask can be rebuilt exactly.
    """

    def __init__(self, wavelength: float, side_length: float, spacing: Tuple[float, float, float],
                 shape: Tuple[int, int, int], height: float, sigma: float, RI_PM: float, prob: float = 0.5,
                 RI_bg: float = 1., tile_map: Optional[Tensor] = None, seed: Optional[int] = None,
                 generator: Optional[torch.Generator] = None):

        super().__init__()

        self.wl = wavelength
        self.Nx, self.Ny = shape[:2] # 100x100
        self.tile_size_px, (self.num_x, self.num_y) = _tile_layout(side_length, spacing, shape)
        self.RI_PM = RI_PM

        self.n0 = RI_bg

        self.height = nn.Parameter(torch.tensor(height))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))

        if tile_map is None:
            if seed is not None and generator is not None:
                raise ValueError("give either seed or generator, not both")
            if seed is None:
                seed = int(torch.randint(0, 2**63 - 1, (), generator=generator))
            self.seed = int(seed)
            tile_map = self._init_random(prob=prob)
        else:
            if seed is not None or generator is not None:
                raise ValueError("seed and generator are only used when tile_map is not given")
            if not isinstance(tile_map, Tensor):
                raise TypeError("tile_map must be a Tensor")
            if tuple(tile_map.shape) != self.get_num_tiles():
                raise ValueError(f"tile_map must be of shape {self.num_x}x{self.num_y}, "
                                 f"got {tuple(tile_map.shape)}")
            self.seed = None

        self.register_buffer('tile_map', tile_map.detach().float().clone())

    @property
    def sigma(self):
        return self.log_sigma.exp()

    @property
    def mask(self) -> Tensor:
        """0/1 map expanded to the field size ``(Nx, Ny)``."""
        return _expand_tiles(self.tile_map, self.tile_size_px, (self.Nx, self.Ny))

    def forward(self, field: Tensor):

        mask = self.mask * self.height

        mask = mask*self.RI_PM + (mask.max() - mask)*self.n0
        mask = CustomGaussianBlur(mask.unsqueeze(0).unsqueeze(0), self.sigma).squeeze()

        return field*torch.exp(1j*(2*torch.pi/self.wl)*mask)

    def _init_random(self, prob: float = 0.5):
        g = torch.Generator(device='cpu').manual_seed(self.seed)
        return (torch.rand([self.num_x, self.num_y], generator=g) > prob).float()

    def get_num_tiles(self):
        return (self.num_x, self.num_y)

    def get_extra_state(self):
        return {'seed': self.seed}

    def set_extra_state(self, state):
        self.seed = state['seed']


class CustomPhaseMask(nn.Module):
    """Phase mask with one trainable height per square tile.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength, in metres.
    side_length : float
        Tile edge length, in metres.
    spacing : tuple of float
        Field sampling ``(dx, dy, ...)`` in metres.
    shape : tuple of int
        Field size ``(Nx, Ny, ...)`` in pixels.
    heightmap : Tensor
        ``(num_x, num_y)`` tile heights in metres (parameter).
    sigma : float
        Width of the Gaussian edge blur, in pixels (parameter, stored as its log).
    RI_PM : float
        Refractive index of the mask material.
    maxheight : float
        Heights are clamped to ``[-maxheight, maxheight]``, in metres.
    RI_bg : float
        Refractive index of the surrounding medium.
    """

    def __init__(self, wavelength: float, side_length: float, spacing: Tuple[float, float, float],
                 shape: Tuple[int, int, int], heightmap: Tensor, sigma: float, RI_PM: float, maxheight: float = 1e-6,
                 RI_bg: float = 1.):

        super().__init__()

        self.wl = wavelength
        self.Nx, self.Ny = shape[:2] # 100x100
        self.tile_size_px, (self.num_x, self.num_y) = _tile_layout(side_length, spacing, shape)
        self.RI_PM = RI_PM
        self.max_height = maxheight

        self.n0 = RI_bg

        if not isinstance(heightmap, Tensor):
            raise TypeError("heightmap must be a Tensor")
        if tuple(heightmap.shape) != self.get_num_tiles():
            raise ValueError(f"heightmap must be of shape {self.num_x}x{self.num_y}, "
                             f"got {tuple(heightmap.shape)}")

        self.heightmap = nn.Parameter(heightmap)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def forward(self, field: Tensor):

        mask = torch.clamp(self.heightmap, min=-1*self.max_height, max=self.max_height) # height cannot be negative
        mask = _expand_tiles(mask, self.tile_size_px, (self.Nx, self.Ny))

        mask = mask*self.RI_PM + (mask.max() - mask)*self.n0
        mask = CustomGaussianBlur(mask.unsqueeze(0).unsqueeze(0), self.sigma).squeeze()

        return field*torch.exp(1j*(2*torch.pi/self.wl)*mask)

    def get_num_tiles(self):
        return (self.num_x, self.num_y)


def CustomGaussianBlur(field: Tensor, sigma: Tensor, channels: int = 1):
    """
    field: (N, C, H, W)
    """
    device = field.device
    dtype = field.dtype

    sigma = sigma.clamp(min=1e-4)

    kernel_size = 2*int(4.*sigma + 0.5) + 1
    radius = kernel_size // 2
    coords = torch.arange(-radius, radius + 1, device=device, dtype=dtype)

    # 1D Gaussian
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()

    # 2D Gaussian (separable outer product)
    kernel2d = torch.outer(kernel, kernel)
    kernel2d = kernel2d / kernel2d.sum()

    # Depthwise convolution kernel
    kernel2d = kernel2d.expand(
        channels, 1, kernel_size, kernel_size
    )

    return F.conv2d(
        field,
        kernel2d,
        padding=radius,
        groups=channels,
    )
