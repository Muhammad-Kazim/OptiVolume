import torch

from typing import Optional, Tuple, List, Union
from torch import nn, Tensor
import torch.nn.functional as F


class BinaryPhaseMask(nn.Module):

    def __init__(self, wavelength: float, side_length: float, spacing: Tuple[float, float, float], 
                 shape: Tuple[int, int, int], height: float, sigma: float, RI_PM: float, prob: float = 0.5, 
                 RI_bg: float = 1., map: Tensor = None):

        super().__init__()

        self.wl = wavelength
        self.Nx, self.Ny = shape[:2] # 100x100
        self.tile_size_px = torch.ceil(torch.tensor(side_length/spacing[0])).int() # 25, sqaure cells in voxels
        self.num_x, self.num_y = torch.ceil(shape[0]/self.tile_size_px).int(), torch.ceil(shape[1]/self.tile_size_px).int() # 16x16, number of tiles
        self.RI_PM = RI_PM

        self.n0 = RI_bg

        self.height = nn.Parameter(torch.tensor(height))
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))

        if map is None:
            map = self._init_random(prob=prob)
        else:
            assert type(map) == Tensor, "Map must be a Tensor"
            assert map.size() == self.get_num_tiles(), f"Param height must be of shape {self.num_x}x{self.num_y}"

        mask = map.repeat_interleave(self.tile_size_px, 0).repeat_interleave(self.tile_size_px, 1)[:self.Nx, :self.Ny]
        self.register_buffer('mask', mask)

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def forward(self, field: Tensor):

        mask = self.mask * self.height

        mask = mask*self.RI_PM + (mask.max() - mask)*self.n0
        mask = CustomGaussianBlur(mask.unsqueeze(0).unsqueeze(0), self.sigma).squeeze()

        return field*torch.exp(1j*(2*torch.pi/self.wl)*mask)

    def _init_random(self, prob: float = 0.5):
        return (torch.rand([self.num_x, self.num_y]) > prob).float()

    def get_num_tiles(self):
        return (self.num_x, self.num_y)


class CustomPhaseMask(nn.Module):

    def __init__(self, wavelength: float, side_length: float, spacing: Tuple[float, float, float], 
                 shape: Tuple[int, int, int], heightmap: Tensor, sigma: float, RI_PM: float, maxheight: float = 1e-6,
                 RI_bg: float = 1.):

        super().__init__()

        self.wl = wavelength
        self.Nx, self.Ny = shape[:2] # 100x100
        self.tile_size_px = torch.ceil(torch.tensor(side_length/spacing[0])).int() # 25, sqaure cells in voxels
        self.num_x, self.num_y = torch.ceil(shape[0]/self.tile_size_px).int(), torch.ceil(shape[1]/self.tile_size_px).int() # 16x16, number of tiles
        self.RI_PM = RI_PM
        self.max_height = maxheight

        self.n0 = RI_bg

        self.heightmap = nn.Parameter(heightmap)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma)))

        assert type(heightmap) == Tensor, "heightmap must be a Tensor"
        assert heightmap.size() == self.get_num_tiles(), f"Param height must be of shape {self.num_x}x{self.num_y}"

    @property
    def sigma(self):
        return self.log_sigma.exp()

    def forward(self, field: Tensor):

        mask = torch.clamp(self.heightmap, min=-1*self.max_height, max=self.max_height) # height cannot be negative
        mask = mask.repeat_interleave(self.tile_size_px, 0).repeat_interleave(self.tile_size_px, 1)[:self.Nx, :self.Ny]

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
