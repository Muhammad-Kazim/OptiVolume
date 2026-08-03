import torch
from torchvision.transforms import GaussianBlur
import torchvision

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
        # self.padding = padding

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
        # self.mask = torch.clamp(self.mask, min=0) # height cannot be negative
        # self.mask = torch.nn.functional.pad(self.mask, pad = ([self.padding]*4)) # to enable padding, need zeros at bounadry. OW random patterns stretched.

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
        # if padding is not None:
        #     self.PAD = torchvision.transforms.Pad(padding) 
        #     self.padding = padding

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
        
        # self.mask = torch.clamp(self.mask, min=0) # height cannot be negative
        # self.mask = torch.nn.functional.pad(self.mask, pad = ([self.padding]*4)) # to enable padding, need zeros at bounadry. OW random patterns stretched.

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

def _soft_step(x: Tensor, softness=1e-6): 
    return torch.sigmoid(x / softness)


if __name__=='__main__':
    from matplotlib import pyplot as plt
    from torch.optim import Adam
    from torch import nn
    
    WL = 500e-9
    SPACING = [100e-9, 100e-9, 100e-9]
    SHAPE = [500, 500, 100]
    NA = 0.9
    DIST_PM_IM = 0.4e-3

    TILE_LEN = 0.25e-6
    PM_HEIGHT = 600e-9
    PM_RI = 1.46
    PM_SIG = 0.6

    N_BG = 1.33
    PAD = 512
    FOC_PLANE_VAR = 0.1e-6

    pm_obj1 = BinaryPhaseMask(WL, TILE_LEN, SPACING, SHAPE, PM_HEIGHT, PM_SIG, PM_RI)

    tile_size_px = torch.ceil(torch.tensor(TILE_LEN/SPACING[0])).int() # 25, sqaure cells in voxels
    num_tiles_x, num_tiles_y = torch.ceil(SHAPE[0]/tile_size_px).int(), torch.ceil(SHAPE[1]/tile_size_px).int() # 16x16, number of tiles
    pm_obj2 = CustomPhaseMask(WL, TILE_LEN, SPACING, SHAPE, torch.rand(num_tiles_x, num_tiles_y).float()*1e-6, PM_SIG, PM_RI)
    
    field = torch.ones(SHAPE[:2], dtype=torch.complex64)
    gt_field = pm_obj1.forward(field).angle().detach()

    optimizer = Adam(pm_obj2.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()

    losses = []
    # heights = []
    for i in range(500):
        optimizer.zero_grad()
        
        field2 = pm_obj2.forward(field).angle()
        if i == 0:
            init_field = field2.detach()

        loss = loss_fn(gt_field, field2)
        print(loss.item())
        losses.append(loss.item())
        # heights.append(pm_obj2.height.detach().item())
        
        loss.backward()
        optimizer.step()
        
    
    # print(loss.item())
    
    plt.plot(losses)
    plt.show()
    
    # plt.plot(heights)
    # plt.show()
    
    fig, axs = plt.subplots(1, 3, figsize=(16, 4))
    
    cm0 = axs[0].imshow(init_field[20:40, 20:40], cmap='gray', vmax=3, vmin=-3)
    cm1 = axs[1].imshow(field2.detach()[20:40, 20:40], cmap='gray', vmax=3, vmin=-3)
    cm2 = axs[2].imshow(gt_field[20:40, 20:40], cmap='gray', vmax=3, vmin=-3)

    plt.colorbar(cm0, ax=axs[0])
    plt.colorbar(cm1, ax=axs[1])
    plt.colorbar(cm2, ax=axs[2])

    plt.show()
    