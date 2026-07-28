import torch
from torchvision.transforms import GaussianBlur

from typing import Optional, Tuple, List
from torch import nn, Tensor


class PhaseMask(nn.Module):
    def __init__(self, side_length: Tensor, spatial_resolution: Tuple[float, float, float], 
                 grid_shape: Tuple[float, float, float], height: Tensor, prob: float = 0.5, binary: bool=True):
        super().__init__()
        
        self.grid_size = grid_shape[:2] # 100x100
        self.tile_size_px = torch.ceil(side_length/spatial_resolution[0]).int() # 25
        self.nx, self.ny = torch.ceil(grid_shape[0]/self.tile_size_px).int(), torch.ceil(grid_shape[1]/self.tile_size_px).int() # 16x16
        
        self.binary = binary
        
        if self.binary:
            assert height.dim() == 0 or height.dim() == 1, f"Height dims is equal to {height.dim()} > 1. Phase mask cannot be binary"
            self.height = nn.Parameter(height)
            self.map = nn.Parameter(self._init_random(prob=prob).float())
        else:
            assert height.size() == self.get_num_tiles(), f"Param height must be of shape {self.nx}x{self.ny}"
            self.map = nn.Parameter(height)
        
    def forward(self, RI_pm: Tensor, wl: float, sigma: float = None, padding: int = 0, RI_bg: float = 1.):
        
        mask = self.map.repeat_interleave(self.tile_size_px, 0).repeat_interleave(self.tile_size_px, 1)[:self.grid_size[0], :self.grid_size[1]]
        mask = torch.clamp(mask, min=0) # height cannot be negative
        mask = torch.nn.functional.pad(mask, pad = ([padding]*4)) # to enable padding, need zeros at bounadry. OW random patterns stretched.
        
        if self.binary:
            mask = _soft_step(mask)*self.height
        
        mask = mask*RI_pm + (mask.max() - mask)*RI_bg
        
        if sigma is not None:
            kernel_size = 2*int(4.*sigma + 0.5) + 1
            mask = GaussianBlur(kernel_size, sigma=sigma)(mask.unsqueeze(0)).squeeze()
    
        return torch.exp(1j*(2*torch.pi/wl)*mask)
    
    def _init_random(self, prob: float = 0.5):
        return torch.rand([self.nx, self.ny]) > prob    
    
    def get_num_tiles(self):
        return (self.nx, self.ny)


class _PhaseMask():
    def __init__(self, side_length: Tensor, spatial_resolution: Tuple[float, float, float], 
                 grid_shape: Tuple[float, float, float]):
        
        self.grid_size = grid_shape[:2]
        self.tile_size_px = torch.ceil(side_length/spatial_resolution[0]).int()
        self.nx, self.ny = torch.ceil(grid_shape[0]/self.tile_size_px).int(), torch.ceil(grid_shape[1]/self.tile_size_px).int()
        
        # self.height_map = None
        
    def forward(self, RI_pm: Tensor, wl: float, sigma: float = None, padding: int = 0, RI_bg: float = 1.):

        assert hasattr(self, 'height_map'), "Create height map first."
        
        mask = self.height_map.repeat_interleave(self.tile_size_px, 0).repeat_interleave(self.tile_size_px, 1)[:self.grid_size[0], :self.grid_size[1]]
        mask = torch.clamp(mask, min=0) # height cannot be negative
        mask = torch.nn.functional.pad(mask, pad = ([padding]*4))
        mask = mask*RI_pm + (mask.max() - mask)*RI_bg
        
        if sigma is not None:
            kernel_size = 2*int(4.*sigma + 0.5) + 1
            mask = GaussianBlur(kernel_size, sigma=sigma)(mask.unsqueeze(0)).squeeze()
    
        return torch.exp(1j*(2*torch.pi/wl)*mask)
    
    def create_height_map(self, height: Tensor, prob: float = 0.5, hmap_grad: bool = False, device: str = 'cpu'):
        
        if height.dim() == 0 or height.dim() == 1:
            self.height_map = self._init_random(prob=prob).float()*height
        else:
            assert height.size() == self.get_num_tiles(), f"Param height must be of shape {self.nx}x{self.ny}"
            self.height_map = height
        
        if hmap_grad:
            self.height_map.requires_grad = True
        
        if device == 'cuda':
            assert torch.cuda.is_available(), "CUDA not available, switch to CPU."
        
        self.height_map.to(device)
    
    def _init_random(self, prob: float = 0.5):
        return torch.rand([self.nx, self.ny]) > prob    
    
    def get_num_tiles(self):
        return (self.nx, self.ny)
    

def _soft_step(x: Tensor, softness=1e-6): 
    return torch.sigmoid(x / softness)


if __name__=='__main__':
    from matplotlib import pyplot as plt
    from torch.optim import Adam
    from torch import nn
    
    # functionality tested. Optim fails however.
    pm_obj1 = PhaseMask(torch.tensor(5.), torch.tensor([1., 1., 1.]), 
                       torch.tensor([100, 100, 100]), torch.tensor(10).float())
    field1 = pm_obj1.forward(torch.tensor(1.3), torch.tensor(500e-9)).detach()
    
    pm_obj2 = PhaseMask(torch.tensor(5.), torch.tensor([1., 1., 1.]), 
                       torch.tensor([100, 100, 100]), (5*pm_obj1.map).detach(), binary=False)
    
    optimizer = Adam(pm_obj2.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    losses = []
    # heights = []
    for i in range(500):
        optimizer.zero_grad()
        
        field2 = pm_obj2.forward(torch.tensor(1.3), torch.tensor(500e-9))
        loss = loss_fn(field1.angle(), field2.angle())
        # print(loss.item())
        losses.append(loss.item())
        # heights.append(pm_obj2.height.detach().item())
        
        loss.backward()
        optimizer.step()
        
    
    # print(loss.item())
    
    plt.plot(losses)
    plt.show()
    
    # plt.plot(heights)
    # plt.show()
    
    plt.imshow(field1.angle().detach())
    plt.colorbar()
    plt.show()
    
    plt.imshow(field2.angle().detach())
    plt.colorbar()
    plt.show()
    