import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List

from py_wave_propagator import torch_volume_prop as propagator


class Freespace(nn.Module):
    def __init__(self, WL: float, spacing: Tuple[float, float], shape: Tuple[int, int], padding=None, pad_mode='edge'):
        super().__init__()
        
        self.prop = propagator.FreeSpacePropagator(WL, spacing, shape, padding=padding, pad_mode=pad_mode)
        
    def forward(self, field: Tensor) -> Tensor:
        field = self.prop.forward(field, self.dist, direction=self.direction)
        
        return field
    
    def set_params(self, dist: float, direction='forward'):
        self.dist = dist
        self.direction = direction
        
        return self
    

class BeamPropMethod(nn.Module):
    def __init__(self, WL: float, spacing: Tuple[float, float, float], shape: Tuple[int, int, int], padding=None, pad_mode: str ='edge'):
        super().__init__()
        
        self.prop = propagator.VolumePropagator(WL, spacing, shape, padding=padding, pad_mode=pad_mode)
        
    def forward(self, field: Tensor, RI_distribution: Tensor, RI_background: float) -> Tensor:
        field = self.prop.forward(field, RI_distribution, RI_background)
        
        return field
    

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    # example: free space propagation using ASM
    WL = 500e-9
    spacing = [10e-6, 10e-6]
    shape = [100, 100]
    
    x = (torch.arange(100)-50)*spacing[0]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    wavefield = torch.where(X**2 + Y**2 < (300e-6)**2, 1+0j, 0+0j)
    
    prop = Freespace(WL, spacing, shape, padding=1000)
    field = prop.set_params(50e-3).forward(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    # example: beam propagation method
    import os
    import sys
    sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-1]))
    import scene
    
    # example 1: create two spheres and visulization the RI volume
    sphere1 = scene.shapes.Sphere(torch.tensor((55, 50, 40)).float(), torch.tensor((10.)), torch.tensor(1.3), softness=1e2)
    sphere2 = scene.shapes.Sphere(torch.tensor((30, 50, 40)).float(), torch.tensor((20.)), torch.tensor(1.3), softness=1e2)
    
    grid = scene.grid.Grid((100, 100, 100), (1, 1, 1))
    
    vol = scene.volume.Volume(grid, n_bg=torch.tensor(1.33))
    vol.add([sphere1, sphere2])
    # print(vol.grid.X.dtype, vol.n_bg)
    
    ri_dist = vol.forward()
    # print(ri_dist.shape)
    
    vol_prop = BeamPropMethod(WL, spacing + [10e-6], shape + [100], padding=0)
    exit_field = vol_prop.forward(torch.ones_like(field), ri_dist, 1.33)
    
    plt.imshow(exit_field.abs().detach())
    plt.colorbar()
    plt.show()
    
    prop = Freespace(WL/1.33, spacing, shape, padding=1000)
    centered_field = prop.set_params(500e-6, direction='backward').forward(exit_field)
    
    plt.imshow(centered_field.abs().detach())
    plt.colorbar()
    plt.show()
    
    print(centered_field.requires_grad)
    