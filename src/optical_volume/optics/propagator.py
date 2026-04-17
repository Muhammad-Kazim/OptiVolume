import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List

from py_wave_propagator import torch_volume_prop as propagator


class Freespace(nn.Module):
    def __init__(self, WL: float, spacing: Tuple[float, float], shape: Tuple[int, int], padding=None, pad_mode='edge'):
        super().__init__()
        
        self.prop = propagator.FreeSpacePropagator(WL, spacing, shape, padding=padding, pad_mode=pad_mode)
        
    def forward(self, field):
        field = self.prop.forward(field, self.dist, direction=self.direction)
        
        return field
    
    def set_params(self, dist: float, direction='forward'):
        self.dist = dist
        self.direction = direction
        
        return self


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    
    
    # example: functioning
    WL = 500e-9
    spacing = [10e-6, 10e-6]
    shape = [100, 100]
    
    x = (torch.arange(100)-50)*spacing[0]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    wavefield = torch.where(X**2 + Y**2 < (300e-6)**2, 1+0j, 0+0j)
    
    prop = Freespace(WL, spacing, shape, padding=1000)
    field = prop(wavefield, 50e-3)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()