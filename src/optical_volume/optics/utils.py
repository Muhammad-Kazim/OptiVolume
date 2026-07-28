import torch
from torch import nn, Tensor
import torchvision
from torchvision.transforms.functional import resize


class Modulator(nn.Module):
    """
    Makes transmission functions compatible with system.OpticalSystem module.
    """
    def __init__(self, operator):
        super().__init__()
        
        self.operator = operator
        
    def forward(self, field):
        return self.operator*field
    
    
class Magnification(nn.Module):
    def __init__(self, mag_factor):
        super().__init__()
        
        self.mag_factor = self.register_buffer('mag_factor', mag_factor)
        
    def forward(self, field):
        field_real = resize(field.unsqueeze(0).real, field.shape[0]*self.mag_factor, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).squeeze()
        field_imag = resize(field.unsqueeze(0).imag, field.shape[0]*self.mag_factor, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).squeeze()
        
        return field_real + 1j*field_imag
    
    def set_mag(self, mag_factor):
        self.mag_factor = mag_factor
        return self

class PixelIntegrator(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        
        self.conv2d = nn.Conv2d(1, 1, stride, self.sum_size.int().item(), bias=False)
        
        self.conv2d.weight = nn.Parameter(torch.ones([stride, stride]).unsqueeze(0).unsqueeze(0)/stride**2)
        self.conv2d.weight.requires_grad = False
        
    def forward(self, field: Tensor):
        return self.conv2d(field.unsqueeze(0)).squeeze()