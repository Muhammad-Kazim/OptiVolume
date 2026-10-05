import torch
from torch import nn, Tensor
import torchvision
from torchvision.transforms.functional import resize
from torchvision.transforms import GaussianBlur



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
    def __init__(self, mag_factor:float):
        super().__init__()
        
        self.mag_factor = int(mag_factor)
        
    def forward(self, field:Tensor):
        field_real = resize(field.unsqueeze(0).real, field.shape[0]*self.mag_factor, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).squeeze()
        field_imag = resize(field.unsqueeze(0).imag, field.shape[0]*self.mag_factor, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).squeeze()
        
        return (field_real + 1j*field_imag)/self.mag_factor**2
    
    def set_mag(self, mag_factor):
        self.mag_factor = mag_factor
        return self

class PixelIntegrator(nn.Module):
    def __init__(self, stride: int):
        super().__init__()
        
        self.conv2d = nn.Conv2d(1, 1, stride, stride, bias=False)
        
        self.conv2d.weight = nn.Parameter(torch.ones([stride, stride], dtype=torch.complex64).unsqueeze(0).unsqueeze(0))
        self.conv2d.weight.requires_grad = False
        
    def forward(self, field: Tensor):
        return self.conv2d(field.unsqueeze(0)).squeeze()
    
    
# def PSC_approximator(self, const_sigma=1e-2, defocus_sigma=10e-2, defocus_dist=0.):
    #     sigma = const_sigma + defocus_sigma*torch.abs(defocus_dist)
    #     kernel_size = 2*int(4.*sigma + 0.5) + 1
    #     self.blur = GaussianBlur(kernel_size, sigma=sigma.item())

#     output_field = self.blur(output_field.real.unsqueeze(0)).squeeze() + 1j*self.blur(output_field.imag.unsqueeze(0)).squeeze()

class GaussBlur(nn.Module):
    def __init__(self, const_sigma: float = 1e-2, defocus_sigma: float = 10e-2, defocus_dist:float = 0.):
        super().__init__()
        
        sigma = const_sigma + defocus_sigma*torch.abs(torch.tensor(defocus_dist))
        kernel_size = 2*int(4.*sigma + 0.5) + 1
        self.blur = GaussianBlur(kernel_size, sigma=sigma.item())
        
    def forward(self, field: Tensor):
        return self.blur(field.real.unsqueeze(0)).squeeze() + 1j*self.blur(field.imag.unsqueeze(0)).squeeze()