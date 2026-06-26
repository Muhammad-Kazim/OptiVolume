from torch import nn
import torchvision
from torchvision.transforms.functional import resize


class Modulator(nn.Module):
    def __init__(self, operator):
        super().__init__()
        
        self.operator = operator
        
    def forward(self, field):
        return self.operator*field
    
    
class Magnification(nn.Module):
    def __init__(self, mag_factor):
        super().__init__()
        
        self.mag_factor = mag_factor
        
    def forward(self, field):
        field_real = resize(field.unsqueeze(0).real, field.shape[0]*self.mag_factor, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).squeeze()
        field_imag = resize(field.unsqueeze(0).imag, field.shape[0]*self.mag_factor, torchvision.transforms.InterpolationMode.BILINEAR, antialias=True).squeeze()
        
        return field_real + 1j*field_imag
    
    def set_mag(self, mag_factor):
        self.mag_factor = mag_factor
