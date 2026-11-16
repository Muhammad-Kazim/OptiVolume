import torch
from torch import nn, Tensor
import torchvision
from torchvision.transforms.functional import resize
from torchvision.transforms import GaussianBlur


class Modulator(nn.Module):
    """Multiply the field by a fixed or trainable transmission function.

    Parameters
    ----------
    operator : Tensor
        Complex or real transmission, broadcastable to the field ``(..., H, W)``.
        Stored as a buffer, or as a parameter if it already is one or ``trainable=True``.
    trainable : bool
        Register ``operator`` as an ``nn.Parameter``.
    """
    def __init__(self, operator: Tensor, trainable: bool = False):
        super().__init__()

        if isinstance(operator, nn.Parameter):
            self.operator = operator  # keep the caller's parameter (and its optimizer)
        elif trainable:
            self.operator = nn.Parameter(torch.as_tensor(operator))
        else:
            self.register_buffer('operator', torch.as_tensor(operator))

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


class GaussBlur(nn.Module):
    def __init__(self, const_sigma: float = 1e-2, defocus_sigma: float = 10e-2, defocus_dist:float = 0.):
        super().__init__()

        sigma = const_sigma + defocus_sigma*torch.abs(torch.tensor(defocus_dist))
        kernel_size = 2*int(4.*sigma + 0.5) + 1
        self.blur = GaussianBlur(kernel_size, sigma=sigma.item())

    def forward(self, field: Tensor):
        return self.blur(field.real.unsqueeze(0)).squeeze() + 1j*self.blur(field.imag.unsqueeze(0)).squeeze()
