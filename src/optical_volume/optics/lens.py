import torch

from typing import Optional, Tuple, List
from torch import nn, Tensor


class F4ImagingSystem(nn.Module):
    def __init__(self, wavelength: float, spacing: Tuple[float, float], shape: Tuple[int, int], NA: float, 
                 pupil_amp_init: Tensor = None, pupil_opd_init: Tensor = None, softness=1e-12):
        super().__init__()

        """maps the wavefield in object focus plane to image plane by modulating with the pupil

        Args:
            wavelength (float): wavelength
            spacing (Tuple[float, float]): lateral resolution in object space (dx, dy)
            shape (Tuple[int, int]): (Nx, Ny) in object space
        """

        self.wl = wavelength
        self.dx, self.dy = spacing[:2]
        self.nx, self.ny = shape[:2]
        self.softness = softness

        self.NA = torch.nn.Parameter(torch.tensor(NA))
        self.pupil_amp = torch.nn.Parameter(torch.ones([self.nx, self.ny])) if pupil_amp_init is None else torch.nn.Parameter(pupil_amp_init)
        self.pupil_opd = torch.nn.Parameter(torch.zeros([self.nx, self.ny]))  if pupil_opd_init is None else torch.nn.Parameter(pupil_opd_init)

    def forward(self, wavefield: Tensor):
        assert wavefield.shape == (self.nx, self.ny), f"Input wavefield shape and class object (nx, ny) must match"

        device = wavefield.device
        wave_spectrum = torch.fft.fftshift(torch.fft.fft2(wavefield))

        kx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, self.dx, device=device))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.ny, self.dy, device=device))
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        fmax = self.NA/self.wl
        K = Kx**2 + Ky**2

        lpf_mask = 1 - torch.sigmoid((K - fmax**2) / self.softness)
        pupil = self.pupil_amp*torch.exp(1j*2*torch.pi/self.wl*self.pupil_opd)

        wave_spectrum = wave_spectrum * lpf_mask * pupil

        return torch.fft.ifft2(torch.fft.ifftshift(wave_spectrum))
