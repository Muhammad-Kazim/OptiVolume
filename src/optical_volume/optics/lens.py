import numpy as np
from scipy.signal import convolve2d as convolve
import torch
import torchvision

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


# class ObjImgMap(nn.Module):
#     def __init__(self, wavelength: float, spatial_resolution: Tuple[float, float], numPx: Tuple[int, int]):
#         super().__init__()
        
#         """maps the wavefield in object focus plane to image plane by modulating with the pupil

#         Args:
#             wavelength (float): wavelength
#             spatial_resolution (Tuple[float, float]): lateral resolution in object space (dx, dy)
#             numPx (Tuple[float, float]): (Nx, Ny) in object space
#         """

#         self.wl = wavelength
#         self.dx, self.dy = spatial_resolution[:2]
#         self.nx, self.ny = numPx[:2]
                
#         kx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, self.dx))
#         ky = torch.fft.fftshift(torch.fft.fftfreq(self.ny, self.dy))
#         self.Kx, self.Ky = torch.meshgrid(kx, ky, indexing='ij')
        
#         # aperture does nothing
#         self.pupil_amp = torch.ones([self.nx, self.ny])
#         self.pupil_phase = torch.zeros([self.nx, self.ny])
        
#     def forward(self, wavefield: Tensor):
        
#         pupil = self.pupil_amp*torch.exp(1j*2*torch.pi/self.wl*self.pupil_phase)
#         wave_spectrum = torch.fft.fftshift(torch.fft.fft2(wavefield))*pupil
        
#         return torch.fft.ifft2(torch.fft.ifftshift(wave_spectrum))
    
#     def displaced_unifrom_pupil_amp(self, pupil_center = [0., 0.], pupil_radius=1e8, softness=1e-12):
            
#         K = (self.Kx - pupil_center[0])**2 + (self.Ky - pupil_center[1])**2
#         self.pupil_amp = 1 - torch.sigmoid((K - pupil_radius**2) / softness)
        
#     def low_pass_filter(self, NA: float, softness=1e-12):
#         fmax = NA/self.wl
#         self.displaced_unifrom_pupil_amp(pupil_center = [0., 0.], pupil_radius=fmax, softness=softness)
#         return self
        
#     def set_pupil_amp(self, amp):
#         assert amp.size() == (self.nx, self.ny), f"Aperture amplitude must have sizes {self.nx}x{self.ny}"
#         self.pupil_amp = amp
#         return self
        
#     def set_pupil_phase(self, phase):
#         assert phase.size() == (self.nx, self.ny), f"Aperture phase must have sizes {self.nx}x{self.ny}"
#         self.pupil_phase = phase
#         return self


# def low_pass_filter(wavelength, spatial_resolution, numPx, NA):
#     lpf = ObjImgMap(wavelength, spatial_resolution, numPx)
#     return lpf.low_pass_filter(NA)


# def low_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
#     fmax = NA/wl
#     dx, dy = spatial_resolution[:2]
    
#     kx = np.fft.fftfreq(wavefield.shape[0], dx)
#     ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
#     Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
#     K = np.sqrt(Kx**2 + Ky**2)
#     mask = np.ones_like(K)
#     mask[K > fmax] = 0.
    
#     wave_spectrum = np.fft.fft2(wavefield)*mask
    
#     return np.fft.ifft2(wave_spectrum)


# def high_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
#     fmax = NA/wl
#     dx, dy = spatial_resolution[:2]
    
#     kx = np.fft.fftfreq(wavefield.shape[0], dx)
#     ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
#     Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
#     K = np.sqrt(Kx**2 + Ky**2)
#     mask = np.ones_like(K)
#     mask[K < fmax] = 0.
    
#     wave_spectrum = np.fft.fft2(wavefield)*mask
    
#     return np.fft.ifft2(wave_spectrum)


# def band_pass_filter_NA(wavefield, wl, spatial_resolution, NA, loc):
#     # ideally loc in space, means loc/(wl*focal_length) in the spectreum
#     # but here loc = f_c
    
#     fmax = NA/wl
#     dx, dy = spatial_resolution[:2]
    
#     kx = np.fft.fftshift(np.fft.fftfreq(wavefield.shape[0], dx))
#     ky = np.fft.fftshift(np.fft.fftfreq(wavefield.shape[1], dy))
    
#     Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
#     K = np.sqrt(Kx**2 + Ky**2)
#     mask = np.ones_like(K)
#     mask[K > fmax] = 0.
    
#     wave_spectrum = np.fft.fftshift(np.fft.fft2(wavefield))*np.roll(mask, (loc[0], loc[1]), axis=(1, 0))
    
#     return np.fft.ifft2(np.fft.ifftshift(wave_spectrum))


if __name__=='__main__':
    from matplotlib import pyplot as plt
    
    # example: functions properly, chaining tested, can be used in an optics system
    WL = 500e-9
    spacing = [10e-6, 10e-6]
    shape = [100, 100]
    NA = 0.01
    
    lens = F4ImagingSystem(WL, spacing, shape, NA)
    
    x = (torch.arange(100)-50)*spacing[0]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    wavefield = torch.where(X**2 + Y**2 < (300e-6)**2, 1+0j, 0+0j)
    
    field = lens.forward(wavefield)
    
    # print(lens)
    plt.imshow(field.abs().detach())
    plt.colorbar()
    plt.show()