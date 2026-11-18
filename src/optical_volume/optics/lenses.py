import math

import torch

from typing import Optional, Tuple, List
from torch import nn, Tensor

#: 10-90 % width of a logistic step in units of its softness: 2 ln 9.
_EDGE_WIDTH_PER_SOFTNESS = 2 * math.log(9)


class FourFSystem(nn.Module):
    """4f imaging: maps the field in the object focal plane to the image plane through a pupil.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength in metres.
    spacing : tuple of float
        Lateral sampling ``(dx, dy)`` of the object-side field, in metres.
    shape : tuple of int
        ``(nx, ny)`` samples of the object-side field.
    NA : float
        Numerical aperture (parameter). The pupil passes spatial frequencies ``|f| < NA / wavelength``.
    pupil_amp_init : Tensor, optional
        ``(nx, ny)`` pupil amplitude (parameter), on the centred frequency grid. Default: ones.
    pupil_opd_init : Tensor, optional
        ``(nx, ny)`` pupil optical path difference in metres (parameter). Default: zeros.
    softness : float
        Width of the soft aperture edge, in units of the frequency spacing
        ``df = 1 / (n * d)`` (dimensionless). The default gives a 10-90 % edge about one
        frequency sample wide; it keeps the image differentiable with respect to ``NA``.

    Notes
    -----
    The aperture is ``sigmoid((NA/wavelength - |f|) / (softness * df))``: 0.5 exactly at the
    cutoff, on any grid. A hard edge (``softness -> 0``) has no gradient with respect to NA.
    """

    def __init__(self, wavelength: float, spacing: Tuple[float, float], shape: Tuple[int, int], NA: float,
                 pupil_amp_init: Tensor = None, pupil_opd_init: Tensor = None,
                 softness: float = 1 / _EDGE_WIDTH_PER_SOFTNESS):
        super().__init__()

        self.wl = wavelength
        self.dx, self.dy = spacing[:2]
        self.nx, self.ny = shape[:2]
        self.softness = float(softness)

        self.NA = torch.nn.Parameter(torch.tensor(NA))
        self.pupil_amp = torch.nn.Parameter(torch.ones([self.nx, self.ny])) if pupil_amp_init is None else torch.nn.Parameter(pupil_amp_init)
        self.pupil_opd = torch.nn.Parameter(torch.zeros([self.nx, self.ny]))  if pupil_opd_init is None else torch.nn.Parameter(pupil_opd_init)

    def cutoff_mask(self, device=None) -> Tensor:
        """Soft aperture ``(nx, ny)`` on the centred (fftshifted) frequency grid, in [0, 1]."""
        device = self.NA.device if device is None else device
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, self.dx, device=device))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.ny, self.dy, device=device))
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
        f = torch.sqrt(Kx**2 + Ky**2)
        df = min(1 / (self.nx * self.dx), 1 / (self.ny * self.dy))
        # The edge width is set in frequency samples, so it is resolved on any grid (A14).
        return torch.sigmoid((self.NA / self.wl - f) / (self.softness * df))

    def forward(self, wavefield: Tensor):
        assert wavefield.shape == (self.nx, self.ny), f"Input wavefield shape and class object (nx, ny) must match"

        wave_spectrum = torch.fft.fftshift(torch.fft.fft2(wavefield))
        pupil = self.pupil_amp*torch.exp(1j*2*torch.pi/self.wl*self.pupil_opd)
        wave_spectrum = wave_spectrum * self.cutoff_mask(wavefield.device) * pupil

        return torch.fft.ifft2(torch.fft.ifftshift(wave_spectrum))
