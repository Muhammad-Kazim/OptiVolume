import torch
from torch import nn, Tensor
import torchvision
from typing import Optional, Tuple, List


class FreeSpacePropagator(nn.Module):
    """
        Propagation through a homogenous medium. Base unit is meters.
        Use when input has to be updated.

        Args:
            field (Tensor): 2d complex field on a plane
            wavelength (float): if not air, than wl =/ RI_background
            spatial_resolution (): _description_
            dist (float): distance bw parallel planes in meters

        Returns:
            NDArray[np.complex128]: field at parallel plane distance dist away 
    """
    def __init__(self,
              wavelength: float, 
              spacing: Tuple[float, float],
              distance: float,
              direction: Optional[str] = 'forward',
              padding: Optional[int] = None, 
              pad_mode: str = 'edge',
    ) -> Tensor:
        super().__init__()

        if padding is not None:
            self.pad_do = True
            self.pad_by = padding
            self.PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode) # edge make sense
        else:
            self.pad_do = False

        self.wavelength = wavelength
        self.dx, self.dy = spacing
        self.direction = direction

        self.distance = nn.Parameter(torch.tensor(distance))

    def forward(self, field: Tensor) -> Tensor:

        device = field.device

        if self.pad_do:
            field = self.PAD(field.real) + 1j*self.PAD(field.imag)

        H, W = field.shape[-2:]

        # Spatial frequency grid
        kx = torch.fft.fftfreq(H, self.dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(W, self.dy, device=device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        Kz = (2 * torch.pi/self.wavelength)**2 - Kx**2 - Ky**2
        mask = torch.sigmoid(Kz/1e-12) # to remove negatives which remove evanascent waves
        Kz = torch.sqrt(Kz*mask)

        field_fft = torch.fft.fft2(field)
        if (self.direction == 'backward' and self.distance > 0.) or (self.direction == 'forward' and self.distance < 0.):
            transfer_function = torch.conj(torch.exp(1j*Kz*self.distance))
        else:
            transfer_function = torch.exp(1j*Kz*self.distance)

        field = torch.fft.ifft2(field_fft * transfer_function)

        if self.pad_do:
            return field[self.pad_by:-1*self.pad_by, self.pad_by:-1*self.pad_by]

        return field

    def set_direction(self, direction):
        self.direction = direction

        return self


class VolumePropagator(nn.Module):
    """
        Propagates a 2D complex wavefield using the beam propagationm method. Base units is meters.

        Args:
            field (Tensor): Input 2D complex wavefield in the (x, y, 0) plane
            RI_distribution (Tensor): 3D refractive index distribution (x, y, z)
            RI_background (float): background refractive index
            wavelength (float): wavelength of source in vacuum
            spatial_resolution (Tuple[float, float, float]): (dx, dy, dz). dz is the propagation step or slice thickness
            padding (Optional[int], optional): Number of pixels added to the field. Defaults to None.

        Returns:
            NDArray[np.complex64]: 2D complex wavefield at (x, y, -1)
    """

    def __init__(
        self, 
        wavelength: float, 
        spacing: Tuple[float, float, float],
        RI_background: float,
        padding: Optional[int] = None, 
        pad_mode: str = 'edge'
    ) -> Tensor:
        super().__init__()

        if padding is not None:
            self.pad_do = True
            self.pad_by = padding
            self.PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode) # edge make sense
        else:
            self.pad_do = False

        self.wavelength = wavelength
        self.dx, self.dy, self.dz = spacing
        self.n0 = RI_background
        self.source_field = None

    def forward(self, RI_distribution: Tensor) -> Tensor:

        Nx, Ny, Nz = RI_distribution.shape
        device = RI_distribution.device

        # Spatial frequency grid
        k0 = 2 * torch.pi / self.wavelength
        kx = torch.fft.fftfreq(Nx, self.dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, self.dy, device=device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        Kz = torch.sqrt(0j + (self.n0*k0)**2 - Kx**2 - Ky**2)
        transfer_function = torch.exp(1j*Kz*self.dz)

        if self.source_field is not None:
            assert self.source_field.shape == (Nx, Ny), f'source field must have dims {Nx}x{Ny}'
            self.source_field = self.source_field.to(device)
        else:
            self.source_field = torch.ones([Nx, Ny], dtype=torch.complex64, device=device) 

        if self.pad_do:
            field = self.PAD(self.source_field.real) + 1j*self.PAD(self.source_field.imag) # edge make sense
        else:
            field = self.source_field.clone()

        # Forward propagation
        for z in range(Nz):
            field_fft = torch.fft.fft2(field)
            phase = torch.exp(1j*k0*(RI_distribution[..., z] - self.n0)*self.dz)

            if self.pad_do:
                phase = self.PAD(phase.real) + 1j*self.PAD(phase.imag) # no delay in the padded region

            field = torch.fft.ifft2(field_fft * transfer_function) * phase

        if self.pad_do:
            return field[self.pad_by:-1*self.pad_by, self.pad_by:-1*self.pad_by]

        return field

    def set_source_field(self, field: Tensor):
        self.source_field = field # field at input plane of the volume z = 0

        return self

    def set_RI_background(self, RI_background: float):
        self.n0 = RI_background

        return self
