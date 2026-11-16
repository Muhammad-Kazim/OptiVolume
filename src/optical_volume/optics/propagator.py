import torch
from torch import nn, Tensor
import torchvision
from typing import Optional, Tuple


def _make_pad(padding: Optional[int], pad_mode: str):
    """Return a lateral pad transform, or ``None`` when ``padding`` is ``None`` or 0."""
    if not padding:
        return None
    return torchvision.transforms.Pad(int(padding), padding_mode=pad_mode)


def _pad_complex(pad, field: Tensor) -> Tensor:
    # torchvision pads real tensors only, so pad real and imaginary parts separately.
    return pad(field.real) + 1j * pad(field.imag)


def _crop(field: Tensor, p: int) -> Tensor:
    """Remove ``p`` pixels from each side of the last two dimensions."""
    H, W = field.shape[-2:]
    return field[..., p:H - p, p:W - p]


class FreeSpacePropagator(nn.Module):
    """Angular-spectrum propagation through a homogeneous medium.

    Parameters
    ----------
    wavelength : float
        Wavelength in the medium, in metres (vacuum wavelength / refractive index).
    spacing : tuple of float
        Lateral sampling ``(dx, dy)`` of the field, in metres.
    distance : float
        Propagation distance between the parallel planes, in metres. Stored as a parameter.
    direction : {'forward', 'backward'}
        Propagation direction.
    padding : int, optional
        Pixels added on each side before propagation and removed afterwards. ``None`` or 0
        means no padding.
    pad_mode : str
        torchvision padding mode, ``'edge'`` by default.

    Notes
    -----
    ``forward`` takes a complex field of shape ``(..., H, W)`` and returns the propagated
    field with the same shape.
    """

    def __init__(self,
              wavelength: float,
              spacing: Tuple[float, float],
              distance: float,
              direction: Optional[str] = 'forward',
              padding: Optional[int] = None,
              pad_mode: str = 'edge',
    ) -> None:
        super().__init__()

        self.PAD = _make_pad(padding, pad_mode)
        self.pad_by = int(padding) if self.PAD is not None else 0

        self.wavelength = wavelength
        self.dx, self.dy = spacing
        self.direction = direction

        self.distance = nn.Parameter(torch.tensor(distance))

    def forward(self, field: Tensor) -> Tensor:

        device = field.device

        if self.PAD is not None:
            field = _pad_complex(self.PAD, field)

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

        if self.PAD is not None:
            return _crop(field, self.pad_by)

        return field

    def set_direction(self, direction):
        self.direction = direction

        return self


class VolumePropagator(nn.Module):
    """Beam propagation method (BPM) through a refractive-index volume.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength of the source, in metres.
    spacing : tuple of float
        ``(dx, dy, dz)`` in metres. ``dz`` is the slice thickness (propagation step).
    RI_background : float
        Refractive index of the surrounding medium (dimensionless).
    padding : int, optional
        Pixels added on each side of the field and of every phase slice; removed from the
        output. ``None`` or 0 means no padding.
    pad_mode : str
        torchvision padding mode, ``'edge'`` by default.

    Notes
    -----
    ``forward(RI_distribution, source=None)`` takes a volume of shape ``(..., Nx, Ny, Nz)``
    and returns the complex exit field ``(..., Nx, Ny)``. The input field is, in order of
    precedence: ``source`` passed to ``forward``, the field stored with
    :meth:`set_source_field` (a registered buffer, so ``.to()`` moves it), or a unit plane
    wave built for the size of each volume.
    """

    def __init__(
        self,
        wavelength: float,
        spacing: Tuple[float, float, float],
        RI_background: float,
        padding: Optional[int] = None,
        pad_mode: str = 'edge'
    ) -> None:
        super().__init__()

        self.PAD = _make_pad(padding, pad_mode)
        self.pad_by = int(padding) if self.PAD is not None else 0

        self.wavelength = wavelength
        self.dx, self.dy, self.dz = spacing
        self.n0 = RI_background
        self.register_buffer('source_field', None)

    def forward(self, RI_distribution: Tensor, source: Optional[Tensor] = None) -> Tensor:

        Nx, Ny, Nz = RI_distribution.shape[-3:]
        device = RI_distribution.device

        if source is None:
            source = self.source_field
        if source is None:
            # Not cached: the next volume may have a different size.
            source = torch.ones([Nx, Ny], dtype=torch.complex64, device=device)
        elif tuple(source.shape[-2:]) != (Nx, Ny):
            raise ValueError(f'source field has lateral shape {tuple(source.shape[-2:])}, '
                             f'but the volume is {Nx}x{Ny}')
        source = source.to(device)

        if self.PAD is not None:
            field = _pad_complex(self.PAD, source) # edge make sense
        else:
            field = source.clone()

        # Spatial frequency grid, built for the padded field
        H, W = field.shape[-2:]
        k0 = 2 * torch.pi / self.wavelength
        kx = torch.fft.fftfreq(H, self.dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(W, self.dy, device=device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        Kz = torch.sqrt(0j + (self.n0*k0)**2 - Kx**2 - Ky**2)
        transfer_function = torch.exp(1j*Kz*self.dz)

        # Forward propagation
        for z in range(Nz):
            field_fft = torch.fft.fft2(field)
            phase = torch.exp(1j*k0*(RI_distribution[..., z] - self.n0)*self.dz)

            if self.PAD is not None:
                phase = _pad_complex(self.PAD, phase) # no delay in the padded region

            field = torch.fft.ifft2(field_fft * transfer_function) * phase

        if self.PAD is not None:
            return _crop(field, self.pad_by)

        return field

    def set_source_field(self, field: Optional[Tensor]):
        """Store the field at the input plane ``z = 0`` as a buffer (``None`` clears it).

        For a trainable source, pass it to ``forward(ri, source=...)`` instead.
        """
        if isinstance(field, nn.Parameter):
            raise TypeError('set_source_field stores a buffer; pass a trainable source '
                            'to forward(ri, source=...) instead')
        self.source_field = field # field at input plane of the volume z = 0

        return self

    def set_RI_background(self, RI_background: float):
        self.n0 = RI_background

        return self
