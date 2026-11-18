import torch
import torch.nn.functional as F
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


_EVANESCENT_MODES = ("zero", "decay")


def _transfer_function(shape, spacing, k, distance, evanescent: str, device) -> Tensor:
    """Angular-spectrum transfer function ``H(kx, ky)`` for propagation over ``distance``.

    Parameters
    ----------
    shape : tuple of int
        Field size ``(H, W)`` in pixels (after padding).
    spacing : tuple of float
        Sampling ``(dx, dy)`` in metres.
    k : float
        Wavenumber in the medium, ``2*pi*n/wavelength``, in 1/m.
    distance : float or Tensor
        Propagation distance in metres.
    evanescent : {'zero', 'decay'}
        ``'zero'``: components with ``kx**2 + ky**2 >= k**2`` are removed (stable in both
        directions). ``'decay'``: they decay as ``exp(-kappa*d)``, the exact solution for
        ``d >= 0``; for ``d < 0`` they are removed, since growth would be unbounded.

    Returns
    -------
    Tensor
        Complex ``(H, W)`` transfer function, in FFT (unshifted) order.
    """
    H, W = shape
    dx, dy = spacing
    kx = torch.fft.fftfreq(H, dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(W, dy, device=device) * 2 * torch.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

    kz2 = k**2 - Kx**2 - Ky**2
    propagating = kz2 > 0
    kz = torch.sqrt(torch.clamp(kz2, min=0))
    transfer = torch.exp(1j * kz * distance)

    if evanescent == "zero":
        return torch.where(propagating, transfer, torch.zeros_like(transfer))
    kappa = torch.sqrt(torch.clamp(-kz2, min=0))
    decay = torch.where(torch.as_tensor(distance, device=device) >= 0,
                        torch.exp(-kappa * distance), torch.zeros_like(kappa))
    return torch.where(propagating, transfer, decay.to(transfer.dtype))


def _check_evanescent(evanescent: str) -> str:
    if evanescent not in _EVANESCENT_MODES:
        raise ValueError(f"evanescent must be one of {_EVANESCENT_MODES}, got {evanescent!r}")
    return evanescent


class AngularSpectrum(nn.Module):
    """Angular-spectrum propagation through a homogeneous medium.

    Parameters
    ----------
    wavelength : float
        Wavelength in the medium, in metres (vacuum wavelength / refractive index).
    spacing : tuple of float
        Lateral sampling ``(dx, dy)`` of the field, in metres.
    distance : float
        Signed propagation distance in metres (negative propagates backwards). Stored as a
        parameter.
    direction : {'forward', 'backward'}
        ``'backward'`` negates ``distance``.
    padding : int, optional
        Pixels added on each side before propagation and removed afterwards. ``None`` or 0
        means no padding.
    pad_mode : str
        torchvision padding mode, ``'edge'`` by default.
    evanescent : {'zero', 'decay'}
        Treatment of evanescent components (``kx**2 + ky**2 >= k**2``). ``'zero'`` (default)
        removes them, which is stable for both directions.

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
              evanescent: str = 'zero',
    ) -> None:
        super().__init__()

        self.evanescent = _check_evanescent(evanescent)
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

        # Signed distance: H = exp(i kz d) is smooth in d through 0, so focus searches work.
        distance = -self.distance if self.direction == 'backward' else self.distance
        transfer_function = _transfer_function(field.shape[-2:], (self.dx, self.dy),
                                               2 * torch.pi / self.wavelength, distance,
                                               self.evanescent, device)

        field_fft = torch.fft.fft2(field)

        field = torch.fft.ifft2(field_fft * transfer_function)

        if self.PAD is not None:
            return _crop(field, self.pad_by)

        return field

    def set_direction(self, direction):
        self.direction = direction

        return self


class BeamPropagation(nn.Module):
    """Beam propagation method (BPM) through a refractive-index volume.

    Parameters
    ----------
    wavelength : float
        Vacuum wavelength of the source, in metres.
    spacing : tuple of float
        ``(dx, dy, dz)`` in metres. ``dz`` is the slice thickness (propagation step).
    n_medium : float
        Refractive index of the surrounding medium (dimensionless).
    padding : int, optional
        Pixels added on each side of the field (with ``pad_mode``) and of every slice (with the
        background index ``n_medium``); removed from the output. ``None`` or 0 means no
        padding.
    pad_mode : str
        torchvision padding mode, ``'edge'`` by default.
    evanescent : {'zero', 'decay'}
        Treatment of evanescent components. ``'decay'`` (default) applies ``exp(-kappa*dz)``
        per slice, which matters because ``dz`` is shorter than the wavelength.

    Notes
    -----
    ``forward(n, source=None)`` takes a volume of shape ``(..., Nx, Ny, Nz)``
    and returns the complex exit field ``(..., Nx, Ny)``. The input field is, in order of
    precedence: ``source`` passed to ``forward``, the field stored with
    :meth:`set_source_field` (a registered buffer, so ``.to()`` moves it), or a unit plane
    wave built for the size of each volume.
    """

    def __init__(
        self,
        wavelength: float,
        spacing: Tuple[float, float, float],
        n_medium: float,
        padding: Optional[int] = None,
        pad_mode: str = 'edge',
        evanescent: str = 'decay',
    ) -> None:
        super().__init__()

        self.evanescent = _check_evanescent(evanescent)
        self.PAD = _make_pad(padding, pad_mode)
        self.pad_by = int(padding) if self.PAD is not None else 0

        self.wavelength = wavelength
        self.dx, self.dy, self.dz = spacing
        self.n_medium = n_medium
        self.register_buffer('source_field', None)

    def forward(self, n: Tensor, source: Optional[Tensor] = None) -> Tensor:

        Nx, Ny, Nz = n.shape[-3:]
        device = n.device

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

        # Transfer function for one slice, built for the padded field
        k0 = 2 * torch.pi / self.wavelength
        transfer_function = _transfer_function(field.shape[-2:], (self.dx, self.dy), self.n_medium * k0,
                                               self.dz, self.evanescent, device)

        # Forward propagation
        for z in range(Nz):
            field_fft = torch.fft.fft2(field)
            delta_n = n[..., z] - self.n_medium
            if self.PAD is not None:
                # The pad is background medium (delta_n = 0, no extra delay), not a copy of the edge.
                delta_n = F.pad(delta_n, (self.pad_by,) * 4)
            phase = torch.exp(1j*k0*delta_n*self.dz)

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

    def set_n_medium(self, n_medium: float):
        self.n_medium = n_medium

        return self
