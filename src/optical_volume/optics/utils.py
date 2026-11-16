import warnings

import torch
from torch import nn, Tensor
import torchvision
from torchvision.transforms.functional import resize
from torchvision.transforms import GaussianBlur


def samples_per(length: float, spacing: float, what: str = "length") -> int:
    """Number of grid samples spanned by ``length``, rounded to the nearest integer.

    Parameters
    ----------
    length : float
        Physical length in metres (a tile, a pixel, ...).
    spacing : float
        Grid spacing in metres.
    what : str
        Name used in the warning.

    Warns
    -----
    UserWarning
        If ``length / spacing`` is not within 1e-6 of a whole number.
    """
    ratio = length / spacing
    n = max(1, round(ratio))
    if abs(ratio - n) > 1e-6 * max(ratio, 1.0):
        warnings.warn(f"{what} {length:g} m is {ratio:.4f} samples of {spacing:g} m; using {n}",
                      stacklevel=2)
    return n


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

def _pair(value):
    """``x`` -> ``(x, x)``; a pair stays a pair (as floats)."""
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value[:2])
    return (float(value), float(value))


def _box_weights(n_in: int, factor: float, offset: Tensor, n_out: int) -> Tensor:
    """``(n_out, n_in)`` overlap of output pixel ``j`` (``[j*f + o, (j+1)*f + o]``) with input
    sample ``i`` (``[i, i+1]``), in units of input samples. Piecewise linear in ``offset``."""
    i = torch.arange(n_in, dtype=offset.dtype, device=offset.device)
    j = torch.arange(n_out, dtype=offset.dtype, device=offset.device)[:, None]
    left = torch.maximum(i, j * factor + offset)
    right = torch.minimum(i + 1, (j + 1) * factor + offset)
    return torch.clamp(right - left, min=0)


class Intensity(nn.Module):
    """Detected intensity ``|E|**2`` of a complex field ``(..., H, W)``."""

    def forward(self, field: Tensor) -> Tensor:
        return field.abs() ** 2


class PixelIntegrator(nn.Module):
    """Sum (or average) the input over detector pixels larger than the grid spacing.

    Works on complex fields and on real intensities, shape ``(..., H, W)``. Put
    :class:`Intensity` before it to integrate intensity (a physical camera), or after it to
    integrate the field first.

    Parameters
    ----------
    factor : float or tuple of float, optional
        Pixel size in units of input samples, per axis. May be non-integer.
    pixel_size : float or tuple of float, optional
        Pixel size in metres; with ``spacing``, an alternative to ``factor``.
    spacing : float or tuple of float, optional
        Input sample spacing in metres. Needed with ``pixel_size`` or ``offset``.
    offset : tuple of float, optional
        Shift of the pixel grid in metres (a trainable parameter). Default: no shift.
    reduce : {'sum', 'mean'}
        Sum over each pixel (default), or average.

    Notes
    -----
    The output has ``floor(N / factor)`` pixels per axis. Integer factors without an offset
    use an exact block sum. Otherwise each input sample is weighted by its overlap with the
    output pixel (box integration), which conserves the total sum; with an offset, samples
    beyond the field count as zero.
    """

    def __init__(self, factor=None, *, pixel_size=None, spacing=None, offset=None, reduce: str = "sum"):
        super().__init__()
        if (factor is None) == (pixel_size is None):
            raise ValueError("give either factor or pixel_size (with spacing)")
        if pixel_size is not None:
            if spacing is None:
                raise ValueError("pixel_size needs spacing")
            factor = tuple(p / d for p, d in zip(_pair(pixel_size), _pair(spacing)))
        # Snap ratios like 2.9999999999 to the integer they are meant to be.
        self.factor = tuple(float(round(f)) if abs(f - round(f)) < 1e-6 * f else f
                            for f in _pair(factor))
        if reduce not in ("sum", "mean"):
            raise ValueError(f"reduce must be 'sum' or 'mean', got {reduce!r}")
        self.reduce = reduce
        self.spacing = None if spacing is None else _pair(spacing)
        if offset is None:
            self.offset = None
        else:
            if self.spacing is None:
                raise ValueError("offset (in metres) needs spacing")
            self.offset = nn.Parameter(torch.as_tensor(offset, dtype=torch.float32))

    def _integer(self) -> bool:
        return self.offset is None and all(f == int(f) for f in self.factor)

    def forward(self, field: Tensor) -> Tensor:
        H, W = field.shape[-2:]
        fx, fy = self.factor
        Ho, Wo = int(H / fx + 1e-9), int(W / fy + 1e-9)
        if self._integer():
            fx, fy = int(fx), int(fy)
            blocks = field[..., :Ho * fx, :Wo * fy].reshape(*field.shape[:-2], Ho, fx, Wo, fy)
            out = blocks.sum(dim=(-3, -1))
        else:
            real = field.real.dtype if field.is_complex() else field.dtype
            if self.offset is None:
                ox = oy = torch.zeros((), dtype=real, device=field.device)
            else:
                ox, oy = (self.offset.to(real) / torch.tensor(self.spacing, dtype=real, device=field.device))
            wx = _box_weights(H, fx, ox, Ho).to(field.dtype)
            wy = _box_weights(W, fy, oy, Wo).to(field.dtype)
            out = wx @ field @ wy.transpose(0, 1)
        if self.reduce == "mean":
            out = out / (self.factor[0] * self.factor[1])
        return out

    def output_spacing(self, spacing):
        """Pixel pitch ``spacing * factor`` (metres), per axis."""
        return tuple(d * f for d, f in zip(_pair(spacing), self.factor))


class GaussBlur(nn.Module):
    def __init__(self, const_sigma: float = 1e-2, defocus_sigma: float = 10e-2, defocus_dist:float = 0.):
        super().__init__()

        sigma = const_sigma + defocus_sigma*torch.abs(torch.tensor(defocus_dist))
        kernel_size = 2*int(4.*sigma + 0.5) + 1
        self.blur = GaussianBlur(kernel_size, sigma=sigma.item())

    def forward(self, field: Tensor):
        return self.blur(field.real.unsqueeze(0)).squeeze() + 1j*self.blur(field.imag.unsqueeze(0)).squeeze()
