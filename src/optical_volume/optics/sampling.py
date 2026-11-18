"""Sampling: grid bookkeeping, physical magnification and numerical resampling."""
import warnings

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torchvision
from torchvision.transforms.functional import resize


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


class Magnification(nn.Module):
    """Ideal (telecentric) magnification by ``M``: the field pattern is stretched, not resampled.

    The array is unchanged; each sample now covers an ``M`` times larger patch, so the spacing
    becomes ``spacing * M`` (see :meth:`output_spacing`). Energy conservation over the
    ``M**2`` larger area scales the amplitude by ``1/M`` (intensity by ``1/M**2``). Image
    inversion is ignored.

    Parameters
    ----------
    magnification : float
        Lateral magnification ``M`` (dimensionless, > 0).
    """

    def __init__(self, magnification: float):
        super().__init__()
        self.magnification = float(magnification)
        if self.magnification <= 0:
            raise ValueError(f"magnification must be positive, got {self.magnification}")

    def forward(self, field: Tensor) -> Tensor:
        return field / self.magnification

    def output_spacing(self, spacing):
        """Spacing ``spacing * M`` in metres, per axis."""
        return tuple(d * self.magnification for d in _pair(spacing))


class Resample(nn.Module):
    """Change the number of samples covering the same physical area.

    The output has ``round(N * factor)`` samples per axis and spacing ``spacing * N / N_out``.
    Values are interpolated, not rescaled: ``|E|**2`` is a density.

    Parameters
    ----------
    factor : float or tuple of float
        Sampling factor per axis (> 1 upsamples). May be non-integer.
    method : {'fourier', 'bilinear'}
        ``'fourier'`` (default): zero-pads or crops the spectrum, exact for band-limited
        periodic fields such as the output of a pupil filter. ``'bilinear'``: torchvision
        bilinear interpolation (antialiased when downsampling); smooths slightly.
    """

    def __init__(self, factor, method: str = "fourier"):
        super().__init__()
        self.factor = _pair(factor)
        if method not in ("bilinear", "fourier"):
            raise ValueError(f"method must be 'bilinear' or 'fourier', got {method!r}")
        self.method = method

    def _out_shape(self, shape):
        return tuple(max(1, round(n * f)) for n, f in zip(shape, self.factor))

    def forward(self, field: Tensor) -> Tensor:
        H, W = field.shape[-2:]
        Ho, Wo = self._out_shape((H, W))
        if self.method == "bilinear":
            def interp(x):
                # torchvision wants (C, H, W): fold any leading dimensions into C.
                y = resize(x.reshape(-1, H, W), [Ho, Wo], torchvision.transforms.InterpolationMode.BILINEAR,
                           antialias=True)
                return y.reshape(*x.shape[:-2], Ho, Wo)
            return torch.complex(interp(field.real), interp(field.imag)) if field.is_complex() else interp(field)
        # 'forward' normalisation keeps sample values when the number of samples changes.
        spec = torch.fft.fftshift(torch.fft.fft2(field, norm="forward"), dim=(-2, -1))
        spec = _center_fit(spec, Ho, Wo)
        out = torch.fft.ifft2(torch.fft.ifftshift(spec, dim=(-2, -1)), norm="forward")
        return out if field.is_complex() else out.real

    def output_spacing(self, spacing, shape=None):
        """Output spacing in metres. Exact with ``shape`` (rounding of the sample count)."""
        if shape is None:
            return tuple(d / f for d, f in zip(_pair(spacing), self.factor))
        out = self._out_shape(shape[-2:])
        return tuple(d * n / m for d, n, m in zip(_pair(spacing), shape[-2:], out))


def _center_fit(spec: Tensor, Ho: int, Wo: int) -> Tensor:
    """Zero-pad or crop a centred spectrum ``(..., H, W)`` to ``(..., Ho, Wo)``."""
    H, W = spec.shape[-2:]
    out = spec.new_zeros(*spec.shape[:-2], Ho, Wo)
    h, w = min(H, Ho), min(W, Wo)
    # Keep the h x w block around the zero frequency, which fftshift puts at index n // 2.
    si, sj = H // 2 - h // 2, W // 2 - w // 2
    di, dj = Ho // 2 - h // 2, Wo // 2 - w // 2
    out[..., di:di + h, dj:dj + w] = spec[..., si:si + h, sj:sj + w]
    return out


def _pair(value):
    """``x`` -> ``(x, x)``; a pair stays a pair (as floats)."""
    if isinstance(value, (tuple, list)):
        return tuple(float(v) for v in value[:2])
    return (float(value), float(value))
