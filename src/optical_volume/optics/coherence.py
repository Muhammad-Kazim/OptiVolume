"""Partial spatial coherence."""
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from .sampling import _pair


class PartialCoherenceBlur(nn.Module):
    """Gaussian blur modelling partial spatial coherence: ``sigma = sigma0 + theta * |defocus|``.

    Light from an extended source arrives over an angular spread ``theta``; at a distance
    ``z`` from the plane in focus each direction is shifted by about ``theta * z``, which
    smears the pattern. The element blurs whatever it receives, and where it sits is the
    user's modelling choice:

    - **Field, object side** (as in example 07): blur of the object's defocused projection,
      ``sigma = theta * |z_obj|``. Averaging complex values approximates averaging the phase,
      so this holds for smooth, weak phase objects; strong phase changes within ``sigma``
      partly cancel and darken.
    - **Intensity, image side** (after :class:`Intensity`): blur of a pattern formed a
      distance ``d`` behind a mask, ``sigma = (theta / M) * d`` for magnification ``M``.

    Parameters
    ----------
    spacing : float or tuple of float
        Sampling ``(dx, dy)`` of the input, in metres.
    sigma0 : float
        Constant blur width (standard deviation) in metres (parameter).
    angular_spread : float
        Illumination angular spread ``theta`` in radians (parameter).
    defocus : float
        Distance from the plane in focus, in metres (parameter).

    Notes
    -----
    Real and complex inputs ``(..., H, W)``; complex fields are blurred in their real and
    imaginary parts. Separable kernel, replicate padding, differentiable in all three
    parameters. ``sigma = 0`` returns the input unchanged.
    """

    def __init__(self, spacing, sigma0: float = 0., angular_spread: float = 0., defocus: float = 0.):
        super().__init__()
        self.spacing = _pair(spacing)
        self.sigma0 = nn.Parameter(torch.tensor(float(sigma0)))
        self.angular_spread = nn.Parameter(torch.tensor(float(angular_spread)))
        self.defocus = nn.Parameter(torch.tensor(float(defocus)))

    @property
    def sigma(self) -> Tensor:
        """Blur width (standard deviation) in metres."""
        return self.sigma0 + self.angular_spread * self.defocus.abs()

    def forward(self, field: Tensor) -> Tensor:
        sigma = self.sigma
        if sigma.item() <= 0:
            return field
        if field.is_complex():
            return torch.complex(self._blur(field.real, sigma), self._blur(field.imag, sigma))
        return self._blur(field, sigma)

    def _blur(self, x: Tensor, sigma: Tensor) -> Tensor:
        H, W = x.shape[-2:]
        y = x.reshape(-1, 1, H, W)
        for axis, d in enumerate(self.spacing):
            s = (sigma / d).to(x.dtype)  # in samples, differentiable
            radius = int(4 * s.item() + 0.5)
            if radius == 0:
                continue
            c = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
            k = torch.exp(-c ** 2 / (2 * s ** 2))
            k = k / k.sum()
            if axis == 0:
                y = F.conv2d(F.pad(y, (0, 0, radius, radius), mode='replicate'), k.view(1, 1, -1, 1))
            else:
                y = F.conv2d(F.pad(y, (radius, radius, 0, 0), mode='replicate'), k.view(1, 1, 1, -1))
        return y.reshape(x.shape)
