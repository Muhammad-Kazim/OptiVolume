"""Least-squares surface fits to images."""
import torch
from torch import Tensor


def fit_linear(img: Tensor):
    
    nx, ny = img.shape
    
    X, Y = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing='ij')
    A = torch.hstack([X.reshape(-1, 1), Y.reshape(-1, 1), torch.ones([nx, ny]).reshape(-1, 1)])

    pinv = torch.linalg.inv(A.T @ A) @ A.T
    abc = pinv @ img.reshape(-1, 1)
    
    return (abc, abc[0]*X + abc[1]*Y + abc[2])


def fit_quadratic(img: Tensor, spacing=None):
    """Least-squares fit of a quadratic surface to a 2D image.

    Fits ``a x^2 + b y^2 + c x y + d x + e y + f`` to ``img[i, j]`` at ``x = i * dx``,
    ``y = j * dy`` (arrays are indexed ``[x, y]``).

    Parameters
    ----------
    img : Tensor
        ``(nx, ny)`` real image.
    spacing : float or tuple of float, optional
        Sample spacing ``(dx, dy)`` in metres. Default: coordinates in samples (``dx = dy = 1``).

    Returns
    -------
    coeffs : Tensor
        ``(6,)`` coefficients ``(a, b, c, d, e, f)``, in units of the image per (metre or
        sample) to the power of the term's degree.
    surface : Tensor
        ``(nx, ny)`` fitted surface.
    """
    nx, ny = img.shape
    dtype = img.dtype if img.is_floating_point() else torch.float32
    dx, dy = (1.0, 1.0) if spacing is None else ((float(spacing),) * 2 if isinstance(spacing, (int, float)) else tuple(map(float, spacing)))
    x = torch.arange(nx, dtype=dtype, device=img.device) * dx
    y = torch.arange(ny, dtype=dtype, device=img.device) * dy
    X, Y = torch.meshgrid(x, y, indexing='ij')
    basis = torch.stack([X**2, Y**2, X * Y, X, Y, torch.ones_like(X)], dim=-1)    # (nx, ny, 6)
    # lstsq solves the normal equations without forming A^T A, which would square the condition number.
    coeffs = torch.linalg.lstsq(basis.reshape(-1, 6), img.reshape(-1, 1).to(dtype)).solution.reshape(6)
    return coeffs, basis @ coeffs
