import pickle
import os
import sys
import numpy as np

import torch
import torchvision
import torchvision.transforms.functional as F
from torch import Tensor


def load_pkl(filename):
    """Reads in a geometry pickled object.

    Args:
        filename (str): path/to/file.pkl

    Returns:
        geometry: geometry object
    """
    
    if os.path.isfile(filename):
        print('Loading geometry object...')
        with open(filename, 'rb') as inp:
            geom = pickle.load(inp)
        
        return geom
    else:
        print('File does not exist.')
        

def normalization(field, totype='int16'):
    """normalizes field to 0-1.

    Args:
        field (float): 2d floating point arrays.
        totype (str): 'int16' or 'int8'
    """
    
    field = (field - field.min())/(field.max() - field.min())
    
    if totype == 'int16':
        return np.array(field*(2**16) - 1, dtype=np.uint16)
    elif totype == 'int8':
        return np.array(field*(2**8) - 1, dtype=np.uint8)
    else:
        print('Wrong totype')
        

def plot(imgs, **imshow_kwargs):
    from matplotlib import pyplot as plt  # optional dependency: pip install optical-volume[plot]

    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            img = F.to_pil_image(img.to("cpu"))
            ax.imshow(np.asarray(img), **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()


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


def torch_grad_optr(image, mode='edge'):

    d0 = image[:-1, :] - image[1:, :] # rows or y-axis
    d1 = image[:, :-1] - image[:, 1:] # columns or x-axis
    
    d0 = torchvision.transforms.Pad((0, 0, 0, 1), padding_mode=mode)(d0)
    d1 = torchvision.transforms.Pad((0, 0, 1, 0), padding_mode=mode)(d1)

    return [d0, d1]


def median_filter_2d(input_tensor: Tensor, kernel_size: int = 3) -> Tensor:
        """
        Apply median filtering to a BxCxHxW tensor (C channels per image).

        Args:
            input_tensor: torch.Tensor of shape (B, C, H, W)
            kernel_size: int, odd filter size

        Returns:
            torch.Tensor of shape (B, C, H, W)
        """
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        B, C, H, W = input_tensor.shape

        pad = kernel_size // 2

        # Apply unfold to each channel separately
        filtered_channels = []
        for c in range(C):
            channel = input_tensor[:, c:c+1, :, :]  # shape: (B, 1, H, W)
            patches = torch.nn.functional.unfold(channel, kernel_size=kernel_size, padding=pad)  # shape: (B, K*K, H*W)
            median = patches.median(dim=1)[0]  # shape: (B, H*W)
            median = median.view(B, 1, H, W)   # shape: (B, 1, H, W)
            filtered_channels.append(median)

        # Concatenate filtered channels back along dim=1
        return torch.cat(filtered_channels, dim=1)  # shape: (B, C, H, W)


def auto_corr_fn(image: Tensor):
    return torch.fft.fftshift(torch.fft.ifft2(torch.fft.fft2(image).abs()**2).real)


def torch_TV(image: Tensor, eps: float = 1e-6):
    del0, del1 = torch_grad_optr(image)
    norm = torch.sqrt(del0**2 + del1**2 + eps)
    
    return norm.sum()/(image.view(-1).size()[0])


def torch_L2_grad(image: Tensor):
    del0, del1 = torch_grad_optr(image)
    norm = del0**2 + del1**2
    
    return norm.sum()/(image.view(-1).size()[0])


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def normalized_cross_corr(x, y, eps=1e-12):
    """Normalized Cross Correlation"""
    x_centered = x - torch.mean(x)
    y_centered = y - torch.mean(y)

    numerator = torch.sum(x_centered * y_centered)

    x_variance = torch.sqrt(torch.sum(x_centered ** 2) + eps)
    y_variance = torch.sqrt(torch.sum(y_centered ** 2) + eps)

    denominator = x_variance * y_variance

    ncc = numerator / denominator
    return ncc
