import pickle
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d as convolve
from scipy.fftpack import dct, idct
import csv
from tifffile import tifffile
import json

from typing import Optional, Tuple, List
from torch import nn, Tensor

import torch
import torchvision
import torchvision.transforms.functional as F
from torch import nn, Tensor
from torch.utils.data import Dataset, DataLoader

from typing import Optional, Tuple, List

from scipy.sparse import spdiags
from scipy.sparse.linalg import spsolve


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
        

def low_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
    fmax = NA/wl
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftfreq(wavefield.shape[0], dx)
    ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K > fmax] = 0.
    
    wave_spectrum = np.fft.fft2(wavefield)*mask
    
    return np.fft.ifft2(wave_spectrum)


def high_pass_filter_NA(wavefield, wl, spatial_resolution, NA):
    fmax = NA/wl
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftfreq(wavefield.shape[0], dx)
    ky = np.fft.fftfreq(wavefield.shape[1], dy)
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K < fmax] = 0.
    
    wave_spectrum = np.fft.fft2(wavefield)*mask
    
    return np.fft.ifft2(wave_spectrum)


def band_pass_filter_NA(wavefield, wl, spatial_resolution, NA, loc):
    # ideally loc in space, means loc/(wl*focal_length) in the spectreum
    # but here loc = f_c
    
    fmax = NA/wl
    dx, dy = spatial_resolution[:2]
    
    kx = np.fft.fftshift(np.fft.fftfreq(wavefield.shape[0], dx))
    ky = np.fft.fftshift(np.fft.fftfreq(wavefield.shape[1], dy))
    
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    
    K = np.sqrt(Kx**2 + Ky**2)
    mask = np.ones_like(K)
    mask[K > fmax] = 0.
    
    wave_spectrum = np.fft.fftshift(np.fft.fft2(wavefield))*np.roll(mask, (loc[0], loc[1]), axis=(1, 0))
    
    return np.fft.ifft2(np.fft.ifftshift(wave_spectrum))


def plot(imgs, **imshow_kwargs):
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


def fit_quadratic(img: Tensor):
    
    nx, ny = img.shape
    
    X, Y = torch.meshgrid(torch.arange(nx), torch.arange(ny), indexing='ij')
    A = torch.hstack([X.reshape(-1, 1)**2, Y.reshape(-1, 1)**2, X.reshape(-1, 1)*Y.reshape(-1, 1), X.reshape(-1, 1), Y.reshape(-1, 1), torch.ones([nx, ny]).reshape(-1, 1)])

    pinv = torch.linalg.inv(A.T @ A) @ A.T
    abc = pinv @ img.reshape(-1, 1)
    
    return (abc, abc[0]*X**2 + abc[0]*Y**2 + abc[0]*X*Y + abc[0]*X + abc[1]*Y + abc[2])


class ObjImgMap():
    def __init__(self, wavelength: float, spatial_resolution: Tuple[float, float], numPx: Tuple[int, int]):
        """maps the wavefield in object focus plane to image plane by modulating with the pupil

        Args:
            wavelength (float): wavelength
            spatial_resolution (Tuple[float, float]): lateral resolution in object space (dx, dy)
            numPx (Tuple[float, float]): (Nx, Ny) in object space
        """

        self.wl = wavelength
        self.dx, self.dy = spatial_resolution[:2]
        self.nx, self.ny = numPx[:2]
                
        kx = torch.fft.fftshift(torch.fft.fftfreq(self.nx, self.dx))
        ky = torch.fft.fftshift(torch.fft.fftfreq(self.ny, self.dy))
        self.Kx, self.Ky = torch.meshgrid(kx, ky, indexing='ij')
        
        # aperture does nothing
        self.pupil_amp = torch.ones([self.nx, self.ny])
        self.pupil_phase = torch.zeros([self.nx, self.ny])
        
    def forward(self, wavefield: Tensor):
        
        pupil = self.pupil_amp*torch.exp(1j*2*torch.pi/self.wl*self.pupil_phase)
        wave_spectrum = torch.fft.fftshift(torch.fft.fft2(wavefield))*pupil
        
        return torch.fft.ifft2(torch.fft.ifftshift(wave_spectrum))
    
    def displaced_unifrom_pupil_amp(self, pupil_center = [0., 0.], pupil_radius=1e8, softness=1e-12):
            
        K = (self.Kx - pupil_center[0])**2 + (self.Ky - pupil_center[1])**2
        self.pupil_amp = 1 - torch.sigmoid((K - pupil_radius**2) / softness)
        
    def low_pass_filter(self, NA: float, softness=1e-12):
        fmax = NA/self.wl
        self.displaced_unifrom_pupil_amp(pupil_center = [0., 0.], pupil_radius=fmax, softness=softness)
        
    def set_pupil_amp(self, amp):
        assert amp.size() == (self.nx, self.ny), f"Aperture amplitude must have sizes {self.nx}x{self.ny}"
        self.pupil_amp = amp
        
    def set_pupil_phase(self, phase):
        assert phase.size() == (self.nx, self.ny), f"Aperture phase must have sizes {self.nx}x{self.ny}"
        self.pupil_phase = phase
        

def low_pass_filter(wavelength, spatial_resolution, numPx, NA):
    lpf = ObjImgMap(wavelength, spatial_resolution, numPx)
    return lpf.low_pass_filter(NA)


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


if __name__=='__main__':
    pass
