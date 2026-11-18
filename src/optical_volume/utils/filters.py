"""Image filters and correlations."""
import torch
from torch import Tensor


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
