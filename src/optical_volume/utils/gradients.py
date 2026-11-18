"""Finite-difference gradients and gradient-based regularisers of images."""
import torch
import torchvision
from torch import Tensor


def torch_grad_optr(image, mode='edge'):

    d0 = image[:-1, :] - image[1:, :] # rows or y-axis
    d1 = image[:, :-1] - image[:, 1:] # columns or x-axis
    
    d0 = torchvision.transforms.Pad((0, 0, 0, 1), padding_mode=mode)(d0)
    d1 = torchvision.transforms.Pad((0, 0, 1, 0), padding_mode=mode)(d1)

    return [d0, d1]


def torch_TV(image: Tensor, eps: float = 1e-6):
    del0, del1 = torch_grad_optr(image)
    norm = torch.sqrt(del0**2 + del1**2 + eps)
    
    return norm.sum()/(image.view(-1).size()[0])


def torch_L2_grad(image: Tensor):
    del0, del1 = torch_grad_optr(image)
    norm = del0**2 + del1**2
    
    return norm.sum()/(image.view(-1).size()[0])
