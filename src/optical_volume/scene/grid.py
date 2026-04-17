import torch
from typing import Optional, Tuple, List

class Grid:
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], device: str='cpu'):
        self.nx, self.ny, self.nz = shape
        self.dx, self.dy, self.dz = spacing

        x = torch.arange(self.nx, device=device) * self.dx
        y = torch.arange(self.ny, device=device) * self.dy
        z = torch.arange(self.nz, device=device) * self.dz

        self.X, self.Y, self.Z = torch.meshgrid(x, y, z, indexing="ij")
    
    def to(self, device: str):
        self.device = device
        self.X = self.X.to(device)
        self.Y = self.Y.to(device)
        self.Z = self.Z.to(device)
        return self