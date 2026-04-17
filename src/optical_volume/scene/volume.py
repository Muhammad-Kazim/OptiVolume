import torch
from torch import nn, Tensor
from typing import Optional, Tuple, List


class Volume(nn.Module):
    def __init__(self, grid: Tensor, n_bg: Tensor):
        super().__init__()
        
        self.grid = grid
        self.n_bg = n_bg
        self.shapes = nn.ModuleList()
        # self.field = nn.Parameter(torch.ones_like(self.grid.X) * self.n_bg)
        
    def add(self, shape: List):
        self.shapes.extend(shape)

    def forward(self):
        field = torch.ones_like(self.grid.X) * self.n_bg
        for shape in self.shapes:
            mask = shape(self.grid)
            field = field * (1 - mask) + shape.RI * mask

        return field
        
    def to(self, device: str):
        super().to(device)
        if self.grid.X.device != torch.device(device):
            self.grid = self.grid.to(device)
        self.n_bg = self.n_bg.to(device)
        
        return self

if __name__ == '__main__':
    from grid import Grid
    from shapes import *
    from matplotlib import pyplot as plt
    
    from torch.optim import Adam
    from torch import nn
    
    
    # example 1: create two spheres and visulization the RI volume
    sphere1 = Sphere(torch.tensor((55, 50, 40)).float(), torch.tensor((10.)), torch.tensor(1.3), softness=1e-2)
    sphere2 = Sphere(torch.tensor((50, 50, 40)).float(), torch.tensor((10.)), torch.tensor(1.3), softness=1e-2)
    
    grid = Grid((100, 100, 100), (1, 1, 1))
    
    vol = Volume(grid, n_bg=torch.tensor(1.33))
    vol.add([sphere1, sphere2])
        
    ri_dist = vol.forward()
    
    print(f'RI Volume: {ri_dist.dtype}, {ri_dist.requires_grad}, {ri_dist.shape}')
    plt.imshow(ri_dist.detach()[:, :, 40])
    plt.colorbar()
    plt.show()
    
    # example 2: Create two volumes and optimize by updating the shape center
    vol1 = Volume(grid, n_bg=torch.tensor(1.33))
    vol2 = Volume(grid, n_bg=torch.tensor(1.33))
    
    vol1.add([sphere1])
    vol2.add([sphere2])
    
    # optimizer = Adam(vol1.parameters(), lr=1e-3)
    optimizer = Adam([sphere1.center], lr=1e-2)
    loss_fn = nn.MSELoss()
    
    centers = []
    for i in range(10):
        ri_dist = vol1.forward()
        loss = loss_fn(ri_dist, vol2.forward())
        print(loss.item())
        
        loss.backward()
        optimizer.step()
        
        # print(vol1.shapes[0].center, vol1.shapes[0].radius, vol1.shapes[0].RI)
        # print(vol2.shapes[0].center, vol2.shapes[0].radius, vol2.shapes[0].RI)
        centers.append(vol1.shapes[0].center[0].item())
        # plt.imshow(ri_dist.detach()[:, :, 40])
        # plt.colorbar()
        # plt.show()
    plt.plot(centers)
    plt.show()
    
    
    # example 3: add multiple shapes
    
    shapes = make_ellipsoids(
        torch.tensor([[55, 50, 40], [75, 50, 40], [30, 50, 40]]).float(),
        torch.tensor([[5, 10, 10], [15, 5, 7], [10, 10, 10]]).float(),
        torch.tensor([3., 1, 2.]).float(), 
        rotation=True
    )
    
    vol = Volume(grid, n_bg=torch.tensor(1.33))
    vol.add(shapes)
    
    ri_dist = vol.forward()
    
    print(f'RI Volume: {ri_dist.dtype}, {ri_dist.requires_grad}, {ri_dist.shape}')
    plt.imshow(ri_dist.detach()[:, :, 40])
    plt.colorbar()
    plt.show()