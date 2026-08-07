import torch
import torch.nn.functional as F

from torch import nn, Tensor
from typing import Optional, Tuple, List

from .utils import from_axis_angle, to_axis_angle, to_matrix

class Grid(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float]):
        super().__init__()

        self.nx, self.ny, self.nz = shape
        self.dx, self.dy, self.dz = spacing

        x = torch.arange(self.nx) * self.dx
        y = torch.arange(self.ny) * self.dy
        z = torch.arange(self.nz) * self.dz

        grid = torch.meshgrid(x, y, z, indexing="ij")
        
        self.register_buffer('grid', torch.stack(grid, dim=-1))
        # self.register_buffer('grid', X)
        # self.register_buffer('Y', Y)
        # self.register_buffer('Z', Z)


class CollectionVolume(Grid):
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], n_bg: float):
        super().__init__(shape, spacing)
   
        self.n_bg = n_bg
        self.shapes = nn.ModuleList()
        
    def add(self, shape: List):
        self.shapes.extend(shape)

    def forward(self):
        field = torch.ones_like(self.grid[..., 0]) * self.n_bg
        for shape in self.shapes:
            mask = shape(self.grid)
            field = field * (1 - mask/shape.RI) + mask

        return field
    

class VoxelVolume(nn.Module):
    def __init__(self, shape: Tuple[int, int, int], spacing: Tuple[float, float, float], volume: Tensor = None, n0 = float,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0., object_center: Tuple = None) -> None:
        super().__init__()

        self.shape = shape
        self.spacing = spacing
        self.n0 = n0

        self.volume = torch.nn.Parameter(torch.ones(self.shape).float()*self.n0) if volume is None else self.set_volume(volume)
        self.quat = torch.nn.Parameter(from_axis_angle(torch.tensor(rotation_axis).float(), 
                                                       torch.deg2rad(torch.tensor(rotation_angle))))
        self.center = torch.tensor(self.nx*self.dx/2, self.ny*self.dy/2, self.nz*self.dz/2) if object_center is None else torch.tensor(object_center)
        # self.rot_position = torch.tensor(self.nx*self.dx/2, self.ny*self.dy/2, self.nz*self.dz/2) if rotation_center is None else torch.tensor(object_center)

    def set_volume(self, volume: Tensor):

        assert volume.shape == self.shape
        self.volume = torch.nn.Parameter(volume.float())
        
        return self

    def forward(self, grid: Tensor):

        # Rotation matrix
        R = to_matrix(self.q).squeeze()

        pts = grid - self.center
        pts = pts @ R.T
        pts = pts + self.center

        # normalize the points to the range [0, 1] for grid_sample
        # pts_normalized = pts.clone()
        pts_normalized[..., 0] = (pts[..., 0] / ((self.nx - 1) * self.dx)) * 2 - 1
        pts_normalized[..., 1] = (pts[..., 1] / ((self.ny - 1) * self.dy)) * 2 - 1
        pts_normalized[..., 2] = (pts[..., 2] / ((self.nz - 1) * self.dz)) * 2 - 1

        # grid_sample expects the grid in the shape (N, D, H, W, 3), so we need to add a batch dimension
        pts_normalized = pts_normalized.unsqueeze(0)

        ri = self.ri_distibution.permute(2, 1, 0)

        # Sample the RI distribution using grid_sample
        sampled_ri = F.grid_sample(
            ri.unsqueeze(0).unsqueeze(0),  # (N, C, D, H, W)
            pts_normalized,
            mode='bilinear',
            padding_mode='border',
            align_corners=True
        )

        return sampled_ri.squeeze()

    def voxelize(self):
        # if the input volume is not in voxels
        pass



if __name__ == '__main__':
    # from grid import Grid
    from shapes import *
    from matplotlib import pyplot as plt
    
    from torch.optim import Adam
    from torch import nn
    
    
    # example 1: create two spheres and visulization the RI volume
    sphere1 = Sphere(torch.tensor((55, 50, 40)).float(), torch.tensor((10.)), torch.tensor(1.3), softness=1e2)
    sphere2 = Sphere(torch.tensor((30, 50, 40)).float(), torch.tensor((10.)), torch.tensor(1.3), softness=1e2)
    
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
    optimizer = Adam([sphere1.center], lr=1e-1)
    loss_fn = nn.MSELoss()
    
    centers = []
    for i in range(1000):
        optimizer.zero_grad()
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