import torch
from torch import Tensor, nn
from scipy.spatial.transform import Rotation as Rot


class Sphere(nn.Module):
    def __init__(self, center: Tensor, radius: Tensor, RI: Tensor, softness=1e-12):
        super().__init__()
        
        self.center = nn.Parameter(center)
        self.radius = nn.Parameter(radius)
        self.RI = nn.Parameter(RI)
        self.softness = softness

    def forward(self, grid: Tensor):
        cx, cy, cz = self.center
        
        X = grid.X - cx
        Y = grid.Y - cy
        Z = grid.Z - cz

        return 1 - _soft_step(X**2 + Y**2 + Z**2 - self.radius**2, self.softness)


def make_spheres(centers: Tensor, radii: Tensor, RIs: Tensor, softness: float = 1e-12):
        
    assert centers.dim() == 2, "Each row must contain sphere attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.size()[0], "Each row must contain sphere attributes"

    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Sphere(centers[i], radii[i], RIs[i], softness=softness))
    
    return shapes
            
            
class Cube(nn.Module):
    def __init__(self, center: Tensor, length: Tensor, RI: Tensor, rotation=None, softness=1e-9):
        super().__init__()
        
        self.center = nn.Parameter(center)
        self.length = nn.Parameter(length)
        self.RI = nn.Parameter(RI)
        
        if rotation:
            self.rotation = torch.tensor(Rot.random().as_matrix()).float()
        else:
            self.rotation = torch.eye(3).float()
        
        self.softness = softness

    def forward(self, grid: Tensor):
        cx, cy, cz = self.center
        
        if self.length.dim() > 0 and self.length.size()[0] > 1:
            sx = self.length[0]/2
            sy = self.length[1]/2
            sz = self.length[2]/2
        else:
            sx = sy = sz = self.length/2
            
        X = grid.X - cx
        Y = grid.Y - cy
        Z = grid.Z - cz
            
        # Stack into vector form
        coords = torch.stack([X, Y, Z], dim=-1)
        rotated = torch.tensordot(coords, self.rotation, dims=([-1], [1]))  # (..., 3)
        
        mask_x = _soft_step(rotated[..., 0] + sx, softness=self.softness) * (1 - _soft_step(rotated[..., 0] - sx, softness=self.softness))
        mask_y = _soft_step(rotated[..., 1] + sy, softness=self.softness) * (1 - _soft_step(rotated[..., 1] - sy, softness=self.softness))
        mask_z = _soft_step(rotated[..., 2] + sz, softness=self.softness) * (1 - _soft_step(rotated[..., 2] - sz, softness=self.softness))

        return mask_x * mask_y * mask_z  # smooth transition between 0 and 1


def make_cubes(centers: Tensor, lengths: Tensor, RIs: Tensor, rotation: bool = False, softness: float = 1e-9):
        
    assert centers.dim()  == 2, "Each row must contain ellipsoid attributes"
    assert RIs.size()[0] == centers.shape[0] == lengths.size()[0], "Each row must contain cube attributes"
    
    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Cube(centers[i], lengths[i], RIs[i], rotation=rotation, softness=softness))

    return shapes

            
class Ellipsoid(nn.Module):
    def __init__(self, center: Tensor, radii: Tensor, RI: Tensor, 
                 rotation: bool = False, softness: float = 1e-12):
        super().__init__()
        
        self.center = nn.Parameter(center)
        self.radii = nn.Parameter(radii)
        self.RI = nn.Parameter(RI)
        
        if rotation:
            self.rotation = torch.tensor(Rot.random().as_matrix()).float()
        else:
            self.rotation = torch.eye(3).float()
        
        self.softness = softness
    
    def forward(self, grid: Tensor):
        cx, cy, cz = self.center
        rx, ry, rz = self.radii
        
        X = grid.X - cx
        Y = grid.Y - cy
        Z = grid.Z - cz
        
        # Stack into vector form
        coords = torch.stack([X, Y, Z], dim=-1)
        rotated = torch.tensordot(coords, self.rotation, dims=([-1], [1]))  # (..., 3)

        # Compute normalized squared distance
        mask_x = (rotated[..., 0] / rx) ** 2
        mask_y = (rotated[..., 1] / ry) ** 2
        mask_z = (rotated[..., 2] / rz) ** 2

        return 1 - _soft_step(mask_x + mask_y + mask_z - 1, softness=self.softness)
    

def make_ellipsoids(centers: Tensor, radii: Tensor, RIs: Tensor, rotation: bool = False, softness: float = 1e-12):
        
    assert centers.dim() == radii.dim() == 2, "Each row must contain ellipsoid attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.shape[0], "Each row must contain ellipsoid attributes"
    
    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Ellipsoid(centers[i], radii[i], RIs[i], rotation=rotation, softness=softness))
    
    return shapes

    
def _soft_step(x: Tensor, softness=1e-6): 
    return torch.sigmoid(x / softness)


if __name__ == '__main__':
    from grid import Grid
    from matplotlib import pyplot as plt
    
    grid = Grid([100, 100, 100], [1., 1., 1.])
    obj = Ellipsoid(torch.tensor([50, 20, 20]).float(), torch.tensor([10, 5, 9]).float(), 
                    torch.tensor(1.5), rotation=True, softness=1e-6)
    
    print(f'Center: {obj.center}')
    print(f'Radii: {obj.radii}')
    print(f'RI: {obj.RI}')
    print(f'Rot Matrix: {obj.rotation}')
    
    plt.imshow(obj(grid)[..., 20].detach())
    plt.plot(20, 50, 'r*')
    plt.colorbar()
    plt.grid()
    plt.show()