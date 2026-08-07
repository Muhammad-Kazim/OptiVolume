import torch
from torch import Tensor, nn
from scipy.spatial.transform import Rotation as Rot

from typing import Tuple

from .utils import from_axis_angle, to_axis_angle, to_matrix

# all shapes are voxel volumes of analytical objects

class Sphere(nn.Module):
    def __init__(self, center: Tensor, radius: Tensor, RI: Tensor, softness=1e-12,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__()
        
        self.center = nn.Parameter(center)
        self.radius = nn.Parameter(radius)
        self.RI = nn.Parameter(RI)
        self.softness = softness

        q = from_axis_angle(torch.tensor(rotation_axis).float(), torch.deg2rad(torch.tensor(rotation_angle)))
        self.quat = nn.Parameter(q)  

    def forward(self, grid: Tensor):

        # Rotation Matrix
        R = to_matrix(self.quat).squeeze()

        # World -> object coordinates
        pts = grid - self.center
        pts = pts @ R

        return self.RI * (1 - _soft_step(pts[..., 0]**2 + pts[..., 1]**2 + pts[..., 2]**2 - self.radius**2, self.softness))
    
    def set_quat(self, rotation_axis, rotation_angle):

        q = from_axis_angle(torch.tensor(rotation_axis).float(), torch.deg2rad(torch.tensor(rotation_angle)))
        self.quat = torch.nn.Parameter(q)
        return self


def make_spheres(centers: Tensor, radii: Tensor, RIs: Tensor, softness: float = 1e-12):
        
    assert centers.dim() == 2, "Each row must contain sphere attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.size()[0], "Each row must contain sphere attributes"

    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Sphere(centers[i], radii[i], RIs[i], softness=softness))
    
    return shapes
            
            
class Cube(nn.Module):
    def __init__(self, center: Tensor, length: Tensor, RI: Tensor, softness=1e-9, 
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
        super().__init__()
        
        self.center = nn.Parameter(center)
        self.length = nn.Parameter(length)
        self.RI = nn.Parameter(RI)

        q = from_axis_angle(torch.tensor(rotation_axis).float(), torch.deg2rad(torch.tensor(rotation_angle)))
        self.quat = nn.Parameter(q)  
        
        self.softness = softness

    def forward(self, grid: Tensor):

        # Rotation Matrix
        R = to_matrix(self.quat).squeeze()

        # World -> object coordinates
        pts = grid - self.center
        pts = pts @ R

        # cx, cy, cz = self.center
        
        if self.length.dim() > 0 and self.length.size()[0] > 1:
            sx = self.length[0]/2
            sy = self.length[1]/2
            sz = self.length[2]/2
        else:
            sx = sy = sz = self.length/2
            
        # X = grid.X - cx
        # Y = grid.Y - cy
        # Z = grid.Z - cz
            
        # # Stack into vector form
        # coords = torch.stack([X, Y, Z], dim=-1)
        # rotated = torch.tensordot(coords, self.rotation, dims=([-1], [1]))  # (..., 3)
        
        mask_x = _soft_step(pts[..., 0] + sx, softness=self.softness) * (1 - _soft_step(pts[..., 0] - sx, softness=self.softness))
        mask_y = _soft_step(pts[..., 1] + sy, softness=self.softness) * (1 - _soft_step(pts[..., 1] - sy, softness=self.softness))
        mask_z = _soft_step(pts[..., 2] + sz, softness=self.softness) * (1 - _soft_step(pts[..., 2] - sz, softness=self.softness))

        return self.RI * (mask_x * mask_y * mask_z)  # smooth transition between 0 and 1
    
    def set_quat(self, rotation_axis, rotation_angle):

        q = from_axis_angle(torch.tensor(rotation_axis).float(), torch.deg2rad(torch.tensor(rotation_angle)))
        self.quat = torch.nn.Parameter(q)
        return self


def make_cubes(centers: Tensor, lengths: Tensor, RIs: Tensor, rotation: bool = False, softness: float = 1e-9):
        
    assert centers.dim()  == 2, "Each row must contain ellipsoid attributes"
    assert RIs.size()[0] == centers.shape[0] == lengths.size()[0], "Each row must contain cube attributes"
    
    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Cube(centers[i], lengths[i], RIs[i], rotation=rotation, softness=softness))

    return shapes

            
class Ellipsoid(nn.Module):
    def __init__(self, center: Tensor, radii: Tensor, RI: Tensor, softness: float = 1e-12,
                 rotation_axis: Tuple = (0, 0, 1), rotation_angle: float = 0.):
                 
        super().__init__()
        
        self.center = nn.Parameter(center)
        self.radii = nn.Parameter(radii)
        self.RI = nn.Parameter(RI)

        q = from_axis_angle(torch.tensor(rotation_axis).float(), torch.deg2rad(torch.tensor(rotation_angle)))
        self.quat = nn.Parameter(q)
        
        self.softness = softness
    
    def forward(self, grid: Tensor):

        # Rotation Matrix
        R = to_matrix(self.quat).squeeze()

        # World -> object coordinates
        pts = grid - self.center
        pts = pts @ R

        # cx, cy, cz = self.center
        rx, ry, rz = self.radii
        
        # X = grid.X - cx
        # Y = grid.Y - cy
        # Z = grid.Z - cz
        
        # # Stack into vector form
        # coords = torch.stack([X, Y, Z], dim=-1)
        # rotated = torch.tensordot(coords, self.rotation, dims=([-1], [1]))  # (..., 3)

        # Compute normalized squared distance
        mask_x = (pts[..., 0] / rx) ** 2
        mask_y = (pts[..., 1] / ry) ** 2
        mask_z = (pts[..., 2] / rz) ** 2

        return self.RI * (1 - _soft_step(mask_x + mask_y + mask_z - 1, softness=self.softness))
    
    def set_quat(self, rotation_axis, rotation_angle):

        q = from_axis_angle(torch.tensor(rotation_axis).float(), torch.deg2rad(torch.tensor(rotation_angle)))
        self.quat = torch.nn.Parameter(q)
        return self
    

def make_ellipsoids(centers: Tensor, radii: Tensor, RIs: Tensor, rotation: bool = False, softness: float = 1e-12):
        
    assert centers.dim() == radii.dim() == 2, "Each row must contain ellipsoid attributes"
    assert RIs.size()[0] == centers.shape[0] == radii.shape[0], "Each row must contain ellipsoid attributes"
    
    shapes = []
    for i in range(centers.shape[0]):
        shapes.append(Ellipsoid(centers[i], radii[i], RIs[i], rotation=rotation, softness=softness))
    
    return shapes
    
class Plane(nn.Module):
    """
        Add a thick plane to the grid. Physical coordinates.

        Args:
            point (float): point that lies on the plane.
            normal (float): normal to the planes.
            RI (float): RI of plane.
            thickness (float, optional): Thickness/2 on either halfspace.
    """
    def __init__(self, point: Tensor, normal: Tensor, thickness: Tensor, RI: float, softness: float = 1e-12):
        super().__init__()
        
        self.point = nn.Parameter(point)
        self.normal = nn.Parameter(normal)/torch.linalg.vector_norm(normal)
        self.thickness = nn.Parameter(thickness)
        self.RI = nn.Parameter(RI)

        self.softness = softness
    
    def forward(self, grid: Tensor):
        px, py, pz = self.point
        nx, ny, nz = self.normal
        
        X = grid.X - px
        Y = grid.Y - py
        Z = grid.Z - pz
        
        # Plane equation: n . (x - p) = 0
        mask_p = self._soft_step(nx * X + ny * Y + nz * Z, softness=self.softness)
        mask_n = 1 - self._soft_step(nx * X + ny * Y + nz * Z - self.thickness, softness=self.softness)
        
        return self.RI * (1 - mask_p * mask_n)
    
    
def add_plane(self, point: Tensor, normal: Tensor, RI: Tensor, thickness: Tensor = None, softness: float = 1e-9):
        """
        Add a thick plane to the grid. Physical coordinates.

        Args:
            point (float): point that lies on the plane.
            normal (float): normal to the planes.
            RI (float): RI of plane.
            thickness (float, optional): Thickness/2 on either halfspace.
        """
        px, py, pz = point
        nx, ny, nz = normal/torch.linalg.vector_norm(normal)
        
        if thickness is None:
            thickness = 2*self.dz
            
        # Plane equation: n . (x - p) = 0
        mask_p = self._soft_step(nx * (self._x_mesh - px) + ny * (self._y_mesh - py) + nz * (self._z_mesh - pz), softness=softness)
        mask_n = 1 - self._soft_step(nx * (self._x_mesh - px) + ny * (self._y_mesh - py) + nz * (self._z_mesh - pz) - thickness, softness=softness)
        
        plane_mask = mask_p * mask_n
        self.grid = self.grid * (1 - plane_mask) + RI * plane_mask
        
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
    
    # make ellipsoids examples
    
    obj_2 = make_ellipsoids(torch.tensor([[50, 20, 20], [50, 20, 50]]).float(), 
                            torch.tensor([[10, 5, 9], [10, 5, 9]]).float(), 
                            torch.tensor([1.5, 1.6]), rotation=True)
    
    print("Two Ellipsoids", obj_2)
    
    plt.imshow(obj_2[1](grid)[:, 20, :].detach())
    plt.plot(20, 50, 'r*')
    plt.colorbar()
    plt.grid()
    plt.show()
    
    print(nn.ModuleList((obj_2 + [obj])))