import pickle
import os
import torch
from typing import Optional, Tuple, List
from torch import nn, Tensor
from scipy.spatial.transform import Rotation as Rot


# add cuboids, hemisphere, prisms
# handle intersecting objects
class Geometry:
    def __init__(self, grid_shape: Tuple[int, int, int], spatial_resolution: Tuple[float, float, float], 
                 n_background: Tensor, device: str = 'cpu', grad: bool = False, grid=None):
        """
        Initialize a single grid with background refractive index.
        
        grid_shape: Tuple of (nx, ny, nz) defining the grid dimensions.
        spatial_resolution: Tuple of (dx, dy, dz) defining spatial resolution.
        n_background: Background refractive index n_bg.
        """
        
        print(f'''Coordiante system with size: \n 
              X = [0, {spatial_resolution[0]*grid_shape[0]*1e6:.2f} um], Res_X = {spatial_resolution[0]*1e6:.2f} um
              Y = [0, {spatial_resolution[1]*grid_shape[1]*1e6:.2f} um], Res_Y = {spatial_resolution[1]*1e6:.2f} um
              Z = [0, {spatial_resolution[2]*grid_shape[2]*1e6:.2f} um], Res_Z = {spatial_resolution[2]*1e6:.2f} um
              Immersion RI: {n_background}
      ''')
        
        self.dx, self.dy, self.dz = spatial_resolution
        self.nx, self.ny, self.nz = grid_shape
        self.n_bg = n_background
        self.device = device

        # Initialize the grid and meshgrid
        if self.device == 'cuda':
            assert torch.cuda.is_available(), "CUDA not available, switch to CPU."
        if grid is None:
            self.grid = torch.ones([self.nx, self.ny, self.nz], device=self.device, requires_grad=grad)*self.n_bg
        else:
            assert [grid.shape[0], grid.shape[1], grid.shape[2]] == [self.nx, self.ny, self.nz], "[nx, ny, nx] not same as grid's shape."
            self.grid = grid.to(self.device)

        x = torch.arange(self.nx) * self.dx
        y = torch.arange(self.ny) * self.dy
        z = torch.arange(self.nz) * self.dz
        self._x_mesh, self._y_mesh, self._z_mesh = torch.meshgrid(x, y, z, indexing="ij")

    def add_cube(self, center: Tuple[float, float, float], side_length: float, RI: float, softness: float = 10e-9, random_rotation: bool = False):
        """
        Add a cube to the grid.
        
        center: Tuple of (cx, cy, cz) defining the cube's center in real units.
        side_length: Length of the cube's side in real units.
        RI: refractive index of homogenous shape.
        """
        
        cx, cy, cz = center
        
        if side_length.dim() > 0 and side_length.size()[0] > 1:
            sx = side_length[0]/2
            sy = side_length[1]/2
            sz = side_length[2]/2
        else:
            sx = sy = sz = side_length/2
            
        X = self._x_mesh - cx
        Y = self._y_mesh - cy
        Z = self._z_mesh - cz
        
        if random_rotation:
            rot_mats = torch.tensor(Rot.random().as_matrix()).float()
        else:
            rot_mats = torch.eye(3).float()
            
        # Stack into vector form
        coords = torch.stack([X, Y, Z], dim=-1)
        rotated = torch.tensordot(coords, rot_mats, dims=([-1], [1]))  # (..., 3)
        
        mask_x = self._soft_step(rotated[..., 0] + sx, softness=softness) * (1 - self._soft_step(rotated[..., 0] - sx, softness=softness))
        mask_y = self._soft_step(rotated[..., 1] + sy, softness=softness) * (1 - self._soft_step(rotated[..., 1] - sy, softness=softness))
        mask_z = self._soft_step(rotated[..., 2] + sz, softness=softness) * (1 - self._soft_step(rotated[..., 2] - sz, softness=softness))

        # mask_x = self._soft_step(self._x_mesh - (cx - sx), softness=softness) * (1 - self._soft_step(self._x_mesh - (cx + sx), softness=softness))
        # mask_y = self._soft_step(self._y_mesh - (cy - sy), softness=softness) * (1 - self._soft_step(self._y_mesh - (cy + sy), softness=softness))
        # mask_z = self._soft_step(self._z_mesh - (cz - sz), softness=softness) * (1 - self._soft_step(self._z_mesh - (cz + sz), softness=softness))

        cube_mask = mask_x * mask_y * mask_z  # smooth transition between 0 and 1

        self.grid = self.grid * (1 - cube_mask) + RI * cube_mask

    
    def add_cubes(self, centers: Tensor, side_lengths: Tensor, RIs: Tensor, softness: float = 10e-9, random_rotation: bool = False):
        
        assert centers.dim()  == 2, "Each row must contain ellipsoid attributes"
        assert RIs.size()[0] == centers.shape[0] == side_lengths.size()[0], "Each row must contain cube attributes"
        
        for i in range(centers.shape[0]):
            self.add_cube(centers[i], side_lengths[i], RIs[i], softness=softness, random_rotation=random_rotation)
            
    
    def add_sphere(self, center: Tensor, radius: Tensor, RI: Tensor, softness: float = 1e-12):
        """
        Add a sphere to the grid.
        
        center: Tuple of (cx, cy, cz) defining the sphere's center in real units.
        radius: Radius of the sphere in real units.
        RI: refractive index of homogenous shape.
        """
        cx, cy, cz = center

        mask_x = (self._x_mesh - cx)**2
        mask_y = (self._y_mesh - cy)**2
        mask_z = (self._z_mesh - cz)**2

        sphere_mask = 1 - self._soft_step(mask_x + mask_y + mask_z - radius**2, softness=softness)  # smooth transition between 0 and 1

        self.grid = self.grid * (1 - sphere_mask) + RI * sphere_mask
    
    
    def add_spheres(self, centers: Tensor, radii: Tensor, RIs: Tensor, softness: float = 1e-12):
        
        assert centers.dim() == 2, "Each row must contain sphere attributes"
        assert RIs.size()[0] == centers.shape[0] == radii.size()[0], "Each row must contain sphere attributes"
        
        for i in range(centers.shape[0]):
            self.add_sphere(centers[i], radii[i], RIs[i], softness=softness)
            
    
    def add_ellipsoid(self, center: Tensor, radii: Tensor, RI: Tensor, random_rotation: bool = False, softness: float = 1e-12):

        cx, cy, cz = center
        rx, ry, rz = radii
        
        X = self._x_mesh - cx
        Y = self._y_mesh - cy
        Z = self._z_mesh - cz

        if random_rotation:
            rot_mats = torch.tensor(Rot.random().as_matrix()).float()
        else:
            rot_mats = torch.eye(3).float()
        
        # Stack into vector form
        coords = torch.stack([X, Y, Z], dim=-1)
        
        rotated = torch.tensordot(coords, rot_mats, dims=([-1], [1]))  # (..., 3)

        # Compute normalized squared distance
        mask_x = (rotated[..., 0] / rx) ** 2
        mask_y = (rotated[..., 1] / ry) ** 2
        mask_z = (rotated[..., 2] / rz) ** 2
            
        # mask_x = ((self._x_mesh - cx)/rx)**2
        # mask_y = ((self._y_mesh - cy)/ry)**2
        # mask_z = ((self._z_mesh - cz)/rz)**2

        ellipsoid_mask = 1 - self._soft_step(mask_x + mask_y + mask_z - 1, softness=softness)  # smooth transition between 0 and 1

        self.grid = self.grid * (1 - ellipsoid_mask) + RI * ellipsoid_mask
    
    
    def add_ellipsoids(self, centers: Tensor, radii: Tensor, RIs: Tensor, random_rotation: bool = False, softness: float = 1e-12):
        
        assert centers.dim() == radii.dim() == 2, "Each row must contain ellipsoid attributes"
        assert RIs.size()[0] == centers.shape[0] == radii.shape[0], "Each row must contain ellipsoid attributes"
        
        for i in range(centers.shape[0]):
            self.add_ellipsoid(centers[i], radii[i], RIs[i], random_rotation=random_rotation, softness=softness)
            
        
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
    
        
    def __add__(self, obj):
        """Concatenates the populated grid of two Geometry class objects.

        Args:
            obj (geometry)
            return: a new geometry object.
        """
        
        if not isinstance(obj, Geometry):
            raise TypeError(f"Cannot add Geometry with {type(obj)}.")
        
        self_attr = [self.dx, self.dy, self.dz, self.nx, self.ny]
        obj_attr = [obj.dx, obj.dy, obj.dz, obj.nx, obj.ny]
        
        if self_attr == obj_attr:
            return Geometry(
                [self.nx, self.ny, self.nz + obj.nz], 
                [self.dx, self.dy, self.dz], self.n_0, 
                np.concatenate([self.get_grid(), obj.get_grid()], axis=2) # remove numpy usage
                ) 
        else:
            raise AssertionError('Objects have different attributes.')
    
    
    def _soft_step(self, x, softness=1e-6): 
        return torch.sigmoid(x / softness)
        
        
    def save(self, filename):
        """Saves instance as a .pkl file.

        Args:
            filename (str): path/to/file.pkl
        """
        
        if os.path.isfile(filename):
            print('File exists.')
        else:
            print('Saving geometry object...')
            
            with open(filename, 'wb') as outp:
                pickle.dump(self, outp, -1)
        

    def get_grid(self):
        """
        Return the current grid with all shapes added.
        """
        return self.grid
    
    
    def reset_grid(self):
        """
        Return the current grid with all shapes added.
        """
        self.grid = torch.ones_like(self.grid)*self.n_bg
    
    
    def __repr__(self):
        
        return f'''Coordiante system with size: \n 
              X = [0, {self.dx*self.nx:.2e}], Res_X = {self.dx}
              Y = [0, {self.dy*self.ny:.2e}], Res_Y = {self.dy}
              Z = [0, {self.dz*self.nz:.2e}], Res_Z = {self.dz}
              Immersion RI: {self.n_0}
              '''


if __name__=='__main__':
    pass