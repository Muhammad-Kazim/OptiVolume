import torch
from torch import nn, Tensor
import torchvision
from typing import Optional, Tuple, List

# from py_wave_propagator import torch_volume_prop as propagator

class FreeSpacePropagator(nn.Module):
    """
        Propagation through a homogenous medium. Base unit is meters.
        Use when input has to be updated.
    
        Args:
            field (Tensor): 2d complex field on a plane
            wavelength (float): if not air, than wl =/ RI_background
            spatial_resolution (): _description_
            dist (float): distance bw parallel planes in meters
    
        Returns:
            NDArray[np.complex128]: field at parallel plane distance dist away 
    """
    def __init__(self,
              wavelength: float, 
              spacing: Tuple[float, float],
              distance: float,
              direction: Optional[str] = 'forward',
              padding: Optional[int] = None, 
              pad_mode: str = 'edge',
    ) -> Tensor:
        super().__init__()

        if padding is not None:
            self.pad_do = True
            self.pad_by = padding
            self.PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode) # edge make sense
        else:
            self.pad_do = False
            
        self.wavelength = wavelength
        self.dx, self.dy = spacing
        self.direction = direction
        
        self.distance = nn.Parameter(torch.tensor(distance))

    def forward(self, field: Tensor) -> Tensor:

        device = field.device

        if self.pad_do:
            field = self.PAD(field.real) + 1j*self.PAD(field.imag)

        H, W = field.shape[-2:]
        
        # Spatial frequency grid
        kx = torch.fft.fftfreq(H, self.dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(W, self.dy, device=device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
        
        Kz = (2 * torch.pi/self.wavelength)**2 - Kx**2 - Ky**2
        mask = torch.sigmoid(Kz/1e-12) # to remove negatives which remove evanascent waves
        Kz = torch.sqrt(Kz*mask)

        field_fft = torch.fft.fft2(field)
        if (self.direction == 'backward' and self.distance > 0.) or (self.direction == 'forward' and self.distance < 0.):
            transfer_function = torch.conj(torch.exp(1j*Kz*self.distance))
        else:
            transfer_function = torch.exp(1j*Kz*self.distance)

        field = torch.fft.ifft2(field_fft * transfer_function)
        
        if self.pad_do:
            return field[self.pad_by:-1*self.pad_by, self.pad_by:-1*self.pad_by]

        return field
    
    def set_direction(self, direction):
        self.direction = direction

        return self
    

class VolumePropagator(nn.Module):
    """
        Propagates a 2D complex wavefield using the beam propagationm method. Base units is meters.
    
        Args:
            field (Tensor): Input 2D complex wavefield in the (x, y, 0) plane
            RI_distribution (Tensor): 3D refractive index distribution (x, y, z)
            RI_background (float): background refractive index
            wavelength (float): wavelength of source in vacuum
            spatial_resolution (Tuple[float, float, float]): (dx, dy, dz). dz is the propagation step or slice thickness
            padding (Optional[int], optional): Number of pixels added to the field. Defaults to None.
    
        Returns:
            NDArray[np.complex64]: 2D complex wavefield at (x, y, -1)
    """
    
    def __init__(
        self, 
        wavelength: float, 
        spacing: Tuple[float, float, float],
        RI_background: float,
        padding: Optional[int] = None, 
        pad_mode: str = 'edge'
    ) -> Tensor:
        super().__init__()
        

        # self.padding = True if padding else False
        # if self.padding:
        #     self.PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode)

        if padding is not None:
            self.pad_do = True
            self.pad_by = padding
            self.PAD = torchvision.transforms.Pad(padding, padding_mode=pad_mode) # edge make sense
        else:
            self.pad_do = False
            
        self.wavelength = wavelength
        self.dx, self.dy, self.dz = spacing
        self.n0 = RI_background
        self.source_field = None

    def forward(self, RI_distribution: Tensor) -> Tensor:

        Nx, Ny, Nz = RI_distribution.shape
        device = RI_distribution.device

        # Spatial frequency grid
        k0 = 2 * torch.pi / self.wavelength
        kx = torch.fft.fftfreq(Nx, self.dx, device=device) * 2 * torch.pi
        ky = torch.fft.fftfreq(Ny, self.dy, device=device) * 2 * torch.pi
        Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')

        Kz = torch.sqrt(0j + (self.n0*k0)**2 - Kx**2 - Ky**2)
        transfer_function = torch.exp(1j*Kz*self.dz)

        if self.source_field is not None:
            assert self.source_field.shape == (Nx, Ny), f'source field must have dims {Nx}x{Ny}'
            self.source_field = self.source_field.to(device)
        else:
            self.source_field = torch.ones([Nx, Ny], dtype=torch.complex64, device=device) 
        
        if self.pad_do:
            field = self.PAD(self.source_field.real) + 1j*self.PAD(self.source_field.imag) # edge make sense
        else:
            field = self.source_field.clone()
        
        # Forward propagation
        for z in range(Nz):
            field_fft = torch.fft.fft2(field)
            phase = torch.exp(1j*k0*(RI_distribution[..., z] - self.n0)*self.dz)
            
            if self.pad_do:
                phase = self.PAD(phase.real) + 1j*self.PAD(phase.imag) # no delay in the padded region
                
            field = torch.fft.ifft2(field_fft * transfer_function) * phase
        
        if self.pad_do:
            return field[self.pad_by:-1*self.pad_by, self.pad_by:-1*self.pad_by]

        return field
    
    def set_source_field(self, field: Tensor):
        self.source_field = field # field at input plane of the volume z = 0

        return self
    
    def set_RI_background(self, RI_background: float):
        self.n0 = RI_background

        return self
    


# class Freespace(nn.Module):
#     def __init__(self, WL: float, spacing: Tuple[float, float], shape: Tuple[int, int], padding=None, pad_mode='edge'):
#         super().__init__()
        
#         self.prop = propagator.FreeSpacePropagator(WL, spacing, shape, padding=padding, pad_mode=pad_mode)
        
#     def forward(self, field: Tensor) -> Tensor:
#         field = self.prop.forward(field, self.dist, direction=self.direction)
        
#         return field
    
#     def set_params(self, dist: float, direction='forward'):
#         self.dist = dist
#         self.direction = direction
        
#         return self
    

# class BeamPropMethod(nn.Module):
#     def __init__(self, WL: float, spacing: Tuple[float, float, float], shape: Tuple[int, int, int], padding=None, pad_mode: str ='edge'):
#         super().__init__()
        
#         self.prop = propagator.VolumePropagator(WL, spacing, shape, padding=padding, pad_mode=pad_mode)
        
#     def forward(self, field: Tensor, RI_distribution: Tensor, RI_background: float) -> Tensor:
#         field = self.prop.forward(field, RI_distribution, RI_background)
        
#         return field
    

# if __name__ == "__main__":
#     from matplotlib import pyplot as plt
    
#     # example: free space propagation using ASM
#     WL = 500e-9
#     spacing = [10e-6, 10e-6]
#     shape = [100, 100]
    
#     x = (torch.arange(100)-50)*spacing[0]
#     X, Y = torch.meshgrid(x, x, indexing='ij')
    
#     wavefield = torch.where(X**2 + Y**2 < (300e-6)**2, 1+0j, 0+0j)
    
#     prop = Freespace(WL, spacing, shape, padding=1000)
#     field = prop.set_params(50e-3).forward(wavefield)
    
#     plt.imshow(field.abs())
#     plt.colorbar()
#     plt.show()
    
#     # example: beam propagation method
#     import os
#     import sys
#     sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-1]))
#     import scene
    
#     # example 1: create two spheres and visulization the RI volume
#     sphere1 = scene.shapes.Sphere(torch.tensor((55, 50, 40)).float(), torch.tensor((10.)), torch.tensor(1.3), softness=1e2)
#     sphere2 = scene.shapes.Sphere(torch.tensor((30, 50, 40)).float(), torch.tensor((20.)), torch.tensor(1.3), softness=1e2)
    
#     grid = scene.grid.Grid((100, 100, 100), (1, 1, 1))
    
#     vol = scene.volume.Volume(grid, n_bg=torch.tensor(1.33))
#     vol.add([sphere1, sphere2])
#     # print(vol.grid.X.dtype, vol.n_bg)
    
#     ri_dist = vol.forward()
#     # print(ri_dist.shape)
    
#     vol_prop = BeamPropMethod(WL, spacing + [10e-6], shape + [100], padding=0)
#     exit_field = vol_prop.forward(torch.ones_like(field), ri_dist, 1.33)
    
#     plt.imshow(exit_field.abs().detach())
#     plt.colorbar()
#     plt.show()
    
#     prop = Freespace(WL/1.33, spacing, shape, padding=1000)
#     centered_field = prop.set_params(500e-6, direction='backward').forward(exit_field)
    
#     plt.imshow(centered_field.abs().detach())
#     plt.colorbar()
#     plt.show()
    
#     print(centered_field.requires_grad)
    