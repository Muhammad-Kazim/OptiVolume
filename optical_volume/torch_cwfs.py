import os
import torch
import torchvision
from torchvision.transforms import GaussianBlur, Resize
from py_wave_propagator import torch_volume_prop as propagator

from typing import Optional, Tuple, List
from torch import nn, Tensor
    
from .torch_geometry import Geometry
from .utils import ObjImgMap, median_filter_2d, torch_grad_optr


class CodedWFSForwardModel(Geometry):
    def __init__(self, wavelength, grid_shape, spatial_resolution, n_background, device='cpu', grad=False, grid=None, padding=256, 
                 im_to_ob_space_scale=20, digital_px_size=6e-6, pad_mode='edge'):
        super().__init__(grid_shape, spatial_resolution, n_background, device=device, grad=grad, grid=grid)
        
        self.wl = wavelength
        self.xyz_sup = [self.nx*self.dx, self.ny*self.dy, self.nz*self.dz]
        self.pad = padding
        self.pad_mode = pad_mode
        self.im_to_ob_space_scale = im_to_ob_space_scale
        self.im_space_res = [self.im_to_ob_space_scale*self.dx, self.im_to_ob_space_scale*self.dy]
        self.sum_size = torch.round(digital_px_size/(self.im_to_ob_space_scale*self.dx))
        
        print(f'Digital pixel size {self.sum_size*self.im_space_res[0]*1e6:.2f} um, but desired {digital_px_size*1e6:.2f} um')
        
    def forward(self, lens, dist_m_im, phase_mask, source_field=None, focus_plane_var=0., digital=True, gradient_median_kernel_size=3):
        # main function that does everything
        spatial_resolution = [self.dx, self.dy, self.dz]
        self.dist_m_im = dist_m_im
        
        if source_field is not None:
            self.wavefield_focus(source_field)
        
        assert hasattr(self, 'wf_focus'), "Propagate field through the grid using method wavefield_focus"
        
        # for focal stacks
        if focus_plane_var + 1e-9 > 0: # when 0, props by 1e-9 along the dorward direction
            output_field = propagator.propagate(self.wf_focus, self.wl, spatial_resolution, torch.abs(focus_plane_var), 
                                                padding=self.pad, direction='forward', pad_mode=self.pad_mode)
        else:
            output_field = propagator.propagate(self.wf_focus, self.wl, spatial_resolution, torch.abs(focus_plane_var), 
                                                padding=self.pad, direction='backward', pad_mode=self.pad_mode)
        
        assert hasattr(self, 'blur'), "Control defocus specific specimen spatial partial coherence with method PSC_approximator"
        output_field = self.blur(output_field.real.unsqueeze(0)).squeeze() + 1j*self.blur(output_field.imag.unsqueeze(0)).squeeze()
        
        assert hasattr(self, 'resize'), "Relative resize field with eff_mag_operator: multiples of 20 supported"
        output_field = self.resize(output_field.unsqueeze(0).real).squeeze() + 1j*self.resize(output_field.unsqueeze(0).imag).squeeze()
        
        # pupil function and lens imaging for NA based low-pass filtering and etc.
        # update lens and call forward again
        if(not isinstance(lens, ObjImgMap)): # low-pass-filtering after resizing to remove aliasing introduced by resizing
            raise ValueError("Lens must be an instance of class ObjImgMap")
        output_field = lens.forward(output_field)
        
        # prop to phase mask plane
        self.field_mask_plane = propagator.propagate(output_field, self.wl, self.im_space_res, self.dist_m_im, 
                                                     padding=self.pad, direction='backward', pad_mode=self.pad_mode)

        # mask modulation and prop to image plane
        obj_field = propagator.propagate(self.field_mask_plane*phase_mask, self.wl, self.im_space_res, self.dist_m_im, 
                                         padding=self.pad, direction='forward', pad_mode=self.pad_mode)    
        ref_field = propagator.propagate(phase_mask, self.wl, self.im_space_res, self.dist_m_im, 
                                         padding=self.pad, direction='forward', pad_mode=self.pad_mode)
        
        if self.sum_size > 1:
            conv2d = torch.nn.Conv2d(1, 1, self.sum_size.int().item(), self.sum_size.int().item(), bias=False)
            conv2d.weight = torch.nn.Parameter(torch.ones([self.sum_size.int().item(), self.sum_size.int().item()], dtype=torch.complex64).unsqueeze(0).unsqueeze(0))
            conv2d.weight.requires_grad = False
            
            obj_field = conv2d(obj_field.unsqueeze(0)).squeeze()
            ref_field = conv2d(ref_field.unsqueeze(0)).squeeze()
        
        gt_grad = self.get_gradeint_fields(gradient_median_kernel_size)
        
        return ref_field, obj_field, gt_grad
    
    def wavefield_focus(self, source_field: Tensor = None):
        spatial_resolution = [self.dx, self.dy, self.dz]
        
        if source_field is None:
            source_field = torch.ones([self.nx, self.ny], dtype=torch.complex64) 
        # Take the source field at z=0., prop through self.grid, and retrun field at z=self.xyz_sup[2]
        output_field = propagator.propagate_beam_vol(source_field, self.grid, self.n_bg, self.wl, spatial_resolution, 
                                                     padding=self.pad*0, pad_mode=self.pad_mode) # setting pad to 0 increases speed
        self.wf_focus = propagator.propagate(output_field, self.wl/self.n_bg, spatial_resolution, self.xyz_sup[2]/2, 
                                             padding=self.pad, direction='backward', pad_mode=self.pad_mode)
        
        return self.wf_focus   
        
    # can add the fourier of this Gaussian to the lens' phase
    def PSC_approximator(self, const_sigma=1e-2, defocus_sigma=10e-2, defocus_dist=0.):
        sigma = const_sigma + defocus_sigma*torch.abs(defocus_dist)
        kernel_size = 2*int(4.*sigma + 0.5) + 1
        self.blur = GaussianBlur(kernel_size, sigma=sigma.item())
        
    def eff_mag_operator(self, magnification): # effective magnification operator
        eff_mag = int(magnification/self.im_to_ob_space_scale)
        self.resize = Resize(size=self.nx*eff_mag, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        
    def get_gradeint_fields(self, median_kernel_size):
        
        k = 2*torch.pi/self.wl
        
        # gradient in pixels
        flows_01 = torch_grad_optr(torch.angle(self.field_mask_plane))
        
        # remove 2pi peaks at phase wrapping junctions
        flow_0 = median_filter_2d(flows_01[0].unsqueeze(0).unsqueeze(0), median_kernel_size).squeeze()
        flow_1 = median_filter_2d(flows_01[1].unsqueeze(0).unsqueeze(0), median_kernel_size).squeeze()
        
        # OPD gradients in image sapce units
        flow_0 = flow_0/(self.im_space_res[0])/k*self.dist_m_im
        flow_1 = flow_1/(self.im_space_res[1])/k*self.dist_m_im
        
        if self.sum_size > 1:
            conv2d = torch.nn.Conv2d(1, 1, self.sum_size.int().item(), self.sum_size.int().item(), bias=False)
            conv2d.weight = torch.nn.Parameter(torch.ones([self.sum_size.int().item(), self.sum_size.int().item()]).unsqueeze(0).unsqueeze(0)/self.sum_size**2)
            conv2d.weight.requires_grad = False
            
            flow_0 = conv2d(flow_0.unsqueeze(0)).squeeze()
            flow_1 = conv2d(flow_1.unsqueeze(0)).squeeze()
            
        flow_0 = flow_0/(self.im_space_res[0]*self.sum_size)
        flow_1 = flow_1/(self.im_space_res[1]*self.sum_size)
        
        return [flow_0, flow_1]
    
    
class PhaseMask():
    def __init__(self, side_length: Tensor, spatial_resolution: Tuple[float, float, float], 
                 grid_shape: Tuple[float, float, float]):
        
        self.grid_size = grid_shape[:2]
        self.tile_size_px = torch.ceil(side_length/spatial_resolution[0]).int()
        self.nx, self.ny = torch.ceil(grid_shape[0]/self.tile_size_px).int(), torch.ceil(grid_shape[1]/self.tile_size_px).int()
        
        # self.height_map = None
        
    def forward(self, RI_pm: Tensor, wl: float, sigma: float = None, padding: int = 0, RI_bg: float = 1.):

        assert hasattr(self, 'height_map'), "Create height map first."
        
        mask = self.height_map.repeat_interleave(self.tile_size_px, 0).repeat_interleave(self.tile_size_px, 1)[:self.grid_size[0], :self.grid_size[1]]
        mask = torch.clamp(mask, min=0) # height cannot be negative
        mask = torch.nn.functional.pad(mask, pad = ([padding]*4))
        mask = mask*RI_pm + (mask.max() - mask)*RI_bg
        
        if sigma is not None:
            kernel_size = 2*int(4.*sigma + 0.5) + 1
            mask = GaussianBlur(kernel_size, sigma=sigma)(mask.unsqueeze(0)).squeeze()
    
        return torch.exp(1j*(2*torch.pi/wl)*mask)
    
    def create_height_map(self, height: Tensor, prob: float = 0.5, hmap_grad: bool = False, device: str = 'cpu'):
        
        if height.dim() == 0 or height.dim() == 1:
            self.height_map = self._init_random(prob=prob).float()*height
        else:
            assert height.size() == self.get_num_tiles(), f"Param height must be of shape {self.nx}x{self.ny}"
            self.height_map = height
        
        if hmap_grad:
            self.height_map.requires_grad = True
        
        if device == 'cuda':
            assert torch.cuda.is_available(), "CUDA not available, switch to CPU."
        
        self.height_map.to(device)
    
    def _init_random(self, prob: float = 0.5):
        return torch.rand([self.nx, self.ny]) > prob    
    
    def get_num_tiles(self):
        return (self.nx, self.ny)
    

if __name__=='__main__':
    pass