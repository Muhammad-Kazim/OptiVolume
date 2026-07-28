import os
import sys
import torch
import torchvision
from torchvision.transforms import GaussianBlur, Resize
from py_wave_propagator import torch_volume_prop as propagator

from typing import Optional, Tuple, List
from torch import nn, Tensor

sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-1]))

import optics
import scene
import utils

from matplotlib import pyplot as plt

# example: functioning
WL = 500e-9
spacing = [100e-9, 100e-9, 100e-9]
shape = [500, 500, 100]
NA = 0.9
DIST_PM_IM = 0.4e-3
TILE_LEN = 0.25e-6
PM_HEIGHT = 600e-9
PM_RI = 1.3
N_BG = torch.tensor(1.33)
PAD = 512
FOC_PLANE_VAR = 0.1e-6

x = (torch.arange(shape[0]) - int(shape[0]/2))*spacing[0]
X, Y = torch.meshgrid(x, x, indexing='ij')

# wavefield = torch.where(X**2 + Y**2 < (10e-6)**2, 1+0j, 0+0j)

lens = optics.lens.ObjImgMap(WL, spacing[:2], shape[:2])

pm_obj = optics.phase_mask.PhaseMask(torch.tensor(TILE_LEN), spacing[:2], shape[:2], height=torch.tensor(PM_HEIGHT))
mask = pm_obj.forward(PM_RI, WL, sigma=0.6)

sphere1 = scene.shapes.Sphere(torch.tensor((20e-6, 35e-6, 4e-6)).float(), torch.tensor((5e-6)), torch.tensor(1.5), softness=1e-12)
sphere2 = scene.shapes.Sphere(torch.tensor((20e-6, 10e-6, 5e-6)).float(), torch.tensor((3e-6)), torch.tensor(1.3), softness=1e-12)
    
grid = scene.grid.Grid(shape, spacing)
vol = scene.volume.Volume(grid, N_BG) # volume should probably inherit from grid instead
vol.add([sphere1, sphere2])

vol_prop = optics.propagator.BeamPropMethod(WL, spacing, shape, padding=0)

# plt.imshow(vol.forward()[:, :, 50].detach())
# plt.colorbar()
# plt.show()

exit_field = vol_prop.forward(torch.ones([500, 500]).type(torch.complex64), vol.forward(), N_BG)
    
prop_obj_space = optics.propagator.Freespace(WL/N_BG, spacing[:2], shape[:2], padding=PAD)
prop_img_space = optics.propagator.Freespace(WL, [x*20 for x in spacing[:2]], shape[:2], padding=PAD)

cwfs_baisc = optics.system.OpticalSystem((
        prop_obj_space.set_params(shape[2]*spacing[2]/2 - FOC_PLANE_VAR, direction='backward'), 
        lens.low_pass_filter(NA), 
        prop_img_space.set_params(dist=DIST_PM_IM, direction='backward'),
        optics.utils.Modulator(mask),
        prop_img_space.set_params(dist=DIST_PM_IM)
    ))

ref_field = cwfs_baisc(torch.ones([500, 500]).type(torch.complex64))
cwfs_baisc.fields.clear()
spn_field = cwfs_baisc(exit_field.type(torch.complex64))


fig, axs = plt.subplots(1, 2, figsize=(15, 5))

cm0 = axs[0].imshow(ref_field.abs().detach()**2, cmap='gray', vmin=0, vmax=5)
plt.colorbar(cm0, ax=axs[0])

cm1 = axs[1].imshow(spn_field.abs().detach()**2, cmap='gray', vmin=0, vmax=5)
plt.colorbar(cm1, ax=axs[1])

plt.show()

print(cwfs_baisc.elements, len(cwfs_baisc.fields))

# opd = grad_unwrap(cwfs_baisc.fields[2])*2*torch.pi/WL
# gradients = utils.grad_optr(opd)*scaling

# use pixel integrator
# use blur for PSC
    
# class _CodedWFSForwardModel(Geometry):
    # def __init__(self, wavelength, grid_shape, spatial_resolution, n_background, device='cpu', grad=False, grid=None, padding=256, 
    #              im_to_ob_space_scale=20, digital_px_size=6e-6, pad_mode='edge'):
    #     super().__init__(grid_shape, spatial_resolution, n_background, device=device, grad=grad, grid=grid)
        
    #     self.wl = wavelength
    #     self.xyz_sup = [self.nx*self.dx, self.ny*self.dy, self.nz*self.dz]
    #     self.pad = padding
    #     self.pad_mode = pad_mode
    #     self.im_to_ob_space_scale = im_to_ob_space_scale
    #     self.im_space_res = [self.im_to_ob_space_scale*self.dx, self.im_to_ob_space_scale*self.dy]
    #     self.sum_size = torch.round(digital_px_size/(self.im_to_ob_space_scale*self.dx))
        
    #     print(f'Digital pixel size {self.sum_size*self.im_space_res[0]*1e6:.2f} um, but desired {digital_px_size*1e6:.2f} um')
        
    # def forward(self, lens, dist_m_im, phase_mask, source_field=None, focus_plane_var=0., digital=True, gradient_median_kernel_size=3):
    #     # main function that does everything
    #     spatial_resolution = [self.dx, self.dy, self.dz]
    #     self.dist_m_im = dist_m_im
        
    #     if source_field is not None:
    #         self.wavefield_focus(source_field)
        
    #     assert hasattr(self, 'wf_focus'), "Propagate field through the grid using method wavefield_focus"
        
    #     # for focal stacks
    #     if focus_plane_var + 1e-9 > 0: # when 0, props by 1e-9 along the dorward direction
    #         output_field = propagator.propagate(self.wf_focus, self.wl, spatial_resolution, torch.abs(focus_plane_var), 
    #                                             padding=self.pad, direction='forward', pad_mode=self.pad_mode)
    #     else:
    #         output_field = propagator.propagate(self.wf_focus, self.wl, spatial_resolution, torch.abs(focus_plane_var), 
    #                                             padding=self.pad, direction='backward', pad_mode=self.pad_mode)
        
    #     assert hasattr(self, 'blur'), "Control defocus specific specimen spatial partial coherence with method PSC_approximator"
    #     output_field = self.blur(output_field.real.unsqueeze(0)).squeeze() + 1j*self.blur(output_field.imag.unsqueeze(0)).squeeze()
        
    #     assert hasattr(self, 'resize'), "Relative resize field with eff_mag_operator: multiples of 20 supported"
    #     output_field = self.resize(output_field.unsqueeze(0).real).squeeze() + 1j*self.resize(output_field.unsqueeze(0).imag).squeeze()
        
    #     # pupil function and lens imaging for NA based low-pass filtering and etc.
    #     # update lens and call forward again
    #     if(not isinstance(lens, ObjImgMap)): # low-pass-filtering after resizing to remove aliasing introduced by resizing
    #         raise ValueError("Lens must be an instance of class ObjImgMap")
    #     output_field = lens.forward(output_field)
        
    #     # prop to phase mask plane
    #     self.field_mask_plane = propagator.propagate(output_field, self.wl, self.im_space_res, self.dist_m_im, 
    #                                                  padding=self.pad, direction='backward', pad_mode=self.pad_mode)

    #     # mask modulation and prop to image plane
    #     obj_field = propagator.propagate(self.field_mask_plane*phase_mask, self.wl, self.im_space_res, self.dist_m_im, 
    #                                      padding=self.pad, direction='forward', pad_mode=self.pad_mode)    
    #     ref_field = propagator.propagate(phase_mask, self.wl, self.im_space_res, self.dist_m_im, 
    #                                      padding=self.pad, direction='forward', pad_mode=self.pad_mode)
        
    #     if self.sum_size > 1:
    #         conv2d = torch.nn.Conv2d(1, 1, self.sum_size.int().item(), self.sum_size.int().item(), bias=False)
    #         conv2d.weight = torch.nn.Parameter(torch.ones([self.sum_size.int().item(), self.sum_size.int().item()], dtype=torch.complex64).unsqueeze(0).unsqueeze(0))
    #         conv2d.weight.requires_grad = False
            
    #         obj_field = conv2d(obj_field.unsqueeze(0)).squeeze()
    #         ref_field = conv2d(ref_field.unsqueeze(0)).squeeze()
        
    #     gt_grad = self.get_gradeint_fields(gradient_median_kernel_size)
        
    #     return ref_field, obj_field, gt_grad
    
    # def wavefield_focus(self, source_field: Tensor = None):
    #     spatial_resolution = [self.dx, self.dy, self.dz]
        
    #     if source_field is None:
    #         source_field = torch.ones([self.nx, self.ny], dtype=torch.complex64) 
    #     # Take the source field at z=0., prop through self.grid, and retrun field at z=self.xyz_sup[2]
    #     output_field = propagator.propagate_beam_vol(source_field, self.grid, self.n_bg, self.wl, spatial_resolution, 
    #                                                  padding=self.pad*0, pad_mode=self.pad_mode) # setting pad to 0 increases speed
    #     self.wf_focus = propagator.propagate(output_field, self.wl/self.n_bg, spatial_resolution, self.xyz_sup[2]/2, 
    #                                          padding=self.pad, direction='backward', pad_mode=self.pad_mode)
        
    #     return self.wf_focus   
        
    # # can add the fourier of this Gaussian to the lens' phase
    # def PSC_approximator(self, const_sigma=1e-2, defocus_sigma=10e-2, defocus_dist=0.):
    #     sigma = const_sigma + defocus_sigma*torch.abs(defocus_dist)
    #     kernel_size = 2*int(4.*sigma + 0.5) + 1
    #     self.blur = GaussianBlur(kernel_size, sigma=sigma.item())
        
    # def eff_mag_operator(self, magnification): # effective magnification operator
    #     eff_mag = int(magnification/self.im_to_ob_space_scale)
    #     self.resize = Resize(size=self.nx*eff_mag, interpolation=torchvision.transforms.InterpolationMode.BILINEAR, antialias=True)
        
    # def get_gradeint_fields(self, median_kernel_size):
        
    #     k = 2*torch.pi/self.wl
        
    #     # gradient in pixels
    #     flows_01 = torch_grad_optr(torch.angle(self.field_mask_plane))
        
    #     # remove 2pi peaks at phase wrapping junctions
    #     flow_0 = median_filter_2d(flows_01[0].unsqueeze(0).unsqueeze(0), median_kernel_size).squeeze()
    #     flow_1 = median_filter_2d(flows_01[1].unsqueeze(0).unsqueeze(0), median_kernel_size).squeeze()
        
    #     # OPD gradients in image sapce units
    #     flow_0 = flow_0/(self.im_space_res[0])/k*self.dist_m_im
    #     flow_1 = flow_1/(self.im_space_res[1])/k*self.dist_m_im
        
    #     if self.sum_size > 1:
    #         conv2d = torch.nn.Conv2d(1, 1, self.sum_size.int().item(), self.sum_size.int().item(), bias=False)
    #         conv2d.weight = torch.nn.Parameter(torch.ones([self.sum_size.int().item(), self.sum_size.int().item()]).unsqueeze(0).unsqueeze(0)/self.sum_size**2)
    #         conv2d.weight.requires_grad = False
            
    #         flow_0 = conv2d(flow_0.unsqueeze(0)).squeeze()
    #         flow_1 = conv2d(flow_1.unsqueeze(0)).squeeze()
            
    #     flow_0 = flow_0/(self.im_space_res[0]*self.sum_size)
    #     flow_1 = flow_1/(self.im_space_res[1]*self.sum_size)
        
    #     return [flow_0, flow_1]
    

if __name__=='__main__':
    pass