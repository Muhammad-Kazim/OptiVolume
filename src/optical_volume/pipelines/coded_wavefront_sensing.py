import os
import sys
import torch
import torchvision
from torchvision.transforms import GaussianBlur, Resize
# from py_wave_propagator import torch_volume_prop as propagator

from typing import Optional, Tuple, List
from torch import nn, Tensor

# sys.path.insert(0, '/'.join(os.getcwd().split('/')[:-1]))

from optical_volume import optics
from optical_volume import scene
from optical_volume import utils

class CodedPhaseCamV3(torch.nn.Module):
    def __init__(self, wavelength: float, spacing: Tuple[float, float], shape: Tuple[int, int],
                 padding: Optional[int] = 512, pad_mode: str = 'edge',softness=1e-12):
        super().__init__()
        # shape: img space
        
        # fixed sensor parameters
        TILE_LEN = 12e-6
        PM_HEIGHT = 570e-9
        PM_RI = 1.46
        PM_SIG = 1.1
        DIST_PM_IM = 2.5e-3
        DIGITAL_PX = 6e-6
       
        SUM_PATCH = int(DIGITAL_PX/spacing[0]) # spacing is image space sampling
        
        pm_obj = optics.BinaryPhaseMask(wavelength, TILE_LEN, spacing, shape, PM_HEIGHT, PM_SIG, PM_RI)
        px_int_obj = optics.utils.PixelIntegrator(SUM_PATCH)
        prop_img_space = optics.FreeSpacePropagator(wavelength, spacing, DIST_PM_IM, direction='backward', 
                                                    padding=padding, pad_mode=pad_mode)
        

        self.pcam_FM = optics.system.OpticalSystem((
            prop_img_space, # prop to mask plane
            pm_obj, # applying the mask
            prop_img_space.set_direction('forward'), # prop to image plane
            px_int_obj # add complex values incident on each pixel patch
            ))
        
    def forward(self, field: Tensor):
        # self.pcam_FM.to(field.get_device)
        self.pcam_FM.fields.clear()
        
        field = self.pcam_FM(field)
        intensity = field.abs()**2
        
        return intensity

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
    from matplotlib import pyplot as plt
    from tifffile import tifffile
    
    WL = 500e-9
    OBJ_SPACING = [150e-9, 150e-9, 100e-9]
    SHAPE = [600, 600, 100]
    NA = 0.75
    # DIST_PM_IM = 1.43e-3

    # TILE_LEN = 12e-6
    # PM_HEIGHT = 200e-9
    # PM_RI = 1.46
    # PM_SIG = 0.3

    N_BG = 1.51
    PAD = 512
    PAD_MODE = 'edge'
    FOC_PLANE_VAR = 0.1e-8

    MAG = 40
    OBJ_IMG_FACTOR = 20
    EFF_MAG = int(40/OBJ_IMG_FACTOR)

    IMG_SPACING = [OBJ_SPACING[i]*OBJ_IMG_FACTOR for i in range(2)]
    IMG_SHAPE = [x*EFF_MAG for x in SHAPE]

    DEVICE = 'cpu'


    sphere1 = scene.shapes.Sphere(torch.tensor((20e-6, 45e-6, 3e-6)).float(), torch.tensor((5e-6)), torch.tensor(1.46), softness=1e-12)
    sphere2 = scene.shapes.Sphere(torch.tensor((30e-6, 20e-6, 5e-6)).float(), torch.tensor((3e-6)), torch.tensor(1.46), softness=1e-12)
        
    vol = scene.CollectionVolume(SHAPE, OBJ_SPACING, N_BG)
    vol.add([sphere1, sphere2])

    vol_prop = optics.VolumePropagator(WL, OBJ_SPACING, N_BG)
    
    # vol_prop.set_source_field(100*torch.ones([SHAPE[0], SHAPE[1]], dtype=torch.complex64, device=DEVICE))

    # plt.imshow(vol.forward()[:, :, 50].detach())
    # plt.colorbar()
    # plt.show()

    psc_approx = optics.utils.GaussBlur(defocus_dist=FOC_PLANE_VAR)
    mag_obj = optics.utils.Magnification(torch.tensor(MAG/OBJ_IMG_FACTOR))
    lens = optics.F4ImagingSystem(WL, [x/EFF_MAG for x in OBJ_SPACING], IMG_SHAPE, NA)

    prop_obj_space = optics.FreeSpacePropagator(WL/N_BG, OBJ_SPACING[:2], SHAPE[2]*OBJ_SPACING[2]/2 - FOC_PLANE_VAR, direction='backward', padding=PAD)

    phase_camera = CodedPhaseCamV3(wavelength=WL, spacing=IMG_SPACING, shape=IMG_SHAPE, padding=PAD, pad_mode=PAD_MODE)

    image_formation = optics.system.OpticalSystem((
            prop_obj_space, # centring + displacing focus plalne
            psc_approx,
            mag_obj, # magnified by MAG/OBJ_IMG_FACTOR
            lens, # imaging the wavefield
            phase_camera
        ))


    vol.to(DEVICE)
    image_formation.to(DEVICE)

    exit_field = vol_prop.forward(vol.forward())

    img_ref = image_formation(torch.ones(SHAPE[:2]).type(torch.complex64).to(DEVICE)).detach().to('cpu')
    
    image_formation.fields.clear()
    img_spn = image_formation(exit_field.type(torch.complex64)).detach().to('cpu')
    
    
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    cm0 = axs[0].imshow(img_ref/img_ref.max(), cmap='gray', vmin=0)
    plt.colorbar(cm0, ax=axs[0])

    cm1 = axs[1].imshow(img_spn/img_ref.max(), cmap='gray', vmin=0)
    plt.colorbar(cm1, ax=axs[1])

    plt.show()

    im_real = torch.tensor(tifffile.imread('/home/syedkazim/sciebo - Kazim, Syed Muhammad (u491036@uni-siegen.de)@uni-siegen.sciebo.de/Lab/Projects/2024_Phase_Camera_FM_Design/phase camera/phase_camera_hardware/2026_09_24_PCam_3-1/Image__2026-09-25__09-59-27.tiff')).float()[:1000, :1000]

    plt.hist((img_spn/img_ref.max()).reshape(-1), bins=25, label='Sim', alpha=0.7)
    plt.hist((im_real/im_real.max()).reshape(-1), bins=25, label='Real', alpha=0.7)
    plt.legend()
    plt.show()

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    cm0 = axs[0].imshow(torch.log10(torch.fft.fftshift(torch.fft.fft2(im_real/im_real.max())).abs()), cmap='gray')
    plt.colorbar(cm0, ax=axs[0])

    padded_int = torch.nn.functional.pad(img_ref[350:850, 350:850]/img_ref.max(), [250]*4, "constant", 0)
    spectrum = torch.fft.fftshift(torch.fft.fft2(padded_int)).abs()

    cm1 = axs[1].imshow(torch.log10(spectrum), cmap='gray')
    plt.colorbar(cm1, ax=axs[1])

    plt.show()
    