import os
import sys
import torch
import torchvision
from torchvision.transforms import GaussianBlur, Resize

from typing import Optional, Tuple, List
from torch import nn, Tensor

from optical_volume import optics
from optical_volume import scene
from optical_volume import utils
from optical_volume import pipelines

from matplotlib import pyplot as plt
from tifffile import tifffile

import matplotlib.animation as animation

WL = 500e-9
OBJ_SPACING = [150e-9, 150e-9, 100e-9]
SHAPE = [600, 600, 100]
NA = 0.75
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

sphere1 = scene.shapes.Sphere(torch.tensor((60e-6, 20e-6, 5e-6)).float(), torch.tensor((7e-6)), torch.tensor(1.46))
sphere2 = scene.shapes.Sphere(torch.tensor((50e-6, 26e-6, 5e-6)).float(), torch.tensor((8e-6)), torch.tensor(1.46))
sphere3 = scene.shapes.Sphere(torch.tensor((40e-6, 33e-6, 5e-6)).float(), torch.tensor((9e-6)), torch.tensor(1.46))
sphere4 = scene.shapes.Sphere(torch.tensor((32e-6, 40e-6, 5e-6)).float(), torch.tensor((10e-6)), torch.tensor(1.46))
sphere5 = scene.shapes.Sphere(torch.tensor((26e-6, 50e-6, 5e-6)).float(), torch.tensor((9e-6)), torch.tensor(1.46))
sphere6 = scene.shapes.Sphere(torch.tensor((22e-6, 57e-6, 5e-6)).float(), torch.tensor((8e-6)), torch.tensor(1.46))
sphere7 = scene.shapes.Sphere(torch.tensor((19e-6, 66e-6, 5e-6)).float(), torch.tensor((7e-6)), torch.tensor(1.46))
    
vol = scene.CollectionVolume(SHAPE, OBJ_SPACING, N_BG)
vol.add([sphere1, sphere2, sphere3, sphere4, sphere5, sphere6, sphere7])


tx = [1.05, 1.075, 1.14, 1.2, 1.14, 1.2]
ty = [1.1, 1.2, 1.28, 1.4, 1.28, 1.2]

with torch.no_grad():
    vols = [scene.CollectionVolume(SHAPE, OBJ_SPACING, N_BG) for _ in range(6)]

    for t, vol_t in enumerate(vols):
        for shape in vol.shapes:
            xcenter = (shape.center[0].detach().clone() - 45e-6)*tx[t]+ 45e-6
            ycenter = (shape.center[1].detach().clone() - 45e-6)*ty[t] + 45e-6
            zcenter = shape.center[2].detach().clone()
            radius = shape.radius.detach().clone()
            RI = shape.RI.detach().clone() - 0.04*t # torch.randn(1)[0]/10
            
            vol_t.add([scene.shapes.Sphere(torch.tensor((xcenter, ycenter, 5e-6)).float(), radius, RI)])
            
            # break

# plt.imshow(vol.forward()[:, :, 50].detach())
# plt.colorbar()
# plt.show()

# projs = []
# projs.append(vol.forward()[:, :, 50].detach())
# for i in range(len(vols)):
#     projs.append(vols[i].forward()[:, :, 50].detach())

# fig, ax = plt.subplots()
# im = ax.imshow(projs[0], vmin=1.3, vmax=1.6, cmap='gray')

# c = 0
# def update(c):
#     im.set_data(projs[c])
#     c += 1
#     return (im)

# ani = animation.FuncAnimation(fig=fig, func=update, frames=6, interval=150)
# plt.show()

vol_prop = optics.VolumePropagator(WL, OBJ_SPACING, N_BG)

# vol_prop.set_source_field(100*torch.ones([SHAPE[0], SHAPE[1]], dtype=torch.complex64, device=DEVICE))

# Partial coherence: no blur for now. Set sigma0 (m) or angular_spread (rad) to enable;
# blurring the field here smooths the object's projection, not the mask pattern.
psc_approx = optics.PartialCoherenceBlur(OBJ_SPACING[:2], sigma0=0., angular_spread=0., defocus=FOC_PLANE_VAR)
resample = optics.Resample(EFF_MAG)  # object field sampled at OBJ_SPACING / EFF_MAG
mag_obj = optics.Magnification(MAG)  # physical: spacing x MAG -> IMG_SPACING, amplitude / MAG
lens = optics.F4ImagingSystem(WL, [x/EFF_MAG for x in OBJ_SPACING], IMG_SHAPE, NA)

prop_obj_space = optics.FreeSpacePropagator(WL/N_BG, OBJ_SPACING[:2], SHAPE[2]*OBJ_SPACING[2]/2 - FOC_PLANE_VAR, direction='backward', padding=PAD)

phase_camera = pipelines.CodedPhaseCamV3(wavelength=WL, spacing=IMG_SPACING, shape=IMG_SHAPE, padding=PAD, pad_mode=PAD_MODE)

image_formation = optics.system.OpticalSystem((
        prop_obj_space, # centring + displacing focus plalne
        psc_approx,
        resample, # finer sampling in object space
        lens, # imaging the wavefield
        mag_obj, # magnification MAG into image space
        phase_camera
    ))

# vol.to(DEVICE)
image_formation.to(DEVICE)
img_ref = image_formation(torch.ones(SHAPE[:2]).type(torch.complex64).to(DEVICE)).detach().to('cpu')

spns = []

with torch.no_grad():
    for vol_t in vols:
        
        exit_field = vol_prop.forward(vol_t.forward())

        image_formation.fields.clear()
        spns.append(image_formation(exit_field.type(torch.complex64)).detach().to('cpu'))


# fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# cm0 = axs[0].imshow(img_ref/img_ref.max(), cmap='gray', vmin=0)
# plt.colorbar(cm0, ax=axs[0])

# cm1 = axs[1].imshow(spns[0]/img_ref.max(), cmap='gray', vmin=0)
# plt.colorbar(cm1, ax=axs[1])

# plt.show()

fig, ax = plt.subplots()
im = ax.imshow(spns[0]/img_ref.max(), vmax=0.9, cmap='gray')

c = 0
def update(c):
    im.set_data(spns[c]/img_ref.max())
    c += 1
    return (im)

ani = animation.FuncAnimation(fig=fig, func=update, frames=6, interval=150)
plt.show()
