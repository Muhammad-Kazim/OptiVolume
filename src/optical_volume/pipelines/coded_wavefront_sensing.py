import torch

from typing import Optional, Tuple, List
from torch import nn, Tensor

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
        field = self.pcam_FM(field)
        intensity = field.abs()**2

        return intensity
