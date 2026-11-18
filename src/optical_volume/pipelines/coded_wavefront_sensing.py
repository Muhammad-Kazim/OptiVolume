import torch

from typing import Optional, Tuple, List
from torch import nn, Tensor

from optical_volume import optics
from optical_volume import scene
from optical_volume import utils

class CodedPhaseCamV3(torch.nn.Module):
    def __init__(self, wavelength: float, spacing: Tuple[float, float], shape: Tuple[int, int],
                 padding: Optional[int] = 512, pad_mode: str = 'edge',softness=1e-12,
                 integrate: str = 'intensity'):
        """Coded wavefront sensor: phase mask in front of a camera at the image plane.

        Parameters
        ----------
        wavelength : float
            Vacuum wavelength, in metres.
        spacing : tuple of float
            Image-space sampling ``(dx, dy)`` in metres.
        shape : tuple of int
            Image-space field size ``(H, W)``.
        padding : int, optional
            Lateral padding of the propagators, in samples.
        pad_mode : str
            torchvision padding mode of the propagators.
        integrate : {'intensity', 'field'}
            ``'intensity'`` (default): each camera pixel sums ``|E|**2`` (a physical camera).
            ``'field'``: sums the complex field first and returns ``|sum E|**2`` (the
            behaviour before CS4b).
        """
        super().__init__()
        if integrate not in ('intensity', 'field'):
            raise ValueError(f"integrate must be 'intensity' or 'field', got {integrate!r}")
        self.integrate = integrate
        # shape: img space

        # fixed sensor parameters
        TILE_LEN = 12e-6
        PM_HEIGHT = 570e-9
        PM_RI = 1.46
        PM_SIG = 1.1
        DIST_PM_IM = 2.5e-3
        DIGITAL_PX = 6e-6

        pm_obj = optics.BinaryPhaseMask(wavelength, TILE_LEN, spacing, shape, PM_HEIGHT, PM_SIG, PM_RI)
        px_int_obj = optics.PixelIntegrator(pixel_size=DIGITAL_PX, spacing=spacing)
        # The sensor sits at the image plane and the mask DIST_PM_IM in front of it: propagate
        # back to the mask, modulate, propagate forward again. Two propagators share one
        # distance Parameter, so training it moves the mask as one physical distance.
        prop_to_mask = optics.FreeSpacePropagator(wavelength, spacing, DIST_PM_IM, direction='backward',
                                                  padding=padding, pad_mode=pad_mode)
        prop_to_sensor = optics.FreeSpacePropagator(wavelength, spacing, DIST_PM_IM, direction='forward',
                                                    padding=padding, pad_mode=pad_mode)
        prop_to_sensor.distance = prop_to_mask.distance

        detector = (optics.Intensity(), px_int_obj) if integrate == 'intensity' else (px_int_obj,)
        self.pcam_FM = optics.system.OpticalSystem((
            prop_to_mask, # image plane -> mask plane
            pm_obj, # applying the mask
            prop_to_sensor, # mask plane -> sensor (image plane)
            *detector, # camera pixels
            ))

    def forward(self, field: Tensor):
        out = self.pcam_FM(field)
        if self.integrate == 'field':
            out = out.abs()**2

        return out
