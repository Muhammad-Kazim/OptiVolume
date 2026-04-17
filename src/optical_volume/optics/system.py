from torch import nn


class OpticalSystem(nn.Module):
    def __init__(self, elements):
        super().__init__()
        self.elements = nn.ModuleList(elements)

    def forward(self, field):
        for element in self.elements:
            field = element.forward(field)
        return field
    
    
if __name__ == "__main__":
    from lens import ObjImgMap
    from phase_mask import PhaseMask
    from propagator import Freespace
    from utils import Modulator
    
    import torch
    from matplotlib import pyplot as plt
    
    
    # example: functioning
    WL = 500e-9
    spacing = [100e-9, 100e-9]
    shape = [500, 500]
    NA = 0.9
    dist = 0.4e-3
    padding = 1000
    TILE_LEN = 0.25e-6
    PM_HEIGHT = 600e-9
    PM_RI = 1.3
    
    x = (torch.arange(shape[0]) - int(shape[0]/2))*spacing[0]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    wavefield = torch.where(X**2 + Y**2 < (10e-6)**2, 1+0j, 0+0j)
    
    prop = Freespace(WL, spacing, shape, padding=padding)
    lens = ObjImgMap(WL, spacing, shape)
    
    sysA = OpticalSystem((
        prop.set_params(dist=5e-3), lens.low_pass_filter(NA), prop.set_params(dist=1e-3)
    ))
    
    field = sysA(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    sysA = OpticalSystem((
        prop.set_params(dist=5e-3), lens.low_pass_filter(NA)
    ))
    
    field = sysA(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    sysA = OpticalSystem((
        prop.set_params(dist=5e-3), lens.low_pass_filter(NA), prop.set_params(dist=1e-3), prop.set_params(dist=1e-3, direction='backward')
    ))
    
    field = sysA(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    pm_obj = PhaseMask(torch.tensor(TILE_LEN), spacing, shape, height=torch.tensor(PM_HEIGHT))
    mask = pm_obj.forward(PM_RI, WL, sigma=0.6)
    
    cwfs_baisc = OpticalSystem((
        prop.set_params(dist=0.1e-3), 
        lens.low_pass_filter(NA), 
        prop.set_params(dist=dist, direction='backward'),
        Modulator(mask),
        prop.set_params(dist=dist)
    ))
    
    field = cwfs_baisc(wavefield)
    
    plt.imshow(field.abs().detach()**2, cmap='gray')
    plt.colorbar()
    plt.show()
    
    print(cwfs_baisc, cwfs_baisc.parameters())