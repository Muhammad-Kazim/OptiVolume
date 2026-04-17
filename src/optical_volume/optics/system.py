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
    
    import torch
    from matplotlib import pyplot as plt
    
    
    # example: functioning
    WL = 500e-9
    spacing = [10e-6, 10e-6]
    shape = [100, 100]
    NA = 1e-2
    dist = 10e-3
    padding = 1000
    
    x = (torch.arange(100)-50)*spacing[0]
    X, Y = torch.meshgrid(x, x, indexing='ij')
    
    wavefield = torch.where(X**2 + Y**2 < (300e-6)**2, 1+0j, 0+0j)
    
    prop = Freespace(WL, spacing, shape, padding=padding)
    lens = ObjImgMap(WL, spacing, shape)
    
    sysA = OpticalSystem((
        prop.set_params(dist=50e-3), lens.low_pass_filter(NA), prop.set_params(dist=10e-3)
    ))
    
    field = sysA(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    sysA = OpticalSystem((
        prop.set_params(dist=50e-3), lens.low_pass_filter(NA)
    ))
    
    field = sysA(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    sysA = OpticalSystem((
        prop.set_params(dist=50e-3), lens.low_pass_filter(NA), prop.set_params(dist=10e-3), prop.set_params(dist=10e-3, direction='backward')
    ))
    
    field = sysA(wavefield)
    
    plt.imshow(field.abs())
    plt.colorbar()
    plt.show()
    
    print(sysA, sysA.parameters())