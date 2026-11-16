from torch import nn


class OpticalSystem(nn.Module):
    def __init__(self, elements):
        super().__init__()
        self.elements = nn.ModuleList(elements)
        self.fields = []

    def forward(self, field):
        for element in self.elements:
            field = element.forward(field)
            self.fields.append(field.detach())
        return field
