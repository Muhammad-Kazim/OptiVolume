from torch import nn


class Modulator(nn.Module):
    def __init__(self, operator):
        super().__init__()
        
        self.operator = operator
        
    def forward(self, field):
        return self.operator*field