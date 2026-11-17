from torch import nn


class OpticalSystem(nn.Module):
    """Apply optical elements in sequence.

    Parameters
    ----------
    elements : sequence of nn.Module
        Elements applied in order; each maps a complex field ``(..., H, W)`` to the next.
    record : bool
        If ``True``, ``fields`` holds the detached output of every element from the most
        recent call. Off by default so repeated calls don't keep fields in memory.
    """

    def __init__(self, elements, record: bool = False):
        super().__init__()
        self.elements = nn.ModuleList(elements)
        self.record = record
        self.fields = []

    def output_spacing(self, spacing):
        """Sample spacing after the last element, for an input with ``spacing`` (metres).

        Elements that change the sampling define ``output_spacing``; the others keep it.
        """
        for element in self.elements:
            if hasattr(element, "output_spacing"):
                spacing = element.output_spacing(spacing)
        return spacing

    def forward(self, field):
        # A new list each call: only the latest pass is kept, and a list a caller took
        # from an earlier call is left untouched.
        self.fields = []
        for element in self.elements:
            field = element(field)  # __call__, so module hooks run
            if self.record:
                self.fields.append(field.detach())
        return field
