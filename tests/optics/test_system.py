"""OpticalSystem and Transmission: memory, hooks and registered state (C1, C3)."""
import torch
from torch import nn

from optical_volume import optics

WL = 500e-9
SPACING = (1e-6, 1e-6)


def _system(record=False):
    return optics.OpticalSystem((
        optics.AngularSpectrum(WL, SPACING, 10e-6),
        optics.Transmission(torch.full((16, 16), 0.5, dtype=torch.complex64)),
        optics.PixelIntegrator(2),
    ), record=record)


def test_system_no_memory_growth():
    field = torch.ones(16, 16, dtype=torch.complex64)
    system = _system()
    for _ in range(10):
        system(field)
    assert system.fields == []

    system = _system(record=True)
    for _ in range(10):
        out = system(field)
    assert len(system.fields) == len(system.elements)
    torch.testing.assert_close(system.fields[-1], out)


def test_system_runs_module_hooks():
    system = _system()
    calls = []
    system.elements[1].register_forward_hook(lambda module, inp, out: calls.append(module))
    system(torch.ones(16, 16, dtype=torch.complex64))
    assert calls == [system.elements[1]]


def test_modulator_registers_operator():
    op = torch.ones(4, 4, dtype=torch.complex64)
    assert "operator" in dict(optics.Transmission(op).named_buffers())
    assert "operator" in dict(optics.Transmission(op, trainable=True).named_parameters())
    param = nn.Parameter(op.clone())
    assert optics.Transmission(param).operator is param
