import torch
import torch.nn as nn
import torch.nn.functional as F


def _fake_quantize(x, scale, zero_point, qmin, qmax):
    # multiply by the inverse scale like fbgemm: the max-abs weight sits exactly on a .5 tie
    q = torch.clamp(torch.round(x * (1.0 / scale)) + zero_point, qmin, qmax)
    return x + ((q - zero_point) * scale - x).detach()


class FakeDynamicInt8Linear(nn.Linear):
    """
    nn.Linear that simulates torch.quantization.quantize_dynamic (fbgemm) in the forward pass:
    weights per-tensor symmetric qint8, inputs per-tensor asymmetric quint8 with reduce_range
    (0..127), both computed on the fly. Gradients pass straight through the rounding.
    Swapped in by changing __class__, so parameters and state_dict keys are unchanged.
    """
    def forward(self, x):
        w_scale = self.weight.detach().abs().max().clamp(min=1e-8) / 127.5
        w = _fake_quantize(self.weight, w_scale, 0, -128, 127)
        x_min = x.detach().min().clamp(max=0)
        x_max = x.detach().max().clamp(min=0)
        x_scale = ((x_max - x_min) / 127).clamp(min=1e-8)
        x_zp = torch.clamp(torch.round(-x_min / x_scale), 0, 127)
        x = _fake_quantize(x, x_scale, x_zp, 0, 127)
        return F.linear(x, w, self.bias)


def enable_fake_int8(model):
    for module in model.modules():
        if type(module) is nn.Linear:
            module.__class__ = FakeDynamicInt8Linear
    return model


def disable_fake_int8(model):
    for module in model.modules():
        if type(module) is FakeDynamicInt8Linear:
            module.__class__ = nn.Linear
    return model
