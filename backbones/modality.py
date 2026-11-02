import torch
import torch.nn as nn
import torch.nn.functional as F

from .timmfr import LoRaLin


class ModulatedLoRaLin(nn.Module):
    """
    Rank-space modality modulation for a factorised linear layer W = U V:
        y = U (s_m * (V x) + t_m) + b
    where s_m (scale, init 1) and t_m (shift, init 0) are per-modality vectors of length `rank`,
    so the layer starts as the pretrained LoRaLin. For a fixed modality s_m folds into U and
    U t_m into b, giving zero inference overhead.
    """
    def __init__(self, lowrank, num_modalities=2):
        super().__init__()
        self.linear1 = lowrank.linear1
        self.linear2 = lowrank.linear2
        self.scale = nn.Parameter(torch.ones(num_modalities, lowrank.rank))
        self.shift = nn.Parameter(torch.zeros(num_modalities, lowrank.rank))
        self.modality = None  # LongTensor [B] set by set_modality()

    def forward(self, x):
        h = self.linear1(x)
        if self.modality is not None:
            shape = (h.shape[0], *([1] * (h.dim() - 2)), h.shape[-1])
            h = h * self.scale[self.modality].view(shape) + self.shift[self.modality].view(shape)
        return self.linear2(h)


class ModulatedLayerNorm(nn.Module):
    """
    Modality-specific affine parameters for a LayerNorm (nn.LayerNorm or timm's channels-first
    LayerNorm2d): y = scale_m * norm(x) + shift_m, every row initialised to the pretrained
    weight/bias. Without a modality set, row 0 (visible) is used.
    """
    def __init__(self, norm, num_modalities=2):
        super().__init__()
        self.normalized_shape = norm.normalized_shape
        self.eps = norm.eps
        self.channels_first = type(norm).__name__ == "LayerNorm2d"
        self.scale = nn.Parameter(norm.weight.detach().clone().repeat(num_modalities, 1))
        self.shift = nn.Parameter(norm.bias.detach().clone().repeat(num_modalities, 1))
        self.modality = None

    def forward(self, x):
        if self.channels_first:
            x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, None, None, self.eps)
        modality = self.modality if self.modality is not None else torch.zeros(
            x.shape[0], dtype=torch.long, device=x.device)
        shape = (x.shape[0], *([1] * (x.dim() - 2)), x.shape[-1])
        x = x * self.scale[modality].view(shape) + self.shift[modality].view(shape)
        if self.channels_first:
            x = x.permute(0, 3, 1, 2)
        return x


MODULATED = (ModulatedLoRaLin, ModulatedLayerNorm)


def add_modality_modulation(model, num_modalities=2, targets=("lowrank",)):
    """targets: "lowrank" (rank-space modulation of LoRaLin) and/or "norm" (LayerNorm affine)"""
    for name, module in model.named_children():
        if "lowrank" in targets and isinstance(module, LoRaLin):
            setattr(model, name, ModulatedLoRaLin(module, num_modalities))
        elif "norm" in targets and isinstance(module, nn.LayerNorm) and module.elementwise_affine:
            setattr(model, name, ModulatedLayerNorm(module, num_modalities))
        else:
            add_modality_modulation(module, num_modalities, targets)
    return model


def set_modality(model, modality):
    """modality: LongTensor [B] of modality ids, or None to disable modulation"""
    for module in model.modules():
        if isinstance(module, MODULATED):
            module.modality = modality


def modulation_parameters(model, kinds=("scale", "shift"), targets=MODULATED):
    return [getattr(m, k) for m in model.modules() if isinstance(m, targets) for k in kinds]
