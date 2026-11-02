import math

import torch
import torch.nn as nn

from .timmfr import LoRaLin


def expand_lowrank(model, extra_rank):
    """
    LoRA for factorised layers: append `extra_rank` components to every LoRaLin, W = U V + B A,
    with B zero-initialised so the output is unchanged. The result is still a LoRaLin (of rank
    rank + extra_rank), so INT8 export and evaluation need no special handling.
    Returns [(module, pretrained_rank)].
    """
    expanded = []
    for module in model.modules():
        if isinstance(module, LoRaLin):
            old1, old2, rank = module.linear1, module.linear2, module.rank
            linear1 = nn.Linear(module.in_features, rank + extra_rank, bias=False)
            linear2 = nn.Linear(rank + extra_rank, module.out_features, bias=old2.bias is not None)
            with torch.no_grad():
                nn.init.kaiming_uniform_(linear1.weight, a=math.sqrt(5))
                linear1.weight[:rank] = old1.weight
                linear2.weight.zero_()
                linear2.weight[:, :rank] = old2.weight
                if old2.bias is not None:
                    linear2.bias.copy_(old2.bias)
            module.linear1, module.linear2, module.rank = linear1, linear2, rank + extra_rank
            expanded.append((module, rank))
    return expanded


def freeze_pretrained_components(expanded):
    """Train only the appended LoRA components; returns their parameters."""
    params = []
    for module, rank in expanded:
        module.linear1.weight.requires_grad_(True)
        module.linear2.weight.requires_grad_(True)
        module.linear1.weight.register_hook(lambda g, r=rank: torch.cat([torch.zeros_like(g[:r]), g[r:]]))
        module.linear2.weight.register_hook(lambda g, r=rank: torch.cat([torch.zeros_like(g[:, :r]), g[:, r:]], 1))
        params += [module.linear1.weight, module.linear2.weight]
    return params
