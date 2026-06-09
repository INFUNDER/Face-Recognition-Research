"""
===============================================================================
Author: Anjith George
Institution: Idiap Research Institute, Martigny, Switzerland.

Copyright (C) 2023 Anjith George

This software is distributed under the terms described in the LICENSE file 
located in the parent directory of this source code repository. 

For inquiries, please contact the author at anjith.george@idiap.ch
===============================================================================
"""



import timm
import torch
import torch.nn as nn
import math

class LoRaLin(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x

def replace_linear_with_lowrank_recursive_2(model, rank_ratio=0.2, replace_all=True):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and (replace_all or not any(h in name.lower() for h in ['head', 'fc', 'classifier'])):
            in_features = module.in_features
            out_features = module.out_features
            rank = max(2,int(min(in_features, out_features) * rank_ratio))
            bias=False
            if module.bias is not None:
                bias=True
            lowrank_module = LoRaLin(in_features, out_features, rank, bias)

            setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_recursive_2(module, rank_ratio, replace_all)

def replace_linear_with_lowrank_2(model, rank_ratio=0.2, replace_all=True):
    replace_linear_with_lowrank_recursive_2(model, rank_ratio, replace_all)
    return model

def replace_linear_with_lowrank_nas_recursive(model, rank_ratios, current_idx, replace_all=True):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear) and (replace_all or not any(h in name.lower() for h in ['head', 'fc', 'classifier'])):
            if current_idx[0] < len(rank_ratios):
                rank_ratio = rank_ratios[current_idx[0]]
                current_idx[0] += 1
                in_features = module.in_features
                out_features = module.out_features
                rank = max(2,int(min(in_features, out_features) * rank_ratio))
                bias=False
                if module.bias is not None:
                    bias=True
                lowrank_module = LoRaLin(in_features, out_features, rank, bias)
                setattr(model, name, lowrank_module)
        else:
            replace_linear_with_lowrank_nas_recursive(module, rank_ratios, current_idx, replace_all)

def replace_linear_with_lowrank_nas(model, rank_ratios, replace_all=True):
    current_idx = [0]
    replace_linear_with_lowrank_nas_recursive(model, rank_ratios, current_idx, replace_all)
    return model

class FakeINT4Linear(nn.Module):
    """
    Simulates INT4 weight-only quantization. 
    It dynamically quantizes the weights to 4-bit representation (16 levels) 
    during the forward pass, serving as a placeholder for actual INT4 hardware kernels.
    """
    def __init__(self, linear_layer):
        super(FakeINT4Linear, self).__init__()
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

    def forward(self, x):
        # Symmetric 4-bit quantization (values from -8 to 7)
        # Find the max absolute value for scaling
        w_max = self.weight.abs().max()
        scale = w_max / 7.0
        # Quantize and dequantize
        quantized_weight = torch.round(self.weight / scale).clamp(-8, 7)
        dequantized_weight = quantized_weight * scale
        return torch.nn.functional.linear(x, dequantized_weight, self.bias)

def replace_linear_with_int4_recursive(model):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            int4_module = FakeINT4Linear(module)
            setattr(model, name, int4_module)
        else:
            replace_linear_with_int4_recursive(module)

def replace_linear_with_int4(model):
    replace_linear_with_int4_recursive(model)
    return model

        
class TimmFRWrapperV2(nn.Module):
    """
    Wraps timm model
    """
    def __init__(self, model_name='edgenext_x_small', featdim=512, batchnorm=False):
        super().__init__()
        self.featdim = featdim
        self.model_name = model_name
        
        self.model = timm.create_model(self.model_name)
        self.model.reset_classifier(self.featdim)

    def forward(self, x):
        x = self.model(x)
        return x


def get_timmfrv2(model_name, **kwargs):
    """
    Create an instance of TimmFRWrapperV2 with the specified `model_name` and additional arguments passed as `kwargs`.
    """
    return TimmFRWrapperV2(model_name=model_name, **kwargs)

