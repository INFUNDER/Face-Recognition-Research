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
from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2, replace_linear_with_int4

import torch

def get_model(name, **kwargs):

    if name=='edgeface_xs_gamma_06':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
    elif name=='edgeface_xs_q':
        model= get_timmfrv2('edgenext_x_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name=='edgeface_xs_q4':
        model= get_timmfrv2('edgenext_x_small', batchnorm=False)
        return replace_linear_with_int4(model)
    elif name=='edgeface_xs_gamma_06_q4':
        model = replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
        return replace_linear_with_int4(model)
    elif name=='edgeface_xs_gamma_06_q':
        model = replace_linear_with_lowrank_2(get_timmfrv2('edgenext_x_small', batchnorm=False), rank_ratio=0.6)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif  name=='edgeface_xxs':
        return get_timmfrv2('edgenext_xx_small', batchnorm=False)
    elif  name=='edgeface_base':
        return get_timmfrv2('edgenext_base', batchnorm=False)
    elif name=='edgeface_xxs_q':
        model=get_timmfrv2('edgenext_xx_small', batchnorm=False)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model   
    elif name=='edgeface_xxs_q4':
        model=get_timmfrv2('edgenext_xx_small', batchnorm=False)
        return replace_linear_with_int4(model)
    elif name=='edgeface_s_gamma_05':
        return replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
    elif name=='edgeface_s_gamma_05_q4':
        model = replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
        return replace_linear_with_int4(model)
    elif name=='edgeface_s_gamma_05_q':
        model = replace_linear_with_lowrank_2(get_timmfrv2('edgenext_small', batchnorm=False), rank_ratio=0.5)
        model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        return model
    elif name=='edgeface_s_q4':
        model = get_timmfrv2('edgenext_small', batchnorm=False)
        return replace_linear_with_int4(model)
    elif name.startswith('edgeface_timm_'):
        parts = name.split('_gamma_')
        if len(parts) == 2:
            model_name = parts[0].replace('edgeface_timm_', '')
            gamma = float(parts[1].replace('_', '.'))
            return replace_linear_with_lowrank_2(get_timmfrv2(model_name, batchnorm=False), rank_ratio=gamma)
        else:
            model_name = name.replace('edgeface_timm_', '')
            return get_timmfrv2(model_name, batchnorm=False)
    else:
        raise ValueError(f"Model {name} not found or not supported.")
