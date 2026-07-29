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
from .timmfr import get_timmfrv2, replace_linear_with_lowrank_2, replace_linear_with_int4, get_dual_timmfrv2

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
        quantize = None
        clean_name = name
        if name.endswith('_q'):
            quantize = 'q'
            clean_name = name[:-2]
        elif name.endswith('_q4'):
            quantize = 'q4'
            clean_name = name[:-3]
            
        parts = clean_name.split('_gamma_')
        if len(parts) == 2:
            model_name = parts[0].replace('edgeface_timm_', '')
            gamma = float(parts[1].replace('_', '.'))
            model = replace_linear_with_lowrank_2(get_timmfrv2(model_name, batchnorm=False), rank_ratio=gamma)
        else:
            model_name = clean_name.replace('edgeface_timm_', '')
            model = get_timmfrv2(model_name, batchnorm=False)
            
        if quantize == 'q':
            model = torch.quantization.quantize_dynamic(model, qconfig_spec={torch.nn.Linear}, dtype=torch.qint8)
        elif quantize == 'q4':
            model = replace_linear_with_int4(model)
            
        return model
    elif name.startswith('edgeface_fusion_'):
        quantize = None
        clean_name = name
        if name.endswith('_q'):
            quantize = 'q'
            clean_name = name[:-2]
        elif name.endswith('_q4'):
            quantize = 'q4'
            clean_name = name[:-3]
            
        model_name = clean_name.replace('edgeface_fusion_', '')
        gamma = None
        if '_gamma_' in model_name:
            parts = model_name.split('_gamma_')
            model_name = parts[0]
            gamma_str = parts[1]
            if len(gamma_str) == 2 and gamma_str.startswith('0'):
                gamma = float(gamma_str) / 10.0
            else:
                gamma = float(gamma_str.replace('_', '.'))
            
        timm_name = 'edgenext_x_small'
        if model_name == 'xs': timm_name = 'edgenext_x_small'
        elif model_name == 's': timm_name = 'edgenext_small'
        elif model_name == 'base': timm_name = 'edgenext_base'
        
        quantize_fusion = quantize is not None
        model = get_dual_timmfrv2(timm_name, batchnorm=False, quantize_fusion=quantize_fusion)
        if gamma is not None:
            model = replace_linear_with_lowrank_2(model, rank_ratio=gamma)
            
        if quantize == 'q':
            # Skip QuantizedLowRankTensorFusion so its linear layers are not replaced by DynamicQuantizedLinear
            from .fusion import QuantizedLowRankTensorFusion
            q_spec = {
                torch.nn.Linear: torch.quantization.default_dynamic_qconfig,
                'fusion': None
            }
            model = torch.quantization.quantize_dynamic(model, qconfig_spec=q_spec, dtype=torch.qint8)
        elif quantize == 'q4':
            model = replace_linear_with_int4(model)
            
        return model
    else:
        raise ValueError(f"Model {name} not found or not supported.")
