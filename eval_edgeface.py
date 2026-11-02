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

import torch
import os
import pandas as pd
import sys
from tabulate import tabulate
from backbones import get_model
import argparse

def _is_dynamic_quantized_linear(module):
    return isinstance(module, torch.ao.nn.quantized.dynamic.Linear)


def count_parameters_in_millions(model):
    model.train()
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    # Dynamic INT8 linears keep their weights in packed buffers that parameters() does not see
    for module in model.modules():
        if _is_dynamic_quantized_linear(module):
            total_params += module.weight().numel()
            if module.bias() is not None:
                total_params += module.bias().numel()
    return total_params / 1e6


def code_to_profile(model):
    _ = model(torch.rand((1,3,112,112)))

def get_mflops(model):
    # The profiler reports no FLOPs for quantized::linear_dynamic, so count them with hooks
    quantized_linear_flops = [0]

    def _count_quantized_linear(module, inputs, output):
        x = inputs[0]
        rows = x.numel() // module.in_features
        quantized_linear_flops[0] += 2 * rows * module.in_features * module.out_features

    hooks = [m.register_forward_hook(_count_quantized_linear)
             for m in model.modules() if _is_dynamic_quantized_linear(m)]

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ], with_flops=True
    ) as p:
        code_to_profile(model)

    for h in hooks:
        h.remove()

    print(p.key_averages().table(
        sort_by="self_cpu_time_total", row_limit=-1))
    # Get the key averages from the profiler
    key_averages = p.key_averages()

    # Initialize total FLOPs counter
    total_flops = 0

    # Iterate over key averages and sum the FLOPs
    for record in key_averages:
        if hasattr(record, 'flops'):
            total_flops += record.flops
    total_flops += quantized_linear_flops[0]

    # Print the total FLOPs
    MFLOPS=total_flops/ 1e6
    print(f"Total FLOPs: {MFLOPS} M")
    
    return MFLOPS


def main(args):

    all_dict={}

    os.makedirs('dummy', exist_ok=True)

    name = args.name
        
    model=get_model(name)
    
    model.eval()
    
    MODEL_PATH=os.path.join('dummy', name+'.pth')
    
    # saves a dummy state dict to see model size
    torch.save(model.state_dict(), MODEL_PATH)

    MPARAMS=count_parameters_in_millions(model)

    ret_dict={}

    ret_dict['MFLOPS']=get_mflops(model)

    ret_dict['MPARAMS']=MPARAMS

    ret_dict['MODEL SIZE-MB']=(os.stat(MODEL_PATH).st_size/1e6)
    
    
    all_dict[name]=ret_dict

    df= pd.DataFrame.from_dict(all_dict).T

    print(tabulate(df, headers='keys', tablefmt='fancy_grid'))

 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Computes the MFLOPS , parameters and model size on disk")
    parser.add_argument("name", 
                        type=str, 
                        help="Name of the model to evaluate",
                        default="edgeface_s_gamma_05")

    main(parser.parse_args())