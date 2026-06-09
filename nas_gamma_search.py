"""
Neural Architecture Search for Layer-wise Gamma (rank_ratio) optimization in EdgeFace
"""

import torch
import random
import os
import argparse
from tabulate import tabulate
import pandas as pd
from backbones.timmfr import get_timmfrv2, replace_linear_with_lowrank_nas

# We'll borrow the FLOP/Param evaluation functions from eval_edgeface.py
def count_parameters_in_millions(model):
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params / 1e6

def get_mflops(model):
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ], with_flops=True
    ) as p:
        _ = model(torch.rand((1,3,112,112)))

    total_flops = 0
    for record in p.key_averages():
        if hasattr(record, 'flops'):
            total_flops += record.flops
    return total_flops / 1e6

def count_linear_layers(model, replace_all=True):
    """Count the number of linear layers that will be replaced."""
    count = 0
    def _count(m):
        nonlocal count
        for name, module in m.named_children():
            if isinstance(module, torch.nn.Linear) and (replace_all or not any(h in name.lower() for h in ['head', 'fc', 'classifier'])):
                count += 1
            else:
                _count(module)
    _count(model)
    return count

def main(args):
    print(f"Starting NAS for {args.model_name}")
    
    # Base model to get linear layer count
    base_model = get_timmfrv2(args.model_name, batchnorm=False)
    num_linear = count_linear_layers(base_model, replace_all=True)
    print(f"Found {num_linear} replaceable nn.Linear layers.")

    best_config = None
    best_score = float('inf')  # Let's say we want to minimize FLOPs + Params
    
    results = []

    for i in range(args.num_trials):
        # Sample random rank ratios between 0.1 and 1.0 for each layer
        rank_ratios = [round(random.uniform(0.1, 1.0), 2) for _ in range(num_linear)]
        
        # Instantiate fresh model and apply NAS configuration
        model = get_timmfrv2(args.model_name, batchnorm=False)
        model = replace_linear_with_lowrank_nas(model, rank_ratios, replace_all=True)
        model.eval()

        mflops = get_mflops(model)
        mparams = count_parameters_in_millions(model)
        
        # A mock fitness score: we want lowest FLOPs and Params
        score = mflops + (mparams * 1000) 
        
        results.append({
            'Trial': i,
            'Gammas': str(rank_ratios),
            'MFLOPS': mflops,
            'MPARAMS': mparams,
            'Score': score
        })

        if score < best_score:
            best_score = score
            best_config = rank_ratios

    df = pd.DataFrame(results)
    print(tabulate(df.sort_values('Score'), headers='keys', tablefmt='fancy_grid', showindex=False))

    print("\nBest Configuration Found:")
    print(best_config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="edgenext_x_small")
    parser.add_argument("--num_trials", type=int, default=5, help="Number of random NAS trials to run")
    main(parser.parse_args())
