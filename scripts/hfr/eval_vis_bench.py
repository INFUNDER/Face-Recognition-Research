"""
Check how much an HFR-fine-tuned checkpoint forgets ordinary visible face verification.
Runs the standard InsightFace verification protocol on .bin benchmarks (LFW, CFP-FP, AgeDB-30, ...).
Modulated checkpoints are evaluated through their visible-modality path (modality id 0).
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

# mxnet (used by load_bin) still references np.bool, removed in NumPy 1.24
if not hasattr(np, "bool"):
    np.bool = bool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backbones.modality import set_modality
from eval import verification
from scripts.hfr.eval_hfr import build_model


class VisiblePath(torch.nn.Module):
    def __init__(self, model, modulation):
        super().__init__()
        self.model = model
        self.modulation = modulation

    def forward(self, x):
        if self.modulation:
            set_modality(self.model, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))
        return self.model(x)


@torch.no_grad()
def main(args):
    model = VisiblePath(build_model(args), args.modulation)
    results = {"weight": args.weight, "int8": args.int8, "modulation": args.modulation}
    for target in args.targets.split(","):
        path = os.path.join(args.bin_dir, target + ".bin")
        if not os.path.exists(path):
            print(f"skip {target}: {path} not found")
            continue
        data_set = verification.load_bin(path, (112, 112))
        _, _, acc, std, _, _ = verification.test(data_set, model, args.batch_size, 10)
        results[target] = {"acc": float(acc), "std": float(std)}
        print(f"{target}: {100 * acc:.3f} +- {100 * std:.3f}", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", default="edgeface_xs_gamma_06")
    parser.add_argument("--weight", default="checkpoints/edgeface_xs_gamma_06.pt")
    parser.add_argument("--modulation", action="store_true")
    parser.add_argument("--mod_targets", default="lowrank")
    parser.add_argument("--int8", action="store_true")
    parser.add_argument("--extra_rank", type=int, default=0)
    parser.add_argument("--int8_stages", default="")
    parser.add_argument("--bin_dir", default="data")
    parser.add_argument("--targets", default="lfw,cfp_fp,agedb_30,calfw,cplfw")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="")
    main(parser.parse_args())
