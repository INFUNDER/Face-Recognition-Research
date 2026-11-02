"""
CPU latency and serialized size of EdgeFace variants for the edge-deployment study:
FP32, full dynamic INT8 and stage-selective (mixed-precision) INT8, single 112x112 face,
for several thread counts. Reports median and 90th-percentile latency in milliseconds.
"""

import argparse
import io
import json
import os
import platform
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.hfr.eval_hfr import build_model


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.getbuffer().nbytes / 2 ** 20


@torch.no_grad()
def latency_ms(model, threads, warmup, iters):
    torch.set_num_threads(threads)
    x = torch.randn(1, 3, 112, 112)
    for _ in range(warmup):
        model(x)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model(x)
        times.append(1000 * (time.perf_counter() - start))
    return float(np.median(times)), float(np.percentile(times, 90))


def main(args):
    variants = [("fp32", False, "")] + [(f"int8[{s or 'all'}]", True, s) for s in args.int8_stages]
    results = {"cpu": platform.processor() or platform.machine(), "torch": torch.__version__, "runs": []}
    for network in args.networks.split(","):
        for name, int8, stages in variants:
            model = build_model(SimpleNamespace(network=network, weight="", modulation=False, mod_targets="lowrank",
                                                extra_rank=0, int8=int8, int8_stages=stages))
            row = {"network": network, "variant": name, "size_mb": size_mb(model)}
            for threads in args.threads:
                row[f"ms@{threads}t"], row[f"p90@{threads}t"] = latency_ms(model, threads, args.warmup, args.iters)
            results["runs"].append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--networks", default="edgeface_xs_gamma_06,edgeface_s_gamma_05")
    parser.add_argument("--int8_stages", nargs="*", default=["", "0,1,2", "0,1", "2,3", "3"],
                        help="stage sets to quantize ('' = all linear layers)")
    parser.add_argument("--threads", nargs="*", type=int, default=[1, 4])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--out", default="")
    main(parser.parse_args())
