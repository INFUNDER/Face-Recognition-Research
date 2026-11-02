"""
Summarise run_experiments.py outputs as a markdown table. Runs named <method>_f<fold>_s<seed>
are grouped by <method> and reported as mean +- std over folds and seeds.
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

METRICS = [("thr2vis", "rank1", "R1"), ("thr2vis", "tar@far=1e-3", "TAR@0.1%"), ("thr2vis", "eer", "EER")]


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def main(args):
    groups = defaultdict(list)
    for run in sorted(glob.glob(os.path.join(args.out_root, "*", "eval_fp32.json"))):
        run_dir = os.path.dirname(run)
        method = re.sub(r"_f\d+_s\d+$", "", os.path.basename(run_dir))
        groups[method].append(run_dir)

    metrics = METRICS if len(args.precisions) <= 2 else METRICS[:2]
    cols = ["params"] + [f"{name} {p}" for p in args.precisions for _, _, name in metrics] + ["LFW"]
    print("| method | n | " + " | ".join(cols) + " |")
    print("|---" * (len(cols) + 2) + "|")
    fmt = lambda v: f"{100 * np.mean(v):.2f} ± {100 * np.std(v):.2f}" if v else "–"
    for method, runs in groups.items():
        row = []
        params = {(load(os.path.join(r, "args.json")) or {}).get("trainable_params") for r in runs}
        row.append(",".join(str(p) for p in sorted(params, key=str)))
        for precision in args.precisions:
            results = [load(os.path.join(r, f"eval_{precision}.json")) for r in runs]
            for key, metric, _ in metrics:
                row.append(fmt([res[key][metric] for res in results if res]))
        row.append(fmt([lfw["lfw"]["acc"] for lfw in (load(os.path.join(r, "lfw.json")) for r in runs) if lfw]))
        print(f"| {method} | {len(runs)} | " + " | ".join(row) + " |")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_root")
    parser.add_argument("--precisions", nargs="*", default=["fp32", "int8"],
                        help="eval_<precision>.json files to report, e.g. fp32 int8 int8_s3 int8_s012")
    main(parser.parse_args())
