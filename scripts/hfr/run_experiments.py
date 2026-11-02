"""
Train + evaluate a list of HFR experiments. Each non-empty, non-# line of the config file is
    <name> <train_hfr.py arguments...>
and produces <out_root>/<name>/{model.pt, eval_fp32.json, eval_int8.json, lfw.json}.
--mode none evaluates the pretrained checkpoint without training (zero-shot baseline).
Finished steps are skipped, so a crashed or killed job can simply be resubmitted.
--shard i/n runs every n-th experiment starting at i, so n PBS jobs can share one config.
"""

import argparse
import json
import os
import shlex
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_SPLIT = {"train": "test", "devtrain": "devval"}


def value(argv, flag, default=None):
    return argv[argv.index(flag) + 1] if flag in argv else default


def run(cmd):
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main(args):
    with open(args.config) as f:
        lines = [l.split("#")[0].strip() for l in f]
    experiments = [shlex.split(l) for l in lines if l]
    index, count = map(int, args.shard.split("/"))
    for i, (name, *train_args) in enumerate(experiments):
        if i % count != index:
            continue
        out = os.path.join(args.out_root, name)
        mode = value(train_args, "--mode", "full")
        model = os.path.join(out, "model.pt")
        if mode == "none":
            model = value(train_args, "--weight", "checkpoints/edgeface_xs_gamma_06.pt")
            os.makedirs(out, exist_ok=True)
            with open(os.path.join(out, "args.json"), "w") as f:
                json.dump({"mode": "none", "weight": model, "trainable_params": 0}, f, indent=2)
        elif not os.path.exists(model):
            run([sys.executable, os.path.join(HERE, "train_hfr.py"), *train_args, "--output", out])

        model_flags = ["--network", value(train_args, "--network", "edgeface_xs_gamma_06"), "--weight", model]
        if mode in ("mod", "mod_full"):
            model_flags += ["--modulation", "--mod_targets", value(train_args, "--mod_targets", "lowrank")]
        if mode == "lora":
            model_flags += ["--extra_rank", value(train_args, "--lora_rank", "1")]
        split_flags = ["--split", EVAL_SPLIT[value(train_args, "--split", "train")]]
        if "--fold" in train_args:
            split_flags = ["--split", "test", "--fold", value(train_args, "--fold")]

        evals = [("fp32", []), ("int8", ["--int8"])]
        evals += [("int8_s" + stages.replace(",", ""), ["--int8", "--int8_stages", stages])
                  for stages in args.int8_stages]
        for precision, extra in evals:
            result = os.path.join(out, f"eval_{precision}.json")
            if not os.path.exists(result):
                run([sys.executable, os.path.join(HERE, "eval_hfr.py"), *model_flags, *split_flags, *extra,
                     "--out", result])
        lfw = os.path.join(out, "lfw.json")
        if args.lfw and not os.path.exists(lfw):
            run([sys.executable, os.path.join(HERE, "eval_vis_bench.py"), *model_flags, "--targets", "lfw",
                 "--out", lfw])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config")
    parser.add_argument("--out_root", default="results/hfr")
    parser.add_argument("--shard", default="0/1")
    parser.add_argument("--lfw", action="store_true", help="also measure visible-domain forgetting on LFW")
    parser.add_argument("--int8_stages", nargs="*", default=[],
                        help="extra mixed-precision evaluations, e.g. 3 0,1,2 -> eval_int8_s3.json, eval_int8_s012.json")
    main(parser.parse_args())
