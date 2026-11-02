"""
Extra analyses of the cross-validation runs for the paper (seed 0, all folds):
  - thermal-to-visible rank-1 / TAR@FAR=1e-3 per head position (FP32)
  - genuine / impostor cosine-score histograms (FP32 and full INT8)
  - rank-1 when the linear layers of a single stage are quantized to INT8
Writes one JSON per method to --out_dir.
"""

import argparse
import json
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from scripts.hfr.eval_hfr import NUM_FOLDS, build_model, collect, embed, get_subjects, score

METHODS = {
    "zeroshot": {},
    "full": {},
    "lora": {"extra_rank": 1},
    "ln": {},
    "msln": {"modulation": True, "mod_targets": "norm"},
    "n1both_msln": {"modulation": True, "mod_targets": "lowrank,norm"},
}
BINS = np.linspace(-0.4, 1.0, 71)


def model_args(method, fold, int8=False, int8_stages=""):
    cfg = METHODS[method]
    weight = ("checkpoints/edgeface_xs_gamma_06.pt" if method == "zeroshot"
              else f"results/hfr/cv/{method}_f{fold}_s0/model.pt")
    return SimpleNamespace(network="edgeface_xs_gamma_06", weight=weight, int8=int8, int8_stages=int8_stages,
                           modulation=cfg.get("modulation", False), mod_targets=cfg.get("mod_targets", "lowrank"),
                           extra_rank=cfg.get("extra_rank", 0))


def evaluate(args, fold, device):
    model = build_model(args).to(device)
    ids = (0, 1) if args.modulation else (None, None)
    subjects = get_subjects("test", fold)
    gallery = collect(args.data_dir, "vis", subjects, trial=1, positions={1})
    g_feats = embed(model, gallery, False, device, 64, False, ids[0])
    g_subs = np.array([s for _, s, _ in gallery])
    template_ids = np.unique(g_subs)
    templates = np.stack([g_feats[g_subs == s].mean(0) for s in template_ids])
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)
    probes = collect(args.data_dir, "thr", list(template_ids), trial=2)
    p_feats = embed(model, probes, True, device, 64, False, ids[1])
    p_ids = np.array([s for _, s, _ in probes])
    p_pos = np.array([p for _, _, p in probes])
    return templates, template_ids, p_feats, p_ids, p_pos


def main(cli):
    os.makedirs(cli.out_dir, exist_ok=True)
    torch.backends.cudnn.enabled = False
    gpu = "cuda" if torch.cuda.is_available() else "cpu"
    for method in cli.methods:
        out = {"method": method, "per_position": {}, "hist": {}, "single_stage_int8": {}}
        per_pos = {p: [] for p in range(1, 10)}
        hists = {"fp32": [np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)],
                 "int8": [np.zeros(len(BINS) - 1), np.zeros(len(BINS) - 1)]}
        stage_r1 = {k: [] for k in range(4)}
        for fold in range(NUM_FOLDS):
            for precision in ("fp32", "int8"):
                a = model_args(method, fold, int8=precision == "int8")
                a.data_dir = cli.data_dir
                templates, tids, feats, pids, ppos = evaluate(a, fold, "cpu" if a.int8 else gpu)
                sims = feats @ templates.T
                genuine = pids[:, None] == tids[None, :]
                hists[precision][0] += np.histogram(sims[genuine], BINS)[0]
                hists[precision][1] += np.histogram(sims[~genuine], BINS)[0]
                if precision == "fp32":
                    for p in range(1, 10):
                        m = ppos == p
                        per_pos[p].append(score(templates, tids, feats[m], pids[m]))
            for k in range(4):
                a = model_args(method, fold, int8=True, int8_stages=str(k))
                a.data_dir = cli.data_dir
                templates, tids, feats, pids, _ = evaluate(a, fold, "cpu")
                stage_r1[k].append(score(templates, tids, feats, pids)["rank1"])
            print(method, "fold", fold, "done", flush=True)
        out["per_position"] = {p: {m: float(np.mean([r[m] for r in rs])) for m in ("rank1", "tar@far=1e-3")}
                               for p, rs in per_pos.items()}
        out["hist"] = {k: {"bins": BINS.tolist(), "genuine": g.tolist(), "impostor": i.tolist()}
                       for k, (g, i) in hists.items()}
        out["single_stage_int8"] = {k: float(np.mean(v)) for k, v in stage_r1.items()}
        with open(os.path.join(cli.out_dir, f"{method}.json"), "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--methods", nargs="*", default=list(METHODS))
    parser.add_argument("--data_dir", default="data/speakingfaces_112")
    parser.add_argument("--out_dir", default="results/hfr/analysis")
    main(parser.parse_args())
