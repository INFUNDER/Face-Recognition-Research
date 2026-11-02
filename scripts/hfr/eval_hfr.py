"""
Thermal-to-visible heterogeneous face recognition (HFR) evaluation on aligned SpeakingFaces crops.

Protocol (subject-disjoint, fixed by subject id):
  train subjects: 1..100, test subjects: 101..142, or with --fold k (5-fold CV) test subjects
  are those with id % 5 == k and train subjects are the rest
  gallery: visible, trial 1, frontal position 1 -> one mean template per subject
  probes : thermal (HFR) or visible (VIS->VIS sanity check), trial 2, positions 1..9
Trial 2 is a separate recording, so probes are never the same frames as the gallery.

Reports rank-1 identification and verification (TAR@FAR, EER) over all probe-template pairs,
for all positions and for the frontal position only.
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import torch
from sklearn.metrics import roc_curve

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backbones import get_model
from backbones.modality import add_modality_modulation, set_modality
from backbones.peft import expand_lowrank

SPLITS = {"train": range(1, 101), "test": range(101, 143), "all": range(1, 143),
          # hyper-parameter tuning only touches subjects of the default train split
          "devtrain": range(1, 81), "devval": range(81, 101)}
NUM_FOLDS = 5


def get_subjects(split, fold=None):
    if fold is None:
        return list(SPLITS[split])
    if split not in ("train", "test"):
        raise ValueError("--fold only applies to the train and test splits")
    return [s for s in SPLITS["all"] if (s % NUM_FOLDS == fold) == (split == "test")]


def load_image(path, gray):
    img = cv2.imread(path)
    if gray:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).float()
    return img.div_(255).sub_(0.5).div_(0.5)


def collect(data_dir, modality, subjects, trial, positions=None):
    items = []
    for sub in subjects:
        for path in sorted(glob.glob(os.path.join(data_dir, modality, f"sub_{sub}", f"trial_{trial}", "*.png"))):
            pos = int(os.path.basename(path).split("_")[0])
            if positions is None or pos in positions:
                items.append((path, sub, pos))
    return items


@torch.no_grad()
def embed(model, items, gray, device, batch_size, flip, modality=None):
    feats = []
    for i in range(0, len(items), batch_size):
        x = torch.stack([load_image(p, gray) for p, _, _ in items[i:i + batch_size]]).to(device)
        if modality is not None:
            set_modality(model, torch.full((x.shape[0],), modality, dtype=torch.long, device=device))
        f = model(x)
        if flip:
            f = f + model(torch.flip(x, dims=[3]))
        feats.append(torch.nn.functional.normalize(f.float(), dim=1).cpu())
    return torch.cat(feats).numpy()


def tar_at_far(fpr, tpr, far):
    idx = np.searchsorted(fpr, far, side="right") - 1
    return float(tpr[max(idx, 0)])


def score(gallery_feats, gallery_ids, probe_feats, probe_ids):
    sims = probe_feats @ gallery_feats.T
    rank1 = float(np.mean(gallery_ids[np.argmax(sims, axis=1)] == probe_ids))
    labels = (probe_ids[:, None] == gallery_ids[None, :]).ravel()
    fpr, tpr, _ = roc_curve(labels, sims.ravel())
    eer_idx = np.nanargmin(np.abs((1 - tpr) - fpr))
    return {
        "rank1": rank1,
        "tar@far=1e-2": tar_at_far(fpr, tpr, 1e-2),
        "tar@far=1e-3": tar_at_far(fpr, tpr, 1e-3),
        "eer": float((fpr[eer_idx] + 1 - tpr[eer_idx]) / 2),
        "n_probes": int(len(probe_ids)),
        "n_gallery": int(len(gallery_ids)),
    }


def build_model(args):
    model = get_model(args.network)
    if args.extra_rank:
        expand_lowrank(model, args.extra_rank)
    if args.modulation:
        model = add_modality_modulation(model, targets=args.mod_targets.split(","))
    if args.weight:
        missing, unexpected = model.load_state_dict(torch.load(args.weight, map_location="cpu"), strict=False)
        # checkpoints saved before the rank-space shift existed have no .shift entries (shift = 0)
        if unexpected or any(not k.endswith(".shift") for k in missing):
            raise RuntimeError(f"state_dict mismatch: missing {missing}, unexpected {unexpected}")
    model.eval()
    if args.int8:
        # Dynamic INT8 kernels are CPU-only. --int8_stages limits quantization to the linear
        # layers of the listed EdgeNeXt stages (the rest, and the head, stay FP32).
        spec = {torch.nn.Linear}
        if args.int8_stages:
            prefixes = tuple(f"model.stages.{k}." for k in args.int8_stages.split(","))
            spec = {name for name, m in model.named_modules()
                    if isinstance(m, torch.nn.Linear) and name.startswith(prefixes)}
        model = torch.quantization.quantize_dynamic(model, qconfig_spec=spec, dtype=torch.qint8)
    return model


def main(args):
    device = "cpu" if args.int8 or not torch.cuda.is_available() else "cuda"
    # cuDNN fails to initialise on this cluster's H100 nodes (same workaround as train_v2.py)
    torch.backends.cudnn.enabled = False
    model = build_model(args).to(device)
    subjects = get_subjects(args.split, args.fold)

    gallery = collect(args.data_dir, "vis", subjects, trial=1, positions={1})
    vis_id, thr_id = (0, 1) if args.modulation else (None, None)
    g_feats = embed(model, gallery, args.gray_vis, device, args.batch_size, args.flip, vis_id)
    g_subs = np.array([s for _, s, _ in gallery])
    template_ids = np.unique(g_subs)
    templates = np.stack([g_feats[g_subs == s].mean(0) for s in template_ids])
    templates /= np.linalg.norm(templates, axis=1, keepdims=True)

    results = {"network": args.network, "weight": args.weight, "int8": args.int8,
               "int8_stages": args.int8_stages, "split": args.split,
               "fold": args.fold,
               "gray_thr": args.gray_thr, "gray_vis": args.gray_vis, "flip": args.flip,
               "modulation": args.modulation}
    for name, modality, gray, mid in [("thr2vis", "thr", args.gray_thr, thr_id),
                                      ("vis2vis", "vis", args.gray_vis, vis_id)]:
        probes = collect(args.data_dir, modality, list(template_ids), trial=2)
        if not probes:
            print(f"no {modality} probes found, skipping {name}")
            continue
        p_feats = embed(model, probes, gray, device, args.batch_size, args.flip, mid)
        p_ids = np.array([s for _, s, _ in probes])
        p_pos = np.array([p for _, _, p in probes])
        results[name] = score(templates, template_ids, p_feats, p_ids)
        results[name + "_frontal"] = score(templates, template_ids, p_feats[p_pos == 1], p_ids[p_pos == 1])

    print(json.dumps(results, indent=2))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--network", default="edgeface_xs_gamma_06")
    parser.add_argument("--weight", default="checkpoints/edgeface_xs_gamma_06.pt")
    parser.add_argument("--data_dir", default="data/speakingfaces_112")
    parser.add_argument("--split", default="test", choices=list(SPLITS))
    parser.add_argument("--fold", type=int, default=None, choices=range(NUM_FOLDS), help="5-fold CV fold")
    parser.add_argument("--int8", action="store_true", help="apply dynamic INT8 to nn.Linear (runs on CPU)")
    parser.add_argument("--gray_thr", action=argparse.BooleanOptionalAction, default=True,
                        help="convert false-colour thermal to grayscale")
    parser.add_argument("--gray_vis", action="store_true", help="convert visible images to grayscale")
    parser.add_argument("--modulation", action="store_true", help="checkpoint has modality modulation")
    parser.add_argument("--mod_targets", default="lowrank", help="comma-separated: lowrank,norm")
    parser.add_argument("--int8_stages", default="", help="e.g. 0,1,2: quantize only these stages")
    parser.add_argument("--extra_rank", type=int, default=0, help="checkpoint was trained with LoRA of this rank")
    parser.add_argument("--flip", action="store_true", help="add horizontally flipped embedding")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", default="")
    main(parser.parse_args())
