"""
Fine-tune a pretrained EdgeFace model for thermal-to-visible HFR on SpeakingFaces train subjects.

Modes:
  full      all backbone weights trained, no modality-specific parameters
  mod       modality modulation (backbones/modality.py) of the layers chosen by --mod_targets:
            rank-space vectors of the factorised layers (--mod_params picks scale/shift) and/or
            modality-specific LayerNorm affines; only these (and the classification head) are
            trained, the backbone is frozen
  mod_full  modulation + all backbone weights trained
  bitfit    PEFT baseline: only bias terms trained (shared by both modalities)
  ln        PEFT baseline: only normalisation affine parameters trained
  lora      PEFT baseline: --lora_rank extra components per factorised layer (backbones/peft.py)
With --anchor_vis the visible-modality vectors stay at their initial values, so (with a frozen
backbone) the visible path is exactly the pretrained model and visible-domain accuracy cannot
degrade.

N2 (quantization-aware HFR):
  --qat           train under simulated dynamic INT8 (backbones/qat.py), as deployed
  --distill_pair  weight of a paired cross-modal distillation loss that pulls each thermal
                  embedding towards the frozen FP32 pretrained embedding of its pixel-registered
                  visible frame, i.e. towards where the visible gallery templates live

Images of both modalities from the train subjects (both trials, all positions) are mixed in each
batch; the ArcFace head is discarded after training.
"""

import argparse
import copy
import glob
import json
import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from backbones import get_model
from backbones.modality import (ModulatedLayerNorm, ModulatedLoRaLin, add_modality_modulation,
                                modulation_parameters, set_modality)
from backbones.peft import expand_lowrank, freeze_pretrained_components
from backbones.qat import disable_fake_int8, enable_fake_int8
from losses import CombinedMarginLoss
from scripts.hfr.eval_hfr import NUM_FOLDS, SPLITS, get_subjects, load_image

MODALITIES = {"vis": 0, "thr": 1}
MOD_PARAMS = {"scale": ("scale",), "shift": ("shift",), "both": ("scale", "shift")}


class SpeakingFacesTrain(Dataset):
    """Yields (image, label, modality id, registered visible frame of the same capture)."""
    def __init__(self, data_dir, subjects, gray_thr=True):
        self.items = []
        self.gray_thr = gray_thr
        for modality, mid in MODALITIES.items():
            for sub in subjects:
                for path in glob.glob(os.path.join(data_dir, modality, f"sub_{sub}", "trial_*", "*.png")):
                    self.items.append((path, sub, mid))
        present = sorted({sub for _, sub, _ in self.items})
        labels = {s: i for i, s in enumerate(present)}
        self.items = [(path, labels[sub], mid) for path, sub, mid in self.items]
        self.num_classes = len(present)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, mid = self.items[idx]
        img = load_image(path, gray=self.gray_thr and mid == MODALITIES["thr"])
        pair = img
        if mid == MODALITIES["thr"]:
            pair_path = path.replace(os.sep + "thr" + os.sep, os.sep + "vis" + os.sep)
            pair = load_image(pair_path, gray=False)
        if random.random() < 0.5:
            img, pair = torch.flip(img, dims=[2]), torch.flip(pair, dims=[2])
        return img, label, mid, pair


def main(args):
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda"
    # cuDNN fails to initialise on this cluster's H100 nodes (same workaround as train_v2.py)
    torch.backends.cudnn.enabled = False

    dataset = SpeakingFacesTrain(args.data_dir, get_subjects(args.split, args.fold))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                        drop_last=True, pin_memory=True)
    print(f"{len(dataset)} images, {dataset.num_classes} identities", flush=True)

    model = get_model(args.network)
    model.load_state_dict(torch.load(args.weight, map_location="cpu"))
    teacher = None
    if args.distill_pair > 0:
        teacher = copy.deepcopy(model).to(device).eval()
        for p in teacher.parameters():
            p.requires_grad = False

    use_mod = args.mode in ("mod", "mod_full")
    frozen_backbone = args.mode not in ("full", "mod_full")
    if use_mod:
        model = add_modality_modulation(model, num_modalities=len(MODALITIES), targets=args.mod_targets.split(","))
    for p in model.parameters():
        p.requires_grad = not frozen_backbone

    n_masked = 0
    peft_params = []
    if args.mode == "bitfit":
        peft_params = [p for n, p in model.named_parameters() if n.endswith("bias")]
    elif args.mode == "ln":
        peft_params = [p for m in model.modules() if "LayerNorm" in type(m).__name__
                       for p in m.parameters(recurse=False)]
    elif args.mode == "lora":
        expanded = expand_lowrank(model, args.lora_rank)
        for p in model.parameters():
            p.requires_grad = False
        peft_params = freeze_pretrained_components(expanded)
        n_masked = sum(m.linear1.weight[:r].numel() + m.linear2.weight[:, :r].numel() for m, r in expanded)

    mod_params, norm_params = [], []
    if use_mod:
        for p in modulation_parameters(model):
            p.requires_grad = False
        mod_params = modulation_parameters(model, MOD_PARAMS[args.mod_params], ModulatedLoRaLin)
        norm_params = modulation_parameters(model, targets=ModulatedLayerNorm)
    for p in mod_params + norm_params + peft_params:
        p.requires_grad = True
    if use_mod and args.anchor_vis:
        vis_row = MODALITIES["vis"]
        for p in mod_params + norm_params:
            p.register_hook(lambda g: g.index_fill(0, torch.tensor([vis_row], device=g.device), 0.0))
            n_masked += p[vis_row].numel()

    if args.qat:
        enable_fake_int8(model)
    model = model.to(device)

    head = torch.nn.Parameter(torch.normal(0, 0.01, (dataset.num_classes, args.embedding_size), device=device))
    margin = CombinedMarginLoss(64, 1.0, 0.5, 0.0)

    # vectors/slices that must stay fixed get no weight decay, otherwise AdamW would still move them
    groups = [{"params": [head], "lr": args.lr_head}]
    if mod_params:
        groups.append({"params": mod_params, "lr": args.lr_mod, "weight_decay": 0.0})
    if norm_params:
        groups.append({"params": norm_params, "lr": args.lr_mod_norm, "weight_decay": 0.0})
    if peft_params:
        groups.append({"params": peft_params, "lr": args.lr_peft, "weight_decay": 0.0})
    if not frozen_backbone:
        mod_ids = {id(p) for p in modulation_parameters(model)}
        groups.append({"params": [p for p in model.parameters() if id(p) not in mod_ids and p.requires_grad],
                       "lr": args.lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
    total_steps = args.epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[g["lr"] for g in groups], total_steps=total_steps, pct_start=0.1)

    n_trainable = sum(p.numel() for g in groups[1:] for p in g["params"]) - n_masked
    print(f"mode={args.mode} trainable backbone params: {n_trainable}", flush=True)

    thr_id = MODALITIES["thr"]
    step = 0
    for epoch in range(args.epochs):
        model.train()
        if frozen_backbone:
            model.eval()  # frozen backbone: no drop-path, keep normalisation behaviour fixed
        for img, label, mid, pair in loader:
            img, label, mid = img.to(device), label.to(device), mid.to(device)
            set_modality(model, mid if use_mod else None)
            emb = F.normalize(model(img), dim=1)
            logits = F.linear(emb, F.normalize(head, dim=1)).clamp(-1, 1)
            loss = F.cross_entropy(margin(logits, label), label)
            kd = torch.zeros((), device=device)
            if teacher is not None and (mid == thr_id).any():
                thr = mid == thr_id
                with torch.no_grad():
                    target = F.normalize(teacher(pair.to(device)[thr]), dim=1)
                kd = (1 - (emb[thr] * target).sum(1)).mean()
                loss = loss + args.distill_pair * kd
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} kd {kd.item():.4f}",
                      flush=True)

    set_modality(model, None)
    disable_fake_int8(model)
    os.makedirs(args.output, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output, "model.pt"))
    with open(os.path.join(args.output, "args.json"), "w") as f:
        json.dump({**vars(args), "trainable_params": n_trainable}, f, indent=2)
    print("saved", os.path.join(args.output, "model.pt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", default="full", choices=["full", "mod", "mod_full", "bitfit", "ln", "lora"])
    parser.add_argument("--network", default="edgeface_xs_gamma_06")
    parser.add_argument("--weight", default="checkpoints/edgeface_xs_gamma_06.pt")
    parser.add_argument("--data_dir", default="data/speakingfaces_112")
    parser.add_argument("--split", default="train", choices=list(SPLITS))
    parser.add_argument("--fold", type=int, default=None, choices=range(NUM_FOLDS), help="5-fold CV fold")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_head", type=float, default=1e-2)
    parser.add_argument("--lr_mod", type=float, default=1e-2)
    parser.add_argument("--lr_peft", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--embedding_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=6)
    parser.add_argument("--mod_targets", default="lowrank", help="comma-separated: lowrank,norm")
    parser.add_argument("--mod_params", default="scale", choices=list(MOD_PARAMS), help="for lowrank targets")
    parser.add_argument("--lr_mod_norm", type=float, default=1e-3, help="lr of modality-specific LayerNorm")
    parser.add_argument("--anchor_vis", action="store_true", help="keep visible-modality vectors at init")
    parser.add_argument("--lora_rank", type=int, default=1)
    parser.add_argument("--qat", action="store_true", help="train under simulated dynamic INT8")
    parser.add_argument("--distill_pair", type=float, default=0.0, help="paired cross-modal distillation weight")
    parser.add_argument("--seed", type=int, default=0)
    main(parser.parse_args())
