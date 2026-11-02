"""
Result figures for the paper, from results/hfr/cv (cross-validation runs) and
results/hfr/analysis (scripts/hfr/analysis.py). Run from the repository root.
  tradeoff.pdf        TAR@FAR=0.1% vs LFW forgetting
  int8_drop.pdf       thermal rank-1 lost to INT8 vs trainable parameters
  pose.pdf            thermal-to-visible rank-1 per head position
  scores.pdf          genuine / impostor score distributions
  stage_int8.pdf      rank-1 lost when a single stage is quantized
"""

import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CV = "results/hfr/cv"
ANALYSIS = "results/hfr/analysis"
OUT = os.path.dirname(os.path.abspath(__file__))
COL_W = 3.45  # IEEE column width in inches

plt.rcParams.update({"font.family": "serif", "font.size": 7.5, "axes.titlesize": 7.5, "legend.fontsize": 6.5,
                     "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})

NAMES = {"zeroshot": "Zero-shot", "full": "Full FT", "lora": "LoRA", "bitfit": "BitFit", "ln": "LN tuning",
         "msln": "msLN", "n1both_msln": "RSM+msLN", "n1both": "RSM", "n1": "RSM-s"}
SHARED = ("full", "lora", "bitfit", "ln")
C_SHARED, C_SPEC, C_ZERO = "#1f4e9c", "#d9711a", "#6b6b6b"
STYLE = {"zeroshot": (C_ZERO, "o", ":"), "full": (C_SHARED, "o", "-"), "lora": (C_SHARED, "^", "--"),
         "ln": (C_SHARED, "s", "-."), "msln": (C_SPEC, "s", "-"), "n1both_msln": (C_SPEC, "D", "--")}


def runs(method):
    seeds = [0] if method == "zeroshot" else [0, 1, 2]
    return [f"{CV}/{method}_f{f}_s{s}" for f in range(5) for s in seeds]


def metric(method, precision="fp32", key="thr2vis", name="rank1"):
    return np.array([json.load(open(f"{r}/eval_{precision}.json"))[key][name] for r in runs(method)])


def lfw(method):
    return np.array([json.load(open(f"{r}/lfw.json"))["lfw"]["acc"] for r in runs(method)])


def params(method):
    return json.load(open(f"{runs(method)[0]}/args.json"))["trainable_params"]


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def tradeoff():
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    base = lfw("zeroshot").mean()
    offsets = {"full": (5, 0), "lora": (5, 0), "bitfit": (0, 7), "ln": (-4, -7), "msln": (6, 3),
               "n1both_msln": (6, -4), "n1both": (6, 0), "n1": (6, 0)}
    for m in ("full", "lora", "bitfit", "ln", "msln", "n1both_msln", "n1both", "n1"):
        x, y = 100 * (base - lfw(m).mean()), 100 * metric(m, name="tar@far=1e-3")
        shared = m in SHARED
        ax.plot(x, y.mean(), "o" if shared else "s", ms=4.5, color=C_SHARED if shared else C_SPEC)
        ax.annotate(NAMES[m], (x, y.mean()), textcoords="offset points", xytext=offsets[m], fontsize=6.5,
                    ha="right" if offsets[m][0] < 0 else "center" if offsets[m][0] == 0 else "left",
                    va="center")
    ax.plot([], [], "o", ms=4.5, color=C_SHARED, label="Shared parameters")
    ax.plot([], [], "s", ms=4.5, color=C_SPEC, label="Modality-specific (visible path frozen)")
    ax.set_xlabel("LFW accuracy drop w.r.t. pretrained model (pp)")
    ax.set_ylabel("TAR@FAR=0.1% (%)")
    ax.set_xlim(-0.15, 2.3)
    ax.set_ylim(30, 92)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="lower right", frameon=False)
    save(fig, "tradeoff")


def int8_drop():
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    offsets = {"full": (5, 0), "lora": (6, 0), "bitfit": (6, 0), "ln": (-6, -5), "msln": (-6, 3),
               "n1both_msln": (6, 6), "n1both": (-6, -14), "n1": (-6, 0)}
    for m in ("full", "lora", "bitfit", "ln", "msln", "n1both_msln", "n1both", "n1"):
        d = 100 * (metric(m) - metric(m, "int8"))
        v = 100 * (metric(m, key="vis2vis") - metric(m, "int8", key="vis2vis"))
        shared = m in SHARED
        ax.errorbar(params(m), d.mean(), yerr=d.std(), fmt="o" if shared else "s", ms=4, capsize=1.5, lw=0.6,
                    color=C_SHARED if shared else C_SPEC)
        ax.scatter(params(m), v.mean(), marker="x", s=14, lw=0.8, color="#444444")
        dx, dy = offsets[m]
        ax.annotate(NAMES[m], (params(m), d.mean()), textcoords="offset points", xytext=(dx, dy), fontsize=6.5,
                    ha="right" if dx < 0 else "left", va="center")
    ax.scatter([], [], marker="o", color=C_SHARED, label="Thermal$\\to$visible, shared")
    ax.scatter([], [], marker="s", color=C_SPEC, label="Thermal$\\to$visible, modality-specific")
    ax.scatter([], [], marker="x", color="#444444", label="Visible$\\to$visible (same models)")
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Rank-1 lost to INT8 (pp)")
    ax.set_ylim(-1, 17)
    ax.set_xlim(3e2, 8e6)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(loc="upper right", frameon=False)
    save(fig, "int8_drop")


def load_analysis(m):
    return json.load(open(f"{ANALYSIS}/{m}.json"))


def pose():
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    for m in ("full", "lora", "ln", "msln", "n1both_msln", "zeroshot"):
        a = load_analysis(m)["per_position"]
        y = [100 * a[str(p)]["rank1"] for p in range(1, 10)]
        c, mk, ls = STYLE[m]
        ax.plot(range(1, 10), y, color=c, marker=mk, ls=ls, ms=3, lw=0.9, label=NAMES[m])
    ax.set_xticks(range(1, 10))
    ax.set_xlabel("Head position of the thermal probe (1 = frontal)")
    ax.set_ylabel("Rank-1 (%)")
    ax.set_ylim(0, 102)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(ncol=3, loc="lower left", frameon=False, bbox_to_anchor=(0, 0.02))
    save(fig, "pose")


def scores():
    methods = ("zeroshot", "full", "msln")
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 1.55), sharey=True)
    for ax, m in zip(axes, methods):
        h = load_analysis(m)["hist"]
        bins = np.array(h["fp32"]["bins"])
        centers = (bins[1:] + bins[:-1]) / 2
        for prec, ls in (("fp32", "-"), ("int8", "--")):
            g, i = np.array(h[prec]["genuine"]), np.array(h[prec]["impostor"])
            ax.plot(centers, g / g.sum(), color="#2a7f3b", ls=ls, lw=0.9)
            ax.plot(centers, i / i.sum(), color="#b2272c", ls=ls, lw=0.9)
        ax.set_title(NAMES[m])
        ax.set_xlabel("Cosine similarity")
        ax.grid(alpha=0.25, lw=0.4)
    axes[0].set_ylabel("Fraction of pairs")
    axes[2].plot([], [], color="#2a7f3b", label="Genuine")
    axes[2].plot([], [], color="#b2272c", label="Impostor")
    axes[2].plot([], [], color="k", ls="-", label="FP32")
    axes[2].plot([], [], color="k", ls="--", label="INT8")
    axes[2].legend(loc="upper right", frameon=False, fontsize=6, ncol=2, columnspacing=0.8)
    axes[0].set_ylim(0, 0.118)
    save(fig, "scores")


def stage_int8():
    methods = ("zeroshot", "full", "lora", "ln", "msln", "n1both_msln")
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    width = 0.18
    colors = ["#c6dbef", "#6baed6", "#2171b5", "#08306b"]
    for k in range(4):
        drops = []
        for m in methods:
            fp32 = np.mean([json.load(open(f"{CV}/{m}_f{f}_s0/eval_fp32.json"))["thr2vis"]["rank1"]
                            for f in range(5)])
            drops.append(100 * (fp32 - load_analysis(m)["single_stage_int8"][str(k)]))
        ax.bar(np.arange(len(methods)) + (k - 1.5) * width, drops, width, color=colors[k],
               label=f"Stage {k + 1} only")
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels([NAMES[m].replace("+", "+\n").replace("-shot", "-\nshot").replace(" tuning", "\ntuning")
                        for m in methods])
    ax.set_ylabel("Rank-1 lost to INT8 (pp)")
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.set_ylim(-0.4, 6.2)
    ax.legend(ncol=4, loc="upper left", frameon=False, columnspacing=0.8, handlelength=1.2)
    save(fig, "stage_int8")


if __name__ == "__main__":
    tradeoff()
    int8_drop()
    if os.path.isdir(ANALYSIS):
        pose()
        scores()
        stage_int8()
