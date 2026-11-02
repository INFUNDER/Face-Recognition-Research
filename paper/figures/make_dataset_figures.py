"""
Dataset figures for the paper, from SpeakingFaces (CC BY 4.0, Abdrakhmanova et al., 2021):
  dataset_samples.pdf  raw registered visible/thermal frames and the aligned 112x112 crops
  poses.pdf            one subject at the nine head positions (aligned visible and thermal)
Run from the repository root after scripts/hfr/align_speakingfaces.py.
"""

import os

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "data/speakingfaces_raw"
ALIGNED = "data/speakingfaces_112"
OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "serif", "font.size": 8})


def rgb(path, gray=False):
    img = cv2.imread(path)
    if gray:
        img = cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def samples(subjects, frame="1_300"):
    cols = ["Visible\n(raw)", "Thermal\n(raw)", "Visible\n(aligned)", "Thermal\n(aligned)"]
    fig, axes = plt.subplots(len(subjects), 4, figsize=(3.45, 0.92 * len(subjects)),
                             gridspec_kw={"width_ratios": [1.33, 1.33, 1, 1]})
    for r, sub in enumerate(subjects):
        imgs = [rgb(f"{RAW}/sub_{sub}/trial_1/rgb_image_aligned/{frame}.png"),
                rgb(f"{RAW}/sub_{sub}/trial_1/thr_image/{frame}.png"),
                rgb(f"{ALIGNED}/vis/sub_{sub}/trial_1/{frame}.png"),
                rgb(f"{ALIGNED}/thr/sub_{sub}/trial_1/{frame}.png", gray=True)]
        for c, img in enumerate(imgs):
            ax = axes[r, c]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.06, left=0.01, right=0.99, top=0.90, bottom=0.01)
    fig.savefig(f"{OUT}/dataset_samples.pdf", dpi=300)


def poses(sub, trial=2, frame=300):
    fig, axes = plt.subplots(2, 9, figsize=(7.1, 1.75))
    for p in range(1, 10):
        for r, (mod, gray) in enumerate((("vis", False), ("thr", True))):
            ax = axes[r, p - 1]
            ax.imshow(rgb(f"{ALIGNED}/{mod}/sub_{sub}/trial_{trial}/{p}_{frame}.png", gray=gray))
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(f"Position {p}", fontsize=7)
        axes[0, 0].set_ylabel("Visible", fontsize=7)
        axes[1, 0].set_ylabel("Thermal", fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.04, left=0.03, right=0.995, top=0.88, bottom=0.01)
    fig.savefig(f"{OUT}/poses.pdf", dpi=300)


if __name__ == "__main__":
    samples([101, 117, 133])
    poses(124)
