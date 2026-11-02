"""
Align SpeakingFaces thermal/visible pairs to 112x112 crops.

The visible stream used here (rgb_image_aligned) is already registered pixel-to-pixel to the
thermal view, so MTCNN landmarks are detected on the visible frame only and the same similarity
transform is applied to both images. This avoids running a face detector on thermal images.

Output layout: <out_dir>/{vis,thr}/sub_<id>/trial_<t>/<pos>_<frame>.png
"""

import argparse
import glob
import os
import sys

import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from face_alignment import mtcnn
from face_alignment.mtcnn_pytorch.src.align_trans import warp_and_crop_face


def largest_face_landmarks(detector, img):
    boxes, landmarks = detector.detect_faces(img, detector.min_face_size, detector.thresholds,
                                             detector.nms_thresholds, detector.factor)
    if len(boxes) == 0:
        return None
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    lm = landmarks[int(np.argmax(areas))]
    return [[lm[j], lm[j + 5]] for j in range(5)]


def main(args):
    detector = mtcnn.MTCNN(device=args.device, crop_size=(112, 112))
    vis_paths = sorted(glob.glob(os.path.join(args.raw_dir, "sub_*", "trial_*", "rgb_image_aligned", "*.png")))
    failed = []
    for vis_path in tqdm(vis_paths, mininterval=30):
        rel = os.path.relpath(vis_path, args.raw_dir).split(os.sep)
        sub, trial, fname = rel[0], rel[1], rel[3]
        out_vis = os.path.join(args.out_dir, "vis", sub, trial, fname)
        out_thr = os.path.join(args.out_dir, "thr", sub, trial, fname)
        if os.path.exists(out_vis) and os.path.exists(out_thr):
            continue
        thr_path = os.path.join(args.raw_dir, sub, trial, "thr_image", fname)
        if not os.path.exists(thr_path):
            failed.append((vis_path, "no thermal pair"))
            continue

        vis = Image.open(vis_path).convert("RGB")
        points = largest_face_landmarks(detector, vis)
        if points is None:
            failed.append((vis_path, "no face"))
            continue
        thr = np.array(Image.open(thr_path).convert("RGB"))

        for out_path, img in [(out_vis, np.array(vis)), (out_thr, thr)]:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            crop = warp_and_crop_face(img, points, detector.refrence, crop_size=(112, 112))
            Image.fromarray(crop).save(out_path)

    with open(os.path.join(args.out_dir, "failed.txt"), "w") as f:
        for path, reason in failed:
            f.write(f"{path}\t{reason}\n")
    print(f"aligned {len(vis_paths) - len(failed)} / {len(vis_paths)} pairs, {len(failed)} failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw_dir", default="data/speakingfaces_raw")
    parser.add_argument("--out_dir", default="data/speakingfaces_112")
    parser.add_argument("--device", default="cuda:0", choices=["cuda:0", "cpu"])
    main(parser.parse_args())
