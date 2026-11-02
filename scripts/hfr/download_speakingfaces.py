"""
Download a frame subset of SpeakingFaces (thermal + registered visible) without fetching the full 1.2 TB.

SpeakingFaces (Abdrakhmanova et al., Sensors 2021) is released under CC BY 4.0 on an ungated
Hugging Face repo. Each subject is one zip; we read individual PNGs from it with HTTP range
requests (remotezip), so only the sampled frames are transferred.

File names inside the zips: {sub}_{trial}_{session}_{position}_{frame}_{stream}.png
  stream 1 = thermal (thr_image), 3 = visible registered to the thermal view (rgb_image_aligned)
  position 1..9 (1 is frontal), frame 1..900
"""

import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from remotezip import RemoteZip

REPO = "issai/Speaking_Faces"
BASE = f"https://huggingface.co/datasets/{REPO}/resolve/main/image_only"
STREAMS = {"thr_image": 1, "rgb_image_aligned": 3}


def list_subjects():
    r = requests.get(f"https://huggingface.co/api/datasets/{REPO}/tree/main/image_only", timeout=60)
    r.raise_for_status()
    names = [os.path.basename(f["path"]) for f in r.json() if f["path"].endswith("_io.zip")]
    return sorted(int(n.split("_")[1]) for n in names)


def download_subject(sub, out_dir, frames, positions, trials):
    done_flag = os.path.join(out_dir, f"sub_{sub}", ".done")
    if os.path.exists(done_flag):
        return sub, 0, 0
    zf = RemoteZip(f"{BASE}/sub_{sub}_io.zip")
    members = set(zf.namelist())
    n_ok, n_missing = 0, 0
    for trial in trials:
        for pos in positions:
            for frame in frames:
                for stream_dir, sid in STREAMS.items():
                    name = f"sub_{sub}_io/trial_{trial}/{stream_dir}/{sub}_{trial}_1_{pos}_{frame}_{sid}.png"
                    if name not in members:
                        n_missing += 1
                        continue
                    dst = os.path.join(out_dir, f"sub_{sub}", f"trial_{trial}", stream_dir,
                                       f"{pos}_{frame}.png")
                    if not os.path.exists(dst):
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        # write-then-rename so an interrupted run never leaves a truncated PNG
                        with open(dst + ".part", "wb") as f:
                            f.write(zf.read(name))
                        os.replace(dst + ".part", dst)
                    n_ok += 1
    zf.close()
    open(done_flag, "w").close()
    return sub, n_ok, n_missing


def main(args):
    frames = [int(f) for f in args.frames.split(",")]
    positions = list(range(1, 10))
    trials = [1, 2]
    subjects = list_subjects()
    print(f"{len(subjects)} subjects, frames {frames}, {len(positions)} positions, trials {trials}", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(download_subject, s, args.out_dir, frames, positions, trials) for s in subjects]
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                sub, ok, missing = fut.result()
                print(f"[{i}/{len(subjects)}] sub {sub}: {ok} files, {missing} missing", flush=True)
            except Exception as e:
                print(f"[{i}/{len(subjects)}] FAILED: {e!r}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="data/speakingfaces_raw")
    parser.add_argument("--frames", default="100,300,500,700", help="comma-separated frame ids to sample")
    parser.add_argument("--workers", type=int, default=8)
    main(parser.parse_args())
