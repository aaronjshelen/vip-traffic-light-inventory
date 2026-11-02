"""Download pretrained KASTEL MobilityLab traffic-light detection weights.

Source repo : https://github.com/KASTEL-MobilityLab/traffic-light-detection
Weights host: KIT BWSync&Share (public share, see repo's download_weights.sh)
Trained on  : DriveU Traffic Light Dataset (DTLD) -- German traffic lights.

These are the exact URLs from the upstream `model_weights/download_weights.sh`
(verified, not invented). The default `traffic_lights_yolov8x.pt` is a standard
single-head YOLOv8 detector and loads with stock `ultralytics` (no Docker, no
modified fork). The `*_split` model is intentionally NOT included here because
it requires KASTEL's modified ultralytics fork.

Usage:
    python download_weights.py                 # default: yolov8x
    python download_weights.py --model yolov8xl
    python download_weights.py --all
"""

import argparse
import os
import urllib.request

OUT_DIR = "weights"

# name -> (public URL, output filename)
WEIGHTS = {
    "yolov8x":  ("https://bwsyncandshare.kit.edu/s/ZLtoyf4q8nTLL7Y/download/traffic_lights_yolov8x.pt",
                 "traffic_lights_yolov8x.pt"),
    "yolov8xl": ("https://bwsyncandshare.kit.edu/s/RrbwrQ7iLktd87W/download/traffic_lights_yolov8xl.pt",
                 "traffic_lights_yolov8xl.pt"),
    "yolov9":   ("https://bwsyncandshare.kit.edu/s/Pc9jKBp6ncNbpf7/download/traffic_lights_yolov9.pt",
                 "traffic_lights_yolov9.pt"),
}


def _download(url: str, dest: str):
    if os.path.exists(dest):
        print(f"  already present: {dest}")
        return
    print(f"  downloading {url}\n           -> {dest}")

    def _hook(block, bsize, total):
        if total > 0:
            pct = min(100, 100 * block * bsize / total)
            print(f"\r    {pct:5.1f}%", end="", flush=True)

    urllib.request.urlretrieve(url, dest, _hook)
    print(f"\r    done ({os.path.getsize(dest) / 1e6:.1f} MB)")


def main():
    ap = argparse.ArgumentParser(description="Download KASTEL traffic-light weights")
    ap.add_argument("--model", choices=list(WEIGHTS), default="yolov8x")
    ap.add_argument("--all", action="store_true", help="Download all models")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    names = list(WEIGHTS) if args.all else [args.model]
    for name in names:
        url, fname = WEIGHTS[name]
        print(f"[{name}]")
        _download(url, os.path.join(OUT_DIR, fname))


if __name__ == "__main__":
    main()
