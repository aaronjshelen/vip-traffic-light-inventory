"""Re-generate D-series classification, localization and report.html for an existing run.

Uses the crops + signs.json already in the output folder; detection/tracking is not re-run.

Example:
  python make_report.py --output outputs/clip_0718_0820 --video ../Recordings/clip_0718_0820.mp4
"""

from __future__ import annotations

import argparse

from sign_ocr.enrich import add_enrich_args, enrich_run
from sign_ocr.ocr import build_ocr_engine


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", required=True, help="existing run_sign_ocr.py output folder")
    p.add_argument("--video", required=True, help="the video that run was made from")
    p.add_argument("--ocr", choices=["paddle", "easy"], default="easy")
    args = add_enrich_args(p).parse_args()
    ocr = build_ocr_engine(prefer=args.ocr, device="cpu" if args.depth_device == "cpu" else "gpu")
    enrich_run(args.output, args.video, ocr, args)


if __name__ == "__main__":
    main()
