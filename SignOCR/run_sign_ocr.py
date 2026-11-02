"""Directional-sign OCR pipeline — fully automatic, standalone (separate from KASTEL).

    dashcam .mp4
      -> YOLO-World open-vocabulary sign-PANEL detection (US directional prompts)
      -> ByteTrack (persistent track_id per physical sign)
      -> automatic best-crop selection (area x sharpness x conf)
      -> OCR (EasyOCR; PaddleOCR adapter also supported) on the best crops
      -> aggregate repeated observations into ONE record per sign
      -> MUTCD D-series description/classification (colour + text rows + arrows)
      -> Depth Anything 3 metric depth -> camera-relative sign position (+ GPS when available)
      -> annotated_video.mp4 + signs.csv + signs.json + observations.csv
         + crops/ + summary.json + report.html

No manual frame selection, cropping, or transcription. One command.

Modes:
  panel     (default) detect sign panels, track, OCR the panel crops.
  scenetext fallback: run OCR directly on (downscaled) full frames — no panel
            detection/tracking, just "find and read directional text".
  auto      probe a few frames; use panel if it finds enough, else scenetext.

Example:
  python run_sign_ocr.py --video "../Recordings/<clip>.mp4" --output outputs/run1
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import cv2
from tqdm import tqdm

from sign_ocr.detector import SignPanelDetector, DEFAULT_PROMPTS, crop_from, sharpness, quality_score
from sign_ocr.ocr import build_ocr_engine
from sign_ocr.dseries import group_rows
from sign_ocr.enrich import enrich_run, add_enrich_args


# ---- OCR on a crop -> joined text + confidence ---------------------------

def ocr_crop(engine, crop):
    """Return (joined_text, mean_conf, raw_lines) for one crop image."""
    if crop is None or crop.size == 0:
        return "", 0.0, []
    lines = engine.read(crop)
    lines = [l for l in lines if l.text and l.confidence > 0.0]
    if not lines:
        return "", 0.0, []
    # reading order: rows top-to-bottom, words left-to-right within a row
    joined = " | ".join(" ".join(l.text for l in row) for row in group_rows(lines))
    mean_conf = sum(l.confidence for l in lines) / len(lines)
    return joined, mean_conf, lines


# ---- drawing --------------------------------------------------------------

def draw(frame, box, label, color, thick):
    p1 = (int(box[0]), int(box[1]))
    p2 = (int(box[2]), int(box[3]))
    cv2.rectangle(frame, p1, p2, color, thick)
    if label:
        fs = 0.6
        (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
        ty = max(p1[1], th + 4)
        cv2.rectangle(frame, (p1[0], ty - th - bl), (p1[0] + tw, ty + bl), color, -1)
        cv2.putText(frame, label, (p1[0], ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), 1, cv2.LINE_AA)


# ---- main -----------------------------------------------------------------

def run(args) -> int:
    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}")
        return 1

    os.makedirs(args.output, exist_ok=True)
    crops_dir = os.path.join(args.output, "crops")
    os.makedirs(crops_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        return 1
    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end_frame = min(total, args.max_frames * args.stride) if args.max_frames else total

    device = args.device
    if device is None:
        import torch
        device = 0 if torch.cuda.is_available() else "cpu"

    print("=" * 66)
    print("DIRECTIONAL-SIGN OCR PIPELINE")
    print("=" * 66)
    print(f"  video    : {args.video}")
    print(f"  frames   : {width}x{height} @ {in_fps:.2f}fps, {total} total")
    print(f"  segment  : [0,{end_frame}) stride={args.stride}  device={device}")
    print(f"  mode     : {args.mode}   imgsz={args.imgsz} conf={args.conf}")

    ocr = build_ocr_engine(prefer=args.ocr, device=("gpu" if device != "cpu" else "cpu"))

    detector = None
    mode = args.mode
    if mode in ("panel", "auto"):
        detector = SignPanelDetector(prompts=args.prompts, imgsz=args.imgsz,
                                     conf=args.conf, device=device)
        print(f"  prompts  : {detector.prompts}")
        if mode == "auto":
            mode = _probe_mode(cap, detector, end_frame)
            print(f"  auto -> selected mode: {mode}")
    print("-" * 66)

    writer = None
    if not args.no_video:
        out_fps = max(1.0, in_fps / args.stride)
        writer = cv2.VideoWriter(os.path.join(args.output, "annotated_video.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (width, height))
    thick = max(2, int(round((width + height) / 1000)))

    observations = []            # raw OCR reads
    # per-track consolidation
    best = {}                    # tid -> dict(quality,text,conf,crop,bbox,frame,lines)
    seen_frames = defaultdict(list)   # tid -> [frame_index,...]
    display_text = {}            # tid -> current best text (for drawing)
    n_ocr_calls = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    fi = 0
    t0 = time.time()
    pbar = tqdm(total=end_frame // max(1, args.stride), unit="f")

    while fi < end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if fi % args.stride != 0:
            fi += 1
            continue
        ts = fi / in_fps

        if mode == "scenetext":
            _process_scenetext(frame, fi, ts, ocr, args, observations, writer, thick)
            n_ocr_calls += 1
        else:
            panels = detector.track(frame)
            clean = frame.copy()   # crop from this: `frame` gets labels drawn on it below
            for p in panels:
                box = [p.x1, p.y1, p.x2, p.y2]
                crop = crop_from(clean, box, pad=4)
                q = quality_score(crop, p.confidence)
                tid = p.track_id
                color = (0, 200, 0)
                if tid is not None:
                    seen_frames[tid].append(fi)
                    prev = best.get(tid)
                    # OCR when this is the first or a clearly better crop for the track
                    if crop is not None and (prev is None or q > prev["quality"] * 1.15):
                        text, conf, lines = ocr_crop(ocr, crop)
                        n_ocr_calls += 1
                        if text:
                            cpath = os.path.join(crops_dir, f"track{tid:04d}_f{fi:06d}.jpg")
                            cv2.imwrite(cpath, crop)
                            rel = os.path.relpath(cpath, args.output)
                            observations.append(dict(
                                frame=fi, timestamp=round(ts, 3), track_id=tid,
                                x1=round(p.x1, 1), y1=round(p.y1, 1), x2=round(p.x2, 1),
                                y2=round(p.y2, 1), det_conf=round(p.confidence, 4),
                                quality=round(q, 1), ocr_text=text,
                                ocr_conf=round(conf, 4), crop_path=rel,
                                class_name=p.class_name))
                            # keep the crop with best (quality) as the track's representative
                            if prev is None or q > prev["quality"]:
                                best[tid] = dict(quality=q, text=text, conf=conf,
                                                 crop=rel, bbox=[round(v, 1) for v in box],
                                                 frame=fi, class_name=p.class_name)
                                display_text[tid] = text
                    label = f"#{tid} {display_text.get(tid, p.class_name)}"[:48]
                else:
                    label = p.class_name[:24]
                if writer is not None:
                    draw(frame, box, label, color, thick)
        if writer is not None and mode != "scenetext":
            writer.write(frame)
        fi += 1
        pbar.update(1)
        pbar.set_postfix(ocr=n_ocr_calls, signs=len(best))

    pbar.close()
    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t0

    # ---- aggregate + write ----
    signs = _aggregate(mode, best, seen_frames, observations)
    _write_outputs(args.output, observations, signs, mode)

    summary = dict(
        video=args.video, mode=mode, device=str(device),
        ocr_backend=getattr(ocr, "name", "?"),
        frames_processed=fi // max(1, args.stride),
        stride=args.stride, imgsz=args.imgsz, conf=args.conf,
        ocr_calls=n_ocr_calls, raw_observations=len(observations),
        unique_signs=len(signs), elapsed_sec=round(elapsed, 1),
        fps=round((fi / max(1, args.stride)) / elapsed, 2) if elapsed else 0,
    )
    with open(os.path.join(args.output, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("-" * 66)
    print(f"  mode={mode} backend={summary['ocr_backend']}  elapsed={elapsed:.1f}s "
          f"({summary['fps']} fps)")
    print(f"  OCR calls={n_ocr_calls}  raw observations={len(observations)}  "
          f"unique signs={len(signs)}")
    print(f"  outputs -> {args.output}")
    for s in signs[:12]:
        print(f"    sign {s.get('track_id','-')}: {s['text']!r} "
              f"(conf {s['ocr_confidence']}, n={s['n_observations']})")
    print("=" * 66)
    if mode == "panel" and not args.no_report:
        enrich_run(args.output, args.video, ocr, args)
    return 0


def _probe_mode(cap, detector, end_frame, n=10, min_hits=3):
    import numpy as np
    idxs = np.linspace(0, max(1, end_frame - 1), n).astype(int)
    hits = 0
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, fr = cap.read()
        if ok and detector.detect(fr):
            hits += 1
    return "panel" if hits >= min_hits else "scenetext"


def _process_scenetext(frame, fi, ts, ocr, args, observations, writer, thick):
    # downscale wide frames for speed; map boxes back
    h, w = frame.shape[:2]
    scale = min(1.0, args.scenetext_width / w)
    small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
    for l in ocr.read(small):
        if l.confidence < args.min_text_conf:
            continue
        box = [v / scale for v in l.box]
        if (box[3] - box[1]) < args.min_text_h:   # skip tiny text
            continue
        observations.append(dict(
            frame=fi, timestamp=round(ts, 3), track_id="",
            x1=round(box[0], 1), y1=round(box[1], 1), x2=round(box[2], 1), y2=round(box[3], 1),
            det_conf="", quality="", ocr_text=l.text, ocr_conf=round(l.confidence, 4),
            crop_path="", class_name="text"))
        if writer is not None:
            draw(frame, box, f"{l.text}", (0, 180, 255), thick)
    if writer is not None:
        writer.write(frame)


def _aggregate(mode, best, seen_frames, observations):
    signs = []
    if mode == "scenetext":
        # group identical (normalized) text strings
        groups = defaultdict(list)
        for o in observations:
            key = o["ocr_text"].strip().lower()
            groups[key].append(o)
        for key, obs in groups.items():
            bestobs = max(obs, key=lambda o: o["ocr_conf"])
            frames = [o["frame"] for o in obs]
            signs.append(dict(
                track_id="", text=bestobs["ocr_text"],
                ocr_confidence=round(sum(o["ocr_conf"] for o in obs) / len(obs), 4),
                n_observations=len(obs), first_frame=min(frames), last_frame=max(frames),
                best_crop="", best_bbox=[bestobs["x1"], bestobs["y1"], bestobs["x2"], bestobs["y2"]],
                sign_type="text", geo=None))
        signs.sort(key=lambda s: s["n_observations"], reverse=True)
    else:
        for tid, b in best.items():
            frames = seen_frames.get(tid, [b["frame"]])
            signs.append(dict(
                track_id=tid, text=b["text"], ocr_confidence=round(b["conf"], 4),
                n_observations=len(frames), first_frame=min(frames), last_frame=max(frames),
                best_crop=b["crop"], best_bbox=b["bbox"], best_frame=b["frame"],
                sign_type=b["class_name"], geo=None))
        signs.sort(key=lambda s: (s["ocr_confidence"], s["n_observations"]), reverse=True)
    return signs


def _write_outputs(out, observations, signs, mode):
    obs_fields = ["frame", "timestamp", "track_id", "x1", "y1", "x2", "y2",
                  "det_conf", "quality", "ocr_text", "ocr_conf", "crop_path", "class_name"]
    with open(os.path.join(out, "observations.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=obs_fields)
        w.writeheader()
        w.writerows(observations)

    sign_fields = ["track_id", "text", "ocr_confidence", "n_observations",
                   "first_frame", "last_frame", "best_frame", "sign_type", "best_bbox", "best_crop", "geo"]
    with open(os.path.join(out, "signs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=sign_fields)
        w.writeheader()
        for s in signs:
            row = dict(s)
            row["best_bbox"] = json.dumps(row["best_bbox"])
            w.writerow(row)
    with open(os.path.join(out, "signs.json"), "w", encoding="utf-8") as f:
        json.dump(signs, f, indent=2)


def build_parser():
    p = argparse.ArgumentParser(description="Directional-sign OCR pipeline")
    p.add_argument("--video", required=True)
    p.add_argument("--output", default="outputs/run")
    p.add_argument("--mode", choices=["panel", "scenetext", "auto"], default="panel")
    p.add_argument("--max-frames", type=int, default=0, help="cap processed frames (0=all)")
    p.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    p.add_argument("--device", default=None, help="'0' GPU, 'cpu', or auto")
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.05, help="panel detection conf threshold")
    p.add_argument("--ocr", choices=["paddle", "easy"], default="easy")
    p.add_argument("--prompts", nargs="*", default=DEFAULT_PROMPTS)
    p.add_argument("--no-video", action="store_true")
    # scenetext knobs
    p.add_argument("--scenetext-width", type=int, default=1280)
    p.add_argument("--min-text-conf", type=float, default=0.4)
    p.add_argument("--min-text-h", type=int, default=14)
    p.add_argument("--no-report", action="store_true",
                   help="skip D-series classification, localization and report.html")
    return add_enrich_args(p)


if __name__ == "__main__":
    raise SystemExit(run(build_parser().parse_args()))
