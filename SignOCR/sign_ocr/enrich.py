"""Post-process a SignOCR run: D-series classification, de-duplication, DA3 localization,
enriched signs.csv/json and report.html.

Runs automatically at the end of run_sign_ocr.py, or standalone on an existing output
folder via make_report.py (no need to re-run detection/tracking).
"""

from __future__ import annotations

import csv
import json
import os
import re

import cv2
import numpy as np

from sign_ocr.dseries import classify_dseries
from sign_ocr.report import write_html

CSV_FIELDS = ["track_id", "text", "is_d_series", "mutcd_code", "mutcd_family", "d_confidence",
              "ocr_confidence", "n_observations", "first_frame", "last_frame", "best_frame",
              "forward_m", "lateral_m", "height_m", "range_m", "lat", "lon",
              "description", "best_bbox", "best_crop"]


def _best_frame(s):
    if s.get("best_frame") is not None:
        return int(s["best_frame"])
    m = re.search(r"_f(\d+)", s.get("best_crop", ""))
    return int(m.group(1)) if m else int(s["first_frame"])


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _dedupe(signs, max_gap=60):
    """Merge ByteTrack fragments of one physical sign: same text, adjacent in time."""
    kept = []
    for s in sorted(signs, key=lambda s: s["first_frame"]):
        for k in kept:
            if _norm(k["text"]) == _norm(s["text"]) and s["first_frame"] - k["last_frame"] <= max_gap:
                k.setdefault("merged_tracks", []).append(s["track_id"])
                k["n_observations"] += s["n_observations"]
                k["last_frame"] = max(k["last_frame"], s["last_frame"])
                if s["ocr_confidence"] > k["ocr_confidence"]:   # keep the better read's crop
                    for f in ("ocr_confidence", "best_crop", "best_bbox", "best_frame"):
                        k[f] = s[f]
                break
        else:
            kept.append(s)
    return kept


def _read_frame(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None


def _resolve_focal(args, cap, width, total):
    from sign_ocr.localize import estimate_focal_da3, focal_from_hfov
    if args.focal_px:
        return args.focal_px, "user-supplied"
    fallback = focal_from_hfov(width, args.hfov)
    try:
        idxs = np.linspace(total * 0.1, total * 0.9, 4).astype(int)
        frames = [f for f in (_read_frame(cap, int(i)) for i in idxs) if f is not None]
        fx = estimate_focal_da3(frames, device=args.depth_device)
        hfov = np.degrees(2 * np.arctan(width / 2 / fx))
        if 40 <= hfov <= 150:
            return fx, f"estimated by DA3-BASE, HFOV {hfov:.0f}&deg;"
        print(f"[localize] DA3 focal {fx:.0f}px (HFOV {hfov:.0f}) implausible -> fallback")
    except Exception as e:  # noqa: BLE001 - localization must not kill the run
        print(f"[localize] focal estimation failed ({e}) -> fallback")
    return fallback, f"assumed HFOV {args.hfov:.0f}&deg;"


def enrich_run(out_dir, video, ocr, args):
    with open(os.path.join(out_dir, "signs.json"), encoding="utf-8") as f:
        signs = json.load(f)
    summary_path = os.path.join(out_dir, "summary.json")
    meta = {}
    if os.path.isfile(summary_path):
        with open(summary_path, encoding="utf-8") as f:
            meta = json.load(f)
    meta["video"] = video

    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ---- 1. re-OCR best crop (need per-word boxes) + D-series description ----
    d_signs, others = [], []
    for s in signs:
        s["best_frame"] = _best_frame(s)
        crop = cv2.imread(os.path.join(out_dir, s["best_crop"])) if s.get("best_crop") else None
        lines = [l for l in ocr.read(crop) if l.confidence > 0] if crop is not None else []
        s.update(classify_dseries(crop, lines))
        if s["rows"]:
            s["text"] = " / ".join(s["rows"])
            s["ocr_confidence"] = round(float(np.mean([l.confidence for l in lines])), 4)
        letters = sum(c.isalpha() for c in s["text"])
        if s["is_d_series"] and (letters < 3 or s["ocr_confidence"] < args.min_sign_conf):
            s["is_d_series"] = False
            s["description"] += " -> excluded: low OCR confidence / too little text"
        (d_signs if s["is_d_series"] else others).append(s)

    d_signs = _dedupe(d_signs)
    for s in d_signs + others:
        s["first_time"] = round(s["first_frame"] / fps, 2)
        s["last_time"] = round(s["last_frame"] / fps, 2)
        s.setdefault("location", None)
        s["geo"] = None

    # ---- 2. Depth Anything 3 localization on each sign's best frame ----
    if not args.no_depth and d_signs:
        from sign_ocr.localize import MetricDepth, sign_position, depth_preview, load_gps
        focal, src = _resolve_focal(args, cap, width, total)
        meta.update(focal_px=round(focal, 1), focal_source=src)
        print(f"[localize] focal = {focal:.0f}px ({src})")
        gps = load_gps(args.gps)
        depth = MetricDepth(focal, device=args.depth_device)
        assets = os.path.join(out_dir, "report_assets")
        os.makedirs(assets, exist_ok=True)
        for s in sorted(d_signs, key=lambda s: s["best_frame"]):
            frame = _read_frame(cap, s["best_frame"])
            if frame is None:
                continue
            dm = depth(frame)
            s["location"] = sign_position(dm, s["best_bbox"], focal) or None
            tag = f"track{int(s['track_id']):04d}_f{s['best_frame']:06d}"
            vis = frame.copy()
            x1, y1, x2, y2 = (int(v) for v in s["best_bbox"])
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.imwrite(os.path.join(assets, tag + "_frame.jpg"), vis)
            cv2.imwrite(os.path.join(assets, tag + "_depth.jpg"), depth_preview(dm, s["best_bbox"]))
            s["frame_preview"] = os.path.join("report_assets", tag + "_frame.jpg")
            s["depth_preview"] = os.path.join("report_assets", tag + "_depth.jpg")
            if gps is not None and s["location"]:
                s["geo"] = list(gps.sign_latlon(s["best_frame"] / fps, s["location"]["forward_m"],
                                                s["location"]["lateral_m"]))
            loc = s["location"] or {}
            print(f"  #{s['track_id']:>4} {s['mutcd_code']:<5} {s['text'][:40]!r:<44} "
                  f"Z={loc.get('forward_m', '-')}m X={loc.get('lateral_m', '-')}m")
    else:
        meta.setdefault("focal_px", 0)
        meta.setdefault("focal_source", "localization skipped")
    cap.release()

    # ---- 3. write outputs ----
    d_signs.sort(key=lambda s: s["first_frame"])
    allsigns = d_signs + others
    with open(os.path.join(out_dir, "signs.json"), "w", encoding="utf-8") as f:
        json.dump(allsigns, f, indent=2)
    with open(os.path.join(out_dir, "signs.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        for s in allsigns:
            loc = s.get("location") or {}
            geo = s.get("geo") or (None, None)
            w.writerow(dict(s, best_bbox=json.dumps(s["best_bbox"]), lat=geo[0], lon=geo[1],
                            **{k: loc.get(k) for k in ("forward_m", "lateral_m", "height_m", "range_m")}))
    report = os.path.join(out_dir, "report.html")
    write_html(report, out_dir, meta, d_signs, others)
    print(f"[report] {len(d_signs)} D-series signs, {len(others)} excluded -> {report}")
    return report


def add_enrich_args(p):
    g = p.add_argument_group("D-series / localization / report")
    g.add_argument("--no-depth", action="store_true", help="skip Depth Anything 3 localization")
    g.add_argument("--depth-device", default="cuda")
    g.add_argument("--focal-px", type=float, default=0.0,
                   help="camera focal length in px (default: estimate with DA3-BASE)")
    g.add_argument("--hfov", type=float, default=110.0,
                   help="fallback horizontal FOV (deg) if focal estimation fails")
    g.add_argument("--gps", default=None, help="GPS log CSV: t,lat,lon[,heading]")
    g.add_argument("--min-sign-conf", type=float, default=0.30,
                   help="min mean OCR confidence to keep a D-series sign")
    return p
