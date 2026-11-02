"""Traffic-light inventory: KASTEL+ByteTrack detections.csv -> one record per physical light.

    detections.csv (per frame)
      -> drop junk tracks (too short / never confident)
      -> split KASTEL class into INDICATION (circle / arrow_*) and STATE (red/yellow/green/off)
      -> indication from the CLOSEST frames (largest boxes), not majority vote
      -> state kept as a timeline (it changes within a track); German red_yellow -> red
      -> MoGe-2 metric depth on the closest frames -> camera-relative position
      -> GPS pose at that frame time -> light lat/lon (median over frames)
      -> merge ByteTrack fragments of one light (close in space, disjoint in time)
      -> lights.csv, lights.geojson, report.html (satellite map + per-light cards)

Run from KASTEL/ (uses KASTEL/.venv for MoGe):
  .venv/Scripts/python -m inventory.build_inventory --run outputs/gps_full \
      --video ../Recordings/2024_05_18_02_09_20_847/123456/IMEINotAvailable/camera/<...>.mp4
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import Counter, defaultdict

import cv2
import numpy as np

from inventory.gps import GpsTrack, find_loc_csv, haversine_m, recording_start_ms
from inventory.report import write_report

STATES = ("red_yellow", "red", "yellow", "green")


def parse_class(name: str):
    """KASTEL joint class -> (indication, state)."""
    if name == "off":
        return "unknown", "off"
    for st in STATES:
        if name.endswith("_" + st):
            return name[: -len(st) - 1], ("red" if st == "red_yellow" else st)
    return name, "unknown"


def load_tracks(det_csv):
    tracks = defaultdict(list)
    run_dir = os.path.dirname(det_csv)
    with open(det_csv, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["track_id"] in ("", "None", "-1"):
                continue
            box = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
            ind, st = parse_class(r["class_name"])
            crop = r.get("crop_path", "")   # stored relative to the run folder
            tracks[int(r["track_id"])].append(dict(
                frame=int(r["frame_index"]), t=float(r["timestamp_sec"]), box=box,
                area=(box[2] - box[0]) * (box[3] - box[1]), conf=float(r["confidence"]),
                cls=r["class_name"], indication=ind, state=st,
                crop=os.path.join(run_dir, crop.replace("\\", "/")) if crop else ""))
    for rows in tracks.values():
        rows.sort(key=lambda d: d["frame"])
    return tracks


def state_timeline(rows, min_run=5):
    """Run-length encode per-frame state; runs shorter than min_run are absorbed (flicker)."""
    runs = []
    for d in rows:
        if runs and runs[-1]["state"] == d["state"]:
            runs[-1].update(t1=d["t"], n=runs[-1]["n"] + 1)
        else:
            runs.append(dict(state=d["state"], t0=d["t"], t1=d["t"], n=1))
    kept = [r for r in runs if r["n"] >= min_run] or runs
    merged = []
    for r in kept:
        if merged and merged[-1]["state"] == r["state"]:
            merged[-1].update(t1=r["t1"], n=merged[-1]["n"] + r["n"])
        else:
            merged.append(dict(r))
    return [dict(state=r["state"], t0=round(r["t0"], 2), t1=round(r["t1"], 2)) for r in merged]


def summarize_track(tid, rows, top_k):
    closest = sorted(rows, key=lambda d: d["area"], reverse=True)[:top_k]
    vote = Counter()
    for d in closest:
        vote[d["indication"]] += d["conf"]
    indication, w = vote.most_common(1)[0]
    majority = Counter(d["indication"] for d in rows).most_common(1)[0][0]
    best = closest[0]
    return dict(
        light_id=tid, track_ids=[tid], n_frames=len(rows),
        first_t=round(rows[0]["t"], 2), last_t=round(rows[-1]["t"], 2),
        indication=indication, indication_share=round(w / sum(vote.values()), 2),
        indication_majority_all_frames=majority,
        states=state_timeline(rows), best_frame=best["frame"], best_box=[round(v, 1) for v in best["box"]],
        best_class=best["cls"], best_crop=best["crop"], max_conf=round(max(d["conf"] for d in rows), 3),
        max_box_h=round(max(d["box"][3] - d["box"][1] for d in rows), 1))


def pick_loc_frames(rows, k, min_gap):
    out = []
    for d in sorted(rows, key=lambda d: d["area"], reverse=True):
        if all(abs(d["frame"] - o["frame"]) >= min_gap for o in out):
            out.append(d)
        if len(out) == k:
            break
    return out


def localize(lights, tracks, args, video, gps, fps, out_dir):
    from inventory.moge_depth import MoGe2, box_position, depth_vis
    moge = MoGe2(device=args.device)
    cap = cv2.VideoCapture(video)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def read(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, fr = cap.read()
        return fr if ok else None

    if args.fov_x:
        moge.fov_x = args.fov_x
        fov_src = "user-supplied"
    else:
        sample = [read(int(i)) for i in np.linspace(total * 0.1, total * 0.9, 6)]
        moge.calibrate_fov([f for f in sample if f is not None])
        fov_src = "estimated by MoGe-2 (median of 6 frames)"
    print(f"[localize] horizontal FOV = {moge.fov_x:.1f} deg ({fov_src})")

    need = defaultdict(list)   # frame -> [(light, row)]
    for L in lights:
        for d in pick_loc_frames(tracks[L["light_id"]], args.loc_frames, args.loc_min_gap):
            need[d["frame"]].append((L, d))
    os.makedirs(os.path.join(out_dir, "assets"), exist_ok=True)
    for i, fi in enumerate(sorted(need)):
        frame = read(fi)
        if frame is None:
            continue
        dm, intr = moge.depth(frame)
        lat_v, lon_v, hdg = gps.pose(fi / fps)
        for L, d in need[fi]:
            p = box_position(dm, intr, d["box"])
            if p is None:
                continue
            lat, lon = GpsTrack.offset(lat_v, lon_v, hdg, p["forward_m"], p["right_m"])
            L.setdefault("_obs", []).append(dict(frame=fi, lat=lat, lon=lon, **p))
            if fi == L["best_frame"] or "depth_preview" not in L:
                x1, y1, x2, y2 = (int(v) for v in d["box"])
                pad = max(40, 3 * (y2 - y1))
                win = (slice(max(0, y1 - pad), y2 + pad), slice(max(0, x1 - pad), x2 + pad))
                vis_rgb, vis_d = frame.copy(), depth_vis(dm)
                for im in (vis_rgb, vis_d):
                    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
                name = f"assets/light{L['light_id']:05d}"
                cv2.imwrite(os.path.join(out_dir, name + "_ctx.jpg"), vis_rgb[win])
                cv2.imwrite(os.path.join(out_dir, name + "_depth.jpg"), vis_d[win])
                L["context_preview"], L["depth_preview"] = name + "_ctx.jpg", name + "_depth.jpg"
        print(f"\r[localize] frame {i + 1}/{len(need)}", end="")
    print()
    cap.release()

    for L in lights:
        obs = L.pop("_obs", [])
        if not obs:
            L["location"] = None
            continue
        lat, lon = float(np.median([o["lat"] for o in obs])), float(np.median([o["lon"] for o in obs]))
        L["location"] = dict(
            lat=round(lat, 7), lon=round(lon, 7), n_obs=len(obs),
            spread_m=round(max(haversine_m(lat, lon, o["lat"], o["lon"]) for o in obs), 1),
            closest_forward_m=round(min(o["forward_m"] for o in obs), 1),
            height_m=round(float(np.median([o["up_m"] for o in obs])), 1))
    return dict(fov_x=round(moge.fov_x, 1), fov_source=fov_src)


def merge_fragments(lights, max_m):
    """Same physical light split into several tracks: close in space AND disjoint in time."""
    lights = sorted(lights, key=lambda L: L["first_t"])
    kept = []
    for L in lights:
        for K in kept:
            if (L["location"] and K["location"] and L["indication"] == K["indication"]
                    and L["first_t"] > K["last_t"]
                    and haversine_m(L["location"]["lat"], L["location"]["lon"],
                                    K["location"]["lat"], K["location"]["lon"]) <= max_m):
                K["track_ids"] += L["track_ids"]
                K["n_frames"] += L["n_frames"]
                K["last_t"] = L["last_t"]
                for s in L["states"]:
                    if K["states"][-1]["state"] == s["state"]:
                        K["states"][-1]["t1"] = s["t1"]
                    else:
                        K["states"].append(s)
                break
        else:
            kept.append(L)
    for n, L in enumerate(kept, 1):
        L["light_no"] = n
    return kept


def main():
    p = argparse.ArgumentParser(description="KASTEL detections -> traffic-light inventory")
    p.add_argument("--run", required=True, help="run_baseline output dir (has detections.csv)")
    p.add_argument("--video", required=True)
    p.add_argument("--loc-csv", default=None, help="GPS *_loc.csv (default: found next to the video)")
    p.add_argument("--utc-offset", type=float, default=-7.0, help="recording local time zone (Tucson -7)")
    p.add_argument("--min-frames", type=int, default=15, help="drop tracks shorter than this")
    p.add_argument("--min-conf", type=float, default=0.45, help="drop tracks never above this conf")
    p.add_argument("--top-k", type=int, default=5, help="closest frames voting on indication")
    p.add_argument("--loc-frames", type=int, default=3, help="frames per light for localization")
    p.add_argument("--loc-min-gap", type=int, default=10, help="min frame gap between those")
    p.add_argument("--fov-x", type=float, default=0.0, help="known horizontal FOV (deg); 0 = estimate")
    p.add_argument("--merge-m", type=float, default=3.0, help="merge fragments closer than this (m)")
    p.add_argument("--no-localize", action="store_true")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None, help="output dir (default: <run>/inventory)")
    args = p.parse_args()

    out_dir = args.out or os.path.join(args.run, "inventory")
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    tracks = load_tracks(os.path.join(args.run, "detections.csv"))
    lights = [summarize_track(tid, rows, args.top_k) for tid, rows in tracks.items()
              if len(rows) >= args.min_frames and max(d["conf"] for d in rows) >= args.min_conf]
    print(f"[inventory] {len(tracks)} tracks -> {len(lights)} after junk filter "
          f"(>= {args.min_frames} frames, max conf >= {args.min_conf})")

    gps = GpsTrack(args.loc_csv or find_loc_csv(args.video), recording_start_ms(args.video, args.utc_offset))
    meta = dict(video=args.video, n_tracks=len(tracks), fps=fps, fov_x=None, fov_source="not localized")
    if not args.no_localize:
        meta.update(localize(lights, tracks, args, args.video, gps, fps, out_dir))
        n0 = len(lights)
        lights = merge_fragments(lights, args.merge_m)
        print(f"[inventory] merged fragments: {n0} -> {len(lights)} lights")
    else:
        for n, L in enumerate(sorted(lights, key=lambda L: L["first_t"]), 1):
            L.update(light_no=n, location=None)
        lights.sort(key=lambda L: L["light_no"])

    # ---- outputs ----
    with open(os.path.join(out_dir, "lights.json"), "w", encoding="utf-8") as f:
        json.dump(dict(meta=meta, lights=lights), f, indent=2)
    fields = ["light_no", "track_ids", "indication", "indication_share", "indication_majority_all_frames",
              "states", "n_frames", "first_t", "last_t", "max_conf", "max_box_h", "best_frame",
              "lat", "lon", "closest_forward_m", "height_m", "spread_m", "best_crop"]
    with open(os.path.join(out_dir, "lights.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for L in lights:
            w.writerow(dict(L, **(L["location"] or {}),
                            track_ids=" ".join(map(str, L["track_ids"])),
                            states=" > ".join(f'{s["state"]}@{s["t0"]:.0f}s' for s in L["states"])))
    feats = [dict(type="Feature", geometry=dict(type="Point", coordinates=[L["location"]["lon"], L["location"]["lat"]]),
                  properties=dict(light_no=L["light_no"], indication=L["indication"], track_ids=L["track_ids"]))
             for L in lights if L["location"]]
    with open(os.path.join(out_dir, "lights.geojson"), "w", encoding="utf-8") as f:
        json.dump(dict(type="FeatureCollection", features=feats), f)
    write_report(os.path.join(out_dir, "report.html"), out_dir, meta, lights, gps.path())
    print(f"[inventory] {len(lights)} lights -> {out_dir}/ (lights.csv, lights.geojson, report.html)")
    for L in lights:
        loc = L["location"] or {}
        print(f"  #{L['light_no']:>3} tracks {L['track_ids']} {L['indication']:<20} "
              f"{' > '.join(s['state'] for s in L['states']):<24} "
              f"{loc.get('lat', '-')}, {loc.get('lon', '-')}  spread {loc.get('spread_m', '-')} m")


if __name__ == "__main__":
    main()
