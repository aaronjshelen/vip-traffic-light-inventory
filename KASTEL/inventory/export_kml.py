"""Export the inventory + your labels to a Google Earth KML, and check label grouping.

Grouping check: every Frames-tab box verdict (tp / fp / skip) is mapped through its
ByteTrack ID to the inventory light that owns that track, then compared with that
light's Lights-tab label (real yes / no). Conflicts are printed and listed in the KML.

KML folders:
  Drive path               - vehicle GPS track
  Lights                   - estimated light position (MoGe-2 + GPS, NOT yet validated),
                             green = you labeled real, red = not real, grey = unlabeled
  Car at closest view      - GPS position of the car at each light's best frame (reliable)
  Sight lines              - car-at-closest-view -> estimated light

  .venv/Scripts/python -m inventory.export_kml --run outputs/gps_full --video <mp4> [--real-only]
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from xml.sax.saxutils import escape

from inventory.build_inventory import load_tracks
from inventory.gps import GpsTrack, find_loc_csv, recording_start_ms

STYLES = {"yes": "ff00c800", "no": "ff0000ff", None: "ff9e9e9e"}  # KML colors are aabbggrr


def box_verdicts_by_track(labels, tracks):
    """Frames-tab verdicts keyed by track ID, using the labeler's box numbering
    (boxes in a frame sorted left to right by x1, numbered from 1)."""
    by_frame = defaultdict(list)
    for tid, rows in tracks.items():
        for d in rows:
            by_frame[d["frame"]].append((tid, d))
    out = defaultdict(Counter)
    for fi, fl in labels["frames"].items():
        dets = sorted(by_frame.get(int(fi), []), key=lambda x: x[1]["box"][0])
        for k, v in fl.get("boxes", {}).items():
            k = int(k)
            if 1 <= k <= len(dets) and v.get("v"):
                out[dets[k - 1][0]][v["v"]] += 1
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--labels", default=None, help="default: <run>/labeling/labels.json")
    p.add_argument("--utc-offset", type=float, default=-7.0)
    p.add_argument("--real-only", action="store_true", help="only lights you labeled real (-> lights_real.kml)")
    args = p.parse_args()

    with open(os.path.join(args.run, "inventory", "lights.json"), encoding="utf-8") as f:
        inv = json.load(f)
    with open(args.labels or os.path.join(args.run, "labeling", "labels.json"), encoding="utf-8") as f:
        labels = json.load(f)
    tracks = load_tracks(os.path.join(args.run, "detections.csv"))
    gps = GpsTrack(find_loc_csv(args.video), recording_start_ms(args.video, args.utc_offset))
    verdicts = box_verdicts_by_track(labels, tracks)

    owner = {t: L["light_no"] for L in inv["lights"] for t in L["track_ids"]}
    orphan = Counter()  # frame verdicts on tracks that are not in the inventory (filtered out)
    for tid, c in verdicts.items():
        if tid not in owner:
            orphan.update(c)

    lights_kml, cars_kml, lines_kml, conflicts = [], [], [], []
    for L in inv["lights"]:
        no = L["light_no"]
        lab = labels["lights"].get(str(no), {})
        real = lab.get("real")
        if args.real_only and real != "yes":
            continue
        fv = Counter()
        for t in L["track_ids"]:
            fv.update(verdicts.get(t, {}))
        if (real == "yes" and fv["fp"]) or (real == "no" and fv["tp"]):
            conflicts.append((no, real, dict(fv)))
        best_t = next(d["t"] for t in L["track_ids"] for d in tracks[t] if d["frame"] == L["best_frame"])
        clat, clon, _ = gps.pose(best_t)
        loc = L.get("location") or {}
        desc = (f"Tracks: {L['track_ids']}<br>Seen {L['first_t']:.1f}–{L['last_t']:.1f} s "
                f"({L['n_frames']} frames), closest at {best_t:.1f} s<br>"
                f"Model: {L['indication']} / {L['best_class']}<br>"
                f"Your label: real={real or '—'}, ind={lab.get('ind', '—')}, state={lab.get('state', '—')}<br>"
                f"Frame-box verdicts: {dict(fv) or 'none sampled'}<br>"
                f"Position: estimated (MoGe-2 + GPS, unvalidated), spread {loc.get('spread_m', '—')} m, "
                f"{loc.get('closest_forward_m', '—')} m ahead at closest view")
        name = f"L{no} {lab.get('ind') or L['indication']}"
        if loc.get("lat") is not None:
            lights_kml.append(
                f"<Placemark><name>{escape(name)}</name><styleUrl>#real_{real or 'none'}</styleUrl>"
                f"<description><![CDATA[{desc}]]></description>"
                f"<Point><coordinates>{loc['lon']},{loc['lat']},0</coordinates></Point></Placemark>")
            lines_kml.append(
                f"<Placemark><name>L{no}</name><styleUrl>#sight</styleUrl><LineString><coordinates>"
                f"{clon},{clat},0 {loc['lon']},{loc['lat']},0</coordinates></LineString></Placemark>")
        cars_kml.append(
            f"<Placemark><name>L{no} @ {best_t:.0f}s</name><styleUrl>#car</styleUrl>"
            f"<description><![CDATA[{desc}]]></description>"
            f"<Point><coordinates>{clon},{clat},0</coordinates></Point></Placemark>")

    path = " ".join(f"{lon},{lat},0" for lat, lon in gps.path())
    styles = "".join(
        f'<Style id="real_{k or "none"}"><IconStyle><color>{c}</color><scale>1.0</scale><Icon><href>'
        f"http://maps.google.com/mapfiles/kml/paddle/wht-blank.png</href></Icon></IconStyle></Style>"
        for k, c in STYLES.items())
    styles += ('<Style id="car"><IconStyle><color>ffffaa00</color><scale>0.6</scale><Icon><href>'
               "http://maps.google.com/mapfiles/kml/shapes/placemark_circle.png</href></Icon></IconStyle>"
               "<LabelStyle><scale>0</scale></LabelStyle></Style>"
               '<Style id="sight"><LineStyle><color>9900ffff</color><width>1.5</width></LineStyle></Style>'
               '<Style id="path"><LineStyle><color>ffff7800</color><width>3</width></LineStyle></Style>')
    kml = ('<?xml version="1.0" encoding="UTF-8"?><kml xmlns="http://www.opengis.net/kml/2.2"><Document>'
           f"<name>{escape(os.path.basename(os.path.normpath(args.run)))} traffic lights</name>{styles}"
           "<Folder><name>Drive path</name><Placemark><name>Drive</name><styleUrl>#path</styleUrl>"
           f"<LineString><tessellate>1</tessellate><coordinates>{path}</coordinates></LineString></Placemark></Folder>"
           f"<Folder><name>Lights (estimated position)</name>{''.join(lights_kml)}</Folder>"
           f"<Folder><name>Car at closest view</name>{''.join(cars_kml)}</Folder>"
           f"<Folder><name>Sight lines</name><visibility>0</visibility>{''.join(lines_kml)}</Folder>"
           "</Document></kml>")
    out = os.path.join(args.run, "inventory", "lights_real.kml" if args.real_only else "lights.kml")
    with open(out, "w", encoding="utf-8") as f:
        f.write(kml)

    n_lab = sum(1 for L in inv["lights"] if str(L["light_no"]) in labels["lights"])
    covered = sum(1 for L in inv["lights"] if any(t in verdicts for t in L["track_ids"]))
    print(f"[kml] {len(inv['lights'])} lights ({n_lab} labeled) -> {out}")
    print(f"[group] lights with >=1 sampled frame box: {covered}/{len(inv['lights'])}")
    print(f"[group] frame verdicts on tracks NOT in the inventory (filtered out): {dict(orphan)}")
    print(f"[group] light-label vs frame-label conflicts: {len(conflicts)}")
    for no, real, fv in conflicts:
        print(f"    L{no}: light real={real}, frame boxes {fv}")


if __name__ == "__main__":
    main()
