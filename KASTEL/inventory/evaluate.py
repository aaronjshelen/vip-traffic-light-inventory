"""Score the pipeline against labels.json exported from labeler.html.

Frame level (Frames tab)   detection precision / recall, per-frame state + indication accuracy
Light level (Lights tab)   light precision, indication accuracy: closest-frame vote vs all-frame
                           majority (does best-frame selection help?), state accuracy at the
                           closest frame, localization error in metres vs satellite-clicked truth

  .venv/Scripts/python -m inventory.evaluate --run outputs/gps_full --labels labels.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np

from inventory.build_inventory import parse_class
from inventory.gps import haversine_m


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a (0 labelled)"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--labels", required=True)
    args = p.parse_args()
    with open(args.labels, encoding="utf-8") as f:
        lab = json.load(f)
    with open(os.path.join(args.run, "inventory", "lights.json"), encoding="utf-8") as f:
        inv = json.load(f)

    # ---- frame level ----
    tp = fp = skip = missed = st_ok = st_n = ind_ok = ind_n = n_frames = 0
    for fr in lab.get("frames", {}).values():
        if not fr.get("done"):
            continue
        n_frames += 1
        missed += int(fr.get("missed") or 0)
        for b in fr.get("boxes", {}).values():
            if b.get("v") == "tp":
                tp += 1
                if b.get("st"):
                    st_n += 1
                    st_ok += b["st"] == "ok"
                if b.get("ind"):
                    ind_n += 1
                    ind_ok += b["ind"] == "ok"
            elif b.get("v") == "fp":
                fp += 1
            elif b.get("v") == "skip":
                skip += 1

    # ---- light level ----
    real = fake = 0
    best_ok = maj_ok = n_ind = state_ok = n_state = 0
    errs, confusion = [], Counter()
    for L in inv["lights"]:
        x = lab.get("lights", {}).get(str(L["light_no"]))
        if not x or x.get("real") is None:
            continue
        if x["real"] == "no":
            fake += 1
            continue
        real += 1
        if x.get("ind"):
            n_ind += 1
            best_ok += L["indication"] == x["ind"]
            maj_ok += L["indication_majority_all_frames"] == x["ind"]
            confusion[(x["ind"], L["indication"])] += 1
        if x.get("state"):
            pred_state = parse_class(L["best_class"])[1]   # same red_yellow -> red rule as the pipeline
            n_state += 1
            state_ok += pred_state == x["state"]
        if x.get("lat") and L.get("location"):
            errs.append(haversine_m(x["lat"], x["lon"], L["location"]["lat"], L["location"]["lon"]))

    lines = ["# Evaluation", "",
             f"Run: `{args.run}`  ·  labels: `{args.labels}`", "",
             "## Detection (sampled frames)",
             f"- Frames labelled: {n_frames}  ·  boxes too far to judge (excluded): {skip}",
             f"- Precision: {pct(tp, tp + fp)}",
             f"- Recall: {pct(tp, tp + missed)}",
             f"- State correct (per frame, true lights): {pct(st_ok, st_n)}",
             f"- Indication correct (per frame, true lights): {pct(ind_ok, ind_n)}", "",
             "## Physical lights",
             f"- Light precision (real / labelled): {pct(real, real + fake)}",
             f"- Indication, closest-frame vote: {pct(best_ok, n_ind)}",
             f"- Indication, all-frame majority (baseline): {pct(maj_ok, n_ind)}",
             f"- State at closest frame: {pct(state_ok, n_state)}", ""]
    if errs:
        e = np.array(errs)
        lines += ["## Localization (MoGe-2 + GPS vs satellite truth)",
                  f"- Lights: {len(e)}",
                  f"- Error median {np.median(e):.1f} m · mean {e.mean():.1f} m · max {e.max():.1f} m",
                  f"- Within 5 m: {pct(int((e <= 5).sum()), len(e))} · within 10 m: {pct(int((e <= 10).sum()), len(e))}", ""]
    if confusion:
        lines += ["## Indication confusion (true → predicted)"] + \
                 [f"- {t} → {p}: {n}" for (t, p), n in confusion.most_common()]
    text = "\n".join(lines)
    sys.stdout.reconfigure(encoding="utf-8")   # Windows console is cp1252
    print(text)
    with open(os.path.join(args.run, "inventory", "evaluation.md"), "w", encoding="utf-8") as f:
        f.write(text + "\n")


if __name__ == "__main__":
    main()
