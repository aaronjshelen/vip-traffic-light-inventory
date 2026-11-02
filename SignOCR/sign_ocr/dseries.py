"""MUTCD D-series (guide sign) heuristics on top of the panel crop + OCR result.

Proof-of-concept: the detector is open-vocabulary YOLO-World, which has no notion
of MUTCD codes. Instead we DESCRIBE each panel from what we can measure:

  * background colour   -> MUTCD guide signs are white legend on GREEN
  * number of text rows -> D1-1 / D1-2 / D1-3 differ by 1 / 2 / 3 destination lines
  * arrow presence      -> white non-text blobs on the green field (heuristic)
  * row content         -> distance numbers => D2 (distance), street suffix => D3 (street name)

and map that description to the most likely D-series code. A trained D-series
classifier (future work) would replace `classify_dseries`, keeping the same fields.

MUTCD D-series reference (2009/2023 MUTCD, Ch. 2D):
  D1-1/-2/-3   Destination sign, 1/2/3 destinations (name + arrow per line)
  D2-1/-2/-3   Distance sign, 1/2/3 destinations (name + miles per line)
  D3-1         Street name sign
"""

from __future__ import annotations

import re
from typing import List

import cv2
import numpy as np

# Street-name suffixes (USPS abbreviations + spelled out) -> D3 street name sign
_STREET_SUFFIX = {
    "RD", "ROAD", "ST", "STREET", "DR", "DRIVE", "AVE", "AV", "AVENUE", "BLVD",
    "BOULEVARD", "LN", "LANE", "WAY", "PKWY", "PKY", "PARKWAY", "CT", "COURT", "PL",
    "PLACE", "CIR", "CIRCLE", "HWY", "HIGHWAY", "TRL", "TRAIL", "TER", "LOOP", "PIKE",
}
_CARDINAL = {"N", "S", "E", "W", "NORTH", "SOUTH", "EAST", "WEST"}
# a row ending in a mileage, e.g. "Tucson 12", "Benson 3 1/2", "Exit 5 MILES"
_DISTANCE_ROW = re.compile(r"[A-Za-z].*\b\d+(\s+\d/\d|\.\d+)?\s*(MI|MILE|MILES)?\s*$", re.I)
# OCR often reads arrows as a lone symbol
_ARROW_TOKEN = re.compile(r"^[<>^vV\-=+←-⇿➔-➿|/\\]{1,3}$")

FAMILY = {"D1": "Destination", "D2": "Distance", "D3": "Street name"}


def group_rows(lines) -> List[list]:
    """Group OCR boxes into text rows (top->bottom), each row sorted left->right."""
    rows: List[list] = []
    for ln in sorted(lines, key=lambda l: (l.box[1] + l.box[3]) / 2):
        y1, y2 = ln.box[1], ln.box[3]
        for row in rows:
            ry1 = min(l.box[1] for l in row)
            ry2 = max(l.box[3] for l in row)
            overlap = min(y2, ry2) - max(y1, ry1)
            if overlap > 0.5 * min(y2 - y1, ry2 - ry1):
                row.append(ln)
                break
        else:
            rows.append([ln])
    rows.sort(key=lambda r: min(l.box[1] for l in r))
    return [sorted(r, key=lambda l: l.box[0]) for r in rows]


def _colour_ratios(crop):
    h, w = crop.shape[:2]
    # ignore the outer 10% (crop padding / background around the panel)
    inner = crop[int(h * .1):max(int(h * .9), 1), int(w * .1):max(int(w * .9), 1)]
    hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (40, 60, 35), (95, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
    n = float(green.size) or 1.0
    return float((green > 0).sum()) / n, float((white > 0).sum()) / n


def _count_arrow_blobs(crop, lines) -> int:
    """White blobs on the green field that are NOT covered by OCR text boxes."""
    h, w = crop.shape[:2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    white = cv2.inRange(hsv, (0, 0, 160), (180, 70, 255))
    for ln in lines:  # mask out recognised text (slightly dilated)
        x1, y1, x2, y2 = (int(v) for v in ln.box)
        pad = max(2, int(0.15 * (y2 - y1)))
        white[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = 0
    n, _, stats, _ = cv2.connectedComponentsWithStats(white, connectivity=8)
    count = 0
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if area < 0.004 * h * w:                       # speckle
            continue
        if bw > 0.6 * w or bh > 0.8 * h:                # sign border / background
            continue
        if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:  # touches crop edge
            continue
        fill = area / float(bw * bh)
        if 0.2 <= fill <= 0.75 and 0.25 <= bw / float(bh) <= 4.0:  # arrow-ish
            count += 1
    return count


def classify_dseries(crop, lines, min_green: float = 0.25) -> dict:
    """Describe a sign panel and guess its MUTCD D-series code."""
    if crop is None or crop.size == 0:
        return dict(is_d_series=False, mutcd_code="", mutcd_family="", d_confidence="none",
                    description="empty crop", rows=[], n_rows=0, arrows=0,
                    green_ratio=0.0, white_ratio=0.0)

    green, white = _colour_ratios(crop)
    rows = group_rows(lines)
    text_rows, arrow_tokens = [], 0
    for r in rows:
        words = [l.text for l in r if not _ARROW_TOKEN.match(l.text)]
        arrow_tokens += sum(1 for l in r if _ARROW_TOKEN.match(l.text))
        if words:
            text_rows.append(" ".join(words))
    arrows = max(arrow_tokens, _count_arrow_blobs(crop, lines))
    n_rows = len(text_rows)

    desc = [f"{'green' if green >= min_green else 'non-green'} panel "
            f"({green:.0%} green, {white:.0%} white)",
            f"{n_rows} text row{'s' if n_rows != 1 else ''}"]
    if arrows:
        desc.append(f"{arrows} arrow/symbol blob{'s' if arrows != 1 else ''}")

    out = dict(rows=text_rows, n_rows=n_rows, arrows=arrows,
               green_ratio=round(green, 3), white_ratio=round(white, 3))

    if green < min_green or n_rows == 0:
        reason = "not a green guide panel" if green < min_green else "no legible text"
        return dict(out, is_d_series=False, mutcd_code="", mutcd_family="",
                    d_confidence="none", description="; ".join(desc) + f" -> {reason}")

    tokens = [t.strip(".,").upper() for row in text_rows for t in row.split()]
    n_dist = sum(1 for row in text_rows if _DISTANCE_ROW.search(row))
    k = min(n_rows, 3)
    if any(t in _STREET_SUFFIX for t in tokens) and n_rows <= 2:
        code, conf = "D3-1", "high" if any(t in _CARDINAL for t in tokens) or n_rows == 1 else "medium"
    elif n_dist and n_dist >= n_rows / 2:
        code, conf = f"D2-{k}", "medium"
    else:
        code, conf = f"D1-{k}", "medium" if arrows else "low"
    family = FAMILY[code[:2]]
    desc.append(f"-> likely {code} ({family})")
    return dict(out, is_d_series=True, mutcd_code=code, mutcd_family=family,
                d_confidence=conf, description="; ".join(desc))
