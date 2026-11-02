"""Self-contained HTML report: one card per sign (crop, OCR text, MUTCD guess, location).

All images are embedded as base64 JPEGs so `report.html` is a single portable file.
"""

from __future__ import annotations

import base64
import html
import os

import cv2

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#5d6470;--line:#e2e5ea;
--sign:#006b3f;--accent:#0b63ce;--warn:#a15c00;--chip:#eef1f5}
@media (prefers-color-scheme:dark){:root{--bg:#111317;--card:#1a1d23;--ink:#e8eaee;
--mute:#9aa1ad;--line:#2b2f37;--accent:#6aa8ff;--warn:#f0b35a;--chip:#252a32}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px}
.sub{color:var(--mute);margin:0 0 20px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:22px}.stat span{color:var(--mute);font-size:13px}
.flow{display:flex;flex-wrap:wrap;gap:6px;align-items:center;font-size:13px;color:var(--mute)}
.flow span{background:var(--chip);border-radius:6px;padding:4px 8px;color:var(--ink)}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--card);
border-radius:0 8px 8px 0;color:var(--mute);font-size:14px;margin:14px 0}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:16px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;overflow:hidden;
display:flex;flex-direction:column}
.crop{background:#000;display:flex;justify-content:center;align-items:center;height:170px}
.crop img{max-width:100%;max-height:170px}
.body{padding:14px 16px;display:flex;flex-direction:column;gap:10px}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.code{font-weight:700;font-size:14px;background:var(--sign);color:#fff;border-radius:6px;padding:3px 9px}
.code.none{background:var(--chip);color:var(--mute)}
.conf{font-size:12px;color:var(--mute)}
.legend{background:var(--sign);color:#fff;border:2px solid #fff;outline:2px solid var(--sign);
border-radius:8px;padding:8px 12px;font-weight:600;font-size:17px;line-height:1.35}
.legend.plain{background:var(--chip);color:var(--ink);outline-color:var(--line);border-color:transparent}
.desc{color:var(--mute);font-size:13px}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0;font-size:13px}
dt{color:var(--mute)}dd{margin:0;font-variant-numeric:tabular-nums}
.loc{display:flex;gap:12px;align-items:center}
.loc svg{flex:none;background:var(--chip);border-radius:8px}
.thumbs{display:grid;grid-template-columns:1fr 1fr;gap:6px}
.thumbs img{width:100%;border-radius:6px;display:block}
.thumbs figcaption{font-size:11px;color:var(--mute)}
figure{margin:0}
details{margin-top:28px}summary{cursor:pointer;font-weight:600}
table{width:100%;border-collapse:collapse;font-size:13px;margin-top:10px}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
th{color:var(--mute);font-weight:600}
"""


def _img_uri(path_or_img, max_w=640, quality=82):
    img = cv2.imread(path_or_img) if isinstance(path_or_img, str) else path_or_img
    if img is None:
        return ""
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, int(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def _topdown_svg(loc, max_m=80.0, size=110):
    """Bird's-eye sketch: camera at bottom-centre, sign at (lateral, forward)."""
    if not loc:
        return ""
    s = size / max_m
    cx, cy = size / 2, size - 8
    x = cx + max(-max_m / 2, min(max_m / 2, loc["lateral_m"])) * s
    y = cy - min(max_m, max(0, loc["forward_m"])) * s
    rings = "".join(f'<circle cx="{cx}" cy="{cy}" r="{r * s}" fill="none" stroke="currentColor" '
                    f'stroke-opacity=".15"/>' for r in (20, 40, 60, 80))
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}" aria-label="top-down position">'
            f'{rings}<line x1="{cx}" y1="{cy}" x2="{x:.1f}" y2="{y:.1f}" stroke="currentColor" '
            f'stroke-opacity=".4" stroke-dasharray="3 3"/>'
            f'<polygon points="{cx - 5},{cy + 4} {cx + 5},{cy + 4} {cx},{cy - 6}" fill="#0b63ce"/>'
            f'<rect x="{x - 5:.1f}" y="{y - 4:.1f}" width="10" height="8" rx="1.5" fill="#006b3f" '
            f'stroke="#fff"/></svg>')


def _location_block(s):
    loc = s.get("location") or {}
    geo = s.get("geo")
    if not loc:
        return '<div class="desc">Location: not computed (run with depth localization)</div>'
    rows = [("Forward", f'{loc["forward_m"]:.1f} m'),
            ("Lateral", f'{loc["lateral_m"]:+.1f} m ({"right" if loc["lateral_m"] >= 0 else "left"})'),
            ("Height", f'{loc["height_m"]:+.1f} m vs camera'),
            ("Range", f'{loc["range_m"]:.1f} m @ {loc["bearing_deg"]:+.0f}&deg;'),
            ("Lat / Lon", f'{geo[0]:.6f}, {geo[1]:.6f}' if geo else
             '<span style="color:var(--warn)">pending GPS log</span>')]
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return f'<div class="loc">{_topdown_svg(loc)}<dl>{dl}</dl></div>'


def _card(s, out_dir):
    d_ok = s.get("is_d_series")
    code = html.escape(s.get("mutcd_code") or "not D-series")
    rows = s.get("rows") or [s.get("text", "")]
    legend = "<br>".join(html.escape(r) for r in rows if r)
    crop = _img_uri(os.path.join(out_dir, s["best_crop"])) if s.get("best_crop") else ""
    thumbs = ""
    if s.get("frame_preview") or s.get("depth_preview"):
        thumbs = '<div class="thumbs">' + "".join(
            f'<figure><img src="{_img_uri(os.path.join(out_dir, s[k]), 480)}" alt="">'
            f'<figcaption>{cap}</figcaption></figure>'
            for k, cap in (("frame_preview", f'frame {s.get("best_frame", "")}'),
                           ("depth_preview", "DA3 metric depth"))
            if s.get(k)) + "</div>"
    t0, t1 = s.get("first_time"), s.get("last_time")
    seen = (f'{s["n_observations"]} frames ({t0:.1f}&ndash;{t1:.1f} s)'
            if t0 is not None else f'{s["n_observations"]} frames')
    meta = (f'<dl><dt>Track</dt><dd>#{s.get("track_id", "")}'
            f'{" + " + ", ".join("#" + str(m) for m in s["merged_tracks"]) if s.get("merged_tracks") else ""}</dd>'
            f'<dt>Seen</dt><dd>{seen}</dd>'
            f'<dt>Detector</dt><dd>{html.escape(str(s.get("sign_type", "")))}</dd></dl>')
    return (f'<article class="card"><div class="crop"><img src="{crop}" alt="sign crop"></div>'
            f'<div class="body"><div class="top"><span class="code{"" if d_ok else " none"}">{code}</span>'
            f'<span class="conf">{html.escape(s.get("mutcd_family", ""))} &middot; '
            f'{s.get("d_confidence", "")} &middot; OCR {s.get("ocr_confidence", 0):.2f}</span></div>'
            f'<div class="legend{"" if d_ok else " plain"}">{legend}</div>'
            f'<div class="desc">{html.escape(s.get("description", ""))}</div>'
            f'{_location_block(s)}{meta}{thumbs}</div></article>')


def write_html(path: str, out_dir: str, meta: dict, signs: list, others: list):
    n_loc = sum(1 for s in signs if s.get("location"))
    stats = [(len(signs), "D-series signs"), (len(others), "other text panels (excluded)"),
             (n_loc, "signs localized"), (meta.get("frames_processed", "?"), "frames processed"),
             (meta.get("fps", "?"), "pipeline fps")]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)
    flow = " &rarr; ".join(f"<span>{x}</span>" for x in (
        "dashcam video", "YOLO-World panel detection", "ByteTrack", "best-crop selection",
        "EasyOCR", "D-series classification (colour + rows + arrows)",
        "Depth Anything 3 metric depth", "camera-relative position", "GPS (pending)"))
    cal = meta.get("focal_source", "")
    note = ("<b>Proof of concept.</b> MUTCD codes are inferred from a description of the panel "
            "(green background, number of white text rows, arrow blobs, distance numbers, "
            "street suffixes) &mdash; not from a detector trained on D-series signs. "
            "Positions are camera-relative (no GPS log yet); "
            f"camera focal length {meta.get('focal_px', 0):.0f} px ({html.escape(cal)}).")
    other_rows = "".join(
        f'<tr><td>#{s.get("track_id", "")}</td><td>{html.escape(s.get("text", ""))}</td>'
        f'<td>{s.get("ocr_confidence", 0):.2f}</td><td>{html.escape(s.get("description", ""))}</td></tr>'
        for s in others)
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Guide Sign Inventory</title><style>{_CSS}</style></head><body><main>
<h1>Guide sign inventory &mdash; MUTCD D-series</h1>
<p class="sub">{html.escape(os.path.basename(meta.get("video", "")))}</p>
<div class="flow">{flow}</div>
<div class="stats">{stat_html}</div>
<div class="note">{note}</div>
<h2>D-series guide signs</h2>
<div class="grid">{"".join(_card(s, out_dir) for s in signs) or "<p>None found.</p>"}</div>
<details><summary>Other text panels excluded from the inventory ({len(others)})</summary>
<table><tr><th>Track</th><th>OCR text</th><th>Conf</th><th>Why excluded</th></tr>{other_rows}</table>
</details></main></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
