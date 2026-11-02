"""report.html for the traffic-light inventory: satellite map + one card per physical light.

Images are embedded (base64) so the file is portable; the map uses Leaflet + Esri imagery
tiles (needs internet to show the satellite background).
"""

from __future__ import annotations

import base64
import html
import json
import os

import cv2

IND_COLOR = {"circle": "#2f7de1", "arrow_left": "#e0752d", "arrow_right": "#9b4fd1",
             "arrow_straight": "#1f9d72", "arrow_straight_left": "#c9a227", "unknown": "#777"}
STATE_COLOR = {"red": "#d93b3b", "yellow": "#e3b21c", "green": "#2ea85a", "off": "#666", "unknown": "#999"}

_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#5d6470;--line:#e2e5ea;--chip:#eef1f5;--warn:#a15c00}
@media (prefers-color-scheme:dark){:root{--bg:#111317;--card:#1a1d23;--ink:#e8eaee;--mute:#9aa1ad;
--line:#2b2f37;--chip:#252a32;--warn:#f0b35a}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1200px;margin:0 auto;padding:28px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:32px 0 12px}.sub{color:var(--mute);margin:0 0 18px}
.flow{display:flex;flex-wrap:wrap;gap:6px;font-size:13px;color:var(--mute)}
.flow span{background:var(--chip);border-radius:6px;padding:4px 8px;color:var(--ink)}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:10px;margin:18px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px}
.stat b{display:block;font-size:22px}.stat span{color:var(--mute);font-size:13px}
.note{border-left:3px solid var(--warn);padding:8px 12px;background:var(--card);border-radius:0 8px 8px 0;
color:var(--mute);font-size:14px;margin:14px 0}
#map{height:520px;border-radius:12px;border:1px solid var(--line)}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:14px}
.card{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:14px;display:flex;
flex-direction:column;gap:10px;scroll-margin-top:12px}
.card:target{outline:3px solid #2f7de1}
.top{display:flex;justify-content:space-between;align-items:center;gap:8px}
.no{font-weight:700;font-size:16px}.ind{color:#fff;border-radius:6px;padding:2px 8px;font-size:13px;font-weight:600}
.imgs{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;align-items:end}
.imgs img{width:100%;border-radius:6px;display:block;background:#000;image-rendering:auto}
.imgs figcaption{font-size:11px;color:var(--mute)}figure{margin:0}
.timeline{display:flex;height:12px;border-radius:6px;overflow:hidden;background:var(--chip)}
dl{display:grid;grid-template-columns:auto 1fr;gap:2px 12px;margin:0;font-size:13px}
dt{color:var(--mute)}dd{margin:0;font-variant-numeric:tabular-nums}
.legend{display:flex;flex-wrap:wrap;gap:10px;font-size:13px;color:var(--mute);margin:8px 0}
.legend i{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:4px}
"""


def _uri(path, max_w=360):
    img = cv2.imread(path) if path and os.path.isfile(path) else None
    if img is None:
        return ""
    if img.shape[1] > max_w:
        s = max_w / img.shape[1]
        img = cv2.resize(img, (max_w, max(1, int(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    elif img.shape[1] < 120:   # tiny crops: upscale for visibility
        s = 120 / img.shape[1]
        img = cv2.resize(img, (120, int(img.shape[0] * s)), interpolation=cv2.INTER_NEAREST)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode() if ok else ""


def _timeline(L):
    t0, t1 = L["first_t"], max(L["last_t"], L["first_t"] + 0.01)
    segs = "".join(
        f'<div title="{s["state"]} {s["t0"]:.1f}-{s["t1"]:.1f}s" style="background:{STATE_COLOR.get(s["state"], "#999")};'
        f'width:{max(1.5, 100 * (s["t1"] - s["t0"]) / (t1 - t0)):.1f}%"></div>' for s in L["states"])
    return f'<div class="timeline">{segs}</div>'


def _card(L, out_dir):
    loc = L.get("location") or {}
    ind = L["indication"]
    imgs = [(L.get("best_crop"), "closest crop")]
    if L.get("context_preview"):
        imgs += [(os.path.join(out_dir, L["context_preview"]), "context"),
                 (os.path.join(out_dir, L["depth_preview"]), "MoGe-2 depth")]
    figs = "".join(f'<figure><img src="{_uri(p)}" alt=""><figcaption>{c}</figcaption></figure>' for p, c in imgs)
    rows = [("Tracks", " + ".join(f"#{t}" for t in L["track_ids"])),
            ("Seen", f'{L["first_t"]:.1f}&ndash;{L["last_t"]:.1f} s, {L["n_frames"]} frames'),
            ("Indication", f'{ind} ({L["indication_share"]:.0%} of closest-frame vote); '
                           f'all-frame majority: {L["indication_majority_all_frames"]}'),
            ("States", " &rarr; ".join(s["state"] for s in L["states"])),
            ("Location", f'{loc["lat"]:.6f}, {loc["lon"]:.6f} (&plusmn;{loc["spread_m"]} m over {loc["n_obs"]} frames)'
             if loc else '<span style="color:var(--warn)">not localized</span>')]
    if loc:
        rows.append(("Geometry", f'closest {loc["closest_forward_m"]} m ahead, {loc["height_m"]:+.1f} m vs camera'))
    dl = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in rows)
    return (f'<article class="card" id="L{L["light_no"]}"><div class="top"><span class="no">Light {L["light_no"]}</span>'
            f'<span class="ind" style="background:{IND_COLOR.get(ind, "#777")}">{html.escape(ind)}</span></div>'
            f'<div class="imgs">{figs}</div>{_timeline(L)}<dl>{dl}</dl></article>')


def write_report(path, out_dir, meta, lights, gps_path):
    n_loc = sum(1 for L in lights if L.get("location"))
    stats = [(meta.get("n_tracks", "?"), "raw ByteTrack IDs"), (len(lights), "physical lights (after filter + merge)"),
             (n_loc, "lights localized"),
             (f'{meta["fov_x"]}&deg;' if meta.get("fov_x") else "&ndash;", "estimated camera HFOV")]
    stat_html = "".join(f'<div class="stat"><b>{v}</b><span>{k}</span></div>' for v, k in stats)
    flow = " &rarr; ".join(f"<span>{s}</span>" for s in (
        "dashcam video", "KASTEL YOLOv8x (DTLD)", "ByteTrack", "junk-track filter",
        "indication from closest frames", "state timeline", "MoGe-2 metric depth", "GPS pose", "lat/lon"))
    legend = "".join(f'<span><i style="background:{c}"></i>{k}</span>' for k, c in IND_COLOR.items())
    pts = [dict(no=L["light_no"], lat=L["location"]["lat"], lon=L["location"]["lon"], ind=L["indication"],
                color=IND_COLOR.get(L["indication"], "#777")) for L in lights if L.get("location")]
    note = ("Detector = KASTEL YOLOv8x trained on German DTLD lights (no US fine-tuning). Positions = MoGe-2 "
            f"monocular metric depth with an estimated (uncalibrated) camera FOV ({html.escape(meta.get('fov_source', ''))}) "
            "+ 1 Hz phone GPS; camera assumed aligned with the vehicle heading. Not yet evaluated against ground truth.")
    doc = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic Light Inventory</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>{_CSS}</style></head><body><main>
<h1>Traffic-light inventory</h1><p class="sub">{html.escape(os.path.basename(meta.get("video", "")))}</p>
<div class="flow">{flow}</div><div class="stats">{stat_html}</div><div class="note">{note}</div>
<h2>Map</h2><div class="legend">{legend}<span>&mdash; vehicle GPS path</span></div><div id="map"></div>
<h2>Lights</h2><div class="grid">{"".join(_card(L, out_dir) for L in lights)}</div>
</main><script>
const pts={json.dumps(pts)}, path={json.dumps(gps_path)};
if (window.L) {{
  const map=L.map('map');
  L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}',
    {{maxZoom:21,maxNativeZoom:19,attribution:'Imagery &copy; Esri'}}).addTo(map);
  const line=L.polyline(path,{{color:'#fff',weight:3,opacity:.8}}).addTo(map);
  pts.forEach(p=>L.circleMarker([p.lat,p.lon],{{radius:7,color:'#fff',weight:2,fillColor:p.color,fillOpacity:1}})
    .bindTooltip('Light '+p.no+' &middot; '+p.ind).on('click',()=>location.hash='L'+p.no).addTo(map));
  map.fitBounds(line.getBounds(),{{padding:[20,20]}});
}} else {{ document.getElementById('map').textContent='Map needs an internet connection.'; }}
</script></body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
