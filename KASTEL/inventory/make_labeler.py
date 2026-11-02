"""Build labeler.html: verify-the-model labeling for detection, classification and localization.

Instead of drawing boxes from scratch, the user CHECKS the pipeline's output:
  Frames tab  - one frame every --every-sec seconds with predicted boxes; mark each box
                correct-light / not-a-light, whether its state and indication are right,
                and how many visible lights were MISSED  -> precision, recall, per-frame accuracy
  Lights tab  - per physical light: crops far -> near; pick the true indication and the true
                state at the closest crop; click its true position on the satellite map
                                                     -> light-level accuracy, localization error
Labels autosave in the browser (localStorage) and export to labels.json for evaluate.py.

  .venv/Scripts/python -m inventory.make_labeler --run outputs/gps_full --video <...mp4>
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import cv2

from inventory.build_inventory import load_tracks, parse_class
from inventory.gps import GpsTrack, find_loc_csv, recording_start_ms

INDICATIONS = ["circle", "arrow_left", "arrow_right", "arrow_straight", "arrow_straight_left", "other"]
STATES = ["red", "yellow", "green", "off"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--video", required=True)
    p.add_argument("--every-sec", type=float, default=5.0, help="frame sampling interval for the Frames tab")
    p.add_argument("--start-sec", type=float, default=0.0)
    p.add_argument("--end-sec", type=float, default=0.0)
    p.add_argument("--crops-per-light", type=int, default=6)
    p.add_argument("--utc-offset", type=float, default=-7.0)
    args = p.parse_args()

    out = os.path.join(args.run, "labeling")
    os.makedirs(os.path.join(out, "frames"), exist_ok=True)
    with open(os.path.join(args.run, "inventory", "lights.json"), encoding="utf-8") as f:
        inv = json.load(f)
    tracks = load_tracks(os.path.join(args.run, "detections.csv"))

    # ---- lights: crops spread from far (first) to near (closest) ----
    lights = []
    for L in inv["lights"]:
        rows = sorted((d for t in L["track_ids"] for d in tracks[t]), key=lambda d: d["frame"])
        n = args.crops_per_light
        pick = [rows[int(i * (len(rows) - 1) / max(1, n - 1))] for i in range(n)] if len(rows) >= n else rows
        # right-most crop = the model's closest frame (what the State label is compared against)
        bestrow = next((d for d in rows if d["frame"] == L["best_frame"]), None)
        pick = [d for d in pick if d is not bestrow][: n - 1] + ([bestrow] if bestrow else [])
        crops = [dict(src=os.path.relpath(d["crop"], out).replace("\\", "/"), t=round(d["t"], 1), cls=d["cls"])
                 for d in sorted(pick[:-1], key=lambda d: d["area"]) + pick[-1:] if d["crop"]]
        best = os.path.relpath(L["best_crop"], out).replace("\\", "/") if L.get("best_crop") else ""
        loc = L.get("location") or {}
        lights.append(dict(no=L["light_no"], tracks=L["track_ids"], pred_ind=L["indication"],
                           pred_state=parse_class(L["best_class"])[1],
                           pred_lat=loc.get("lat"), pred_lon=loc.get("lon"), crops=crops, best=best,
                           t0=L["first_t"], t1=L["last_t"]))

    # ---- frames: uniform time sampling (includes frames with NO detections, for recall) ----
    by_frame = defaultdict(list)
    for tid, rows in tracks.items():
        for d in rows:
            by_frame[d["frame"]].append((tid, d))
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    end = min(total, int(args.end_sec * fps)) if args.end_sec else total
    frames = []
    fi = int(args.start_sec * fps)
    while fi < end:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, img = cap.read()
        if not ok:
            break
        boxes = []
        for k, (tid, d) in enumerate(sorted(by_frame.get(fi, []), key=lambda x: x[1]["box"][0]), 1):
            x1, y1, x2, y2 = (int(v) for v in d["box"])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(img, str(k), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
            cv2.putText(img, str(k), (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            boxes.append(dict(k=k, track=tid, cls=d["cls"], conf=round(d["conf"], 2), h=y2 - y1))
        name = f"frames/f{fi:06d}.jpg"
        cv2.imwrite(os.path.join(out, name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        frames.append(dict(frame=fi, t=round(fi / fps, 1), src=name, boxes=boxes))
        fi += int(round(args.every_sec * fps))
    cap.release()

    gps = GpsTrack(find_loc_csv(args.video), recording_start_ms(args.video, args.utc_offset))
    data = dict(run=os.path.basename(os.path.normpath(args.run)), lights=lights, frames=frames,
                path=gps.path(), indications=INDICATIONS, states=STATES)
    with open(os.path.join(out, "labeler.html"), "w", encoding="utf-8") as f:
        f.write(_HTML.replace("/*DATA*/", json.dumps(data)))
    print(f"[labeler] {len(lights)} lights, {len(frames)} frames -> {os.path.join(out, 'labeler.html')}")


_HTML = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Traffic Light Labeler</title>
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css">
<script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
<style>
:root{--bg:#f6f7f9;--card:#fff;--ink:#16181d;--mute:#5d6470;--line:#e2e5ea;--chip:#eef1f5;--on:#2f7de1;--ok:#2ea85a;--bad:#d93b3b}
@media (prefers-color-scheme:dark){:root{--bg:#111317;--card:#1a1d23;--ink:#e8eaee;--mute:#9aa1ad;--line:#2b2f37;--chip:#252a32}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,Segoe UI,Roboto,sans-serif}
header{position:sticky;top:0;z-index:1000;background:var(--card);border-bottom:1px solid var(--line);padding:10px 16px;
display:flex;flex-wrap:wrap;gap:10px;align-items:center}
header h1{font-size:17px;margin:0 12px 0 0}.tab{border:1px solid var(--line);background:var(--chip);color:var(--ink);
border-radius:8px;padding:6px 12px;cursor:pointer;font:inherit}.tab.on{background:var(--on);color:#fff;border-color:var(--on)}
.spacer{flex:1}.prog{color:var(--mute)}button.act{font:inherit;border-radius:8px;border:1px solid var(--line);
background:var(--card);color:var(--ink);padding:6px 12px;cursor:pointer}
main{max-width:1280px;margin:0 auto;padding:16px}.help{color:var(--mute);margin:0 0 14px;max-width:900px}
.frame{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:16px}
.frame.done{border-color:var(--ok)}.frame img{width:100%;border-radius:8px;display:block}
.row{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin-top:8px}
.box{display:flex;flex-wrap:wrap;gap:6px;align-items:center;background:var(--chip);border-radius:8px;padding:6px 8px}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:7px;overflow:hidden}
.seg button{font:inherit;font-size:13px;border:0;background:var(--card);color:var(--ink);padding:4px 8px;cursor:pointer}
.seg button.on{background:var(--on);color:#fff}.seg button.on.ok{background:var(--ok)}.seg button.on.bad{background:var(--bad)}
input[type=number]{width:60px;font:inherit;padding:3px 6px;border-radius:6px;border:1px solid var(--line);background:var(--card);color:var(--ink)}
.layout{display:grid;grid-template-columns:minmax(0,1fr) 440px;gap:16px}@media(max-width:980px){.layout{grid-template-columns:1fr}}
#map{height:calc(100vh - 110px);position:sticky;top:70px;border-radius:12px;border:1px solid var(--line)}
@media(max-width:980px){#map{height:360px;position:static}}
.light{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:12px;margin-bottom:14px}
.light.done{border-color:var(--ok)}.light.picking{outline:3px solid var(--on)}
.crops{display:flex;gap:6px;align-items:flex-end;overflow-x:auto;padding-bottom:4px}
.crops figure{margin:0;text-align:center;flex:none}.crops img{height:96px;border-radius:6px;image-rendering:pixelated;background:#000}
.crops figcaption{font-size:11px;color:var(--mute)}.pred{color:var(--mute);font-size:13px}
</style></head><body>
<header><h1>Traffic-light labeler</h1>
<button class="tab on" data-tab="frames">1. Frames (detector)</button>
<button class="tab" data-tab="lights">2. Lights (type + location)</button>
<span class="spacer"></span><span class="prog" id="prog"></span>
<button class="act" id="exp">Export labels.json</button>
<label class="act" style="cursor:pointer">Import<input type="file" id="imp" accept=".json" hidden></label></header>
<main><section id="frames"></section><section id="lights" hidden></section></main>
<script>
const D=/*DATA*/;
const KEY='tl-labels-'+D.run;
let S; try{S=JSON.parse(localStorage.getItem(KEY))||{}}catch(e){S={}}
S.frames=S.frames||{}; S.lights=S.lights||{};
const save=()=>{try{localStorage.setItem(KEY,JSON.stringify(S))}catch(e){} prog();};
const el=(t,a={},...k)=>{const e=document.createElement(t);for(const[x,y]of Object.entries(a)){if(x==='on')for(const[ev,f]of Object.entries(y))e.addEventListener(ev,f);else if(x==='class')e.className=y;else e.setAttribute(x,y)}k.flat().forEach(c=>e.append(c));return e};
function seg(opts,get,set,cls=''){const w=el('span',{class:'seg'});const draw=()=>{w.replaceChildren(...opts.map(([v,l,c])=>el('button',{class:(get()===v?'on ':'')+(c||''),on:{click:()=>{set(get()===v?undefined:v);draw()}}},l)))};draw();return w}
function prog(){const fd=D.frames.filter(f=>S.frames[f.frame]?.done).length;
 const ld=D.lights.filter(l=>{const x=S.lights[l.no];return x&&x.real!==undefined&&(x.real==='no'||(x.ind&&x.state))}).length;
 const lp=D.lights.filter(l=>S.lights[l.no]?.lat).length;
 document.getElementById('prog').textContent=`frames ${fd}/${D.frames.length} · lights ${ld}/${D.lights.length} · located ${lp}`}

// ---------- Frames tab ----------
const FH=el('p',{class:'help'},'For each sampled frame: for every numbered box choose ',el('b',{},'Light'),' or ',el('b',{},'Not a light'),
 ' (or ',el('b',{},"Can't tell"),' if it is too small/far to judge — those are left out of the score); for real lights mark whether the predicted ',el('b',{},'state'),' (colour) and ',el('b',{},'indication'),' (circle/arrow) are right. Then enter how many traffic-light heads you can ',el('b',{},'clearly make out'),' have ',el('b',{},'no box'),' (missed — ignore ones too far to judge), and tick Done. Frames with no boxes still count: enter missed lights (often 0) and tick Done.');
document.getElementById('frames').append(FH,...D.frames.map(f=>{
 const L=S.frames[f.frame]||(S.frames[f.frame]={boxes:{}});
 const card=el('div',{class:'frame'+(L.done?' done':'')});
 const boxes=f.boxes.map(b=>{const B=L.boxes[b.k]||(L.boxes[b.k]={});
  return el('div',{class:'box'},el('b',{},'#'+b.k),el('span',{class:'pred'},b.cls+' '+b.conf+' · '+b.h+'px'),
   seg([['tp','Light','ok'],['fp','Not a light','bad'],['skip',"Can't tell"]],()=>B.v,v=>{B.v=v;save()}),
   el('span',{class:'pred'},'state'),seg([['ok','✓','ok'],['bad','✗','bad']],()=>B.st,v=>{B.st=v;save()}),
   el('span',{class:'pred'},'indication'),seg([['ok','✓','ok'],['bad','✗','bad']],()=>B.ind,v=>{B.ind=v;save()}))});
 const missed=el('input',{type:'number',min:0,value:L.missed??''});missed.addEventListener('input',()=>{L.missed=missed.value===''?undefined:+missed.value;save()});
 const done=el('input',{type:'checkbox'});done.checked=!!L.done;done.addEventListener('change',()=>{L.done=done.checked;card.classList.toggle('done',L.done);save()});
 card.append(el('div',{class:'row'},el('b',{},`t = ${f.t}s`),el('span',{class:'pred'},`frame ${f.frame} · ${f.boxes.length} box(es)`)),
  el('img',{src:f.src,loading:'lazy',alt:''}),el('div',{class:'row'},...boxes),
  el('div',{class:'row'},'Missed lights:',missed,el('label',{},done,' Done')));return card}));

// ---------- Lights tab ----------
let picking=null, map=null, marks={};
function initMap(){ if(map||!window.L)return;
 map=L.map('map');L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',{maxZoom:22,maxNativeZoom:19,attribution:'Imagery © Esri'}).addTo(map);
 const line=L.polyline(D.path,{color:'#fff',weight:3,opacity:.7}).addTo(map);map.fitBounds(line.getBounds());
 D.lights.forEach(l=>{if(l.pred_lat)L.circleMarker([l.pred_lat,l.pred_lon],{radius:5,color:'#fff',weight:1,fillColor:'#2f7de1',fillOpacity:.9}).bindTooltip('pred '+l.no).addTo(map);drawTrue(l.no)});
 map.on('click',e=>{if(picking==null)return;const x=S.lights[picking]||(S.lights[picking]={});x.lat=+e.latlng.lat.toFixed(7);x.lon=+e.latlng.lng.toFixed(7);
  drawTrue(picking);document.querySelector(`#light${picking} .loc`).textContent=`true: ${x.lat}, ${x.lon}`;
  document.getElementById('light'+picking).classList.remove('picking');picking=null;save()});}
function drawTrue(no){const x=S.lights[no];if(marks[no]){map.removeLayer(marks[no]);delete marks[no]}
 if(x?.lat)marks[no]=L.marker([x.lat,x.lon]).bindTooltip('true '+no,{permanent:true,direction:'right'}).addTo(map)}
const LH=el('p',{class:'help'},'For each physical light (crops go far → near): is it a real traffic light? Pick its ',el('b',{},'true indication'),' and the ',el('b',{},'true state in the closest (right-most) crop'),'. Then press ',el('b',{},'Pick location'),' and click the light on the satellite map (blue dots = model predictions; zoom in to the signal mast / pole).');
const list=el('div',{});
document.getElementById('lights').append(el('div',{class:'layout'},el('div',{},LH,list),el('div',{id:'map'})));
D.lights.forEach(l=>{const x=S.lights[l.no]||(S.lights[l.no]={});
 const card=el('div',{class:'light',id:'light'+l.no});
 const upd=()=>card.classList.toggle('done',x.real==='no'||(x.real==='yes'&&!!x.ind&&!!x.state));upd();
 const crops=el('div',{class:'crops'},...l.crops.map((c,i)=>el('figure',{},el('img',{src:c.src,alt:''}),el('figcaption',{},c.t+'s'+(i===l.crops.length-1?' · closest':'')))));
 card.append(el('div',{class:'row'},el('b',{},'Light '+l.no),el('span',{class:'pred'},`tracks ${l.tracks.join(', ')} · ${l.t0}–${l.t1}s · model: ${l.pred_ind}, ${l.pred_state} (closest)`)),crops,
  el('div',{class:'row'},'Real light?',seg([['yes','Yes','ok'],['no','No','bad']],()=>x.real,v=>{x.real=v;upd();save()})),
  el('div',{class:'row'},'Indication',seg(D.indications.map(i=>[i,i.replace('arrow_','→')]),()=>x.ind,v=>{x.ind=v;upd();save()})),
  el('div',{class:'row'},'State (closest crop)',seg(D.states.map(s=>[s,s]),()=>x.state,v=>{x.state=v;upd();save()})),
  el('div',{class:'row'},el('button',{class:'act',on:{click:()=>{document.querySelectorAll('.light.picking').forEach(c=>c.classList.remove('picking'));picking=l.no;card.classList.add('picking')}}},'Pick location'),
   el('span',{class:'pred loc'},x.lat?`true: ${x.lat}, ${x.lon}`:'not set')));
 list.append(card)});

document.querySelectorAll('.tab').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('on',t===b));
 document.getElementById('frames').hidden=b.dataset.tab!=='frames';document.getElementById('lights').hidden=b.dataset.tab!=='lights';
 if(b.dataset.tab==='lights'){initMap();setTimeout(()=>map&&map.invalidateSize(),50)}}));
document.getElementById('exp').addEventListener('click',()=>{const a=el('a',{href:URL.createObjectURL(new Blob([JSON.stringify(S,null,1)],{type:'application/json'})),download:'labels.json'});a.click()});
document.getElementById('imp').addEventListener('change',async e=>{const f=e.target.files[0];if(!f)return;S=JSON.parse(await f.text());save();location.reload()});
prog();
</script></body></html>"""


if __name__ == "__main__":
    main()
