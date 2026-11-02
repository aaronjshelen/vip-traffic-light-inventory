# Command Guide — Traffic-Light Detection Baseline

Quick reference for running `detector/run_baseline.py`. All commands are run from
**inside this `KASTEL/` folder**: `C:\lair\school\VIP - Traffic Light\KASTEL`.

The dashcam clips live one level up in `../Recordings/` (shared across models),
so the examples below use that relative path.

---

## TL;DR — recommended full run

Full video, high resolution (best on small/distant lights), no crop flood —
just a bounded, evenly-spread sample of crops in their own folder:

```powershell
python -m detector.run_baseline `
  --video "../Recordings/2024_05_18_02_09_20_847/123456/IMEINotAvailable/camera/2024_05_18_02_09_20_847_cam_2024_05_18_02_09_20_847.mp4" `
  --imgsz 1920 --conf 0.2 `
  --crop-every 25 --max-crops 300 --min-crop-conf 0.3 `
  --output-dir outputs/full_run
```

(~20–25 min on the GPU for the full 10.7-min drive. In PowerShell, the backtick `` ` `` continues a line — or
just put it all on one line.)

---

## Outputs

Everything lands under `--output-dir` (default `outputs/`):

```
<output-dir>/
    annotated_video.mp4   # ONE video, boxes + labels drawn
    detections.csv        # ONE row per detection (frame, time, bbox, conf, class)
    summary.json          # run stats
    crops/                # small .jpg per saved crop (only if crops enabled)
```

Re-running **overwrites** `annotated_video.mp4`, `detections.csv`, `summary.json`.
Use a different `--output-dir` to keep runs side by side. `crops/` files are named
by frame, so old ones persist unless overwritten — use a fresh folder per run.

---

## Common recipes

### Fast first pass (lower res, ~10 min)
```powershell
python -m detector.run_baseline --video "../Recordings/2024_05_18_02_09_20_847/123456/IMEINotAvailable/camera/2024_05_18_02_09_20_847_cam_2024_05_18_02_09_20_847.mp4" --imgsz 1280 --conf 0.25 --no-save-crops
```

### 30-second preview (quick sanity check)
```powershell
python -m detector.run_baseline --video "../Recordings/2024_05_18_02_09_20_847/123456/IMEINotAvailable/camera/2024_05_18_02_09_20_847_cam_2024_05_18_02_09_20_847.mp4" --end-sec 30
```

### A specific time window (e.g. 2:00–2:30 into the clip)
```powershell
python -m detector.run_baseline --video "../Recordings/2024_05_18_02_09_20_847/123456/IMEINotAvailable/camera/2024_05_18_02_09_20_847_cam_2024_05_18_02_09_20_847.mp4" --start-sec 120 --end-sec 150
```

### No crops at all (just video + CSV)
```powershell
python -m detector.run_baseline --video "../Recordings/..." --no-save-crops
```

### CSV + stats only, no annotated video (fastest)
```powershell
python -m detector.run_baseline --video "../Recordings/..." --no-video --no-save-crops
```

### Process every 5th frame (skim a long drive quickly)
```powershell
python -m detector.run_baseline --video "../Recordings/..." --stride 5
```
Output video FPS is scaled by the stride, so timing stays real.

---

## Controlling how many crops are saved

Crops are the only output that can balloon. Three knobs, combinable:

| Flag | Effect |
|------|--------|
| `--no-save-crops` | Save **zero** crops. |
| `--crop-every N` | Save only every **N-th** eligible detection — a small set spread **evenly** across the whole video. `1` = every detection. |
| `--max-crops N` | Hard cap on total crops (stops saving after N). |
| `--min-crop-conf X` | Only crop detections with confidence ≥ X (skip low-confidence junk). |

Example — about ~100 clean crops spread across the drive:
```powershell
... --crop-every 25 --max-crops 150 --min-crop-conf 0.3
```

> Crops are always reproducible later from `detections.csv` + the video, so
> saving few (or none) now loses nothing permanent.

---

## All options

| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | (required) | Input dashcam `.mp4` |
| `--weights` | `weights/traffic_lights_yolov8x.pt` | YOLO `.pt` weights |
| `--output-dir` | `outputs` | Where results go |
| `--conf` | `0.25` | Confidence threshold (lower = more detections + more false positives) |
| `--iou` | `0.7` | NMS IoU threshold |
| `--imgsz` | `1280` | Inference size; higher helps small/distant lights, slower (KASTEL used ~1920) |
| `--device` | auto | `0` = GPU, `cpu` = force CPU |
| `--stride` | `1` | Process every Nth frame |
| `--start-sec` / `--end-sec` | `0` / end | Process only a time segment (seconds) |
| `--max-frames` | `0` | Cap number of processed frames (0 = no cap) |
| `--save-crops` / `--no-save-crops` | on | Toggle crop saving |
| `--crop-every` | `1` | Save every Nth eligible crop |
| `--max-crops` | `0` | Cap total crops (0 = unlimited) |
| `--min-crop-conf` | `0.0` | Minimum confidence to save a crop |
| `--crop-padding` | `6` | Pixels of padding around each crop |
| `--no-video` | off | Skip writing the annotated video |
| `--track` | off | Enable tracking (assigns a persistent `track_id` per light; use `--stride 1`) |
| `--tracker` | `bytetrack.yaml` | Tracker config: `bytetrack.yaml` or `botsort.yaml` |

---

## Re-downloading / switching weights

```powershell
python download_weights.py                 # default yolov8x
python download_weights.py --model yolov8xl # larger variant
python download_weights.py --all
```
Then point a run at it with `--weights weights/<file>.pt`.
