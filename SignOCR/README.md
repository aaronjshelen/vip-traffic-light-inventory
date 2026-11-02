# SignOCR — Directional-Sign OCR Pipeline

> **Status (2026-10-25): PAUSED.** Secondary feature (traffic lights are the main pipeline).
> Detection → OCR → D-series classification → `report.html` works. Depth Anything 3
> localization is implemented but **not yet run/validated** — use `--no-depth` for now.

A **fully automatic, standalone** pipeline that reads a dashcam video and extracts
the text on **directional road signs** (highway guide, exit, destination, and
street-name signs). Completely separate from the `KASTEL/` traffic-light pipeline —
it shares only the `../Recordings/` clips.

```
dashcam .mp4
  → YOLO-World open-vocabulary sign-PANEL detection (US directional prompts)
  → ByteTrack (one persistent track_id per physical sign)
  → automatic best-crop selection (area × sharpness × detection confidence)
  → OCR (EasyOCR; PaddleOCR adapter also supported)
  → aggregate repeated observations → ONE record per sign
  → annotated_video.mp4 + signs.csv + signs.json + observations.csv + crops/ + summary.json
```

No manual frame selection, cropping, or transcription. One command.

## Models
- **Detector:** YOLO-World (`yolov8x-worldv2.pt`) via Ultralytics — open-vocabulary,
  prompted with US directional-sign phrases. No training required.
- **OCR:** EasyOCR (GPU, torch-based). PaddleOCR was preferred but hit a
  paddle/torch Windows DLL conflict on Python 3.13 (see `requirements.txt`).
- **Tracking:** ByteTrack (Ultralytics built-in), same approach as KASTEL.

## Setup (one time)
```powershell
cd SignOCR
python -m venv .venv --system-site-packages   # inherits global GPU torch + ultralytics
.venv/Scripts/python -m pip install easyocr
```

## Run
```powershell
cd SignOCR
.venv/Scripts/python run_sign_ocr.py `
  --video "../Recordings/clip_0718_0820.mp4" `
  --output outputs/demo
```

### Options
| Flag | Default | Meaning |
|------|---------|---------|
| `--video` | (required) | Input dashcam `.mp4` |
| `--output` | `outputs/run` | Output directory |
| `--mode` | `panel` | `panel` (detect+track sign panels), `scenetext` (OCR whole frames), or `auto` |
| `--max-frames` | `0` | Cap processed frames (0 = all) |
| `--stride` | `1` | Process every Nth frame |
| `--device` | auto | `0` = GPU, `cpu` |
| `--imgsz` | `1280` | Detection image size |
| `--conf` | `0.05` | Panel detection confidence (low = high recall; OCR confidence filters junk) |
| `--ocr` | `easy` | `easy` or `paddle` |
| `--prompts` | US directional set | Open-vocabulary detection prompts |
| `--no-video` | off | Skip annotated video |
| `--scenetext-width` / `--min-text-conf` / `--min-text-h` | 1280 / 0.4 / 14 | scenetext-mode knobs |

## Outputs
- `annotated_video.mp4` — sign boxes + track IDs + recognized text
- `signs.csv` / `signs.json` — one consolidated record per physical sign
- `observations.csv` — every raw OCR read (preserved)
- `crops/` — automatically extracted sign crops
- `summary.json` — processing statistics

## Known limitations
- Open-vocabulary detection fires on many non-sign green rectangles at low conf;
  OCR confidence is the practical filter for real text.
- Distant/motion-blurred signs OCR poorly — text only resolves as the car approaches.
  Best-crop selection mitigates this but cannot read what the camera can't resolve.
- `panel` mode tracks whole panels (good for later geographic localization — one
  point per sign); `scenetext` mode does not track.

## D-series classification, localization, HTML report (added 2026-10-25)
After tracking + OCR, each sign is post-processed (`sign_ocr/enrich.py`):
1. **MUTCD D-series description** (`sign_ocr/dseries.py`) — green-background ratio, number of
   white text rows, arrow blobs, distance numbers, street suffixes → likely `D1-1/2/3`
   (destination), `D2-1/2/3` (distance) or `D3-1` (street name). Non-green panels
   (regulatory/warning/business) are excluded from the inventory. Heuristic proof of concept,
   not a trained D-series detector.
2. **Localization** (`sign_ocr/localize.py`) — Depth Anything 3 (`DA3METRIC-LARGE`) metric depth on
   each sign's best frame → median depth in the box → camera-relative forward/lateral/height (m).
   Focal length estimated with `DA3-BASE` (override `--focal-px`, fallback `--hfov`).
   `--gps log.csv` (`t,lat,lon[,heading]`) converts positions to lat/lon once a GPS log exists.
   DA3 is loaded from `third_party/Depth-Anything-3/src` (NOT pip-installed: it pins numpy<2 and
   pulls xformers, which would clobber the cu128 torch). Only `addict` was added to `.venv`.
3. **`report.html`** — self-contained page: crop, OCR text, MUTCD code + description, location.

Re-generate the report for an existing run without re-detecting:
```powershell
.venv/Scripts/python make_report.py --output outputs/demo --video ../Recordings/clip_0718_0820.mp4
```
`--no-depth` skips localization.
