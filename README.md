# Roadside Asset Inventory from Dashcam Video

Build a roadway-infrastructure inventory automatically from
forward-facing dashcam video plus phone GPS. For each asset the pipeline reports what it is, what it
shows, and where it is.

Two independent pipelines read the same recordings:

| Folder | Pipeline | Status |
|--------|----------|--------|
| [`KASTEL/`](KASTEL/) | **Traffic lights** (main): detect → track → classify → localize → inventory | Working; localization not yet validated |
| [`SignOCR/`](SignOCR/) | **Directional guide signs** (secondary): detect → track → OCR → MUTCD D-series → report | Paused; depth localization written but not run |

## Traffic-light pipeline (`KASTEL/`)

```
dashcam .mp4
  → KASTEL YOLOv8x (DTLD weights): detect + classify in one model (20 classes = shape × color)
  → ByteTrack: one track ID per signal head
  → junk-track filter (≥15 frames and max confidence ≥0.45)
  → indication (circle / arrow) from the 5 largest (closest) boxes; state (red/yellow/green) as a timeline
  → MoGe-2 monocular metric depth + field of view, combined with GPS pose → latitude / longitude
  → merge track fragments of the same head → lights.csv / .geojson / .kml + report.html
```

## Sign OCR pipeline (`SignOCR/`)

YOLO-World open-vocabulary panel detection → ByteTrack → best-crop selection (area × sharpness × confidence)
→ EasyOCR → heuristic MUTCD D-series description → self-contained `report.html`.
On the demo clip it found 4 street-name signs (D3-1). Text accuracy has not been formally measured yet.
See [`SignOCR/README.md`](SignOCR/README.md).

## Setup

Tested on Windows, Python 3.13, NVIDIA RTX 5070 Ti.

```powershell
# GPU torch first (RTX 50xx / Blackwell needs the cu128 build)
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision
pip install -r KASTEL/requirements.txt lap

# Traffic-light weights
cd KASTEL
python download_weights.py                      # -> weights/traffic_lights_yolov8x.pt

# MoGe-2 for localization (imported from source)
git clone https://github.com/microsoft/MoGe.git third_party/MoGe
python -m venv .venv --system-site-packages
.venv/Scripts/python -m pip install --no-deps utils3d-moge
```

For SignOCR setup (separate venv with EasyOCR; Depth Anything 3 cloned to
`SignOCR/third_party/Depth-Anything-3` from https://github.com/ByteDance-Seed/Depth-Anything-3),
see [`SignOCR/README.md`](SignOCR/README.md).

## Running (from `KASTEL/`)

```powershell
$V = "../Recordings/<drive>/.../camera/<video>.mp4"   # GPS *_loc.csv is found next to the video

# 1. Detect + track (tracking needs every frame: --stride 1)
python -m detector.run_baseline --video $V --track --imgsz 1920 --output-dir outputs/gps_full

# 2. Inventory: filter, classify, localize, report
.venv/Scripts/python -m inventory.build_inventory --run outputs/gps_full --video $V

# 3. Optional: labeling tool, evaluation, Google Earth export
.venv/Scripts/python -m inventory.make_labeler --run outputs/gps_full --video $V
.venv/Scripts/python -m inventory.evaluate     --run outputs/gps_full --labels outputs/gps_full/labeling/labels.json
.venv/Scripts/python -m inventory.export_kml   --run outputs/gps_full --video $V --real-only
```

More options: [`KASTEL/README.md`](KASTEL/README.md) and [`KASTEL/COMMANDS.md`](KASTEL/COMMANDS.md).

## Not included in this repository

- **Recordings** (video, GPS, IMU): too large, and lab data.
- **Model weights** and **third-party source** (MoGe, Depth Anything 3): download or clone as shown above.
- **Pipeline outputs** (`KASTEL/outputs/`, `SignOCR/outputs/`): regenerate with the commands above.

## Next steps

1. Measure localization error against light positions hand-placed on satellite imagery.
2. Label more drives with independent ground truth (not verify-the-model).
3. Run and validate depth-based localization for guide signs, and measure OCR character error rate.
