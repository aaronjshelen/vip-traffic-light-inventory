# Traffic-Light Detection Baseline

A minimal, swappable baseline for running a **pretrained traffic-light detector**
over dashcam `.mp4` footage and producing:

- an annotated output video (original dimensions, ~original FPS),
- a machine-readable `detections.csv`,
- per-detection crops (for later state / indication classification),
- console + `summary.json` statistics.

With `--track`, ByteTrack assigns a persistent `track_id` per signal head. Post-processing
(junk-track filtering, best-frame indication, state timelines, MoGe-2 + GPS localization,
inventory report) lives in `inventory/`; see the top-level README. The detector is
isolated behind `detector/detector.py` so it can be replaced later.

## Detector

Default weights are the **KASTEL MobilityLab** YOLOv8x model
(`traffic_lights_yolov8x.pt`), from
<https://github.com/KASTEL-MobilityLab/traffic-light-detection> (TLD-READY,
ITSC 2024), trained on the **DriveU Traffic Light Dataset (DTLD)**.

- Loads with **stock `ultralytics`** — no Docker, no modified fork required.
- **20 classes jointly encode state + indication**, e.g. `circle_red`,
  `circle_green`, `arrow_left_green`, `arrow_straight_red`, `off`.
- ⚠️ **Domain gap:** DTLD is German footage (mostly vertical, near-side signal
  heads). US / Georgia lights are typically horizontal and overhead, so expect
  reduced recall — a key thing to evaluate.

## Setup

```bash
# 1) (GPU only) install CUDA torch first — RTX 50xx/Blackwell needs cu128+
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision

# 2) install the rest
pip install -r requirements.txt

# 3) download pretrained weights (real KASTEL URLs)
python download_weights.py            # -> weights/traffic_lights_yolov8x.pt
```

## Run

```bash
python -m detector.run_baseline \
    --video "../Recordings/<your_clip>.mp4" \
    --conf 0.25 --imgsz 1280
```

Outputs land in `outputs/`:

```
outputs/
    annotated_video.mp4
    detections.csv          # frame, timestamp, bbox, confidence, class, crop_path
    summary.json
    crops/                  # f<frame>_<class>_<conf>.jpg
```

### Useful options

| Flag | Meaning |
|------|---------|
| `--conf` | Confidence threshold (default 0.25) |
| `--imgsz` | Inference size; higher helps small/distant lights (default 1280; KASTEL used ~1920) |
| `--stride N` | Process every Nth frame (output FPS scales so timing stays real) |
| `--start-sec` / `--end-sec` | Process only a time segment |
| `--max-frames N` | Cap number of processed frames (quick preview) |
| `--device 0` / `--device cpu` | Force GPU / CPU (auto by default) |
| `--no-video` | Skip writing the annotated video |
| `--min-crop-conf` / `--max-crops` | Control which / how many crops are saved |

### Quick preview (first 30 s)

```bash
python -m detector.run_baseline --video "../Recordings/<your_clip>.mp4" --end-sec 30
```

## Swapping the detector later

Anything exposing `.detect(frame_bgr) -> list[Detection]` and a `.names` dict
can drop in. `Detection` (see `detector/detector.py`) is the single hand-off
type that a future tracking / classification / localization stage will consume.
