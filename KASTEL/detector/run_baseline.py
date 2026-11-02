"""Traffic-light detection BASELINE runner.

    dashcam .mp4
        -> pretrained traffic-light detector (KASTEL / DTLD YOLOv8 by default)
        -> bounding boxes + confidence drawn on each frame
        -> annotated .mp4  (original dimensions, ~original FPS)
        -> detections.csv  (frame, timestamp, bbox, confidence, class)
        -> crops/          (one image per detection, for later state/indication work)
        -> summary.json + console statistics

This is a research baseline, intentionally small. It does NOT do tracking,
best-frame selection, classification, or localization. The detector is isolated
behind `detector.TrafficLightDetector` so it can be swapped later.

Example:
    python -m detector.run_baseline \
        --video "../Recordings/2024_05_18_02_09_20_847/123456/IMEINotAvailable/camera/2024_05_18_02_09_20_847_cam_2024_05_18_02_09_20_847.mp4" \
        --conf 0.25 --imgsz 1280
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from collections import Counter

import cv2
from tqdm import tqdm

from .detector import TrafficLightDetector


# ---- drawing -------------------------------------------------------------

def _color_for_class(name: str):
    """BGR color keyed off the state encoded in the class name."""
    n = name.lower()
    if "red_yellow" in n:
        return (0, 165, 255)   # orange
    if "yellow" in n:
        return (0, 215, 255)   # amber
    if "red" in n:
        return (0, 0, 255)     # red
    if "green" in n:
        return (0, 200, 0)     # green
    return (160, 160, 160)     # off / unknown -> gray


def draw_detections(frame, detections, line_thickness: int):
    for det in detections:
        color = _color_for_class(det.class_name)
        p1 = (int(det.x1), int(det.y1))
        p2 = (int(det.x2), int(det.y2))
        cv2.rectangle(frame, p1, p2, color, line_thickness)
        id_prefix = f"#{det.track_id} " if det.track_id is not None else ""
        label = f"{id_prefix}{det.class_name} {det.confidence:.2f}"
        font_scale = max(0.4, line_thickness * 0.25)
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        ty = max(p1[1], th + 4)
        cv2.rectangle(frame, (p1[0], ty - th - baseline),
                      (p1[0] + tw, ty + baseline), color, -1)
        cv2.putText(frame, label, (p1[0], ty),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1,
                    cv2.LINE_AA)
    return frame


# ---- crops ---------------------------------------------------------------

def save_crop(frame, det, frame_index, crops_dir, padding: int) -> str:
    h, w = frame.shape[:2]
    x1 = max(0, int(det.x1) - padding)
    y1 = max(0, int(det.y1) - padding)
    x2 = min(w, int(det.x2) + padding)
    y2 = min(h, int(det.y2) + padding)
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = frame[y1:y2, x1:x2]
    # When tracking, lead with the track id so a light's crops group together.
    tid = f"track{det.track_id:04d}_" if det.track_id is not None else ""
    fname = f"{tid}f{frame_index:06d}_{det.class_name}_{det.confidence:.2f}.jpg"
    path = os.path.join(crops_dir, fname)
    cv2.imwrite(path, crop)
    return path


# ---- main ----------------------------------------------------------------

def run(args) -> int:
    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}")
        return 1
    if not os.path.isfile(args.weights):
        print(f"ERROR: weights not found: {args.weights}\n"
              f"       Run: python download_weights.py")
        return 1

    out_dir = args.output_dir
    crops_dir = os.path.join(out_dir, "crops")
    os.makedirs(out_dir, exist_ok=True)
    if args.save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: could not open video: {args.video}")
        return 1

    in_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(args.start_sec * in_fps) if args.start_sec else 0
    if args.end_sec:
        end_frame = min(total_frames, int(args.end_sec * in_fps))
    else:
        end_frame = total_frames
    if args.max_frames:
        end_frame = min(end_frame, start_frame + args.max_frames * args.stride)

    # Output FPS is scaled by stride so the annotated video keeps real-world timing.
    out_fps = max(1.0, in_fps / args.stride)

    print("=" * 66)
    print("TRAFFIC-LIGHT DETECTION BASELINE")
    print("=" * 66)
    print(f"  video        : {args.video}")
    print(f"  resolution   : {width}x{height}  @ {in_fps:.2f} fps  ({total_frames} frames)")
    print(f"  weights      : {args.weights}")
    print(f"  segment      : frames [{start_frame}, {end_frame})  stride={args.stride}")
    print(f"  conf / iou   : {args.conf} / {args.iou}   imgsz={args.imgsz}")
    print(f"  tracking     : {args.tracker if args.track else 'off (detect only)'}")
    if args.track and args.stride != 1:
        print("  WARNING: trackers assume consecutive frames; --stride > 1 will "
              "degrade tracking. Use --stride 1 for real tracking.")

    detector = TrafficLightDetector(
        weights_path=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
    )
    print(f"  device       : {detector.device}")
    print(f"  classes ({len(detector.names)}) : {list(detector.names.values())}")
    print("-" * 66)

    writer = None
    if not args.no_video:
        annotated_path = os.path.join(out_dir, "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(annotated_path, fourcc, out_fps, (width, height))

    csv_path = os.path.join(out_dir, "detections.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "frame_index", "timestamp_sec", "track_id",   # track_id reserved for later
        "x1", "y1", "x2", "y2", "confidence",
        "class_id", "class_name", "crop_path",
    ])

    line_thickness = max(2, int(round((width + height) / 1200)))

    # stats
    processed = 0
    frames_with_dets = 0
    total_dets = 0
    conf_sum = 0.0
    class_counts: Counter = Counter()
    crops_saved = 0
    crop_candidates = 0   # detections eligible to be cropped (for --crop-every sampling)
    track_ids = set()     # distinct track ids seen (only when --track)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_index = start_frame
    t0 = time.time()

    pbar = tqdm(total=(end_frame - start_frame) // args.stride, unit="frame")
    while frame_index < end_frame:
        ok, frame = cap.read()
        if not ok:
            break

        # honor stride: only run inference on every Nth frame
        if (frame_index - start_frame) % args.stride != 0:
            frame_index += 1
            continue

        detections = detector.track(frame, tracker=args.tracker) if args.track \
            else detector.detect(frame)
        timestamp = frame_index / in_fps if in_fps else ""

        if detections:
            frames_with_dets += 1
        for det in detections:
            total_dets += 1
            conf_sum += det.confidence
            class_counts[det.class_name] += 1
            if det.track_id is not None:
                track_ids.add(det.track_id)

            crop_path = ""
            if args.save_crops and det.confidence >= args.min_crop_conf:
                crop_candidates += 1
                # --crop-every N: keep only every Nth eligible detection, so a
                # capped set is spread evenly across the whole video.
                take = (crop_candidates % args.crop_every == 0)
                under_cap = (args.max_crops == 0 or crops_saved < args.max_crops)
                if take and under_cap:
                    crop_path = save_crop(frame, det, frame_index, crops_dir,
                                          args.crop_padding)
                    if crop_path:
                        crops_saved += 1
                        crop_path = os.path.relpath(crop_path, out_dir)

            csv_writer.writerow([
                frame_index,
                f"{timestamp:.3f}" if timestamp != "" else "",
                det.track_id if det.track_id is not None else "",
                f"{det.x1:.1f}", f"{det.y1:.1f}", f"{det.x2:.1f}", f"{det.y2:.1f}",
                f"{det.confidence:.4f}",
                det.class_id, det.class_name, crop_path,
            ])

        if writer is not None:
            draw_detections(frame, detections, line_thickness)
            writer.write(frame)

        processed += 1
        frame_index += 1
        pbar.update(1)
        pbar.set_postfix(dets=total_dets)

    pbar.close()
    elapsed = time.time() - t0

    cap.release()
    if writer is not None:
        writer.release()
    csv_file.close()

    avg_conf = (conf_sum / total_dets) if total_dets else 0.0
    proc_fps = (processed / elapsed) if elapsed else 0.0

    summary = {
        "video": args.video,
        "resolution": [width, height],
        "input_fps": round(in_fps, 3),
        "weights": args.weights,
        "device": str(detector.device),
        "imgsz": args.imgsz,
        "conf_threshold": args.conf,
        "iou_threshold": args.iou,
        "stride": args.stride,
        "frames_processed": processed,
        "frames_with_detections": frames_with_dets,
        "total_detections": total_dets,
        "tracking": args.tracker if args.track else None,
        "unique_tracks": len(track_ids) if args.track else None,
        "average_confidence": round(avg_conf, 4),
        "crops_saved": crops_saved,
        "elapsed_sec": round(elapsed, 2),
        "processing_fps": round(proc_fps, 2),
        "class_counts": dict(class_counts.most_common()),
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("-" * 66)
    print("DONE")
    print(f"  frames processed        : {processed}")
    print(f"  frames with detections  : {frames_with_dets} "
          f"({(100*frames_with_dets/processed) if processed else 0:.1f}%)")
    print(f"  total detections        : {total_dets}")
    if args.track:
        print(f"  unique tracks (lights)  : {len(track_ids)}")
    print(f"  average confidence      : {avg_conf:.3f}")
    print(f"  crops saved             : {crops_saved}")
    print(f"  elapsed                 : {elapsed:.1f}s  ->  {proc_fps:.1f} fps")
    if class_counts:
        print("  detections per class    :")
        for name, c in class_counts.most_common():
            print(f"      {name:<28} {c}")
    print("-" * 66)
    print(f"  annotated video : {os.path.join(out_dir, 'annotated_video.mp4') if not args.no_video else '(skipped)'}")
    print(f"  detections csv  : {csv_path}")
    print(f"  crops dir       : {crops_dir if args.save_crops else '(skipped)'}")
    print(f"  summary json    : {os.path.join(out_dir, 'summary.json')}")
    print("=" * 66)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Traffic-light detection baseline runner")
    p.add_argument("--video", required=True, help="Input dashcam .mp4")
    p.add_argument("--weights", default=os.path.join("weights", "traffic_lights_yolov8x.pt"),
                   help="Path to YOLO .pt weights")
    p.add_argument("--output-dir", default="outputs", help="Output directory")

    p.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="Inference image size (higher = better on small/distant lights, slower)")
    p.add_argument("--device", default=None,
                   help="'0' for first GPU, 'cpu', or leave unset for auto")

    # segment / sampling controls (useful on CPU or for quick previews)
    p.add_argument("--stride", type=int, default=1, help="Process every Nth frame")
    p.add_argument("--start-sec", type=float, default=0.0, help="Start time (seconds)")
    p.add_argument("--end-sec", type=float, default=0.0, help="End time (seconds); 0 = to end")
    p.add_argument("--max-frames", type=int, default=0,
                   help="Max frames to PROCESS (after stride); 0 = no cap")

    # crops
    p.add_argument("--no-save-crops", dest="save_crops", action="store_false",
                   help="Disable saving per-detection crops")
    p.add_argument("--min-crop-conf", type=float, default=0.0,
                   help="Only save crops at/above this confidence (0 = use --conf gating)")
    p.add_argument("--crop-padding", type=int, default=6, help="Pixels of padding around each crop")
    p.add_argument("--crop-every", type=int, default=1,
                   help="Save only every Nth eligible crop (spread evenly across the video). 1 = every detection")
    p.add_argument("--max-crops", type=int, default=0, help="Cap total crops saved; 0 = unlimited")

    p.add_argument("--no-video", action="store_true", help="Skip writing the annotated video")

    # tracking
    p.add_argument("--track", action="store_true",
                   help="Enable multi-object tracking (assigns a persistent track_id per light)")
    p.add_argument("--tracker", default="bytetrack.yaml",
                   help="Ultralytics tracker config: bytetrack.yaml or botsort.yaml")

    p.set_defaults(save_crops=True)
    return p


def main():
    args = build_parser().parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
