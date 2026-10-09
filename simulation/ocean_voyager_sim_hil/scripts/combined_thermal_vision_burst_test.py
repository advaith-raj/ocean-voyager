from pathlib import Path
import argparse
import csv
import json
import subprocess
import sys
import time

import numpy as np
import board
import busio
import adafruit_mlx90640
from PIL import Image, ImageDraw

# Reuse the already-working thermal anomaly code.
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from thermal_anomaly_test import (  # noqa: E402
    read_frame_with_retries,
    sanitize_frame,
    analyze_frame,
    save_heatmap,
)


def capture_image(out_path, timeout_ms=1500, hflip=True, vflip=True):
    cmd = [
        "rpicam-still",
        "-o",
        str(out_path),
        "--timeout",
        str(timeout_ms),
        "--nopreview",
    ]

    if hflip:
        cmd.append("--hflip")
    if vflip:
        cmd.append("--vflip")

    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_yolo_on_image(model, image_path, conf, expected_labels):
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    names = result.names

    detections = []
    positive = False

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, cls_id))
            score = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            is_expected = label.lower() in expected_labels

            if is_expected:
                positive = True

            detections.append(
                {
                    "label": label,
                    "confidence": round(score, 4),
                    "box_xyxy": [round(v, 2) for v in xyxy],
                    "is_expected": is_expected,
                }
            )

    return positive, detections


def annotate_image(image_path, out_path, detections):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = det["label"]
        conf = det["confidence"]
        text = f"{label} {conf:.2f}"

        draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
        draw.text((x1 + 5, max(0, y1 - 22)), text, fill="lime")

    img.save(out_path)


def make_contact_sheet(image_paths, out_path, thumb_width=400):
    thumbs = []

    for p in image_paths:
        try:
            img = Image.open(p).convert("RGB")
            ratio = thumb_width / img.width
            thumb_height = int(img.height * ratio)
            thumbs.append(img.resize((thumb_width, thumb_height)))
        except Exception:
            pass

    if not thumbs:
        return

    cols = min(3, len(thumbs))
    rows = int(np.ceil(len(thumbs) / cols))
    thumb_height = thumbs[0].height

    sheet = Image.new("RGB", (cols * thumb_width, rows * thumb_height), "white")

    for idx, img in enumerate(thumbs):
        x = (idx % cols) * thumb_width
        y = (idx // cols) * thumb_height
        sheet.paste(img, (x, y))

    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--label", default="combined_test")
    ap.add_argument("--output-root", default="outputs/combined_sensor_tests")

    ap.add_argument("--mode", choices=["warm", "cold", "both"], default="cold")
    ap.add_argument("--thermal-samples", type=int, default=10)
    ap.add_argument("--thermal-delay-sec", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--min-blob-pixels", type=int, default=6)
    ap.add_argument("--thermal-vote-k", type=int, default=1)
    ap.add_argument("--fov-x-deg", type=float, default=55.0)

    ap.add_argument("--camera-captures", type=int, default=5)
    ap.add_argument("--camera-timeout-ms", type=int, default=1500)
    ap.add_argument("--camera-delay-sec", type=float, default=0.5)
    ap.add_argument("--no-hflip", action="store_true")
    ap.add_argument("--no-vflip", action="store_true")

    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--conf", type=float, default=0.03)
    ap.add_argument("--vote-k", type=int, default=2)
    ap.add_argument("--expected-labels", default="waste,bottle,plastic,plastic_bottle")

    ap.add_argument("--clear", action="store_true")

    args = ap.parse_args()

    out = Path(args.output_root) / args.label
    thermal_frames_dir = out / "thermal_frames"
    thermal_preview_dir = out / "thermal_preview"
    camera_raw_dir = out / "camera_raw"
    camera_annotated_dir = out / "camera_annotated"

    if args.clear and out.exists():
        import shutil

        shutil.rmtree(out)

    for d in [thermal_frames_dir, thermal_preview_dir, camera_raw_dir, camera_annotated_dir]:
        d.mkdir(parents=True, exist_ok=True)

    expected_labels = {
        x.strip().lower()
        for x in args.expected_labels.split(",")
        if x.strip()
    }

    print("=== Combined thermal + vision burst test ===")
    print("label:", args.label)
    print("output:", out)
    print("thermal mode:", args.mode)
    print("thermal threshold:", args.threshold)
    print("vision model:", args.model)
    print("vision conf:", args.conf)

    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ

    print("\nThermal sensor settling...")
    time.sleep(1.0)

    thermal_rows = []

    for sample in range(1, args.thermal_samples + 1):
        arr, err = read_frame_with_retries(mlx)

        if arr is None:
            row = {
                "sample": sample,
                "ok": False,
                "candidate": False,
                "error": err,
            }
            thermal_rows.append(row)
            print(f"[thermal {sample}] FAILED {err}")
            continue

        analysis = analyze_frame(
            arr,
            mode=args.mode,
            threshold_delta_c=args.threshold,
            min_blob_pixels=args.min_blob_pixels,
            fov_x_deg=args.fov_x_deg,
        )

        frame_path = thermal_frames_dir / f"thermal_{sample:03d}.npy"
        preview_path = thermal_preview_dir / f"thermal_{sample:03d}.png"

        np.save(frame_path, sanitize_frame(arr))

        title = (
            f"thermal {sample} | mode={analysis['selected_mode']} | "
            f"candidate={analysis['candidate']} | "
            f"delta={analysis['delta_c']}C | "
            f"bearing={analysis['bearing_deg']}deg"
        )
        save_heatmap(arr, analysis, preview_path, title)

        row = {
            "sample": sample,
            "ok": True,
            **analysis,
            "frame_path": str(frame_path),
            "preview_path": str(preview_path),
            "error": "",
        }
        thermal_rows.append(row)

        print(
            f"[thermal {sample}] candidate={analysis['candidate']} "
            f"mode={analysis['selected_mode']} "
            f"bearing={analysis['bearing_deg']} "
            f"delta={analysis['delta_c']} "
            f"blob_px={analysis['blob_px']} "
            f"conf={analysis['confidence']}"
        )

        time.sleep(args.thermal_delay_sec)

    thermal_ok_rows = [r for r in thermal_rows if r.get("ok") is True]
    thermal_candidate_rows = [r for r in thermal_ok_rows if r.get("candidate") is True]
    thermal_triggered = len(thermal_candidate_rows) >= args.thermal_vote_k

    best_thermal = None
    if thermal_ok_rows:
        best_thermal = max(thermal_ok_rows, key=lambda r: float(r.get("delta_c", 0)))

    thermal_csv = out / "thermal_readings.csv"
    thermal_fieldnames = sorted({k for row in thermal_rows for k in row.keys()})

    with thermal_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=thermal_fieldnames)
        writer.writeheader()
        writer.writerows(thermal_rows)

    print("\nLoading YOLO model...")
    from ultralytics import YOLO

    model = YOLO(args.model)

    vision_rows = []
    annotated_paths = []

    print("Capturing AI camera burst...")

    for i in range(1, args.camera_captures + 1):
        raw_path = camera_raw_dir / f"camera_{i:03d}.jpg"
        ann_path = camera_annotated_dir / f"camera_{i:03d}_annotated.jpg"

        try:
            capture_image(
                raw_path,
                timeout_ms=args.camera_timeout_ms,
                hflip=not args.no_hflip,
                vflip=not args.no_vflip,
            )

            positive, detections = run_yolo_on_image(
                model,
                raw_path,
                conf=args.conf,
                expected_labels=expected_labels,
            )

            annotate_image(raw_path, ann_path, detections)
            annotated_paths.append(ann_path)
            err = ""

        except Exception as exc:
            positive = False
            detections = []
            err = f"{type(exc).__name__}: {exc}"

        row = {
            "capture": i,
            "ok": err == "",
            "positive": positive,
            "raw_path": str(raw_path),
            "annotated_path": str(ann_path),
            "num_detections": len(detections),
            "detections_json": json.dumps(detections),
            "error": err,
        }
        vision_rows.append(row)

        print(
            f"[vision {i}] ok={row['ok']} "
            f"positive={positive} "
            f"detections={len(detections)} "
            f"error={err}"
        )

        time.sleep(args.camera_delay_sec)

    vision_positive_rows = [r for r in vision_rows if r["positive"] is True]
    vision_vote_passed = len(vision_positive_rows) >= args.vote_k

    vision_csv = out / "vision_readings.csv"

    with vision_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "capture",
                "ok",
                "positive",
                "raw_path",
                "annotated_path",
                "num_detections",
                "detections_json",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(vision_rows)

    contact_sheet = out / "vision_contact_sheet.jpg"
    make_contact_sheet(annotated_paths, contact_sheet)

    if thermal_triggered and vision_vote_passed:
        final_decision = "confirmed_waste"
    elif thermal_triggered and not vision_vote_passed:
        final_decision = "thermal_candidate_but_vision_rejected"
    elif not thermal_triggered and vision_vote_passed:
        final_decision = "vision_positive_without_thermal_trigger"
    else:
        final_decision = "no_candidate"

    summary = {
        "label": args.label,
        "output_dir": str(out),
        "thermal_mode": args.mode,
        "thermal_samples": args.thermal_samples,
        "thermal_ok_samples": len(thermal_ok_rows),
        "thermal_candidate_count": len(thermal_candidate_rows),
        "thermal_vote_k": args.thermal_vote_k,
        "thermal_triggered": thermal_triggered,
        "best_thermal": best_thermal,
        "thermal_csv": str(thermal_csv),
        "vision_captures": args.camera_captures,
        "vision_positive_frames": len(vision_positive_rows),
        "vision_vote_k": args.vote_k,
        "vision_vote_passed": vision_vote_passed,
        "vision_csv": str(vision_csv),
        "vision_contact_sheet": str(contact_sheet),
        "final_decision": final_decision,
    }

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved output:", out)


if __name__ == "__main__":
    main()
