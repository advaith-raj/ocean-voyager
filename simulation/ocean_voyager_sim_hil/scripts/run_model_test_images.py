#!/usr/bin/env python3
"""Run the trained YOLO model on positive/negative test image folders.

Example:
  python scripts/run_model_test_images.py --model models/best.pt --images assets/test_images --output outputs/model_eval

Expected input layout:
  assets/test_images/positives/*.jpg|png  -> expected target present
  assets/test_images/negatives/*.jpg|png  -> expected no target
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import cv2
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ocean_voyager.core.types import CameraFrame
from ocean_voyager.perception.vision_model import YoloVisionModel

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_images(root: Path) -> Iterable[tuple[Path, str]]:
    positives = root / "positives"
    negatives = root / "negatives"
    for expected, folder in [("target", positives), ("non_target", negatives)]:
        if not folder.exists():
            continue
        for p in sorted(folder.rglob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                yield p, expected


def classify_row(expected: str, has_target: bool) -> str:
    if expected == "target" and has_target:
        return "TP"
    if expected == "target" and not has_target:
        return "FN"
    if expected == "non_target" and has_target:
        return "FP"
    return "TN"


def annotate(image_path: Path, detections: list, output_path: Path) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        return
    for d in detections:
        if not d.bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{d.label} {d.confidence:.2f}"
        cv2.putText(img, label, (x1, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), img)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to YOLO .pt model")
    parser.add_argument("--images", default="assets/test_images", help="Root containing positives/ and negatives/")
    parser.add_argument("--output", default="outputs/model_eval", help="Output directory")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--target-labels", default="waste,bottle,plastic,plastic_bottle", help="Comma-separated labels treated as target")
    args = parser.parse_args()

    root = Path(args.images)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ann_dir = out_dir / "annotated_images"
    target_labels = {x.strip().lower() for x in args.target_labels.split(",") if x.strip()}

    model = YoloVisionModel(args.model, args.conf, sorted(target_labels))
    rows = []
    for image_path, expected in iter_images(root):
        frame = CameraFrame(camera_id="eval", image_path=str(image_path), visible_objects=[])
        detections = model.predict(frame)
        has_target = any(d.label.lower() in target_labels and d.confidence >= args.conf for d in detections)
        best = max(detections, key=lambda d: d.confidence) if detections else None
        result = classify_row(expected, has_target)
        row = {
            "image": str(image_path),
            "expected": expected,
            "has_target": has_target,
            "result": result,
            "best_label": best.label if best else "none",
            "best_confidence": round(float(best.confidence), 4) if best else 0.0,
            "detections": [d.as_dict() for d in detections],
        }
        rows.append(row)
        annotate(image_path, detections, ann_dir / image_path.name)

    csv_path = out_dir / "predictions.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "expected", "has_target", "result", "best_label", "best_confidence"])
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    (out_dir / "predictions.json").write_text(json.dumps(rows, indent=2))
    total = len(rows)
    counts = {k: sum(1 for r in rows if r["result"] == k) for k in ["TP", "TN", "FP", "FN"]}
    summary = {
        "model": args.model,
        "images_root": str(root),
        "confidence_threshold": args.conf,
        "target_labels": sorted(target_labels),
        "total_images": total,
        **counts,
        "accuracy": round((counts["TP"] + counts["TN"]) / total, 4) if total else None,
        "note": "If total_images is 0, add images under positives/ and negatives/.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
