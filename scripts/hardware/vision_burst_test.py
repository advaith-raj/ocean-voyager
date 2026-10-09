from pathlib import Path
import argparse
import csv
import json
import shutil
import subprocess
import time

import cv2
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


def capture_image(path: Path, timeout_ms: int, hflip: bool, vflip: bool):
    cmd = [
        "rpicam-still",
        "-o", str(path),
        "--timeout", str(timeout_ms),
        "--nopreview",
    ]
    if hflip:
        cmd.append("--hflip")
    if vflip:
        cmd.append("--vflip")
    subprocess.run(cmd, check=True)


def make_contact_sheet(raw_dir, ann_dir, out_path, title):
    raw_files = sorted(raw_dir.glob("*.jpg"))
    ann_files = sorted(ann_dir.glob("*.jpg"))

    pairs = list(zip(raw_files, ann_files))
    if not pairs:
        return

    thumbs = []
    for raw, ann in pairs:
        raw_img = Image.open(raw).convert("RGB")
        ann_img = Image.open(ann).convert("RGB")

        raw_img.thumbnail((320, 240))
        ann_img.thumbnail((320, 240))

        canvas = Image.new("RGB", (660, 280), "white")
        canvas.paste(raw_img, (0, 30))
        canvas.paste(ann_img, (340, 30))

        draw = ImageDraw.Draw(canvas)
        draw.text((5, 5), raw.name, fill="black")
        draw.text((345, 5), "YOLO annotated", fill="black")
        thumbs.append(canvas)

    sheet_w = 660
    sheet_h = 60 + len(thumbs) * 280
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill="black")

    y = 50
    for t in thumbs:
        sheet.paste(t, (0, y))
        y += 280

    sheet.save(out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True, help="Object/test name, e.g. green_bottle")
    ap.add_argument("--captures", type=int, default=10)
    ap.add_argument("--delay-sec", type=float, default=0.75)
    ap.add_argument("--timeout-ms", type=int, default=1500)
    ap.add_argument("--conf", type=float, default=0.03)
    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--output-root", default="outputs/vision_object_tests")
    ap.add_argument("--expected", choices=["waste", "non_waste", "unknown"], default="unknown")
    ap.add_argument("--clear", action="store_true")
    ap.add_argument("--hflip", action="store_true", default=True)
    ap.add_argument("--vflip", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.output_root) / args.label
    raw_dir = out_dir / "raw"
    ann_dir = out_dir / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.clear and out_dir.exists():
        shutil.rmtree(out_dir)

    raw_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    print("Loading model:", args.model)
    model = YOLO(args.model)

    rows = []
    detections_json = []

    print(f"Capturing {args.captures} images for: {args.label}")
    print("Output:", out_dir)

    for i in range(1, args.captures + 1):
        raw_path = raw_dir / f"{args.label}_{i:02d}.jpg"
        ann_path = ann_dir / f"{args.label}_{i:02d}_annotated.jpg"

        print(f"\n[{i}/{args.captures}] Capturing image...")
        capture_image(raw_path, args.timeout_ms, args.hflip, args.vflip)

        print(f"[{i}/{args.captures}] Running YOLO...")
        results = model.predict(
            source=str(raw_path),
            conf=args.conf,
            imgsz=640,
            device="cpu",
            verbose=False,
        )

        r = results[0]
        frame_detections = []

        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            label = r.names[cls]
            xyxy = [round(float(x), 1) for x in b.xyxy[0].tolist()]
            frame_detections.append(
                {
                    "label": label,
                    "confidence": round(conf, 4),
                    "box_xyxy": xyxy,
                }
            )

        annotated = r.plot()
        cv2.imwrite(str(ann_path), annotated)

        waste_detected = any(d["label"] == "waste" for d in frame_detections)
        best_conf = max([d["confidence"] for d in frame_detections], default=0.0)

        rows.append({
            "frame": i,
            "raw_image": str(raw_path),
            "annotated_image": str(ann_path),
            "waste_detected": waste_detected,
            "num_detections": len(frame_detections),
            "best_confidence": best_conf,
            "detections": json.dumps(frame_detections),
        })

        detections_json.append({
            "frame": i,
            "raw_image": str(raw_path),
            "annotated_image": str(ann_path),
            "waste_detected": waste_detected,
            "detections": frame_detections,
        })

        print(f"[{i}/{args.captures}] waste_detected={waste_detected} best_conf={best_conf}")

        time.sleep(args.delay_sec)

    positive_frames = sum(1 for r in rows if r["waste_detected"])
    vote_2_of_5 = positive_frames >= 2 if args.captures >= 5 else positive_frames >= 1
    vote_3_of_10 = positive_frames >= 3 if args.captures >= 10 else positive_frames >= 1

    summary = {
        "label": args.label,
        "expected": args.expected,
        "captures": args.captures,
        "confidence_threshold": args.conf,
        "positive_frames": positive_frames,
        "negative_frames": args.captures - positive_frames,
        "positive_rate": round(positive_frames / args.captures, 3),
        "vote_2_of_5_style_pass": vote_2_of_5,
        "vote_3_of_10_style_pass": vote_3_of_10,
        "output_dir": str(out_dir),
        "raw_dir": str(raw_dir),
        "annotated_dir": str(ann_dir),
    }

    with (out_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "frame",
                "raw_image",
                "annotated_image",
                "waste_detected",
                "num_detections",
                "best_confidence",
                "detections",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "detections.json").write_text(json.dumps(detections_json, indent=2))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    make_contact_sheet(
        raw_dir,
        ann_dir,
        out_dir / "contact_sheet.jpg",
        f"Vision burst test: {args.label}",
    )

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved:")
    print(" ", out_dir / "summary.json")
    print(" ", out_dir / "predictions.csv")
    print(" ", out_dir / "contact_sheet.jpg")


if __name__ == "__main__":
    main()
