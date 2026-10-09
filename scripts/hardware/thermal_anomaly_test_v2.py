from pathlib import Path
import argparse
import csv
import json
import time
from collections import deque

import numpy as np
import board
import busio
import adafruit_mlx90640
import matplotlib.pyplot as plt


def read_frame_with_retries(mlx, retries=8, delay_sec=0.25):
    frame = [0.0] * 768
    last_error = None

    for _ in range(retries):
        try:
            mlx.getFrame(frame)
            arr = np.array(frame, dtype=float).reshape((24, 32))
            if not np.isfinite(arr).all():
                raise RuntimeError("non-finite thermal frame values")
            return arr, None
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(delay_sec)

    return None, last_error


def sanitize_frame(arr):
    arr = np.array(arr, dtype=float)
    arr = np.where(np.isfinite(arr), arr, np.nan)

    vals = arr[np.isfinite(arr)]
    if vals.size == 0:
        raise RuntimeError("thermal frame has no finite values")

    p2, p5, p50, p95, p98 = np.percentile(vals, [2, 5, 50, 95, 98])

    # Some readings may appear Fahrenheit-like. Convert if the frame median
    # looks like indoor Fahrenheit rather than Celsius.
    if p50 > 45 and p95 < 160:
        arr = (arr - 32.0) * 5.0 / 9.0
        vals = arr[np.isfinite(arr)]
        p2, p5, p50, p95, p98 = np.percentile(vals, [2, 5, 50, 95, 98])

    # Clip extreme bad pixels, but keep real object contrast.
    arr = np.clip(arr, p2, p98)

    return arr


def largest_connected_component(mask):
    mask = np.array(mask, dtype=bool)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)

    best_pixels = []

    for y in range(h):
        for x in range(w):
            if not mask[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True
            pixels = []

            while q:
                cy, cx = q.popleft()
                pixels.append((cy, cx))

                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            q.append((ny, nx))

            if len(pixels) > len(best_pixels):
                best_pixels = pixels

    out = np.zeros_like(mask, dtype=bool)
    for y, x in best_pixels:
        out[y, x] = True

    return out


def choose_localized_mask(score, base_threshold, max_coverage_frac=0.35):
    peak = float(np.max(score))

    thresholds = [
        base_threshold,
        max(base_threshold, peak * 0.35),
        max(base_threshold, peak * 0.50),
        max(base_threshold, peak * 0.65),
        max(base_threshold, peak * 0.80),
    ]

    best_mask = None
    best_threshold = thresholds[-1]
    best_coverage = 1.0

    for th in thresholds:
        mask = score >= th
        mask = largest_connected_component(mask)

        coverage = float(mask.sum()) / float(mask.size)

        best_mask = mask
        best_threshold = th
        best_coverage = coverage

        if 0 < coverage <= max_coverage_frac:
            break

    return best_mask, best_threshold, best_coverage


def analyze_frame(arr, mode, threshold_delta_c, min_blob_pixels, fov_x_deg):
    arr = sanitize_frame(arr)

    valid = arr[np.isfinite(arr)]
    background = float(np.median(valid))

    warm_score = arr - background
    cold_score = background - arr

    warm_peak = float(np.max(warm_score))
    cold_peak = float(np.max(cold_score))

    if mode == "warm":
        selected_mode = "warm"
        score = warm_score
        peak_delta = warm_peak
        object_temp = float(np.max(arr))
    elif mode == "cold":
        selected_mode = "cold"
        score = cold_score
        peak_delta = cold_peak
        object_temp = float(np.min(arr))
    elif mode == "both":
        if warm_peak >= cold_peak:
            selected_mode = "warm"
            score = warm_score
            peak_delta = warm_peak
            object_temp = float(np.max(arr))
        else:
            selected_mode = "cold"
            score = cold_score
            peak_delta = cold_peak
            object_temp = float(np.min(arr))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    raw_mask = score >= threshold_delta_c
    raw_blob_px = int(raw_mask.sum())
    raw_coverage_frac = raw_blob_px / int(arr.size)

    mask, used_threshold, coverage_frac = choose_localized_mask(
        score,
        base_threshold=threshold_delta_c,
        max_coverage_frac=0.35,
    )

    blob_px = int(mask.sum())

    full_frame_rejected = bool(raw_coverage_frac > 0.75)
    too_large_rejected = bool(coverage_frac > 0.35)

    candidate = bool(
        peak_delta >= threshold_delta_c
        and blob_px >= min_blob_pixels
        and not full_frame_rejected
        and not too_large_rejected
    )

    if blob_px > 0:
        ys, xs = np.where(mask)
        cx = float(xs.mean())
        cy = float(ys.mean())
    else:
        py, px = np.unravel_index(np.argmax(score), score.shape)
        cx = float(px)
        cy = float(py)

    x_center = (32 - 1) / 2.0
    rel_x = (cx - x_center) / x_center
    bearing_deg = rel_x * (fov_x_deg / 2.0)

    confidence = max(0.0, min(1.0, peak_delta / (threshold_delta_c * 3.0)))

    if peak_delta >= threshold_delta_c * 3:
        strength = "strong"
    elif peak_delta >= threshold_delta_c:
        strength = "medium"
    else:
        strength = "weak"

    return {
        "candidate": candidate,
        "selected_mode": selected_mode,
        "background_temp_c": round(background, 3),
        "object_temp_c": round(object_temp, 3),
        "delta_c": round(peak_delta, 3),
        "warm_peak_delta_c": round(warm_peak, 3),
        "cold_peak_delta_c": round(cold_peak, 3),
        "raw_blob_px": raw_blob_px,
        "raw_coverage_frac": round(raw_coverage_frac, 3),
        "blob_px": blob_px,
        "coverage_frac": round(coverage_frac, 3),
        "used_threshold_delta_c": round(float(used_threshold), 3),
        "full_frame_rejected": full_frame_rejected,
        "too_large_rejected": too_large_rejected,
        "centroid_x": round(cx, 3),
        "centroid_y": round(cy, 3),
        "bearing_deg": round(bearing_deg, 3),
        "confidence": round(confidence, 3),
        "strength": strength,
    }


def save_heatmap(arr, analysis, out_path, title):
    arr = sanitize_frame(arr)

    plt.figure(figsize=(8, 6))
    plt.imshow(arr, cmap="inferno", origin="lower")
    plt.colorbar(label="Temperature (°C)")

    cx = analysis["centroid_x"]
    cy = analysis["centroid_y"]
    plt.scatter([cx], [cy], marker="x", s=120)

    plt.title(title)
    plt.xlabel("X thermal pixel")
    plt.ylabel("Y thermal pixel")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["warm", "cold", "both"], default="cold")
    ap.add_argument("--samples", type=int, default=15)
    ap.add_argument("--delay-sec", type=float, default=1.0)
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--min-blob-pixels", type=int, default=6)
    ap.add_argument("--fov-x-deg", type=float, default=55.0)
    ap.add_argument("--output", default="outputs/current_thermal_anomaly_test")
    ap.add_argument("--clear", action="store_true")
    args = ap.parse_args()

    out = Path(args.output)
    frames_dir = out / "thermal_frames"
    preview_dir = out / "thermal_preview"

    if args.clear and out.exists():
        import shutil
        shutil.rmtree(out)

    frames_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    print("=== Thermal anomaly test ===")
    print("mode:", args.mode)
    print("threshold_delta_c:", args.threshold)
    print("min_blob_pixels:", args.min_blob_pixels)
    print("output:", out)

    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ

    print("Warming up / settling sensor...")
    time.sleep(1.0)

    rows = []

    for sample in range(1, args.samples + 1):
        arr, err = read_frame_with_retries(mlx)

        if arr is None:
            row = {
                "sample": sample,
                "ok": False,
                "candidate": False,
                "selected_mode": args.mode,
                "background_temp_c": "",
                "object_temp_c": "",
                "delta_c": "",
                "warm_peak_delta_c": "",
                "cold_peak_delta_c": "",
                "raw_blob_px": "",
                "raw_coverage_frac": "",
                "blob_px": "",
                "coverage_frac": "",
                "used_threshold_delta_c": "",
                "full_frame_rejected": "",
                "too_large_rejected": "",
                "centroid_x": "",
                "centroid_y": "",
                "bearing_deg": "",
                "confidence": "",
                "strength": "",
                "frame_path": "",
                "preview_path": "",
                "error": err,
            }
            rows.append(row)
            print(f"[{sample}] FAILED {err}")
            continue

        analysis = analyze_frame(
            arr,
            mode=args.mode,
            threshold_delta_c=args.threshold,
            min_blob_pixels=args.min_blob_pixels,
            fov_x_deg=args.fov_x_deg,
        )

        frame_path = frames_dir / f"thermal_{sample:03d}.npy"
        preview_path = preview_dir / f"thermal_{sample:03d}.png"

        np.save(frame_path, sanitize_frame(arr))

        title = (
            f"sample {sample} | mode={analysis['selected_mode']} | "
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
        rows.append(row)

        print(
            f"[{sample}] candidate={analysis['candidate']} "
            f"mode={analysis['selected_mode']} "
            f"bearing={analysis['bearing_deg']} "
            f"delta={analysis['delta_c']} "
            f"cold_delta={analysis['cold_peak_delta_c']} "
            f"warm_delta={analysis['warm_peak_delta_c']} "
            f"blob_px={analysis['blob_px']} "
            f"coverage={analysis['coverage_frac']} "
            f"raw_blob_px={analysis['raw_blob_px']} "
            f"raw_coverage={analysis['raw_coverage_frac']} "
            f"full_frame_rejected={analysis['full_frame_rejected']} "
            f"conf={analysis['confidence']} "
            f"strength={analysis['strength']}"
        )

        time.sleep(args.delay_sec)

    csv_path = out / "thermal_anomaly_readings.csv"
    fieldnames = [
        "sample",
        "ok",
        "candidate",
        "selected_mode",
        "background_temp_c",
        "object_temp_c",
        "delta_c",
        "warm_peak_delta_c",
        "cold_peak_delta_c",
        "raw_blob_px",
        "raw_coverage_frac",
        "blob_px",
        "coverage_frac",
        "used_threshold_delta_c",
        "full_frame_rejected",
        "too_large_rejected",
        "centroid_x",
        "centroid_y",
        "bearing_deg",
        "confidence",
        "strength",
        "frame_path",
        "preview_path",
        "error",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    ok_rows = [r for r in rows if r["ok"] is True]
    candidate_rows = [r for r in ok_rows if r["candidate"] is True]

    best = None
    if ok_rows:
        best = max(ok_rows, key=lambda r: float(r["delta_c"]))

    summary = {
        "mode": args.mode,
        "threshold_delta_c": args.threshold,
        "min_blob_pixels": args.min_blob_pixels,
        "samples": args.samples,
        "ok_samples": len(ok_rows),
        "candidate_count": len(candidate_rows),
        "triggered": len(candidate_rows) > 0,
        "best": best,
        "output_dir": str(out),
        "csv_path": str(csv_path),
        "preview_dir": str(preview_dir),
        "frames_dir": str(frames_dir),
    }

    summary_path = out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))
    print("\nSaved output:", out)


if __name__ == "__main__":
    main()
