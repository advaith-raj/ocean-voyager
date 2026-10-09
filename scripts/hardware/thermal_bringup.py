#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import csv
import json
import time
from copy import deepcopy
from typing import Any

import numpy as np

from ocean_voyager.core.config_loader import load_config, load_scenario, parse_world
from ocean_voyager.core.factory import build_runtime_bundle
from ocean_voyager.core.logger import MissionLogger
from ocean_voyager.core.startup import run_startup_sequence
from ocean_voyager.simulation.world import SimWorld


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def thermal_strength(reading) -> str:
    c = float(getattr(reading, "confidence", 0.0))
    if c >= 0.67:
        return "strong"
    if c >= 0.34:
        return "medium"
    if c > 0:
        return "weak"
    return "none"


def save_preview(frame_path: str | None, output_path: Path) -> str | None:
    if not frame_path:
        return None
    p = Path(frame_path)
    if not p.exists() or p.suffix.lower() != ".npy":
        return None
    arr = np.load(p)
    arr = np.asarray(arr, dtype=float)
    mn, mx = float(np.min(arr)), float(np.max(arr))
    if mx - mn < 1e-9:
        norm = np.zeros_like(arr, dtype=np.uint8)
    else:
        norm = ((arr - mn) / (mx - mn) * 255).clip(0, 255).astype(np.uint8)
    try:
        from PIL import Image
        img = Image.fromarray(norm, mode="L").resize((320, 240))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        return str(output_path)
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Bring up a real/simulated MLX90640-style thermal sensor and log candidate/bearing readings.")
    parser.add_argument("--config", default="config/pi_hardware_sensors_log_only.yaml")
    parser.add_argument("--scenario", default="scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml")
    parser.add_argument("--run-name", default="thermal_bringup")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--delay-sec", type=float, default=0.5)
    parser.add_argument("--interactive", action="store_true", help="Prompt before every sample.")
    parser.add_argument("--use-sim", action="store_true", help="Force simulated thermal; useful for testing the script on a laptop/Pi without hardware.")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    scenario = load_scenario(ROOT / args.scenario)
    cfg["_scenario"] = scenario
    cfg["scenario_name"] = scenario.get("scenario_name", Path(args.scenario).stem)
    if args.use_sim:
        cfg.setdefault("sources", {})["thermal"] = "sim"
        cfg.setdefault("sources", {})["camera"] = "sim"
        cfg.setdefault("sources", {})["motors"] = "log_only"
    else:
        cfg.setdefault("sources", {})["motors"] = "log_only"

    # For thermal-only bring-up, camera is not required.
    cfg.setdefault("startup", {})
    cfg["startup"]["require_devices"] = ["thermal", "motors", "vision_model", "safety"]
    cfg["startup"]["abort_on_failure"] = True
    cfg["startup"]["allow_degraded_mode"] = False

    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool=pool, robot_pose=pose, objects=objects)
    logger = MissionLogger(cfg.get("logging", {}).get("base_output_dir", "outputs/hardware_bringup"), args.run_name)
    logger.save_static_inputs(args.config, args.scenario)

    runtime = build_runtime_bundle(world, cfg, logger.output_dir, command_log=[])
    report = run_startup_sequence(cfg, logger.output_dir, runtime.as_startup_dict(), bus=runtime.device_bus)
    if report.abort_mission:
        summary = {"mode": "thermal_bringup", "outcome": "startup_failed", "startup": report.as_dict(), "output_dir": str(logger.output_dir)}
        logger.finalize(summary)
        print("Thermal startup failed. Check startup/startup_log.txt")
        return 2

    rows = []
    previews_dir = logger.output_dir / "thermal_preview"
    for i in range(1, args.samples + 1):
        if args.interactive:
            input(f"[Sample {i}/{args.samples}] Point thermal sensor at setup, then press Enter...")
        readings = runtime.thermal.scan()
        now = time.time()
        for j, r in enumerate(readings):
            preview_path = save_preview(r.frame_path, previews_dir / f"thermal_{i:03d}_{j:02d}.png")
            row = {
                "sample": i,
                "reading_index": j,
                "time_unix": round(now, 3),
                "candidate": r.candidate,
                "bearing_deg": r.bearing_deg,
                "confidence": r.confidence,
                "strength": thermal_strength(r),
                "observed_delta_c": r.observed_delta_c,
                "blob_pixel_count": r.blob_pixel_count,
                "peak_pixel": r.peak_pixel,
                "blob_centroid_pixel": r.blob_centroid_pixel,
                "frame_path": r.frame_path,
                "preview_path": preview_path,
                "range_known": r.range_known,
                "range_source": r.range_source,
            }
            rows.append(row)
            print(f"[{i}] candidate={row['candidate']} bearing={row['bearing_deg']} conf={row['confidence']} strength={row['strength']} delta={row['observed_delta_c']} blob_px={row['blob_pixel_count']}")
        if not args.interactive and i < args.samples:
            time.sleep(args.delay_sec)

    csv_path = logger.output_dir / "thermal_readings.csv"
    json_path = logger.output_dir / "thermal_summary.json"
    fields = [
        "sample", "reading_index", "time_unix", "candidate", "bearing_deg", "confidence", "strength",
        "observed_delta_c", "blob_pixel_count", "peak_pixel", "blob_centroid_pixel", "frame_path",
        "preview_path", "range_known", "range_source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "mode": "thermal_bringup",
        "outcome": "complete",
        "samples": args.samples,
        "candidate_count": sum(1 for r in rows if str(r["candidate"]).lower() == "true" or r["candidate"] is True),
        "max_confidence": max([float(r["confidence"]) for r in rows], default=0.0),
        "max_delta_c": max([float(r["observed_delta_c"]) for r in rows], default=0.0),
        "startup": report.as_dict(),
        "thermal_readings_csv": str(csv_path),
        "thermal_summary_json": str(json_path),
        "output_dir": str(logger.output_dir),
    }
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    logger.finalize(summary)

    print(f"Thermal bring-up complete. Output directory: {logger.output_dir}")
    print(f"Readings CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
