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
from copy import deepcopy
from typing import Any

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Bring up real/simulated cameras and optionally run the vision model.")
    parser.add_argument("--config", default="config/pi_hardware_sensors_log_only.yaml")
    parser.add_argument("--scenario", default="scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml")
    parser.add_argument("--run-name", default="camera_bringup")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--camera-ids", default=None, help="Comma-separated list. Default from config.")
    parser.add_argument("--run-vision", action="store_true", help="Run configured vision model on captured frames.")
    parser.add_argument("--use-sim", action="store_true", help="Force simulated cameras; useful for testing the script without hardware.")
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

    # For camera-only bring-up, thermal is not required.
    cfg.setdefault("startup", {})
    cfg["startup"]["require_devices"] = ["camera_array", "motors", "vision_model", "safety"]
    cfg["startup"]["abort_on_failure"] = True
    cfg["startup"]["allow_degraded_mode"] = False

    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool=pool, robot_pose=pose, objects=objects)
    logger = MissionLogger(cfg.get("logging", {}).get("base_output_dir", "outputs/hardware_bringup"), args.run_name)
    logger.save_static_inputs(args.config, args.scenario)

    runtime = build_runtime_bundle(world, cfg, logger.output_dir, command_log=[])
    report = run_startup_sequence(cfg, logger.output_dir, runtime.as_startup_dict(), bus=runtime.device_bus)
    if report.abort_mission:
        summary = {"mode": "camera_bringup", "outcome": "startup_failed", "startup": report.as_dict(), "output_dir": str(logger.output_dir)}
        logger.finalize(summary)
        print("Camera startup failed. Check startup/startup_log.txt")
        return 2

    camera_ids = [x.strip() for x in args.camera_ids.split(",")] if args.camera_ids else cfg.get("sensors", {}).get("camera", {}).get("camera_ids", ["front", "left", "right"])
    rows = []
    detections = []
    for i in range(1, args.samples + 1):
        for camera_id in camera_ids:
            try:
                frame = runtime.camera.get_frame(camera_id)
                row = {
                    "sample": i,
                    "camera_id": camera_id,
                    "ok": True,
                    "image_path": frame.image_path,
                    "visible_objects_sim_only": [o.as_dict() for o in frame.visible_objects],
                    "error": "",
                }
                print(f"[{i}] camera={camera_id} image={frame.image_path}")
                if args.run_vision:
                    preds = runtime.vision_model.predict(frame)
                    pred_dicts = [p.as_dict() for p in preds]
                    detections.append({"sample": i, "camera_id": camera_id, "image_path": frame.image_path, "detections": pred_dicts})
                    print(f"    detections={pred_dicts}")
            except Exception as exc:
                row = {
                    "sample": i,
                    "camera_id": camera_id,
                    "ok": False,
                    "image_path": "",
                    "visible_objects_sim_only": [],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"[{i}] camera={camera_id} ERROR {row['error']}")
            rows.append(row)

    csv_path = logger.output_dir / "camera_frames.csv"
    det_path = logger.output_dir / "vision_detections.json"
    fields = ["sample", "camera_id", "ok", "image_path", "visible_objects_sim_only", "error"]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    det_path.write_text(json.dumps(detections, indent=2, default=str) + "\n")

    summary = {
        "mode": "camera_bringup",
        "outcome": "complete",
        "samples": args.samples,
        "camera_ids": camera_ids,
        "frames_ok": sum(1 for r in rows if r["ok"]),
        "frames_failed": sum(1 for r in rows if not r["ok"]),
        "vision_run": args.run_vision,
        "startup": report.as_dict(),
        "camera_frames_csv": str(csv_path),
        "vision_detections_json": str(det_path),
        "output_dir": str(logger.output_dir),
    }
    logger.finalize(summary)

    print(f"Camera bring-up complete. Output directory: {logger.output_dir}")
    print(f"Frames CSV: {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
