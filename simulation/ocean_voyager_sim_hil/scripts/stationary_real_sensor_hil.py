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

from ocean_voyager.core.config_loader import load_config, load_scenario, parse_world
from ocean_voyager.core.factory import build_runtime_bundle
from ocean_voyager.core.logger import MissionLogger
from ocean_voyager.core.startup import run_startup_sequence
from ocean_voyager.core.types import MotionCommand
from ocean_voyager.simulation.world import SimWorld


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def strength(confidence: float) -> str:
    if confidence >= 0.67:
        return "strong"
    if confidence >= 0.34:
        return "medium"
    if confidence > 0:
        return "weak"
    return "none"


def best_reading(readings):
    candidates = [r for r in readings if getattr(r, "candidate", False)]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(getattr(r, "confidence", 0.0)))


def is_waste_detection(det, target_labels: set[str], min_conf: float) -> bool:
    return det.label.lower() in target_labels and float(det.confidence) >= min_conf


def main() -> int:
    parser = argparse.ArgumentParser(description="Stationary real-sensor HIL: real thermal + real camera, motors log-only, operator manually moves the target.")
    parser.add_argument("--config", default="config/stationary_real_thermal_real_camera_log_only.yaml")
    parser.add_argument("--scenario", default="scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml")
    parser.add_argument("--run-name", default="stationary_real_sensor_hil")
    parser.add_argument("--cycles", type=int, default=8, help="Number of thermal->camera cycles.")
    parser.add_argument("--camera-ids", default=None, help="Comma-separated camera IDs; default from config.")
    parser.add_argument("--interactive", action="store_true", help="Prompt operator to move object closer after each simulated approach command.")
    parser.add_argument("--auto-continue-sec", type=float, default=0.0, help="Delay between cycles when not interactive.")
    parser.add_argument("--thermal-min-confidence", type=float, default=None)
    parser.add_argument("--vision-min-confidence", type=float, default=None)
    parser.add_argument("--target-labels", default=None, help="Comma-separated labels treated as waste; default from config.")
    parser.add_argument("--use-sim", action="store_true", help="Force simulated thermal/camera for dry-run testing.")
    parser.add_argument("--mock-vision", action="store_true", help="Force mock vision to avoid YOLO dependency.")
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

    if args.mock_vision:
        cfg["vision_model"] = {"type": "mock", "confidence_by_type": {"bottle": 0.90, "seaweed": 0.78, "glare": 0.70}}

    cfg.setdefault("startup", {})
    cfg["startup"]["require_devices"] = ["thermal", "camera_array", "motors", "vision_model", "safety"]
    cfg["startup"]["abort_on_failure"] = True
    cfg["startup"]["allow_degraded_mode"] = False

    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool=pool, robot_pose=pose, objects=objects)
    logger = MissionLogger(cfg.get("logging", {}).get("base_output_dir", "outputs/hardware_bringup"), args.run_name)
    logger.save_static_inputs(args.config, args.scenario)

    runtime = build_runtime_bundle(world, cfg, logger.output_dir, command_log=[])
    report = run_startup_sequence(cfg, logger.output_dir, runtime.as_startup_dict(), bus=runtime.device_bus)
    if report.abort_mission:
        summary = {"mode": "stationary_real_sensor_hil", "outcome": "startup_failed", "startup": report.as_dict(), "output_dir": str(logger.output_dir)}
        logger.finalize(summary)
        print("Stationary real-sensor HIL startup failed. Check startup/startup_log.txt")
        return 2

    mission_cfg = cfg.get("mission", {})
    thermal_min = float(args.thermal_min_confidence if args.thermal_min_confidence is not None else mission_cfg.get("thermal_min_confidence", 0.12))
    vision_min = float(args.vision_min_confidence if args.vision_min_confidence is not None else mission_cfg.get("vision_min_confidence", 0.50))
    camera_ids = [x.strip() for x in args.camera_ids.split(",")] if args.camera_ids else cfg.get("sensors", {}).get("camera", {}).get("camera_ids", ["front", "left", "right"])
    labels_from_cfg = cfg.get("vision_model", {}).get("target_labels", ["waste", "bottle", "plastic", "plastic_bottle"])
    target_labels = {x.strip().lower() for x in args.target_labels.split(",")} if args.target_labels else {str(x).lower() for x in labels_from_cfg}
    step_distance = float(mission_cfg.get("manual_hil_step_distance_m", 0.25))
    command_duration = float(mission_cfg.get("bearing_approach_duration_s", 0.45))
    command_forward = float(mission_cfg.get("bearing_approach_forward", 0.22))

    events = []
    detections_log = []
    confirmed = False
    rejected = False

    print("Stationary real-sensor HIL started.")
    print("Robot will NOT physically move; motors are log-only.")
    print("When prompted, manually move the target closer to the stationary sensors.")

    for cycle in range(1, args.cycles + 1):
        readings = runtime.thermal.scan()
        r = best_reading(readings)
        thermal_candidate = r is not None and float(r.confidence) >= thermal_min
        event = {
            "cycle": cycle,
            "thermal_candidate": thermal_candidate,
            "thermal_bearing_deg": "" if r is None else r.bearing_deg,
            "thermal_confidence": "" if r is None else r.confidence,
            "thermal_strength": "none" if r is None else strength(float(r.confidence)),
            "thermal_delta_c": "" if r is None else r.observed_delta_c,
            "thermal_blob_pixel_count": "" if r is None else r.blob_pixel_count,
            "simulated_action": "hold",
            "vision_confirmed": False,
            "vision_best_label": "",
            "vision_best_confidence": "",
            "decision": "continue",
        }

        if thermal_candidate:
            bearing = float(r.bearing_deg)
            yaw = max(-1.0, min(1.0, bearing / 45.0))
            cmd = MotionCommand(yaw=yaw, forward=command_forward, duration_s=command_duration, action="APPROACH_STEP", reason=f"manual HIL approach toward bearing {bearing:.1f} deg")
            runtime.motors.execute(cmd)
            event["simulated_action"] = f"approach_step bearing={bearing:.1f}deg forward~{step_distance:.2f}m"
            print(f"[Cycle {cycle}] THERMAL {event['thermal_strength']} candidate: bearing={bearing:.1f}°, conf={float(r.confidence):.2f}, delta={float(r.observed_delta_c):.2f}, blob_px={r.blob_pixel_count}")
            print(f"[Cycle {cycle}] Simulated Voyager would move about {step_distance:.2f} m toward bearing {bearing:.1f}°.")
            if args.interactive:
                input("Move the target about that much closer/aligned with the bearing, then press Enter...")
        else:
            print(f"[Cycle {cycle}] No thermal candidate above threshold {thermal_min}.")
            if args.interactive:
                input("Adjust object/scene if needed, then press Enter to capture camera frames...")

        best_det = None
        for camera_id in camera_ids:
            try:
                frame = runtime.camera.get_frame(camera_id)
                preds = runtime.vision_model.predict(frame)
                pred_dicts = [p.as_dict() for p in preds]
                detections_log.append({"cycle": cycle, "camera_id": camera_id, "image_path": frame.image_path, "detections": pred_dicts})
                for det in preds:
                    if best_det is None or float(det.confidence) > float(best_det.confidence):
                        best_det = det
                print(f"[Cycle {cycle}] camera={camera_id} image={frame.image_path} detections={pred_dicts}")
            except Exception as exc:
                detections_log.append({"cycle": cycle, "camera_id": camera_id, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[Cycle {cycle}] camera={camera_id} ERROR {type(exc).__name__}: {exc}")

        if best_det is not None:
            event["vision_best_label"] = best_det.label
            event["vision_best_confidence"] = round(float(best_det.confidence), 3)
            if is_waste_detection(best_det, target_labels, vision_min):
                event["vision_confirmed"] = True
                event["decision"] = "confirmed_waste"
                confirmed = True
                print(f"[Cycle {cycle}] DECISION: confirmed waste ✓ label={best_det.label} confidence={float(best_det.confidence):.2f}")
                if args.interactive:
                    cont = input("Confirmed. Press Enter to continue or type q to stop: ").strip().lower()
                    if cont == "q":
                        break
            elif thermal_candidate:
                event["decision"] = "thermal_seen_but_vision_not_confirmed"
                print(f"[Cycle {cycle}] DECISION: thermal candidate seen, vision not confirmed yet.")
        elif thermal_candidate:
            event["decision"] = "thermal_seen_no_vision_detection"
            print(f"[Cycle {cycle}] DECISION: thermal candidate seen, no vision detection yet.")

        events.append(event)

        if confirmed and not args.interactive:
            # In automatic mode, stop after first confirmation to keep logs focused.
            break

        if args.auto_continue_sec > 0 and not args.interactive:
            time.sleep(args.auto_continue_sec)

    if not confirmed and events:
        rejected = any(e["thermal_candidate"] for e in events) and not confirmed

    events_csv = logger.output_dir / "stationary_hil_events.csv"
    detections_json = logger.output_dir / "stationary_hil_detections.json"
    fields = [
        "cycle", "thermal_candidate", "thermal_bearing_deg", "thermal_confidence", "thermal_strength",
        "thermal_delta_c", "thermal_blob_pixel_count", "simulated_action", "vision_confirmed",
        "vision_best_label", "vision_best_confidence", "decision",
    ]
    with events_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    detections_json.write_text(json.dumps(detections_log, indent=2, default=str) + "\n")

    outcome = "confirmed_waste" if confirmed else ("candidate_not_confirmed" if rejected else "no_candidate")
    summary = {
        "mode": "stationary_real_sensor_hil",
        "outcome": outcome,
        "cycles_completed": len(events),
        "thermal_min_confidence": thermal_min,
        "vision_min_confidence": vision_min,
        "target_labels": sorted(target_labels),
        "motors": "log_only",
        "startup": report.as_dict(),
        "events_csv": str(events_csv),
        "detections_json": str(detections_json),
        "output_dir": str(logger.output_dir),
    }
    logger.finalize(summary)

    print(f"Stationary real-sensor HIL complete: {outcome}")
    print(f"Output directory: {logger.output_dir}")
    print(f"Events CSV: {events_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
