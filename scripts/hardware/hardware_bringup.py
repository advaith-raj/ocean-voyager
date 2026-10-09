#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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
    parser = argparse.ArgumentParser(description="Ocean Voyager Raspberry Pi peripheral bring-up checker")
    parser.add_argument("--config", default="config/pi_hardware_sensors_log_only.yaml", help="Hardware bring-up config")
    parser.add_argument("--scenario", default="scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml", help="Scenario used only for pool/world context")
    parser.add_argument("--run-name", default="hardware_bringup", help="Output run folder name")
    parser.add_argument("--probe", action="store_true", help="After startup, capture one thermal scan and one frame per camera if devices are ready")
    parser.add_argument("--allow-real-motors", action="store_true", help="Allow config to keep real motor source. Default forces motors to log_only.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scenario = load_scenario(args.scenario)
    cfg["_scenario"] = scenario
    cfg["scenario_name"] = scenario.get("scenario_name", Path(args.scenario).stem)
    if "startup" in scenario:
        cfg.setdefault("startup", {})
        deep_update(cfg["startup"], scenario["startup"])

    # Safety default: sensors can be real, motors stay log-only unless explicitly requested.
    if not args.allow_real_motors:
        cfg.setdefault("sources", {})["motors"] = "log_only"
        cfg.setdefault("motors", {})["arm_real_motors"] = False

    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool=pool, robot_pose=pose, objects=objects)
    logger = MissionLogger(cfg.get("logging", {}).get("base_output_dir", "outputs"), args.run_name)
    logger.save_static_inputs(args.config, args.scenario)

    runtime = build_runtime_bundle(world, cfg, logger.output_dir, command_log=[])
    report = run_startup_sequence(cfg, logger.output_dir, runtime.as_startup_dict(), bus=runtime.device_bus)

    probe_results: dict[str, Any] = {}
    if args.probe and not report.abort_mission:
        try:
            readings = runtime.thermal.scan()
            probe_results["thermal_scan"] = [r.as_dict() for r in readings]
        except Exception as exc:
            probe_results["thermal_scan_error"] = f"{type(exc).__name__}: {exc}"

        camera_results = {}
        for camera_id in cfg.get("sensors", {}).get("camera", {}).get("camera_ids", ["front", "left", "right"]):
            try:
                frame = runtime.camera.get_frame(camera_id)
                camera_results[camera_id] = frame.as_dict()
            except Exception as exc:
                camera_results[camera_id] = {"error": f"{type(exc).__name__}: {exc}"}
        probe_results["camera_frames"] = camera_results

        try:
            runtime.motors.stop()
            probe_results["motors_stop"] = "sent/logged"
        except Exception as exc:
            probe_results["motors_stop_error"] = f"{type(exc).__name__}: {exc}"

    summary = {
        "mode": "hardware_bringup",
        "outcome": "ready" if not report.abort_mission else "not_ready",
        "startup": report.as_dict(),
        "probe_results": probe_results,
        "output_dir": str(logger.output_dir),
    }
    logger.finalize(summary)

    print(f"Bring-up outcome: {summary['outcome']}")
    print(f"Startup status: {report.overall_status}")
    print(f"Output directory: {logger.output_dir}")
    print(f"Startup report: {logger.output_dir / 'startup' / 'startup_status.json'}")
    print(f"Peripheral status: {logger.output_dir / 'startup' / 'peripheral_status.json'}")
    if report.abort_mission:
        print("One or more required devices were not ready. Check startup_log.txt.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
