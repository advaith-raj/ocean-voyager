from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any

from ocean_voyager.core.config_loader import load_config, load_scenario, parse_world
from ocean_voyager.core.factory import build_runtime_bundle
from ocean_voyager.core.logger import MissionLogger
from ocean_voyager.core.mission_controller import MissionController
from ocean_voyager.core.startup import run_startup_sequence
from ocean_voyager.core.stationary_hil import StationaryHILRunner
from ocean_voyager.simulation.replay import build_replay_outputs


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def apply_scenario_overrides(cfg: dict[str, Any], scenario: dict[str, Any]) -> dict[str, Any]:
    """Allow test scenarios to tune mission timing, thermal range, etc."""
    for key in ["mission", "sensors", "wall_guard", "motors", "robot_dynamics", "vision_model", "coverage_grid", "startup"]:
        if key in scenario:
            cfg.setdefault(key, {})
            deep_update(cfg[key], scenario[key])
    return cfg


def main() -> int:
    parser = argparse.ArgumentParser(description="Ocean Voyager simulation/HIL runner")
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--scenario", required=True, help="Path to scenario YAML")
    parser.add_argument("--run-name", default=None, help="Optional output run name")
    parser.add_argument("--steps", type=int, default=None, help="Override max steps or HIL steps")
    parser.add_argument("--startup-only", action="store_true", help="Run peripheral init/health checks and stop before mission")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scenario = load_scenario(args.scenario)
    cfg = apply_scenario_overrides(cfg, scenario)
    cfg["scenario_name"] = scenario.get("scenario_name", Path(args.scenario).stem)
    cfg["_scenario"] = scenario
    pool, pose, objects = parse_world(scenario)
    from ocean_voyager.simulation.world import SimWorld
    world = SimWorld(pool=pool, robot_pose=pose, objects=objects)

    output_base = cfg.get("logging", {}).get("base_output_dir", "outputs")
    logger = MissionLogger(output_base, args.run_name)
    logger.save_static_inputs(args.config, args.scenario)
    command_log: list[dict] = []

    runtime = build_runtime_bundle(world, cfg, logger.output_dir, command_log)
    thermal, camera, motors, pan_servo, wall_guard, vision_model = runtime.as_tuple()

    startup_cfg = cfg.get("startup", {})
    startup_enabled = bool(startup_cfg.get("enabled", True))
    startup_report = None
    if startup_enabled:
        startup_report = run_startup_sequence(
            cfg,
            logger.output_dir,
            adapters=runtime.as_startup_dict(),
            bus=runtime.device_bus,
        )
        print(f"Startup status: {startup_report.overall_status}")
        print(f"Startup files: {logger.output_dir / 'startup'}")
        if startup_report.abort_mission or args.startup_only:
            summary = {
                "scenario_name": cfg.get("scenario_name", "scenario"),
                "outcome": "startup_only" if args.startup_only and not startup_report.abort_mission else "startup_failed",
                "startup": startup_report.as_dict(),
                "output_dir": str(logger.output_dir),
            }
            logger.finalize(summary)
            print(f"Ocean Voyager startup complete: {summary['outcome']}")
            print(f"Output directory: {logger.output_dir}")
            return 0 if args.startup_only and not startup_report.abort_mission else 2

    runtime_mode = cfg.get("runtime", {}).get("mode", "sim_full")
    controller: MissionController | None = None
    if runtime_mode == "stationary_hil":
        runner = StationaryHILRunner(world, thermal, camera, vision_model, logger, cfg)
        summary = runner.run(steps=args.steps or int(cfg.get("mission", {}).get("hil_steps", 5)))
    else:
        if args.steps is not None:
            cfg.setdefault("mission", {})["max_steps"] = args.steps
        controller = MissionController(world, thermal, camera, motors, wall_guard, vision_model, logger, cfg)
        summary_obj = controller.run()
        summary = summary_obj.as_dict()
        if controller.coverage_grid:
            coverage_artifacts = controller.coverage_grid.write_outputs(logger.output_dir)
            cg_summary = controller.coverage_grid.summary()
            cg_summary["lawnmower_pass_equivalent_coverage_percent"] = round((controller.coverage_pass_count * 100.0) + float(cg_summary.get("path_covered_percent", 0.0)), 2)
            cg_summary["coverage_pass_count"] = controller.coverage_pass_count
            summary["coverage_grid"] = cg_summary
            summary["coverage_grid_outputs"] = coverage_artifacts
            summary["coverage_waypoints_skipped_as_covered"] = controller.coverage.skipped_covered_waypoints
            summary["mission_objective"] = controller.objective
            summary["coverage_pass_count"] = controller.coverage_pass_count
            summary["time_boxed_patrol"] = controller.keep_patrolling_until_time_expires

    logger.write_object_coordinates(world.objects)
    replay_paths = build_replay_outputs(world, logger.output_dir, title=f"Ocean Voyager: {cfg.get('scenario_name', 'scenario')}")
    summary["replay_outputs"] = replay_paths
    summary["object_coordinate_log"] = str(logger.object_coordinate_path)
    summary["ground_truth_objects_csv"] = str(logger.ground_truth_object_path)
    summary["object_coordinates_json"] = str(logger.object_coordinate_json_path)
    summary["thermal_scans_json"] = str(logger.thermal_scan_path)
    summary["device_bus_trace"] = str(logger.output_dir / "device_bus" / "traces" / "bus_trace.jsonl")
    summary["startup_status"] = str(logger.output_dir / "startup" / "startup_status.json")
    if startup_report is not None:
        summary["startup"] = startup_report.as_dict()
    logger.finalize(summary)

    print(f"Ocean Voyager run complete: {summary.get('outcome', summary.get('mode'))}")
    print(f"Output directory: {logger.output_dir}")
    if replay_paths.get("mission_animation"):
        print(f"Mission animation: {replay_paths['mission_animation']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
