#!/usr/bin/env python3
from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path

SCENARIOS = [
    "20_pool_empty_return_home.yaml",
    "21_pool_single_bottle_return_home.yaml",
    "22_pool_bottle_and_seaweed_return_home.yaml",
    "23_pool_bottle_near_wall_return_home.yaml",
    "24_pool_multiple_objects_return_home.yaml",
    "25_pool_start_near_corner_return_home.yaml",
    "26_pool_low_thermal_delta_return_home.yaml",
    "27_pool_false_positive_only_return_home.yaml",
    "28_delayed_bottle_detection.yaml",
    "29_delayed_bottle_and_seaweed.yaml",
    "30_object_outside_thermal_range.yaml",
    "31_object_enters_range_after_coverage_move.yaml",
    "32_low_delta_becomes_detectable_when_close.yaml",
    "33_three_bottles_edges_full_survey.yaml",
    "34_delayed_bottle_full_physical_coverage.yaml",
    "35_middle_and_edges_full_survey.yaml",
    "36_time_boxed_three_bottles_full_pool_patrol.yaml",
    "37_time_boxed_four_bottles_plus_seaweed_patrol.yaml",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Build 0.5.4 coordinate-pool replay suite.")
    parser.add_argument("--config", default="config/sim_hardware_faithful_pool.yaml")
    parser.add_argument("--only", nargs="*", help="Optional scenario filenames or numeric prefixes, e.g. 21 22")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    scenarios = SCENARIOS
    if args.only:
        keep = set(args.only)
        scenarios = [s for s in scenarios if s in keep or s.split("_")[0] in keep]
    for s in scenarios:
        scenario_path = root / "scenarios" / s
        run_name = s.replace(".yaml", "")
        cmd = [sys.executable, "main.py", "--config", args.config, "--scenario", str(scenario_path), "--run-name", run_name]
        print("\n===", run_name, "===")
        subprocess.run(cmd, cwd=root, check=True)
    print("\nBuild 0.5.4 suite complete. Outputs are under outputs/build_0_5_pool_runs/.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
