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
import subprocess
from datetime import datetime


QUICK_SCENARIOS = [
    "scenarios/02_single_bottle_center.yaml",
    "scenarios/06_bottle_plus_seaweed.yaml",
    "scenarios/28_delayed_bottle_detection.yaml",
    "scenarios/29_delayed_bottle_and_seaweed.yaml",
    "scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml",
    "scenarios/37_time_boxed_four_bottles_plus_seaweed_patrol.yaml",
]


def discover_scenarios(preset: str, scenario_glob: str) -> list[Path]:
    if preset == "quick":
        return [ROOT / p for p in QUICK_SCENARIOS if (ROOT / p).exists()]
    if preset == "all":
        return sorted(ROOT.glob(scenario_glob))
    raise ValueError(f"unknown preset: {preset}")


def load_summary(run_dir: Path) -> dict:
    p = run_dir / "summary.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Ocean Voyager scenarios on the current machine and summarize results.")
    parser.add_argument("--config", default="config/sim_full.yaml", help="Config to use for each scenario.")
    parser.add_argument("--preset", choices=["quick", "all"], default="quick", help="Scenario set to run.")
    parser.add_argument("--scenario-glob", default="scenarios/*.yaml", help="Used only with --preset all.")
    parser.add_argument("--steps", type=int, default=None, help="Optional step override for each run.")
    parser.add_argument("--output-dir", default="outputs/pi_mocked_scenario_sweep", help="Directory for sweep summary files.")
    parser.add_argument("--continue-on-failure", action="store_true", help="Continue running even when one scenario fails.")
    args = parser.parse_args()

    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    scenarios = discover_scenarios(args.preset, args.scenario_glob)
    rows: list[dict] = []
    started = datetime.now().isoformat(timespec="seconds")

    for idx, scenario_path in enumerate(scenarios, start=1):
        run_name = f"{scenario_path.stem}"
        cmd = [
            sys.executable,
            str(ROOT / "main.py"),
            "--config",
            str(ROOT / args.config),
            "--scenario",
            str(scenario_path),
            "--run-name",
            run_name,
        ]
        if args.steps is not None:
            cmd += ["--steps", str(args.steps)]

        print(f"[{idx}/{len(scenarios)}] Running {scenario_path.name}")
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        run_dir = ROOT / "outputs" / run_name
        # If config has a different base output dir, infer from stdout if possible.
        for line in proc.stdout.splitlines():
            if line.startswith("Output directory:"):
                run_dir = ROOT / line.split(":", 1)[1].strip()
        summary = load_summary(run_dir)

        row = {
            "scenario": scenario_path.name,
            "returncode": proc.returncode,
            "run_dir": str(run_dir.relative_to(ROOT) if run_dir.exists() else run_dir),
            "outcome": summary.get("outcome", "process_failed" if proc.returncode else "unknown"),
            "steps": summary.get("steps", ""),
            "objects_confirmed": summary.get("objects_confirmed", ""),
            "objects_rejected": summary.get("objects_rejected", ""),
            "returned_home": summary.get("returned_home", ""),
            "startup_status": summary.get("startup", {}).get("overall_status", ""),
            "mission_time_s": summary.get("mission_time_s", ""),
            "path_length_m": summary.get("path_length_m", ""),
            "stdout_tail": proc.stdout[-500:],
            "stderr_tail": proc.stderr[-500:],
        }
        rows.append(row)

        if proc.returncode != 0 and not args.continue_on_failure:
            print(f"FAILED: {scenario_path.name}; stopping. Use --continue-on-failure to continue.")
            break

    csv_path = out / "scenario_sweep_summary.csv"
    json_path = out / "scenario_sweep_summary.json"
    fields = [
        "scenario",
        "returncode",
        "run_dir",
        "outcome",
        "steps",
        "objects_confirmed",
        "objects_rejected",
        "returned_home",
        "startup_status",
        "mission_time_s",
        "path_length_m",
        "stdout_tail",
        "stderr_tail",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps({"started_at": started, "rows": rows}, indent=2, default=str) + "\n")

    print(f"Scenario sweep complete: {len(rows)} runs")
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")
    failures = [r for r in rows if r["returncode"] != 0]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
