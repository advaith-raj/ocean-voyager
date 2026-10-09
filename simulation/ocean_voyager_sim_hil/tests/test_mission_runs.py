from pathlib import Path
import subprocess
import sys
import json

ROOT = Path(__file__).resolve().parents[1]


def test_single_bottle_mission_runs(tmp_path):
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--config", str(ROOT / "config" / "sim_full.yaml"),
        "--scenario", str(ROOT / "scenarios" / "02_single_bottle_center.yaml"),
        "--run-name", "pytest_single_bottle",
        "--steps", "40",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    summary = ROOT / "outputs" / "pytest_single_bottle" / "summary.json"
    assert summary.exists()
    data = json.loads(summary.read_text())
    assert data["steps"] > 0
    assert (ROOT / "outputs" / "pytest_single_bottle" / "world_replay.png").exists()


def test_stationary_hil_simulated_runs():
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--config", str(ROOT / "config" / "stationary_hil.yaml"),
        "--scenario", str(ROOT / "scenarios" / "06_bottle_plus_seaweed.yaml"),
        "--run-name", "pytest_stationary_hil",
        "--steps", "2",
    ]
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    actions = ROOT / "outputs" / "pytest_stationary_hil" / "recommended_actions.json"
    assert actions.exists()
    data = json.loads(actions.read_text())
    assert len(data) == 2
    assert data[0]["motor_execution"] == "disabled_log_only"
