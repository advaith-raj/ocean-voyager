from pathlib import Path
import json

from ocean_voyager.core.config_loader import load_config, load_scenario, parse_world
from ocean_voyager.core.factory import build_runtime_bundle
from ocean_voyager.core.startup import run_startup_sequence
from ocean_voyager.simulation.world import SimWorld


ROOT = Path(__file__).resolve().parents[1]


def test_startup_sequence_writes_status_files(tmp_path):
    cfg = load_config(ROOT / "config" / "sim_full.yaml")
    scenario = load_scenario(ROOT / "scenarios" / "02_single_bottle_center.yaml")
    cfg["_scenario"] = scenario
    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool, pose, objects)
    runtime = build_runtime_bundle(world, cfg, tmp_path, command_log=[])
    report = run_startup_sequence(cfg, tmp_path, runtime.as_startup_dict(), bus=runtime.device_bus)

    assert report.overall_status in {"ready", "degraded"}
    assert not report.abort_mission
    assert (tmp_path / "startup" / "startup_log.txt").exists()
    assert (tmp_path / "startup" / "startup_status.json").exists()
    assert (tmp_path / "startup" / "peripheral_status.json").exists()

    data = json.loads((tmp_path / "startup" / "startup_status.json").read_text())
    assert data["all_required_ready"] is True
    names = {d["device"] for d in data["devices"]}
    assert {"thermal", "camera_array", "motors", "vision_model", "safety"}.issubset(names)
