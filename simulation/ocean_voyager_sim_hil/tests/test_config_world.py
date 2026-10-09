from pathlib import Path
import pytest
from ocean_voyager.core.config_loader import load_config, load_scenario, parse_world

ROOT = Path(__file__).resolve().parents[1]


def test_load_all_scenarios():
    for path in (ROOT / "scenarios").glob("*.yaml"):
        scenario = load_scenario(path)
        pool, pose, objects = parse_world(scenario)
        assert pool.length_m > 0
        assert pool.width_m > 0
        assert pool.contains(pose.x, pose.y, margin=0.0)
        for obj in objects:
            assert pool.contains(obj.x, obj.y, margin=0.0)


def test_config_loads():
    cfg = load_config(ROOT / "config" / "sim_full.yaml")
    assert cfg["runtime"]["mode"] == "sim_full"
    assert cfg["sources"]["thermal"] == "sim"
