from ocean_voyager.core.config_loader import load_scenario, parse_world
from ocean_voyager.core.types import distance
from ocean_voyager.device_bus.file_bus import FileDeviceBus
from pathlib import Path


def test_coordinate_pool_origin_and_bounds():
    scenario = load_scenario("scenarios/21_pool_single_bottle_return_home.yaml")
    pool, pose, objects = parse_world(scenario)
    assert pose.x == 0.0 and pose.y == 0.0
    assert pool.x_min < 0 < pool.x_max
    assert pool.y_min < 0 < pool.y_max
    assert pool.contains(0, 0, margin=0.0)
    assert objects[0].x > 0


def test_file_device_bus_writes_trace(tmp_path: Path):
    bus = FileDeviceBus(tmp_path / "device_bus")
    bus.transact("thermal", "capture_frame", {"pan_angle_deg": 0}, {"ok": True}, time_sec=1.25)
    assert (tmp_path / "device_bus" / "traces" / "bus_trace.jsonl").exists()
    assert len(list((tmp_path / "device_bus" / "commands" / "processed").glob("*.json"))) == 1
    assert len(list((tmp_path / "device_bus" / "responses" / "processed").glob("*.json"))) == 1
