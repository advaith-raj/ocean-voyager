from ocean_voyager.core.config_loader import load_config
from ocean_voyager.core.factory import build_runtime_bundle
from ocean_voyager.core.config_loader import load_scenario, parse_world
from ocean_voyager.simulation.world import SimWorld


def test_ai_camera_config_builds_rpicam_adapter(tmp_path):
    cfg = load_config("config/camera_only_ai_camera_bringup.yaml")
    scenario = load_scenario("scenarios/02_single_bottle_center.yaml")
    cfg["_scenario"] = scenario
    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool=pool, robot_pose=pose, objects=objects)
    runtime = build_runtime_bundle(world, cfg, tmp_path, command_log=[])
    assert runtime.camera.__class__.__name__ == "RpicamCameraArray"
    caps = runtime.camera.capabilities()
    assert caps["backend"] == "rpicam_still"
    assert caps["camera_ids"] == ["front"]
