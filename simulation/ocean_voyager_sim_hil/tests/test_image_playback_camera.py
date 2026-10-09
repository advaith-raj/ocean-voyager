from pathlib import Path

from ocean_voyager.adapters.sim.image_playback_camera import ImagePlaybackCameraArray
from ocean_voyager.core.config_loader import load_scenario, parse_world
from ocean_voyager.simulation.world import SimWorld


def test_image_playback_camera_copies_front_frame(tmp_path):
    scenario = load_scenario("scenarios/16_hil_image_bottle_confirm.yaml")
    pool, pose, objects = parse_world(scenario)
    world = SimWorld(pool, pose, objects)
    cam = ImagePlaybackCameraArray(
        world,
        {"camera_ids": ["front", "left", "right"]},
        tmp_path,
        scenario["image_playback"],
    )
    frames = cam.get_all_frames()
    assert set(frames) == {"front", "left", "right"}
    assert Path(frames["front"].image_path).exists()
    assert frames["front"].wall_observation["expected"] == "target"
    assert frames["front"].wall_observation["source_image"].endswith("plastic_001.jpg")
