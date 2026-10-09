from pathlib import Path
from ocean_voyager.core.config_loader import load_config, load_scenario, parse_world
from ocean_voyager.simulation.world import SimWorld
from ocean_voyager.adapters.sim.sim_pan_servo import SimPanServo
from ocean_voyager.adapters.sim.sim_thermal import SimThermalSensor
from ocean_voyager.adapters.sim.sim_camera import SimCameraArray

ROOT = Path(__file__).resolve().parents[1]


def make_world():
    scenario = load_scenario(ROOT / "scenarios" / "02_single_bottle_center.yaml")
    return SimWorld(*parse_world(scenario))


def test_thermal_detects_bottle():
    cfg = load_config(ROOT / "config" / "sim_full.yaml")
    world = make_world()
    thermal = SimThermalSensor(world, SimPanServo(), cfg["sensors"]["thermal"])
    readings = thermal.scan()
    # Build 0.5.3 is hardware-faithful: thermal detects a blob/bearing but does not
    # reveal object identity or range to the autonomy code.
    assert any(r.candidate and not r.range_known and r.distance_m < 0 for r in readings)
    assert all(r.object_id is None for r in readings)


def test_camera_sees_bottle_when_facing():
    cfg = load_config(ROOT / "config" / "sim_full.yaml")
    world = make_world()
    camera = SimCameraArray(world, cfg["sensors"]["camera"], ROOT / "outputs" / "test_frames")
    frame = camera.get_frame("front")
    assert any(obj.object_id == "bottle_01" for obj in frame.visible_objects)
    assert frame.image_path is not None
