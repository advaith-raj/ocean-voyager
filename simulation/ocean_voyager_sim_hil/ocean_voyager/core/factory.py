from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ocean_voyager.adapters.sim.sim_camera import SimCameraArray
from ocean_voyager.adapters.sim.image_playback_camera import ImagePlaybackCameraArray
from ocean_voyager.adapters.sim.sim_motors import LogOnlyMotorExecutor, SimMotorExecutor
from ocean_voyager.adapters.sim.sim_pan_servo import SimPanServo
from ocean_voyager.adapters.sim.sim_thermal import SimThermalSensor
from ocean_voyager.adapters.sim.sim_wall_guard import SimWallGuard
from ocean_voyager.device_bus.file_bus import FileDeviceBus, NullDeviceBus
from ocean_voyager.perception.vision_model import MockVisionModel, YoloVisionModel
from ocean_voyager.simulation.robot_dynamics import DifferentialThrustDynamics
from ocean_voyager.simulation.world import SimWorld


@dataclass
class RuntimeAdapters:
    thermal: Any
    camera: Any
    motors: Any
    pan_servo: Any
    wall_guard: Any
    vision_model: Any
    device_bus: Any

    def as_startup_dict(self) -> dict[str, Any]:
        return {
            "thermal": self.thermal,
            "camera_array": self.camera,
            "motors": self.motors,
            "pan_servo": self.pan_servo,
            "wall_guard": self.wall_guard,
            "vision_model": self.vision_model,
        }

    def as_tuple(self):
        return self.thermal, self.camera, self.motors, self.pan_servo, self.wall_guard, self.vision_model


def build_device_bus(cfg: dict[str, Any], output_dir: str | Path):
    bus_cfg = cfg.get("device_bus", {})
    if not bool(bus_cfg.get("enabled", False)):
        return NullDeviceBus()
    root = Path(output_dir) / bus_cfg.get("dir_name", "device_bus")
    run_mode = str(bus_cfg.get("run_mode", cfg.get("runtime", {}).get("mode", "sim")))
    return FileDeviceBus(root, run_mode=run_mode)


def build_runtime_bundle(world: SimWorld, cfg: dict[str, Any], output_dir: str | Path, command_log: list[dict] | None = None) -> RuntimeAdapters:
    sensors = cfg.get("sensors", {})
    sources = cfg.get("sources", {})
    bus = build_device_bus(cfg, output_dir)

    pan_servo = SimPanServo(bus=bus)

    thermal_source = sources.get("thermal", "sim")
    if str(thermal_source).startswith("real"):
        from ocean_voyager.adapters.hardware.real_thermal_mlx90640 import RealMLX90640Thermal
        thermal = RealMLX90640Thermal(sensors.get("thermal", {}), output_dir=output_dir, bus=bus, pan_servo=pan_servo)
    else:
        thermal = SimThermalSensor(world, pan_servo, sensors.get("thermal", {}), output_dir=output_dir, bus=bus)

    camera_source = sources.get("camera", "sim")
    playback_cfg = cfg.get("_scenario", {}).get("image_playback") or cfg.get("image_playback")
    if camera_source == "image_playback":
        camera = ImagePlaybackCameraArray(world, sensors.get("camera", {}), output_dir, playback_cfg=playback_cfg, bus=bus)
    elif str(camera_source) in {"rpicam", "rpicam_still", "ai_camera", "rpicam_ai_camera", "raspberry_pi_ai_camera"}:
        from ocean_voyager.adapters.hardware.rpicam_camera_array import RpicamCameraArray
        camera = RpicamCameraArray(sensors.get("camera", {}), output_dir=output_dir, bus=bus)
    elif str(camera_source).startswith("real"):
        from ocean_voyager.adapters.hardware.real_camera_array import RealCameraArray
        camera = RealCameraArray(sensors.get("camera", {}), output_dir=output_dir, bus=bus)
    else:
        camera = SimCameraArray(world, sensors.get("camera", {}), output_dir, bus=bus)

    motor_source = sources.get("motors", "sim")
    if motor_source == "real":
        from ocean_voyager.adapters.hardware.real_motor_pca9685 import RealPCA9685MotorExecutor
        motors = RealPCA9685MotorExecutor(cfg.get("motors", {}), bus=bus)
    elif motor_source in {"log_only", "disabled"}:
        motors = LogOnlyMotorExecutor(command_log=command_log, bus=bus)
    else:
        dynamics_cfg = dict(cfg.get("robot_dynamics", {}))
        # Allow motor/hardware profile values to flow into the motion model.
        dynamics_cfg.update(cfg.get("motors", {}))
        dynamics = DifferentialThrustDynamics(world, **dynamics_cfg)
        motors = SimMotorExecutor(dynamics, command_log=command_log, bus=bus)

    wall_guard = SimWallGuard(world, cfg.get("wall_guard", {}))

    model_cfg = cfg.get("vision_model", {})
    model_type = model_cfg.get("type", "mock")
    if model_type == "yolo":
        vision_model = YoloVisionModel(model_cfg["model_path"], float(model_cfg.get("confidence_threshold", 0.25)), model_cfg.get("target_labels"))
    else:
        vision_model = MockVisionModel(model_cfg)

    return RuntimeAdapters(
        thermal=thermal,
        camera=camera,
        motors=motors,
        pan_servo=pan_servo,
        wall_guard=wall_guard,
        vision_model=vision_model,
        device_bus=bus,
    )


def build_runtime_adapters(world: SimWorld, cfg: dict[str, Any], output_dir: str | Path, command_log: list[dict] | None = None):
    """Backward-compatible tuple return used by older tests/scripts."""
    return build_runtime_bundle(world, cfg, output_dir, command_log).as_tuple()
