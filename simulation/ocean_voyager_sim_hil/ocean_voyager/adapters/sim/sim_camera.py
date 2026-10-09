from __future__ import annotations
from pathlib import Path
from typing import Any
from ocean_voyager.core.types import CameraFrame, CameraVisibleObject
from ocean_voyager.interfaces.camera import CameraArrayInterface
from ocean_voyager.simulation.visibility import visible_objects
from ocean_voyager.simulation.world import SimWorld
from ocean_voyager.simulation.renderer import render_camera_view
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class SimCameraArray(CameraArrayInterface):
    def __init__(self, world: SimWorld, cfg: dict[str, Any], output_dir: str | Path, bus: Any | None = None):
        self.world = world
        self.max_range_m = float(cfg.get("max_range_m", cfg.get("max_detection_range_m", 3.5)))
        self.fov_deg = float(cfg.get("fov_deg", 70.0))
        self.camera_mount_angles = cfg.get("mount_angles_deg", {"front": 0.0, "left": 75.0, "right": -75.0})
        self.output_dir = Path(output_dir)
        self.frame_counter = 0
        self.frame_width = int(cfg.get("frame_width", 640))
        self.frame_height = int(cfg.get("frame_height", 480))
        self.fps = float(cfg.get("fps", 10))
        self.latency_ms = float(cfg.get("latency_ms", 100))
        self.bus = bus or NullDeviceBus()

    def get_frame(self, camera_id: str) -> CameraFrame:
        mount = float(self.camera_mount_angles.get(camera_id, 0.0))
        visible = visible_objects(self.world.robot_pose, self.world.objects, self.max_range_m, self.fov_deg, mount)
        objs: list[CameraVisibleObject] = []
        for obj, d, bearing in visible:
            apparent = max(0.05, min(1.0, 1.0 / max(d, 0.1)))
            objs.append(CameraVisibleObject(obj.id, obj.type, d, bearing, 0.0, 0.0, apparent))
        self.frame_counter += 1
        image_path = self.output_dir / "frames" / f"{self.frame_counter:05d}_{camera_id}.png"
        render_camera_view(camera_id, objs, image_path)
        self.bus.transact(
            device="camera_array",
            command="capture",
            params={"camera_id": camera_id, "width": self.frame_width, "height": self.frame_height, "fov_deg": self.fov_deg, "mount_angle_deg": mount},
            response_payload={"image_path": str(image_path), "camera_id": camera_id, "visible_objects": [o.as_dict() for o in objs], "latency_ms_configured": self.latency_ms},
            time_sec=self.world.time_s,
        )
        return CameraFrame(camera_id=camera_id, image_path=str(image_path), visible_objects=objs)
