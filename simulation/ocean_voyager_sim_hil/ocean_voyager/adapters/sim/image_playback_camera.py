from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil

from PIL import Image, ImageDraw

from ocean_voyager.core.types import CameraFrame
from ocean_voyager.interfaces.camera import CameraArrayInterface
from ocean_voyager.simulation.world import SimWorld
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class ImagePlaybackCameraArray(CameraArrayInterface):
    """Camera adapter for stationary HIL/image-playback tests."""

    def __init__(self, world: SimWorld, cfg: dict[str, Any], output_dir: str | Path, playback_cfg: dict[str, Any] | None = None, bus: Any | None = None):
        self.world = world
        self.cfg = cfg or {}
        self.output_dir = Path(output_dir)
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.playback_cfg = playback_cfg or {}
        self.sequence = list(self.playback_cfg.get("frames", []))
        if not self.sequence:
            raise ValueError("ImagePlaybackCameraArray requires image_playback.frames in config or scenario.")
        self.repeat_last = bool(self.playback_cfg.get("repeat_last", True))
        self.step_index = 0
        self.frame_counter = 0
        self.cameras = tuple(self.cfg.get("camera_ids", ["front", "left", "right"]))
        self.blank_color = tuple(self.cfg.get("blank_color_rgb", [180, 210, 230]))
        self.blank_size = tuple(self.cfg.get("blank_size", [640, 480]))
        self.bus = bus or NullDeviceBus()

    def _current_spec(self) -> dict[str, Any]:
        if self.step_index < len(self.sequence):
            return dict(self.sequence[self.step_index])
        if self.repeat_last:
            return dict(self.sequence[-1])
        return {"camera": "front", "image_path": None, "expected": "none", "instruction": "No more playback frames."}

    def _copy_or_blank(self, src: str | None, out_path: Path, camera_id: str, spec: dict[str, Any]) -> str:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if src:
            src_path = Path(src)
            if not src_path.is_absolute():
                src_path = Path.cwd() / src_path
            if src_path.exists():
                shutil.copy2(src_path, out_path)
                return str(out_path)
        w, h = int(self.blank_size[0]), int(self.blank_size[1])
        img = Image.new("RGB", (w, h), self.blank_color)
        d = ImageDraw.Draw(img)
        d.text((20, 20), f"{camera_id} blank", fill=(30, 30, 30))
        d.text((20, 48), str(spec.get("instruction", ""))[:80], fill=(30, 30, 30))
        img.save(out_path)
        return str(out_path)

    def get_frame(self, camera_id: str) -> CameraFrame:
        spec = self._current_spec()
        active_camera = str(spec.get("camera", "front"))
        self.frame_counter += 1
        suffix = Path(str(spec.get("image_path", ""))).suffix.lower() or ".png"
        if camera_id == active_camera and spec.get("image_path"):
            out_path = self.frames_dir / f"{self.frame_counter:05d}_{camera_id}_playback{suffix}"
            image_path = self._copy_or_blank(str(spec.get("image_path")), out_path, camera_id, spec)
        else:
            out_path = self.frames_dir / f"{self.frame_counter:05d}_{camera_id}_blank.png"
            image_path = self._copy_or_blank(None, out_path, camera_id, spec)
        meta = {
            "playback_step": self.step_index,
            "active_camera": active_camera,
            "expected": spec.get("expected", "unknown"),
            "instruction": spec.get("instruction", ""),
            "source_image": spec.get("image_path"),
            "category": spec.get("category", ""),
        }
        self.bus.transact(
            device="camera_array",
            command="capture_playback",
            params={"camera_id": camera_id, "playback_step": self.step_index},
            response_payload={"image_path": image_path, "metadata": meta},
            time_sec=self.world.time_s,
        )
        return CameraFrame(camera_id=camera_id, image_path=image_path, visible_objects=[], wall_observation=meta)

    def get_all_frames(self) -> dict[str, CameraFrame]:
        frames = {cam: self.get_frame(cam) for cam in self.cameras}
        self.step_index += 1
        return frames
