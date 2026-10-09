from __future__ import annotations

from pathlib import Path
from typing import Any

from ocean_voyager.interfaces.camera import CameraArrayInterface
from ocean_voyager.core.types import CameraFrame
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class RealCameraArray(CameraArrayInterface):
    """Raspberry Pi / USB camera adapter with safe bring-up behavior.

    Uses OpenCV VideoCapture if available. On Pi Camera modules, Picamera2 can be
    added later; this class is intentionally safe and reports readiness through
    health_check().
    """

    def __init__(self, cfg: dict[str, Any], output_dir: str | Path | None = None, bus: Any | None = None):
        self.cfg = cfg
        self.output_dir = Path(output_dir or "outputs")
        self.frames_dir = self.output_dir / "real_camera_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.bus = bus or NullDeviceBus()
        self.frame_counter = 0
        self.camera_ids = list(cfg.get("camera_ids", ["front", "left", "right"]))
        self.device_by_id = {
            "front": cfg.get("front_device", 0),
            "left": cfg.get("left_device", 1),
            "right": cfg.get("right_device", 2),
        }
        self.frame_width = int(cfg.get("frame_width", cfg.get("width", 640)))
        self.frame_height = int(cfg.get("frame_height", cfg.get("height", 480)))
        self.fps = float(cfg.get("fps", 10))
        self.require_all_cameras = bool(cfg.get("require_all_cameras", False))
        self.available = False
        self.cv2 = None
        self.last_error: str | None = None
        self.warnings: list[str] = []
        try:
            import cv2  # type: ignore
            self.cv2 = cv2
            self.available = True
        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.warnings.append("OpenCV is not available. Install opencv-python or configure a Picamera2 backend.")

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "opencv_videocapture",
            "camera_ids": self.camera_ids,
            "device_by_id": self.device_by_id,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "fps": self.fps,
            "require_all_cameras": self.require_all_cameras,
        }

    def _probe_one(self, camera_id: str) -> tuple[bool, str | None]:
        if not self.available or self.cv2 is None:
            return False, self.last_error
        dev = self.device_by_id.get(camera_id, 0)
        cap = self.cv2.VideoCapture(dev)
        try:
            if not cap.isOpened():
                return False, f"camera {camera_id} device {dev} did not open"
            cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            ok, frame = cap.read()
            if not ok or frame is None:
                return False, f"camera {camera_id} device {dev} opened but did not return a frame"
            return True, None
        finally:
            cap.release()

    def health_check(self) -> dict[str, Any]:
        if not self.available:
            return {
                "status": "failed",
                "self_test": "fail",
                "capabilities": self.capabilities(),
                "last_error": self.last_error,
                "warnings": self.warnings,
            }
        results = {}
        errors = []
        for camera_id in self.camera_ids:
            ok, err = self._probe_one(camera_id)
            results[camera_id] = ok
            if err:
                errors.append(err)
        any_ok = any(results.values())
        all_ok = all(results.values()) if results else False
        if all_ok or (any_ok and not self.require_all_cameras):
            status = "ready" if all_ok else "degraded"
            self_test = "pass" if all_ok else "degraded"
            warnings = self.warnings + ([] if all_ok else errors)
            last_error = None if all_ok else "; ".join(errors)
        else:
            status = "failed"
            self_test = "fail"
            warnings = self.warnings
            last_error = "; ".join(errors) if errors else "no camera frames available"
        return {
            "status": status,
            "self_test": self_test,
            "capabilities": self.capabilities() | {"probe_results": results},
            "last_error": last_error,
            "warnings": warnings,
        }

    def get_frame(self, camera_id: str) -> CameraFrame:
        if not self.available or self.cv2 is None:
            raise RuntimeError(f"RealCameraArray is not ready: {self.last_error}")
        dev = self.device_by_id.get(camera_id, 0)
        cap = self.cv2.VideoCapture(dev)
        try:
            if not cap.isOpened():
                raise RuntimeError(f"camera {camera_id} device {dev} did not open")
            cap.set(self.cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(self.cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError(f"camera {camera_id} device {dev} did not return a frame")
            self.frame_counter += 1
            image_path = self.frames_dir / f"{self.frame_counter:06d}_{camera_id}.jpg"
            self.cv2.imwrite(str(image_path), frame)
        finally:
            cap.release()
        self.bus.transact(
            device="camera_array",
            command="capture",
            params={"camera_id": camera_id, "device": dev},
            response_payload={"image_path": str(image_path), "camera_id": camera_id},
            time_sec=None,
        )
        return CameraFrame(camera_id=camera_id, image_path=str(image_path), visible_objects=[])
