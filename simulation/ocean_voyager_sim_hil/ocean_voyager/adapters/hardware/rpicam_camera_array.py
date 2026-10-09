from __future__ import annotations

from pathlib import Path
from typing import Any
import shutil
import subprocess

from ocean_voyager.interfaces.camera import CameraArrayInterface
from ocean_voyager.core.types import CameraFrame
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class RpicamCameraArray(CameraArrayInterface):
    """Raspberry Pi libcamera/rpicam-still camera adapter.

    This adapter is intended for CSI cameras that work with Raspberry Pi's
    rpicam/libcamera stack, including the Raspberry Pi AI Camera / IMX500.
    It avoids OpenCV VideoCapture and captures still frames with rpicam-still,
    which is the path validated during Raspberry Pi hardware bring-up.

    Notes:
    - One physical CSI camera is supported in this first-pass adapter.
    - The logical camera_id is normally "front".
    - Motors remain controlled by the configured motor adapter; this class only
      captures images.
    """

    def __init__(self, cfg: dict[str, Any], output_dir: str | Path | None = None, bus: Any | None = None):
        self.cfg = cfg
        self.output_dir = Path(output_dir or "outputs")
        self.frames_dir = self.output_dir / "rpicam_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.bus = bus or NullDeviceBus()
        self.frame_counter = 0

        self.camera_ids = list(cfg.get("camera_ids", ["front"]))
        self.require_all_cameras = bool(cfg.get("require_all_cameras", True))
        self.command = str(cfg.get("rpicam_command", cfg.get("command", "rpicam-still")))
        self.camera_index = int(cfg.get("camera_index", cfg.get("rpicam_camera_index", 0)))
        self.timeout_ms = int(cfg.get("timeout_ms", cfg.get("capture_timeout_ms", 1000)))
        self.process_timeout_sec = float(cfg.get("process_timeout_sec", max(8.0, self.timeout_ms / 1000.0 + 6.0)))
        self.width = cfg.get("frame_width", cfg.get("width"))
        self.height = cfg.get("frame_height", cfg.get("height"))
        self.use_width_height_args = bool(cfg.get("use_width_height_args", True))
        self.nopreview = bool(cfg.get("nopreview", True))
        self.health_check_capture = bool(cfg.get("health_check_capture", True))
        self.list_cameras_first = bool(cfg.get("list_cameras_first", True))
        self.extra_args = [str(x) for x in cfg.get("extra_args", [])]
        self.last_error: str | None = None
        self.warnings: list[str] = []

    def capabilities(self) -> dict[str, Any]:
        return {
            "backend": "rpicam_still",
            "camera_ids": self.camera_ids,
            "camera_index": self.camera_index,
            "command": self.command,
            "timeout_ms": self.timeout_ms,
            "process_timeout_sec": self.process_timeout_sec,
            "frame_width": self.width,
            "frame_height": self.height,
            "require_all_cameras": self.require_all_cameras,
            "health_check_capture": self.health_check_capture,
        }

    def _command_exists(self) -> bool:
        return shutil.which(self.command) is not None

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.process_timeout_sec,
            check=False,
        )

    def _list_cameras(self) -> tuple[bool, str]:
        if not self._command_exists():
            return False, f"{self.command!r} not found on PATH"
        try:
            proc = self._run([self.command, "--list-cameras"])
        except subprocess.TimeoutExpired:
            return False, f"{self.command} --list-cameras timed out"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        text = (proc.stdout or "") + "\n" + (proc.stderr or "")
        ok = proc.returncode == 0 and "Available cameras" in text and "no cameras available" not in text.lower()
        if not ok:
            return False, text.strip() or f"{self.command} --list-cameras returned {proc.returncode}"
        return True, text.strip()

    def _capture_to_path(self, image_path: Path) -> tuple[bool, str | None]:
        if not self._command_exists():
            return False, f"{self.command!r} not found on PATH"

        image_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [self.command, "-o", str(image_path), "--timeout", str(self.timeout_ms)]
        if self.nopreview:
            cmd.append("--nopreview")
        # Raspberry Pi rpicam-still supports --camera. It is harmless for a single
        # AI Camera and useful once more CSI devices are attached.
        cmd.extend(["--camera", str(self.camera_index)])
        if self.use_width_height_args and self.width and self.height:
            cmd.extend(["--width", str(int(self.width)), "--height", str(int(self.height))])
        cmd.extend(self.extra_args)

        try:
            proc = self._run(cmd)
        except subprocess.TimeoutExpired:
            return False, f"capture timed out after {self.process_timeout_sec:.1f}s: {' '.join(cmd)}"
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

        if proc.returncode != 0:
            text = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
            return False, text or f"capture command returned {proc.returncode}: {' '.join(cmd)}"
        if not image_path.exists() or image_path.stat().st_size <= 0:
            return False, f"capture command completed but did not create a non-empty image at {image_path}"
        return True, None

    def _probe_one(self, camera_id: str) -> tuple[bool, str | None]:
        if camera_id not in self.camera_ids:
            return False, f"camera_id {camera_id!r} is not configured; configured camera_ids={self.camera_ids}"
        if self.list_cameras_first:
            ok, detail = self._list_cameras()
            if not ok:
                return False, detail
        if not self.health_check_capture:
            return True, None
        probe_path = self.frames_dir / "health_check" / f"probe_{camera_id}.jpg"
        return self._capture_to_path(probe_path)

    def health_check(self) -> dict[str, Any]:
        results: dict[str, bool] = {}
        errors: list[str] = []
        for camera_id in self.camera_ids:
            ok, err = self._probe_one(camera_id)
            results[camera_id] = ok
            if err:
                errors.append(f"camera {camera_id}: {err}")
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
            last_error = "; ".join(errors) if errors else "no rpicam camera frames available"
        return {
            "status": status,
            "self_test": self_test,
            "capabilities": self.capabilities() | {"probe_results": results},
            "last_error": last_error,
            "warnings": warnings,
        }

    def get_frame(self, camera_id: str) -> CameraFrame:
        if camera_id not in self.camera_ids:
            raise RuntimeError(f"camera_id {camera_id!r} is not configured; configured camera_ids={self.camera_ids}")
        self.frame_counter += 1
        image_path = self.frames_dir / f"{self.frame_counter:06d}_{camera_id}_rpicam.jpg"
        ok, err = self._capture_to_path(image_path)
        if not ok:
            raise RuntimeError(f"rpicam capture failed for camera {camera_id}: {err}")
        self.bus.transact(
            device="camera_array",
            command="rpicam_capture",
            params={"camera_id": camera_id, "camera_index": self.camera_index, "backend": "rpicam_still"},
            response_payload={"image_path": str(image_path), "camera_id": camera_id},
            time_sec=None,
        )
        return CameraFrame(camera_id=camera_id, image_path=str(image_path), visible_objects=[])
