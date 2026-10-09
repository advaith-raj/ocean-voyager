from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import numpy as np

from ocean_voyager.interfaces.thermal import ThermalInterface
from ocean_voyager.core.types import ThermalReading
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class RealMLX90640Thermal(ThermalInterface):
    """Raspberry Pi + MLX90640 adapter with warm-up/retry behavior."""

    def __init__(self, cfg: dict[str, Any], output_dir: str | Path | None = None, bus: Any | None = None, pan_servo: Any | None = None):
        self.cfg = cfg
        self.output_dir = Path(output_dir or "outputs")
        self.frames_dir = self.output_dir / "thermal_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)

        self.bus = bus or NullDeviceBus()
        self.pan_servo = pan_servo

        self.resolution_w = int(cfg.get("resolution_w", 32))
        self.resolution_h = int(cfg.get("resolution_h", 24))
        self.fov_deg = float(cfg.get("fov_deg", cfg.get("fov_x_deg", 55.0)))
        self.fov_y_deg = float(cfg.get("fov_y_deg", 35.0))
        self.threshold_delta_c = float(cfg.get("threshold_delta_c", cfg.get("min_detectable_delta_c", 0.35)))
        self.min_blob_pixels = int(cfg.get("min_blob_pixels", 4))
        self.refresh_rate_hz = float(cfg.get("refresh_rate_hz", 1))
        self.read_retries = int(cfg.get("read_retries", 8))
        self.retry_delay_sec = float(cfg.get("retry_delay_sec", 0.25))

        self.frame_counter = 0
        self.available = False
        self.mlx = None
        self.last_error: str | None = None
        self.warnings: list[str] = []

        try:
            import board  # type: ignore
            import busio  # type: ignore
            import adafruit_mlx90640  # type: ignore

            i2c = busio.I2C(board.SCL, board.SDA)
            self.mlx = adafruit_mlx90640.MLX90640(i2c)

            rate_map = {
                1: "REFRESH_1_HZ",
                2: "REFRESH_2_HZ",
                4: "REFRESH_4_HZ",
                8: "REFRESH_8_HZ",
            }
            attr = rate_map.get(int(round(self.refresh_rate_hz)), "REFRESH_1_HZ")
            try:
                self.mlx.refresh_rate = getattr(adafruit_mlx90640.RefreshRate, attr)
            except Exception as exc:
                self.warnings.append(f"Could not set MLX90640 refresh rate {attr}: {type(exc).__name__}: {exc}")

            self.available = True
            time.sleep(1.0)

        except Exception as exc:
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.warnings.append("MLX90640 library/device not available yet. Startup will report failed.")

    def capabilities(self) -> dict[str, Any]:
        return {
            "sensor": "MLX90640",
            "frame_shape": [self.resolution_h, self.resolution_w],
            "bearing_only": True,
            "range_known": False,
            "fov_x_deg": self.fov_deg,
            "fov_y_deg": self.fov_y_deg,
            "threshold_delta_c": self.threshold_delta_c,
            "refresh_rate_hz": self.refresh_rate_hz,
            "read_retries": self.read_retries,
        }

    def _read_frame_with_retries(self) -> list[float]:
        if self.mlx is None:
            raise RuntimeError("MLX90640 object is not initialized")

        last_exc: Exception | None = None
        frame = [0.0] * (self.resolution_w * self.resolution_h)

        for attempt in range(1, self.read_retries + 1):
            try:
                self.mlx.getFrame(frame)
                arr = np.array(frame, dtype=float)
                if not np.isfinite(arr).all():
                    raise RuntimeError("Frame contains non-finite values")
                return frame
            except Exception as exc:
                last_exc = exc
                time.sleep(self.retry_delay_sec)

        raise RuntimeError(f"MLX90640 frame read failed after {self.read_retries} retries: {type(last_exc).__name__}: {last_exc}")

    def health_check(self) -> dict[str, Any]:
        if not self.available:
            return {
                "status": "failed",
                "self_test": "fail",
                "capabilities": self.capabilities(),
                "last_error": self.last_error,
                "warnings": self.warnings,
            }

        try:
            frame = self._read_frame_with_retries()
            return {
                "status": "ready",
                "self_test": "pass",
                "capabilities": self.capabilities() | {
                    "sample_min_c": round(float(min(frame)), 3),
                    "sample_max_c": round(float(max(frame)), 3),
                },
                "last_error": None,
                "warnings": self.warnings,
            }
        except Exception as exc:
            return {
                "status": "failed",
                "self_test": "fail",
                "capabilities": self.capabilities(),
                "last_error": f"{type(exc).__name__}: {exc}",
                "warnings": self.warnings,
            }

    def _bearing_from_peak(self, x: int) -> float:
        pan = self.pan_servo.current_angle() if self.pan_servo is not None and hasattr(self.pan_servo, "current_angle") else 0.0
        x_center = (self.resolution_w - 1) / 2.0
        rel = (x - x_center) / max(x_center, 1e-6)
        return float(pan + rel * (self.fov_deg / 2.0))

    def scan(self) -> list[ThermalReading]:
        if not self.available or self.mlx is None:
            raise RuntimeError(f"RealMLX90640Thermal is not ready: {self.last_error}")

        frame_list = self._read_frame_with_retries()
        frame = np.array(frame_list, dtype=float).reshape((self.resolution_h, self.resolution_w))

        self.frame_counter += 1
        frame_path = self.frames_dir / f"real_thermal_{self.frame_counter:06d}.npy"
        np.save(frame_path, frame)

        background = float(np.median(frame))
        delta = frame - background
        max_delta = float(delta.max())

        peak_y, peak_x = tuple(int(v) for v in np.unravel_index(np.argmax(delta), delta.shape))
        mask = delta >= self.threshold_delta_c
        blob_count = int(mask.sum())

        candidate = bool(max_delta >= self.threshold_delta_c and blob_count >= self.min_blob_pixels)

        if blob_count > 0:
            ys, xs = np.where(mask)
            centroid_x = float(xs.mean())
            centroid_y = float(ys.mean())
        else:
            centroid_x = float(peak_x)
            centroid_y = float(peak_y)

        bearing = self._bearing_from_peak(int(round(centroid_x)))
        confidence = max(0.0, min(1.0, max_delta / max(self.threshold_delta_c * 3.0, 1e-6)))

        reading = ThermalReading(
            object_id=None,
            bearing_deg=round(bearing, 3),
            distance_m=-1.0,
            observed_delta_c=round(max_delta, 3),
            confidence=round(confidence, 3),
            pan_angle_deg=float(self.pan_servo.current_angle() if self.pan_servo is not None and hasattr(self.pan_servo, "current_angle") else 0.0),
            candidate=candidate,
            frame_path=str(frame_path),
            peak_pixel=(peak_x, peak_y),
            blob_centroid_pixel=(centroid_x, centroid_y),
            blob_pixel_count=blob_count,
            range_known=False,
            range_source="real_mlx90640_bearing_only",
        )

        self.bus.transact(
            device="thermal",
            command="capture_frame",
            params={"source": "real_mlx90640"},
            response_payload={"reading": reading.as_dict(), "frame_path": str(frame_path)},
            time_sec=None,
        )

        return [reading]
