from __future__ import annotations
import random
from pathlib import Path
from typing import Any
import math
import numpy as np

from ocean_voyager.core.types import ThermalReading, ObjectStatus
from ocean_voyager.interfaces.thermal import ThermalInterface
from ocean_voyager.simulation.visibility import visible_objects
from ocean_voyager.simulation.world import SimWorld
from ocean_voyager.interfaces.pan_servo import PanServoInterface
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class SimThermalSensor(ThermalInterface):
    """Hardware-faithful MLX90640-like thermal simulator.

    The main program receives a 24x32-style thermal frame summary: blob bearing,
    pixel count, max delta, and confidence. It does NOT receive true object range
    or ground-truth object coordinates from the thermal sensor.
    """

    def __init__(self, world: SimWorld, pan_servo: PanServoInterface, cfg: dict[str, Any], output_dir: str | Path | None = None, bus: Any | None = None):
        self.world = world
        self.pan_servo = pan_servo
        self.max_range_m = float(cfg.get("max_range_m", cfg.get("max_detection_range_m", 4.0)))
        self.fov_deg = float(cfg.get("fov_deg", cfg.get("fov_x_deg", 55.0)))
        self.fov_y_deg = float(cfg.get("fov_y_deg", 35.0))
        self.noise_std_c = float(cfg.get("noise_std_c", 0.08))
        self.threshold_delta_c = float(cfg.get("threshold_delta_c", cfg.get("min_detectable_delta_c", 0.35)))
        self.scan_min_deg = float(cfg.get("scan_min_deg", -60.0))
        self.scan_max_deg = float(cfg.get("scan_max_deg", 60.0))
        self.scan_step_deg = float(cfg.get("scan_step_deg", 15.0))
        self.distance_decay_power = float(cfg.get("distance_decay_power", 1.1))
        self.resolution_w = int(cfg.get("resolution_w", 32))
        self.resolution_h = int(cfg.get("resolution_h", 24))
        self.background_temp_c = float(cfg.get("background_temp_c", 22.0))
        self.random = random.Random(int(cfg.get("seed", 7)))
        self.np_random = np.random.default_rng(int(cfg.get("seed", 7)))
        self.output_dir = Path(output_dir or "outputs")
        self.frames_dir = self.output_dir / "thermal_frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.frame_counter = 0
        self.bus = bus or NullDeviceBus()
        self.expose_sim_debug = bool(cfg.get("expose_sim_debug_ground_truth", False))
        self.min_blob_pixels = int(cfg.get("min_blob_pixels", 4))

    def _thermal_frame(self, visible: list[tuple], mount_angle: float) -> tuple[np.ndarray, dict[str, Any]]:
        frame = self.background_temp_c + self.np_random.normal(0.0, self.noise_std_c, (self.resolution_h, self.resolution_w))
        metadata: dict[str, Any] = {"debug_visible_objects": [], "peak_pixel": None}
        for obj, d, bearing in visible:
            # Processed objects remain in the simulated world, but the autonomy layer should
            # ignore already processed zones. To keep demos stable, processed objects no
            # longer generate new thermal candidates.
            if obj.status in {ObjectStatus.CONFIRMED, ObjectStatus.REJECTED, ObjectStatus.LOST}:
                continue
            attenuation = max(0.0, 1.0 - (d / self.max_range_m) ** self.distance_decay_power)
            observed = obj.thermal_delta_c * attenuation
            x_center = int(round((bearing + self.fov_deg / 2.0) / max(self.fov_deg, 1e-6) * (self.resolution_w - 1)))
            y_center = self.resolution_h // 2
            sigma = max(0.8, 2.4 / max(d, 0.4))
            for yy in range(self.resolution_h):
                for xx in range(self.resolution_w):
                    blob = math.exp(-(((xx - x_center) ** 2 + (yy - y_center) ** 2) / (2 * sigma * sigma)))
                    frame[yy, xx] += observed * blob
            metadata["debug_visible_objects"].append({
                "id": obj.id, "type": obj.type.value, "distance_m": d,
                "bearing_relative_to_pan_deg": bearing, "observed_delta_c": observed,
            })
        max_idx = np.unravel_index(int(np.argmax(frame)), frame.shape)
        metadata["peak_pixel"] = [int(max_idx[1]), int(max_idx[0])]
        metadata["max_temp_c"] = float(np.max(frame))
        metadata["min_temp_c"] = float(np.min(frame))
        metadata["max_delta_c"] = float(np.max(frame) - self.background_temp_c)
        # Blob stats are derived from the returned frame, not ground-truth object range.
        mask = (frame - self.background_temp_c) >= self.threshold_delta_c
        ys, xs = np.where(mask)
        if len(xs) > 0:
            weights = np.maximum(frame[ys, xs] - self.background_temp_c, 1e-6)
            cx = float(np.average(xs, weights=weights))
            cy = float(np.average(ys, weights=weights))
            centroid_bearing_relative_pan = ((cx / max(self.resolution_w - 1, 1)) - 0.5) * self.fov_deg
        else:
            cx = cy = None
            centroid_bearing_relative_pan = None
        metadata["blob_pixel_count"] = int(len(xs))
        metadata["blob_centroid_pixel"] = None if cx is None else [round(cx, 3), round(cy, 3)]
        metadata["blob_bearing_relative_pan_deg"] = None if centroid_bearing_relative_pan is None else float(centroid_bearing_relative_pan)
        return frame, metadata

    def scan(self) -> list[ThermalReading]:
        readings: list[ThermalReading] = []
        angle = self.scan_min_deg
        while angle <= self.scan_max_deg + 1e-9:
            if hasattr(self.pan_servo, "set_context"):
                self.pan_servo.set_context(time_s=self.world.time_s)
            self.pan_servo.rotate_to(angle)
            visible = visible_objects(
                self.world.robot_pose,
                self.world.objects,
                max_range_m=self.max_range_m,
                fov_deg=self.fov_deg,
                mount_angle_deg=angle,
            )
            frame, meta = self._thermal_frame(visible, angle)
            self.frame_counter += 1
            frame_path = self.frames_dir / f"thermal_{self.frame_counter:06d}_pan_{angle:+05.1f}.npy"
            np.save(frame_path, frame)

            candidate = bool(meta["blob_pixel_count"] >= self.min_blob_pixels and meta["max_delta_c"] >= self.threshold_delta_c)
            confidence = max(0.0, min(1.0, (meta["max_delta_c"] / max(self.threshold_delta_c * 3.0, 1e-6)) * min(1.0, meta["blob_pixel_count"] / 8.0)))
            if meta["blob_bearing_relative_pan_deg"] is not None:
                bearing = float(angle + meta["blob_bearing_relative_pan_deg"])
            else:
                bearing = float(angle)
            payload = {
                "frame_path": str(frame_path),
                "frame_shape": [self.resolution_h, self.resolution_w],
                "background_temp_c": self.background_temp_c,
                "max_temp_c": round(meta["max_temp_c"], 4),
                "min_temp_c": round(meta["min_temp_c"], 4),
                "max_delta_c": round(meta["max_delta_c"], 4),
                "peak_pixel": meta["peak_pixel"],
                "blob_centroid_pixel": meta["blob_centroid_pixel"],
                "blob_pixel_count": meta["blob_pixel_count"],
                "bearing_deg": round(bearing, 4) if candidate else None,
                "range_known": False,
                "range_source": "bearing_only_no_range",
            }
            if self.expose_sim_debug:
                payload["sim_debug_visible_objects"] = meta["debug_visible_objects"]
            self.bus.transact(
                device="thermal",
                command="capture_frame",
                params={
                    "pan_angle_deg": angle,
                    "resolution": [self.resolution_h, self.resolution_w],
                    "fov_x_deg": self.fov_deg,
                    "fov_y_deg": self.fov_y_deg,
                },
                response_payload=payload,
                time_sec=self.world.time_s,
            )
            readings.append(
                ThermalReading(
                    object_id=None,
                    bearing_deg=bearing,
                    distance_m=-1.0,
                    observed_delta_c=float(meta["max_delta_c"]),
                    confidence=confidence,
                    pan_angle_deg=angle,
                    candidate=candidate,
                    frame_path=str(frame_path),
                    peak_pixel=tuple(meta["peak_pixel"]) if meta.get("peak_pixel") else None,
                    blob_centroid_pixel=tuple(meta["blob_centroid_pixel"]) if meta.get("blob_centroid_pixel") else None,
                    blob_pixel_count=int(meta["blob_pixel_count"]),
                    range_known=False,
                    range_source="bearing_only_no_range",
                )
            )
            angle += self.scan_step_deg
        readings.sort(key=lambda r: r.confidence, reverse=True)
        return readings
