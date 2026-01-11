#!/usr/bin/env python3
"""
All-in-one controller program (single file)

Supports two modes:
  1) Raspberry Pi + PCA9685 + MLX90640 (I2C)
  2) Dry-run simulation (no hardware)

Features included:
  - Hardware Abstraction Layer (HAL) interfaces + implementations (Pi + DryRun)
  - Explicit finite-state machine: SCAN → COARSE_ALIGN → FINE_ALIGN → APPROACH → VERIFY → DONE/ABORT
  - Concurrency:
      * Thermal capture thread (latest-frame buffer)
      * Verification worker thread (async verify requests)
      * Watchdog thread (failsafe stop if controller stalls)
  - Logging + recording:
      * events.jsonl (structured event stream)
      * optional thermal frame dumps (.npz) + summary stats

Usage:
  - Write a config template:
      python3 controller_all_in_one.py --write-config-template controller_config.json

  - Dry run:
      python3 controller_all_in_one.py --config controller_config.json --hal dryrun

  - Raspberry Pi (PCA9685 + MLX90640):
      python3 controller_all_in_one.py --config controller_config.json --hal pi_pca9685
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import queue
import random
import signal
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Optional: OpenCV for camera capture + wall-guard vision
try:
    import cv2  # type: ignore
    HAS_CV2 = True
except Exception:
    cv2 = None  # type: ignore
    HAS_CV2 = False



# ============================================================
# Utilities
# ============================================================
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))

def now_s() -> float:
    return time.time()

def _is_tbd(v: Any) -> bool:
    return v is None or (isinstance(v, str) and "TBD" in v.upper())

def require_int(name: str, v: Any) -> int:
    if _is_tbd(v):
        raise RuntimeError(f"Config value '{name}' is TBD. Edit your config file and set an integer.")
    try:
        return int(v)
    except Exception as e:
        raise RuntimeError(f"Config value '{name}' must be an integer, got {v!r}") from e

def require_float(name: str, v: Any) -> float:
    if _is_tbd(v):
        raise RuntimeError(f"Config value '{name}' is TBD. Edit your config file and set a number.")
    try:
        return float(v)
    except Exception as e:
        raise RuntimeError(f"Config value '{name}' must be a number, got {v!r}") from e

def load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    try:
        return json.loads(p.read_text())
    except Exception as e:
        raise RuntimeError(f"Failed to parse JSON: {path}") from e


# ============================================================
# Recording / Logging
# ============================================================
class RunRecorder:
    def __init__(self, base_dir: str, enable_npz: bool, npz_every_n: int):
        self.base_dir = base_dir or "runs"
        ts = time.strftime("%Y%m%d-%H%M%S")
        self.run_dir = Path(self.base_dir) / ts
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"
        self.enable_npz = enable_npz
        self.npz_every_n = max(1, int(npz_every_n))
        self._frame_count = 0
        self._lock = threading.Lock()

    def write_event(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event["t"] = now_s()
        line = json.dumps(event, ensure_ascii=False)
        with self._lock:
            with self.events_path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def maybe_dump_frame(self, frame: np.ndarray, tag: str) -> None:
        if not self.enable_npz:
            return
        self._frame_count += 1
        if (self._frame_count % self.npz_every_n) != 0:
            return
        p = self.run_dir / f"thermal_{self._frame_count:06d}_{tag}.npz"
        # Store lightweight summary too
        np.savez_compressed(
            p,
            frame=frame.astype(np.float32),
            min=float(np.nanmin(frame)),
            max=float(np.nanmax(frame)),
            mean=float(np.nanmean(frame)),
        )


# ============================================================
# Blob detection + tracking
# ============================================================
@dataclass
class Blob:
    id: int
    area: int
    cy: float
    cx: float
    max_temp: float
    mean_temp: float

def connected_components(binary: np.ndarray) -> Tuple[np.ndarray, int]:
    """Simple 4-connected labeling (good enough for 24x32)."""
    h, w = binary.shape
    labels = np.zeros((h, w), dtype=np.int32)
    current = 0
    stack: List[Tuple[int, int]] = []
    for y in range(h):
        for x in range(w):
            if not binary[y, x] or labels[y, x] != 0:
                continue
            current += 1
            labels[y, x] = current
            stack.append((y, x))
            while stack:
                yy, xx = stack.pop()
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = yy + dy, xx + dx
                    if 0 <= ny < h and 0 <= nx < w and binary[ny, nx] and labels[ny, nx] == 0:
                        labels[ny, nx] = current
                        stack.append((ny, nx))
    return labels, current

def extract_blobs(temp: np.ndarray, thresh_c: float, min_area: int) -> List[Blob]:
    binary = temp > float(thresh_c)
    labels, n = connected_components(binary)
    blobs: List[Blob] = []
    for bid in range(1, n + 1):
        pts = np.argwhere(labels == bid)
        area = int(pts.shape[0])
        if area < int(min_area):
            continue
        cy, cx = pts.mean(axis=0)
        vals = temp[labels == bid]
        blobs.append(
            Blob(
                id=bid,
                area=area,
                cy=float(cy),
                cx=float(cx),
                max_temp=float(vals.max()) if vals.size else float("nan"),
                mean_temp=float(vals.mean()) if vals.size else float("nan"),
            )
        )
    blobs.sort(key=lambda b: (b.area, b.max_temp), reverse=True)
    return blobs

@dataclass
class TrackerState:
    last_cx: Optional[float] = None
    last_cy: Optional[float] = None
    stable_count: int = 0

def pick_target(blobs: List[Blob], tracker: TrackerState, max_jump_px: float, stable_required: int) -> Optional[Blob]:
    """Choose a blob with temporal stability (avoids 'largest blob' flipping)."""
    if not blobs:
        tracker.last_cx = tracker.last_cy = None
        tracker.stable_count = 0
        return None

    if tracker.last_cx is None:
        tgt = blobs[0]
        tracker.last_cx, tracker.last_cy = tgt.cx, tgt.cy
        tracker.stable_count = 1
        return tgt

    # Nearest to previous centroid
    best = None
    best_d = 1e9
    for b in blobs:
        d = math.hypot(b.cx - tracker.last_cx, b.cy - tracker.last_cy)
        if d < best_d:
            best = b
            best_d = d

    if best is None or best_d > float(max_jump_px):
        tgt = blobs[0]
        tracker.last_cx, tracker.last_cy = tgt.cx, tgt.cy
        tracker.stable_count = 1
        return tgt

    tracker.stable_count = tracker.stable_count + 1 if best_d < 1.0 else 1
    tracker.last_cx, tracker.last_cy = best.cx, best.cy

    return best if tracker.stable_count >= int(stable_required) else None


# ============================================================
# HAL Interfaces
# ============================================================
class ThermalCamera:
    def read(self) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        pass

class Thrusters:
    def set(self, yaw: float, forward: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass

class PanServo:
    def rotate_to(self, deg: float) -> None:
        raise NotImplementedError

    def park(self) -> None:
        pass

    def close(self) -> None:
        pass

class EStop:
    def is_pressed(self) -> bool:
        return False

class Verifier:
    def verify(self) -> Tuple[bool, float]:
        """Return (is_target, confidence in [0,1])."""
        return False, 0.0


# ============================================================
# Raspberry Pi HAL: PCA9685 + MLX90640
# ============================================================
def _import_pi_deps():
    try:
        import board  # type: ignore
        import busio  # type: ignore
        import adafruit_mlx90640  # type: ignore
        from adafruit_pca9685 import PCA9685  # type: ignore
        return board, busio, adafruit_mlx90640, PCA9685
    except Exception as e:
        raise RuntimeError(
            "Missing Raspberry Pi hardware dependencies. Install:\n"
            "  pip3 install adafruit-blinka adafruit-circuitpython-pca9685 adafruit-circuitpython-mlx90640\n"
            "and ensure I2C is enabled."
        ) from e

def us_to_duty(pulse_us: float, frequency_hz: float) -> int:
    period_us = 1_000_000.0 / float(frequency_hz)
    frac = clamp(pulse_us / period_us, 0.0, 1.0)
    return int(frac * 0xFFFF)

class PiPCA9685Driver:
    def __init__(self, address: int, frequency_hz: int, i2c_freq_hz: int):
        board, busio, _, PCA9685 = _import_pi_deps()
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=int(i2c_freq_hz))
        self.pca = PCA9685(self.i2c, address=int(address))
        self.pca.frequency = int(frequency_hz)

    def set_pulse_us(self, channel: int, pulse_us: float) -> None:
        if not (0 <= int(channel) <= 15):
            raise ValueError(f"PCA9685 channel must be 0..15, got {channel}")
        duty = us_to_duty(float(pulse_us), float(self.pca.frequency))
        self.pca.channels[int(channel)].duty_cycle = duty

    def close(self) -> None:
        try:
            self.pca.deinit()
        except Exception:
            pass

class PiPCA9685Thrusters(Thrusters):
    def __init__(
        self,
        driver: PiPCA9685Driver,
        left_ch: int,
        right_ch: int,
        neutral_us: int,
        min_us: int,
        max_us: int,
        yaw_gain_us: float,
        fwd_gain_us: float,
        invert_left: bool,
        invert_right: bool,
        init_neutral_s: float,
    ):
        self.d = driver
        self.left = int(left_ch)
        self.right = int(right_ch)
        self.neutral = int(neutral_us)
        self.min_us = int(min_us)
        self.max_us = int(max_us)
        self.yaw_gain = float(yaw_gain_us)
        self.fwd_gain = float(fwd_gain_us)
        self.inv_left = bool(invert_left)
        self.inv_right = bool(invert_right)
        self.stop()
        time.sleep(float(init_neutral_s))

    def stop(self) -> None:
        try:
            self.d.set_pulse_us(self.left, self.neutral)
            self.d.set_pulse_us(self.right, self.neutral)
        except Exception:
            pass

    def set(self, yaw: float, forward: float) -> None:
        yaw = clamp(float(yaw), -1.0, 1.0)
        forward = clamp(float(forward), -1.0, 1.0)

        forward_l = -forward if self.inv_left else forward
        forward_r = -forward if self.inv_right else forward

        left_us = self.neutral + forward_l * self.fwd_gain - yaw * self.yaw_gain
        right_us = self.neutral + forward_r * self.fwd_gain + yaw * self.yaw_gain
        left_us = clamp(left_us, self.min_us, self.max_us)
        right_us = clamp(right_us, self.min_us, self.max_us)

        try:
            self.d.set_pulse_us(self.left, left_us)
            self.d.set_pulse_us(self.right, right_us)
        except Exception:
            self.stop()
            raise

class PiPCA9685Servo(PanServo):
    def __init__(
        self,
        driver: PiPCA9685Driver,
        channel: int,
        min_deg: float,
        max_deg: float,
        min_us: int,
        max_us: int,
        park_deg: float,
    ):
        self.d = driver
        self.ch = int(channel)
        self.min_deg = float(min_deg)
        self.max_deg = float(max_deg)
        self.min_us = int(min_us)
        self.max_us = int(max_us)
        self.park_deg = float(park_deg)
        self.rotate_to(self.park_deg)

    def _deg_to_us(self, deg: float) -> float:
        deg = clamp(float(deg), self.min_deg, self.max_deg)
        frac = (deg - self.min_deg) / (self.max_deg - self.min_deg) if self.max_deg != self.min_deg else 0.5
        return self.min_us + frac * (self.max_us - self.min_us)

    def rotate_to(self, deg: float) -> None:
        self.d.set_pulse_us(self.ch, self._deg_to_us(deg))

    def park(self) -> None:
        try:
            self.rotate_to(self.park_deg)
        except Exception:
            pass

class PiMLX90640Thermal(ThermalCamera):
    def __init__(self, i2c_freq_hz: int, refresh_rate_name: str, max_retries: int, retry_delay_s: float):
        board, busio, adafruit_mlx90640, _ = _import_pi_deps()
        self.i2c = busio.I2C(board.SCL, board.SDA, frequency=int(i2c_freq_hz))
        self.mlx = adafruit_mlx90640.MLX90640(self.i2c)
        rate = getattr(adafruit_mlx90640.RefreshRate, str(refresh_rate_name), None)
        if rate is None:
            raise ValueError(f"Invalid MLX refresh rate: {refresh_rate_name}")
        self.mlx.refresh_rate = rate
        self.frame = [0.0] * (24 * 32)
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay_s)

    def read(self) -> np.ndarray:
        for _ in range(self.max_retries):
            try:
                self.mlx.getFrame(self.frame)
                img = np.array(self.frame, dtype=np.float32).reshape((24, 32))
                # sanity checks
                if not np.isfinite(img).all():
                    time.sleep(self.retry_delay)
                    continue
                if img.min() < -40 or img.max() > 300:
                    time.sleep(self.retry_delay)
                    continue
                return img
            except ValueError:
                time.sleep(self.retry_delay)
            except OSError as e:
                raise RuntimeError("I2C error reading MLX90640") from e
        raise RuntimeError("MLX90640 read retries exceeded")


# ============================================================
# Dry-run HAL (simulation)
# ============================================================
class DryRunThermal(ThermalCamera):
    """Simulated 24x32 thermal frames with a moving warm blob + noise."""
    def __init__(self, ambient_c: float, blob_c: float, noise_c: float, drift: float):
        self.ambient = float(ambient_c)
        self.blob = float(blob_c)
        self.noise = float(noise_c)
        self.drift = float(drift)
        self.t = 0.0
        # start near center
        self.cx = 16.0
        self.cy = 12.0

    def read(self) -> np.ndarray:
        self.t += 1.0
        # random walk
        self.cx = clamp(self.cx + random.uniform(-self.drift, self.drift), 2.0, 29.0)
        self.cy = clamp(self.cy + random.uniform(-self.drift, self.drift), 2.0, 21.0)

        img = np.full((24, 32), self.ambient, dtype=np.float32)
        # blob as gaussian-ish hill
        for y in range(24):
            for x in range(32):
                d2 = (x - self.cx) ** 2 + (y - self.cy) ** 2
                img[y, x] += self.blob * math.exp(-d2 / 10.0)
        img += np.random.normal(0.0, self.noise, size=img.shape).astype(np.float32)
        return img

class DryRunThrusters(Thrusters):
    def __init__(self):
        self.last = (0.0, 0.0)

    def set(self, yaw: float, forward: float) -> None:
        self.last = (float(yaw), float(forward))

    def stop(self) -> None:
        self.last = (0.0, 0.0)

class DryRunServo(PanServo):
    def __init__(self, park_deg: float):
        self.pos = float(park_deg)

    def rotate_to(self, deg: float) -> None:
        self.pos = float(deg)

    def park(self) -> None:
        pass

class DryRunVerifier(Verifier):
    def __init__(self, accept_prob: float):
        self.accept_prob = float(accept_prob)

    def verify(self) -> Tuple[bool, float]:
        ok = random.random() < self.accept_prob
        conf = random.uniform(0.6, 0.95) if ok else random.uniform(0.05, 0.4)
        return ok, conf


# ============================================================
# Threads: thermal capture, verification worker, watchdog
# ============================================================
class LatestValue:
    """Thread-safe latest value container."""
    def __init__(self):
        self._lock = threading.Lock()
        self._val = None
        self._ts = 0.0

    def set(self, val: Any) -> None:
        with self._lock:
            self._val = val
            self._ts = now_s()

    def get(self) -> Tuple[Any, float]:
        with self._lock:
            return self._val, self._ts

class ThermalCaptureThread(threading.Thread):
    def __init__(self, cam: ThermalCamera, latest: LatestValue, stop_event: threading.Event, period_s: float, recorder: RunRecorder):
        super().__init__(daemon=True)
        self.cam = cam
        self.latest = latest
        self.stop_event = stop_event
        self.cameras = cameras
        self.wall_guard = wall_guard
        self.period = float(period_s)
        self.recorder = recorder

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                frame = self.cam.read()
                self.latest.set(frame)
                self.recorder.maybe_dump_frame(frame, tag="cap")
            except Exception as e:
                self.latest.set(None)
                self.recorder.write_event({"event": "thermal_read_error", "error": str(e)})
                time.sleep(0.2)
            time.sleep(self.period)

class VerifyRequest:
    def __init__(self, req_id: int):
        self.req_id = req_id

class VerifyResult:
    def __init__(self, req_id: int, ok: bool, conf: float):
        self.req_id = req_id
        self.ok = ok
        self.conf = conf

class VerifierWorker(threading.Thread):
    def __init__(self, verifier: Verifier, in_q: "queue.Queue[VerifyRequest]", out_q: "queue.Queue[VerifyResult]",
                 stop_event: threading.Event, recorder: RunRecorder):
        super().__init__(daemon=True)
        self.verifier = verifier
        self.in_q = in_q
        self.out_q = out_q
        self.stop_event = stop_event
        self.cameras = cameras
        self.wall_guard = wall_guard
        self.recorder = recorder

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                req = self.in_q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                ok, conf = self.verifier.verify()
                self.out_q.put(VerifyResult(req.req_id, ok, conf))
            except Exception as e:
                self.recorder.write_event({"event": "verify_error", "error": str(e)})
                self.out_q.put(VerifyResult(req.req_id, False, 0.0))

class WatchdogThread(threading.Thread):
    def __init__(self, get_heartbeat: callable, thrusters: Thrusters, stop_event: threading.Event,
                 timeout_s: float, recorder: RunRecorder):
        super().__init__(daemon=True)
        self.get_heartbeat = get_heartbeat
        self.thrusters = thrusters
        self.stop_event = stop_event
        self.cameras = cameras
        self.wall_guard = wall_guard
        self.timeout_s = float(timeout_s)
        self.recorder = recorder
        self._tripped = False

    def run(self) -> None:
        while not self.stop_event.is_set():
            hb = self.get_heartbeat()
            if hb is not None and (now_s() - hb) > self.timeout_s and not self._tripped:
                self._tripped = True
                try:
                    self.thrusters.stop()
                except Exception:
                    pass
                self.recorder.write_event({"event": "watchdog_trip", "timeout_s": self.timeout_s})
            time.sleep(0.05)


# ============================================================
# Config
# ============================================================
@dataclass
class Config:
    # mode
    hal: str  # "pi_pca9685" or "dryrun"

    # recording
    run_base_dir: str
    save_npz: bool
    npz_every_n: int

    # watchdog
    watchdog_timeout_s: float

    # PCA9685
    pca_address: int
    pca_frequency: int
    pca_i2c_freq: int
    left_thruster_ch: int
    right_thruster_ch: int
    servo_ch: int

    esc_min_us: int
    esc_neutral_us: int
    esc_max_us: int
    yaw_gain_us: float
    fwd_gain_us: float
    invert_left: bool
    invert_right: bool
    esc_init_neutral_s: float

    servo_min_deg: float
    servo_max_deg: float
    servo_min_us: int
    servo_max_us: int
    servo_park_deg: float

    # thermal
    mlx_i2c_freq: int
    mlx_refresh_rate: str
    mlx_retries: int
    mlx_retry_delay_s: float

    # dryrun thermal
    dry_ambient_c: float
    dry_blob_c: float
    dry_noise_c: float
    dry_drift: float

    # detect/track
    blob_temp_threshold_c: float
    blob_min_area_px: int
    tracker_max_jump_px: float
    tracker_stable_frames: int

    # control
    yaw_kp_coarse: float
    yaw_kp_fine: float
    yaw_deadband_px: float
    yaw_cmd_limit: float
    forward_cmd: float

    # scanning
    scan_servo_min_deg: float
    scan_servo_max_deg: float
    scan_servo_step_deg: float
    scan_dwell_s: float

    # timeouts
    scan_timeout_s: float
    coarse_align_timeout_s: float
    fine_align_timeout_s: float
    approach_timeout_s: float
    verify_timeout_s: float

    # verify voting
    verify_votes: int
    verify_accept_count: int
    dry_verify_accept_prob: float


def config_template() -> Dict[str, Any]:
    # Reasonable defaults + explicit TBD fields where needed
    return {
        "hal": "TBD (pi_pca9685 or dryrun)",
        "run_base_dir": "runs",
        "save_npz": False,
        "npz_every_n": 10,
        "watchdog_timeout_s": 1.5,

        "pca_address": "TBD (usually 0x40)",
        "pca_frequency": 50,
        "pca_i2c_freq": 400000,
        "left_thruster_ch": "TBD (0-15)",
        "right_thruster_ch": "TBD (0-15)",
        "servo_ch": "TBD (0-15)",

        "esc_min_us": 1100,
        "esc_neutral_us": 1500,
        "esc_max_us": 1900,
        "yaw_gain_us": 250.0,
        "fwd_gain_us": 250.0,
        "invert_left": False,
        "invert_right": False,
        "esc_init_neutral_s": 0.4,

        "servo_min_deg": -90.0,
        "servo_max_deg": 90.0,
        "servo_min_us": 600,
        "servo_max_us": 2400,
        "servo_park_deg": 0.0,

        "mlx_i2c_freq": 800000,
        "mlx_refresh_rate": "REFRESH_2_HZ",
        "mlx_retries": 5,
        "mlx_retry_delay_s": 0.03,

        "dry_ambient_c": 24.0,
        "dry_blob_c": 18.0,
        "dry_noise_c": 0.6,
        "dry_drift": 0.6,

        "blob_temp_threshold_c": 32.0,
        "blob_min_area_px": 8,
        "tracker_max_jump_px": 6.0,
        "tracker_stable_frames": 3,

        "yaw_kp_coarse": 0.07,
        "yaw_kp_fine": 0.04,
        "yaw_deadband_px": 1.5,
        "yaw_cmd_limit": 0.6,
        "forward_cmd": 0.35,

        "scan_servo_min_deg": -60.0,
        "scan_servo_max_deg": 60.0,
        "scan_servo_step_deg": 10.0,
        "scan_dwell_s": 0.6,

        "scan_timeout_s": 30.0,
        "coarse_align_timeout_s": 12.0,
        "fine_align_timeout_s": 12.0,
        "approach_timeout_s": 20.0,
        "verify_timeout_s": 6.0,

        "verify_votes": 5,
        "verify_accept_count": 3,
        "dry_verify_accept_prob": 0.25,

        "TBD": {
            "fill_these_first": [
                "hal",
                "pca_address (if not 0x40)",
                "left_thruster_ch",
                "right_thruster_ch",
                "servo_ch"
            ],
            "hint": "PCA9685 channels are printed 0–15 on the board. Use those numbers."
        }
    }

def parse_config(d: Dict[str, Any]) -> Config:
    hal = str(d.get("hal", "")).strip()
    if _is_tbd(hal) or hal == "":
        raise RuntimeError("Config value 'hal' is TBD. Set it to 'pi_pca9685' or 'dryrun'.")

    return Config(
        hal=hal,

        run_base_dir=str(d.get("run_base_dir", "runs")),
        save_npz=bool(d.get("save_npz", False)),
        npz_every_n=int(d.get("npz_every_n", 10)),

        watchdog_timeout_s=float(d.get("watchdog_timeout_s", 1.5)),

        pca_address=require_int("pca_address", d.get("pca_address")),
        pca_frequency=int(d.get("pca_frequency", 50)),
        pca_i2c_freq=int(d.get("pca_i2c_freq", 400000)),
        left_thruster_ch=require_int("left_thruster_ch", d.get("left_thruster_ch")),
        right_thruster_ch=require_int("right_thruster_ch", d.get("right_thruster_ch")),
        servo_ch=require_int("servo_ch", d.get("servo_ch")),

        esc_min_us=int(d.get("esc_min_us", 1100)),
        esc_neutral_us=int(d.get("esc_neutral_us", 1500)),
        esc_max_us=int(d.get("esc_max_us", 1900)),
        yaw_gain_us=float(d.get("yaw_gain_us", 250.0)),
        fwd_gain_us=float(d.get("fwd_gain_us", 250.0)),
        invert_left=bool(d.get("invert_left", False)),
        invert_right=bool(d.get("invert_right", False)),
        esc_init_neutral_s=float(d.get("esc_init_neutral_s", 0.4)),

        servo_min_deg=float(d.get("servo_min_deg", -90.0)),
        servo_max_deg=float(d.get("servo_max_deg", 90.0)),
        servo_min_us=int(d.get("servo_min_us", 600)),
        servo_max_us=int(d.get("servo_max_us", 2400)),
        servo_park_deg=float(d.get("servo_park_deg", 0.0)),

        mlx_i2c_freq=int(d.get("mlx_i2c_freq", 800000)),
        mlx_refresh_rate=str(d.get("mlx_refresh_rate", "REFRESH_2_HZ")),
        mlx_retries=int(d.get("mlx_retries", 5)),
        mlx_retry_delay_s=float(d.get("mlx_retry_delay_s", 0.03)),

        dry_ambient_c=float(d.get("dry_ambient_c", 24.0)),
        dry_blob_c=float(d.get("dry_blob_c", 18.0)),
        dry_noise_c=float(d.get("dry_noise_c", 0.6)),
        dry_drift=float(d.get("dry_drift", 0.6)),

        blob_temp_threshold_c=float(d.get("blob_temp_threshold_c", 32.0)),
        blob_min_area_px=int(d.get("blob_min_area_px", 8)),
        tracker_max_jump_px=float(d.get("tracker_max_jump_px", 6.0)),
        tracker_stable_frames=int(d.get("tracker_stable_frames", 3)),

        yaw_kp_coarse=float(d.get("yaw_kp_coarse", 0.07)),
        yaw_kp_fine=float(d.get("yaw_kp_fine", 0.04)),
        yaw_deadband_px=float(d.get("yaw_deadband_px", 1.5)),
        yaw_cmd_limit=float(d.get("yaw_cmd_limit", 0.6)),
        forward_cmd=float(d.get("forward_cmd", 0.35)),

        scan_servo_min_deg=float(d.get("scan_servo_min_deg", -60.0)),
        scan_servo_max_deg=float(d.get("scan_servo_max_deg", 60.0)),
        scan_servo_step_deg=float(d.get("scan_servo_step_deg", 10.0)),
        scan_dwell_s=float(d.get("scan_dwell_s", 0.6)),

        scan_timeout_s=float(d.get("scan_timeout_s", 30.0)),
        coarse_align_timeout_s=float(d.get("coarse_align_timeout_s", 12.0)),
        fine_align_timeout_s=float(d.get("fine_align_timeout_s", 12.0)),
        approach_timeout_s=float(d.get("approach_timeout_s", 20.0)),
        verify_timeout_s=float(d.get("verify_timeout_s", 6.0)),

        verify_votes=int(d.get("verify_votes", 5)),
        verify_accept_count=int(d.get("verify_accept_count", 3)),
        dry_verify_accept_prob=float(d.get("dry_verify_accept_prob", 0.25)),
    )


# ============================================================
# Controller FSM
# ============================================================
class State:
    WALL_AVOID = "WALL_AVOID"
    SEARCH_PATTERN = "SEARCH_PATTERN"
    SCAN = "SCAN"
    COARSE_ALIGN = "COARSE_ALIGN"
    FINE_ALIGN = "FINE_ALIGN"
    APPROACH = "APPROACH"
    VERIFY = "VERIFY"
    DONE = "DONE"
    ABORT = "ABORT"

@dataclass
class ControllerContext:
    state: str = State.SCAN
    state_enter_t: float = 0.0
    target: Optional[Blob] = None
    scan_deg: float = 0.0
    scan_dir: int = 1
    scan_next_step_t: float = 0.0
    verify_req_id: int = 0
    verify_sent_t: float = 0.0
    verify_votes_ok: int = 0
    verify_votes_total: int = 0
    approach_burst_end_t: float = 0.0
    approach_pause_end_t: float = 0.0
    approach_motion_started_t: float = 0.0
    search_phase: str = ""
    search_phase_end_t: float = 0.0
    search_started_t: float = 0.0
    search_cycle_count: int = 0
    search_dir: int = 1
    wall_avoid_end_t: float = 0.0
    wall_avoid_cooldown_end_t: float = 0.0
    wall_avoid_yaw_dir: int = 1
    wall_guard_last_debug_t: float = 0.0

def compute_yaw_cmd(cx: float, center_x: float, kp: float, deadband_px: float, limit: float) -> float:
    err = cx - center_x
    if abs(err) <= deadband_px:
        return 0.0
    return clamp(kp * err, -limit, limit)

def is_timeout(ctx: ControllerContext, timeout_s: float) -> bool:
    return (now_s() - ctx.state_enter_t) > float(timeout_s)


class Controller:
    def __init__(
        self,
        cfg: Config,
        recorder: RunRecorder,
        thermal_latest: LatestValue,
        thrusters: Thrusters,
        servo: PanServo,
        verify_in: "queue.Queue[VerifyRequest]",
        verify_out: "queue.Queue[VerifyResult]",
        stop_event: threading.Event,
    ):
        self.cfg = cfg
        self.rec = recorder
        self.latest = thermal_latest
        self.thr = thrusters
        self.servo = servo
        self.verify_in = verify_in
        self.verify_out = verify_out
        self.stop_event = stop_event
        self.cameras = cameras
        self.wall_guard = wall_guard

        self.ctx = ControllerContext(state_enter_t=now_s(), scan_deg=cfg.servo_park_deg)
        # Pool mode starts with an open-loop search pattern to reduce straight-line drift.
        if self.cfg.pool_mode:
            self.ctx.state = State.SEARCH_PATTERN
            self.ctx.search_started_t = now_s()
            self.ctx.search_phase = "ROTATE"
            self.ctx.search_dir = 1
            self.ctx.search_phase_end_t = now_s() + self.cfg.pool_search_rotate_s
            self.rec.write_event({"event": "state", "state": State.SEARCH_PATTERN})
        self.tracker = TrackerState()
        self._heartbeat = now_s()

        self.center_x = 16.0  # 32 wide thermal

    def heartbeat(self) -> float:
        return self._heartbeat

    def _enter(self, new_state: str, **event_fields: Any) -> None:
        self.ctx.state = new_state
        self.ctx.state_enter_t = now_s()
        self.rec.write_event({"event": "state", "state": new_state, **event_fields})

        # Reset per-state counters
        if new_state == State.WALL_AVOID:
            self.ctx.wall_avoid_end_t = 0.0
        if new_state == State.SEARCH_PATTERN:
            self.ctx.search_started_t = now_s()
            self.ctx.search_phase = "ROTATE"
            self.ctx.search_dir = 1
            self.ctx.search_phase_end_t = now_s() + self.cfg.pool_search_rotate_s
            self.ctx.search_cycle_count = 0
        if new_state == State.APPROACH:
            self.ctx.approach_burst_end_t = 0.0
            self.ctx.approach_pause_end_t = 0.0
            self.ctx.approach_motion_started_t = now_s()
        if new_state == State.VERIFY:
            self.ctx.verify_req_id += 1
            self.ctx.verify_sent_t = now_s()
            self.ctx.verify_votes_ok = 0
            self.ctx.verify_votes_total = 0

    def _get_frame_and_target(self) -> Tuple[Optional[np.ndarray], Optional[Blob], List[Blob]]:
        frame, ts = self.latest.get()
        if frame is None:
            return None, None, []
        blobs = extract_blobs(frame, self.cfg.blob_temp_threshold_c, self.cfg.blob_min_area_px)
        target = pick_target(blobs, self.tracker, self.cfg.tracker_max_jump_px, self.cfg.tracker_stable_frames)
        return frame, target, blobs

    def step(self) -> None:
        self._heartbeat = now_s()

        frame, target, blobs = self._get_frame_and_target()
        if frame is not None:
            self.rec.maybe_dump_frame(frame, tag=self.ctx.state.lower())

        # Safety: stop motion if stopping
        if self.stop_event.is_set():
            self.thr.stop()
            return

        # WALL_AVOID: reactive avoidance using cameras (pool safety)
        if self.ctx.state == State.WALL_AVOID:
            now = now_s()
            if self.ctx.wall_avoid_end_t <= 0.0:
                self.ctx.wall_avoid_end_t = now + clamp(self.cfg.wall_guard_avoid_s, 0.1, 5.0)
                self.ctx.wall_avoid_cooldown_end_t = self.ctx.wall_avoid_end_t + clamp(self.cfg.wall_guard_cooldown_s, 0.1, 10.0)
            if now < self.ctx.wall_avoid_end_t:
                yaw = clamp(float(self.cfg.wall_guard_avoid_yaw_cmd) * float(self.ctx.wall_avoid_yaw_dir), -0.6, 0.6)
                self.thr.set(yaw=yaw, forward=clamp(float(self.cfg.wall_guard_avoid_reverse_cmd), -0.4, 0.0))
            else:
                self.thr.stop()
                if self.cfg.pool_mode:
                    self._enter(State.SEARCH_PATTERN, reason="wall_avoid_done")
                else:
                    self._enter(State.SCAN, reason="wall_avoid_done")
            return

        # Optional: vision-based wall guard (pool boundary safety)
        if self.wall_guard is not None and self.cameras is not None:
            # Determine whether we are *actually* commanding forward motion right now.
            moving_forward = False
            now = now_s()
            try:
                if self.ctx.state == State.APPROACH:
                    if not self.cfg.pool_mode:
                        moving_forward = float(self.cfg.forward_cmd) >= float(self.cfg.wall_guard_min_forward_cmd)
                    else:
                        moving_forward = (now < float(self.ctx.approach_burst_end_t)) and (float(self.cfg.forward_cmd) >= float(self.cfg.wall_guard_min_forward_cmd))
                elif self.ctx.state == State.SEARCH_PATTERN:
                    moving_forward = (getattr(self.ctx, "search_phase", "") == "FORWARD") and (now < float(getattr(self.ctx, "search_phase_end_t", 0.0)))
            except Exception:
                moving_forward = False

            if now >= float(self.ctx.wall_avoid_cooldown_end_t):
                frames = self.cameras.read_all()
                trig, info = self.wall_guard.check(frames, moving_forward=moving_forward)
                try:
                    _wall_guard_debug_dump(self.rec, self.cfg, self.ctx, frames, info, trig)
                except Exception:
                    pass
                if trig:
                    self.ctx.wall_avoid_yaw_dir = int((info or {}).get("yaw_dir", 1))
                    self.thr.stop()
                    self._enter(State.WALL_AVOID, reason="wall_guard")
                    return


        # SEARCH_PATTERN (pool mode): open-loop rotate/step pattern to reduce wall impacts.
        # - Rotates in place in alternating directions with pauses.
        # - Every N rotate cycles, performs a short forward step.
        # - Continuously checks for a stable thermal target; if found, transitions to COARSE_ALIGN.
        if self.ctx.state == State.SEARCH_PATTERN:
            if not self.cfg.pool_mode:
                self._enter(State.SCAN, reason="search_pattern_disabled")
                return

            if (now_s() - self.ctx.search_started_t) > self.cfg.pool_search_timeout_s:
                self.thr.stop()
                self._enter(State.ABORT, reason="pool_search_timeout")
                return

            # If a stable target is available at any point, switch to alignment.
            if target is not None:
                self.ctx.target = target
                self.thr.stop()
                self._enter(State.COARSE_ALIGN, cx=target.cx, cy=target.cy, area=target.area)
                return

            now = now_s()

            # Phase scheduler
            if self.ctx.search_phase == "":
                self.ctx.search_phase = "ROTATE"
                self.ctx.search_dir = 1
                self.ctx.search_phase_end_t = now + self.cfg.pool_search_rotate_s

            if self.ctx.search_phase == "ROTATE":
                yaw = clamp(self.cfg.pool_search_rotate_cmd * float(self.ctx.search_dir), -self.cfg.pool_max_yaw_cmd, self.cfg.pool_max_yaw_cmd)
                self.thr.set(yaw=yaw, forward=0.0)
                if now >= self.ctx.search_phase_end_t:
                    self.thr.stop()
                    self.ctx.search_phase = "PAUSE"
                    self.ctx.search_phase_end_t = now + self.cfg.pool_search_pause_s
                    self.rec.write_event({"event": "pool_search_rotate_done", "dir": int(self.ctx.search_dir)})

            elif self.ctx.search_phase == "PAUSE":
                self.thr.stop()
                if now >= self.ctx.search_phase_end_t:
                    # After each rotate+pause, flip direction and count cycles
                    self.ctx.search_cycle_count += 1
                    self.ctx.search_dir *= -1

                    # Every N cycles, do a forward step
                    if self.ctx.search_cycle_count % max(1, int(self.cfg.pool_search_cycles_per_forward)) == 0:
                        self.ctx.search_phase = "FORWARD"
                        self.ctx.search_phase_end_t = now + self.cfg.pool_search_forward_s
                        self.rec.write_event({"event": "pool_search_forward_start"})
                    else:
                        self.ctx.search_phase = "ROTATE"
                        self.ctx.search_phase_end_t = now + self.cfg.pool_search_rotate_s

            elif self.ctx.search_phase == "FORWARD":
                fwd = clamp(self.cfg.pool_search_forward_cmd, -0.4, float(self.cfg.pool_forward_cmd_limit))
                self.thr.set(yaw=0.0, forward=fwd)
                if now >= self.ctx.search_phase_end_t:
                    self.thr.stop()
                    self.ctx.search_phase = "PAUSE"
                    self.ctx.search_phase_end_t = now + self.cfg.pool_search_pause_s
                    self.rec.write_event({"event": "pool_search_forward_done"})
            return

        # SCAN: sweep servo and look for stable target blob
        if self.ctx.state == State.SCAN:
            if is_timeout(self.ctx, self.cfg.scan_timeout_s):
                self.thr.stop()
                self._enter(State.ABORT, reason="scan_timeout")
                return

            if now_s() >= self.ctx.scan_next_step_t:
                # step the scan angle
                step = self.cfg.scan_servo_step_deg * self.ctx.scan_dir
                new_deg = self.ctx.scan_deg + step
                if new_deg > self.cfg.scan_servo_max_deg:
                    new_deg = self.cfg.scan_servo_max_deg
                    self.ctx.scan_dir = -1
                if new_deg < self.cfg.scan_servo_min_deg:
                    new_deg = self.cfg.scan_servo_min_deg
                    self.ctx.scan_dir = 1
                self.ctx.scan_deg = new_deg
                try:
                    self.servo.rotate_to(self.ctx.scan_deg)
                except Exception as e:
                    self.rec.write_event({"event": "servo_error", "error": str(e)})
                self.ctx.scan_next_step_t = now_s() + self.cfg.scan_dwell_s

            if target is not None:
                self.ctx.target = target
                self.thr.stop()
                self._enter(State.COARSE_ALIGN, cx=target.cx, cy=target.cy, area=target.area)
                return

            # Keep thrusters neutral while scanning
            self.thr.stop()
            return

        # COARSE_ALIGN: yaw-only control to center blob
        if self.ctx.state == State.COARSE_ALIGN:
            if is_timeout(self.ctx, self.cfg.coarse_align_timeout_s):
                self.thr.stop()
                self._enter(State.ABORT, reason="coarse_align_timeout")
                return

            if target is None:
                self.thr.stop()
                return

            yaw = compute_yaw_cmd(target.cx, self.center_x, self.cfg.yaw_kp_coarse, self.cfg.yaw_deadband_px, self.cfg.yaw_cmd_limit)
            self.thr.set(yaw=yaw, forward=0.0)
            if yaw == 0.0:
                self.thr.stop()
                self._enter(State.FINE_ALIGN, cx=target.cx, cy=target.cy, area=target.area)
            return

        # FINE_ALIGN: smaller gain to settle before approach
        if self.ctx.state == State.FINE_ALIGN:
            if is_timeout(self.ctx, self.cfg.fine_align_timeout_s):
                self.thr.stop()
                self._enter(State.ABORT, reason="fine_align_timeout")
                return

            if target is None:
                self.thr.stop()
                return

            yaw = compute_yaw_cmd(target.cx, self.center_x, self.cfg.yaw_kp_fine, self.cfg.yaw_deadband_px, self.cfg.yaw_cmd_limit)
            self.thr.set(yaw=yaw, forward=0.0)
            if yaw == 0.0:
                self.thr.stop()
                self._enter(State.APPROACH, cx=target.cx, cy=target.cy, area=target.area)
            return

        # APPROACH: drive forward while maintaining yaw alignment
        if self.ctx.state == State.APPROACH:
            # In pool mode we avoid long straight runs:
            # - clamp forward command
            # - pulse forward for short bursts with pauses to re-align / reduce wall risk
            # - enforce a max continuous motion time (fails safe)
            if is_timeout(self.ctx, self.cfg.approach_timeout_s):
                self.thr.stop()
                self._enter(State.VERIFY, reason="approach_timeout")
                return

            if self.cfg.pool_mode:
                # Extra safety: never keep moving continuously too long in a pool
                if (now_s() - self.ctx.approach_motion_started_t) > self.cfg.pool_max_continuous_motion_s:
                    self.thr.stop()
                    if self.cfg.pool_reverse_on_timeout:
                        # brief reverse to back away from whatever we might be approaching (wall)
                        self.thr.set(yaw=0.0, forward=clamp(self.cfg.pool_reverse_cmd, -0.3, 0.0))
                        self.rec.write_event({"event": "pool_reverse", "reason": "max_continuous_motion"})
                        time.sleep(clamp(self.cfg.pool_reverse_burst_s, 0.0, 2.0))
                        self.thr.stop()
                    self._enter(State.VERIFY, reason="pool_max_continuous_motion")
                    return

            if target is None:
                self.thr.stop()
                # In pool mode, if we lose the target, pause longer to reduce drift
                if self.cfg.pool_mode:
                    time.sleep(clamp(self.cfg.pool_reacquire_stop_s, 0.0, 2.0))
                return

            yaw_limit = self.cfg.pool_max_yaw_cmd if self.cfg.pool_mode else self.cfg.yaw_cmd_limit
            yaw = compute_yaw_cmd(
                target.cx,
                self.center_x,
                self.cfg.yaw_kp_fine,
                self.cfg.yaw_deadband_px,
                yaw_limit,
            )

            if not self.cfg.pool_mode:
                # Normal behavior (open water / bench tests)
                self.thr.set(yaw=yaw, forward=self.cfg.forward_cmd)
            else:
                # Pool behavior: pulse forward, then pause
                fwd = clamp(self.cfg.forward_cmd, -1.0, 1.0)
                fwd = clamp(fwd, -0.4, float(self.cfg.pool_forward_cmd_limit))

                now = now_s()
                # If we're in a pause window, hold stop
                if now < self.ctx.approach_pause_end_t:
                    self.thr.stop()
                    return

                # Start a new burst if needed
                if now >= self.ctx.approach_burst_end_t:
                    self.ctx.approach_burst_end_t = now + clamp(self.cfg.pool_forward_burst_s, 0.05, 3.0)
                    self.ctx.approach_pause_end_t = self.ctx.approach_burst_end_t + clamp(self.cfg.pool_pause_s, 0.05, 5.0)
                    self.rec.write_event({"event": "pool_burst", "yaw": float(yaw), "forward": float(fwd)})

                # During burst: allow yaw correction + forward
                if now < self.ctx.approach_burst_end_t:
                    self.thr.set(yaw=yaw, forward=fwd)
                else:
                    self.thr.stop()

            # Heuristic: if blob is very large, assume we're close and verify
            if target.area >= max(30, self.cfg.blob_min_area_px * 4):
                self.thr.stop()
                self._enter(State.VERIFY, reason="blob_large", area=target.area)
            return

            if target is None:
                self.thr.stop()
                return

            yaw = compute_yaw_cmd(target.cx, self.center_x, self.cfg.yaw_kp_fine, self.cfg.yaw_deadband_px, self.cfg.yaw_cmd_limit)
            self.thr.set(yaw=yaw, forward=self.cfg.forward_cmd)
            # Heuristic: if blob is very large, assume we're close and verify
            if target.area >= max(30, self.cfg.blob_min_area_px * 4):
                self.thr.stop()
                self._enter(State.VERIFY, reason="blob_large", area=target.area)
            return

        # VERIFY: request N async verifications, accept if enough positive votes
        if self.ctx.state == State.VERIFY:
            # send verify requests until votes collected
            if (now_s() - self.ctx.verify_sent_t) > self.cfg.verify_timeout_s:
                self._enter(State.ABORT, reason="verify_timeout", votes_ok=self.ctx.verify_votes_ok, votes_total=self.ctx.verify_votes_total)
                return

            # Issue requests until we reach desired total votes
            while self.ctx.verify_votes_total < self.cfg.verify_votes and not self.stop_event.is_set():
                rid = self.ctx.verify_req_id * 1000 + self.ctx.verify_votes_total
                try:
                    self.verify_in.put_nowait(VerifyRequest(rid))
                    self.ctx.verify_votes_total += 1
                    self.rec.write_event({"event": "verify_request", "rid": rid})
                except queue.Full:
                    break

            # Consume results
            while True:
                try:
                    res = self.verify_out.get_nowait()
                except queue.Empty:
                    break
                if res.ok:
                    self.ctx.verify_votes_ok += 1
                self.rec.write_event({"event": "verify_result", "rid": res.req_id, "ok": res.ok, "conf": res.conf})

            if self.ctx.verify_votes_ok >= self.cfg.verify_accept_count:
                self._enter(State.DONE, votes_ok=self.ctx.verify_votes_ok, votes_total=self.ctx.verify_votes_total)
                return

            # If we have all votes but not enough OK, abort
            if self.ctx.verify_votes_total >= self.cfg.verify_votes and self.ctx.verify_votes_ok < self.cfg.verify_accept_count:
                self._enter(State.ABORT, reason="verify_rejected", votes_ok=self.ctx.verify_votes_ok, votes_total=self.ctx.verify_votes_total)
                return

            return

        # DONE/ABORT: do nothing (main loop will stop)
        if self.ctx.state in (State.DONE, State.ABORT):
            self.thr.stop()
            return


# ============================================================
# Cameras (front/left/right) and wall-guard vision (optional)
# ============================================================

def _parse_cam_device(dev: Any) -> Any:
    """Accepts int index (0,1,2), string '/dev/video0' or '0', or None."""
    if dev is None:
        return None
    if isinstance(dev, int):
        return dev
    if isinstance(dev, str):
        s = dev.strip()
        if s.startswith("TBD") or s == "":
            return None
        if s.isdigit():
            return int(s)
        return s
    return dev


class CameraStream:
    def __init__(self, name: str, device: Any, width: int, height: int, fps: int, flip: bool, recorder: "RunRecorder"):
        self.name = name
        self.device = _parse_cam_device(device)
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self.flip = bool(flip)
        self.rec = recorder
        self._cap = None
        self._thread = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest = None  # np.ndarray BGR

    def start(self) -> None:
        if self.device is None or not HAS_CV2:
            return
        self._cap = cv2.VideoCapture(self.device)
        try:
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        except Exception:
            pass
        self._thread = threading.Thread(target=self._run, name=f"cam_{self.name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = 1.0 / max(1, self.fps)
        while not self._stop.is_set():
            try:
                if self._cap is None:
                    break
                ok, frame = self._cap.read()
                if not ok or frame is None:
                    self.rec.write_event({"event": "camera_read_fail", "camera": self.name})
                    time.sleep(0.1)
                    continue
                if self.flip:
                    frame = cv2.flip(frame, 1)
                with self._lock:
                    self._latest = frame
            except Exception as e:
                self.rec.write_event({"event": "camera_error", "camera": self.name, "error": str(e)})
                time.sleep(0.1)
            time.sleep(period)

    def latest(self):
        with self._lock:
            return None if self._latest is None else self._latest.copy()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            try:
                self._cap.release()
            except Exception:
                pass


class CameraArray:
    def __init__(self, cfg: "Config", recorder: "RunRecorder"):
        self.cfg = cfg
        self.rec = recorder
        self.front = CameraStream("front", cfg.camera_front_device, cfg.camera_width, cfg.camera_height, cfg.camera_fps, cfg.camera_flip_front, recorder)
        self.left = CameraStream("left", cfg.camera_left_device, cfg.camera_width, cfg.camera_height, cfg.camera_fps, cfg.camera_flip_left, recorder)
        self.right = CameraStream("right", cfg.camera_right_device, cfg.camera_width, cfg.camera_height, cfg.camera_fps, cfg.camera_flip_right, recorder)

    def start(self) -> None:
        if not self.cfg.camera_enabled:
            return
        if not HAS_CV2:
            self.rec.write_event({"event": "camera_disabled", "reason": "opencv_not_available"})
            return
        self.front.start()
        self.left.start()
        self.right.start()
        time.sleep(clamp(self.cfg.camera_warmup_s, 0.0, 5.0))
        self.rec.write_event({"event": "camera_started"})

    def read_all(self):
        return {"front": self.front.latest(), "left": self.left.latest(), "right": self.right.latest()}

    def stop(self) -> None:
        self.front.stop()
        self.left.stop()
        self.right.stop()


class WallGuard:
    """Pool safety reflex using front/left/right cameras."""

    def __init__(self, cfg: "Config", recorder: "RunRecorder"):
        self.cfg = cfg
        self.rec = recorder
        self.prev_gray = {"front": None, "left": None, "right": None}
        self._last_check_t = 0.0

    def _prep_gray(self, frame):
        if frame is None or not HAS_CV2:
            return None
        try:
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            g = cv2.resize(g, (160, 120))
            return g
        except Exception:
            return None

    def _score_edges(self, gray) -> float:
        if gray is None or not HAS_CV2:
            return 0.0
        e = cv2.Canny(gray, 60, 140)
        return float(np.mean(e > 0)) * 100.0  # percent

    def _score_flow(self, gray, prev) -> float:
        if gray is None or prev is None or not HAS_CV2:
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 2, 15, 2, 5, 1.2, 0)
        mag, _ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag))

    def check(self, frames: dict, moving_forward: bool):
        if not self.cfg.wall_guard_enabled or not self.cfg.camera_enabled or not HAS_CV2:
            return False, {}
        now = now_s()
        hz = max(0.5, float(self.cfg.wall_guard_check_hz))
        if (now - self._last_check_t) < (1.0 / hz):
            return False, {}
        self._last_check_t = now

        method = str(self.cfg.wall_guard_method).lower().strip()
        scores = {}
        for name in ("front", "left", "right"):
            gray = self._prep_gray(frames.get(name))
            prev = self.prev_gray.get(name)
            scores[name] = self._score_edges(gray) if method == "edges" else self._score_flow(gray, prev)
            self.prev_gray[name] = gray

        # Only trigger when forward motion is active (reduces false triggers on pure rotation)
        if not moving_forward:
            return False, {"scores": scores, "moving_forward": False, "threshold": float(self.cfg.wall_guard_threshold), "method": method}

        thr = float(self.cfg.wall_guard_threshold)
        triggered = any(v >= thr for v in scores.values())
        if triggered:
            left_v = scores.get("left", 0.0)
            right_v = scores.get("right", 0.0)
            if left_v > right_v:
                yaw_dir = +1  # turn right
            elif right_v > left_v:
                yaw_dir = -1  # turn left
            else:
                yaw_dir = 1 if random.random() < 0.5 else -1
            info = {"scores": scores, "threshold": thr, "yaw_dir": yaw_dir, "method": method}
            self.rec.write_event({"event": "wall_guard_trigger", **info})
            return True, info

        return False, {"scores": scores, "moving_forward": True, "threshold": thr, "method": method}


def _wall_guard_debug_dump(recorder: "RunRecorder", cfg: "Config", ctx: "ControllerContext", frames: dict, info: dict, triggered: bool) -> None:
    """Save annotated camera frames for wall-guard tuning (best-effort)."""
    if not (cfg.wall_guard_debug_overlay and HAS_CV2):
        return
    now = now_s()
    max_hz = max(0.1, float(cfg.wall_guard_debug_max_hz))
    if (now - float(getattr(ctx, "wall_guard_last_debug_t", 0.0))) < (1.0 / max_hz):
        return
    if cfg.wall_guard_debug_trigger_only and not triggered:
        return
    ctx.wall_guard_last_debug_t = now

    scores = (info or {}).get("scores", {}) or {}
    thr = (info or {}).get("threshold", cfg.wall_guard_threshold)
    method = (info or {}).get("method", cfg.wall_guard_method)
    state = getattr(ctx, "state", "UNKNOWN")

    for name, frame in (frames or {}).items():
        if frame is None:
            continue
        try:
            img = frame.copy()
            s = float(scores.get(name, 0.0))
            text1 = f"WALL_GUARD {method}  state={state}"
            text2 = f"{name}: score={s:.3f}  thr={float(thr):.3f}  trig={triggered}"
            cv2.putText(img, text1, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(img, text2, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
            p = recorder.run_dir / f"wallguard_{int(now*1000)}_{name}.jpg"
            cv2.imwrite(str(p), img)
            recorder.write_event({"event": "wall_guard_debug_frame", "path": str(p), "camera": name, "score": s, "threshold": float(thr), "triggered": bool(triggered)})
        except Exception as e:
            recorder.write_event({"event": "wall_guard_debug_error", "camera": name, "error": str(e)})

# ============================================================
# HAL factory
# ============================================================
@dataclass
class HALBundle:
    thermal: ThermalCamera
    thrusters: Thrusters
    servo: PanServo
    verifier: Verifier
    driver_close: Optional[callable] = None

def make_hal(cfg: Config) -> HALBundle:
    if cfg.hal == "dryrun":
        thermal = DryRunThermal(cfg.dry_ambient_c, cfg.dry_blob_c, cfg.dry_noise_c, cfg.dry_drift)
        thrusters = DryRunThrusters()
        servo = DryRunServo(cfg.servo_park_deg)
        verifier = DryRunVerifier(cfg.dry_verify_accept_prob)
        return HALBundle(thermal=thermal, thrusters=thrusters, servo=servo, verifier=verifier)

    if cfg.hal == "pi_pca9685":
        driver = PiPCA9685Driver(cfg.pca_address, cfg.pca_frequency, cfg.pca_i2c_freq)
        thrusters = PiPCA9685Thrusters(
            driver=driver,
            left_ch=cfg.left_thruster_ch,
            right_ch=cfg.right_thruster_ch,
            neutral_us=cfg.esc_neutral_us,
            min_us=cfg.esc_min_us,
            max_us=cfg.esc_max_us,
            yaw_gain_us=cfg.yaw_gain_us,
            fwd_gain_us=cfg.fwd_gain_us,
            invert_left=cfg.invert_left,
            invert_right=cfg.invert_right,
            init_neutral_s=cfg.esc_init_neutral_s,
        )
        servo = PiPCA9685Servo(
            driver=driver,
            channel=cfg.servo_ch,
            min_deg=cfg.servo_min_deg,
            max_deg=cfg.servo_max_deg,
            min_us=cfg.servo_min_us,
            max_us=cfg.servo_max_us,
            park_deg=cfg.servo_park_deg,
        )
        thermal = PiMLX90640Thermal(cfg.mlx_i2c_freq, cfg.mlx_refresh_rate, cfg.mlx_retries, cfg.mlx_retry_delay_s)
        verifier = Verifier()  # stub; replace with your model/camera
        return HALBundle(thermal=thermal, thrusters=thrusters, servo=servo, verifier=verifier, driver_close=driver.close)

    raise ValueError(f"Unknown hal mode: {cfg.hal}")


# ============================================================
# Main
# ============================================================
def write_template(path: str) -> None:
    p = Path(path)
    p.write_text(json.dumps(config_template(), indent=2), encoding="utf-8")
    print(f"Wrote config template: {p}")

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="", help="Path to controller config JSON.")
    ap.add_argument("--hal", type=str, default="", help="Override HAL mode: pi_pca9685 or dryrun.")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--write-config-template", type=str, default="", help="Write a config template JSON to this path and exit.")
    args = ap.parse_args()

    if args.write_config_template:
        write_template(args.write_config_template)
        return

    if not args.config:
        raise SystemExit("Missing --config. Use --write-config-template to generate one.")

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    raw = load_json(args.config)
    if args.hal:
        raw["hal"] = args.hal

    cfg = parse_config(raw)

    recorder = RunRecorder(cfg.run_base_dir, enable_npz=cfg.save_npz, npz_every_n=cfg.npz_every_n)
    recorder.write_event({"event": "program_start", "hal": cfg.hal})

    cameras = None
    wall_guard = None
    if cfg.camera_enabled:
        cameras = CameraArray(cfg, recorder)
        cameras.start()
        wall_guard = WallGuard(cfg, recorder) if cfg.wall_guard_enabled else None

    stop_event = threading.Event()

    # Signal handling
    def _sig_handler(signum, frame):
        stop_event.set()
        recorder.write_event({"event": "signal", "signum": int(signum)})

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    hal = make_hal(cfg)

    # Ensure safe neutral at start
    try:
        hal.thrusters.stop()
        hal.servo.park()
    except Exception:
        pass

    latest = LatestValue()
    verify_in: "queue.Queue[VerifyRequest]" = queue.Queue(maxsize=20)
    verify_out: "queue.Queue[VerifyResult]" = queue.Queue(maxsize=20)

    # Threads
    capture = ThermalCaptureThread(hal.thermal, latest, stop_event, period_s=0.05, recorder=recorder)
    verifier_worker = VerifierWorker(hal.verifier, verify_in, verify_out, stop_event, recorder=recorder)

    # Controller
    controller = Controller(cfg, recorder, latest, hal.thrusters, hal.servo, verify_in, verify_out, stop_event, cameras=cameras, wall_guard=wall_guard)

    watchdog = WatchdogThread(controller.heartbeat, hal.thrusters, stop_event, timeout_s=cfg.watchdog_timeout_s, recorder=recorder)

    capture.start()
    verifier_worker.start()
    watchdog.start()

    recorder.write_event({"event": "run_start", "run_dir": str(recorder.run_dir)})

    try:
        # main loop
        while not stop_event.is_set():
            controller.step()
            if controller.ctx.state in (State.DONE, State.ABORT):
                stop_event.set()
                break
            time.sleep(0.03)
    except Exception as e:
        recorder.write_event({"event": "fatal_error", "error": str(e)})
        stop_event.set()
    finally:
        # failsafe stop
        try:
            hal.thrusters.stop()
        except Exception:
            pass
        try:
            if cameras is not None:
                cameras.stop()
        except Exception:
            pass
        try:
            hal.servo.park()
        except Exception:
            pass
        try:
            hal.thermal.close()
        except Exception:
            pass
        try:
            hal.thrusters.close()
        except Exception:
            pass
        try:
            hal.servo.close()
        except Exception:
            pass
        try:
            if hal.driver_close:
                hal.driver_close()
        except Exception:
            pass

        recorder.write_event({"event": "program_end", "final_state": controller.ctx.state})
        logging.info("Finished with state: %s (logs in %s)", controller.ctx.state, recorder.run_dir)

if __name__ == "__main__":
    main()
