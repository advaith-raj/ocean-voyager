from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path
from typing import Any
import json
import traceback


@dataclass
class DeviceStatus:
    device: str
    adapter: str
    mode: str
    source: str
    required: bool
    status: str = "unknown"              # ready, degraded, failed, skipped
    initialized: bool = False
    self_test: str = "not_run"            # pass, fail, degraded, skipped
    capabilities: dict[str, Any] = field(default_factory=dict)
    last_error: str | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def ok_for_mission(self) -> bool:
        if self.required:
            return self.status == "ready"
        return self.status in {"ready", "degraded", "skipped"}


@dataclass
class StartupReport:
    started_at: str
    completed_at: str | None
    sequence: list[str]
    overall_status: str
    all_required_ready: bool
    abort_mission: bool
    devices: list[DeviceStatus]
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["devices"] = [asdict(x) for x in self.devices]
        return d


class StartupManager:
    """Formal boot/readiness sequence for simulation and real-hardware-style runs.

    This manager deliberately separates *adapter construction* from *mission arming*.
    A device can be present but degraded; required devices must be ready before a
    mission starts unless the config explicitly allows degraded mode.
    """

    DEFAULT_SEQUENCE = [
        "SYSTEM_BOOT",
        "LOAD_CONFIG",
        "START_DEVICE_BUS",
        "INIT_PERIPHERALS",
        "HEALTH_CHECK",
        "CALIBRATION_CHECK",
        "ARM_MISSION",
        "MISSION_READY",
    ]

    def __init__(self, cfg: dict[str, Any], output_dir: str | Path, bus: Any | None = None):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.bus = bus
        self.startup_dir = self.output_dir / "startup"
        self.startup_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.startup_dir / "startup_log.txt"
        self.status_path = self.startup_dir / "startup_status.json"
        self.peripheral_path = self.startup_dir / "peripheral_status.json"
        self.trace_path = self.output_dir / "device_bus" / "traces" / "startup_trace.jsonl"
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        self._events: list[dict[str, Any]] = []

    def _log(self, line: str) -> None:
        stamp = datetime.now().isoformat(timespec="seconds")
        msg = f"[{stamp}] {line}"
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(msg + "\n")
        event = {"wall_clock": stamp, "message": line}
        self._events.append(event)
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, default=str) + "\n")
        if self.bus is not None and hasattr(self.bus, "log_event"):
            try:
                self.bus.log_event("startup", {"message": line})
            except Exception:
                pass

    def _device_required(self, device: str) -> bool:
        startup_cfg = self.cfg.get("startup", {})
        required = set(startup_cfg.get("require_devices", [
            "thermal",
            "camera_array",
            "pan_servo",
            "vision_model",
            "motors",
            "wall_guard",
            "safety",
        ]))
        return device in required

    def _source_for(self, device: str) -> str:
        sources = self.cfg.get("sources", {})
        if device == "thermal":
            return str(sources.get("thermal", "sim"))
        if device == "camera_array":
            return str(sources.get("camera", "sim"))
        if device == "motors":
            return str(sources.get("motors", "sim"))
        if device == "vision_model":
            return str(self.cfg.get("vision_model", {}).get("type", "mock"))
        if device == "pan_servo":
            return "sim_or_pwm"
        if device == "wall_guard":
            return "sim_or_safety"
        if device == "safety":
            return "software"
        return "unknown"

    def _status_from_adapter(self, device: str, adapter: Any) -> DeviceStatus:
        required = self._device_required(device)
        source = self._source_for(device)
        mode = str(self.cfg.get("runtime", {}).get("mode", "unknown"))
        status = DeviceStatus(
            device=device,
            adapter=adapter.__class__.__name__ if adapter is not None else "None",
            mode=mode,
            source=source,
            required=required,
            initialized=adapter is not None,
        )

        if adapter is None:
            status.status = "failed" if required else "skipped"
            status.self_test = "fail" if required else "skipped"
            status.last_error = "adapter is None"
            return status

        try:
            if hasattr(adapter, "health_check"):
                payload = adapter.health_check()
                status.status = str(payload.get("status", "ready"))
                status.self_test = str(payload.get("self_test", "pass" if status.status == "ready" else "degraded"))
                status.capabilities = dict(payload.get("capabilities", {}))
                status.last_error = payload.get("last_error")
                status.warnings = list(payload.get("warnings", []))
            else:
                status.status = "ready"
                status.self_test = "pass"
                status.capabilities = infer_capabilities(device, adapter)
        except Exception as exc:
            status.status = "failed" if required else "degraded"
            status.self_test = "fail"
            status.last_error = f"{type(exc).__name__}: {exc}"
            status.warnings.append(traceback.format_exc(limit=2))

        return status

    def run(self, adapters: dict[str, Any]) -> StartupReport:
        started_at = datetime.now().isoformat(timespec="seconds")
        if self.log_path.exists():
            self.log_path.unlink()
        if self.trace_path.exists():
            self.trace_path.unlink()

        self._log("[BOOT] SYSTEM_BOOT")
        self._log("[BOOT] Loading runtime configuration")
        self._log(f"[BOOT] Runtime mode: {self.cfg.get('runtime', {}).get('mode', 'unknown')}")
        self._log("[BUS] START_DEVICE_BUS")
        self._log("[PERIPHERALS] INIT_PERIPHERALS")

        device_order = [
            "pan_servo",
            "thermal",
            "camera_array",
            "vision_model",
            "motors",
            "wall_guard",
            "safety",
        ]

        statuses: list[DeviceStatus] = []
        self._log("[HEALTH] HEALTH_CHECK")
        for name in device_order:
            adapter = adapters.get(name)
            if name == "safety":
                status = self._safety_status()
            else:
                status = self._status_from_adapter(name, adapter)
            statuses.append(status)
            label = "READY" if status.status == "ready" else status.status.upper()
            self._log(f"[{name.upper()}] {label} adapter={status.adapter} required={status.required} self_test={status.self_test}")
            if status.last_error:
                self._log(f"[{name.upper()}] ERROR {status.last_error}")
            for warning in status.warnings:
                self._log(f"[{name.upper()}] WARNING {warning}")

        self._log("[CALIBRATION] CALIBRATION_CHECK")
        calibration_notes = calibration_warnings(self.cfg)
        for note in calibration_notes:
            self._log(f"[CALIBRATION] WARNING {note}")

        allow_degraded = bool(self.cfg.get("startup", {}).get("allow_degraded_mode", False))
        abort_on_failure = bool(self.cfg.get("startup", {}).get("abort_on_failure", True))
        all_required_ready = all(s.ok_for_mission or allow_degraded for s in statuses)
        abort_mission = abort_on_failure and not all_required_ready

        if abort_mission:
            overall = "failed"
            self._log("[MISSION] Required peripheral is not ready; mission not armed")
        else:
            overall = "ready" if all(s.status == "ready" or not s.required for s in statuses) else "degraded"
            self._log("[MISSION] ARM_MISSION")
            self._log("[MISSION] MISSION_READY")

        report = StartupReport(
            started_at=started_at,
            completed_at=datetime.now().isoformat(timespec="seconds"),
            sequence=self.DEFAULT_SEQUENCE,
            overall_status=overall,
            all_required_ready=all_required_ready,
            abort_mission=abort_mission,
            devices=statuses,
            notes=calibration_notes,
        )

        write_json(self.status_path, report.as_dict())
        write_json(self.peripheral_path, {"devices": [asdict(s) for s in statuses]})
        return report

    def _safety_status(self) -> DeviceStatus:
        startup_cfg = self.cfg.get("startup", {})
        motors_cfg = self.cfg.get("motors", {})
        sources = self.cfg.get("sources", {})
        required = self._device_required("safety")
        warnings: list[str] = []
        status = "ready"
        self_test = "pass"

        motor_source = str(sources.get("motors", "sim"))
        arm_real_motors = bool(motors_cfg.get("arm_real_motors", False))
        require_estop = bool(startup_cfg.get("require_emergency_stop_clear", True))
        emergency_stop_clear = bool(startup_cfg.get("emergency_stop_clear", True))

        if motor_source == "real" and not arm_real_motors:
            status = "degraded"
            self_test = "degraded"
            warnings.append("sources.motors is real but motors.arm_real_motors is false; real motor commands will be blocked/log-only where supported.")

        if require_estop and not emergency_stop_clear:
            status = "failed"
            self_test = "fail"
            warnings.append("emergency stop is not clear.")

        return DeviceStatus(
            device="safety",
            adapter="StartupSafetyChecks",
            mode=str(self.cfg.get("runtime", {}).get("mode", "unknown")),
            source="software",
            required=required,
            status=status,
            initialized=True,
            self_test=self_test,
            capabilities={
                "require_emergency_stop_clear": require_estop,
                "emergency_stop_clear": emergency_stop_clear,
                "motor_source": motor_source,
                "arm_real_motors": arm_real_motors,
                "neutral_pwm_us": motors_cfg.get("neutral_pwm_us", motors_cfg.get("pwm_neutral_us", 1500)),
            },
            warnings=warnings,
        )


def infer_capabilities(device: str, adapter: Any) -> dict[str, Any]:
    if hasattr(adapter, "capabilities"):
        try:
            return dict(adapter.capabilities())
        except Exception:
            return {}
    caps: dict[str, Any] = {}
    if device == "thermal":
        for attr in ["resolution_w", "resolution_h", "fov_deg", "fov_y_deg", "max_range_m", "threshold_delta_c"]:
            if hasattr(adapter, attr):
                caps[attr] = getattr(adapter, attr)
        caps["bearing_only"] = True
    elif device == "camera_array":
        for attr in ["frame_width", "frame_height", "fps", "fov_deg", "max_range_m", "camera_mount_angles"]:
            if hasattr(adapter, attr):
                caps[attr] = getattr(adapter, attr)
    elif device == "motors":
        caps["stop_supported"] = hasattr(adapter, "stop")
        caps["execute_supported"] = hasattr(adapter, "execute")
    elif device == "pan_servo":
        caps["current_angle_deg"] = adapter.current_angle() if hasattr(adapter, "current_angle") else None
    return caps


def calibration_warnings(cfg: dict[str, Any]) -> list[str]:
    warnings: list[str] = []
    sources = cfg.get("sources", {})
    if str(sources.get("thermal", "sim")).startswith("real"):
        warnings.append("real thermal source selected; run 1m/2m/3m/4m thermal envelope calibration before mission claims.")
    if str(sources.get("camera", "sim")).startswith("real"):
        warnings.append("real camera source selected; verify exposure, focus, frame paths, and camera IDs before mission.")
    if str(sources.get("motors", "sim")) == "real":
        warnings.append("real motor source selected; keep arm_real_motors false until neutral PWM and emergency stop are verified.")
    return warnings


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
        f.write("\n")
    tmp.replace(path)


def run_startup_sequence(cfg: dict[str, Any], output_dir: str | Path, adapters: dict[str, Any], bus: Any | None = None) -> StartupReport:
    manager = StartupManager(cfg, output_dir, bus=bus)
    return manager.run(adapters)
