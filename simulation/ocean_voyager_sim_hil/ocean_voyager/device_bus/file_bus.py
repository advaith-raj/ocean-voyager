from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from datetime import datetime
import json
import time


class NullDeviceBus:
    enabled = False

    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        return None

    def transact(self, device: str, command: str, params: dict[str, Any], response_payload: dict[str, Any] | None = None, tick: int | None = None, time_sec: float | None = None, status: str = "ok") -> dict[str, Any]:
        return {"request_id": "disabled", "device": device, "command": command, "status": status, "payload": response_payload or {}}


@dataclass
class FileDeviceBus:
    """File-based command/response trace for hardware-faithful simulation.

    Build 0.5 still runs the simulated peripheral in-process for convenience, but
    it writes the same command and response files that a separate real peripheral
    service would use later. This makes sim-vs-real comparison straightforward.
    """

    root_dir: Path
    run_mode: str = "simulated_peripherals"
    enabled: bool = True

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.commands_dir = self.root_dir / "commands" / "processed"
        self.responses_dir = self.root_dir / "responses" / "processed"
        self.traces_dir = self.root_dir / "traces"
        self.commands_dir.mkdir(parents=True, exist_ok=True)
        self.responses_dir.mkdir(parents=True, exist_ok=True)
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.traces_dir / "bus_trace.jsonl"
        self.counter = 0

    def _next_id(self, device: str) -> str:
        self.counter += 1
        return f"{device}_{self.counter:06d}"

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
            f.write("\n")
        tmp.replace(path)

    def _append_trace(self, record: dict[str, Any]) -> None:
        with self.trace_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Append a non-command event to the bus trace, e.g. startup/health messages."""
        record = {
            "wall_clock": datetime.now().isoformat(timespec="milliseconds"),
            "run_mode": self.run_mode,
            "direction": "system_event",
            "event_type": event_type,
            "payload": payload,
        }
        self._append_trace(record)

    def transact(
        self,
        device: str,
        command: str,
        params: dict[str, Any],
        response_payload: dict[str, Any] | None = None,
        tick: int | None = None,
        time_sec: float | None = None,
        status: str = "ok",
    ) -> dict[str, Any]:
        request_id = self._next_id(device)
        t0 = time.perf_counter()
        wall_clock = datetime.now().isoformat(timespec="milliseconds")
        cmd = {
            "request_id": request_id,
            "wall_clock": wall_clock,
            "tick": tick,
            "time_sec": time_sec,
            "run_mode": self.run_mode,
            "direction": "cpu_to_device",
            "device": device,
            "command": command,
            "params": params,
        }
        self._write_json(self.commands_dir / f"{request_id}.json", cmd)
        self._append_trace(cmd)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)
        resp = {
            "request_id": request_id,
            "wall_clock": datetime.now().isoformat(timespec="milliseconds"),
            "tick": tick,
            "time_sec": time_sec,
            "run_mode": self.run_mode,
            "direction": "device_to_cpu",
            "device": device,
            "command": command,
            "status": status,
            "latency_ms": latency_ms,
            "payload": response_payload or {},
        }
        self._write_json(self.responses_dir / f"{request_id}.json", resp)
        self._append_trace(resp)
        return resp
