from __future__ import annotations
from typing import Any
from ocean_voyager.interfaces.pan_servo import PanServoInterface
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class SimPanServo(PanServoInterface):
    def __init__(self, initial_angle_deg: float = 0.0, bus: Any | None = None):
        self._angle = initial_angle_deg
        self.bus = bus or NullDeviceBus()
        self.tick = 0
        self.time_s = 0.0

    def set_context(self, tick: int | None = None, time_s: float | None = None) -> None:
        if tick is not None:
            self.tick = tick
        if time_s is not None:
            self.time_s = time_s

    def rotate_to(self, angle_deg: float) -> None:
        self.bus.transact(
            device="pan_servo",
            command="set_angle",
            params={"angle_deg": float(angle_deg), "settle_time_sec": 0.20},
            response_payload={"angle_deg": float(angle_deg), "settled": True},
            tick=self.tick,
            time_sec=self.time_s,
        )
        self._angle = float(angle_deg)

    def current_angle(self) -> float:
        return self._angle
