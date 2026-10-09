from __future__ import annotations

from typing import Any

from ocean_voyager.interfaces.motors import MotorExecutorInterface
from ocean_voyager.core.types import MotionCommand
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class RealPCA9685MotorExecutor(MotorExecutorInterface):
    """PCA9685 -> ESC/thruster adapter with safe default behavior.

    By default, this adapter is NOT armed. It will report readiness of the
    software path but will not send real PWM unless `arm_real_motors: true` is set.
    """

    def __init__(self, cfg: dict[str, Any], bus: Any | None = None):
        self.cfg = cfg
        self.bus = bus or NullDeviceBus()
        self.arm_real_motors = bool(cfg.get("arm_real_motors", False))
        self.left_channel = int(cfg.get("left_channel", 0))
        self.right_channel = int(cfg.get("right_channel", 1))
        self.neutral_pwm_us = int(cfg.get("neutral_pwm_us", cfg.get("pwm_neutral_us", 1500)))
        self.min_pwm_us = int(cfg.get("min_pwm_us", cfg.get("pwm_min_us", 1100)))
        self.max_pwm_us = int(cfg.get("max_pwm_us", cfg.get("pwm_max_us", 1900)))
        self.available = False
        self.last_error: str | None = None
        self.warnings: list[str] = []
        self.pca = None
        if self.arm_real_motors:
            try:
                import board  # type: ignore
                import busio  # type: ignore
                from adafruit_pca9685 import PCA9685  # type: ignore
                i2c = busio.I2C(board.SCL, board.SDA)
                self.pca = PCA9685(i2c)
                self.pca.frequency = int(cfg.get("pwm_frequency_hz", 50))
                self.available = True
                self._send_neutral()
            except Exception as exc:
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.warnings.append("PCA9685 library/device not available or failed to initialize.")
        else:
            self.available = True
            self.warnings.append("Real motor adapter is in safe log-only mode because arm_real_motors is false.")

    def capabilities(self) -> dict[str, Any]:
        return {
            "controller": "PCA9685",
            "left_channel": self.left_channel,
            "right_channel": self.right_channel,
            "neutral_pwm_us": self.neutral_pwm_us,
            "min_pwm_us": self.min_pwm_us,
            "max_pwm_us": self.max_pwm_us,
            "arm_real_motors": self.arm_real_motors,
            "safe_default": "log_only_unless_armed",
        }

    def health_check(self) -> dict[str, Any]:
        if self.arm_real_motors and not self.available:
            return {
                "status": "failed",
                "self_test": "fail",
                "capabilities": self.capabilities(),
                "last_error": self.last_error,
                "warnings": self.warnings,
            }
        status = "ready" if self.arm_real_motors else "degraded"
        return {
            "status": status,
            "self_test": "pass" if self.arm_real_motors else "degraded",
            "capabilities": self.capabilities(),
            "last_error": self.last_error,
            "warnings": self.warnings,
        }

    def _pwm_us_to_duty_cycle(self, pwm_us: int) -> int:
        # 50 Hz period = 20,000 us. 16-bit duty cycle.
        pwm_us = max(self.min_pwm_us, min(self.max_pwm_us, int(pwm_us)))
        return int((pwm_us / 20000.0) * 0xFFFF)

    def _send_channel(self, channel: int, pwm_us: int) -> None:
        if self.pca is None:
            return
        self.pca.channels[channel].duty_cycle = self._pwm_us_to_duty_cycle(pwm_us)

    def _send_neutral(self) -> None:
        self._send_channel(self.left_channel, self.neutral_pwm_us)
        self._send_channel(self.right_channel, self.neutral_pwm_us)

    def execute(self, command: MotionCommand) -> None:
        command.clamp()
        # Safe placeholder mapping. Actual pool motion calibration should refine this.
        left_pwm = self.neutral_pwm_us + int(250 * command.forward - 150 * command.yaw)
        right_pwm = self.neutral_pwm_us + int(250 * command.forward + 150 * command.yaw)
        left_pwm = max(self.min_pwm_us, min(self.max_pwm_us, left_pwm))
        right_pwm = max(self.min_pwm_us, min(self.max_pwm_us, right_pwm))

        if self.arm_real_motors and self.available:
            self._send_channel(self.left_channel, left_pwm)
            self._send_channel(self.right_channel, right_pwm)
            executed = True
            reason = "real_pwm_sent"
        else:
            executed = False
            reason = "safe_log_only_real_motors_not_armed"

        self.bus.transact(
            device="motors",
            command="set_pwm_for_duration",
            params={**command.as_dict(), "left_pwm_us": left_pwm, "right_pwm_us": right_pwm},
            response_payload={"executed": executed, "reason": reason},
            time_sec=None,
        )

    def stop(self) -> None:
        if self.arm_real_motors and self.available:
            self._send_neutral()
            executed = True
        else:
            executed = False
        self.bus.transact(
            device="motors",
            command="stop",
            params={"left_pwm_us": self.neutral_pwm_us, "right_pwm_us": self.neutral_pwm_us},
            response_payload={"executed": executed, "neutral_pwm_sent": executed},
            time_sec=None,
        )
