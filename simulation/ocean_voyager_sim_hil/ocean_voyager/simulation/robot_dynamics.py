from __future__ import annotations
import math
from typing import Any
from ocean_voyager.core.types import MotionCommand, normalize_angle_deg, heading_vector
from ocean_voyager.simulation.world import SimWorld


class DifferentialThrustDynamics:
    """Hardware-faithful-ish ASV motion model.

    The model maps high-level forward/yaw commands to PWM values, then estimates
    speed, yaw rate, and pose update. It is intentionally configurable because real
    pool tests should calibrate the gains.
    """

    def __init__(
        self,
        world: SimWorld,
        max_speed_mps: float = 0.55,
        max_yaw_rate_dps: float = 95.0,
        pwm_min_us: int = 1100,
        pwm_neutral_us: int = 1500,
        pwm_max_us: int = 1900,
        forward_pwm_span_us: int | None = None,
        yaw_pwm_span_us: int | None = None,
        thrust_deadband: float = 0.06,
        acceleration_lag_s: float = 0.35,
        boat_mass_kg: float = 4.5,
        thruster_spacing_m: float = 0.38,
        battery_voltage_v: float = 12.0,
        linear_drag_coeff: float = 1.8,
        angular_drag_coeff: float = 0.9,
        **_: Any,
    ):
        self.world = world
        self.max_speed_mps = float(max_speed_mps)
        self.max_yaw_rate_dps = float(max_yaw_rate_dps)
        self.pwm_min_us = int(pwm_min_us)
        self.pwm_neutral_us = int(pwm_neutral_us)
        self.pwm_max_us = int(pwm_max_us)
        self.forward_pwm_span_us = int(forward_pwm_span_us or (self.pwm_max_us - self.pwm_neutral_us))
        self.yaw_pwm_span_us = int(yaw_pwm_span_us or (self.pwm_max_us - self.pwm_neutral_us))
        self.thrust_deadband = float(thrust_deadband)
        self.acceleration_lag_s = float(acceleration_lag_s)
        self.boat_mass_kg = float(boat_mass_kg)
        self.thruster_spacing_m = float(thruster_spacing_m)
        self.battery_voltage_v = float(battery_voltage_v)
        self.linear_drag_coeff = float(linear_drag_coeff)
        self.angular_drag_coeff = float(angular_drag_coeff)
        self.current_speed_mps = 0.0
        self.current_yaw_rate_dps = 0.0

    def command_to_pwm(self, command: MotionCommand) -> dict[str, int]:
        command.clamp()
        left_norm = command.forward - command.yaw
        right_norm = command.forward + command.yaw
        left_us = self.pwm_neutral_us + int(round(left_norm * self.forward_pwm_span_us))
        right_us = self.pwm_neutral_us + int(round(right_norm * self.forward_pwm_span_us))
        left_us = max(self.pwm_min_us, min(self.pwm_max_us, left_us))
        right_us = max(self.pwm_min_us, min(self.pwm_max_us, right_us))
        return {"left_pwm_us": left_us, "right_pwm_us": right_us}

    def _pwm_to_norm(self, pwm_us: int) -> float:
        norm = (pwm_us - self.pwm_neutral_us) / max(self.pwm_max_us - self.pwm_neutral_us, 1)
        if abs(norm) < self.thrust_deadband:
            return 0.0
        return max(-1.0, min(1.0, norm))

    def apply(self, command: MotionCommand) -> dict[str, Any]:
        command.clamp()
        start_pose = self.world.robot_pose.copy()
        pwm = self.command_to_pwm(command)
        left_norm = self._pwm_to_norm(pwm["left_pwm_us"])
        right_norm = self._pwm_to_norm(pwm["right_pwm_us"])
        forward_norm = (left_norm + right_norm) / 2.0
        yaw_norm = (right_norm - left_norm) / 2.0
        target_speed = forward_norm * self.max_speed_mps
        target_yaw_rate = yaw_norm * self.max_yaw_rate_dps
        dt = command.duration_s
        # First-order lag toward target velocity/rate.
        alpha = 1.0 if self.acceleration_lag_s <= 0 else min(1.0, dt / self.acceleration_lag_s)
        self.current_speed_mps += (target_speed - self.current_speed_mps) * alpha
        self.current_yaw_rate_dps += (target_yaw_rate - self.current_yaw_rate_dps) * alpha
        pose = self.world.robot_pose
        pose.heading_deg = normalize_angle_deg(pose.heading_deg + self.current_yaw_rate_dps * dt)
        dx, dy = heading_vector(pose.heading_deg)
        distance_m = self.current_speed_mps * dt
        pose.x += dx * distance_m
        pose.y += dy * distance_m
        self.world.pool.clamp(pose)
        self.world.step_time(dt)
        end_pose = pose.copy()
        return {
            "pwm_output": pwm,
            "normalized_thrusters": {"left": round(left_norm, 4), "right": round(right_norm, 4)},
            "estimated_motion": {
                "start_pose": start_pose.as_dict(),
                "end_pose": end_pose.as_dict(),
                "distance_m": round(abs(distance_m), 4),
                "avg_speed_mps": round(abs(self.current_speed_mps), 4),
                "yaw_rate_dps": round(self.current_yaw_rate_dps, 4),
                "duration_sec": dt,
            },
            "hardware_profile": {
                "pwm_min_us": self.pwm_min_us,
                "pwm_neutral_us": self.pwm_neutral_us,
                "pwm_max_us": self.pwm_max_us,
                "battery_voltage_v": self.battery_voltage_v,
                "boat_mass_kg": self.boat_mass_kg,
                "thruster_spacing_m": self.thruster_spacing_m,
            },
        }
