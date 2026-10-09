from __future__ import annotations
from ocean_voyager.core.types import MotionCommand, WallObservation


def avoidance_command(obs: WallObservation) -> MotionCommand:
    if not (obs.front_close or obs.left_close or obs.right_close):
        return MotionCommand(0, 0, 0, action="NONE", reason="wall clear")
    # Conservative default: rotate away, no forward motion.
    yaw = -0.55 if obs.right_close else 0.55
    return MotionCommand(yaw=yaw, forward=-0.05, duration_s=0.6, action="AVOID_WALL", reason=f"wall close: {obs.closest_side} {obs.closest_distance_m:.2f}m")
