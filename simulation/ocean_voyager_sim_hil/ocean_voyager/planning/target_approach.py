from __future__ import annotations
from ocean_voyager.core.types import Pose, TargetCandidate, MotionCommand, bearing_from_to, distance


def command_toward_point(pose: Pose, x: float, y: float, stop_distance_m: float = 0.35, action: str = "GOTO", reason: str = "") -> MotionCommand:
    d = distance(pose.x, pose.y, x, y)
    if d <= stop_distance_m:
        return MotionCommand(0.0, 0.0, 0.0, action="STOP", reason=reason or f"within {stop_distance_m:.2f} m of point")
    bearing = bearing_from_to(pose, x, y)
    yaw = max(-0.8, min(0.8, bearing / 55.0))
    forward = 0.30 if abs(bearing) < 50 else 0.04
    if d < 0.9:
        forward = min(forward, 0.18)
    return MotionCommand(yaw=yaw, forward=forward, duration_s=0.45, action=action, reason=reason or f"toward ({x:.2f}, {y:.2f})")


def command_toward_bearing(bearing_deg: float, forward: float = 0.22, duration_s: float = 0.45, yaw_gain_deg: float = 55.0, action: str = "BEARING_APPROACH", reason: str = "") -> MotionCommand:
    """Hardware-faithful approach: thermal gives bearing only, not range.

    The command rotates toward the bearing and moves one short step. The mission
    loop must rescan thermal and capture RGB after each step.
    """
    yaw = max(-0.75, min(0.75, bearing_deg / max(yaw_gain_deg, 1e-6)))
    # If the bearing is far off center, primarily turn with only a small creep.
    fwd = float(forward) if abs(bearing_deg) < 45.0 else min(float(forward), 0.06)
    return MotionCommand(yaw=yaw, forward=fwd, duration_s=duration_s, action=action, reason=reason or f"step toward thermal bearing {bearing_deg:.1f} deg")


def command_toward_candidate(pose: Pose, candidate: TargetCandidate, stop_distance_m: float = 0.65) -> MotionCommand:
    """Legacy coordinate approach retained for non-hardware-faithful experiments."""
    d = distance(pose.x, pose.y, candidate.estimated_x, candidate.estimated_y)
    if d <= stop_distance_m:
        return MotionCommand(0.0, 0.0, 0.0, action="VERIFY", reason=f"within verify distance of {candidate.id}")
    return command_toward_point(pose, candidate.estimated_x, candidate.estimated_y, stop_distance_m=stop_distance_m, action="APPROACH", reason=f"approach {candidate.id}")
