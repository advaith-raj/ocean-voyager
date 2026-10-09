from __future__ import annotations
from typing import Any
from ocean_voyager.core.types import Pool, Pose, MotionCommand, bearing_from_to, distance


class LawnmowerCoveragePlanner:
    def __init__(self, pool: Pool, lane_spacing_m: float = 1.0, safety_offset_m: float = 0.55):
        self.pool = pool
        self.lane_spacing_m = max(0.25, lane_spacing_m)
        self.safety_offset_m = max(0.0, safety_offset_m)
        self.waypoints = self._build_waypoints()
        self.index = 0
        self.skipped_covered_waypoints = 0
        self.resume_events: list[dict[str, Any]] = []

    def _build_waypoints(self) -> list[tuple[float, float]]:
        m = min(self.pool.wall_margin_m + self.safety_offset_m, min(self.pool.length_m, self.pool.width_m) / 3.0)
        y = self.pool.y_min + m
        waypoints: list[tuple[float, float]] = []
        flip = False
        while y <= self.pool.y_max - m + 1e-6:
            left_x = self.pool.x_min + m
            right_x = self.pool.x_max - m
            if not flip:
                waypoints.append((left_x, y)); waypoints.append((right_x, y))
            else:
                waypoints.append((right_x, y)); waypoints.append((left_x, y))
            y += self.lane_spacing_m
            flip = not flip
        return waypoints

    def current_waypoint(self) -> tuple[float, float] | None:
        if self.index >= len(self.waypoints):
            return None
        return self.waypoints[self.index]

    def completed(self) -> bool:
        return self.index >= len(self.waypoints)

    def reset_for_new_pass(self) -> None:
        """Start a new lawnmower pass over the same waypoint set.

        Used by time-boxed patrol mode: once one pass completes, Voyager keeps
        surveying until the mission timer expires instead of returning immediately.
        """
        self.index = 0
        self.resume_events.append({"event": "new_lawnmower_pass"})

    def skip_covered_waypoints(self, coverage_grid: Any | None = None) -> int:
        """Advance over already-explored waypoint neighborhoods.

        This is the "blocked covered area" behavior: after a target detour, the robot may
        return to the old lawnmower waypoint, but the planner can skip waypoints whose
        neighborhood is already covered/scanned/processed in the coverage grid.
        """
        if coverage_grid is None:
            return 0
        skipped = 0
        while self.index < len(self.waypoints):
            wp = self.waypoints[self.index]
            try:
                covered = coverage_grid.is_point_explored(wp[0], wp[1])
            except Exception:
                covered = False
            if not covered:
                break
            self.resume_events.append({"event": "skip_covered_waypoint", "waypoint_index": self.index, "waypoint": [wp[0], wp[1]]})
            self.index += 1
            skipped += 1
        self.skipped_covered_waypoints += skipped
        return skipped

    def next_command(self, pose: Pose, waypoint_tolerance_m: float = 0.35, coverage_grid: Any | None = None, skip_covered: bool = True) -> MotionCommand:
        if skip_covered:
            self.skip_covered_waypoints(coverage_grid)
        wp = self.current_waypoint()
        if wp is None:
            return MotionCommand(0, 0, 0, action="STOP", reason="coverage complete")
        d = distance(pose.x, pose.y, wp[0], wp[1])
        if d <= waypoint_tolerance_m:
            self.index += 1
            if skip_covered:
                self.skip_covered_waypoints(coverage_grid)
            wp = self.current_waypoint()
            if wp is None:
                return MotionCommand(0, 0, 0, action="STOP", reason="coverage complete")
        bearing = bearing_from_to(pose, wp[0], wp[1])
        yaw = max(-0.7, min(0.7, bearing / 60.0))
        forward = 0.28 if abs(bearing) < 55 else 0.05
        return MotionCommand(yaw=yaw, forward=forward, duration_s=0.5, action="COVERAGE", reason=f"toward waypoint {self.index}: {wp}")
