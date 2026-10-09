from __future__ import annotations
from typing import Any
from ocean_voyager.core.types import WallObservation
from ocean_voyager.interfaces.wall_guard import WallGuardInterface
from ocean_voyager.simulation.world import SimWorld


class SimWallGuard(WallGuardInterface):
    def __init__(self, world: SimWorld, cfg: dict[str, Any]):
        self.world = world
        self.threshold_m = float(cfg.get("threshold_m", 0.55))
        self.caution_m = float(cfg.get("caution_m", 0.9))

    def check(self) -> WallObservation:
        p = self.world.robot_pose
        distances = {
            "left_pool_wall": p.x - self.world.pool.x_min,
            "right_pool_wall": self.world.pool.x_max - p.x,
            "bottom_pool_wall": p.y - self.world.pool.y_min,
            "top_pool_wall": self.world.pool.y_max - p.y,
        }
        side, d = min(distances.items(), key=lambda kv: kv[1])
        close = d <= self.threshold_m
        score = max(0.0, (self.caution_m - d) / max(self.caution_m, 1e-6))
        return WallObservation(front_close=close, left_close=close and "left" in side, right_close=close and "right" in side, closest_side=side, closest_distance_m=d, score=score)
