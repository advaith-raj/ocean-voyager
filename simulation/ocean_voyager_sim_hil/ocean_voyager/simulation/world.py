from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ocean_voyager.core.types import Pool, Pose, WorldObject, distance


@dataclass
class SimWorld:
    pool: Pool
    robot_pose: Pose
    objects: list[WorldObject]
    time_s: float = 0.0

    def step_time(self, dt_s: float) -> None:
        self.time_s += max(0.0, dt_s)

    def get_object(self, object_id: str) -> Optional[WorldObject]:
        return next((o for o in self.objects if o.id == object_id), None)

    def nearest_wall_distance(self) -> float:
        p = self.robot_pose
        return min(p.x - self.pool.x_min, self.pool.x_max - p.x, p.y - self.pool.y_min, self.pool.y_max - p.y)

    def distance_to_object(self, obj: WorldObject) -> float:
        return distance(self.robot_pose.x, self.robot_pose.y, obj.x, obj.y)
