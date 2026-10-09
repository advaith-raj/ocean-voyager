from __future__ import annotations
from typing import Iterable
from ocean_voyager.core.types import Pose, WorldObject, bearing_from_to, distance


def visible_objects(
    pose: Pose,
    objects: Iterable[WorldObject],
    max_range_m: float,
    fov_deg: float,
    mount_angle_deg: float = 0.0,
) -> list[tuple[WorldObject, float, float]]:
    results = []
    for obj in objects:
        d = distance(pose.x, pose.y, obj.x, obj.y)
        if d > max_range_m:
            continue
        bearing = bearing_from_to(pose, obj.x, obj.y) - mount_angle_deg
        while bearing < -180:
            bearing += 360
        while bearing >= 180:
            bearing -= 360
        if abs(bearing) <= fov_deg / 2.0:
            results.append((obj, d, bearing))
    results.sort(key=lambda item: item[1])
    return results
