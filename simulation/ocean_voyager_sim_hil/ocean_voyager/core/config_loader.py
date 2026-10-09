from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import yaml

from ocean_voyager.core.types import ObjectType, Pool, Pose, WorldObject


class ConfigError(ValueError):
    pass


def load_yaml(path: str | Path) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"YAML file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigError(f"YAML root must be a mapping: {p}")
    return data


def load_config(config_path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(config_path)
    cfg.setdefault("runtime", {})
    cfg.setdefault("sensors", {})
    cfg.setdefault("mission", {})
    cfg.setdefault("logging", {})
    return cfg


def load_scenario(scenario_path: str | Path) -> Dict[str, Any]:
    scn = load_yaml(scenario_path)
    required = ["pool", "robot", "objects"]
    for key in required:
        if key not in scn:
            raise ConfigError(f"Scenario missing required key '{key}': {scenario_path}")
    return scn


def _parse_pool(pool_data: Dict[str, Any]) -> Pool:
    # New Build 0.5 coordinate-frame format.
    bounds = pool_data.get("bounds_relative_to_start") or pool_data.get("bounds")
    if bounds:
        x_min = float(bounds.get("x_min", 0.0))
        x_max = float(bounds["x_max"])
        y_min = float(bounds.get("y_min", 0.0))
        y_max = float(bounds["y_max"])
        return Pool(
            length_m=x_max - x_min,
            width_m=y_max - y_min,
            wall_margin_m=float(pool_data.get("wall_margin_m", 0.35)),
            x_min=x_min,
            y_min=y_min,
        )
    # Legacy format.
    return Pool(
        length_m=float(pool_data["length_m"]),
        width_m=float(pool_data["width_m"]),
        wall_margin_m=float(pool_data.get("wall_margin_m", 0.35)),
        x_min=float(pool_data.get("x_min", 0.0)),
        y_min=float(pool_data.get("y_min", 0.0)),
    )


def parse_world(scenario: Dict[str, Any]) -> Tuple[Pool, Pose, list[WorldObject]]:
    pool_data = scenario["pool"]
    robot_data = scenario["robot"]
    pool = _parse_pool(pool_data)
    pose = Pose(
        x=float(robot_data.get("start_x", 0.0)),
        y=float(robot_data.get("start_y", 0.0)),
        heading_deg=float(robot_data.get("start_heading_deg", 0.0)),
    )
    objects = []
    seen_ids = set()
    for raw in scenario.get("objects", []):
        obj_id = str(raw["id"])
        if obj_id in seen_ids:
            raise ConfigError(f"Duplicate object id in scenario: {obj_id}")
        seen_ids.add(obj_id)
        obj_type = ObjectType(raw.get("type", "unknown"))
        objects.append(
            WorldObject(
                id=obj_id,
                type=obj_type,
                x=float(raw["x"]),
                y=float(raw["y"]),
                thermal_delta_c=float(raw.get("thermal_delta_c", 0.0)),
                radius_m=float(raw.get("radius_m", 0.15)),
                image=raw.get("image"),
                orientation_deg=float(raw.get("orientation_deg", 0.0)),
                expected=str(raw.get("expected", "")),
            )
        )
    validate_world(pool, pose, objects)
    return pool, pose, objects


def validate_world(pool: Pool, pose: Pose, objects: list[WorldObject]) -> None:
    if pool.length_m <= 0 or pool.width_m <= 0:
        raise ConfigError("Pool dimensions must be positive.")
    if not pool.contains(pose.x, pose.y, margin=0.0):
        raise ConfigError(f"Robot start pose outside pool: {pose}")
    for obj in objects:
        if not pool.contains(obj.x, obj.y, margin=0.0):
            raise ConfigError(f"Object {obj.id} outside pool: ({obj.x}, {obj.y}) not in {pool.as_dict()}")
