from __future__ import annotations
from pathlib import Path
from typing import Optional
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_voyager.core.types import ObjectType, Pose
from ocean_voyager.simulation.world import SimWorld


def plot_world(world: SimWorld, output_path: str | Path, trajectory: Optional[list[Pose]] = None, candidates: Optional[list[dict]] = None) -> None:
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_xlim(world.pool.x_min, world.pool.x_max)
    ax.set_ylim(world.pool.y_min, world.pool.y_max)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Ocean Voyager Simulated Pool World")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.add_patch(plt.Rectangle((world.pool.x_min, world.pool.y_min), world.pool.length_m, world.pool.width_m, fill=False, linewidth=2))
    margin = world.pool.wall_margin_m
    ax.add_patch(plt.Rectangle((world.pool.x_min + margin, world.pool.y_min + margin), world.pool.length_m - 2*margin, world.pool.width_m - 2*margin, fill=False, linestyle="--", linewidth=1))
    marker_map = {ObjectType.BOTTLE: "o", ObjectType.SEAWEED: "s", ObjectType.GLARE: "*", ObjectType.UNKNOWN: "x"}
    for obj in world.objects:
        ax.scatter([obj.x], [obj.y], marker=marker_map.get(obj.type, "x"), s=80)
        ax.text(obj.x + 0.05, obj.y + 0.05, f"{obj.id}\n{obj.type.value}", fontsize=8)
    if trajectory:
        xs = [pt.x for pt in trajectory]
        ys = [pt.y for pt in trajectory]
        ax.plot(xs, ys, linewidth=1)
    pose = world.robot_pose
    ax.scatter([pose.x], [pose.y], marker="^", s=120)
    hx = math.cos(math.radians(pose.heading_deg)) * 0.4
    hy = math.sin(math.radians(pose.heading_deg)) * 0.4
    ax.arrow(pose.x, pose.y, hx, hy, head_width=0.12, length_includes_head=True)
    if candidates:
        for c in candidates:
            ax.scatter([c.get("estimated_x")], [c.get("estimated_y")], marker="+", s=100)
    fig.tight_layout()
    fig.savefig(p, dpi=150)
    plt.close(fig)


def render_camera_view(camera_id: str, visible: list, output_path: str | Path, wall_observation: Optional[dict] = None) -> str:
    """Generate a lightweight camera-like debug image for sim testing."""
    p = Path(output_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.set_xlim(0, 640)
    ax.set_ylim(480, 0)
    ax.set_title(f"Sim camera: {camera_id}")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.add_patch(plt.Rectangle((0, 0), 640, 480, fill=False, linewidth=2))
    ax.text(10, 20, "synthetic water/pool view", fontsize=9)
    if wall_observation and wall_observation.get("closest_distance_m") is not None:
        d = wall_observation["closest_distance_m"]
        if d < 1.0:
            ax.add_patch(plt.Rectangle((0, 0), 640, 55, alpha=0.25))
            ax.text(10, 45, f"wall cue close: {d:.2f} m", fontsize=9)
    for obj in visible:
        x_center = 320 + obj.bearing_deg * 5
        size = max(20, min(160, obj.apparent_size * 130))
        y_center = 260 + min(120, obj.distance_m * 10)
        ax.add_patch(plt.Rectangle((x_center - size/2, y_center - size/2), size, size, fill=False, linewidth=2))
        ax.text(x_center - size/2, y_center - size/2 - 5, f"{obj.type.value} {obj.distance_m:.1f}m", fontsize=8)
    fig.tight_layout()
    fig.savefig(p, dpi=100)
    plt.close(fig)
    return str(p)
