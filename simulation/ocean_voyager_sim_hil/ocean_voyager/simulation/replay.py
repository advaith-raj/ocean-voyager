from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from ocean_voyager.simulation.world import SimWorld


def read_trajectory(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            row["step"] = int(float(row["step"]))
            row["time_s"] = float(row["time_s"])
            row["x"] = float(row["x"])
            row["y"] = float(row["y"])
            row["heading_deg"] = float(row["heading_deg"])
            rows.append(row)
    return rows


def read_coverage_grid(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            def b(v: str) -> bool:
                return str(v).lower() in {"true", "1", "yes"}
            row["center_x"] = float(row["center_x"])
            row["center_y"] = float(row["center_y"])
            for k in ["wall_margin", "path_covered", "thermal_scanned", "vision_checked", "explored"]:
                row[k] = b(row.get(k, ""))
            rows.append(row)
    return rows


def read_thermal_scans(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _object_style(obj_type: str, status: str) -> tuple[str, str, str]:
    if status == "confirmed":
        return "o", "green", "confirmed waste"
    if status == "rejected":
        return "x", "red", "rejected"
    if status == "lost":
        return "?", "purple", "lost"
    if obj_type == "seaweed":
        return "s", "darkorange", "seaweed"
    if obj_type == "glare":
        return "*", "gold", "glare"
    return "o", "blue", obj_type


def _draw_coverage_matplotlib(ax, coverage_rows: list[dict[str, Any]], cell_size: float = 0.35) -> None:
    if not coverage_rows:
        return
    # Draw explored cells as a faint background. Wall-margin cells are light gray.
    for row in coverage_rows:
        x = row["center_x"] - cell_size / 2
        y = row["center_y"] - cell_size / 2
        if row.get("wall_margin"):
            color, alpha = "lightgray", 0.22
        elif row.get("processed_object_id"):
            color, alpha = "limegreen" if row.get("processed_status") == "confirmed" else "salmon", 0.40
        elif row.get("vision_checked") and row.get("path_covered"):
            color, alpha = "plum", 0.34
        elif row.get("vision_checked"):
            color, alpha = "violet", 0.22
        elif row.get("path_covered") and row.get("thermal_scanned"):
            color, alpha = "lightskyblue", 0.34
        elif row.get("thermal_scanned"):
            color, alpha = "khaki", 0.24
        elif row.get("path_covered"):
            color, alpha = "lightsteelblue", 0.22
        else:
            continue
        ax.add_patch(Rectangle((x, y), cell_size, cell_size, facecolor=color, edgecolor="none", alpha=alpha))


def write_static_replay(world: SimWorld, output_path: str | Path, trajectory_rows: list[dict[str, Any]] | None = None, title: str = "Ocean Voyager Mission Replay", coverage_rows: list[dict[str, Any]] | None = None, thermal_scans: list[dict[str, Any]] | None = None) -> None:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    trajectory_rows = trajectory_rows or []
    coverage_rows = coverage_rows or []
    thermal_scans = thermal_scans or []
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(world.pool.x_min - 0.3, world.pool.x_max + 0.3)
    ax.set_ylim(world.pool.y_min - 0.3, world.pool.y_max + 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("x relative to start (m)")
    ax.set_ylabel("y relative to start (m)")
    _draw_coverage_matplotlib(ax, coverage_rows)
    ax.add_patch(plt.Rectangle((world.pool.x_min, world.pool.y_min), world.pool.length_m, world.pool.width_m, fill=False, linewidth=2, edgecolor="black"))
    m = world.pool.wall_margin_m
    ax.add_patch(plt.Rectangle((world.pool.x_min + m, world.pool.y_min + m), world.pool.length_m - 2*m, world.pool.width_m - 2*m, fill=False, linestyle="--", linewidth=1, edgecolor="gray"))
    ax.scatter([0], [0], marker="*", s=180, c="black", zorder=5)
    ax.text(0.05, 0.05, "HOME / START (0,0)", fontsize=8)

    # Thermal scan trail: gray dots = no candidate; orange dots/lines = candidate found.
    no_scan_x, no_scan_y, hit_x, hit_y = [], [], [], []
    for s in thermal_scans:
        pose = s.get("pose") or {}
        if s.get("candidate_found"):
            if s.get("candidate_estimated_x") is not None and s.get("candidate_estimated_y") is not None:
                hit_x.append(float(s["candidate_estimated_x"])); hit_y.append(float(s["candidate_estimated_y"]))
        else:
            if pose.get("x") is not None and pose.get("y") is not None:
                no_scan_x.append(float(pose["x"])); no_scan_y.append(float(pose["y"]))
    if no_scan_x:
        ax.scatter(no_scan_x, no_scan_y, s=8, c="gray", alpha=0.22, label="thermal scan: no candidate")
    if hit_x:
        ax.scatter(hit_x, hit_y, s=45, c="orange", marker="D", alpha=0.85, label="thermal candidate estimate")

    for obj in world.objects:
        marker, color, _label = _object_style(obj.type.value, obj.status.value)
        ax.scatter([obj.x], [obj.y], marker=marker, s=100, c=color, edgecolors="black", zorder=6)
        ax.text(obj.x + 0.05, obj.y + 0.05, f"{obj.id}\n{obj.type.value}\n({obj.x:.2f}, {obj.y:.2f}) m\n{obj.status.value}", fontsize=8)
    if trajectory_rows:
        xs = [r["x"] for r in trajectory_rows]
        ys = [r["y"] for r in trajectory_rows]
        ax.plot(xs, ys, linewidth=1.8, c="black", label="voyager path", zorder=4)
        ax.scatter([xs[0]], [ys[0]], marker="^", s=100, c="black", zorder=5)
        ax.scatter([xs[-1]], [ys[-1]], marker="D", s=80, c="black", zorder=5)
        pose = trajectory_rows[-1]
        hx = math.cos(math.radians(pose["heading_deg"])) * 0.35
        hy = math.sin(math.radians(pose["heading_deg"])) * 0.35
        ax.arrow(pose["x"], pose["y"], hx, hy, head_width=0.10, length_includes_head=True, color="black")
    ax.text(world.pool.x_min + 0.05, world.pool.y_max + 0.08, "overlay: blue=physical path, yellow=thermal scan, purple=RGB camera captured, green/red=processed zone", fontsize=8)
    if no_scan_x or hit_x or trajectory_rows:
        ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def write_animation(world: SimWorld, output_path: str | Path, trajectory_rows: list[dict[str, Any]], title: str = "Ocean Voyager Mission Animation", max_frames: int = 60, coverage_rows: list[dict[str, Any]] | None = None, thermal_scans: list[dict[str, Any]] | None = None) -> None:
    """Fast GIF writer using PIL, designed for automated scenario replay."""
    from PIL import Image, ImageDraw
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(trajectory_rows) < 2:
        return
    coverage_rows = coverage_rows or []
    thermal_scans = thermal_scans or []
    stride = max(1, math.ceil(len(trajectory_rows) / max_frames))
    frames_rows = trajectory_rows[::stride]
    if frames_rows[-1] is not trajectory_rows[-1]:
        frames_rows.append(trajectory_rows[-1])
    W, H = 980, 560
    pad = 60
    x0, x1 = world.pool.x_min, world.pool.x_max
    y0, y1 = world.pool.y_min, world.pool.y_max
    sx = (W - 2 * pad) / max(x1 - x0, 1e-6)
    sy = (H - 2 * pad) / max(y1 - y0, 1e-6)
    scale = min(sx, sy)
    cell_size_px = max(2, int(0.35 * scale))

    def xy(x: float, y: float) -> tuple[int, int]:
        px = int(pad + (x - x0) * scale)
        py = int(H - pad - (y - y0) * scale)
        return px, py

    def rect(a: tuple[int, int], b: tuple[int, int]) -> list[int]:
        return [min(a[0], b[0]), min(a[1], b[1]), max(a[0], b[0]), max(a[1], b[1])]

    def draw_cell(d: ImageDraw.ImageDraw, row: dict[str, Any]) -> None:
        x, y = xy(row["center_x"], row["center_y"])
        if row.get("wall_margin"):
            fill = (226, 226, 226)
        elif row.get("processed_object_id"):
            fill = (188, 235, 188) if row.get("processed_status") == "confirmed" else (255, 205, 205)
        elif row.get("vision_checked") and row.get("path_covered"):
            fill = (225, 205, 235)
        elif row.get("vision_checked"):
            fill = (240, 218, 242)
        elif row.get("path_covered") and row.get("thermal_scanned"):
            fill = (205, 232, 245)
        elif row.get("thermal_scanned"):
            fill = (246, 238, 180)
        elif row.get("path_covered"):
            fill = (222, 232, 245)
        else:
            return
        d.rectangle([x-cell_size_px//2, y-cell_size_px//2, x+cell_size_px//2, y+cell_size_px//2], fill=fill)

    # Pre-group thermal scans by time for animation.
    thermal_by_time = sorted(thermal_scans, key=lambda s: float(s.get("time_s", 0)))
    images: list[Image.Image] = []
    path_points: list[tuple[int, int]] = []
    for idx, row in enumerate(frames_rows):
        img = Image.new("RGB", (W, H), "white")
        d = ImageDraw.Draw(img)
        d.text((18, 12), title, fill="black")
        d.text((18, 32), f"step {row['step']} | t={row['time_s']:.1f}s | {row['state']} | x={row['x']:.2f}, y={row['y']:.2f}", fill="black")
        d.text((18, 52), "Blue=physical path, yellow=thermal scan, purple=RGB camera captured, orange=bearing ray", fill="black")

        for cg in coverage_rows:
            draw_cell(d, cg)
        d.rectangle(rect(xy(world.pool.x_min, world.pool.y_min), xy(world.pool.x_max, world.pool.y_max)), outline="black", width=3)
        m = world.pool.wall_margin_m
        d.rectangle(rect(xy(world.pool.x_min + m, world.pool.y_min + m), xy(world.pool.x_max - m, world.pool.y_max - m)), outline="gray", width=1)
        hx, hy = xy(0, 0)
        d.ellipse([hx - 6, hy - 6, hx + 6, hy + 6], fill="black")
        d.text((hx + 8, hy - 8), "HOME", fill="black")

        # Draw thermal history up to current time.
        current_t = float(row["time_s"])
        recent_candidates = 0
        recent_no_candidates = 0
        for s in thermal_by_time:
            if float(s.get("time_s", 0)) > current_t:
                break
            pose = s.get("pose") or {}
            if s.get("candidate_found") and s.get("candidate_estimated_x") is not None:
                cx, cy = xy(float(s["candidate_estimated_x"]), float(s["candidate_estimated_y"]))
                d.polygon([(cx, cy - 6), (cx + 6, cy), (cx, cy + 6), (cx - 6, cy)], outline="orange", fill="orange")
                recent_candidates += 1
            elif pose.get("x") is not None:
                px, py = xy(float(pose["x"]), float(pose["y"]))
                d.ellipse([px-2, py-2, px+2, py+2], fill=(170,170,170))
                recent_no_candidates += 1
        d.text((W-275, 28), f"thermal scans so far: {recent_no_candidates + recent_candidates}", fill="black")
        d.text((W-275, 45), f"thermal candidate hits: {recent_candidates}", fill="black")

        for obj in world.objects:
            ox, oy = xy(obj.x, obj.y)
            label = f"{obj.id}\n{obj.type.value}\n({obj.x:.2f},{obj.y:.2f})m"
            if obj.status.value == "confirmed":
                d.ellipse([ox - 8, oy - 8, ox + 8, oy + 8], outline="green", width=4)
            elif obj.status.value == "rejected":
                d.line([ox - 9, oy - 9, ox + 9, oy + 9], fill="red", width=4)
                d.line([ox - 9, oy + 9, ox + 9, oy - 9], fill="red", width=4)
            elif obj.type.value == "seaweed":
                d.rectangle([ox - 6, oy - 6, ox + 6, oy + 6], outline="darkorange", width=2)
            else:
                d.ellipse([ox - 6, oy - 6, ox + 6, oy + 6], outline="blue", width=2)
            d.text((ox + 8, oy + 4), label, fill="black")

        path_points.append(xy(row["x"], row["y"]))
        if len(path_points) > 1:
            d.line(path_points, fill="black", width=2)
        rx, ry = path_points[-1]
        heading = math.radians(row["heading_deg"])
        tip = (int(rx + 16 * math.cos(heading)), int(ry - 16 * math.sin(heading)))
        d.ellipse([rx - 6, ry - 6, rx + 6, ry + 6], fill="black")
        d.line([(rx, ry), tip], fill="black", width=3)
        images.append(img)
    images[0].save(out, save_all=True, append_images=images[1:], duration=120, loop=0)


def build_replay_outputs(world: SimWorld, output_dir: str | Path, title: str | None = None) -> dict[str, str]:
    out = Path(output_dir)
    rows = read_trajectory(out / "trajectory.csv")
    coverage_rows = read_coverage_grid(out / "coverage_grid_final.csv")
    thermal_scans = read_thermal_scans(out / "thermal_scans.json")
    static_path = out / "world_replay.png"
    gif_path = out / "mission_animation.gif"
    title = title or "Ocean Voyager Mission Replay"
    write_static_replay(world, static_path, rows, title=title, coverage_rows=coverage_rows, thermal_scans=thermal_scans)
    if rows:
        write_animation(world, gif_path, rows, title=title, coverage_rows=coverage_rows, thermal_scans=thermal_scans)
    return {"world_replay": str(static_path), "mission_animation": str(gif_path) if gif_path.exists() else ""}


def compare_bus_traces(sim_trace: str | Path, real_trace: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    def load(path: str | Path) -> list[dict[str, Any]]:
        p = Path(path)
        if not p.exists():
            return []
        return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    sim = load(sim_trace)
    real = load(real_trace)
    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        out: dict[str, Any] = {"count": len(rows), "by_device": {}}
        for r in rows:
            dev = r.get("device", "unknown")
            out["by_device"].setdefault(dev, 0)
            out["by_device"][dev] += 1
        return out
    report = {"sim": summary(sim), "real": summary(real), "note": "Detailed sim-vs-real numeric calibration is added after real hardware traces are collected."}
    if output_path:
        Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
