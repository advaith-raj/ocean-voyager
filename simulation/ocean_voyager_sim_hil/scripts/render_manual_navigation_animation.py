from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def safe_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def load_events(path):
    rows = []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for r in reader:
            x = safe_float(r.get("x_ft"))
            y = safe_float(r.get("y_ft"))
            heading = safe_float(r.get("heading_deg"), 0.0)
            if x is None or y is None:
                continue

            r["_x"] = x
            r["_y"] = y
            r["_heading"] = heading
            rows.append(r)
    return rows


def load_confirmed(path):
    p = Path(path)
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text())
    except Exception:
        return []


def setup_axes(ax, events, confirmed, title):
    xs = [r["_x"] for r in events]
    ys = [r["_y"] for r in events]

    for obj in confirmed:
        if "estimated_x_ft" in obj and "estimated_y_ft" in obj:
            xs.append(float(obj["estimated_x_ft"]))
            ys.append(float(obj["estimated_y_ft"]))

    if not xs:
        xs = [0, 1]
        ys = [0, 1]

    pad = 0.75
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("x position (ft)")
    ax.set_ylabel("y position (ft)")
    ax.set_title(title)


def draw_static(ax, events, confirmed):
    # Full planned/actual path
    xs = [r["_x"] for r in events]
    ys = [r["_y"] for r in events]
    ax.plot(xs, ys, linestyle="--", linewidth=1.2, alpha=0.45, label="Robot path")

    # Scan points
    scan_x = [r["_x"] for r in events if r.get("phase") == "coverage_scan"]
    scan_y = [r["_y"] for r in events if r.get("phase") == "coverage_scan"]
    if scan_x:
        ax.scatter(scan_x, scan_y, s=35, marker="o", label="Coverage scans")

    # Thermal candidate points
    cand_x = [
        r["_x"] for r in events
        if str(r.get("thermal_candidate")).lower() == "true"
    ]
    cand_y = [
        r["_y"] for r in events
        if str(r.get("thermal_candidate")).lower() == "true"
    ]
    if cand_x:
        ax.scatter(cand_x, cand_y, s=70, marker="^", label="Thermal candidates")

    # Confirmed waste locations
    for i, obj in enumerate(confirmed, start=1):
        x = float(obj["estimated_x_ft"])
        y = float(obj["estimated_y_ft"])
        ax.scatter([x], [y], s=180, marker="*", label="Confirmed waste" if i == 1 else None)
        ax.text(x + 0.08, y + 0.08, f"W{i}", fontsize=10, weight="bold")

    # Event labels for confirmed decisions
    for r in events:
        if r.get("decision") == "confirmed_waste":
            ax.text(r["_x"] + 0.06, r["_y"] - 0.16, f"E{r.get('event_id')}", fontsize=8)


def render_png(events, confirmed, out_png, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_axes(ax, events, confirmed, title)
    draw_static(ax, events, confirmed)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def render_gif(events, confirmed, out_gif, title, fps):
    fig, ax = plt.subplots(figsize=(8, 6))
    setup_axes(ax, events, confirmed, title)

    path_line, = ax.plot([], [], linewidth=2.0, label="Robot traveled path")
    robot_dot = ax.scatter([], [], s=110, marker="o", label="Robot")
    heading_arrow = ax.arrow(0, 0, 0, 0)

    status_text = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    # Pre-draw confirmed waste markers as faint final markers
    for i, obj in enumerate(confirmed, start=1):
        x = float(obj["estimated_x_ft"])
        y = float(obj["estimated_y_ft"])
        ax.scatter([x], [y], s=160, marker="*", alpha=0.35)
        ax.text(x + 0.08, y + 0.08, f"W{i}", fontsize=10, weight="bold")

    thermal_candidate_scatter = ax.scatter([], [], s=80, marker="^", label="Thermal candidate")
    confirmed_scatter = ax.scatter([], [], s=180, marker="*", label="Confirmed waste")

    ax.legend(loc="best")

    xs_so_far = []
    ys_so_far = []
    cand_x = []
    cand_y = []
    conf_x = []
    conf_y = []

    def update(frame_idx):
        nonlocal heading_arrow

        r = events[frame_idx]
        x = r["_x"]
        y = r["_y"]
        heading = r["_heading"]

        xs_so_far.append(x)
        ys_so_far.append(y)

        path_line.set_data(xs_so_far, ys_so_far)
        robot_dot.set_offsets([[x, y]])

        # Remove and redraw heading arrow
        try:
            heading_arrow.remove()
        except Exception:
            pass

        dx = 0.35 * math.cos(math.radians(heading))
        dy = 0.35 * math.sin(math.radians(heading))
        heading_arrow = ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.12,
            head_length=0.14,
            length_includes_head=True,
        )

        if str(r.get("thermal_candidate")).lower() == "true":
            cand_x.append(x)
            cand_y.append(y)

        if r.get("decision") == "confirmed_waste":
            conf_x.append(x)
            conf_y.append(y)

        if cand_x:
            thermal_candidate_scatter.set_offsets(list(zip(cand_x, cand_y)))
        else:
            thermal_candidate_scatter.set_offsets(np.empty((0, 2)))

        if conf_x:
            confirmed_scatter.set_offsets(list(zip(conf_x, conf_y)))
        else:
            confirmed_scatter.set_offsets(np.empty((0, 2)))

        status_text.set_text(
            f"Event {r.get('event_id')} | {r.get('phase')}\n"
            f"pos=({x:.2f}, {y:.2f}) ft | heading={heading:.1f}°\n"
            f"thermal={r.get('thermal_candidate')} "
            f"delta={r.get('thermal_delta_c')} "
            f"bearing={r.get('thermal_bearing_deg')}\n"
            f"vision={r.get('vision_positive')} "
            f"decision={r.get('decision')}"
        )

        return path_line, robot_dot, thermal_candidate_scatter, confirmed_scatter, status_text

    anim = FuncAnimation(
        fig,
        update,
        frames=len(events),
        interval=int(1000 / fps),
        blit=False,
        repeat=True,
    )

    fig.tight_layout()
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run-dir",
        default="outputs/manual_navigation_tests/room_manual_hil_test_01",
    )
    ap.add_argument("--fps", type=int, default=1)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    events_csv = run_dir / "mission_events.csv"
    confirmed_json = run_dir / "confirmed_objects.json"

    events = load_events(events_csv)
    confirmed = load_confirmed(confirmed_json)

    if not events:
        raise SystemExit(f"No events found in {events_csv}")

    out_png = run_dir / "trajectory_map.png"
    out_gif = run_dir / "trajectory_animation.gif"

    title = "Ocean Voyager Manual-Mobile HIL Trajectory"

    render_png(events, confirmed, out_png, title)
    render_gif(events, confirmed, out_gif, title, fps=args.fps)

    print("Created:")
    print(out_png)
    print(out_gif)


if __name__ == "__main__":
    main()
