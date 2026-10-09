from pathlib import Path
import argparse
import csv
import json
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def safe_float(v, default=None):
    try:
        if v is None or v == "":
            return default
        return float(v)
    except Exception:
        return default


def as_bool(v):
    return str(v).strip().lower() == "true"


def resolve_path(p):
    if not p:
        return None
    q = Path(p)
    if q.is_absolute():
        return q
    return PROJECT_ROOT / q


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
            r["_thermal_candidate"] = as_bool(r.get("thermal_candidate"))
            r["_vision_positive"] = as_bool(r.get("vision_positive"))
            r["_decision"] = (r.get("decision") or "").strip()
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


def setup_map_axes(ax, events, confirmed, title):
    xs = [r["_x"] for r in events]
    ys = [r["_y"] for r in events]

    for obj in confirmed:
        if "estimated_x_ft" in obj and "estimated_y_ft" in obj:
            xs.append(float(obj["estimated_x_ft"]))
            ys.append(float(obj["estimated_y_ft"]))

    if not xs:
        xs = [0, 1]
        ys = [0, 1]

    pad = 0.8
    ax.set_xlim(min(xs) - pad, max(xs) + pad)
    ax.set_ylim(min(ys) - pad, max(ys) + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.35)
    ax.set_xlabel("x position (ft)")
    ax.set_ylabel("y position (ft)")
    ax.set_title(title)


def render_static_png(events, confirmed, out_png, title):
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.2])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])

    setup_map_axes(ax_map, events, confirmed, title)

    path_x = [r["_x"] for r in events]
    path_y = [r["_y"] for r in events]
    ax_map.plot(path_x, path_y, linestyle="--", linewidth=1.5, alpha=0.6, label="Robot path")

    scan_x = [r["_x"] for r in events if r.get("phase") == "coverage_scan"]
    scan_y = [r["_y"] for r in events if r.get("phase") == "coverage_scan"]
    if scan_x:
        ax_map.scatter(scan_x, scan_y, s=35, marker="o", label="Coverage scans")

    cand_x = [r["_x"] for r in events if r["_thermal_candidate"]]
    cand_y = [r["_y"] for r in events if r["_thermal_candidate"]]
    if cand_x:
        ax_map.scatter(cand_x, cand_y, s=70, marker="^", label="Thermal candidates")

    for i, obj in enumerate(confirmed, start=1):
        x = float(obj["estimated_x_ft"])
        y = float(obj["estimated_y_ft"])
        ax_map.scatter([x], [y], s=180, marker="*", label="Confirmed waste" if i == 1 else None)
        ax_map.text(x + 0.08, y + 0.08, f"W{i}", fontsize=10, weight="bold")

    ax_map.legend(loc="best")

    # Show the last confirmed image on the static panel if available
    ax_img.axis("off")
    shown = False
    for r in reversed(events):
        if r["_decision"] == "confirmed_waste":
            img_path = resolve_path(r.get("vision_annotated_path"))
            if img_path and img_path.exists():
                img = plt.imread(img_path)
                ax_img.imshow(img)
                ax_img.set_title(f"Confirmed waste image\nEvent {r.get('event_id')}")
                shown = True
                break

    if not shown:
        ax_img.text(0.5, 0.5, "No confirmed image found", ha="center", va="center")
        ax_img.set_title("Confirmed waste image")

    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def render_rich_gif(events, confirmed, out_gif, title, fps):
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.2, 1.2])

    ax_map = fig.add_subplot(gs[0, 0])
    ax_img = fig.add_subplot(gs[0, 1])

    setup_map_axes(ax_map, events, confirmed, title)
    ax_img.axis("off")
    ax_img.set_title("Latest confirmed waste image")

    path_line, = ax_map.plot([], [], linewidth=2.0, label="Robot traveled path")
    robot_scatter = ax_map.scatter([], [], s=110, marker="o", label="Robot")
    thermal_scatter = ax_map.scatter([], [], s=70, marker="^", label="Thermal candidates")
    confirmed_scatter = ax_map.scatter([], [], s=180, marker="*", label="Confirmed waste")

    status_text = ax_map.text(
        0.02, 0.98, "",
        transform=ax_map.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    ax_map.legend(loc="best")

    xs_so_far = []
    ys_so_far = []
    cand_pts = []
    conf_pts = []
    conf_labels_seen = set()
    arrow_ref = {"artist": None}
    last_confirmed_img = {"img": None, "title": "Latest confirmed waste image"}

    # map event_id -> W1/W2/... for labels
    waste_labels = {}
    for i, obj in enumerate(confirmed, start=1):
        waste_labels[int(obj["event_id"])] = f"W{i}"

    def update(frame_idx):
        r = events[frame_idx]
        x = r["_x"]
        y = r["_y"]
        heading = r["_heading"]

        xs_so_far.append(x)
        ys_so_far.append(y)
        path_line.set_data(xs_so_far, ys_so_far)
        robot_scatter.set_offsets(np.array([[x, y]]))

        if arrow_ref["artist"] is not None:
            try:
                arrow_ref["artist"].remove()
            except Exception:
                pass

        dx = 0.35 * math.cos(math.radians(heading))
        dy = 0.35 * math.sin(math.radians(heading))
        arrow_ref["artist"] = ax_map.arrow(
            x, y, dx, dy,
            head_width=0.12,
            head_length=0.14,
            length_includes_head=True,
        )

        if r["_thermal_candidate"]:
            cand_pts.append([x, y])
            thermal_scatter.set_offsets(np.array(cand_pts))
        else:
            if cand_pts:
                thermal_scatter.set_offsets(np.array(cand_pts))
            else:
                thermal_scatter.set_offsets(np.empty((0, 2)))

        if r["_decision"] == "confirmed_waste":
            conf_pts.append([x, y])
            confirmed_scatter.set_offsets(np.array(conf_pts))

            event_id = int(r.get("event_id"))
            if event_id not in conf_labels_seen:
                label = waste_labels.get(event_id, f"E{event_id}")
                ax_map.text(x + 0.08, y + 0.08, label, fontsize=10, weight="bold")
                conf_labels_seen.add(event_id)

            img_path = resolve_path(r.get("vision_annotated_path"))
            if img_path and img_path.exists():
                try:
                    last_confirmed_img["img"] = plt.imread(img_path)
                    last_confirmed_img["title"] = f"Confirmed waste image\nEvent {event_id}"
                except Exception:
                    pass
        else:
            if conf_pts:
                confirmed_scatter.set_offsets(np.array(conf_pts))
            else:
                confirmed_scatter.set_offsets(np.empty((0, 2)))

        ax_img.clear()
        ax_img.axis("off")
        ax_img.set_title(last_confirmed_img["title"])
        if last_confirmed_img["img"] is not None:
            ax_img.imshow(last_confirmed_img["img"])
        else:
            ax_img.text(0.5, 0.5, "No confirmed waste image yet", ha="center", va="center")

        status_text.set_text(
            f"Event {r.get('event_id')} | {r.get('phase')}\n"
            f"pos=({x:.2f}, {y:.2f}) ft | heading={heading:.1f}°\n"
            f"thermal={r.get('thermal_candidate')} "
            f"delta={r.get('thermal_delta_c')} "
            f"bearing={r.get('thermal_bearing_deg')}\n"
            f"vision={r.get('vision_positive')} "
            f"decision={r.get('decision')}"
        )

        return []

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

    out_png = run_dir / "trajectory_rich_map.png"
    out_gif = run_dir / "trajectory_rich_animation.gif"

    title = "Ocean Voyager Manual-Mobile HIL Trajectory"

    render_static_png(events, confirmed, out_png, title)
    render_rich_gif(events, confirmed, out_gif, title, fps=args.fps)

    print("Created:")
    print(out_png)
    print(out_gif)


if __name__ == "__main__":
    main()
