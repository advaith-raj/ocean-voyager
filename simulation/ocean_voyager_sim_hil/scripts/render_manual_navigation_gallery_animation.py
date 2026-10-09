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
            r["_event_id_int"] = int(r.get("event_id")) if r.get("event_id") else -1
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


def get_image_for_event_row(r):
    p = resolve_path(r.get("vision_annotated_path"))
    if p and p.exists():
        return p
    p = resolve_path(r.get("vision_raw_path"))
    if p and p.exists():
        return p
    return None


def render_gallery_gif(events, confirmed, out_gif, title, fps):
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[2.2, 1.4],
        height_ratios=[2.0, 1.0]
    )

    ax_map = fig.add_subplot(gs[:, 0])     # full left column
    ax_main_img = fig.add_subplot(gs[0, 1])  # top-right
    ax_gallery = fig.add_subplot(gs[1, 1])   # bottom-right

    setup_map_axes(ax_map, events, confirmed, title)

    ax_main_img.axis("off")
    ax_main_img.set_title("Current approach image")

    ax_gallery.axis("off")
    ax_gallery.set_title("All approach images for current thermal event")

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

    waste_labels = {}
    for i, obj in enumerate(confirmed, start=1):
        waste_labels[int(obj["event_id"])] = f"W{i}"

    def draw_gallery(ax, event_rows, current_index_within_event):
        ax.clear()
        ax.axis("off")
        ax.set_title("All approach images for current thermal event")

        if not event_rows:
            ax.text(0.5, 0.5, "No approach images yet", ha="center", va="center")
            return

        n = len(event_rows)
        cols = min(3, n)
        rows = int(np.ceil(n / cols))

        margin = 0.02
        box_w = (1.0 - margin * (cols + 1)) / cols
        box_h = (1.0 - margin * (rows + 1)) / rows

        for idx, r in enumerate(event_rows):
            img_path = get_image_for_event_row(r)
            c = idx % cols
            rr = idx // cols

            x0 = margin + c * (box_w + margin)
            y0 = 1.0 - margin - (rr + 1) * box_h - rr * margin

            # highlight current image
            edge_color = "red" if idx == current_index_within_event else "black"
            lw = 3 if idx == current_index_within_event else 1

            rect = plt.Rectangle((x0, y0), box_w, box_h, fill=False, linewidth=lw,
                                 edgecolor=edge_color, transform=ax.transAxes, clip_on=False)
            ax.add_patch(rect)

            if img_path and img_path.exists():
                try:
                    img = plt.imread(img_path)
                    ax.imshow(
                        img,
                        extent=(x0, x0 + box_w, y0, y0 + box_h),
                        transform=ax.transAxes,
                        aspect="auto",
                        zorder=0
                    )
                except Exception:
                    ax.text(
                        x0 + box_w / 2, y0 + box_h / 2,
                        "img err",
                        ha="center", va="center",
                        fontsize=8,
                        transform=ax.transAxes
                    )
            else:
                ax.text(
                    x0 + box_w / 2, y0 + box_h / 2,
                    "no image",
                    ha="center", va="center",
                    fontsize=8,
                    transform=ax.transAxes
                )

            label = f"{idx + 1}"
            if r.get("_decision") == "confirmed_waste":
                label += " ✓"
            ax.text(
                x0 + 0.01, y0 + box_h - 0.03,
                label,
                fontsize=8,
                color="yellow",
                bbox=dict(boxstyle="round,pad=0.15", alpha=0.6),
                transform=ax.transAxes
            )

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
            thermal_scatter.set_offsets(np.array(cand_pts) if cand_pts else np.empty((0, 2)))

        if r["_decision"] == "confirmed_waste":
            conf_pts.append([x, y])
            confirmed_scatter.set_offsets(np.array(conf_pts))

            event_id = r["_event_id_int"]
            if event_id not in conf_labels_seen:
                label = waste_labels.get(event_id, f"E{event_id}")
                ax_map.text(x + 0.08, y + 0.08, label, fontsize=10, weight="bold")
                conf_labels_seen.add(event_id)
        else:
            confirmed_scatter.set_offsets(np.array(conf_pts) if conf_pts else np.empty((0, 2)))

        # ---------- right top panel: current image ----------
        ax_main_img.clear()
        ax_main_img.axis("off")
        ax_main_img.set_title("Current approach image")

        # ---------- right bottom panel: all images in current thermal event ----------
        current_event_id = r["_event_id_int"]

        event_rows = [
            e for e in events[:frame_idx + 1]
            if e["_event_id_int"] == current_event_id and e.get("phase") == "approach"
        ]

        current_img_idx = max(0, len(event_rows) - 1)

        # show current event image if available
        if event_rows:
            current_row = event_rows[-1]
            img_path = get_image_for_event_row(current_row)
            if img_path and img_path.exists():
                try:
                    img = plt.imread(img_path)
                    ax_main_img.imshow(img)
                    ax_main_img.set_title(
                        f"Current approach image\nEvent {current_event_id} | Step {len(event_rows)}"
                    )
                except Exception:
                    ax_main_img.text(0.5, 0.5, "Image load error", ha="center", va="center")
            else:
                ax_main_img.text(0.5, 0.5, "No approach image yet", ha="center", va="center")
        else:
            ax_main_img.text(0.5, 0.5, "No approach image yet", ha="center", va="center")

        draw_gallery(ax_gallery, event_rows, current_img_idx)

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

    out_gif = run_dir / "trajectory_gallery_animation.gif"
    title = "Ocean Voyager Manual-Mobile HIL Trajectory"

    render_gallery_gif(events, confirmed, out_gif, title, fps=args.fps)

    print("Created:")
    print(out_gif)


if __name__ == "__main__":
    main()
