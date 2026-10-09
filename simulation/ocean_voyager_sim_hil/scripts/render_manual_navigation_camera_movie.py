#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def as_bool(v):
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def resolve_path(value, run_dir):
    if value in (None, ""):
        return None

    p = Path(str(value))

    candidates = []

    if p.is_absolute():
        candidates.append(p)

    candidates += [
        PROJECT_ROOT / p,
        p,
        run_dir / p,
        run_dir / p.name,
        run_dir / "paired_frames" / p.name,
        run_dir / "thermal_preview" / p.name,
    ]

    if len(p.parts) >= 2:
        candidates.append(run_dir / p.parts[-2] / p.name)

    for c in candidates:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            pass

    return None


def load_events(run_dir):
    csv_path = run_dir / "mission_events.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"mission_events.csv not found: {csv_path}")

    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def get_rgb_path(row, run_dir):
    # Prefer YOLO-annotated image.
    for key in ("vision_annotated_path", "vision_raw_path"):
        p = resolve_path(row.get(key), run_dir)
        if p is not None:
            return p

    return None


def get_thermal_path(row, run_dir):
    return resolve_path(row.get("thermal_preview_path"), run_dir)


def rgb_purpose(row):
    phase = str(row.get("phase") or "")

    if phase == "pre_move_vision_guard":
        return "PRE-MOVE COLLISION GUARD"

    if phase == "approach":
        return "DEBRIS VERIFICATION + COLLISION GUARD"

    return "RGB / YOLO OBSERVATION"



def is_nonwaste_rejection(row):
    """
    True only once the controller has actually concluded that a
    thermal candidate is not waste.

    Do NOT mark an unresolved candidate as rejected.
    """
    decision = str(row.get("decision") or "").strip().lower()
    rejected_status = str(row.get("rejected_status") or "").strip().lower()

    known_rejection_decisions = {
        "thermal_candidate_but_vision_rejected",
        "rejected",
        "non_waste",
        "non-waste",
        "not_waste",
        "not-waste",
    }

    if decision in known_rejection_decisions:
        return True

    if "reject" in decision and "duplicate" not in decision:
        return True

    if rejected_status:
        if (
            "reject" in rejected_status
            or "non_waste" in rejected_status
            or "non-waste" in rejected_status
        ):
            return True

    return False


def collision_text(row):
    risk = row.get("collision_risk")

    if risk in (None, ""):
        return "Collision: n/a"

    return "Collision: RISK" if as_bool(risk) else "Collision: clear"


def detection_text(row):
    n = row.get("vision_num_detections", "")
    label = row.get("vision_best_label", "")
    conf = row.get("vision_best_conf", "")

    if n in ("", None):
        return "YOLO: observation"

    try:
        n_int = int(float(n))
    except Exception:
        n_int = 0

    if n_int == 0:
        return "YOLO: no detection"

    result = f"YOLO: {n_int} detection"
    if n_int != 1:
        result += "s"

    if label:
        result += f" | {label}"

    if conf not in ("", None):
        try:
            result += f" {float(conf):.2f}"
        except Exception:
            pass

    return result


def build_camera_frames(events, run_dir):
    """
    Build one chronological frame for every logged RGB observation.

    The thermal panel uses:
      1. thermal from the same row when available;
      2. otherwise the latest thermal observation preceding that RGB image.

    This avoids falsely claiming synchronization while still showing
    the thermal context that immediately preceded the camera observation.
    """

    frames = []
    latest_thermal = None
    latest_thermal_row = None

    for i, row in enumerate(events):
        thermal = get_thermal_path(row, run_dir)

        if thermal is not None:
            latest_thermal = thermal
            latest_thermal_row = row

        rgb = get_rgb_path(row, run_dir)

        if rgb is None:
            continue

        same_row_thermal = get_thermal_path(row, run_dir)

        if same_row_thermal is not None:
            thermal_for_rgb = same_row_thermal
            thermal_label = "Corresponding Thermal Observation"
            thermal_row = row
        else:
            thermal_for_rgb = latest_thermal
            thermal_label = "Latest Thermal Observation"
            thermal_row = latest_thermal_row

        frames.append({
            "event_index": i,
            "row": row,
            "rgb": rgb,
            "thermal": thermal_for_rgb,
            "thermal_label": thermal_label,
            "thermal_row": thermal_row,
        })

    return frames


def draw_image(ax, path, empty_text):
    ax.clear()
    ax.axis("off")

    if path is None:
        ax.text(
            0.5, 0.5,
            empty_text,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    try:
        with Image.open(path) as im:
            img = im.convert("RGB")
            ax.imshow(img)
    except Exception as e:
        ax.text(
            0.5, 0.5,
            f"Image load error\n{type(e).__name__}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )



def draw_rgb_state_overlay(ax, row):
    """
    Add mission-state visualization directly on the RGB panel.

    Visual vocabulary:
      RED    = collision danger
      ORANGE = investigated thermal candidate rejected as non-waste

    We intentionally do not invent an object bounding box when exact
    pixel coordinates are unavailable in mission_events.csv.
    """

    collision = as_bool(row.get("collision_risk"))
    rejected = is_nonwaste_rejection(row)

    # --------------------------------------------------------
    # ORANGE: NON-WASTE / REJECTED
    #
    # Use four corner brackets so this remains visually distinct
    # from a green YOLO detection box.
    # --------------------------------------------------------
    if rejected:

        x0, x1 = 0.025, 0.975
        y0, y1 = 0.035, 0.965
        arm = 0.075

        bracket_kw = dict(
            transform=ax.transAxes,
            linewidth=5.0,
            solid_capstyle="round",
            zorder=80,
            color="darkorange",
        )

        # top-left
        ax.plot([x0, x0 + arm], [y1, y1], **bracket_kw)
        ax.plot([x0, x0], [y1, y1 - arm], **bracket_kw)

        # top-right
        ax.plot([x1 - arm, x1], [y1, y1], **bracket_kw)
        ax.plot([x1, x1], [y1, y1 - arm], **bracket_kw)

        # bottom-left
        ax.plot([x0, x0 + arm], [y0, y0], **bracket_kw)
        ax.plot([x0, x0], [y0, y0 + arm], **bracket_kw)

        # bottom-right
        ax.plot([x1 - arm, x1], [y0, y0], **bracket_kw)
        ax.plot([x1, x1], [y0, y0 + arm], **bracket_kw)

        ax.text(
            0.5,
            0.055,
            "NON-WASTE — REJECTED",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
            color="black",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="darkorange",
                edgecolor="black",
                linewidth=1.5,
                alpha=0.92,
            ),
            zorder=90,
        )

    # --------------------------------------------------------
    # RED: COLLISION RISK
    #
    # Collision gets visual priority over rejection because it is
    # an immediate safety/navigation condition.
    # --------------------------------------------------------
    if collision:

        border = plt.Rectangle(
            (0.012, 0.012),
            0.976,
            0.976,
            transform=ax.transAxes,
            fill=False,
            linewidth=7.0,
            edgecolor="red",
            zorder=100,
            clip_on=False,
        )
        ax.add_patch(border)

        reason = str(row.get("collision_reason") or "").strip()

        label = "⚠  COLLISION RISK  ⚠"

        if reason:
            # Keep the overlay readable; full reason remains in CSV/status.
            reason_short = reason.replace("_", " ")

            if len(reason_short) > 55:
                reason_short = reason_short[:52] + "..."

            label += "\n" + reason_short

        ax.text(
            0.5,
            0.955,
            label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=15,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor="red",
                edgecolor="white",
                linewidth=2.0,
                alpha=0.90,
            ),
            zorder=110,
        )


def render_movie(frames, out_gif, fps, hold):
    if not frames:
        raise RuntimeError("No RGB camera frames found in mission_events.csv")

    # RGB intentionally dominates the movie.
    fig = plt.figure(figsize=(15, 8))

    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[3.2, 1.25],
        height_ratios=[5.0, 1.0],
    )

    ax_rgb = fig.add_subplot(gs[0, 0])
    ax_thermal = fig.add_subplot(gs[0, 1])
    ax_status = fig.add_subplot(gs[1, :])

    ax_status.axis("off")

    display_indices = []

    for i in range(len(frames)):
        display_indices.extend([i] * max(1, hold))

    def update(display_idx):
        item = frames[display_indices[display_idx]]
        row = item["row"]

        draw_image(
            ax_rgb,
            item["rgb"],
            "No RGB image",
        )

        # Make important controller decisions visible directly in
        # the camera view rather than requiring the viewer to read
        # only the status strip.
        draw_rgb_state_overlay(ax_rgb, row)

        ax_rgb.set_title(
            f"Voyager Camera — {rgb_purpose(row)}",
            fontsize=13,
            fontweight="bold",
        )

        draw_image(
            ax_thermal,
            item["thermal"],
            "No thermal observation",
        )

        ax_thermal.set_title(
            item["thermal_label"],
            fontsize=11,
            fontweight="bold",
        )

        ax_status.clear()
        ax_status.axis("off")

        event_id = row.get("event_id", "")
        phase = row.get("phase", "")

        x = row.get("x_ft", "")
        y = row.get("y_ft", "")
        heading = row.get("heading_deg", "")

        decision = row.get("decision", "")
        instruction = row.get("instruction", "")

        line1 = (
            f"Event {event_id}  |  {rgb_purpose(row)}"
        )

        line2 = (
            f"Position: ({x}, {y}) ft  |  "
            f"Heading: {heading}°  |  "
            f"{detection_text(row)}  |  "
            f"{collision_text(row)}"
        )

        line3_parts = []

        if phase:
            line3_parts.append(f"Phase: {phase}")

        if decision:
            line3_parts.append(f"Decision: {decision}")

        if is_nonwaste_rejection(row):
            line3_parts.append("Classification: NON-WASTE / REJECTED")

        if instruction:
            line3_parts.append(f"Action: {instruction}")

        line3 = "  |  ".join(line3_parts)

        ax_status.text(
            0.01,
            0.76,
            line1,
            fontsize=12,
            fontweight="bold",
            transform=ax_status.transAxes,
        )

        ax_status.text(
            0.01,
            0.43,
            line2,
            fontsize=10,
            transform=ax_status.transAxes,
        )

        ax_status.text(
            0.01,
            0.10,
            line3,
            fontsize=9,
            transform=ax_status.transAxes,
        )

        return []

    anim = FuncAnimation(
        fig,
        update,
        frames=len(display_indices),
        interval=int(1000 / fps),
        blit=False,
        repeat=True,
    )

    fig.tight_layout()

    writer = PillowWriter(fps=fps)

    anim.save(
        out_gif,
        writer=writer,
    )

    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create a camera-centric Ocean Voyager mission movie "
            "with RGB/YOLO dominant and thermal context alongside."
        )
    )

    ap.add_argument(
        "--run-dir",
        required=True,
        help="Manual-navigation run directory",
    )

    ap.add_argument(
        "--fps",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--hold",
        type=int,
        default=2,
        help="Number of movie frames to hold each RGB observation",
    )

    ap.add_argument(
        "--output",
        default=None,
        help="Output GIF path",
    )

    args = ap.parse_args()

    run_dir = Path(args.run_dir)

    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir

    run_dir = run_dir.resolve()

    events = load_events(run_dir)
    frames = build_camera_frames(events, run_dir)

    if args.output:
        out_gif = Path(args.output)
        if not out_gif.is_absolute():
            out_gif = PROJECT_ROOT / out_gif
    else:
        out_gif = run_dir / "voyager_camera_thermal_movie.gif"

    print("=== Ocean Voyager Camera + Thermal Movie ===")
    print("Run:", run_dir)
    print("Mission events:", len(events))
    print("RGB observations:", len(frames))
    print("Output:", out_gif)
    print()

    for i, item in enumerate(frames[:5], start=1):
        row = item["row"]
        print(
            f"[{i}] event={row.get('event_id')} "
            f"phase={row.get('phase')} "
            f"rgb={item['rgb'].name} "
            f"thermal="
            f"{item['thermal'].name if item['thermal'] else 'none'} "
            f"({item['thermal_label']})"
        )

    if len(frames) > 5:
        print(f"... plus {len(frames) - 5} more RGB observations")

    print()
    print("Rendering...")

    render_movie(
        frames,
        out_gif,
        fps=args.fps,
        hold=args.hold,
    )

    print()
    print("DONE")
    print("Saved:", out_gif)


if __name__ == "__main__":
    main()
