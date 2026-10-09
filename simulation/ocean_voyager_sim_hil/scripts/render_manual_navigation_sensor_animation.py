from pathlib import Path
import argparse
import csv
import json
import re
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image
from thermal_anomaly_test import analyze_frame


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Cache raw MLX90640 analysis so repeated GIF hold frames
# do not repeatedly load/analyze the same .npy file.
THERMAL_FRAME_CACHE = {}




RUN_DIR_GLOBAL = None

def resolve_run_path(path_value):
    """
    Resolve paths saved in mission_events.csv.

    Handles:
    - absolute paths
    - paths relative to project root
    - paths relative to run_dir
    - basename paths inside thermal_preview / paired_frames
    """
    if path_value is None:
        return Path("__missing__")

    text = str(path_value).strip()
    if not text:
        return Path("__missing__")

    p = Path(text)
    candidates = []

    if p.is_absolute():
        candidates.append(p)
    else:
        candidates.append(p)
        candidates.append(Path.cwd() / p)

        if RUN_DIR_GLOBAL is not None:
            run_dir = Path(RUN_DIR_GLOBAL)
            candidates.append(run_dir / p)
            candidates.append(run_dir / p.name)

            parts = set(p.parts)
            if "thermal_preview" in parts or "thermal" in p.name:
                candidates.append(run_dir / "thermal_preview" / p.name)
            if "paired_frames" in parts or "camera" in p.name:
                candidates.append(run_dir / "paired_frames" / p.name)

    for c in candidates:
        try:
            if c.exists():
                return c
        except Exception:
            pass

    return p


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


def load_rejected(path):
    """Load confirmed non-waste / rejected target records."""
    p = Path(path)
    if not p.exists():
        return []
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []



def object_x_for_render(obj):
    """Support both old and v2/v2.3 confirmed object schemas."""
    for key in [
        "estimated_x_ft",
        "estimated_object_x_ft",
        "raw_estimated_object_x_ft",
        "robot_x_ft",
        "x_ft",
    ]:
        v = obj.get(key)
        if v not in (None, ""):
            return float(v)
    return 0.0


def object_y_for_render(obj):
    """Support both old and v2/v2.3 confirmed object schemas."""
    for key in [
        "estimated_y_ft",
        "estimated_object_y_ft",
        "raw_estimated_object_y_ft",
        "robot_y_ft",
        "y_ft",
    ]:
        v = obj.get(key)
        if v not in (None, ""):
            return float(v)
    return 0.0



def object_event_id_for_render(obj, fallback_index=0):
    """
    Support v1 and v2/v2.3 object schemas.

    Old renderer expected obj["event_id"].
    v2.3 confirmed/rejected objects may only have object_id plus image paths.
    Prefer event_id, then parse it from paths like approach_0037_02_camera.jpg,
    then fall back to object_id or loop index.
    """
    for key in ["event_id"]:
        v = obj.get(key)
        if v not in (None, ""):
            try:
                return int(float(v))
            except Exception:
                pass

    for key in [
        "vision_raw_path",
        "vision_annotated_path",
        "thermal_preview_path",
    ]:
        v = str(obj.get(key, ""))
        m = re.search(r"(?:coverage|approach|premove)_(\d+)", v)
        if m:
            return int(m.group(1))

    for key in ["object_id", "matched_object_id"]:
        v = obj.get(key)
        if v not in (None, ""):
            try:
                return int(float(v))
            except Exception:
                pass

    return int(fallback_index)



def resolve_existing_image_path(path_value):
    """
    Resolve image paths recorded in manual-navigation CSV files.

    CSV paths may be:
      - absolute paths
      - project-root-relative paths such as
        outputs/manual_navigation_tests/RUN/thermal_preview/file.png
      - run-relative paths such as thermal_preview/file.png
      - bare filenames
    """
    if path_value in (None, ""):
        return Path("")

    p = Path(str(path_value))
    candidates = []

    if p.is_absolute():
        candidates.append(p)

    # Most current CSV entries are relative to repository/project root.
    candidates.append(PROJECT_ROOT / p)

    # Also allow current-working-directory resolution.
    candidates.append(p)

    if RUN_DIR_GLOBAL is not None:
        rd = Path(RUN_DIR_GLOBAL)

        candidates.append(rd / p)
        candidates.append(rd / p.name)
        candidates.append(rd / "thermal_preview" / p.name)
        candidates.append(rd / "paired_frames" / p.name)

        if len(p.parts) >= 2:
            candidates.append(rd / p.parts[-2] / p.name)

    # Remove duplicates while preserving order.
    seen = set()
    unique = []
    for c in candidates:
        try:
            key = str(c.resolve(strict=False))
        except Exception:
            key = str(c)

        if key not in seen:
            seen.add(key)
            unique.append(c)

    for c in unique:
        try:
            if c.exists() and c.is_file():
                return c
        except Exception:
            pass

    # Return the most likely path so the caller's error output is useful.
    return unique[0] if unique else p

def setup_map(ax, events, confirmed):
    """
    Draw the real 8x8 HIL room while allowing the plotting canvas to
    expand if logged coordinates went outside the permitted boundary.

    This is intentionally a visualization/debugging behavior only.
    It does NOT redefine the legal operating area.
    """

    xs = [float(r["_x"]) for r in events]
    ys = [float(r["_y"]) for r in events]

    for obj in confirmed:
        try:
            xs.append(float(object_x_for_render(obj)))
            ys.append(float(object_y_for_render(obj)))
        except Exception:
            pass

    ROOM_X_MIN = 0.0
    ROOM_Y_MIN = 0.0
    ROOM_X_MAX = 8.0
    ROOM_Y_MAX = 8.0

    pad = 0.6

    data_x_min = min(xs) if xs else ROOM_X_MIN
    data_x_max = max(xs) if xs else ROOM_X_MAX
    data_y_min = min(ys) if ys else ROOM_Y_MIN
    data_y_max = max(ys) if ys else ROOM_Y_MAX

    plot_x_min = min(ROOM_X_MIN, data_x_min)
    plot_x_max = max(ROOM_X_MAX, data_x_max)
    plot_y_min = min(ROOM_Y_MIN, data_y_min)
    plot_y_max = max(ROOM_Y_MAX, data_y_max)

    # Keep the normal room corner clean when there are no negative values.
    if plot_x_min >= 0:
        plot_x_min = 0.0
    else:
        plot_x_min -= pad

    if plot_y_min >= 0:
        plot_y_min = 0.0
    else:
        plot_y_min -= pad

    if plot_x_max > ROOM_X_MAX:
        plot_x_max += pad

    if plot_y_max > ROOM_Y_MAX:
        plot_y_max += pad

    ax.set_xlim(plot_x_min, plot_x_max)
    ax.set_ylim(plot_y_min, plot_y_max)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)

    # The rectangle is the TRUE permitted HIL room.
    room_rect = plt.Rectangle(
        (ROOM_X_MIN, ROOM_Y_MIN),
        ROOM_X_MAX - ROOM_X_MIN,
        ROOM_Y_MAX - ROOM_Y_MIN,
        fill=False,
        linewidth=2.5,
        linestyle="-",
        zorder=2,
    )
    ax.add_patch(room_rect)

    ax.text(
        0.15,
        7.82,
        "VALID 8 x 8 ft HIL AREA",
        fontsize=8,
        weight="bold",
        va="top",
    )

    # Shade outside-area conceptually with an explicit warning rather
    # than pretending an expanded plot means an expanded room.
    out_of_bounds = any(
        (
            float(r["_x"]) < ROOM_X_MIN
            or float(r["_x"]) > ROOM_X_MAX
            or float(r["_y"]) < ROOM_Y_MIN
            or float(r["_y"]) > ROOM_Y_MAX
        )
        for r in events
    )

    if out_of_bounds:
        ax.text(
            0.5,
            1.015,
            "Recorded trajectory contains OUT-OF-BOUNDS coordinates",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=9,
            weight="bold",
        )

    ax.set_xlabel("X position (ft)")
    ax.set_ylabel("Y position (ft)")
    ax.set_title("Ocean Voyager Manual HIL Navigation")


def get_rgb_image_path(r):
    p = resolve_path(r.get("vision_annotated_path"))
    if p and p.exists():
        return p
    p = resolve_path(r.get("vision_raw_path"))
    if p and p.exists():
        return p
    return None


def get_thermal_image_path(r):
    p = resolve_run_path(r.get("thermal_preview_path"))
    if p and p.exists():
        return p
    return None


def draw_image_panel(ax, path, title, empty_text):
    ax.clear()
    ax.axis("off")
    ax.set_title(title)

    if path and path.exists():
        try:
            img = plt.imread(path)
            ax.imshow(img)
        except Exception:
            ax.text(0.5, 0.5, "Image load error", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, empty_text, ha="center", va="center")



def draw_thermal_panel(ax, path, title, empty_text, row=None):
    """
    Render the current MLX90640 observation.

    For old runs, recover the true target centroid from the saved .npy
    thermal frame using the same analyze_frame() logic used during HIL.

    Results are cached so repeated GIF hold frames do not repeatedly
    load and analyze the same thermal file.
    """

    ax.clear()
    ax.set_title(title)

    if not path:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            empty_text,
            ha="center",
            va="center",
        )
        return

    try:
        resolved = resolve_existing_image_path(path)

        npy_path = (
            resolved.parent.parent
            / "thermal_frames"
            / f"{resolved.stem}.npy"
        )

        arr = None
        analysis = None

        mode = "cold"

        if row is not None:
            mode_value = row.get("thermal_mode")
            if mode_value not in (None, ""):
                mode = str(mode_value)

        # ----------------------------------------------------
        # CACHE:
        # each raw thermal frame + mode is analyzed only once.
        # ----------------------------------------------------
        cache_key = (str(npy_path.resolve()), mode)

        if npy_path.exists():

            if cache_key in THERMAL_FRAME_CACHE:
                arr, analysis = THERMAL_FRAME_CACHE[cache_key]

            else:
                arr = np.load(npy_path)

                analysis = analyze_frame(
                    arr,
                    mode=mode,
                    threshold_delta_c=2.0,
                    min_blob_pixels=6,
                    fov_x_deg=55.0,
                )

                THERMAL_FRAME_CACHE[cache_key] = (
                    arr,
                    analysis,
                )

            ax.imshow(
                arr,
                cmap="inferno",
                origin="lower",
                interpolation="nearest",
                aspect="auto",
            )

            # Hide pixel-axis labels in mission animation.
            # The target marker communicates location directly.

        else:
            # Fallback for runs that only contain preview PNGs.
            img = Image.open(resolved)
            img.load()
            img = img.convert("RGB")

            ax.imshow(img)
            ax.axis("off")

        thermal_candidate = False

        if row is not None:
            thermal_candidate = bool(
                row.get("_thermal_candidate", False)
            )

        if not thermal_candidate:
            return

        # ----------------------------------------------------
        # TRUE TARGET CENTROID
        # ----------------------------------------------------
        cx = None
        cy = None

        # New runs may log centroid directly.
        if row is not None:
            try:
                vx = row.get("thermal_centroid_x")
                vy = row.get("thermal_centroid_y")

                if vx not in (None, "") and vy not in (None, ""):
                    cx = float(vx)
                    cy = float(vy)

            except Exception:
                cx = None
                cy = None

        # Old runs: recover centroid from raw .npy frame.
        if (cx is None or cy is None) and analysis is not None:
            cx = float(analysis["centroid_x"])
            cy = float(analysis["centroid_y"])

        # Never show a fake approximate target.
        if cx is None or cy is None or arr is None:
            ax.text(
                0.02,
                0.98,
                "THERMAL TARGET\ncentroid unavailable",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    alpha=0.75,
                ),
            )
            return

        # ----------------------------------------------------
        # Large marker centered on exact MLX90640 centroid.
        # ----------------------------------------------------
        circle = plt.Circle(
            (cx, cy),
            radius=2.0,
            fill=False,
            linewidth=3.0,
            edgecolor="yellow",
            zorder=20,
        )
        ax.add_patch(circle)

        circle2 = plt.Circle(
            (cx, cy),
            radius=2.7,
            fill=False,
            linewidth=1.5,
            edgecolor="magenta",
            zorder=21,
        )
        ax.add_patch(circle2)

        # Annotation is placed on the opposite side so the arrow
        # terminates precisely at the detected thermal centroid.
        if cx > 15.5:
            text_x = 1.0
            ha = "left"
        else:
            text_x = 30.0
            ha = "right"

        text_y = 22.0

        ax.annotate(
            "THERMAL\nTARGET",
            xy=(cx, cy),
            xycoords="data",
            xytext=(text_x, text_y),
            textcoords="data",
            ha=ha,
            va="top",
            fontsize=11,
            fontweight="bold",
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.25",
                facecolor="black",
                edgecolor="yellow",
                alpha=0.80,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="magenta",
                linewidth=2.5,
            ),
            zorder=30,
        )

    except Exception as e:
        print(
            f"[THERMAL RENDER ERROR] "
            f"path={path!r} "
            f"error={type(e).__name__}: {e}"
        )

        ax.axis("off")

        ax.text(
            0.5,
            0.5,
            "Thermal image load error",
            ha="center",
            va="center",
        )

def draw_gallery(ax, event_rows, current_index):
    ax.clear()
    ax.axis("off")
    ax.set_title("RGB / YOLO — Approach Verification History")

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
        img_path = get_rgb_image_path(r)

        c = idx % cols
        rr = idx // cols

        x0 = margin + c * (box_w + margin)
        y0 = 1.0 - margin - (rr + 1) * box_h - rr * margin

        edge_color = "red" if idx == current_index else "black"
        lw = 3 if idx == current_index else 1

        rect = plt.Rectangle(
            (x0, y0),
            box_w,
            box_h,
            fill=False,
            linewidth=lw,
            edgecolor=edge_color,
            transform=ax.transAxes,
            clip_on=False,
        )
        ax.add_patch(rect)

        if img_path and img_path.exists():
            try:
                img = plt.imread(img_path)
                ax.imshow(
                    img,
                    extent=(x0, x0 + box_w, y0, y0 + box_h),
                    transform=ax.transAxes,
                    aspect="auto",
                )
            except Exception:
                ax.text(x0 + box_w / 2, y0 + box_h / 2, "img err",
                        ha="center", va="center", fontsize=8, transform=ax.transAxes)
        else:
            ax.text(x0 + box_w / 2, y0 + box_h / 2, "no image",
                    ha="center", va="center", fontsize=8, transform=ax.transAxes)

        label = f"{idx + 1}"
        if r["_decision"] == "confirmed_waste":
            label += " ✓"

        ax.text(
            x0 + 0.01,
            y0 + box_h - 0.04,
            label,
            fontsize=8,
            color="yellow",
            bbox=dict(boxstyle="round,pad=0.15", alpha=0.6),
            transform=ax.transAxes,
        )


def build_display_frame_indices(events, base_hold, candidate_hold, confirmed_hold):
    """
    Build animation display stages.

    Each entry is:
        (event_index, stage)

    stage is one of:
        normal
        thermal_only
        paired

    A row containing both a thermal frame and RGB observation is
    deliberately rendered twice:
        thermal first
        then thermal + RGB together
    """
    indices = []

    for i, r in enumerate(events):
        hold = base_hold

        if r["_thermal_candidate"]:
            hold = max(hold, candidate_hold)

        if r["_decision"] == "confirmed_waste":
            hold = max(hold, confirmed_hold)

        has_thermal = bool(r.get("thermal_preview_path"))
        has_rgb = bool(
            r.get("vision_annotated_path")
            or r.get("vision_raw_path")
        )

        if has_thermal and has_rgb:
            # First allow the viewer to see what thermal saw.
            thermal_hold = max(
                1,
                candidate_hold if r["_thermal_candidate"] else base_hold,
            )

            indices.extend(
                [(i, "thermal_only")] * thermal_hold
            )

            # Then show the paired RGB observation alongside exactly
            # the same thermal frame.
            indices.extend(
                [(i, "paired")] * hold
            )

        else:
            indices.extend(
                [(i, "normal")] * hold
            )

    return indices

def rgb_purpose_for_row(row):
    """
    Explain why the current RGB image exists.
    """
    if row is None:
        return "CURRENT OBSERVATION"

    phase = str(row.get("phase") or "")

    if phase == "pre_move_vision_guard":
        return "PRE-MOVE COLLISION GUARD"

    if phase == "approach":
        return "DEBRIS VERIFICATION + COLLISION GUARD"

    if phase == "coverage_scan":
        return "COVERAGE / SAFETY OBSERVATION"

    return "CURRENT OBSERVATION"


def render_animation(events, confirmed, rejected, out_gif, fps, base_hold, candidate_hold, confirmed_hold, fast=False):
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[2.1, 1.35],
        height_ratios=[1.2, 1.2, 1.0],
    )

    ax_map = fig.add_subplot(gs[:, 0])

    # Sensor panels follow actual control-flow order:
    # thermal sensing first, RGB/YOLO second.
    ax_thermal = fig.add_subplot(gs[0, 1])
    ax_rgb = fig.add_subplot(gs[1, 1])
    ax_gallery = fig.add_subplot(gs[2, 1])

    setup_map(ax_map, events, confirmed)

    path_line, = ax_map.plot([], [], linewidth=2.0, label="Robot path")
    robot_dot = ax_map.scatter([], [], s=110, marker="o", label="Robot")
    thermal_pts = ax_map.scatter([], [], s=80, marker="^", label="Thermal candidate")
    # Where Voyager was physically located when waste was confirmed.
    confirmed_from_pts = ax_map.scatter(
        [],
        [],
        s=120,
        marker="X",
        label="Waste confirmed from here",
        zorder=10,
    )

    # Estimated physical waste location.
    confirmed_pts = ax_map.scatter(
        [],
        [],
        s=180,
        marker="*",
        label="Estimated waste location",
        zorder=11,
    )

    # Estimated physical location of a candidate explicitly rejected
    # as non-waste. These are revealed chronologically.
    rejected_pts = ax_map.scatter(
        [],
        [],
        s=130,
        marker="X",
        label="Confirmed non-waste",
        zorder=9,
    )
    out_of_bounds_pts = ax_map.scatter(
        [],
        [],
        s=55,
        marker="x",
        label="Out of bounds",
        zorder=8,
    )

    status_text = ax_map.text(
        0.02,
        0.98,
        "",
        transform=ax_map.transAxes,
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round", alpha=0.12),
    )

    ax_map.legend(loc="best")

    # Object JSON records do not reliably carry mission event IDs.
    # Associate them with mission decisions by chronological order.
    #
    # confirmed_waste event #1 -> confirmed[0]
    # confirmed_waste event #2 -> confirmed[1]
    # ...
    #
    # rejected_non_waste event #1 -> rejected[0]
    # ...
    confirmation_index = 0
    rejection_index = 0

    xs_so_far = []
    ys_so_far = []

    cand_pts = []

    # Robot position where confirmation happened.
    conf_from_pts = []

    # Estimated physical waste location.
    conf_pts = []

    # Estimated non-waste locations revealed so far.
    reject_pts = []

    oob_pts = []

    conf_labels_seen = set()
    rejected_labels_seen = set()

    arrow = {"artist": None}

    # Sensor panels are intentionally stateful.
    # A thermal image remains visible until the next thermal image.
    # RGB remains paired with its thermal frame until a newer thermal
    # acquisition starts, at which point RGB is cleared.
    sensor_state = {
        "thermal_path": None,
        "thermal_row": None,
        "rgb_path": None,
    }

    frame_indices = build_display_frame_indices(
        events,
        base_hold=base_hold,
        candidate_hold=candidate_hold,
        confirmed_hold=confirmed_hold,
    )

    if fast:
        # ----------------------------------------------------
        # ADAPTIVE FAST MODE
        #
        # Routine mission activity stays quick, while visually
        # important AI decisions are held long enough to inspect.
        #
        # At fps=2:
        #   normal              = 0.5 sec
        #   vision positive     = 1.5 sec
        #   non-waste rejection = 2.0 sec
        #   confirmed waste     = 3.0 sec
        # ----------------------------------------------------

        compact = []

        for item in frame_indices:
            if not compact or compact[-1] != item:
                compact.append(item)

        adaptive = []

        for event_idx, stage in compact:
            r = events[event_idx]

            decision = str(
                r.get("_decision")
                or r.get("decision")
                or ""
            ).strip().lower()

            rejected_status = str(
                r.get("rejected_status")
                or ""
            ).strip().lower()

            vision_positive = str(
                r.get("vision_positive")
                or ""
            ).strip().lower() in {
                "true", "1", "yes", "y"
            }

            # Default: fly through routine operation.
            hold = 1

            # A positive RGB/YOLO observation deserves enough time
            # to inspect the actual camera result.
            if (
                stage == "paired"
                and vision_positive
            ):
                hold = 3

            # Candidate investigated and determined not to be waste.
            is_rejected = (
                (
                    "reject" in decision
                    and "duplicate" not in decision
                )
                or "non_waste" in decision
                or "non-waste" in decision
                or "not_waste" in decision
                or "not-waste" in decision
                or "reject" in rejected_status
            )

            if is_rejected:
                hold = max(hold, 4)

            # Confirmed debris is the most important visual moment.
            if decision == "confirmed_waste":
                hold = max(hold, 6)

            adaptive.extend(
                [(event_idx, stage)] * hold
            )

        frame_indices = adaptive

        print(
            f"[FAST MODE] Adaptive preview: "
            f"{len(compact)} logical stages -> "
            f"{len(frame_indices)} displayed frames."
        )

    def update(display_idx):
        event_idx, display_stage = frame_indices[display_idx]
        r = events[event_idx]

        x = r["_x"]
        y = r["_y"]
        heading = r["_heading"]

        # Only append path when first time seeing this event frame.
        if (
            display_idx == 0
            or frame_indices[display_idx - 1][0] != event_idx
        ):
            xs_so_far.append(x)
            ys_so_far.append(y)

            if r["_thermal_candidate"]:
                cand_pts.append([x, y])

            if x < 0.0 or x > 8.0 or y < 0.0 or y > 8.0:
                oob_pts.append([x, y])

            if r["_decision"] == "confirmed_waste":
                # ------------------------------------------------
                # Preserve BOTH useful positions:
                #
                # 1. (x,y) = where Voyager was when confirmation
                #    happened.
                #
                # 2. estimated object coordinate = where Voyager
                #    believes the physical waste actually is.
                # ------------------------------------------------
                conf_from_pts.append([x, y])

                confirm_number = len(conf_from_pts)
                obj = None

                if confirm_number <= len(confirmed):
                    obj = confirmed[confirm_number - 1]

                if obj is not None:
                    obj_x = object_x_for_render(obj)
                    obj_y = object_y_for_render(obj)
                else:
                    # Older/incomplete run fallback.
                    obj_x = safe_float(
                        r.get("estimated_object_x_ft"),
                        x,
                    )
                    obj_y = safe_float(
                        r.get("estimated_object_y_ft"),
                        y,
                    )

                    if obj_x is None:
                        obj_x = x
                    if obj_y is None:
                        obj_y = y

                conf_pts.append([obj_x, obj_y])

                label_key = confirm_number

                if label_key not in conf_labels_seen:
                    # Mark the confirmation / observation location.
                    ax_map.text(
                        x + 0.06,
                        y - 0.10,
                        f"W{confirm_number} confirmed here",
                        fontsize=8,
                        weight="bold",
                        va="top",
                        zorder=15,
                    )

                    # Mark and label estimated physical object location.
                    ax_map.text(
                        obj_x + 0.08,
                        obj_y + 0.08,
                        (
                            f"W{confirm_number}\n"
                            f"est. x={obj_x:.2f}, "
                            f"y={obj_y:.2f} ft"
                        ),
                        fontsize=9,
                        weight="bold",
                        zorder=15,
                    )

                    conf_labels_seen.add(label_key)

        # ----------------------------------------------------
        # Chronological non-waste discovery
        #
        # Do NOT preload rejected_objects.json onto frame 1.
        # Reveal each rejected target only when its rejection event
        # actually occurs, then keep it visible afterward.
        # ----------------------------------------------------
        decision_lower = str(
            r.get("_decision")
            or r.get("decision")
            or ""
        ).strip().lower()

        rejected_status_lower = str(
            r.get("rejected_status")
            or ""
        ).strip().lower()

        is_nonwaste_rejection = (
            decision_lower == "rejected_non_waste"
            or "non_waste" in decision_lower
            or "non-waste" in decision_lower
            or "not_waste" in decision_lower
            or "not-waste" in decision_lower
            or (
                "reject" in decision_lower
                and "duplicate" not in decision_lower
            )
            or "non_waste" in rejected_status_lower
            or "non-waste" in rejected_status_lower
            or (
                "reject" in rejected_status_lower
                and "duplicate" not in rejected_status_lower
            )
        )

        # Only reveal once per logical event, not once per held GIF frame.
        first_display_of_event = (
            display_idx == 0
            or frame_indices[display_idx - 1][0] != event_idx
        )

        if is_nonwaste_rejection and first_display_of_event:
            reject_number = len(reject_pts) + 1

            obj = None
            if reject_number <= len(rejected):
                obj = rejected[reject_number - 1]

            if obj is not None:
                reject_x = object_x_for_render(obj)
                reject_y = object_y_for_render(obj)
            else:
                reject_x = safe_float(
                    r.get("estimated_object_x_ft"),
                    x,
                )
                reject_y = safe_float(
                    r.get("estimated_object_y_ft"),
                    y,
                )

                if reject_x is None:
                    reject_x = x
                if reject_y is None:
                    reject_y = y

            reject_pts.append([reject_x, reject_y])

            if reject_number not in rejected_labels_seen:
                ax_map.text(
                    reject_x + 0.08,
                    reject_y + 0.08,
                    (
                        f"NW{reject_number}\n"
                        f"non-waste"
                    ),
                    fontsize=8,
                    weight="bold",
                    zorder=15,
                )

                rejected_labels_seen.add(reject_number)

        path_line.set_data(xs_so_far, ys_so_far)
        robot_dot.set_offsets(np.array([[x, y]]))
        thermal_pts.set_offsets(np.array(cand_pts) if cand_pts else np.empty((0, 2)))
        confirmed_from_pts.set_offsets(
            np.array(conf_from_pts)
            if conf_from_pts
            else np.empty((0, 2))
        )

        confirmed_pts.set_offsets(
            np.array(conf_pts)
            if conf_pts
            else np.empty((0, 2))
        )

        rejected_pts.set_offsets(
            np.array(reject_pts)
            if reject_pts
            else np.empty((0, 2))
        )

        out_of_bounds_pts.set_offsets(
            np.array(oob_pts) if oob_pts else np.empty((0, 2))
        )

        if arrow["artist"] is not None:
            try:
                arrow["artist"].remove()
            except Exception:
                pass

        dx = 0.35 * math.cos(math.radians(heading))
        dy = 0.35 * math.sin(math.radians(heading))
        arrow["artist"] = ax_map.arrow(
            x,
            y,
            dx,
            dy,
            head_width=0.12,
            head_length=0.14,
            length_includes_head=True,
        )

        row_rgb_path = get_rgb_image_path(r)
        row_thermal_path = get_thermal_image_path(r)

        # ----------------------------------------------------
        # Sensor pairing / persistence
        # ----------------------------------------------------

        if display_stage == "thermal_only":
            # A new thermal observation appears first.
            if row_thermal_path is not None:
                sensor_state["thermal_path"] = row_thermal_path
                sensor_state["thermal_row"] = r

            # Do not show the previous target's RGB beside a new thermal.
            sensor_state["rgb_path"] = None

        elif display_stage == "paired":
            # Keep exactly the same thermal frame visible and add
            # the RGB/YOLO image that corresponds to this observation.
            if row_thermal_path is not None:
                sensor_state["thermal_path"] = row_thermal_path
                sensor_state["thermal_row"] = r

            if row_rgb_path is not None:
                sensor_state["rgb_path"] = row_rgb_path

        else:
            # Normal event. If a genuinely new thermal frame arrives,
            # replace the prior thermal and clear stale RGB.
            if row_thermal_path is not None:
                old_thermal = sensor_state["thermal_path"]

                old_key = str(old_thermal) if old_thermal is not None else ""
                new_key = str(row_thermal_path)

                if old_key != new_key:
                    sensor_state["thermal_path"] = row_thermal_path
                    sensor_state["thermal_row"] = r
                    sensor_state["rgb_path"] = None
                else:
                    sensor_state["thermal_row"] = r

            # Vision appearing later is shown next to the currently
            # active thermal frame.
            if row_rgb_path is not None:
                sensor_state["rgb_path"] = row_rgb_path

        rgb_purpose = rgb_purpose_for_row(r)

        draw_image_panel(
            ax_rgb,
            sensor_state["rgb_path"],
            f"RGB / YOLO — {rgb_purpose}",
            "Waiting for current RGB observation",
        )

        draw_thermal_panel(
            ax_thermal,
            sensor_state["thermal_path"],
            "MLX90640 — Current Thermal Observation",
            "No thermal observation yet",
            row=sensor_state["thermal_row"],
        )

        current_event_id = r["_event_id_int"]

        # ----------------------------------------------------
        # Approach history must obey playback chronology.
        #
        # During thermal_only we deliberately DO NOT reveal the
        # RGB image that belongs to the upcoming paired stage.
        # ----------------------------------------------------
        history_end_idx = event_idx

        if (
            display_stage == "thermal_only"
            and r.get("phase") == "approach"
            and get_rgb_image_path(r) is not None
        ):
            history_end_idx = event_idx - 1

        event_rows = [
            e for e in events[:history_end_idx + 1]
            if (
                e["_event_id_int"] == current_event_id
                and e.get("phase") == "approach"
                and get_rgb_image_path(e) is not None
            )
        ]

        current_img_idx = max(0, len(event_rows) - 1)

        draw_gallery(
            ax_gallery,
            event_rows,
            current_img_idx,
        )

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
        frames=len(frame_indices),
        interval=int(1000 / fps),
        blit=False,
        repeat=True,
    )

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.38)
    anim.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="outputs/manual_navigation_tests/room_manual_hil_test_01")
    ap.add_argument("--fps", type=int, default=1)
    ap.add_argument("--base-hold", type=int, default=2)
    ap.add_argument("--candidate-hold", type=int, default=3)
    ap.add_argument("--confirmed-hold", type=int, default=5)
    ap.add_argument(
            "--fast",
            action="store_true",
            help=(
                "Fast preview: render one frame per logical display stage "
                "instead of repeated hold frames."
            ),
        )

        
    args = ap.parse_args()

    global RUN_DIR_GLOBAL
    run_dir = Path(args.run_dir)
    RUN_DIR_GLOBAL = run_dir
    events = load_events(run_dir / "mission_events.csv")
    confirmed = load_confirmed(run_dir / "confirmed_objects.json")
    rejected = load_rejected(run_dir / "rejected_objects.json")

    if not events:
        raise SystemExit("No events found.")

    out_gif = run_dir / "trajectory_sensor_animation_slow.gif"

    render_animation(
        events,
        confirmed,
        rejected,
        out_gif,
        fps=args.fps,
        base_hold=args.base_hold,
        candidate_hold=args.candidate_hold,
        confirmed_hold=args.confirmed_hold,
        fast=args.fast,
)

    print("Created:")
    print(out_gif)


if __name__ == "__main__":
    main()
