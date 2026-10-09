# v45 renderer v1: start=(2.5,0), heading=+Y; X=5 ft, Y=7 ft; parses v45 fusion/cooldown events.
from pathlib import Path
import argparse
import csv
import json
import re
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from PIL import Image, ImageDraw
# Offline renderer: do not import thermal_anomaly_test because that module
# imports Raspberry Pi hardware packages (board/Blinka).  The renderer only
# needs saved .npy frames, so use a small NumPy-only analyzer here.
def analyze_frame(arr, mode="cold", threshold_delta_c=2.0, min_blob_pixels=6, fov_x_deg=55.0):
    a = np.asarray(arr, dtype=float)
    finite = np.isfinite(a)
    if not finite.any():
        return None

    vals = a[finite]
    baseline = float(np.median(vals))
    if str(mode).lower() == "warm":
        mask = finite & (a >= baseline + float(threshold_delta_c))
    else:
        mask = finite & (a <= baseline - float(threshold_delta_c))

    # Find connected components without scipy/OpenCV.
    H, W = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    best = []
    for yy in range(H):
        for xx in range(W):
            if not mask[yy, xx] or seen[yy, xx]:
                continue
            stack = [(yy, xx)]
            seen[yy, xx] = True
            comp = []
            while stack:
                y0, x0 = stack.pop()
                comp.append((y0, x0))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        y1, x1 = y0 + dy, x0 + dx
                        if 0 <= y1 < H and 0 <= x1 < W and mask[y1, x1] and not seen[y1, x1]:
                            seen[y1, x1] = True
                            stack.append((y1, x1))
            if len(comp) > len(best):
                best = comp

    if len(best) < int(min_blob_pixels):
        return None

    cy = float(np.mean([p[0] for p in best]))
    cx = float(np.mean([p[1] for p in best]))
    bearing = ((cx - (W - 1) / 2.0) / max(1.0, (W - 1))) * float(fov_x_deg)
    return {
        "centroid_x": cx,
        "centroid_y": cy,
        "bearing_deg": bearing,
        "blob_pixels": len(best),
        "baseline_c": baseline,
    }


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
        marker="*",
        label="Verified bottle detection",
        zorder=10,
    )

    # Estimated physical waste location.
    confirmed_pts = ax_map.scatter(
        [],
        [],
        s=180,
        marker="*",
        label="_nolegend_",
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
        label="_nolegend_",
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
            if vision_positive:
                hold = max(hold, 3)

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

        # v9 fused event banner: deliberately large and transient so the
        # RGB+thermal success cannot be missed in the animation.
        old_banner = getattr(fig, "_v9_fused_banner", None)
        if old_banner is not None:
            try: old_banner.remove()
            except Exception: pass
            fig._v9_fused_banner = None
        if r.get("v44_success_kind") == "fused":
            rt=float(r.get("v44_rgb_t",0.0)); tt=float(r.get("v44_th_t",0.0))
            dtf=float(r.get("v44_fusion_dt",abs(rt-tt)))
            fig._v9_fused_banner = fig.text(
                0.72,0.965,
                f"★★ FUSED RGB + THERMAL  |  RGB {rt:.3f}s  TH {tt:.3f}s  Δt={dtf:.3f}s",
                ha="center",va="top",fontsize=12,weight="bold",
                bbox=dict(boxstyle="round",facecolor="white",alpha=0.95),
                zorder=100)

        # Only append path when first time seeing this event frame.
        if (
            display_idx == 0
            or frame_indices[display_idx - 1][0] != event_idx
        ):
            xs_so_far.append(x)
            ys_so_far.append(y)

            if r["_thermal_candidate"]:
                cand_pts.append([x, y])
                if r.get("v44_th_hit", False):
                    th_no = sum(1 for rr in events[:event_idx+1] if rr.get("v44_th_hit",False))
                    ax_map.scatter([x],[y],s=55,marker="^",zorder=11)
                    ax_map.text(x+0.04,y-0.07,f"T{th_no}",fontsize=6.5,weight="bold",zorder=12)

            # v9 diagnostic map: every YOLO-positive observation is visible.
            # This is an observation marker, not a claim of a unique bottle.
            if r.get("v44_rgb_hit", False):
                hit_no = sum(1 for rr in events[:event_idx+1] if rr.get("v44_rgb_hit",False))
                ax_map.scatter([x],[y],s=70,marker="o",facecolors="none",
                               edgecolors="black",linewidths=1.4,zorder=12)
                ax_map.text(x+0.05,y+0.05,f"V{hit_no}",fontsize=7,weight="bold",zorder=13)

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

                # v44 has no external object localization. Mark the
                # Voyager position at the verified detection timestamp.
                obj_x, obj_y = x, y

                conf_pts.append([obj_x, obj_y])

                label_key = confirm_number

                if label_key not in conf_labels_seen:
                    # Mark the confirmation / observation location.
                    fused_now = r.get('v44_success_kind')=='fused'
                    ax_map.scatter([x],[y],s=(210 if fused_now else 130),
                                   marker=('*' if fused_now else '*'),
                                   linewidths=1.2,zorder=15)
                    ax_map.text(
                        x + 0.06,
                        y - 0.10,
                        f"{'★★ FUSED' if fused_now else '★ VISUAL'} #{confirm_number}",
                        fontsize=(9 if fused_now else 8),
                        weight="bold",
                        va="top",
                        zorder=16,
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


if False and __name__ == "__main__":
    main()

# V44 pool adapter is defined below and replaces the legacy main when this file is
# invoked through v45_main().  The legacy renderer above is intentionally retained.


# -----------------------------------------------------------------------------
# v45 POOL ADAPTER
# -----------------------------------------------------------------------------
def _v44_ts(line):
    m = re.match(r"\[\s*([0-9.]+)s\s*\|", line)
    return float(m.group(1)) if m else None


def _v44_find_file(run_dir, name, kind):
    if not name:
        return ""
    p = Path(name)
    alts = [p.name]
    if p.name.startswith("rgb_obs_"):
        alts.append(p.name.replace("rgb_obs_", "rgb_", 1))
    dirs = [Path(run_dir), Path(run_dir)/kind]
    if kind == "thermal":
        dirs += [Path(run_dir)/"thermal_preview", Path(run_dir)/"thermal"]
    else:
        dirs += [Path(run_dir)/"rgb", Path(run_dir)/"paired_frames"]
    for d in dirs:
        for n in alts:
            q=d/n
            if q.exists(): return str(q)
    return str(Path(run_dir)/kind/p.name)


def load_v45_events(run_dir, forward_fps=0.32, start_x=0.0, start_y=0.0,
                    start_heading=90.0, pool_short=5.0, pool_long=7.0, left_deg_sec=30.0, right_deg_sec=45.0):
    """Translate v45 events.log into the legacy renderer's row schema.

    Position between wall anchors is dead-reckoned from MOTOR state.  Wall-anchor
    lines override the relevant coordinate exactly as logged.  This is animation
    reconstruction only; it is not external localization.
    """
    run_dir=Path(run_dir); log=run_dir/"events.log"
    if not log.exists(): raise SystemExit(f"Missing {log}")
    lines=log.read_text(errors="ignore").splitlines()
    # Build motor-state and anchor timeline.
    motor=[]; anchors=[]; rgb=[]; thermal=[]; fused=[]
    for line in lines:
        wall_t=_v44_ts(line)
        m=re.search(r'MOTOR epoch=\d+ -> ([A-Z_]+)', line)
        if m and wall_t is not None: motor.append((wall_t,m.group(1),line))
        m=re.search(r'POSE WALL_ANCHOR (RIGHT|LEFT|FAR|NEAR): \(([-0-9.]+),([-0-9.]+)\) -> \(([-0-9.]+),([-0-9.]+)\)', line)
        if m and wall_t is not None:
            anchors.append((wall_t,m.group(1),float(m.group(4)),float(m.group(5))))
        # Parse every saved RGB observation, including hit=NONE.
        # This keeps the camera panel continuously populated during the run.
        m=re.search(r'RGB_OBS t=([0-9.]+) file=(\S+)(.*)$', line)
        if m:
            rt=float(m.group(1)); rf=m.group(2); tail=m.group(3)
            mh=re.search(r'strength=(\S+) conf=([0-9.]+) box=\[([^]]+)\] bearing=([+\-0-9.]+) area=([0-9.]+)', tail)
            if mh:
                rgb.append(dict(t=rt,file=rf,hit=True,strength=mh.group(1),
                                conf=float(mh.group(2)),box=mh.group(3),
                                bearing=float(mh.group(4)),area=float(mh.group(5))))
            else:
                rgb.append(dict(t=rt,file=rf,hit=False,strength='NONE',
                                conf=0.0,box='',bearing=0.0,area=0.0))
        m=re.search(r'THERMAL_OBS t=([0-9.]+) file=(\S+) raw=(\S+) delta=([0-9.]+)C bearing=([+\-0-9.]+)', line)
        if m:
            key=(float(m.group(1)),m.group(2))
            if not any(abs(x['t']-key[0])<1e-4 and x['file']==key[1] for x in thermal):
                thermal.append(dict(t=key[0],file=m.group(2),raw=m.group(3),delta=float(m.group(4)),bearing=float(m.group(5))))
        if ('FUSED_CANDIDATE' in line or ('PAIR_CHECK' in line and 'fused=1' in line)):
            mr=re.search(r'RGB=\{t=([0-9.]+)',line); mt=re.search(r'TH=\{t=([0-9.]+)',line)
            if mr:
                fused.append((float(mr.group(1)), float(mt.group(1)) if mt else None))
        elif 'FUSED_SUPPRESSED' in line:
            # Cooldown suppression is still genuine RGB+thermal agreement. Pair it
            # to the most recent logged RGB and thermal observations for display.
            if rgb and thermal:
                pair=(float(rgb[-1]['t']), float(thermal[-1]['t']))
                if pair not in fused: fused.append(pair)

    # Timeline includes every sensor/motor/anchor event.  v7 reconstructs a
    # DENSE continuous path first, then samples it at these display times.
    all_t={0.0}
    all_t.update(t for t,_,_ in motor); all_t.update(t for t,_,_,_ in anchors)
    all_t.update(x['t'] for x in rgb); all_t.update(x['t'] for x in thermal)
    end=max([_v44_ts(x) or 0 for x in lines] or [0]); all_t.add(end)
    times=sorted(all_t)

    verified = {
      'pool_v44_fusion_01': {'rgb_obs_00026.384.jpg':1,'rgb_obs_00360.486.jpg':2,'rgb_00026.384.jpg':1,'rgb_00360.486.jpg':2},
      'pool_v44_fusion_03': {'rgb_obs_00035.385.jpg':1,'rgb_obs_00241.821.jpg':1,'rgb_obs_00245.619.jpg':1,'rgb_obs_00405.777.jpg':1,
                             'rgb_obs_00200.934.jpg':-1,'rgb_obs_00204.634.jpg':-1},
    }.get(run_dir.name,{})

    # ---------- v8 segment-constrained continuous reconstruction ----------
    # Principle:
    #   motor history -> continuous relative path shape
    #   wall observations -> endpoints of preceding path segments
    #
    # Unlike v5, anchors NEVER assign x/y at a single frame.  Unlike v7, they
    # are not weak local forces.  Each credible wall observation smoothly
    # stretches/translates the WHOLE preceding segment so that its endpoint is
    # genuinely near the observed wall while preserving continuity.

    dt_dense=0.20
    dense_t=np.arange(0.0, end+dt_dense, dt_dense, dtype=float)
    raw_x=np.empty_like(dense_t); raw_y=np.empty_like(dense_t); raw_h=np.empty_like(dense_t)
    raw_x[0]=float(start_x); raw_y[0]=float(start_y); raw_h[0]=float(start_heading)

    motor_sorted=sorted(motor,key=lambda q:q[0])
    mi=0; state='STOP'
    v_forward=float(forward_fps)
    v_turn=0.010
    left_turn_rate=float(left_deg_sec)
    right_turn_rate=float(right_deg_sec)

    # Build an entirely continuous raw motor trace.
    for k in range(1,len(dense_t)):
        t0=dense_t[k-1]; dt=dense_t[k]-t0
        while mi<len(motor_sorted) and motor_sorted[mi][0] <= t0+1e-9:
            state=motor_sorted[mi][1]; mi+=1
        x0=raw_x[k-1]; y0=raw_y[k-1]; h0=raw_h[k-1]
        if state=='TURN_LEFT':
            h1=h0+left_turn_rate*dt; speed=v_turn
        elif state=='TURN_RIGHT':
            h1=h0-right_turn_rate*dt; speed=v_turn
        else:
            h1=h0; speed=v_forward if state=='FORWARD' else 0.0
        hm=math.radians((h0+h1)*0.5)
        raw_x[k]=x0+speed*dt*math.cos(hm)
        raw_y[k]=y0+speed*dt*math.sin(hm)
        raw_h[k]=h1

    dense_x=raw_x.copy(); dense_y=raw_y.copy()

    def ray_wall(px,py,ang_deg,margin=0.12):
        a=math.radians(ang_deg); vx=math.cos(a); vy=math.sin(a)
        c=[]
        if vx>1e-6: c.append(((pool_short-px)/vx,'xmax'))
        if vx<-1e-6: c.append(((0.0-px)/vx,'xmin'))
        if vy>1e-6: c.append(((pool_long-py)/vy,'ymax'))
        if vy<-1e-6: c.append(((0.0-py)/vy,'ymin'))
        c=[z for z in c if z[0]>=0]
        if not c: return px,py,None
        d,w=min(c,key=lambda z:z[0])
        tx=px+d*vx; ty=py+d*vy
        if w=='xmax': tx=pool_short-margin
        elif w=='xmin': tx=margin
        elif w=='ymax': ty=pool_long-margin
        elif w=='ymin': ty=margin
        return tx,ty,w

    # Treat anchor names as robot-relative observations, never global walls.
    # Segment corrections are cumulative and continuous.
    anchor_list=sorted(anchors,key=lambda q:q[0])
    previous_anchor_t=0.0

    for at,side,_,_ in anchor_list:
        j=int(np.argmin(np.abs(dense_t-at)))
        if j<=1: continue

        px=float(dense_x[j]); py=float(dense_y[j])
        hh=float(raw_h[j])
        if side=='RIGHT': ray=hh-90.0
        elif side=='LEFT': ray=hh+90.0
        elif side=='FAR': ray=hh
        else: ray=hh+180.0

        tx,ty,wall=ray_wall(px,py,ray)
        if wall is None: continue

        dx=tx-px; dy=ty-py

        # Do not allow one ambiguous observation to relocate the entire path
        # across the pool.  Large corrections are still allowed, but capped to
        # a physically credible segment-scale adjustment.
        mag=math.hypot(dx,dy)
        # v9: a logged wall anchor is a strong endpoint constraint.  Do NOT
        # cap the correction and leave the boat in mid-pool.  Continuity is
        # preserved because this full correction is spread over the preceding
        # segment with smoothstep below; no single frame is relocated.

        # Begin fitting after the previous anchor, but use at least a 12-second
        # lead-in. This makes the vessel APPROACH the wall instead of suddenly
        # arriving there.
        seg_start=max(previous_anchor_t, at-40.0)
        i0=int(np.searchsorted(dense_t,seg_start,side='left'))
        i0=max(0,min(i0,j-1))

        # Smoothstep correction across the whole preceding segment.
        denom=max(1,j-i0)
        for qidx in range(i0,j+1):
            u=(qidx-i0)/denom
            w=u*u*(3.0-2.0*u)
            dense_x[qidx]+=dx*w
            dense_y[qidx]+=dy*w

        # Carry the endpoint correction forward so the next frame begins at
        # exactly the same location. Future anchors may then smoothly modify
        # their own preceding segments. This is the continuity guarantee.
        if j+1 < len(dense_t):
            dense_x[j+1:]+=dx
            dense_y[j+1:]+=dy

        previous_anchor_t=at

    # Soft physical-domain correction. If accumulated dead reckoning drifts
    # outside, distribute the correction over several seconds instead of
    # clipping an individual frame.
    def smooth_domain(axis, lo, hi):
        arr=axis.copy()
        n=len(arr); ramp=max(2,int(5.0/dt_dense))
        for j in range(n):
            target=None
            if arr[j] < lo: target=lo
            elif arr[j] > hi: target=hi
            if target is None: continue
            delta=target-arr[j]
            i0=max(0,j-ramp)
            den=max(1,j-i0)
            for q in range(i0,j+1):
                u=(q-i0)/den
                w=u*u*(3.0-2.0*u)
                arr[q]+=delta*w
            if j+1<n: arr[j+1:]+=delta
        return arr

    dense_x=smooth_domain(dense_x,0.08,float(pool_short)-0.08)
    dense_y=smooth_domain(dense_y,0.08,float(pool_long)-0.08)

    # Gentle smoothing removes numerical kinks but preserves wall approaches.
    for _ in range(2):
        xx=dense_x.copy(); yy=dense_y.copy()
        dense_x[1:-1]=0.20*xx[:-2]+0.60*xx[1:-1]+0.20*xx[2:]
        dense_y[1:-1]=0.20*yy[:-2]+0.60*yy[1:-1]+0.20*yy[2:]

    def _sample_pose(t):
        x=float(np.interp(t,dense_t,dense_x))
        y=float(np.interp(t,dense_t,dense_y))
        j=int(np.argmin(np.abs(dense_t-t)))
        a=max(0,j-2); b=min(len(dense_t)-1,j+2)
        dx=float(dense_x[b]-dense_x[a]); dy=float(dense_y[b]-dense_y[a])
        hh=float(raw_h[j])
        if math.hypot(dx,dy)>1e-4:
            hh=math.degrees(math.atan2(dy,dx))
        return x,y,hh

    rows=[]; eid=0
    for t in times:
        x,y,h=_sample_pose(t)
        near_rgb=min(rgb,key=lambda z:abs(z['t']-t)) if rgb else None
        near_th=min(thermal,key=lambda z:abs(z['t']-t)) if thermal else None
        use_rgb=near_rgb if near_rgb and abs(near_rgb['t']-t)<0.02 else None
        use_th=near_th if near_th and abs(near_th['t']-t)<0.02 else None
        if not (use_rgb or use_th or any(abs(a[0]-t)<.02 for a in anchors) or any(abs(m[0]-t)<.02 for m in motor)):
            continue
        eid+=1
        r={'event_id':str(eid),'phase':'coverage_scan','x_ft':str(x),'y_ft':str(y),'heading_deg':str(h),
           '_x':x,'_y':y,'_heading':h,'_event_id_int':eid,'_decision':'','_thermal_candidate':bool(use_th),'_vision_positive':False,
           'thermal_candidate':'True' if use_th else 'False','vision_positive':'False','decision':'','thermal_mode':'warm',
           'thermal_preview_path':'','vision_raw_path':'','vision_annotated_path':'','thermal_delta_c':'','thermal_bearing_deg':''}
        if use_th:
            r['thermal_preview_path']=_v44_find_file(run_dir,use_th['file'],'thermal'); r['thermal_delta_c']=str(use_th['delta']); r['thermal_bearing_deg']=str(use_th['bearing']); r['v44_th_hit']=True; r['v44_th_t']=float(use_th['t']); r['v44_th_file']=use_th['file']
        if use_rgb:
            r['vision_raw_path']=_v44_find_file(run_dir,use_rgb['file'],'rgb')
            r['_vision_positive']=bool(use_rgb.get('hit',False))
            r['vision_positive']='True' if r['_vision_positive'] else 'False'
            r['v44_rgb_file']=use_rgb['file']
            r['v44_conf']=use_rgb.get('conf',0.0)
            r['v44_box']=use_rgb.get('box','')
            r['v44_rgb_hit']=bool(use_rgb.get('hit',False))
            r['v44_rgb_t']=float(use_rgb.get('t',t))

            # Determine fusion FIRST. This fixes the v8 bug where the RGB image
            # was annotated before PAIR_CHECK was applied, so a real fused hit
            # could never visibly become ★★ in the RGB panel.
            fp=None
            if use_rgb.get('hit',False):
                fp=next((p for p in fused if abs(p[0]-use_rgb['t'])<0.08),None)

            mark=0
            if fp is not None:
                mark=2
                th_target=fp[1] if fp[1] is not None else use_rgb['t']
                thx=min(thermal,key=lambda z:abs(z['t']-th_target)) if thermal else None
                if thx is not None:
                    r['_thermal_candidate']=True
                    r['thermal_candidate']='True'
                    r['thermal_preview_path']=_v44_find_file(run_dir,thx['file'],'thermal')
                    r['thermal_delta_c']=str(thx['delta'])
                    r['thermal_bearing_deg']=str(thx['bearing'])
                    r['v44_th_hit']=True
                    r['v44_th_t']=float(thx['t'])
                    r['v44_th_file']=thx['file']
                    r['v44_fusion_dt']=abs(float(use_rgb['t'])-float(thx['t']))
                r['_decision']='confirmed_waste'
                r['decision']='confirmed_waste'
                r['v44_success_kind']='fused'
            elif use_rgb.get('hit',False):
                # Keep manually reviewed labels only for visual-only semantic
                # confirmation / known false positives. Every YOLO hit is still
                # rendered and numbered even when mark remains 0.
                reviewed=verified.get(use_rgb['file'],0)
                if reviewed==1:
                    mark=1
                    r['_decision']='confirmed_waste'
                    r['decision']='confirmed_waste'
                    r['v44_success_kind']='visual'
                elif reviewed==-1:
                    mark=-1
                    r['_decision']='false_positive'
                    r['decision']='false_positive'

            r['v44_mark']=mark

            # Annotate only AFTER fusion/review classification is final.
            if use_rgb.get('hit',False):
                r['vision_raw_path']=_v44_annotated_rgb(
                    run_dir,r['vision_raw_path'],use_rgb.get('box',''),mark,
                    use_rgb.get('conf',0.0),use_rgb['file'])
        rows.append(r)
    return rows


def _v44_annotated_rgb(run_dir, src_path, box_text, mark, conf, filename):
    """Create a cached display copy with YOLO box + explicit success label."""
    if not src_path or not box_text:
        return src_path
    try:
        vals=[float(v.strip()) for v in str(box_text).split(',')]
        if len(vals)!=4:
            return src_path
        outdir=Path(run_dir)/"_animation_annotated_rgb"
        outdir.mkdir(exist_ok=True)
        out=outdir/Path(filename).name
        if out.exists():
            return str(out)
        im=Image.open(src_path).convert("RGB")
        dr=ImageDraw.Draw(im)
        x1,y1,x2,y2=vals
        width=max(4,int(min(im.size)*0.007))
        dr.rectangle((x1,y1,x2,y2),outline="yellow",width=width)
        if mark==2:
            label=f"★★ RGB + THERMAL BOTTLE   conf={conf:.3f}"
        elif mark==1:
            label=f"★ VISUAL BOTTLE   conf={conf:.3f}"
        elif mark==-1:
            label=f"YOLO FALSE POSITIVE   conf={conf:.3f}"
        else:
            label=f"YOLO HIT   conf={conf:.3f}"
        tx=max(4,int(x1)); ty=max(4,int(y1)-28)
        dr.rectangle((tx,ty, min(im.width-2,tx+max(260,12*len(label))), ty+26), fill="black")
        dr.text((tx+4,ty+4),label,fill="yellow")
        im.save(out,quality=92)
        return str(out)
    except Exception:
        return src_path


def setup_v45_pool_map(ax, events, confirmed, short_ft=5.0, long_ft=7.0):
    """Physical axes: X=short side (5 ft), Y=long side (7 ft)."""
    ax.set_xlim(0.0,short_ft)
    ax.set_ylim(0.0,long_ft)
    ax.set_aspect('equal',adjustable='box')
    ax.grid(True,alpha=.25)
    ax.set_xlabel('X position (ft) — reconstructed estimate')
    ax.set_ylabel('Y position (ft) — reconstructed estimate')
    ax.set_title('Ocean Voyager v45 — stochastic navigation reconstructed journey')


def v45_main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run-dir',required=True); ap.add_argument('--fps',type=int,default=2)
    ap.add_argument('--base-hold',type=int,default=1); ap.add_argument('--candidate-hold',type=int,default=3); ap.add_argument('--confirmed-hold',type=int,default=6)
    ap.add_argument('--fast',action='store_true'); ap.add_argument('--pool-short-ft',type=float,default=5); ap.add_argument('--pool-long-ft',type=float,default=7)
    ap.add_argument('--forward-ft-sec',type=float,default=.175,help='v45 display dead-reckoning forward speed')
    ap.add_argument('--left-deg-sec',type=float,default=30.0,help='v45 display TURN_LEFT rate')
    ap.add_argument('--right-deg-sec',type=float,default=45.0,help='v45 display TURN_RIGHT rate')
    ap.add_argument('--start-x',type=float,default=2.5); ap.add_argument('--start-y',type=float,default=0); ap.add_argument('--start-heading',type=float,default=90)
    args=ap.parse_args(); global RUN_DIR_GLOBAL; RUN_DIR_GLOBAL=Path(args.run_dir)
    events=load_v45_events(RUN_DIR_GLOBAL,args.forward_ft_sec,args.start_x,args.start_y,args.start_heading,args.pool_short_ft,args.pool_long_ft,args.left_deg_sec,args.right_deg_sec)
    if not events: raise SystemExit('No v45 events parsed.')
    rgb_hits=sum(1 for r in events if r.get('v44_rgb_hit',False))
    th_hits=sum(1 for r in events if r.get('v44_th_hit',False))
    fused_hits=sum(1 for r in events if r.get('v44_success_kind')=='fused')
    visual_confirmed=sum(1 for r in events if r.get('v44_success_kind')=='visual')
    print(f"[v45 v1 SENSOR SUMMARY] RGB detections={rgb_hits} | thermal detections={th_hits} | fused detections={fused_hits} | reviewed visual confirmations={visual_confirmed}")
    # Temporarily substitute pool map while preserving the mature renderer.
    global setup_map; old_setup=setup_map
    setup_map=lambda ax,e,c: setup_v45_pool_map(ax,e,c,args.pool_short_ft,args.pool_long_ft)
    out=RUN_DIR_GLOBAL/'trajectory_sensor_animation_v45_v1.gif'
    try: render_animation(events,[],[],out,args.fps,args.base_hold,args.candidate_hold,args.confirmed_hold,args.fast)
    finally: setup_map=old_setup
    print('Created:'); print(out); print(f'Parsed {len(events)} v45 display events')


if __name__ == '__main__':
    v45_main()
