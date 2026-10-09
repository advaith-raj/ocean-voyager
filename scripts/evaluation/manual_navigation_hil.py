from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

import board
import busio
import adafruit_mlx90640

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from thermal_anomaly_test import (  # noqa: E402
    read_frame_with_retries,
    sanitize_frame,
    analyze_frame,
    save_heatmap,
)



CURRENT_POSE = {
    "x_ft": 0.0,
    "y_ft": 0.0,
    "heading_deg": 0.0,
}


class ReturnHomeRequested(Exception):
    def __init__(self, x_ft=None, y_ft=None, heading_deg=None):
        super().__init__("User requested return home")
        self.x_ft = x_ft
        self.y_ft = y_ft
        self.heading_deg = heading_deg


def set_current_pose(x_ft, y_ft, heading_deg):
    CURRENT_POSE["x_ft"] = float(x_ft)
    CURRENT_POSE["y_ft"] = float(y_ft)
    CURRENT_POSE["heading_deg"] = float(heading_deg)


def get_current_pose():
    return (
        CURRENT_POSE["x_ft"],
        CURRENT_POSE["y_ft"],
        CURRENT_POSE["heading_deg"],
    )


def now_s():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def prompt_user(message):
    print("\n" + "=" * 72)
    print(message)
    print("=" * 72)
    ans = input("Press ENTER when done, or type q/quit + ENTER to return home: ").strip().lower()
    if ans in ("q", "quit", "home", "return"):
        x_ft, y_ft, heading_deg = get_current_pose()
        raise ReturnHomeRequested(x_ft, y_ft, heading_deg)


def norm_heading(deg):
    deg = deg % 360.0
    if deg < 0:
        deg += 360.0
    return deg


def move_pose(x, y, heading_deg, distance_ft):
    rad = math.radians(heading_deg)
    x2 = x + distance_ft * math.cos(rad)
    y2 = y + distance_ft * math.sin(rad)
    set_current_pose(x2, y2, heading_deg)
    return x2, y2



def signed_heading_delta(current_deg, target_deg):
    # Returns shortest turn from current heading to target heading.
    # Positive = left / counterclockwise. Negative = right / clockwise.
    return (target_deg - current_deg + 180.0) % 360.0 - 180.0


def guide_return_home(x, y, heading_deg, home_x, home_y, step_ft, events_csv=None, event_id=0):
    print("\n" + "#" * 72)
    print("RETURN-HOME GUIDE STARTED")
    print("#" * 72)
    print(f"Current estimated pose: x={x:.2f} ft, y={y:.2f} ft, heading={heading_deg:.1f}°")
    print(f"Home/start point:       x={home_x:.2f} ft, y={home_y:.2f} ft")
    print(f"Return step size:       {step_ft:.2f} ft")
    print()
    print("Follow each instruction physically. Press ENTER after each action.")
    print("Type stop + ENTER to stop return-home guidance.")
    print("#" * 72)

    step_ft = float(step_ft)
    if step_ft <= 0:
        step_ft = 1.0

    x = float(x)
    y = float(y)
    heading_deg = norm_heading(float(heading_deg))
    home_x = float(home_x)
    home_y = float(home_y)
    set_current_pose(x, y, heading_deg)

    return_step = 0

    while True:
        dx_home = home_x - x
        dy_home = home_y - y
        dist_home = math.hypot(dx_home, dy_home)

        if dist_home <= 0.15:
            x = home_x
            y = home_y
            set_current_pose(x, y, heading_deg)

            event_id += 1
            if events_csv is not None:
                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "return_home",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "instruction": "home reached",
                    "decision": "home_reached",
                    "notes": "manual return-home complete",
                })

            print("\nHOME REACHED.")
            print(f"Final estimated pose: x={x:.2f} ft, y={y:.2f} ft, heading={heading_deg:.1f}°")
            return x, y, heading_deg, event_id

        target_heading = norm_heading(math.degrees(math.atan2(dy_home, dx_home)))
        turn_delta = signed_heading_delta(heading_deg, target_heading)

        if abs(turn_delta) > 3.0:
            direction = "LEFT" if turn_delta > 0 else "RIGHT"
            ans = input(
                f"\nRETURN STEP {return_step + 1}: Rotate {direction} "
                f"{abs(turn_delta):.1f}° toward home. "
                f"Press ENTER when done, or type stop: "
            ).strip().lower()

            if ans in ("stop", "q", "quit"):
                print("Return-home guidance stopped.")
                return x, y, heading_deg, event_id

            heading_deg = norm_heading(heading_deg + turn_delta)
            set_current_pose(x, y, heading_deg)

            event_id += 1
            if events_csv is not None:
                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "return_home",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "instruction": f"rotate {direction} {abs(turn_delta):.1f} deg toward home",
                    "decision": "returning_home",
                    "notes": f"target_heading={target_heading:.1f}",
                })

        dx_home = home_x - x
        dy_home = home_y - y
        dist_home = math.hypot(dx_home, dy_home)

        move_dist = min(step_ft, dist_home)

        ans = input(
            f"\nRETURN STEP {return_step + 1}: Move FORWARD {move_dist:.2f} ft toward home. "
            f"Remaining after this will be about {max(0.0, dist_home - move_dist):.2f} ft. "
            f"Press ENTER when done, or type stop: "
        ).strip().lower()

        if ans in ("stop", "q", "quit"):
            print("Return-home guidance stopped.")
            return x, y, heading_deg, event_id

        x, y = move_pose(x, y, heading_deg, move_dist)
        return_step += 1

        event_id += 1
        if events_csv is not None:
            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id,
                "phase": "return_home",
                "x_ft": round(x, 3),
                "y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "instruction": f"move forward {move_dist:.2f} ft toward home",
                "decision": "returning_home",
                "notes": f"remaining_distance_ft={max(0.0, dist_home - move_dist):.2f}",
            })

        print(f"Updated estimated pose: x={x:.2f} ft, y={y:.2f} ft, heading={heading_deg:.1f}°")


def turn_instruction_from_bearing(bearing_deg):
    # In our thermal image convention:
    # negative bearing = object appears left of center
    # positive bearing = object appears right of center
    if bearing_deg < -3:
        return "LEFT", abs(bearing_deg), +abs(bearing_deg)
    if bearing_deg > 3:
        return "RIGHT", abs(bearing_deg), -abs(bearing_deg)
    return "CENTER", 0.0, 0.0


def capture_image(out_path, timeout_ms=1500, hflip=True, vflip=True):
    cmd = [
        "rpicam-still",
        "-o", str(out_path),
        "--timeout", str(timeout_ms),
        "--nopreview",
    ]
    if hflip:
        cmd.append("--hflip")
    if vflip:
        cmd.append("--vflip")

    subprocess.run(cmd, check=True, capture_output=True, text=True)


def run_yolo_on_image(model, image_path, conf, expected_labels):
    result = model.predict(source=str(image_path), conf=conf, verbose=False)[0]
    names = result.names

    detections = []
    positive = False

    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label = str(names.get(cls_id, cls_id))
            score = float(box.conf.item())
            xyxy = [float(v) for v in box.xyxy[0].tolist()]
            is_expected = label.lower() in expected_labels

            if is_expected:
                positive = True

            detections.append({
                "label": label,
                "confidence": round(score, 4),
                "box_xyxy": [round(v, 2) for v in xyxy],
                "is_expected": is_expected,
            })

    return positive, detections


def annotate_image(image_path, out_path, detections):
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    for det in detections:
        x1, y1, x2, y2 = det["box_xyxy"]
        label = det["label"]
        conf = det["confidence"]
        text = f"{label} {conf:.2f}"
        draw.rectangle([x1, y1, x2, y2], outline="lime", width=5)
        draw.text((x1 + 5, max(0, y1 - 22)), text, fill="lime")

    img.save(out_path)



def _as_float(v, default=0.0):
    try:
        return float(v)
    except Exception:
        return default


def _det_xyxy(det):
    # Supports several possible detection dictionary formats.
    for key in ("bbox_xyxy", "xyxy", "box_xyxy"):
        if key in det and det[key] is not None:
            vals = det[key]
            if len(vals) >= 4:
                return [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]

    if "bbox" in det and det["bbox"] is not None:
        vals = det["bbox"]
        if len(vals) >= 4:
            return [float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])]

    keys = ("x1", "y1", "x2", "y2")
    if all(k in det for k in keys):
        return [float(det["x1"]), float(det["y1"]), float(det["x2"]), float(det["y2"])]

    return None


def vision_collision_guard(detections, image_path, args):
    """
    Estimate whether the confirmed vision target is too close.

    Uses YOLO box geometry:
    - large area fraction
    - tall box
    - box bottom near bottom of image

    Returns: collision_risk, info_dict
    """
    info = {
        "collision_risk": False,
        "bbox_area_frac": 0.0,
        "bbox_height_frac": 0.0,
        "bbox_bottom_frac": 0.0,
        "best_conf": 0.0,
        "reason": "",
    }

    if not detections:
        return False, info

    try:
        from PIL import Image
        with Image.open(image_path) as im:
            w, h = im.size
    except Exception:
        w, h = 4056, 3040

    best = None
    best_conf = -1.0
    for det in detections:
        xyxy = _det_xyxy(det)
        if xyxy is None:
            continue
        conf = _as_float(det.get("confidence", det.get("conf", 0.0)), 0.0)
        if conf > best_conf:
            best = (det, xyxy)
            best_conf = conf

    if best is None:
        return False, info

    _, xyxy = best
    x1, y1, x2, y2 = xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    area_frac = (bw * bh) / max(1.0, float(w * h))
    height_frac = bh / max(1.0, float(h))
    bottom_frac = max(y1, y2) / max(1.0, float(h))

    area_thr = getattr(args, "collision_bbox_area_frac", 0.15)
    height_thr = getattr(args, "collision_bbox_height_frac", 0.35)
    bottom_thr = getattr(args, "collision_bbox_bottom_frac", 0.80)

    reasons = []
    if area_frac >= area_thr:
        reasons.append(f"bbox_area_frac={area_frac:.3f}>={area_thr:.3f}")
    if height_frac >= height_thr:
        reasons.append(f"bbox_height_frac={height_frac:.3f}>={height_thr:.3f}")
    if bottom_frac >= bottom_thr:
        reasons.append(f"bbox_bottom_frac={bottom_frac:.3f}>={bottom_thr:.3f}")

    risk = bool(reasons)

    info.update({
        "collision_risk": risk,
        "bbox_area_frac": round(area_frac, 5),
        "bbox_height_frac": round(height_frac, 5),
        "bbox_bottom_frac": round(bottom_frac, 5),
        "best_conf": round(best_conf, 5),
        "reason": "; ".join(reasons),
    })
    return risk, info


def load_json_list(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, list) else []
    except Exception:
        return []


def next_object_id(confirmed_json):
    existing = load_json_list(confirmed_json)
    max_id = 0
    for obj in existing:
        try:
            max_id = max(max_id, int(obj.get("object_id", 0)))
        except Exception:
            pass
    return max_id + 1


def find_duplicate_confirmed(confirmed_json, obj, radius_ft):
    existing = load_json_list(confirmed_json)
    x, y = _object_xy_for_duplicate(obj)

    best = None
    best_d = None
    for old in existing:
        if old.get("duplicate_suppressed"):
            continue
        ox, oy = _object_xy_for_duplicate(old)
        d = math.hypot(x - ox, y - oy)
        if best_d is None or d < best_d:
            best = old
            best_d = d

    if best is not None and best_d is not None and best_d <= radius_ft:
        return best, best_d
    return None, None


def append_duplicate_detection(confirmed_json, obj):
    dup_path = Path(confirmed_json).with_name("duplicate_detections.json")
    existing = load_json_list(dup_path)
    existing.append(obj)
    dup_path.write_text(json.dumps(existing, indent=2))



def best_detection_geometry(detections, image_path):
    if not detections:
        return None

    try:
        from PIL import Image
        with Image.open(image_path) as im:
            img_w, img_h = im.size
    except Exception:
        img_w, img_h = 4056, 3040

    best_xyxy = None
    best_conf = -1.0

    for det in detections:
        xyxy = _det_xyxy(det)
        if xyxy is None:
            continue
        conf = _as_float(det.get("confidence", det.get("conf", 0.0)), 0.0)
        if conf > best_conf:
            best_conf = conf
            best_xyxy = xyxy

    if best_xyxy is None:
        return None

    x1, y1, x2, y2 = best_xyxy
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0

    return {
        "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
        "confidence": round(best_conf, 5),
        "img_w": img_w,
        "img_h": img_h,
        "bbox_center_x_frac": cx / max(1.0, img_w),
        "bbox_height_frac": bh / max(1.0, img_h),
        "bbox_area_frac": (bw * bh) / max(1.0, img_w * img_h),
        "bbox_bottom_frac": max(y1, y2) / max(1.0, img_h),
    }


def estimate_object_location_from_vision(
    robot_x_ft,
    robot_y_ft,
    robot_heading_deg,
    detections,
    image_path,
    args,
    prev_bbox_height_frac=None,
    last_forward_move_ft=None,
):
    geom = best_detection_geometry(detections, image_path)
    if geom is None:
        return {}

    h_fov = float(getattr(args, "vision_horizontal_fov_deg", 66.0))
    center_offset = geom["bbox_center_x_frac"] - 0.5
    visual_offset_deg = center_offset * h_fov
    object_bearing_deg = norm_heading(float(robot_heading_deg) + visual_offset_deg)

    current_h = geom["bbox_height_frac"]
    estimated_range_ft = None
    range_method = "fallback_bbox_size"

    if (
        prev_bbox_height_frac is not None
        and last_forward_move_ft is not None
        and prev_bbox_height_frac > 0.001
        and current_h > prev_bbox_height_frac
    ):
        scale_ratio = current_h / prev_bbox_height_frac
        if scale_ratio >= float(getattr(args, "min_bbox_growth_ratio", 1.08)):
            estimated_range_ft = float(last_forward_move_ft) / max(0.01, scale_ratio - 1.0)
            range_method = "bbox_expansion"

    if estimated_range_ft is None:
        if current_h >= 0.35:
            estimated_range_ft = 0.75
        elif current_h >= 0.25:
            estimated_range_ft = 1.0
        elif current_h >= 0.15:
            estimated_range_ft = 1.5
        elif current_h >= 0.08:
            estimated_range_ft = 2.5
        else:
            estimated_range_ft = float(getattr(args, "fallback_object_distance_ft", 3.5))

    estimated_range_ft = max(
        float(getattr(args, "min_estimated_object_distance_ft", 0.25)),
        min(float(getattr(args, "max_estimated_object_distance_ft", 6.0)), float(estimated_range_ft)),
    )

    rad = math.radians(object_bearing_deg)
    raw_obj_x = float(robot_x_ft) + estimated_range_ft * math.cos(rad)
    raw_obj_y = float(robot_y_ft) + estimated_range_ft * math.sin(rad)

    length = float(getattr(args, "room_length_ft", 8.0))
    width = float(getattr(args, "room_width_ft", 8.0))
    margin = float(getattr(args, "boundary_margin_ft", 0.15))

    obj_x = max(0.0, min(length, raw_obj_x))
    obj_y = max(0.0, min(width, raw_obj_y))

    near_boundary = (
        obj_x <= margin or obj_x >= length - margin or
        obj_y <= margin or obj_y >= width - margin
    )

    return {
        "estimated_object_x_ft": round(obj_x, 3),
        "estimated_object_y_ft": round(obj_y, 3),
        "raw_estimated_object_x_ft": round(raw_obj_x, 3),
        "raw_estimated_object_y_ft": round(raw_obj_y, 3),
        "estimated_object_range_ft": round(estimated_range_ft, 3),
        "estimated_object_bearing_deg": round(object_bearing_deg, 2),
        "visual_offset_deg": round(visual_offset_deg, 2),
        "range_method": range_method,
        "bbox_height_frac": round(current_h, 5),
        "bbox_area_frac": round(geom["bbox_area_frac"], 5),
        "bbox_bottom_frac": round(geom["bbox_bottom_frac"], 5),
        "object_estimate_near_boundary": bool(near_boundary),
    }


def _object_xy_for_duplicate(obj):
    for xk, yk in [
        ("estimated_object_x_ft", "estimated_object_y_ft"),
        ("estimated_x_ft", "estimated_y_ft"),
        ("x_ft", "y_ft"),
    ]:
        if obj.get(xk) not in (None, "") and obj.get(yk) not in (None, ""):
            return _as_float(obj.get(xk), 0.0), _as_float(obj.get(yk), 0.0)
    return 0.0, 0.0


def next_pose_out_of_bounds(x, y, heading_deg, move_ft, args):
    rad = math.radians(float(heading_deg))
    nx = float(x) + float(move_ft) * math.cos(rad)
    ny = float(y) + float(move_ft) * math.sin(rad)

    length = float(getattr(args, "room_length_ft", 8.0))
    width = float(getattr(args, "room_width_ft", 8.0))
    margin = float(getattr(args, "boundary_margin_ft", 0.15))

    out = (nx < -margin or nx > length + margin or ny < -margin or ny > width + margin)
    return out, nx, ny


def scan_thermal(mlx, out_dir, sample_name, args):
    arr, err = read_frame_with_retries(
        mlx,
        retries=args.read_retries,
        delay_sec=args.read_retry_delay_sec,
    )

    if arr is None:
        return {
            "ok": False,
            "candidate": False,
            "error": err,
        }

    analysis = analyze_frame(
        arr,
        mode=args.mode,
        threshold_delta_c=args.threshold,
        min_blob_pixels=args.min_blob_pixels,
        fov_x_deg=args.fov_x_deg,
    )

    frame_path = out_dir / "thermal_frames" / f"{sample_name}.npy"
    preview_path = out_dir / "thermal_preview" / f"{sample_name}.png"

    np.save(frame_path, sanitize_frame(arr))

    title = (
        f"{sample_name} | mode={analysis['selected_mode']} | "
        f"candidate={analysis['candidate']} | "
        f"delta={analysis['delta_c']}C | "
        f"bearing={analysis['bearing_deg']}deg"
    )
    save_heatmap(arr, analysis, preview_path, title)

    return {
        "ok": True,
        **analysis,
        "frame_path": str(frame_path),
        "preview_path": str(preview_path),
        "error": "",
    }


def write_event(events_csv, row):
    fieldnames = [
        "time",
        "event_id",
        "phase",
        "x_ft",
        "y_ft",
        "heading_deg",
        "instruction",
        "thermal_ok",
        "thermal_candidate",
        "thermal_mode",
        "thermal_delta_c",
        "thermal_bearing_deg",
        "thermal_blob_px",
        "thermal_centroid_x",
        "thermal_centroid_y",
        "thermal_preview_path",
        "vision_positive",
        "vision_num_detections",
        "vision_raw_path",
        "vision_annotated_path",
        "collision_risk",
        "vision_bbox_area_frac",
        "vision_bbox_height_frac",
        "vision_bbox_bottom_frac",
        "duplicate_status",
        "object_id",
        "matched_object_id",
        "decision",
        "notes",
    ]

    exists = events_csv.exists()

    with events_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()

        clean = {k: row.get(k, "") for k in fieldnames}
        writer.writerow(clean)


def generate_lawnmower_plan(length_ft, width_ft, lane_spacing_ft, step_ft):
    commands = []
    y = 0.0
    direction = 1  # +1 means east/forward along first lane, -1 means west/back
    lane = 0

    while y <= width_ft + 1e-6:
        remaining = length_ft
        while remaining > 1e-6:
            d = min(step_ft, remaining)
            commands.append({
                "type": "move",
                "distance_ft": d,
                "label": f"Lane {lane}: move forward {d:.2f} ft",
            })
            remaining -= d

        next_y = y + lane_spacing_ft
        if next_y > width_ft + 1e-6:
            break

        if direction == 1:
            turn = "LEFT"
            heading_delta = +90.0
        else:
            turn = "RIGHT"
            heading_delta = -90.0

        commands.append({
            "type": "rotate",
            "turn": turn,
            "degrees": 90.0,
            "heading_delta": heading_delta,
            "label": f"End of lane {lane}: rotate {turn} 90°",
        })

        commands.append({
            "type": "move",
            "distance_ft": lane_spacing_ft,
            "label": f"Shift to next lane: move forward {lane_spacing_ft:.2f} ft",
        })

        commands.append({
            "type": "rotate",
            "turn": turn,
            "degrees": 90.0,
            "heading_delta": heading_delta,
            "label": f"Align with lane {lane + 1}: rotate {turn} 90°",
        })

        y = next_y
        direction *= -1
        lane += 1

    return commands


def approach_candidate(
    mlx,
    model,
    out_dir,
    events_csv,
    confirmed_json,
    x,
    y,
    heading_deg,
    event_id,
    args,
    expected_labels,
):
    print("\nTHERMAL CANDIDATE FOUND — entering manual approach mode.")

    vision_positive_count = 0
    prev_bbox_height_frac = None
    last_forward_move_ft = None
    object_estimate = {}
    paired_rows = []
    confirmed_objects = []

    for step in range(1, args.max_approach_steps + 1):
        sample_name = f"approach_{event_id:04d}_{step:02d}_thermal"
        thermal = scan_thermal(mlx, out_dir, sample_name, args)

        if thermal.get("candidate"):
            bearing = float(thermal.get("bearing_deg", 0.0))
            turn_dir, turn_deg, heading_change = turn_instruction_from_bearing(bearing)

            if turn_dir != "CENTER":
                prompt_user(
                    f"APPROACH STEP {step}\n"
                    f"Thermal anomaly bearing = {bearing:.1f}°.\n"
                    f"Rotate Voyager {turn_dir} by about {turn_deg:.1f}°."
                )
                heading_deg = norm_heading(heading_deg + heading_change)
                set_current_pose(x, y, heading_deg)
            else:
                print(f"APPROACH STEP {step}: thermal anomaly is centered.")

        else:
            print(f"APPROACH STEP {step}: thermal candidate not visible in this frame.")

        raw_path = out_dir / "paired_frames" / f"approach_{event_id:04d}_{step:02d}_camera.jpg"
        ann_path = out_dir / "paired_frames" / f"approach_{event_id:04d}_{step:02d}_camera_annotated.jpg"

        try:
            capture_image(
                raw_path,
                timeout_ms=args.camera_timeout_ms,
                hflip=not args.no_hflip,
                vflip=not args.no_vflip,
            )
            positive, detections = run_yolo_on_image(
                model,
                raw_path,
                conf=args.conf,
                expected_labels=expected_labels,
            )
            annotate_image(raw_path, ann_path, detections)
            collision_risk, collision_info = vision_collision_guard(detections, raw_path, args)
            vision_error = ""
        except Exception as exc:
            positive = False
            detections = []
            collision_risk = False
            collision_info = {"bbox_area_frac": 0.0, "bbox_height_frac": 0.0, "bbox_bottom_frac": 0.0, "reason": ""}
            vision_error = f"{type(exc).__name__}: {exc}"

        object_estimate = {}
        if positive:
            vision_positive_count += 1
            object_estimate = estimate_object_location_from_vision(
                x, y, heading_deg, detections, raw_path, args,
                prev_bbox_height_frac=prev_bbox_height_frac,
                last_forward_move_ft=last_forward_move_ft,
            )
            if object_estimate:
                print(
                    "\nOBJECT LOCATION ESTIMATE: "
                    f"x={object_estimate.get('estimated_object_x_ft')} ft, "
                    f"y={object_estimate.get('estimated_object_y_ft')} ft, "
                    f"range={object_estimate.get('estimated_object_range_ft')} ft, "
                    f"bearing={object_estimate.get('estimated_object_bearing_deg')} deg, "
                    f"method={object_estimate.get('range_method')}"
                )
                if object_estimate.get("object_estimate_near_boundary"):
                    print("OBJECT ESTIMATE IS NEAR ROOM BOUNDARY — stop pursuit if needed and continue scan.")

        decision = ""
        if vision_positive_count >= args.vision_vote_k:
            decision = "confirmed_waste"

        row = {
            "time": now_s(),
            "event_id": event_id,
            "phase": "approach",
            "x_ft": round(x, 3),
            "y_ft": round(y, 3),
            "heading_deg": round(heading_deg, 2),
            "instruction": f"approach step {step}",
            "thermal_ok": thermal.get("ok"),
            "thermal_candidate": thermal.get("candidate"),
            "thermal_mode": thermal.get("selected_mode", args.mode),
            "thermal_delta_c": thermal.get("delta_c", ""),
            "thermal_bearing_deg": thermal.get("bearing_deg", ""),
            "thermal_blob_px": thermal.get("blob_px", ""),
            "thermal_centroid_x": thermal.get("centroid_x", ""),
            "thermal_centroid_y": thermal.get("centroid_y", ""),
            "thermal_preview_path": thermal.get("preview_path", ""),
            "vision_positive": positive,
            "vision_num_detections": len(detections),
            "vision_raw_path": str(raw_path),
            "vision_annotated_path": str(ann_path),
            "collision_risk": bool(collision_risk),
            "vision_bbox_area_frac": collision_info.get("bbox_area_frac", 0.0),
            "vision_bbox_height_frac": collision_info.get("bbox_height_frac", 0.0),
            "vision_bbox_bottom_frac": collision_info.get("bbox_bottom_frac", 0.0),
            "estimated_object_x_ft": object_estimate.get("estimated_object_x_ft", ""),
            "estimated_object_y_ft": object_estimate.get("estimated_object_y_ft", ""),
            "estimated_object_range_ft": object_estimate.get("estimated_object_range_ft", ""),
            "estimated_object_bearing_deg": object_estimate.get("estimated_object_bearing_deg", ""),
            "visual_offset_deg": object_estimate.get("visual_offset_deg", ""),
            "range_method": object_estimate.get("range_method", ""),
            "object_estimate_near_boundary": object_estimate.get("object_estimate_near_boundary", ""),
            "decision": decision,
            "notes": vision_error,
        }
        write_event(events_csv, row)
        paired_rows.append(row)

        print(
            f"[approach {step}] thermal={thermal.get('candidate')} "
            f"bearing={thermal.get('bearing_deg', '')} "
            f"vision_positive={positive} "
            f"vision_votes={vision_positive_count}/{args.vision_vote_k}"
        )

        if decision == "confirmed_waste":
            obj = {
                "time": now_s(),
                "event_id": event_id,
                "estimated_x_ft": round(x, 3),
                "estimated_y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "thermal": thermal,
                "vision_positive_count": vision_positive_count,
                "vision_vote_k": args.vision_vote_k,
                "decision": "confirmed_waste",
            }
            if object_estimate:
                obj.update(object_estimate)

            duplicate_obj, duplicate_dist = find_duplicate_confirmed(
                confirmed_json, obj, args.duplicate_radius_ft
            )

            if duplicate_obj is not None:
                matched_id = duplicate_obj.get("object_id", "")
                obj["duplicate_suppressed"] = True
                obj["matched_object_id"] = matched_id
                obj["matched_distance_ft"] = round(float(duplicate_dist), 3)
                obj["duplicate_radius_ft"] = args.duplicate_radius_ft
                append_duplicate_detection(confirmed_json, obj)

                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "approach",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "thermal_candidate": thermal.get("candidate"),
                    "thermal_bearing_deg": thermal.get("bearing_deg"),
                    "thermal_delta_c": thermal.get("delta_c"),
                    "thermal_blob_px": thermal.get("blob_px"),
                    "thermal_centroid_x": thermal.get("centroid_x", ""),
                    "thermal_centroid_y": thermal.get("centroid_y", ""),
                    "thermal_coverage_frac": thermal.get("coverage_frac"),
                    "thermal_preview_path": thermal.get("preview_path", ""),
                    "vision_positive": True,
                    "vision_num_detections": len(detections),
                    "vision_raw_path": str(raw_path),
                    "vision_annotated_path": str(ann_path),
                    "collision_risk": bool(collision_risk),
                    "vision_bbox_area_frac": collision_info.get("bbox_area_frac", 0.0),
                    "vision_bbox_height_frac": collision_info.get("bbox_height_frac", 0.0),
                    "vision_bbox_bottom_frac": collision_info.get("bbox_bottom_frac", 0.0),
                    "duplicate_status": "duplicate_suppressed",
                    "object_id": "",
                    "matched_object_id": matched_id,
                    "instruction": "duplicate target suppressed",
                    "decision": "duplicate_suppressed",
                    "notes": f"matched_object_id={matched_id}; distance_ft={duplicate_dist:.2f}; radius_ft={args.duplicate_radius_ft:.2f}",
                })

                print(
                    f"\nDUPLICATE SUPPRESSED: matched existing object_id={matched_id} "
                    f"at distance {duplicate_dist:.2f} ft. Not counting as a new object."
                )
                return x, y, heading_deg, []

            obj["object_id"] = next_object_id(confirmed_json)
            obj["duplicate_suppressed"] = False
            confirmed_objects.append(obj)

            existing = []
            if confirmed_json.exists():
                try:
                    existing = json.loads(confirmed_json.read_text())
                except Exception:
                    existing = []
            existing.extend(confirmed_objects)
            confirmed_json.write_text(json.dumps(existing, indent=2))

            print("\nCONFIRMED WASTE.")
            print(f"Estimated robot coordinate at confirmation: x={x:.2f} ft, y={y:.2f} ft")
            print(f"Saved confirmed object to: {confirmed_json}")
            return x, y, heading_deg, True

        move_step_ft = args.close_approach_step_ft if (positive or vision_positive_count > 0) else args.approach_step_ft

        out_of_bounds, next_x, next_y = next_pose_out_of_bounds(x, y, heading_deg, move_step_ft, args)
        if out_of_bounds:
            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id,
                "phase": "approach",
                "x_ft": round(x, 3),
                "y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "instruction": "boundary guard stop",
                "decision": "boundary_guard_stop_continue_scan",
                "notes": f"next move would leave test area: next_x={next_x:.2f}, next_y={next_y:.2f}",
            })
            print(
                f"\nBOUNDARY GUARD: next approach move would leave test area "
                f"(next x={next_x:.2f}, y={next_y:.2f}). Stop pursuit and continue scan."
            )
            return x, y, heading_deg, []

        prompt_user(
            f"Vision has not confirmed yet.\n"
            f"Move Voyager FORWARD {move_step_ft:.2f} ft toward the target."
        )
        x, y = move_pose(x, y, heading_deg, move_step_ft)
        last_forward_move_ft = move_step_ft

        if object_estimate and object_estimate.get("bbox_height_frac"):
            prev_bbox_height_frac = float(object_estimate.get("bbox_height_frac"))

    write_event(events_csv, {
        "time": now_s(),
        "event_id": event_id,
        "phase": "approach",
        "x_ft": round(x, 3),
        "y_ft": round(y, 3),
        "heading_deg": round(heading_deg, 2),
        "instruction": "max approach steps reached",
        "decision": "thermal_candidate_but_vision_rejected",
        "notes": f"vision_votes={vision_positive_count}/{args.vision_vote_k}",
    })

    print("\nThermal candidate was not confirmed by vision after max approach steps.")
    return x, y, heading_deg, False


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--label", default="manual_lawnmower_hil")
    ap.add_argument("--output-root", default="outputs/manual_navigation_tests")

    ap.add_argument("--room-length-ft", type=float, default=8.0)
    ap.add_argument("--room-width-ft", type=float, default=4.0)
    ap.add_argument("--lane-spacing-ft", type=float, default=2.0)
    ap.add_argument("--coverage-step-ft", type=float, default=1.0)

    ap.add_argument("--approach-step-ft", type=float, default=0.75)
    ap.add_argument("--close-approach-step-ft", type=float, default=0.5)
    ap.add_argument("--return-step-ft", type=float, default=2.0)
    ap.add_argument("--max-approach-steps", type=int, default=6)

    ap.add_argument("--mode", choices=["warm", "cold", "both"], default="cold")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--min-blob-pixels", type=int, default=6)
    ap.add_argument("--fov-x-deg", type=float, default=55.0)
    ap.add_argument("--read-retries", type=int, default=8)
    ap.add_argument("--read-retry-delay-sec", type=float, default=0.25)

    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--conf", type=float, default=0.03)
    ap.add_argument("--vision-vote-k", type=int, default=2)

    ap.add_argument("--collision-bbox-area-frac", type=float, default=0.15)
    ap.add_argument("--collision-bbox-height-frac", type=float, default=0.35)
    ap.add_argument("--collision-bbox-bottom-frac", type=float, default=0.80)
    ap.add_argument("--duplicate-radius-ft", type=float, default=2.5)
    ap.add_argument("--vision-horizontal-fov-deg", type=float, default=66.0)
    ap.add_argument("--fallback-object-distance-ft", type=float, default=3.5)
    ap.add_argument("--min-estimated-object-distance-ft", type=float, default=0.25)
    ap.add_argument("--max-estimated-object-distance-ft", type=float, default=6.0)
    ap.add_argument("--min-bbox-growth-ratio", type=float, default=1.08)
    ap.add_argument("--boundary-margin-ft", type=float, default=0.15)
    ap.add_argument("--expected-labels", default="waste,bottle,plastic,plastic_bottle")

    ap.add_argument("--camera-timeout-ms", type=int, default=1500)
    ap.add_argument("--no-hflip", action="store_true")
    ap.add_argument("--no-vflip", action="store_true")

    ap.add_argument("--start-x-ft", type=float, default=0.0)
    ap.add_argument("--start-y-ft", type=float, default=0.0)
    ap.add_argument("--start-heading-deg", type=float, default=0.0)

    ap.add_argument("--clear", action="store_true")

    args = ap.parse_args()

    out_dir = Path(args.output_root) / args.label

    if args.clear and out_dir.exists():
        import shutil
        shutil.rmtree(out_dir)

    for d in [
        out_dir,
        out_dir / "thermal_frames",
        out_dir / "thermal_preview",
        out_dir / "paired_frames",
    ]:
        d.mkdir(parents=True, exist_ok=True)

    events_csv = out_dir / "mission_events.csv"
    confirmed_json = out_dir / "confirmed_objects.json"
    summary_json = out_dir / "summary.json"

    expected_labels = {
        x.strip().lower()
        for x in args.expected_labels.split(",")
        if x.strip()
    }

    print("=== Manual Navigation HIL ===")
    print("You physically move the robot. The script only gives instructions.")
    print("No motors are controlled by this script.")
    print()
    print(f"Room/map length: {args.room_length_ft} ft")
    print(f"Room/map width:  {args.room_width_ft} ft")
    print(f"Lane spacing:    {args.lane_spacing_ft} ft")
    print(f"Coverage step:   {args.coverage_step_ft} ft")
    print(f"Start coordinate: x={args.start_x_ft}, y={args.start_y_ft}")
    print(f"Start heading:    {args.start_heading_deg}°")
    print(f"Output:           {out_dir}")

    prompt_user(
        "Place Voyager at the START point.\n"
        "Point it along the first lawnmower lane.\n"
        "Coordinate convention: x = first forward direction, y = lane-to-lane direction."
    )

    print("\nInitializing thermal camera...")
    i2c = busio.I2C(board.SCL, board.SDA)
    mlx = adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = adafruit_mlx90640.RefreshRate.REFRESH_1_HZ

    print("Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO(args.model)

    x = args.start_x_ft
    y = args.start_y_ft
    heading_deg = norm_heading(args.start_heading_deg)
    set_current_pose(x, y, heading_deg)

    commands = generate_lawnmower_plan(
        args.room_length_ft,
        args.room_width_ft,
        args.lane_spacing_ft,
        args.coverage_step_ft,
    )

    event_id = 0
    confirmed_count = 0
    started = time.time()

    try:
        for idx, cmd in enumerate(commands, start=1):
            event_id += 1

            sample_name = f"coverage_{event_id:04d}_thermal"
            thermal = scan_thermal(mlx, out_dir, sample_name, args)

            print(
                f"\n[coverage scan {event_id}] "
                f"pos=({x:.2f}, {y:.2f}) heading={heading_deg:.1f}° "
                f"candidate={thermal.get('candidate')} "
                f"bearing={thermal.get('bearing_deg', '')} "
                f"delta={thermal.get('delta_c', '')}"
            )

            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id,
                "phase": "coverage_scan",
                "x_ft": round(x, 3),
                "y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "instruction": "thermal scan before movement",
                "thermal_ok": thermal.get("ok"),
                "thermal_candidate": thermal.get("candidate"),
                "thermal_mode": thermal.get("selected_mode", args.mode),
                "thermal_delta_c": thermal.get("delta_c", ""),
                "thermal_bearing_deg": thermal.get("bearing_deg", ""),
                "thermal_blob_px": thermal.get("blob_px", ""),
                "thermal_centroid_x": thermal.get("centroid_x", ""),
                "thermal_centroid_y": thermal.get("centroid_y", ""),
                "thermal_preview_path": thermal.get("preview_path", ""),
                "decision": "thermal_candidate" if thermal.get("candidate") else "no_candidate",
            })

            if thermal.get("candidate"):
                # Save exact lawnmower pose before candidate investigation.
                coverage_x = x
                coverage_y = y
                coverage_heading_deg = heading_deg

                x, y, heading_deg, confirmed = approach_candidate(
                    mlx=mlx,
                    model=model,
                    out_dir=out_dir,
                    events_csv=events_csv,
                    confirmed_json=confirmed_json,
                    x=x,
                    y=y,
                    heading_deg=heading_deg,
                    event_id=event_id,
                    args=args,
                    expected_labels=expected_labels,
                )

                if confirmed:
                    confirmed_count += 1

                prompt_user(
                    "Return Voyager to the last lawnmower coverage position if needed.\n"
                    "Then align it with the planned lawnmower heading and continue."
                )

                # Operator has returned Voyager to the saved lawnmower pose.
                # Restore the software pose so coverage does not accumulate
                # offsets introduced during thermal/vision investigation.
                x = coverage_x
                y = coverage_y
                heading_deg = coverage_heading_deg
                set_current_pose(x, y, heading_deg)

                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "coverage_resume",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "instruction": "returned to saved lawnmower coverage pose",
                    "decision": "resume_coverage",
                    "notes": "software pose restored after candidate investigation",
                })

            if cmd["type"] == "rotate":
                prompt_user(cmd["label"])
                heading_deg = norm_heading(heading_deg + cmd["heading_delta"])
                set_current_pose(x, y, heading_deg)

            elif cmd["type"] == "move":
                # Hard safety guard for lawnmower coverage movement.
                out_of_bounds, next_x, next_y = next_pose_out_of_bounds(
                    x,
                    y,
                    heading_deg,
                    cmd["distance_ft"],
                    args,
                )

                if out_of_bounds:
                    write_event(events_csv, {
                        "time": now_s(),
                        "event_id": event_id,
                        "phase": "coverage_boundary_guard",
                        "x_ft": round(x, 3),
                        "y_ft": round(y, 3),
                        "heading_deg": round(heading_deg, 2),
                        "instruction": cmd["label"],
                        "decision": "coverage_move_blocked",
                        "notes": (
                            f"planned move would leave test area: "
                            f"next_x={next_x:.2f}, next_y={next_y:.2f}"
                        ),
                    })

                    print(
                        f"\nCOVERAGE BOUNDARY GUARD: refusing move. "
                        f"Next pose would be x={next_x:.2f}, y={next_y:.2f}."
                    )
                    continue

                prompt_user(cmd["label"])
                x, y = move_pose(x, y, heading_deg, cmd["distance_ft"])
                set_current_pose(x, y, heading_deg)
            summary = {
                "label": args.label,
                "output_dir": str(out_dir),
                "current_x_ft": round(x, 3),
                "current_y_ft": round(y, 3),
                "current_heading_deg": round(heading_deg, 2),
                "commands_completed": idx,
                "total_commands": len(commands),
                "confirmed_count": confirmed_count,
                "elapsed_sec": round(time.time() - started, 2),
                "events_csv": str(events_csv),
                "confirmed_objects_json": str(confirmed_json),
            }
            summary_json.write_text(json.dumps(summary, indent=2))

    except ReturnHomeRequested as exc:
        print("\nReturn-home requested by user.")

        rx = exc.x_ft if exc.x_ft is not None else x
        ry = exc.y_ft if exc.y_ft is not None else y
        rh = exc.heading_deg if exc.heading_deg is not None else heading_deg

        x, y, heading_deg, event_id = guide_return_home(
            x=rx,
            y=ry,
            heading_deg=rh,
            home_x=args.start_x_ft,
            home_y=args.start_y_ft,
            step_ft=args.return_step_ft,
            events_csv=events_csv,
            event_id=event_id,
        )

    except KeyboardInterrupt:
        print("\nStopped by user with Ctrl-C. Return-home guide was not run.")

    final_summary = {
        "label": args.label,
        "output_dir": str(out_dir),
        "final_x_ft": round(x, 3),
        "final_y_ft": round(y, 3),
        "final_heading_deg": round(heading_deg, 2),
        "confirmed_count": confirmed_count,
        "elapsed_sec": round(time.time() - started, 2),
        "events_csv": str(events_csv),
        "confirmed_objects_json": str(confirmed_json),
    }
    summary_json.write_text(json.dumps(final_summary, indent=2))

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
