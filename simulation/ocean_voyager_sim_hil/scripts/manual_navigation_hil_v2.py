#!/usr/bin/env python3

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import manual_navigation_hil as base


class ReturnHomeRequested(Exception):
    pass


FIELDNAMES = [
    "time",
    "event_id",
    "phase",
    "x_ft",
    "y_ft",
    "heading_deg",
    "active_lane_x_ft",
    "active_lane_y_ft",
    "planned_heading_deg",
    "instruction",
    "thermal_ok",
    "thermal_candidate",
    "thermal_mode",
    "thermal_delta_c",
    "thermal_bearing_deg",
    "thermal_blob_px",
    "thermal_preview_path",
    "vision_positive",
    "vision_num_detections",
    "vision_best_label",
    "vision_best_conf",
    "vision_raw_path",
    "vision_annotated_path",
    "estimated_object_x_ft",
    "estimated_object_y_ft",
    "estimated_object_range_ft",
    "estimated_object_bearing_deg",
    "visual_offset_deg",
    "range_method",
    "collision_clearance_ft",
    "collision_risk",
    "collision_reason",
    "duplicate_status",
    "object_id",
    "matched_object_id",
    "rejected_status",
    "detour_status",
    "decision",
    "notes",
]


def now_s():
    return base.now_s() if hasattr(base, "now_s") else time.strftime("%Y-%m-%dT%H:%M:%S")


def norm_heading(deg):
    return base.norm_heading(deg)


def move_pose(x, y, heading_deg, distance_ft):
    return base.move_pose(x, y, heading_deg, distance_ft)


def set_pose(x, y, heading_deg):
    if hasattr(base, "set_current_pose"):
        base.set_current_pose(x, y, heading_deg)


def signed_turn_delta(current_deg, target_deg):
    """
    Positive means turn LEFT/CCW.
    Negative means turn RIGHT/CW.
    """
    return ((target_deg - current_deg + 180.0) % 360.0) - 180.0


def turn_text(delta_deg):
    if abs(delta_deg) < 1.0:
        return "CENTER", 0.0
    if delta_deg > 0:
        return "LEFT", abs(delta_deg)
    return "RIGHT", abs(delta_deg)


def prompt_user(message):
    print("\n" + "=" * 72)
    print(message)
    print("=" * 72)
    ans = input("Press ENTER when done, or type q/quit/home/return: ").strip().lower()
    if ans in {"q", "quit", "home", "return"}:
        raise ReturnHomeRequested()


def write_event(events_csv, row):
    exists = events_csv.exists()
    with events_csv.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def append_json_list(path, obj):
    path = Path(path)
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(obj)
    path.write_text(json.dumps(existing, indent=2))


def detection_label(det):
    for k in ["label", "name", "class_name", "cls_name", "class"]:
        v = det.get(k)
        if v not in (None, ""):
            return str(v).strip().lower()
    return ""


def detection_conf(det):
    for k in ["confidence", "conf", "score"]:
        try:
            return float(det.get(k))
        except Exception:
            pass
    return 0.0


def best_detection_summary(detections):
    if not detections:
        return "", ""
    best = max(detections, key=detection_conf)
    return detection_label(best), round(detection_conf(best), 5)


def has_non_expected_detection(detections, expected_labels):
    """
    Reject immediately only if the detector gives a clear label
    that is not in expected waste labels.
    """
    for det in detections:
        lab = detection_label(det)
        if lab and lab not in expected_labels:
            return True, lab
    return False, ""


def add_collision_clearance(object_estimate, args):
    if not object_estimate:
        return object_estimate

    try:
        rng = float(object_estimate.get("estimated_object_range_ft", ""))
    except Exception:
        return object_estimate

    clearance = rng - float(args.collision_safety_buffer_ft)
    object_estimate["collision_clearance_ft"] = round(clearance, 3)
    return object_estimate



def calibrate_fallback_object_range(est, x, y, heading_deg, args):
    """
    BBox-size-only range is not calibrated yet for this camera/boat setup.
    Scale it upward and mark it as calibrated fallback, not true range.

    This prevents a visually visible object 3.5+ ft away from being treated
    as a near-collision just because the bbox is large.
    """
    if not est:
        return est

    method = str(est.get("range_method", ""))
    if method != "fallback_bbox_size":
        return est

    try:
        old_range = float(est.get("estimated_object_range_ft", ""))
        bearing = float(est.get("estimated_object_bearing_deg", heading_deg))
    except Exception:
        return est

    multiplier = float(getattr(args, "fallback_range_multiplier_ft", 2.5))
    min_range = float(getattr(args, "fallback_min_range_ft", 3.0))
    max_range = float(getattr(args, "max_estimated_object_distance_ft", 6.0))

    new_range = max(min_range, min(max_range, old_range * multiplier))

    rad = math.radians(bearing)
    raw_x = float(x) + new_range * math.cos(rad)
    raw_y = float(y) + new_range * math.sin(rad)

    length = float(getattr(args, "room_length_ft", 8.0))
    width = float(getattr(args, "room_width_ft", 8.0))
    margin = float(getattr(args, "boundary_margin_ft", 0.15))

    obj_x = max(0.0, min(length, raw_x))
    obj_y = max(0.0, min(width, raw_y))

    near_boundary = (
        obj_x <= margin or obj_x >= length - margin or
        obj_y <= margin or obj_y >= width - margin
    )

    est["fallback_original_range_ft"] = round(old_range, 3)
    est["estimated_object_range_ft"] = round(new_range, 3)
    est["raw_estimated_object_x_ft"] = round(raw_x, 3)
    est["raw_estimated_object_y_ft"] = round(raw_y, 3)
    est["estimated_object_x_ft"] = round(obj_x, 3)
    est["estimated_object_y_ft"] = round(obj_y, 3)
    est["range_method"] = "fallback_bbox_size_calibrated"
    est["object_estimate_near_boundary"] = bool(near_boundary)
    return est


def collision_stop_required(collision_risk, object_estimate, args):
    """
    v2.3:
    Stop is based primarily on estimated clearance, not just collision_risk.

    collision_risk can miss cases when the object is slightly off-center,
    but if the object is visible and clearance is low, normal forward motion
    must be blocked.
    """
    if not object_estimate:
        return False

    try:
        clearance = float(object_estimate.get("collision_clearance_ft", "999"))
    except Exception:
        clearance = 999.0

    method = str(object_estimate.get("range_method", ""))

    if method.startswith("fallback_bbox_size"):
        stop_threshold = float(getattr(args, "fallback_collision_stop_ft", 0.75))
    else:
        stop_threshold = float(getattr(args, "collision_clearance_stop_ft", 0.75))

    if clearance <= stop_threshold:
        return True

    # If bbox/forward-zone logic says risk, obey normal threshold.
    if collision_risk and clearance <= float(getattr(args, "collision_clearance_stop_ft", 0.75)):
        return True

    return False

def estimate_object(detections, raw_path, x, y, heading_deg, args, prev_bbox_height_frac=None, last_forward_move_ft=None):
    if not detections:
        return {}

    if hasattr(base, "estimate_object_location_from_vision"):
        est = base.estimate_object_location_from_vision(
            x,
            y,
            heading_deg,
            detections,
            raw_path,
            args,
            prev_bbox_height_frac=prev_bbox_height_frac,
            last_forward_move_ft=last_forward_move_ft,
        )
    else:
        est = {}

    est = calibrate_fallback_object_range(est, x, y, heading_deg, args)
    return add_collision_clearance(est, args)


def conservative_collision_risk(detections, raw_path, object_estimate, args):
    """
    Conservative image-based collision guard.

    Triggers if object is in the forward image zone and either:
    - bbox bottom is low in the image,
    - bbox height is moderately large,
    - estimated collision clearance is small.
    """
    if not detections:
        return False, {
            "collision_reason": "no_detections",
            "bbox_area_frac": "",
            "bbox_height_frac": "",
            "bbox_bottom_frac": "",
        }

    geom = None
    if hasattr(base, "best_detection_geometry"):
        try:
            geom = base.best_detection_geometry(detections, raw_path)
        except Exception:
            geom = None

    reasons = []
    in_forward_zone = False
    height_frac = 0.0
    bottom_frac = 0.0
    area_frac = 0.0

    if geom:
        cx = float(geom.get("bbox_center_x_frac", 0.5))
        height_frac = float(geom.get("bbox_height_frac", 0.0))
        bottom_frac = float(geom.get("bbox_bottom_frac", 0.0))
        area_frac = float(geom.get("bbox_area_frac", 0.0))

        in_forward_zone = (
            float(args.forward_zone_left_frac)
            <= cx
            <= float(args.forward_zone_right_frac)
        )

        if in_forward_zone:
            reasons.append("object_in_forward_path_zone")
        if bottom_frac >= float(args.collision_bbox_bottom_frac):
            reasons.append("bbox_low_in_image")
        if height_frac >= float(args.collision_bbox_height_frac):
            reasons.append("bbox_moderately_large")

    clearance = None
    try:
        clearance = float(object_estimate.get("collision_clearance_ft", ""))
    except Exception:
        clearance = None

    if clearance is not None and clearance <= float(args.collision_clearance_stop_ft):
        reasons.append(f"low_collision_clearance_{clearance:.2f}ft")

    risk = False
    if in_forward_zone and (
        bottom_frac >= float(args.collision_bbox_bottom_frac)
        or height_frac >= float(args.collision_bbox_height_frac)
        or (clearance is not None and clearance <= float(args.collision_clearance_stop_ft))
    ):
        risk = True

    return risk, {
        "collision_reason": "; ".join(reasons) if reasons else "not_in_collision_zone",
        "bbox_area_frac": round(area_frac, 5),
        "bbox_height_frac": round(height_frac, 5),
        "bbox_bottom_frac": round(bottom_frac, 5),
    }


def print_object_report(prefix, object_estimate, collision_risk, action):
    if not object_estimate:
        print(f"\n{prefix}: no object coordinate estimate available.")
        return

    print(f"\n{prefix}")
    print(f"Object x/y:              x={object_estimate.get('estimated_object_x_ft')} ft, y={object_estimate.get('estimated_object_y_ft')} ft")
    print(f"Range from Voyager:      {object_estimate.get('estimated_object_range_ft')} ft")
    print(f"Collision clearance:     {object_estimate.get('collision_clearance_ft', '')} ft")
    print(f"Bearing from Voyager:    {object_estimate.get('estimated_object_bearing_deg')} deg")
    print(f"Method:                  {object_estimate.get('range_method')}")
    print(f"Collision risk:          {'YES' if collision_risk else 'NO'}")
    print(f"Recommended action:      {action}")


def capture_and_detect(model, out_dir, sample_name, args, expected_labels):
    raw_path = out_dir / "paired_frames" / f"{sample_name}_camera.jpg"
    ann_path = out_dir / "paired_frames" / f"{sample_name}_camera_annotated.jpg"

    try:
        base.capture_image(
            raw_path,
            timeout_ms=args.camera_timeout_ms,
            hflip=not args.no_hflip,
            vflip=not args.no_vflip,
        )
        positive, detections = base.run_yolo_on_image(
            model,
            raw_path,
            conf=args.conf,
            expected_labels=expected_labels,
        )
        base.annotate_image(raw_path, ann_path, detections)
        error = ""
    except Exception as exc:
        positive = False
        detections = []
        error = f"{type(exc).__name__}: {exc}"

    best_label, best_conf = best_detection_summary(detections)
    return {
        "raw_path": raw_path,
        "ann_path": ann_path,
        "positive": bool(positive),
        "detections": detections,
        "best_label": best_label,
        "best_conf": best_conf,
        "error": error,
    }



def estimate_uses_fallback(obj):
    method = str(obj.get("range_method", ""))
    return "fallback" in method or bool(obj.get("object_estimate_clamped_to_boundary", False))


def find_duplicate_confirmed_v2(confirmed_json, obj, args):
    """
    Duplicate matching with a wider radius for fallback/boundary-clamped estimates.
    Normal bbox-expansion estimates use duplicate_radius_ft.
    Fallback estimates use fallback_duplicate_radius_ft.
    """
    existing = base.load_json_list(confirmed_json) if hasattr(base, "load_json_list") else []
    if not existing:
        return None, None, None

    if hasattr(base, "_object_xy_for_duplicate"):
        x, y = base._object_xy_for_duplicate(obj)
    else:
        x = float(obj.get("estimated_object_x_ft", obj.get("robot_x_ft", 0.0)))
        y = float(obj.get("estimated_object_y_ft", obj.get("robot_y_ft", 0.0)))

    best = None
    best_d = None
    best_radius = None

    for old in existing:
        if old.get("duplicate_suppressed"):
            continue

        if hasattr(base, "_object_xy_for_duplicate"):
            ox, oy = base._object_xy_for_duplicate(old)
        else:
            ox = float(old.get("estimated_object_x_ft", old.get("robot_x_ft", 0.0)))
            oy = float(old.get("estimated_object_y_ft", old.get("robot_y_ft", 0.0)))

        d = math.hypot(x - ox, y - oy)

        radius = float(args.duplicate_radius_ft)
        if estimate_uses_fallback(obj) or estimate_uses_fallback(old):
            radius = float(args.fallback_duplicate_radius_ft)

        if best_d is None or d < best_d:
            best = old
            best_d = d
            best_radius = radius

    if best is not None and best_d is not None and best_d <= best_radius:
        return best, best_d, best_radius

    return None, best_d, best_radius


def object_record_base(x, y, heading_deg, object_estimate, thermal, vision, decision, reason):
    obj = {
        "time": now_s(),
        "robot_x_ft": round(x, 3),
        "robot_y_ft": round(y, 3),
        "heading_deg": round(heading_deg, 2),
        "decision": decision,
        "reason": reason,
        "thermal_delta_c": thermal.get("delta_c", ""),
        "thermal_bearing_deg": thermal.get("bearing_deg", ""),
        "thermal_blob_px": thermal.get("blob_px", ""),
        "thermal_preview_path": thermal.get("preview_path", ""),
        "vision_raw_path": str(vision.get("raw_path", "")),
        "vision_annotated_path": str(vision.get("ann_path", "")),
        "vision_best_label": vision.get("best_label", ""),
        "vision_best_conf": vision.get("best_conf", ""),
        "vision_num_detections": len(vision.get("detections", [])),
    }
    if object_estimate:
        obj.update(object_estimate)
    return obj


def approach_candidate_v2(
    mlx,
    model,
    out_dir,
    events_csv,
    confirmed_json,
    rejected_json,
    x,
    y,
    heading_deg,
    event_id,
    args,
    expected_labels,
    active_lane_x,
    active_lane_y,
    planned_heading_deg,
):
    print("\nTHERMAL CANDIDATE FOUND — entering manual approach mode.")

    waste_votes = 0
    visual_negative_count = 0
    prev_bbox_height_frac = None
    last_forward_move_ft = None
    last_thermal = {}
    last_vision = {"detections": []}
    object_estimate = {}
    collision_hold_frames = 0

    for step in range(1, args.max_approach_steps + 1):
        sample_name = f"approach_{event_id:04d}_{step:02d}_thermal"
        thermal = base.scan_thermal(mlx, out_dir, sample_name, args)
        last_thermal = thermal

        if thermal.get("candidate"):
            bearing = float(thermal.get("bearing_deg", 0.0))
            turn_dir, turn_deg, heading_change = base.turn_instruction_from_bearing(bearing)
            if turn_dir != "CENTER":
                prompt_user(
                    f"APPROACH STEP {step}\n"
                    f"Thermal anomaly bearing = {bearing:.1f}°.\n"
                    f"Rotate Voyager {turn_dir} by about {turn_deg:.1f}°."
                )
                heading_deg = norm_heading(heading_deg + heading_change)
                set_pose(x, y, heading_deg)
            else:
                print(f"APPROACH STEP {step}: thermal anomaly is centered.")
        else:
            print(f"APPROACH STEP {step}: thermal candidate not visible in this frame.")

        vision = capture_and_detect(
            model,
            out_dir,
            f"approach_{event_id:04d}_{step:02d}",
            args,
            expected_labels,
        )
        last_vision = vision

        positive = vision["positive"]
        detections = vision["detections"]
        non_expected, non_expected_label = has_non_expected_detection(detections, expected_labels)

        object_estimate = estimate_object(
            detections,
            vision["raw_path"],
            x,
            y,
            heading_deg,
            args,
            prev_bbox_height_frac=prev_bbox_height_frac,
            last_forward_move_ft=last_forward_move_ft,
        )

        collision_risk, collision_info = conservative_collision_risk(
            detections,
            vision["raw_path"],
            object_estimate,
            args,
        )

        if collision_stop_required(collision_risk, object_estimate, args):
            collision_hold_frames = max(
                collision_hold_frames,
                int(args.collision_hold_verify_frames),
            )

        if positive and object_estimate:
            candidate_obj = object_record_base(
                x, y, heading_deg, object_estimate, thermal, vision,
                "candidate_duplicate_check",
                "early_positive_duplicate_check",
            )
            early_duplicate_obj, early_duplicate_dist, early_duplicate_radius = find_duplicate_confirmed_v2(
                confirmed_json,
                candidate_obj,
                args,
            )

            if early_duplicate_obj is not None:
                matched_id = early_duplicate_obj.get("object_id", "")
                print(
                    f"\nKNOWN CONFIRMED OBJECT SEEN: matched object_id={matched_id} "
                    f"at distance {early_duplicate_dist:.2f} ft "
                    f"(radius {early_duplicate_radius:.2f} ft). Suppressing duplicate."
                )
                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "approach",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "vision_positive": True,
                    "vision_num_detections": len(detections),
                    "vision_best_label": vision.get("best_label", ""),
                    "vision_best_conf": vision.get("best_conf", ""),
                    "vision_raw_path": str(vision["raw_path"]),
                    "vision_annotated_path": str(vision["ann_path"]),
                    "estimated_object_x_ft": object_estimate.get("estimated_object_x_ft", ""),
                    "estimated_object_y_ft": object_estimate.get("estimated_object_y_ft", ""),
                    "estimated_object_range_ft": object_estimate.get("estimated_object_range_ft", ""),
                    "estimated_object_bearing_deg": object_estimate.get("estimated_object_bearing_deg", ""),
                    "range_method": object_estimate.get("range_method", ""),
                    "collision_clearance_ft": object_estimate.get("collision_clearance_ft", ""),
                    "duplicate_status": "early_duplicate_suppressed",
                    "matched_object_id": matched_id,
                    "decision": "duplicate_suppressed",
                    "notes": f"distance_ft={early_duplicate_dist:.2f}; radius_ft={early_duplicate_radius:.2f}",
                })
                return x, y, heading_deg, "duplicate_suppressed"

        if positive:
            waste_votes += 1
        else:
            visual_negative_count += 1

        if object_estimate:
            print_object_report(
                "OBJECT LOCATION ESTIMATE",
                object_estimate,
                collision_risk,
                "continue approach" if not collision_risk else "stop / verify from same position",
            )

        decision = ""
        notes = vision["error"]

        if positive and waste_votes >= args.vision_vote_k:
            decision = "confirmed_waste"
        elif non_expected:
            decision = "rejected_non_waste"
            notes = f"non_expected_label={non_expected_label}"
        else:
            reject_threshold = args.non_waste_negative_votes
            try:
                clearance = float(object_estimate.get("collision_clearance_ft", "999"))
            except Exception:
                clearance = 999.0

            if collision_risk or clearance <= args.collision_clearance_stop_ft:
                reject_threshold = min(reject_threshold, args.close_non_waste_negative_votes)

            if visual_negative_count >= reject_threshold:
                decision = "rejected_non_waste"
                notes = f"no_expected_waste_label_{visual_negative_count}_frames"

        write_event(events_csv, {
            "time": now_s(),
            "event_id": event_id,
            "phase": "approach",
            "x_ft": round(x, 3),
            "y_ft": round(y, 3),
            "heading_deg": round(heading_deg, 2),
            "active_lane_x_ft": round(active_lane_x, 3),
            "active_lane_y_ft": round(active_lane_y, 3),
            "planned_heading_deg": round(planned_heading_deg, 2),
            "instruction": f"approach step {step}",
            "thermal_ok": thermal.get("ok"),
            "thermal_candidate": thermal.get("candidate"),
            "thermal_mode": thermal.get("selected_mode", args.mode),
            "thermal_delta_c": thermal.get("delta_c", ""),
            "thermal_bearing_deg": thermal.get("bearing_deg", ""),
            "thermal_blob_px": thermal.get("blob_px", ""),
            "thermal_preview_path": thermal.get("preview_path", ""),
            "vision_positive": positive,
            "vision_num_detections": len(detections),
            "vision_best_label": vision.get("best_label", ""),
            "vision_best_conf": vision.get("best_conf", ""),
            "vision_raw_path": str(vision["raw_path"]),
            "vision_annotated_path": str(vision["ann_path"]),
            "estimated_object_x_ft": object_estimate.get("estimated_object_x_ft", ""),
            "estimated_object_y_ft": object_estimate.get("estimated_object_y_ft", ""),
            "estimated_object_range_ft": object_estimate.get("estimated_object_range_ft", ""),
            "estimated_object_bearing_deg": object_estimate.get("estimated_object_bearing_deg", ""),
            "visual_offset_deg": object_estimate.get("visual_offset_deg", ""),
            "range_method": object_estimate.get("range_method", ""),
            "collision_clearance_ft": object_estimate.get("collision_clearance_ft", ""),
            "collision_risk": bool(collision_risk),
            "collision_reason": collision_info.get("collision_reason", ""),
            "decision": decision,
            "notes": notes,
        })

        print(
            f"[approach {step}] thermal={thermal.get('candidate')} "
            f"bearing={thermal.get('bearing_deg', '')} "
            f"vision_positive={positive} "
            f"vision_votes={waste_votes}/{args.vision_vote_k} "
            f"visual_negatives={visual_negative_count}"
        )

        if decision == "confirmed_waste":
            obj = object_record_base(
                x, y, heading_deg, object_estimate, thermal, vision,
                "confirmed_waste",
                "vision_votes_met",
            )

            duplicate_obj, duplicate_dist, duplicate_radius = find_duplicate_confirmed_v2(
                confirmed_json,
                obj,
                args,
            )

            if duplicate_obj is not None:
                matched_id = duplicate_obj.get("object_id", "")
                obj["duplicate_suppressed"] = True
                obj["matched_object_id"] = matched_id
                obj["matched_distance_ft"] = round(float(duplicate_dist), 3)
                obj["duplicate_radius_ft"] = duplicate_radius if duplicate_radius is not None else args.duplicate_radius_ft

                if hasattr(base, "append_duplicate_detection"):
                    base.append_duplicate_detection(confirmed_json, obj)
                else:
                    append_json_list(Path(confirmed_json).with_name("duplicate_detections.json"), obj)

                print(
                    f"\nDUPLICATE SUPPRESSED: matched existing object_id={matched_id} "
                    f"at distance {duplicate_dist:.2f} ft. Not counting as new object."
                )

                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "approach",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "duplicate_status": "duplicate_suppressed",
                    "matched_object_id": matched_id,
                    "decision": "duplicate_suppressed",
                    "notes": f"distance_ft={duplicate_dist:.2f}",
                })
                return x, y, heading_deg, "duplicate_suppressed"

            obj["object_id"] = base.next_object_id(confirmed_json) if hasattr(base, "next_object_id") else 1
            obj["duplicate_suppressed"] = False
            append_json_list(confirmed_json, obj)

            print("\nCONFIRMED WASTE.")
            print(f"Saved confirmed object to: {confirmed_json}")
            return x, y, heading_deg, "confirmed_waste"

        if decision == "rejected_non_waste":
            obj = object_record_base(
                x, y, heading_deg, object_estimate, thermal, vision,
                "rejected_non_waste",
                notes or "visual_non_confirmation",
            )
            append_json_list(rejected_json, obj)

            print("\nREJECTED NON-WASTE.")
            print(f"Reason: {obj.get('reason')}")
            print(f"Saved rejected object to: {rejected_json}")
            return x, y, heading_deg, "rejected_non_waste"

        if collision_risk and not collision_stop_required(collision_risk, object_estimate, args):
            print(
                "\nCOLLISION CAUTION: object is visible/in path zone, "
                "but estimated clearance is not low enough for hard stop. "
                "Proceeding only with short/controlled approach."
            )

        if collision_stop_required(collision_risk, object_estimate, args):
            prompt_user(
                "COLLISION STOP: object appears very close/in path.\n"
                "Do NOT move forward. Press ENTER to take another verification image "
                "from the same position."
            )
            continue

        if collision_hold_frames > 0 and not detections:
            collision_hold_frames -= 1
            prompt_user(
                "COLLISION GUARD STILL ACTIVE: a close/path object was seen recently, "
                "but this frame missed it.\n"
                "Do NOT move forward yet. Press ENTER to take another verification image "
                "from the same position."
            )
            continue

        move_step_ft = args.close_approach_step_ft if (positive or detections) else args.approach_step_ft
        out_of_bounds, next_x, next_y = base.next_pose_out_of_bounds(x, y, heading_deg, move_step_ft, args)

        if out_of_bounds:
            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id,
                "phase": "approach",
                "x_ft": round(x, 3),
                "y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "decision": "boundary_guard_stop_continue_scan",
                "notes": f"next_x={next_x:.2f}, next_y={next_y:.2f}",
            })
            print(
                f"\nBOUNDARY GUARD: next approach move would leave test area "
                f"(next x={next_x:.2f}, y={next_y:.2f}). Stop pursuit and continue scan."
            )
            return x, y, heading_deg, "boundary_guard_stop"

        prompt_user(f"Move Voyager FORWARD {move_step_ft:.2f} ft toward the target.")
        x, y = move_pose(x, y, heading_deg, move_step_ft)
        set_pose(x, y, heading_deg)
        last_forward_move_ft = move_step_ft

        if object_estimate and object_estimate.get("bbox_height_frac"):
            prev_bbox_height_frac = float(object_estimate.get("bbox_height_frac"))

    obj = object_record_base(
        x, y, heading_deg, object_estimate, last_thermal, last_vision,
        "rejected_non_waste",
        "max_approach_steps_without_waste_confirmation",
    )
    append_json_list(rejected_json, obj)

    print("\nREJECTED NON-WASTE after max approach steps without enough waste votes.")
    return x, y, heading_deg, "rejected_non_waste"


def rejoin_lane_and_realign(x, y, heading_deg, lane_x, lane_y, planned_heading_deg, args):
    """
    v2.1 behavior:
    Heading-only rolling continuation.

    Do NOT force lateral return to the old lane coordinate.
    Manual HIL lateral dead-reckoning is too noisy, so we continue from
    current estimated pose and only realign heading.
    """
    theta = math.radians(planned_heading_deg)
    left_vec = (-math.sin(theta), math.cos(theta))

    dx = x - lane_x
    dy = y - lane_y
    cross_track = dx * left_vec[0] + dy * left_vec[1]

    print()
    print("TARGET HANDLED — ROLLING CONTINUATION MODE")
    print("Do NOT manually return Voyager to the previous coverage point.")
    print(f"Current estimated pose: x={x:.2f}, y={y:.2f}, heading={heading_deg:.1f}°")
    print(f"Active lane reference:  x={lane_x:.2f}, y={lane_y:.2f}, heading={planned_heading_deg:.1f}°")

    if abs(cross_track) > args.rejoin_lane_tolerance_ft:
        print(
            f"Lane offset estimate: {abs(cross_track):.2f} ft. "
            "Not forcing lateral rejoin; continuing from current pose."
        )

    d = signed_turn_delta(heading_deg, planned_heading_deg)
    tdir, tdeg = turn_text(d)

    if tdir != "CENTER":
        prompt_user(
            "REALIGN WITH LAWNMOWER PATH\n"
            f"Current heading: {heading_deg:.1f}°\n"
            f"Planned lawnmower heading: {planned_heading_deg:.1f}°\n"
            f"Rotate Voyager {tdir} by {tdeg:.1f}°.\n"
            "Do NOT move back to the old lane position."
        )
        heading_deg = planned_heading_deg
        set_pose(x, y, heading_deg)
    else:
        print()
        print("Already aligned with planned lawnmower heading.")

    print()
    print(
        f"Continuing from current pose: "
        f"x={x:.2f}, y={y:.2f}, heading={heading_deg:.1f}°."
    )
    return x, y, heading_deg

def detour_positions(x, y, heading_deg, side, args):
    sign = 1.0 if side == "LEFT" else -1.0
    angle = float(args.detour_angle_deg)
    side_step = float(args.detour_side_step_ft)
    forward_step = float(args.detour_forward_step_ft)

    pts = []
    h = norm_heading(heading_deg + sign * angle)
    x, y = move_pose(x, y, h, side_step)
    pts.append((x, y, h))

    h = norm_heading(h - sign * angle)
    x, y = move_pose(x, y, h, forward_step)
    pts.append((x, y, h))

    h = norm_heading(h - sign * angle)
    x, y = move_pose(x, y, h, side_step)
    pts.append((x, y, h))

    h = norm_heading(h + sign * angle)
    pts.append((x, y, h))
    return pts


def detour_fits(x, y, heading_deg, side, args):
    margin = args.boundary_margin_ft
    for px, py, _ in detour_positions(x, y, heading_deg, side, args):
        if px < -margin or px > args.room_length_ft + margin:
            return False
        if py < -margin or py > args.room_width_ft + margin:
            return False
    return True


def perform_detour(x, y, heading_deg, events_csv, event_id, args):
    preferred = args.detour_preferred_side.upper()
    candidates = [preferred, "RIGHT" if preferred == "LEFT" else "LEFT"]

    side = None
    for c in candidates:
        if detour_fits(x, y, heading_deg, c, args):
            side = c
            break

    if side is None:
        prompt_user(
            "MINI DETOUR NEEDED, but both left/right detours appear to leave the test area.\n"
            "Manually stop and clear the path, then press ENTER to continue."
        )
        return x, y, heading_deg, "detour_not_possible"

    sign = 1.0 if side == "LEFT" else -1.0
    opposite = "RIGHT" if side == "LEFT" else "LEFT"
    angle = args.detour_angle_deg

    print("\nMINI DETOUR REQUIRED")
    print(f"Chosen side: {side}")
    print(f"Turn {side} {angle:.1f}°")
    print(f"Move forward {args.detour_side_step_ft:.2f} ft")
    print(f"Turn {opposite} {angle:.1f}°")
    print(f"Move forward {args.detour_forward_step_ft:.2f} ft")
    print(f"Turn {opposite} {angle:.1f}°")
    print(f"Move forward {args.detour_side_step_ft:.2f} ft")
    print(f"Turn {side} {angle:.1f}° to realign")

    prompt_user(f"DETOUR STEP 1: Turn {side} by {angle:.1f}°.")
    heading_deg = norm_heading(heading_deg + sign * angle)
    set_pose(x, y, heading_deg)

    prompt_user(f"DETOUR STEP 2: Move forward {args.detour_side_step_ft:.2f} ft.")
    x, y = move_pose(x, y, heading_deg, args.detour_side_step_ft)
    set_pose(x, y, heading_deg)

    prompt_user(f"DETOUR STEP 3: Turn {opposite} by {angle:.1f}°.")
    heading_deg = norm_heading(heading_deg - sign * angle)
    set_pose(x, y, heading_deg)

    prompt_user(f"DETOUR STEP 4: Move forward {args.detour_forward_step_ft:.2f} ft.")
    x, y = move_pose(x, y, heading_deg, args.detour_forward_step_ft)
    set_pose(x, y, heading_deg)

    prompt_user(f"DETOUR STEP 5: Turn {opposite} by {angle:.1f}°.")
    heading_deg = norm_heading(heading_deg - sign * angle)
    set_pose(x, y, heading_deg)

    prompt_user(f"DETOUR STEP 6: Move forward {args.detour_side_step_ft:.2f} ft.")
    x, y = move_pose(x, y, heading_deg, args.detour_side_step_ft)
    set_pose(x, y, heading_deg)

    prompt_user(f"DETOUR STEP 7: Turn {side} by {angle:.1f}° to realign.")
    heading_deg = norm_heading(heading_deg + sign * angle)
    set_pose(x, y, heading_deg)

    write_event(events_csv, {
        "time": now_s(),
        "event_id": event_id,
        "phase": "detour",
        "x_ft": round(x, 3),
        "y_ft": round(y, 3),
        "heading_deg": round(heading_deg, 2),
        "detour_status": f"completed_{side.lower()}",
        "decision": "mini_detour_completed",
    })

    return x, y, heading_deg, f"completed_{side.lower()}"


def pre_move_vision_guard(model, out_dir, events_csv, x, y, heading_deg, event_id, args, expected_labels):
    if args.disable_premove_vision_guard:
        return x, y, heading_deg, False

    vision = capture_and_detect(
        model,
        out_dir,
        f"premove_{event_id:04d}",
        args,
        expected_labels,
    )

    detections = vision["detections"]
    object_estimate = estimate_object(detections, vision["raw_path"], x, y, heading_deg, args)
    collision_risk, collision_info = conservative_collision_risk(
        detections,
        vision["raw_path"],
        object_estimate,
        args,
    )

    if detections:
        print_object_report(
            "PRE-MOVE VISION GUARD",
            object_estimate,
            collision_risk,
            "normal move" if not collision_risk else "mini detour",
        )

    write_event(events_csv, {
        "time": now_s(),
        "event_id": event_id,
        "phase": "pre_move_vision_guard",
        "x_ft": round(x, 3),
        "y_ft": round(y, 3),
        "heading_deg": round(heading_deg, 2),
        "vision_positive": vision["positive"],
        "vision_num_detections": len(detections),
        "vision_best_label": vision.get("best_label", ""),
        "vision_best_conf": vision.get("best_conf", ""),
        "vision_raw_path": str(vision["raw_path"]),
        "vision_annotated_path": str(vision["ann_path"]),
        "estimated_object_x_ft": object_estimate.get("estimated_object_x_ft", ""),
        "estimated_object_y_ft": object_estimate.get("estimated_object_y_ft", ""),
        "estimated_object_range_ft": object_estimate.get("estimated_object_range_ft", ""),
        "estimated_object_bearing_deg": object_estimate.get("estimated_object_bearing_deg", ""),
        "range_method": object_estimate.get("range_method", ""),
        "collision_clearance_ft": object_estimate.get("collision_clearance_ft", ""),
        "collision_risk": bool(collision_risk),
        "collision_reason": collision_info.get("collision_reason", ""),
        "decision": "detour_required" if collision_risk else "path_clear",
        "notes": vision.get("error", ""),
    })

    stop_required = collision_stop_required(collision_risk, object_estimate, args)

    if collision_risk and not stop_required:
        print(
            "\nPRE-MOVE CAUTION: object visible in path zone, "
            "but not close enough for automatic detour. Continue carefully."
        )

    if stop_required:
        print(
            "\nPRE-MOVE STOP: object/confirmed target is too close for normal forward motion. "
            "Starting mini detour."
        )
        x, y, heading_deg, detour_status = perform_detour(x, y, heading_deg, events_csv, event_id, args)
        return x, y, heading_deg, True

    return x, y, heading_deg, False


def guide_return_home(x, y, heading_deg, args, events_csv=None, event_id=None):
    """
    Manually guide Voyager back to the start and, when an events CSV is
    supplied, record the return trajectory so it can be reconstructed
    by the trajectory/sensor animation.
    """
    dist = math.hypot(x, y)

    if dist < 0.25:
        print("\nAlready close to home/start.")
        if events_csv is not None:
            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id if event_id is not None else "",
                "phase": "return_home_complete",
                "x_ft": 0.0,
                "y_ft": 0.0,
                "heading_deg": round(heading_deg, 2),
                "instruction": "Return home complete",
                "decision": "returned_home",
            })
        return 0.0, 0.0, heading_deg

    # Record where the return-home sequence actually begins.
    if events_csv is not None:
        write_event(events_csv, {
            "time": now_s(),
            "event_id": event_id if event_id is not None else "",
            "phase": "return_home_start",
            "x_ft": round(x, 3),
            "y_ft": round(y, 3),
            "heading_deg": round(heading_deg, 2),
            "instruction": "Begin return home",
            "decision": "return_home_started",
        })

    target_heading = norm_heading(math.degrees(math.atan2(-y, -x)))
    d = signed_turn_delta(heading_deg, target_heading)
    tdir, tdeg = turn_text(d)

    if tdir != "CENTER":
        prompt_user(
            f"RETURN HOME\nCurrent pose: x={x:.2f}, y={y:.2f}, heading={heading_deg:.1f}°\n"
            f"Rotate {tdir} by {tdeg:.1f}° toward home."
        )
        heading_deg = target_heading
        set_pose(x, y, heading_deg)

        if events_csv is not None:
            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id if event_id is not None else "",
                "phase": "return_home_rotate",
                "x_ft": round(x, 3),
                "y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "instruction": f"Return home: rotate {tdir} {tdeg:.1f} deg",
                "decision": "return_home_rotate",
            })

    remaining = math.hypot(x, y)

    while remaining > 0.25:
        step = min(args.return_step_ft, remaining)
        prompt_user(f"RETURN HOME: Move forward {step:.2f} ft toward start.")

        x, y = move_pose(x, y, heading_deg, step)
        set_pose(x, y, heading_deg)

        if events_csv is not None:
            write_event(events_csv, {
                "time": now_s(),
                "event_id": event_id if event_id is not None else "",
                "phase": "return_home_move",
                "x_ft": round(x, 3),
                "y_ft": round(y, 3),
                "heading_deg": round(heading_deg, 2),
                "instruction": f"Return home: move forward {step:.2f} ft",
                "decision": "return_home_move",
            })

        remaining = math.hypot(x, y)

    print("\nReturn-home complete or close to start. Snapping internal pose to x=0, y=0.")

    if events_csv is not None:
        write_event(events_csv, {
            "time": now_s(),
            "event_id": event_id if event_id is not None else "",
            "phase": "return_home_complete",
            "x_ft": 0.0,
            "y_ft": 0.0,
            "heading_deg": round(heading_deg, 2),
            "instruction": "Return home complete",
            "decision": "returned_home",
        })

    return 0.0, 0.0, heading_deg


def build_arg_parser():
    ap = argparse.ArgumentParser(description="Manual Navigation HIL v2 with lane rejoin, detour, collision clearance, and non-waste rejection.")

    ap.add_argument("--label", default="manual_lawnmower_hil_v2")
    ap.add_argument("--output-root", default="outputs/manual_navigation_tests")

    ap.add_argument("--room-length-ft", type=float, default=8.0)
    ap.add_argument("--room-width-ft", type=float, default=8.0)
    ap.add_argument("--lane-spacing-ft", type=float, default=2.0)
    ap.add_argument("--coverage-step-ft", type=float, default=1.0)

    ap.add_argument("--approach-step-ft", type=float, default=1.0)
    ap.add_argument("--close-approach-step-ft", type=float, default=0.5)
    ap.add_argument("--return-step-ft", type=float, default=2.0)
    ap.add_argument("--max-approach-steps", type=int, default=10)

    ap.add_argument("--mode", choices=["warm", "cold", "both"], default="cold")
    ap.add_argument("--threshold", type=float, default=2.0)
    ap.add_argument("--min-blob-pixels", type=int, default=6)
    ap.add_argument("--fov-x-deg", type=float, default=55.0)
    ap.add_argument("--read-retries", type=int, default=8)
    ap.add_argument("--read-retry-delay-sec", type=float, default=0.25)

    ap.add_argument("--model", default="models/best.pt")
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--vision-vote-k", type=int, default=2)
    ap.add_argument("--expected-labels", default="waste,bottle,plastic,plastic_bottle")

    ap.add_argument("--vision-horizontal-fov-deg", type=float, default=66.0)
    ap.add_argument("--fallback-object-distance-ft", type=float, default=3.5)
    ap.add_argument("--fallback-range-multiplier-ft", type=float, default=2.5)
    ap.add_argument("--fallback-min-range-ft", type=float, default=3.0)
    ap.add_argument("--min-estimated-object-distance-ft", type=float, default=0.25)
    ap.add_argument("--max-estimated-object-distance-ft", type=float, default=6.0)
    ap.add_argument("--min-bbox-growth-ratio", type=float, default=1.08)

    ap.add_argument("--collision-safety-buffer-ft", type=float, default=0.5)
    ap.add_argument("--collision-clearance-stop-ft", type=float, default=0.75)
    ap.add_argument("--fallback-collision-stop-ft", type=float, default=0.25)
    ap.add_argument("--collision-hold-verify-frames", type=int, default=2)
    ap.add_argument("--collision-bbox-area-frac", type=float, default=0.05)
    ap.add_argument("--collision-bbox-height-frac", type=float, default=0.18)
    ap.add_argument("--collision-bbox-bottom-frac", type=float, default=0.60)
    ap.add_argument("--forward-zone-left-frac", type=float, default=0.30)
    ap.add_argument("--forward-zone-right-frac", type=float, default=0.70)

    ap.add_argument("--duplicate-radius-ft", type=float, default=1.0)
    ap.add_argument("--fallback-duplicate-radius-ft", type=float, default=3.0)
    ap.add_argument("--boundary-margin-ft", type=float, default=0.15)

    ap.add_argument("--non-waste-negative-votes", type=int, default=3)
    ap.add_argument("--close-non-waste-negative-votes", type=int, default=2)

    ap.add_argument("--disable-premove-vision-guard", action="store_true")
    ap.add_argument("--detour-angle-deg", type=float, default=45.0)
    ap.add_argument("--detour-side-step-ft", type=float, default=1.0)
    ap.add_argument("--detour-forward-step-ft", type=float, default=1.5)
    ap.add_argument("--detour-preferred-side", choices=["LEFT", "RIGHT", "left", "right"], default="LEFT")

    ap.add_argument("--rejoin-lane-tolerance-ft", type=float, default=0.75)

    ap.add_argument("--camera-timeout-ms", type=int, default=1500)
    ap.add_argument("--no-hflip", action="store_true")
    ap.add_argument("--no-vflip", action="store_true")

    ap.add_argument("--start-x-ft", type=float, default=0.0)
    ap.add_argument("--start-y-ft", type=float, default=0.0)
    ap.add_argument("--start-heading-deg", type=float, default=0.0)

    ap.add_argument("--clear", action="store_true")

    return ap


def main():
    args = build_arg_parser().parse_args()
    args.detour_preferred_side = args.detour_preferred_side.upper()

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
    rejected_json = out_dir / "rejected_objects.json"
    summary_json = out_dir / "summary.json"

    expected_labels = {
        x.strip().lower()
        for x in args.expected_labels.split(",")
        if x.strip()
    }

    print("=== Manual Navigation HIL v2 ===")
    print("You physically move the robot. The script gives exact movement instructions.")
    print("No motors are controlled by this script.")
    print()
    print(f"Room/map length: {args.room_length_ft} ft")
    print(f"Room/map width:  {args.room_width_ft} ft")
    print(f"Lane spacing:    {args.lane_spacing_ft} ft")
    print(f"Coverage step:   {args.coverage_step_ft} ft")
    print(f"Output:          {out_dir}")

    prompt_user(
        "Place Voyager at the START point.\n"
        "Point it along the first lawnmower lane.\n"
        "Coordinate convention: x = first forward direction, y = lane-to-lane direction."
    )

    print("\nInitializing thermal camera...")
    i2c = base.busio.I2C(base.board.SCL, base.board.SDA)
    mlx = base.adafruit_mlx90640.MLX90640(i2c)
    mlx.refresh_rate = base.adafruit_mlx90640.RefreshRate.REFRESH_1_HZ

    print("Loading YOLO model...")
    from ultralytics import YOLO
    model = YOLO(args.model)

    x = args.start_x_ft
    y = args.start_y_ft
    heading_deg = norm_heading(args.start_heading_deg)
    set_pose(x, y, heading_deg)

    commands = base.generate_lawnmower_plan(
        args.room_length_ft,
        args.room_width_ft,
        args.lane_spacing_ft,
        args.coverage_step_ft,
    )

    event_id = 0
    confirmed_count = 0
    rejected_count = 0
    started = time.time()

    try:
        for idx, cmd in enumerate(commands, start=1):
            event_id += 1

            active_lane_x = x
            active_lane_y = y
            planned_heading_deg = heading_deg

            sample_name = f"coverage_{event_id:04d}_thermal"
            thermal = base.scan_thermal(mlx, out_dir, sample_name, args)

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
                "active_lane_x_ft": round(active_lane_x, 3),
                "active_lane_y_ft": round(active_lane_y, 3),
                "planned_heading_deg": round(planned_heading_deg, 2),
                "instruction": "thermal scan before movement",
                "thermal_ok": thermal.get("ok"),
                "thermal_candidate": thermal.get("candidate"),
                "thermal_mode": thermal.get("selected_mode", args.mode),
                "thermal_delta_c": thermal.get("delta_c", ""),
                "thermal_bearing_deg": thermal.get("bearing_deg", ""),
                "thermal_blob_px": thermal.get("blob_px", ""),
                "thermal_preview_path": thermal.get("preview_path", ""),
                "decision": "thermal_candidate" if thermal.get("candidate") else "no_candidate",
            })

            if thermal.get("candidate"):
                x, y, heading_deg, status = approach_candidate_v2(
                    mlx,
                    model,
                    out_dir,
                    events_csv,
                    confirmed_json,
                    rejected_json,
                    x,
                    y,
                    heading_deg,
                    event_id,
                    args,
                    expected_labels,
                    active_lane_x,
                    active_lane_y,
                    planned_heading_deg,
                )

                if status == "confirmed_waste":
                    confirmed_count += 1
                if status == "rejected_non_waste":
                    rejected_count += 1

                x, y, heading_deg = rejoin_lane_and_realign(
                    x,
                    y,
                    heading_deg,
                    active_lane_x,
                    active_lane_y,
                    planned_heading_deg,
                    args,
                )

            if cmd["type"] == "rotate":
                turn = cmd["turn"]
                deg = cmd["degrees"]
                prompt_user(f"{cmd['label']}\nRotate Voyager {turn} by {deg:.1f}°.")
                heading_deg = norm_heading(heading_deg + cmd["heading_delta"])
                set_pose(x, y, heading_deg)

                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "coverage_rotate",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "instruction": cmd["label"],
                    "decision": "rotated",
                })

            elif cmd["type"] == "move":
                x, y, heading_deg, detoured = pre_move_vision_guard(
                    model,
                    out_dir,
                    events_csv,
                    x,
                    y,
                    heading_deg,
                    event_id,
                    args,
                    expected_labels,
                )

                if detoured:
                    print("\nOriginal forward command skipped because detour already advanced Voyager.")
                    continue

                d = cmd["distance_ft"]

                # Hard safety guard for ordinary lawnmower coverage movement.
                # Approach/detour logic has separate boundary checks; this
                # protects the final planned coverage step as well.
                out_of_bounds, next_x, next_y = base.next_pose_out_of_bounds(
                    x,
                    y,
                    heading_deg,
                    d,
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
                        "active_lane_x_ft": round(active_lane_x, 3),
                        "active_lane_y_ft": round(active_lane_y, 3),
                        "planned_heading_deg": round(planned_heading_deg, 2),
                        "instruction": cmd["label"],
                        "decision": "coverage_move_blocked",
                        "notes": (
                            f"planned coverage move would leave test area: "
                            f"next_x={next_x:.2f}, next_y={next_y:.2f}"
                        ),
                    })

                    print(
                        f"\nCOVERAGE BOUNDARY GUARD: move blocked. "
                        f"Predicted next pose x={next_x:.2f}, y={next_y:.2f} "
                        f"is outside the configured test area."
                    )
                    continue
                prompt_user(f"{cmd['label']}\nMove Voyager FORWARD {d:.2f} ft.")
                x, y = move_pose(x, y, heading_deg, d)
                set_pose(x, y, heading_deg)

                write_event(events_csv, {
                    "time": now_s(),
                    "event_id": event_id,
                    "phase": "coverage_move",
                    "x_ft": round(x, 3),
                    "y_ft": round(y, 3),
                    "heading_deg": round(heading_deg, 2),
                    "instruction": cmd["label"],
                    "decision": "moved",
                })

    except ReturnHomeRequested:
        print("\nReturn/home requested.")
        x, y, heading_deg = guide_return_home(
        x,
        y,
        heading_deg,
        args,
        events_csv=events_csv,
        event_id=event_id,
    )
    finally:
        elapsed = time.time() - started
        summary = {
            "label": args.label,
            "duration_sec": round(elapsed, 2),
            "final_x_ft": round(x, 3),
            "final_y_ft": round(y, 3),
            "final_heading_deg": round(heading_deg, 2),
            "confirmed_count": confirmed_count,
            "rejected_count": rejected_count,
            "events_csv": str(events_csv),
            "confirmed_json": str(confirmed_json),
            "rejected_json": str(rejected_json),
            "notes": "Manual Navigation HIL v2: rolling continuation, lane rejoin, collision clearance, pre-move guard, mini-detour, non-waste rejection.",
        }
        summary_json.write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to: {summary_json}")


if __name__ == "__main__":
    main()
