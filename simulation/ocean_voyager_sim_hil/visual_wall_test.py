#!/usr/bin/env python3

"""
Ocean Voyager - Visual Wall Test #1

PURPOSE
-------
Stationary visual test only.

NO pumps.
NO thermal.
NO YOLO.
NO autonomous motion.

For each test:
    1. User enters approximate real wall distance.
    2. RGB image is captured.
    3. Image is rotated 180 degrees.
    4. Basic image geometry is analyzed.
    5. Original + annotated images are saved.
    6. Measurements are logged to CSV.

This first experiment is CALIBRATION.

We are NOT yet claiming that monocular vision can accurately
produce distance in feet. We collect real pool observations first.

Controls:
    ENTER  capture another frame
    q      quit
"""

from pathlib import Path
from datetime import datetime
import subprocess
import csv
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

CAMERA_ROTATION = 180

# Analyze at lower resolution for speed.
ANALYSIS_WIDTH = 960

# Edge/Hough parameters.
CANNY_LOW = 60
CANNY_HIGH = 160

HOUGH_THRESHOLD = 70
HOUGH_MIN_LINE_LENGTH = 100
HOUGH_MAX_LINE_GAP = 30

# A pool rim/water boundary should generally be more horizontal
# than vertical when the camera is aimed sensibly.
MAX_HORIZONTAL_SLOPE = 0.35

OUTPUT_ROOT = Path("outputs/visual_wall_test")


# ============================================================
# HELPERS
# ============================================================

def rotate_image(img):
    if CAMERA_ROTATION == 180:
        return cv2.rotate(img, cv2.ROTATE_180)

    if CAMERA_ROTATION == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

    if CAMERA_ROTATION == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return img


def resize_for_analysis(img):
    h, w = img.shape[:2]

    if w <= ANALYSIS_WIDTH:
        return img.copy()

    scale = ANALYSIS_WIDTH / float(w)

    return cv2.resize(
        img,
        (
            ANALYSIS_WIDTH,
            int(h * scale)
        ),
        interpolation=cv2.INTER_AREA
    )


def capture_rgb(path):
    """
    Capture using Raspberry Pi camera tools.
    """

    cmd = [
        "rpicam-still",
        "-o", str(path),
        "--timeout", "700",
        "--nopreview"
    ]

    print("Capturing RGB image...")

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if result.returncode != 0:
        print(result.stderr)
        raise RuntimeError("rpicam-still capture failed")


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_wall_geometry(img):
    """
    This does NOT yet return feet.

    It looks for strong near-horizontal lines that could
    correspond to the pool rim / water boundary.

    Returns:
        annotated image
        edge image
        candidate information
    """

    work = resize_for_analysis(img)

    h, w = work.shape[:2]

    gray = cv2.cvtColor(
        work,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    edges = cv2.Canny(
        gray,
        CANNY_LOW,
        CANNY_HIGH
    )

    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180.0,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP
    )

    annotated = work.copy()

    candidates = []

    if lines is not None:

        for item in lines:

            x1, y1, x2, y2 = item[0]

            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1:
                continue

            slope = dy / float(dx)

            length = float(
                np.hypot(dx, dy)
            )

            # Keep primarily horizontal lines.
            if abs(slope) <= MAX_HORIZONTAL_SLOPE:

                mid_x = (x1 + x2) / 2.0
                mid_y = (y1 + y2) / 2.0

                # Score favors:
                # 1. long lines
                # 2. lines closer to image center
                #
                # We deliberately do not over-engineer this
                # until we see real pool images.

                center_penalty = (
                    abs(mid_x - w / 2.0)
                    / (w / 2.0)
                )

                score = (
                    length
                    * (1.0 - 0.25 * center_penalty)
                )

                candidates.append({
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "mid_x": float(mid_x),
                    "mid_y": float(mid_y),
                    "length": length,
                    "slope": float(slope),
                    "score": float(score),
                })


    candidates.sort(
        key=lambda x: x["score"],
        reverse=True
    )


    # Draw top candidates.
    for c in candidates[:15]:

        cv2.line(
            annotated,
            (c["x1"], c["y1"]),
            (c["x2"], c["y2"]),
            (0, 255, 255),
            2
        )


    best = (
        candidates[0]
        if candidates
        else None
    )


    if best is not None:

        y_fraction = (
            best["mid_y"] / h
        )

        # Image-space quantity only.
        #
        # If the pool boundary moves up/down consistently
        # with known distance, we'll calibrate this.

        water_fraction_below = (
            (h - best["mid_y"]) / h
        )

        cv2.line(
            annotated,
            (best["x1"], best["y1"]),
            (best["x2"], best["y2"]),
            (0, 0, 255),
            5
        )

        cx = int(best["mid_x"])
        cy = int(best["mid_y"])

        cv2.circle(
            annotated,
            (cx, cy),
            9,
            (0, 0, 255),
            -1
        )

        text1 = (
            f"BEST boundary y={best['mid_y']:.0f}px "
            f"({y_fraction:.3f} image height)"
        )

        text2 = (
            f"water-below fraction="
            f"{water_fraction_below:.3f}"
        )

        cv2.putText(
            annotated,
            text1,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            annotated,
            text2,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    else:

        y_fraction = None
        water_fraction_below = None

        cv2.putText(
            annotated,
            "NO STRONG HORIZONTAL BOUNDARY FOUND",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )


    return {
        "analysis_image": work,
        "annotated": annotated,
        "edges": edges,
        "best": best,
        "candidate_count": len(candidates),
        "y_fraction": y_fraction,
        "water_fraction_below": water_fraction_below,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        OUTPUT_ROOT
        / f"run_{stamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    csv_path = (
        run_dir
        / "measurements.csv"
    )

    csv_file = csv_path.open(
        "w",
        newline=""
    )

    writer = csv.writer(
        csv_file
    )

    writer.writerow([
        "frame",
        "actual_distance_ft",
        "view_angle_note",
        "candidate_count",
        "best_boundary_y_px",
        "boundary_y_fraction",
        "water_fraction_below",
        "best_line_length",
        "best_line_slope",
        "raw_file",
        "corrected_file",
        "annotated_file",
        "edge_file",
    ])

    csv_file.flush()


    print()
    print("==============================================")
    print(" OCEAN VOYAGER - VISUAL WALL TEST #1")
    print("==============================================")
    print()
    print("NO MOTOR CONTROL.")
    print()
    print("Aim Voyager camera toward a pool wall.")
    print()
    print("For each capture, enter the approximate")
    print("REAL distance from CAMERA to that wall.")
    print()
    print("Examples:")
    print()
    print("  0.5")
    print("  0.75")
    print("  1")
    print("  1.5")
    print("  2")
    print("  3")
    print()
    print("Images and measurements will be saved.")
    print()
    print(f"Run directory:")
    print(run_dir)
    print()


    frame_number = 1


    try:

        while True:

            print()
            print("----------------------------------------------")

            raw = input(
                "Actual wall distance in feet [q=quit]: "
            ).strip()

            if raw.lower() == "q":
                break

            try:
                actual_distance = float(raw)

            except ValueError:
                print("Please enter a number such as 0.75 or 2")
                continue


            angle_note = input(
                "Viewing angle/note [straight]: "
            ).strip()

            if not angle_note:
                angle_note = "straight"


            base = f"frame_{frame_number:03d}"

            raw_path = (
                run_dir
                / f"{base}_raw.jpg"
            )

            corrected_path = (
                run_dir
                / f"{base}_corrected.jpg"
            )

            annotated_path = (
                run_dir
                / f"{base}_annotated.jpg"
            )

            edge_path = (
                run_dir
                / f"{base}_edges.png"
            )


            capture_rgb(
                raw_path
            )


            raw_img = cv2.imread(
                str(raw_path)
            )

            if raw_img is None:
                raise RuntimeError(
                    f"Could not read {raw_path}"
                )


            corrected = rotate_image(
                raw_img
            )

            cv2.imwrite(
                str(corrected_path),
                corrected
            )


            result = analyze_wall_geometry(
                corrected
            )


            cv2.imwrite(
                str(annotated_path),
                result["annotated"]
            )

            cv2.imwrite(
                str(edge_path),
                result["edges"]
            )


            best = result["best"]


            if best is None:

                best_y = ""
                best_length = ""
                best_slope = ""

                print()
                print("RESULT:")
                print("No strong horizontal boundary detected.")

            else:

                best_y = (
                    f"{best['mid_y']:.2f}"
                )

                best_length = (
                    f"{best['length']:.2f}"
                )

                best_slope = (
                    f"{best['slope']:.4f}"
                )

                print()
                print("RESULT:")
                print(
                    f"candidate lines : "
                    f"{result['candidate_count']}"
                )

                print(
                    f"best boundary Y: "
                    f"{best['mid_y']:.1f} px"
                )

                print(
                    f"Y fraction      : "
                    f"{result['y_fraction']:.3f}"
                )

                print(
                    f"water below     : "
                    f"{result['water_fraction_below']:.3f}"
                )

                print(
                    f"line slope      : "
                    f"{best['slope']:+.3f}"
                )


            writer.writerow([
                frame_number,
                actual_distance,
                angle_note,
                result["candidate_count"],
                best_y,
                (
                    ""
                    if result["y_fraction"] is None
                    else f"{result['y_fraction']:.5f}"
                ),
                (
                    ""
                    if result["water_fraction_below"] is None
                    else f"{result['water_fraction_below']:.5f}"
                ),
                best_length,
                best_slope,
                raw_path.name,
                corrected_path.name,
                annotated_path.name,
                edge_path.name,
            ])

            csv_file.flush()


            print()
            print("Saved:")
            print(f"  {corrected_path.name}")
            print(f"  {annotated_path.name}")
            print(f"  {edge_path.name}")
            print()

            print(
                "Move/rotate Voyager or adjust the camera "
                "and repeat."
            )

            frame_number += 1


    except KeyboardInterrupt:

        print()
        print("Stopped.")


    finally:

        csv_file.close()

        print()
        print("==============================================")
        print("TEST COMPLETE")
        print("==============================================")
        print()
        print("Results:")
        print(run_dir)
        print()
        print("Measurements:")
        print(csv_path)


if __name__ == "__main__":
    main()
