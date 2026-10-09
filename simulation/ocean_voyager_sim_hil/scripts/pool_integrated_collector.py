#!/usr/bin/env python3

from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from datetime import datetime
from pathlib import Path
from queue import Queue, Full, Empty

import csv
import math
import sys
import threading
import subprocess
import signal
import numpy as np

# Reuse the EXISTING, proven HIL sensor/vision implementation.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import manual_navigation_hil as base


# ============================================================
# OCEAN VOYAGER - INTEGRATED POOL DATA COLLECTOR
#
# PHYSICAL MOTION:
#   l <sec> <repeat>                 physical LEFT pump
#   r <sec> <repeat>                 physical RIGHT pump
#   a <left_sec> <right_sec> <rep>   alternate LEFT/RIGHT
#   p <sec>                          pumps off / coast
#
# REPEATED BLOCK:
#   st <commands> en <repeat>
#
# Example:
#
#   st a 1 1 10 l 1 9 r 2 10 en 20 l 1 5 r 2 6
#
# Sensors NEVER alter propulsion in this experiment.
# Ctrl-C stops pumps immediately.
# ============================================================


# -----------------------------
# GPIO
# -----------------------------

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)

R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)


# -----------------------------
# MOTION SETTINGS
# -----------------------------

MAX_PULSE_SEC = 3.0
MAX_PAUSE_SEC = 15.0

GAP_SEC = 0.05

# Navigation integration/logging
NAV_DT_SEC = 0.05

# PLACEHOLDERS until calibrated from pool video.
SINGLE_PUMP_FORWARD_FPS = 0.20
LEFT_PUMP_TURN_DEG_PER_SEC = -20.0
RIGHT_PUMP_TURN_DEG_PER_SEC = +20.0

COAST_DECAY_PER_SEC = 1.5


# -----------------------------
# RGB / YOLO
# -----------------------------

RGB_INTERVAL_SEC = 1.0
RGB_QUEUE_SIZE = 3

MODEL_PATH = "models/best.pt"
YOLO_CONF = 0.15

EXPECTED_LABELS = {
    "waste",
    "bottle",
    "plastic",
    "plastic_bottle",
}

CAMERA_TIMEOUT_MS = 1500

# Existing living-room code flipped both axes by default.
CAMERA_HFLIP = True
CAMERA_VFLIP = True


# -----------------------------
# THERMAL
# -----------------------------

THERMAL_INTERVAL_SEC = 1.0

# Pool targets are WARMER than water.
THERMAL_MODE = "warm"

THERMAL_THRESHOLD_C = 2.0
THERMAL_MIN_BLOB_PIXELS = 6
THERMAL_FOV_X_DEG = 55.0


# -----------------------------
# GLOBAL RUN STATE
# -----------------------------

RUN_START = None
RUN_DIR = None

stop_event = threading.Event()

state_lock = threading.Lock()

motor_left = False
motor_right = False

current_step = ""
current_cycle = ""
current_block_cycle = ""

x_est = 0.0
y_est = 0.0
heading_est = 0.0
coast_speed = 0.0

vision_queue = Queue(maxsize=RGB_QUEUE_SIZE)

model = None
mlx = None


def elapsed():
    return monotonic() - RUN_START


def wall_time():
    return datetime.now().isoformat(timespec="milliseconds")


# ============================================================
# MOTOR CONTROL
# ============================================================

def stop_pumps():
    global motor_left, motor_right

    L1.off()
    L2.off()
    R1.off()
    R2.off()

    with state_lock:
        motor_left = False
        motor_right = False


def set_left(on):
    global motor_left, motor_right

    if on:
        R1.off()
        R2.off()
        L1.on()
        L2.off()

        with state_lock:
            motor_left = True
            motor_right = False
    else:
        L1.off()
        L2.off()

        with state_lock:
            motor_left = False


def set_right(on):
    global motor_left, motor_right

    if on:
        L1.off()
        L2.off()
        R1.on()
        R2.off()

        with state_lock:
            motor_left = False
            motor_right = True
    else:
        R1.off()
        R2.off()

        with state_lock:
            motor_right = False


# ============================================================
# COMMAND PARSER
# ============================================================

def parse_single(tokens, i):

    cmd = tokens[i].lower()

    if cmd in ("l", "r"):

        if i + 2 >= len(tokens):
            raise ValueError(
                f"{cmd} requires: {cmd} <seconds> <repeats>"
            )

        sec = float(tokens[i + 1])
        rep = int(tokens[i + 2])

        if not (0 < sec <= MAX_PULSE_SEC):
            raise ValueError(
                f"{cmd}: seconds must be >0 and <= {MAX_PULSE_SEC}"
            )

        if rep < 1:
            raise ValueError("repeats must be >= 1")

        return {
            "cmd": cmd,
            "seconds": sec,
            "repeats": rep,
        }, i + 3

    if cmd == "a":

        if i + 3 >= len(tokens):
            raise ValueError(
                "a requires: "
                "a <left_seconds> <right_seconds> <repeats>"
            )

        ls = float(tokens[i + 1])
        rs = float(tokens[i + 2])
        rep = int(tokens[i + 3])

        if not (0 < ls <= MAX_PULSE_SEC):
            raise ValueError("invalid left pulse")

        if not (0 < rs <= MAX_PULSE_SEC):
            raise ValueError("invalid right pulse")

        if rep < 1:
            raise ValueError("repeats must be >= 1")

        return {
            "cmd": "a",
            "left_sec": ls,
            "right_sec": rs,
            "repeats": rep,
        }, i + 4

    if cmd == "p":

        if i + 1 >= len(tokens):
            raise ValueError("p requires: p <seconds>")

        sec = float(tokens[i + 1])

        if not (0 < sec <= MAX_PAUSE_SEC):
            raise ValueError(
                f"pause must be >0 and <= {MAX_PAUSE_SEC}"
            )

        return {
            "cmd": "p",
            "seconds": sec,
        }, i + 2

    raise ValueError(f"Unknown command '{cmd}'")


def parse_program(text):

    tokens = text.lower().split()
    program = []

    i = 0

    while i < len(tokens):

        if tokens[i] == "st":

            i += 1
            block = []

            while i < len(tokens) and tokens[i] != "en":

                if tokens[i] == "st":
                    raise ValueError(
                        "Nested st/en blocks are not supported."
                    )

                item, i = parse_single(tokens, i)
                block.append(item)

            if i >= len(tokens):
                raise ValueError("st without matching en")

            # Skip en
            i += 1

            if i >= len(tokens):
                raise ValueError(
                    "en must be followed by block repeat count"
                )

            repeat = int(tokens[i])

            if repeat < 1:
                raise ValueError(
                    "block repeat must be >= 1"
                )

            i += 1

            program.append({
                "cmd": "block",
                "program": block,
                "repeats": repeat,
            })

        elif tokens[i] == "en":

            raise ValueError("en without matching st")

        else:

            item, i = parse_single(tokens, i)
            program.append(item)

    if not program:
        raise ValueError("Program is empty.")

    return program


def describe_item(item, indent=""):

    cmd = item["cmd"]

    if cmd == "l":
        print(
            f"{indent}LEFT  {item['seconds']:.2f}s "
            f"x {item['repeats']}"
        )

    elif cmd == "r":
        print(
            f"{indent}RIGHT {item['seconds']:.2f}s "
            f"x {item['repeats']}"
        )

    elif cmd == "a":
        print(
            f"{indent}ALTERNATE "
            f"L={item['left_sec']:.2f}s "
            f"R={item['right_sec']:.2f}s "
            f"x {item['repeats']}"
        )

    elif cmd == "p":
        print(
            f"{indent}COAST {item['seconds']:.2f}s"
        )


def print_program(program):

    print()
    print("============== PROGRAM LOADED ==============")

    for n, item in enumerate(program, 1):

        if item["cmd"] == "block":

            print(
                f"{n:2d}. SEQUENCE BLOCK "
                f"x {item['repeats']}"
            )

            for child in item["program"]:
                describe_item(child, "       ")

        else:
            print(f"{n:2d}. ", end="")
            describe_item(item)

    print("============================================")
    print()


# ============================================================
# NAVIGATION / DEAD RECKONING THREAD
# ============================================================

def navigation_worker():

    global x_est, y_est, heading_est, coast_speed

    path = RUN_DIR / "navigation.csv"

    with path.open("w", newline="") as fh:

        writer = csv.writer(fh)

        writer.writerow([
            "elapsed_sec",
            "wall_time",
            "x_est_ft",
            "y_est_ft",
            "heading_est_deg",
            "physical_left",
            "physical_right",
            "step",
            "cycle",
            "block_cycle",
        ])

        last = monotonic()

        while not stop_event.is_set():

            now = monotonic()
            dt = now - last
            last = now

            with state_lock:

                left = motor_left
                right = motor_right

                if left and not right:

                    v = SINGLE_PUMP_FORWARD_FPS
                    omega = LEFT_PUMP_TURN_DEG_PER_SEC
                    coast_speed = v

                elif right and not left:

                    v = SINGLE_PUMP_FORWARD_FPS
                    omega = RIGHT_PUMP_TURN_DEG_PER_SEC
                    coast_speed = v

                elif left and right:

                    v = 2.0 * SINGLE_PUMP_FORWARD_FPS
                    omega = 0.0
                    coast_speed = v

                else:

                    v = coast_speed
                    omega = 0.0

                    coast_speed *= math.exp(
                        -COAST_DECAY_PER_SEC * dt
                    )

                    if coast_speed < 0.005:
                        coast_speed = 0.0

                mid_heading = (
                    heading_est + 0.5 * omega * dt
                )

                rad = math.radians(mid_heading)

                x_est += v * math.cos(rad) * dt
                y_est += v * math.sin(rad) * dt

                heading_est = (
                    heading_est + omega * dt
                ) % 360.0

                row = [
                    f"{elapsed():.4f}",
                    wall_time(),
                    f"{x_est:.4f}",
                    f"{y_est:.4f}",
                    f"{heading_est:.3f}",
                    int(left),
                    int(right),
                    current_step,
                    current_cycle,
                    current_block_cycle,
                ]

            writer.writerow(row)
            fh.flush()

            stop_event.wait(NAV_DT_SEC)


# ============================================================
# RGB CAPTURE THREAD
# ============================================================

def rgb_capture_worker():

    capture_dir = RUN_DIR / "rgb_raw"
    capture_dir.mkdir(parents=True, exist_ok=True)

    path = RUN_DIR / "rgb_capture.csv"

    with path.open("w", newline="") as fh:

        writer = csv.writer(fh)

        writer.writerow([
            "capture_elapsed_sec",
            "wall_time",
            "frame_id",
            "raw_path",
            "capture_ok",
            "capture_error",
            "queue_status",
        ])

        frame_id = 0

        while not stop_event.is_set():

            cycle_start = monotonic()
            frame_id += 1

            capture_t = elapsed()

            raw_path = (
                capture_dir /
                f"rgb_{frame_id:06d}.jpg"
            )

            ok = 0
            error = ""
            queue_status = ""

            try:

                base.capture_image(
                    raw_path,
                    timeout_ms=CAMERA_TIMEOUT_MS,
                    hflip=CAMERA_HFLIP,
                    vflip=CAMERA_VFLIP,
                )

                ok = 1

                item = {
                    "frame_id": frame_id,
                    "capture_elapsed_sec": capture_t,
                    "raw_path": str(raw_path),
                }

                try:
                    vision_queue.put_nowait(item)
                    queue_status = "QUEUED"

                except Full:

                    # Keep raw image.
                    # Do not let slow YOLO affect propulsion/capture.
                    queue_status = "QUEUE_FULL_NOT_PROCESSED"

            except Exception as exc:

                error = f"{type(exc).__name__}: {exc}"

            writer.writerow([
                f"{capture_t:.4f}",
                wall_time(),
                frame_id,
                str(raw_path) if ok else "",
                ok,
                error,
                queue_status,
            ])

            fh.flush()

            spent = monotonic() - cycle_start

            stop_event.wait(
                max(0.0, RGB_INTERVAL_SEC - spent)
            )


# ============================================================
# YOLO PROCESSING THREAD
# ============================================================

def best_detection(detections):

    if not detections:
        return "", 0.0, ""

    def conf_of(d):
        try:
            return float(
                d.get(
                    "confidence",
                    d.get("conf", 0.0)
                )
            )
        except Exception:
            return 0.0

    best = max(detections, key=conf_of)

    label = str(
        best.get(
            "label",
            best.get(
                "class_name",
                best.get("name", "")
            )
        )
    )

    conf = conf_of(best)

    bbox = best.get(
        "bbox",
        best.get(
            "xyxy",
            best.get("box", "")
        )
    )

    return label, conf, bbox


def yolo_worker():

    annotated_dir = RUN_DIR / "rgb_annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    path = RUN_DIR / "rgb_results.csv"

    with path.open("w", newline="") as fh:

        writer = csv.writer(fh)

        writer.writerow([
            "capture_elapsed_sec",
            "processed_elapsed_sec",
            "frame_id",
            "classification",
            "vision_positive",
            "num_detections",
            "best_label",
            "best_confidence",
            "best_bbox",
            "raw_path",
            "annotated_path",
            "processing_error",
        ])

        while (
            not stop_event.is_set()
            or not vision_queue.empty()
        ):

            try:
                item = vision_queue.get(timeout=0.2)

            except Empty:
                continue

            frame_id = item["frame_id"]
            raw_path = Path(item["raw_path"])

            ann_path = (
                annotated_dir /
                f"rgb_{frame_id:06d}_annotated.jpg"
            )

            positive = False
            detections = []
            error = ""

            try:

                positive, detections = (
                    base.run_yolo_on_image(
                        model,
                        raw_path,
                        conf=YOLO_CONF,
                        expected_labels=EXPECTED_LABELS,
                    )
                )

                base.annotate_image(
                    raw_path,
                    ann_path,
                    detections,
                )

            except Exception as exc:

                error = f"{type(exc).__name__}: {exc}"

            label, conf, bbox = best_detection(
                detections
            )

            if error:
                classification = "ERROR"

            elif positive:
                classification = "WASTE"

            elif detections:
                classification = "NON_WASTE"

            else:
                classification = "NOTHING"

            writer.writerow([
                f"{item['capture_elapsed_sec']:.4f}",
                f"{elapsed():.4f}",
                frame_id,
                classification,
                int(bool(positive)),
                len(detections),
                label,
                f"{conf:.5f}",
                str(bbox),
                str(raw_path),
                str(ann_path)
                    if ann_path.exists() else "",
                error,
            ])

            fh.flush()
            vision_queue.task_done()


# ============================================================
# THERMAL
# ============================================================

def analyse_thermal(frame):

    arr = np.asarray(frame, dtype=float).reshape(24, 32)

    # Robust background estimate.
    background = float(np.median(arr))

    if THERMAL_MODE == "warm":

        delta_map = arr - background
        mask = delta_map >= THERMAL_THRESHOLD_C

        object_temp = float(np.max(arr))
        delta = object_temp - background

    else:

        delta_map = background - arr
        mask = delta_map >= THERMAL_THRESHOLD_C

        object_temp = float(np.min(arr))
        delta = background - object_temp

    ys, xs = np.where(mask)

    blob_px = len(xs)

    candidate = (
        blob_px >= THERMAL_MIN_BLOB_PIXELS
    )

    centroid_x = ""
    centroid_y = ""
    bearing = ""

    if blob_px:

        centroid_x = float(np.mean(xs))
        centroid_y = float(np.mean(ys))

        # Image center = 15.5 for 32 columns.
        normalized = (
            centroid_x - 15.5
        ) / 16.0

        bearing = (
            normalized *
            (THERMAL_FOV_X_DEG / 2.0)
        )

    # Preserve a continuous evidence value.
    # This is NOT a calibrated probability.
    thermal_score = max(
        0.0,
        min(
            1.0,
            delta /
            max(THERMAL_THRESHOLD_C * 2.0, 0.001)
        )
    )

    return {
        "array": arr,
        "candidate": candidate,
        "background": background,
        "object_temp": object_temp,
        "delta": delta,
        "blob_px": blob_px,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "bearing": bearing,
        "thermal_score": thermal_score,
    }


def thermal_worker():

    thermal_dir = RUN_DIR / "thermal_raw"
    thermal_dir.mkdir(parents=True, exist_ok=True)

    path = RUN_DIR / "thermal.csv"

    with path.open("w", newline="") as fh:

        writer = csv.writer(fh)

        writer.writerow([
            "capture_elapsed_sec",
            "wall_time",
            "frame_id",
            "thermal_ok",
            "candidate",
            "mode",
            "background_temp_c",
            "object_temp_c",
            "delta_c",
            "thermal_score",
            "blob_px",
            "centroid_x",
            "centroid_y",
            "bearing_deg",
            "raw_npy_path",
            "error",
        ])

        frame_id = 0

        while not stop_event.is_set():

            cycle_start = monotonic()
            frame_id += 1

            capture_t = elapsed()

            ok = 0
            result = None
            error = ""
            npy_path = ""

            try:

                frame = [0.0] * 768
                mlx.getFrame(frame)

                result = analyse_thermal(frame)

                npy_path = (
                    thermal_dir /
                    f"thermal_{frame_id:06d}.npy"
                )

                np.save(
                    npy_path,
                    result["array"]
                )

                ok = 1

            except Exception as exc:

                error = f"{type(exc).__name__}: {exc}"

            if result:

                writer.writerow([
                    f"{capture_t:.4f}",
                    wall_time(),
                    frame_id,
                    ok,
                    int(result["candidate"]),
                    THERMAL_MODE,
                    f"{result['background']:.3f}",
                    f"{result['object_temp']:.3f}",
                    f"{result['delta']:.3f}",
                    f"{result['thermal_score']:.4f}",
                    result["blob_px"],
                    result["centroid_x"],
                    result["centroid_y"],
                    result["bearing"],
                    str(npy_path),
                    error,
                ])

            else:

                writer.writerow([
                    f"{capture_t:.4f}",
                    wall_time(),
                    frame_id,
                    0,
                    "",
                    THERMAL_MODE,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    error,
                ])

            fh.flush()

            spent = monotonic() - cycle_start

            stop_event.wait(
                max(0.0, THERMAL_INTERVAL_SEC - spent)
            )


# ============================================================
# MOTOR EVENT LOG
# ============================================================

motor_log_fh = None
motor_log_writer = None


def log_motor(
    event,
    left,
    right,
    step="",
    cycle="",
    block_cycle="",
    notes="",
):

    motor_log_writer.writerow([
        f"{elapsed():.4f}",
        wall_time(),
        event,
        int(left),
        int(right),
        step,
        cycle,
        block_cycle,
        notes,
    ])

    motor_log_fh.flush()


# ============================================================
# EXECUTION
# ============================================================

def interruptible_sleep(seconds):

    end = monotonic() + seconds

    while monotonic() < end:

        if stop_event.is_set():
            raise KeyboardInterrupt

        sleep(
            min(
                0.02,
                max(0.0, end - monotonic())
            )
        )


def execute_item(
    item,
    step_number,
    block_cycle="",
):

    global current_step
    global current_cycle
    global current_block_cycle

    cmd = item["cmd"]

    current_step = step_number
    current_block_cycle = block_cycle

    if cmd in ("l", "r"):

        for cycle in range(
            1,
            item["repeats"] + 1
        ):

            current_cycle = cycle

            if cmd == "l":

                set_left(True)

                log_motor(
                    "LEFT_ON",
                    1,
                    0,
                    step_number,
                    cycle,
                    block_cycle,
                    f"{item['seconds']:.3f}s",
                )

                interruptible_sleep(
                    item["seconds"]
                )

                set_left(False)

                log_motor(
                    "LEFT_OFF",
                    0,
                    0,
                    step_number,
                    cycle,
                    block_cycle,
                )

            else:

                set_right(True)

                log_motor(
                    "RIGHT_ON",
                    0,
                    1,
                    step_number,
                    cycle,
                    block_cycle,
                    f"{item['seconds']:.3f}s",
                )

                interruptible_sleep(
                    item["seconds"]
                )

                set_right(False)

                log_motor(
                    "RIGHT_OFF",
                    0,
                    0,
                    step_number,
                    cycle,
                    block_cycle,
                )

            if cycle < item["repeats"]:
                interruptible_sleep(GAP_SEC)

    elif cmd == "a":

        for cycle in range(
            1,
            item["repeats"] + 1
        ):

            current_cycle = cycle

            set_left(True)

            log_motor(
                "LEFT_ON",
                1,
                0,
                step_number,
                cycle,
                block_cycle,
                f"{item['left_sec']:.3f}s",
            )

            interruptible_sleep(
                item["left_sec"]
            )

            set_left(False)

            log_motor(
                "LEFT_OFF",
                0,
                0,
                step_number,
                cycle,
                block_cycle,
            )

            interruptible_sleep(GAP_SEC)

            set_right(True)

            log_motor(
                "RIGHT_ON",
                0,
                1,
                step_number,
                cycle,
                block_cycle,
                f"{item['right_sec']:.3f}s",
            )

            interruptible_sleep(
                item["right_sec"]
            )

            set_right(False)

            log_motor(
                "RIGHT_OFF",
                0,
                0,
                step_number,
                cycle,
                block_cycle,
            )

            if cycle < item["repeats"]:
                interruptible_sleep(GAP_SEC)

    elif cmd == "p":

        current_cycle = ""
        stop_pumps()

        log_motor(
            "COAST_START",
            0,
            0,
            step_number,
            "",
            block_cycle,
            f"{item['seconds']:.3f}s",
        )

        interruptible_sleep(
            item["seconds"]
        )

        log_motor(
            "COAST_END",
            0,
            0,
            step_number,
            "",
            block_cycle,
        )


def execute_program(program):

    for step_number, item in enumerate(
        program,
        start=1
    ):

        if stop_event.is_set():
            raise KeyboardInterrupt

        if item["cmd"] == "block":

            for block_cycle in range(
                1,
                item["repeats"] + 1
            ):

                print(
                    f">>> BLOCK {step_number}: "
                    f"cycle {block_cycle}/"
                    f"{item['repeats']} <<<"
                )

                for child_number, child in enumerate(
                    item["program"],
                    start=1
                ):

                    execute_item(
                        child,
                        f"{step_number}.{child_number}",
                        block_cycle,
                    )

        else:

            execute_item(
                item,
                step_number,
                "",
            )


# ============================================================
# INITIALIZATION
# ============================================================

def init_thermal():

    global mlx

    try:

        print("Initializing MLX90640...")

        i2c = base.busio.I2C(
            base.board.SCL,
            base.board.SDA
        )

        mlx = base.adafruit_mlx90640.MLX90640(
            i2c
        )

        mlx.refresh_rate = (
            base.adafruit_mlx90640
            .RefreshRate
            .REFRESH_1_HZ
        )

        print(
            "THERMAL: READY "
            f"({THERMAL_MODE.upper()} mode)"
        )

        return True

    except Exception as exc:

        print(
            "THERMAL: UNAVAILABLE - "
            f"{type(exc).__name__}: {exc}"
        )

        mlx = None
        return False


def init_yolo():

    global model

    try:

        print(
            f"Loading YOLO: {MODEL_PATH}"
        )

        from ultralytics import YOLO

        model = YOLO(MODEL_PATH)

        print("YOLO: READY")
        return True

    except Exception as exc:

        print(
            "YOLO: UNAVAILABLE - "
            f"{type(exc).__name__}: {exc}"
        )

        model = None
        return False


def check_camera():

    try:

        result = subprocess.run(
            [
                "rpicam-hello",
                "--list-cameras",
            ],
            capture_output=True,
            text=True,
            timeout=8,
        )

        text = (
            result.stdout + result.stderr
        )

        if "imx500" in text.lower():

            print("RGB CAMERA: READY (IMX500)")
            return True

        print("RGB CAMERA: NOT DETECTED")
        return False

    except Exception as exc:

        print(
            "RGB CAMERA CHECK ERROR: "
            f"{type(exc).__name__}: {exc}"
        )

        return False


# ============================================================
# MAIN
# ============================================================

def main():

    global RUN_START
    global RUN_DIR
    global motor_log_fh
    global motor_log_writer

    stop_pumps()

    print()
    print("==============================================")
    print(" OCEAN VOYAGER - INTEGRATED POOL COLLECTOR")
    print("==============================================")
    print()
    print("Sensors are PASSIVE.")
    print("They cannot alter propulsion.")
    print("Ctrl-C immediately stops pumps.")
    print()

    # ----------------------------------------------------
    # PROGRAM REPEAT / HISTORY
    # ----------------------------------------------------
    last_program_path = Path("outputs/pool_integrated/last_program.txt")
    last_program_path.parent.mkdir(parents=True, exist_ok=True)

    last_program = ""

    if last_program_path.exists():
        try:
            last_program = last_program_path.read_text().strip()
        except Exception:
            last_program = ""

    if last_program:
        print("Last program:")
        print(last_program)
        print()
        text = input(
            "Enter new program, or press ENTER to repeat last:\n> "
        ).strip()

        if not text:
            text = last_program
            print()
            print("REUSING LAST PROGRAM:")
            print(text)
    else:
        text = input(
            "Enter program:\n> "
        ).strip()

    try:
        program = parse_program(text)

    except Exception as exc:
        print(f"PROGRAM ERROR: {exc}")
        return

    print_program(program)

    # Save only AFTER successful parsing.
    # A mistyped/invalid command never replaces the previous good one.
    try:
        last_program_path.write_text(text + "\n")
    except Exception as exc:
        print(f"WARNING: could not save last program: {exc}")

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    RUN_DIR = (
        Path("outputs/pool_integrated") /
        f"run_{stamp}"
    )

    RUN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save entered program.
    (
        RUN_DIR / "program.txt"
    ).write_text(text + "\n")

    print(f"Run directory: {RUN_DIR}")
    print()

    rgb_available = check_camera()
    yolo_available = (
        init_yolo()
        if rgb_available
        else False
    )

    thermal_available = init_thermal()

    print()
    print("------------- SENSOR STATUS -------------")
    print(
        f"RGB capture: "
        f"{'YES' if rgb_available else 'NO'}"
    )
    print(
        f"YOLO best.pt: "
        f"{'YES' if yolo_available else 'NO'}"
    )
    print(
        f"Thermal: "
        f"{'YES' if thermal_available else 'NO'}"
    )
    print("-----------------------------------------")
    print()

    input(
        "Position Voyager. "
        "Press ENTER to begin the run."
    )

    motor_path = (
        RUN_DIR / "motor_events.csv"
    )

    motor_log_fh = motor_path.open(
        "w",
        newline=""
    )

    motor_log_writer = csv.writer(
        motor_log_fh
    )

    motor_log_writer.writerow([
        "elapsed_sec",
        "wall_time",
        "event",
        "physical_left",
        "physical_right",
        "step",
        "cycle",
        "block_cycle",
        "notes",
    ])

    motor_log_fh.flush()

    RUN_START = monotonic()

    threads = []

    nav_thread = threading.Thread(
        target=navigation_worker,
        daemon=True,
    )

    threads.append(nav_thread)

    if rgb_available:

        rgb_thread = threading.Thread(
            target=rgb_capture_worker,
            daemon=True,
        )

        threads.append(rgb_thread)

    if yolo_available:

        yolo_thread = threading.Thread(
            target=yolo_worker,
            daemon=True,
        )

        threads.append(yolo_thread)

    if thermal_available:

        thermal_thread = threading.Thread(
            target=thermal_worker,
            daemon=True,
        )

        threads.append(thermal_thread)

    for t in threads:
        t.start()

    log_motor(
        "PROGRAM_START",
        0,
        0,
    )

    try:

        execute_program(program)

        stop_pumps()

        log_motor(
            "PROGRAM_END",
            0,
            0,
        )

        print()
        print(">>> PROGRAM COMPLETE - PUMPS OFF <<<")

    except KeyboardInterrupt:

        # MOTOR SHUTDOWN FIRST.
        stop_pumps()

        try:
            log_motor(
                "CTRL_C_ABORT",
                0,
                0,
            )
        except Exception:
            pass

        print()
        print(
            ">>> CTRL-C: PUMPS OFF IMMEDIATELY <<<"
        )

    finally:

        # Pumps first, always.
        stop_pumps()

        # Then tell passive workers to finish.
        stop_event.set()

        for t in threads:
            t.join(timeout=3.0)

        if motor_log_fh:
            motor_log_fh.flush()
            motor_log_fh.close()

        L1.close()
        L2.close()
        R1.close()
        R2.close()

        print()
        print("==============================================")
        print("RUN SAVED")
        print(RUN_DIR)
        print("==============================================")
        print()

        # ----------------------------------------------------
        # KEEP / DELETE RUN
        #
        # Default is ALWAYS KEEP.
        # Only an explicit "n" or "N" deletes the run.
        # This happens only after pumps are OFF and files have
        # been flushed/closed.
        # ----------------------------------------------------
        try:
            answer = input("Keep this run? [Y/n]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            answer = ""

        if answer == "n":
            import shutil

            print(f"Deleting run: {RUN_DIR}")
            shutil.rmtree(RUN_DIR, ignore_errors=True)
            print("RUN DELETED")
        else:
            print(f"RUN KEPT: {RUN_DIR}")

        print()


if __name__ == "__main__":
    main()
