from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from datetime import datetime
from pathlib import Path
import subprocess
import threading
import csv
import sys
import numpy as np

import board
import busio
import adafruit_mlx90640

# Reuse the already-tested living-room thermal analysis.
from thermal_anomaly_test import (
    read_frame_with_retries,
    analyze_frame,
    save_heatmap,
)

# ============================================================
# OCEAN VOYAGER
# PREPROGRAMMED PROPULSION + PASSIVE THERMAL/RGB CAPTURE
#
# Sensors NEVER change motor commands in this script.
#
# Command syntax:
#
#   l <seconds> <repeats>       physical LEFT pump
#   r <seconds> <repeats>       physical RIGHT pump
#   a <L_seconds> <R_seconds> <repeats>
#   p <seconds>                 coast / pumps OFF
#
# Example:
#   a .25 .75 6 p 2 a .25 .75 6 p 3
# ============================================================

# -----------------------------
# MOTOR CONFIG
# -----------------------------

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)

R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

MAX_PULSE_SEC = 3.0
MAX_PAUSE_SEC = 15.0
MAX_REPEATS = 100

GAP_SEC = 0.05


# -----------------------------
# SENSOR CONFIG
# -----------------------------

SENSOR_INTERVAL_SEC = 1.0

# Pool objects expected warmer than water.
THERMAL_MODE = "warm"

THERMAL_THRESHOLD_C = 2.0
THERMAL_MIN_BLOB_PIXELS = 6
THERMAL_FOV_X_DEG = 55.0

RGB_WIDTH = 1280
RGB_HEIGHT = 720

# Keep RGB capture relatively quick.
RGB_TIMEOUT_MS = 500


# -----------------------------
# SHARED STATE
# -----------------------------

sensor_stop = threading.Event()


# -----------------------------
# MOTOR FUNCTIONS
# -----------------------------

def stop():
    L1.off()
    L2.off()
    R1.off()
    R2.off()


def set_left(on):
    if on:
        R1.off()
        R2.off()

        L1.on()
        L2.off()
    else:
        L1.off()
        L2.off()


def set_right(on):
    if on:
        L1.off()
        L2.off()

        R1.on()
        R2.off()
    else:
        R1.off()
        R2.off()


# -----------------------------
# PROGRAM PARSER
# -----------------------------

def parse_program(text):
    tokens = text.lower().split()

    program = []
    i = 0

    while i < len(tokens):

        cmd = tokens[i]

        if cmd in ("l", "r"):

            if i + 2 >= len(tokens):
                raise ValueError(
                    f"{cmd} requires: {cmd} <seconds> <repeats>"
                )

            seconds = float(tokens[i + 1])
            repeats = int(tokens[i + 2])

            if not (0 < seconds <= MAX_PULSE_SEC):
                raise ValueError(
                    f"{cmd}: seconds must be >0 and <= {MAX_PULSE_SEC}"
                )

            if not (1 <= repeats <= MAX_REPEATS):
                raise ValueError(
                    f"{cmd}: repeats must be 1..{MAX_REPEATS}"
                )

            program.append({
                "cmd": cmd,
                "seconds": seconds,
                "repeats": repeats,
            })

            i += 3


        elif cmd == "a":

            if i + 3 >= len(tokens):
                raise ValueError(
                    "a requires: "
                    "a <left_seconds> <right_seconds> <repeats>"
                )

            left_sec = float(tokens[i + 1])
            right_sec = float(tokens[i + 2])
            repeats = int(tokens[i + 3])

            if not (0 < left_sec <= MAX_PULSE_SEC):
                raise ValueError(
                    f"a: LEFT seconds must be <= {MAX_PULSE_SEC}"
                )

            if not (0 < right_sec <= MAX_PULSE_SEC):
                raise ValueError(
                    f"a: RIGHT seconds must be <= {MAX_PULSE_SEC}"
                )

            if not (1 <= repeats <= MAX_REPEATS):
                raise ValueError(
                    f"a: repeats must be 1..{MAX_REPEATS}"
                )

            program.append({
                "cmd": "a",
                "left_sec": left_sec,
                "right_sec": right_sec,
                "repeats": repeats,
            })

            i += 4


        elif cmd == "p":

            if i + 1 >= len(tokens):
                raise ValueError("p requires: p <seconds>")

            seconds = float(tokens[i + 1])

            if not (0 < seconds <= MAX_PAUSE_SEC):
                raise ValueError(
                    f"p: seconds must be <= {MAX_PAUSE_SEC}"
                )

            program.append({
                "cmd": "p",
                "seconds": seconds,
            })

            i += 2


        else:
            raise ValueError(
                f"Unknown command '{cmd}'. Use l, r, a, p."
            )

    if not program:
        raise ValueError("Program is empty.")

    return program


def print_program(program):

    print("\n================ PROGRAM LOADED ================")

    for n, step in enumerate(program, 1):

        if step["cmd"] == "l":

            print(
                f"{n:2d}. PHYSICAL LEFT   "
                f"{step['seconds']:.2f}s x {step['repeats']}"
            )

        elif step["cmd"] == "r":

            print(
                f"{n:2d}. PHYSICAL RIGHT  "
                f"{step['seconds']:.2f}s x {step['repeats']}"
            )

        elif step["cmd"] == "a":

            print(
                f"{n:2d}. ALTERNATE       "
                f"L={step['left_sec']:.2f}s "
                f"R={step['right_sec']:.2f}s "
                f"x {step['repeats']}"
            )

        elif step["cmd"] == "p":

            print(
                f"{n:2d}. COAST / PAUSE   "
                f"{step['seconds']:.2f}s"
            )

    print("================================================\n")


# ============================================================
# SENSOR THREAD
# ============================================================

def sensor_worker(run_dir, start_time):

    sensor_csv_path = run_dir / "sensor_events.csv"

    thermal_dir = run_dir / "thermal_frames"
    preview_dir = run_dir / "thermal_preview"
    rgb_dir = run_dir / "rgb_frames"

    thermal_dir.mkdir(exist_ok=True)
    preview_dir.mkdir(exist_ok=True)
    rgb_dir.mkdir(exist_ok=True)

    with sensor_csv_path.open("w", newline="") as fh:

        fieldnames = [
            "elapsed_sec",
            "wall_time",
            "sample",
            "thermal_ok",
            "thermal_candidate",
            "thermal_mode",
            "background_temp_c",
            "object_temp_c",
            "thermal_delta_c",
            "thermal_bearing_deg",
            "thermal_blob_px",
            "thermal_centroid_x",
            "thermal_centroid_y",
            "thermal_npy",
            "thermal_preview",
            "rgb_ok",
            "rgb_path",
            "thermal_error",
            "rgb_error",
        ]

        writer = csv.DictWriter(
            fh,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        fh.flush()

        # ---------------------------------
        # Initialize thermal once.
        # ---------------------------------

        mlx = None
        thermal_init_error = ""

        try:

            i2c = busio.I2C(
                board.SCL,
                board.SDA,
                frequency=400000,
            )

            mlx = adafruit_mlx90640.MLX90640(i2c)

            mlx.refresh_rate = (
                adafruit_mlx90640.RefreshRate.REFRESH_1_HZ
            )

            print(">>> THERMAL INITIALIZED <<<")

        except Exception as exc:

            thermal_init_error = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                f">>> THERMAL INIT FAILED: "
                f"{thermal_init_error} <<<"
            )


        sample = 0

        while not sensor_stop.is_set():

            sample += 1

            cycle_start = monotonic()

            elapsed = cycle_start - start_time

            stamp = f"{sample:04d}"

            row = {
                "elapsed_sec": f"{elapsed:.4f}",
                "wall_time":
                    datetime.now().isoformat(
                        timespec="milliseconds"
                    ),
                "sample": sample,

                "thermal_ok": 0,
                "thermal_candidate": "",
                "thermal_mode": "",
                "background_temp_c": "",
                "object_temp_c": "",
                "thermal_delta_c": "",
                "thermal_bearing_deg": "",
                "thermal_blob_px": "",
                "thermal_centroid_x": "",
                "thermal_centroid_y": "",
                "thermal_npy": "",
                "thermal_preview": "",

                "rgb_ok": 0,
                "rgb_path": "",

                "thermal_error": thermal_init_error,
                "rgb_error": "",
            }


            # ==================================================
            # THERMAL - BEST EFFORT
            # ==================================================

            if mlx is not None:

                try:

                    arr, err = read_frame_with_retries(
                        mlx,
                        retries=3,
                        delay_sec=0.08,
                    )

                    if arr is None:

                        row["thermal_error"] = err

                    else:

                        analysis = analyze_frame(
                            arr,
                            mode=THERMAL_MODE,
                            threshold_delta_c=THERMAL_THRESHOLD_C,
                            min_blob_pixels=THERMAL_MIN_BLOB_PIXELS,
                            fov_x_deg=THERMAL_FOV_X_DEG,
                        )

                        npy_path = (
                            thermal_dir /
                            f"thermal_{stamp}.npy"
                        )

                        preview_path = (
                            preview_dir /
                            f"thermal_{stamp}.png"
                        )

                        np.save(npy_path, arr)

                        try:

                            title = (
                                f"t={elapsed:.1f}s | "
                                f"{analysis['selected_mode']} | "
                                f"dT={analysis['delta_c']:.2f}C | "
                                f"bearing="
                                f"{analysis['bearing_deg']:.1f} deg"
                            )

                            save_heatmap(
                                arr,
                                analysis,
                                preview_path,
                                title,
                            )

                        except Exception as preview_exc:

                            print(
                                "Thermal preview warning:",
                                preview_exc
                            )

                        row.update({
                            "thermal_ok": 1,
                            "thermal_candidate":
                                int(analysis["candidate"]),

                            "thermal_mode":
                                analysis["selected_mode"],

                            "background_temp_c":
                                analysis["background_temp_c"],

                            "object_temp_c":
                                analysis["object_temp_c"],

                            "thermal_delta_c":
                                analysis["delta_c"],

                            "thermal_bearing_deg":
                                analysis["bearing_deg"],

                            "thermal_blob_px":
                                analysis["blob_px"],

                            "thermal_centroid_x":
                                analysis["centroid_x"],

                            "thermal_centroid_y":
                                analysis["centroid_y"],

                            "thermal_npy":
                                str(npy_path),

                            "thermal_preview":
                                str(preview_path),

                            "thermal_error": "",
                        })

                        print(
                            f"[SENSOR {sample:04d}] "
                            f"THERMAL "
                            f"candidate={analysis['candidate']} "
                            f"dT={analysis['delta_c']:.2f}C "
                            f"bearing={analysis['bearing_deg']:.1f}"
                        )

                except Exception as exc:

                    row["thermal_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )


            # ==================================================
            # RGB - BEST EFFORT
            # ==================================================

            rgb_path = (
                rgb_dir /
                f"rgb_{stamp}.jpg"
            )

            try:

                result = subprocess.run(
                    [
                        "rpicam-still",
                        "-n",
                        "-t",
                        str(RGB_TIMEOUT_MS),
                        "--width",
                        str(RGB_WIDTH),
                        "--height",
                        str(RGB_HEIGHT),
                        "-o",
                        str(rgb_path),
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=3.0,
                )

                if (
                    result.returncode == 0
                    and rgb_path.exists()
                    and rgb_path.stat().st_size > 0
                ):

                    row["rgb_ok"] = 1
                    row["rgb_path"] = str(rgb_path)

                    print(
                        f"[SENSOR {sample:04d}] "
                        f"RGB saved"
                    )

                else:

                    row["rgb_error"] = (
                        f"rpicam-still return code "
                        f"{result.returncode}"
                    )

            except Exception as exc:

                row["rgb_error"] = (
                    f"{type(exc).__name__}: {exc}"
                )


            writer.writerow(row)
            fh.flush()


            # ---------------------------------
            # approximately 1 Hz cycle
            # ---------------------------------

            spent = monotonic() - cycle_start

            remaining = (
                SENSOR_INTERVAL_SEC - spent
            )

            if remaining > 0:

                sensor_stop.wait(remaining)


# ============================================================
# MOTOR LOG
# ============================================================

def open_motor_log(run_dir):

    path = run_dir / "motor_events.csv"

    fh = path.open("w", newline="")

    writer = csv.writer(fh)

    writer.writerow([
        "elapsed_sec",
        "wall_time",
        "event",
        "physical_left",
        "physical_right",
        "step",
        "cycle",
        "notes",
    ])

    fh.flush()

    return fh, writer


def log_motor(
    writer,
    fh,
    start_time,
    event,
    left,
    right,
    step="",
    cycle="",
    notes="",
):

    elapsed = monotonic() - start_time

    writer.writerow([
        f"{elapsed:.4f}",
        datetime.now().isoformat(
            timespec="milliseconds"
        ),
        event,
        int(left),
        int(right),
        step,
        cycle,
        notes,
    ])

    fh.flush()


# ============================================================
# EXECUTION
# ============================================================

def run_program(
    program,
    writer,
    fh,
    start_time,
):

    log_motor(
        writer,
        fh,
        start_time,
        "PROGRAM_START",
        False,
        False,
    )


    for step_number, step in enumerate(
        program,
        start=1,
    ):

        cmd = step["cmd"]


        # ---------------------------------
        # PAUSE / COAST
        # ---------------------------------

        if cmd == "p":

            stop()

            print(
                f"[{monotonic()-start_time:7.2f}s] "
                f"STEP {step_number}: "
                f"COAST {step['seconds']:.2f}s"
            )

            log_motor(
                writer,
                fh,
                start_time,
                "COAST_START",
                False,
                False,
                step_number,
                "",
                f"{step['seconds']:.3f}s",
            )

            sleep(step["seconds"])

            log_motor(
                writer,
                fh,
                start_time,
                "COAST_END",
                False,
                False,
                step_number,
            )

            continue


        # ---------------------------------
        # INDIVIDUAL PUMP
        # ---------------------------------

        if cmd in ("l", "r"):

            for cycle in range(
                1,
                step["repeats"] + 1,
            ):

                seconds = step["seconds"]

                if cmd == "l":

                    print(
                        f"[{monotonic()-start_time:7.2f}s] "
                        f"LEFT {seconds:.2f}s"
                    )

                    set_left(True)

                    log_motor(
                        writer,
                        fh,
                        start_time,
                        "LEFT_ON",
                        True,
                        False,
                        step_number,
                        cycle,
                        f"{seconds:.3f}s",
                    )

                    sleep(seconds)

                    stop()

                    log_motor(
                        writer,
                        fh,
                        start_time,
                        "LEFT_OFF",
                        False,
                        False,
                        step_number,
                        cycle,
                    )

                else:

                    print(
                        f"[{monotonic()-start_time:7.2f}s] "
                        f"RIGHT {seconds:.2f}s"
                    )

                    set_right(True)

                    log_motor(
                        writer,
                        fh,
                        start_time,
                        "RIGHT_ON",
                        False,
                        True,
                        step_number,
                        cycle,
                        f"{seconds:.3f}s",
                    )

                    sleep(seconds)

                    stop()

                    log_motor(
                        writer,
                        fh,
                        start_time,
                        "RIGHT_OFF",
                        False,
                        False,
                        step_number,
                        cycle,
                    )

                if cycle < step["repeats"]:
                    sleep(GAP_SEC)

            continue


        # ---------------------------------
        # ALTERNATING
        # ---------------------------------

        if cmd == "a":

            for cycle in range(
                1,
                step["repeats"] + 1,
            ):

                # LEFT
                set_left(True)

                log_motor(
                    writer,
                    fh,
                    start_time,
                    "ALT_LEFT_ON",
                    True,
                    False,
                    step_number,
                    cycle,
                    f"{step['left_sec']:.3f}s",
                )

                sleep(step["left_sec"])

                stop()

                log_motor(
                    writer,
                    fh,
                    start_time,
                    "ALT_LEFT_OFF",
                    False,
                    False,
                    step_number,
                    cycle,
                )

                sleep(GAP_SEC)


                # RIGHT
                set_right(True)

                log_motor(
                    writer,
                    fh,
                    start_time,
                    "ALT_RIGHT_ON",
                    False,
                    True,
                    step_number,
                    cycle,
                    f"{step['right_sec']:.3f}s",
                )

                sleep(step["right_sec"])

                stop()

                log_motor(
                    writer,
                    fh,
                    start_time,
                    "ALT_RIGHT_OFF",
                    False,
                    False,
                    step_number,
                    cycle,
                )

                print(
                    f"[{monotonic()-start_time:7.2f}s] "
                    f"ALT cycle "
                    f"{cycle}/{step['repeats']} "
                    f"L={step['left_sec']:.2f} "
                    f"R={step['right_sec']:.2f}"
                )

                if cycle < step["repeats"]:
                    sleep(GAP_SEC)


    stop()

    log_motor(
        writer,
        fh,
        start_time,
        "PROGRAM_END",
        False,
        False,
    )


# ============================================================
# MAIN
# ============================================================

run_dir = None

try:

    stop()

    print("""
==========================================================
 OCEAN VOYAGER
 PROPULSION + PASSIVE THERMAL / RGB CAPTURE
==========================================================

Sensors are OBSERVATION ONLY.

Thermal detections and RGB images WILL NOT
change direction or motor commands.

Commands:

 l <sec> <repeat>       physical LEFT
 r <sec> <repeat>       physical RIGHT
 a <L_sec> <R_sec> <repeat>
 p <sec>                coast

Example:

 a .25 .75 6 p 2 a .25 .75 6 p 3

==========================================================
""")

    text = input(
        "Enter complete motion program:\n> "
    ).strip()

    program = parse_program(text)

    print_program(program)

    print(
        "Press ENTER when Voyager is positioned."
    )

    print(
        "Ctrl-C = emergency stop."
    )

    input()

    print("\nStarting in 5 seconds...")

    for n in range(5, 0, -1):
        print(n, flush=True)
        sleep(1)


    # ---------------------------------
    # One folder per experiment
    # ---------------------------------

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    run_dir = (
        Path("outputs/pool_sensor_calibration")
        / f"run_{stamp}"
    )

    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ---------------------------------
    # Shared experiment clock
    # ---------------------------------

    start_time = monotonic()


    # ---------------------------------
    # Start passive sensor thread
    # ---------------------------------

    sensor_stop.clear()

    sensor_thread = threading.Thread(
        target=sensor_worker,
        args=(run_dir, start_time),
        daemon=True,
    )

    sensor_thread.start()


    # ---------------------------------
    # Execute propulsion
    # ---------------------------------

    motor_fh, motor_writer = (
        open_motor_log(run_dir)
    )

    try:

        run_program(
            program,
            motor_writer,
            motor_fh,
            start_time,
        )

    finally:

        stop()

        motor_fh.close()


    # Give final sensor capture a moment,
    # then stop observer thread.

    sleep(0.3)

    sensor_stop.set()

    sensor_thread.join(timeout=4.0)


    print("\n======================================")
    print(">>> RUN COMPLETE - PUMPS OFF <<<")
    print("======================================")

    print(
        f"\nDATA SAVED TO:\n{run_dir}"
    )

    print(
        "\nContains:"
        "\n  motor_events.csv"
        "\n  sensor_events.csv"
        "\n  thermal_frames/*.npy"
        "\n  thermal_preview/*.png"
        "\n  rgb_frames/*.jpg"
    )


except KeyboardInterrupt:

    sensor_stop.set()

    stop()

    print(
        "\n>>> ABORTED - PUMPS OFF <<<"
    )


except Exception as exc:

    sensor_stop.set()

    stop()

    print(
        f"\nERROR: {type(exc).__name__}: {exc}"
    )

    sys.exit(1)


finally:

    sensor_stop.set()

    stop()

    L1.close()
    L2.close()
    R1.close()
    R2.close()
