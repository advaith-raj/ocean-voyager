from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from datetime import datetime
from pathlib import Path
import csv
import sys

# ============================================================
# OCEAN VOYAGER - POOL PROPULSION SEQUENCE CALIBRATION
#
# Physical pump mapping:
#   LEFT  pump: GPIO17 / GPIO27
#   RIGHT pump: GPIO22 / GPIO23
#
# Command syntax:
#
#   l <seconds> <repeats>
#       Physical LEFT pump
#
#   r <seconds> <repeats>
#       Physical RIGHT pump
#
#   a <left_seconds> <right_seconds> <repeats>
#       Alternate LEFT then RIGHT
#
#   p <seconds>
#       Both pumps OFF -- coast/pause measurement
#
# Example:
#
#   a .25 .75 6 p 2 a .25 .75 6 p 3 l .5 2 p 2 r .5 2
#
# IMPORTANT:
#   "l" and "r" mean PHYSICAL pump, not turn direction.
# ============================================================

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)

R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

MAX_PULSE_SEC = 3.0
MAX_PAUSE_SEC = 15.0
MAX_REPEATS = 100

# Small gap between alternating pump pulses.
GAP_SEC = 0.05

LOG_DIR = Path("outputs/pool_calibration")


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


def parse_program(text):
    tokens = text.lower().split()
    program = []
    i = 0

    while i < len(tokens):
        cmd = tokens[i]

        if cmd in ("l", "r"):
            if i + 2 >= len(tokens):
                raise ValueError(
                    f"'{cmd}' requires: {cmd} <seconds> <repeats>"
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
                    "'a' requires: "
                    "a <left_seconds> <right_seconds> <repeats>"
                )

            left_sec = float(tokens[i + 1])
            right_sec = float(tokens[i + 2])
            repeats = int(tokens[i + 3])

            if not (0 < left_sec <= MAX_PULSE_SEC):
                raise ValueError(
                    f"a: left seconds must be >0 and <= {MAX_PULSE_SEC}"
                )

            if not (0 < right_sec <= MAX_PULSE_SEC):
                raise ValueError(
                    f"a: right seconds must be >0 and <= {MAX_PULSE_SEC}"
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
                raise ValueError("'p' requires: p <seconds>")

            seconds = float(tokens[i + 1])

            if not (0 < seconds <= MAX_PAUSE_SEC):
                raise ValueError(
                    f"p: seconds must be >0 and <= {MAX_PAUSE_SEC}"
                )

            program.append({
                "cmd": "p",
                "seconds": seconds,
            })

            i += 2

        else:
            raise ValueError(
                f"Unknown command '{cmd}'. Use l, r, a, or p."
            )

    if not program:
        raise ValueError("Program is empty.")

    return program


def print_program(program):
    print("\n================ PROGRAM LOADED ================")

    for n, step in enumerate(program, start=1):
        cmd = step["cmd"]

        if cmd == "l":
            print(
                f"{n:2d}. PHYSICAL LEFT  "
                f"{step['seconds']:.2f}s x {step['repeats']}"
            )

        elif cmd == "r":
            print(
                f"{n:2d}. PHYSICAL RIGHT "
                f"{step['seconds']:.2f}s x {step['repeats']}"
            )

        elif cmd == "a":
            print(
                f"{n:2d}. ALTERNATE      "
                f"L={step['left_sec']:.2f}s "
                f"R={step['right_sec']:.2f}s "
                f"x {step['repeats']}"
            )

        elif cmd == "p":
            print(
                f"{n:2d}. COAST / PAUSE  "
                f"{step['seconds']:.2f}s"
            )

    print("================================================\n")


def open_log():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = LOG_DIR / f"pool_motion_calibration_{stamp}.csv"

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

    return path, fh, writer


def log_event(writer, fh, start_time,
              event, left, right,
              step="", cycle="", notes=""):

    elapsed = monotonic() - start_time

    writer.writerow([
        f"{elapsed:.4f}",
        datetime.now().isoformat(timespec="milliseconds"),
        event,
        int(left),
        int(right),
        step,
        cycle,
        notes,
    ])

    fh.flush()


def run_program(program, writer, fh):
    start_time = monotonic()

    log_event(
        writer, fh, start_time,
        "PROGRAM_START", False, False
    )

    for step_number, step in enumerate(program, start=1):

        cmd = step["cmd"]

        if cmd == "p":
            stop()

            print(
                f"[{monotonic()-start_time:7.2f}s] "
                f"STEP {step_number}: COAST "
                f"{step['seconds']:.2f}s"
            )

            log_event(
                writer, fh, start_time,
                "COAST_START",
                False, False,
                step_number, "",
                f"{step['seconds']:.3f}s"
            )

            sleep(step["seconds"])

            log_event(
                writer, fh, start_time,
                "COAST_END",
                False, False,
                step_number
            )

            continue

        if cmd in ("l", "r"):

            for cycle in range(1, step["repeats"] + 1):

                seconds = step["seconds"]

                if cmd == "l":
                    print(
                        f"[{monotonic()-start_time:7.2f}s] "
                        f"STEP {step_number} "
                        f"cycle {cycle}/{step['repeats']}: "
                        f"LEFT {seconds:.2f}s"
                    )

                    set_left(True)

                    log_event(
                        writer, fh, start_time,
                        "LEFT_ON",
                        True, False,
                        step_number, cycle,
                        f"{seconds:.3f}s"
                    )

                    sleep(seconds)

                    stop()

                    log_event(
                        writer, fh, start_time,
                        "LEFT_OFF",
                        False, False,
                        step_number, cycle
                    )

                else:
                    print(
                        f"[{monotonic()-start_time:7.2f}s] "
                        f"STEP {step_number} "
                        f"cycle {cycle}/{step['repeats']}: "
                        f"RIGHT {seconds:.2f}s"
                    )

                    set_right(True)

                    log_event(
                        writer, fh, start_time,
                        "RIGHT_ON",
                        False, True,
                        step_number, cycle,
                        f"{seconds:.3f}s"
                    )

                    sleep(seconds)

                    stop()

                    log_event(
                        writer, fh, start_time,
                        "RIGHT_OFF",
                        False, False,
                        step_number, cycle
                    )

                if cycle < step["repeats"]:
                    sleep(GAP_SEC)

            continue

        if cmd == "a":

            for cycle in range(1, step["repeats"] + 1):

                # LEFT pulse
                print(
                    f"[{monotonic()-start_time:7.2f}s] "
                    f"STEP {step_number} "
                    f"cycle {cycle}/{step['repeats']}: "
                    f"ALT LEFT {step['left_sec']:.2f}s"
                )

                set_left(True)

                log_event(
                    writer, fh, start_time,
                    "ALT_LEFT_ON",
                    True, False,
                    step_number, cycle,
                    f"{step['left_sec']:.3f}s"
                )

                sleep(step["left_sec"])

                stop()

                log_event(
                    writer, fh, start_time,
                    "ALT_LEFT_OFF",
                    False, False,
                    step_number, cycle
                )

                sleep(GAP_SEC)

                # RIGHT pulse
                print(
                    f"[{monotonic()-start_time:7.2f}s] "
                    f"STEP {step_number} "
                    f"cycle {cycle}/{step['repeats']}: "
                    f"ALT RIGHT {step['right_sec']:.2f}s"
                )

                set_right(True)

                log_event(
                    writer, fh, start_time,
                    "ALT_RIGHT_ON",
                    False, True,
                    step_number, cycle,
                    f"{step['right_sec']:.3f}s"
                )

                sleep(step["right_sec"])

                stop()

                log_event(
                    writer, fh, start_time,
                    "ALT_RIGHT_OFF",
                    False, False,
                    step_number, cycle
                )

                if cycle < step["repeats"]:
                    sleep(GAP_SEC)

    stop()

    log_event(
        writer, fh, start_time,
        "PROGRAM_END", False, False
    )

    print(
        f"\n>>> PROGRAM COMPLETE "
        f"({monotonic()-start_time:.2f}s) - PUMPS OFF <<<"
    )


try:
    stop()

    print("""
========================================================
 OCEAN VOYAGER - POOL MOTION SEQUENCE CALIBRATION
========================================================

Commands:

  l <sec> <repeat>
      PHYSICAL LEFT pump

  r <sec> <repeat>
      PHYSICAL RIGHT pump

  a <left_sec> <right_sec> <repeat>
      ALTERNATING LEFT / RIGHT

  p <sec>
      pumps OFF / coast

Example:

  a .25 .75 6 p 2 a .25 .75 6 p 3 l .5 2 p 2 r .5 2

IMPORTANT:
  l/r refer to PHYSICAL pumps.

  a is ALWAYS:
      LEFT seconds
      RIGHT seconds
      repeat count

========================================================
""")

    text = input("Enter complete motion program:\n> ").strip()

    program = parse_program(text)

    print_program(program)

    print("Press ENTER when Voyager is positioned and ready.")
    print("Type Ctrl-C at any time to abort and switch pumps OFF.")
    input()

    print("\nStarting in 5 seconds...")

    for n in range(5, 0, -1):
        print(n, flush=True)
        sleep(1)

    log_path, log_fh, log_writer = open_log()

    try:
        run_program(program, log_writer, log_fh)

    finally:
        stop()
        log_fh.close()

    print(f"\nCalibration log saved to:\n{log_path}")

except KeyboardInterrupt:
    stop()
    print("\n>>> ABORTED - PUMPS OFF <<<")

except Exception as exc:
    stop()
    print(f"\nERROR: {exc}")
    sys.exit(1)

finally:
    stop()
    L1.close()
    L2.close()
    R1.close()
    R2.close()
