#!/usr/bin/env python3

from gpiozero import DigitalOutputDevice
from time import sleep

# ============================================================
# OCEAN VOYAGER - POOL NAVIGATION
#
# Patterns:
#
#   1 = LAWNMOWER
#       Repeated short-side traverses
#
#   2 = BOUNDARY
#       One complete rectangular lap
#
# DEFAULT POOL CALIBRATION:
#
#   Straight:
#       LEFT  = 1.6 sec
#       RIGHT = 1.0 sec
#
#   Safe travel:
#       SHORT side = 5 balanced forward cycles
#       LONG side  = 7 balanced forward cycles
#
#   Approx 90-degree turns:
#       physical LEFT pump  = 3 x 1 sec
#       physical RIGHT pump = 2 x 1 sec
#
# Ctrl-C always switches pumps OFF.
# ============================================================


# ------------------------------------------------------------
# GPIO
# ------------------------------------------------------------

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)

R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)


# ------------------------------------------------------------
# DEFAULT CALIBRATION
# ------------------------------------------------------------

DEFAULT_SHORT_CYCLES = 5
DEFAULT_LONG_CYCLES = 7

DEFAULT_LEFT_RIGHT_RATIO = 1.6

DEFAULT_LEFT_90_BURSTS = 3
DEFAULT_RIGHT_90_BURSTS = 2

RIGHT_BASE_SEC = 1.0
TURN_BURST_SEC = 1.0

GAP_SEC = 0.05

# Let the boat settle slightly after turns.
TURN_SETTLE_SEC = 0.40

# Small settling period at the end of each straight leg.
LEG_SETTLE_SEC = 0.25


# ============================================================
# LOW-LEVEL PUMP CONTROL
# ============================================================

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


def pulse_left(seconds):

    set_left(True)

    sleep(seconds)

    stop()


def pulse_right(seconds):

    set_right(True)

    sleep(seconds)

    stop()


# ============================================================
# STRAIGHT MOTION
# ============================================================

def forward_cycle(balance_ratio):

    left_seconds = RIGHT_BASE_SEC * balance_ratio

    right_seconds = RIGHT_BASE_SEC

    pulse_left(left_seconds)

    sleep(GAP_SEC)

    pulse_right(right_seconds)


def forward(cycles, balance_ratio, name=""):

    print()
    print(
        f"FORWARD {name}: {cycles} balanced cycles"
    )

    for n in range(1, cycles + 1):

        print(
            f"  cycle {n}/{cycles}",
            flush=True
        )

        forward_cycle(balance_ratio)

        if n < cycles:
            sleep(GAP_SEC)

    stop()

    sleep(LEG_SETTLE_SEC)


# ============================================================
# TURN MOTION
# ============================================================

def left_pump_turn(bursts):

    print()
    print(
        f"TURN using PHYSICAL LEFT pump: "
        f"{bursts} x {TURN_BURST_SEC:.1f}s"
    )

    for n in range(1, bursts + 1):

        print(
            f"  burst {n}/{bursts}",
            flush=True
        )

        pulse_left(TURN_BURST_SEC)

        if n < bursts:
            sleep(GAP_SEC)

    stop()

    sleep(TURN_SETTLE_SEC)


def right_pump_turn(bursts):

    print()
    print(
        f"TURN using PHYSICAL RIGHT pump: "
        f"{bursts} x {TURN_BURST_SEC:.1f}s"
    )

    for n in range(1, bursts + 1):

        print(
            f"  burst {n}/{bursts}",
            flush=True
        )

        pulse_right(TURN_BURST_SEC)

        if n < bursts:
            sleep(GAP_SEC)

    stop()

    sleep(TURN_SETTLE_SEC)


# ============================================================
# BOUNDARY PATTERN
# ============================================================

def boundary(
    short_cycles,
    long_cycles,
    balance_ratio,
    left90,
    right90,
    turn_side
):

    print()
    print("========================================")
    print("BOUNDARY LAP")
    print("========================================")

    legs = [

        ("LONG 1", long_cycles),

        ("SHORT 1", short_cycles),

        ("LONG 2", long_cycles),

        ("SHORT 2", short_cycles),

    ]

    for corner, (name, cycles) in enumerate(
        legs,
        start=1
    ):

        forward(
            cycles,
            balance_ratio,
            name
        )

        print(
            f"\nCORNER {corner}/4"
        )

        if turn_side == "r":

            right_pump_turn(
                right90
            )

        else:

            left_pump_turn(
                left90
            )

    stop()

    print()
    print("========================================")
    print("BOUNDARY LAP COMPLETE")
    print("PUMPS OFF")
    print("========================================")


# ============================================================
# LAWNMOWER PATTERN
# ============================================================

def lawnmower(
    short_cycles,
    long_cycles,
    balance_ratio,
    left90,
    right90
):

    print()
    print("========================================")
    print("LAWNMOWER")
    print("========================================")

    print(
        f"Short-side traverse = "
        f"{short_cycles} cycles"
    )

    print(
        f"Long-side available = "
        f"{long_cycles} cycles"
    )

    # For now use one balanced forward cycle
    # as the step between adjacent lanes.
    lane_step_cycles = 1

    # With long=7 this gives 8 passes.
    number_of_lanes = (
        long_cycles // lane_step_cycles
    ) + 1

    print(
        f"Lane step = {lane_step_cycles} cycle"
    )

    print(
        f"Number of lanes = {number_of_lanes}"
    )

    for lane in range(
        1,
        number_of_lanes + 1
    ):

        print()
        print(
            f"========== LANE "
            f"{lane}/{number_of_lanes} =========="
        )

        forward(
            short_cycles,
            balance_ratio,
            f"LANE {lane}"
        )

        if lane == number_of_lanes:
            break

        # Snake pattern.
        #
        # One end:
        #   RIGHT-pump turn
        #   move one step
        #   RIGHT-pump turn
        #
        # Other end:
        #   LEFT-pump turn
        #   move one step
        #   LEFT-pump turn

        if lane % 2 == 1:

            right_pump_turn(
                right90
            )

            forward(
                lane_step_cycles,
                balance_ratio,
                "LANE STEP"
            )

            right_pump_turn(
                right90
            )

        else:

            left_pump_turn(
                left90
            )

            forward(
                lane_step_cycles,
                balance_ratio,
                "LANE STEP"
            )

            left_pump_turn(
                left90
            )

    stop()

    print()
    print("========================================")
    print("LAWNMOWER COMPLETE")
    print("PUMPS OFF")
    print("========================================")


# ============================================================
# INPUT HELPERS
# ============================================================

def get_int(
    prompt,
    default
):

    text = input(
        f"{prompt} [{default}]: "
    ).strip()

    if text == "":
        return int(default)

    value = int(text)

    if value <= 0:
        raise ValueError(
            "Value must be greater than zero."
        )

    return value


def get_ratio():

    text = input(
        "LEFT:RIGHT straight timing ratio "
        f"[{DEFAULT_LEFT_RIGHT_RATIO}:1]: "
    ).strip()

    if text == "":

        return DEFAULT_LEFT_RIGHT_RATIO

    if ":" in text:

        left, right = text.split(
            ":",
            1
        )

        left = float(left)

        right = float(right)

        if left <= 0 or right <= 0:

            raise ValueError(
                "Ratio values must be positive."
            )

        return left / right

    value = float(text)

    if value <= 0:

        raise ValueError(
            "Ratio must be positive."
        )

    return value


# ============================================================
# MAIN
# ============================================================

def main():

    stop()

    print("""
========================================================
 OCEAN VOYAGER - POOL NAVIGATION
========================================================

Patterns:

  1 = LAWNMOWER

  2 = BOUNDARY LAP


CURRENT DEFAULTS:

  safe SHORT side:
      5 balanced cycles

  safe LONG side:
      7 balanced cycles

  straight:
      LEFT : RIGHT = 1.6 : 1

  approx 90 degree:

      PHYSICAL LEFT pump
          3 x 1-second bursts

      PHYSICAL RIGHT pump
          2 x 1-second bursts


Press ENTER to keep any default.

Ctrl-C at ANY TIME = PUMPS OFF.

========================================================
""")

    pattern = input(
        "Pattern - 1 lawnmower / 2 boundary [2]: "
    ).strip()

    if pattern == "":
        pattern = "2"

    if pattern not in (
        "1",
        "2"
    ):

        raise ValueError(
            "Pattern must be 1 or 2."
        )

    short_cycles = get_int(
        "SHORT-side balanced cycles",
        DEFAULT_SHORT_CYCLES
    )

    long_cycles = get_int(
        "LONG-side balanced cycles",
        DEFAULT_LONG_CYCLES
    )

    balance_ratio = get_ratio()

    left90 = get_int(
        "LEFT-pump ~90-degree burst count",
        DEFAULT_LEFT_90_BURSTS
    )

    right90 = get_int(
        "RIGHT-pump ~90-degree burst count",
        DEFAULT_RIGHT_90_BURSTS
    )

    turn_side = "r"

    if pattern == "2":

        text = input(
            "Boundary corner pump "
            "- r=RIGHT / l=LEFT [r]: "
        ).strip().lower()

        if text != "":

            if text not in (
                "r",
                "l"
            ):

                raise ValueError(
                    "Turn pump must be r or l."
                )

            turn_side = text


    print()
    print("========================================")
    print("RUN SETTINGS")
    print("========================================")

    if pattern == "1":

        print(
            "Pattern        : LAWNMOWER"
        )

    else:

        print(
            "Pattern        : BOUNDARY"
        )

    print(
        f"SHORT cycles   : {short_cycles}"
    )

    print(
        f"LONG cycles    : {long_cycles}"
    )

    print(
        f"Straight L:R   : "
        f"{balance_ratio:.3f}:1"
    )

    print(
        f"LEFT 90        : "
        f"{left90} x 1 sec"
    )

    print(
        f"RIGHT 90       : "
        f"{right90} x 1 sec"
    )

    if pattern == "2":

        print(
            "Boundary turns : "
            + (
                "PHYSICAL RIGHT pump"
                if turn_side == "r"
                else "PHYSICAL LEFT pump"
            )
        )

    print("========================================")

    print()

    if pattern == "2":

        print(
            "Place Voyager near a corner."
        )

        print(
            "Point it along the LONG side."
        )

    else:

        print(
            "Place Voyager near a corner."
        )

        print(
            "Point it across the SHORT side."
        )

    print()

    print(
        "Press ENTER when ready."
    )

    print(
        "Ctrl-C immediately stops all pumps."
    )

    input()

    print()
    print(
        "Starting in 5 seconds..."
    )

    for n in range(
        5,
        0,
        -1
    ):

        print(
            n,
            flush=True
        )

        sleep(1)

    if pattern == "2":

        boundary(
            short_cycles,
            long_cycles,
            balance_ratio,
            left90,
            right90,
            turn_side
        )

    else:

        lawnmower(
            short_cycles,
            long_cycles,
            balance_ratio,
            left90,
            right90
        )


try:

    main()

except KeyboardInterrupt:

    stop()

    print()
    print(
        ">>> ABORTED - PUMPS OFF <<<"
    )

except Exception as exc:

    stop()

    print()
    print(
        f"ERROR: {exc}"
    )
