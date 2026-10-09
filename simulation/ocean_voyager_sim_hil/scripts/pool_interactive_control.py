#!/usr/bin/env python3

from gpiozero import DigitalOutputDevice
from time import sleep

# ============================================================
# OCEAN VOYAGER - INTERACTIVE POOL CONTROL
#
# Commands:
#
#   l <seconds> <repeats>
#       physical LEFT pump
#
#   r <seconds> <repeats>
#       physical RIGHT pump
#
#   a <left_seconds> <right_seconds> <repeats>
#       alternate LEFT then RIGHT
#
#   p <seconds>
#       pumps OFF / coast
#
#   st <commands...> en <repeats>
#       repeat an entire command block
#
# Examples:
#
#   l 1 3
#
#   r 1 2
#
#   l 1.6 1 r 1 1
#
#   st l 1.6 1 r 1 1 en 5
#
#   st l 1 1 r 1 1 en 3
#
#   p 2
#
# Special:
#
#   help
#   stop
#   quit
#
# Ctrl-C during motion:
#   immediately stops pumps and returns to prompt.
# ============================================================


# ------------------------------------------------------------
# GPIO - SAME AS EXISTING CALIBRATION SCRIPT
# ------------------------------------------------------------

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)

R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)


# ------------------------------------------------------------
# SAFETY LIMITS
# ------------------------------------------------------------

MAX_PULSE_SEC = 10.0
MAX_PAUSE_SEC = 30.0

MAX_REPEATS = 100
MAX_SEQUENCE_REPEATS = 100

GAP_SEC = 0.05


# ============================================================
# PUMP CONTROL
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


# ============================================================
# REPEAT-BLOCK EXPANSION
# ============================================================

def expand_repeat_blocks(tokens):

    expanded = []

    i = 0

    while i < len(tokens):

        token = tokens[i]

        if token == "st":

            try:

                en_index = tokens.index(
                    "en",
                    i + 1
                )

            except ValueError:

                raise ValueError(
                    "'st' is missing 'en'. "
                    "Use: st <commands> en <repeat>"
                )

            block = tokens[
                i + 1:
                en_index
            ]

            if not block:

                raise ValueError(
                    "Repeat block cannot be empty."
                )

            if "st" in block or "en" in block:

                raise ValueError(
                    "Nested st/en blocks are not supported."
                )

            if en_index + 1 >= len(tokens):

                raise ValueError(
                    "'en' requires repeat count."
                )

            try:

                repeats = int(
                    tokens[
                        en_index + 1
                    ]
                )

            except ValueError:

                raise ValueError(
                    "Repeat count after en "
                    "must be an integer."
                )

            if not (
                1
                <= repeats
                <= MAX_SEQUENCE_REPEATS
            ):

                raise ValueError(
                    "Sequence repeats must be "
                    f"1..{MAX_SEQUENCE_REPEATS}"
                )

            expanded.extend(
                block * repeats
            )

            i = en_index + 2

            continue


        if token == "en":

            raise ValueError(
                "'en' found without matching 'st'."
            )


        expanded.append(
            token
        )

        i += 1


    return expanded


# ============================================================
# PARSER
# ============================================================

def parse_program(text):

    tokens = text.lower().split()

    tokens = expand_repeat_blocks(
        tokens
    )

    program = []

    i = 0


    while i < len(tokens):

        cmd = tokens[i]


        # ----------------------------------------------------
        # LEFT / RIGHT
        # ----------------------------------------------------

        if cmd in (
            "l",
            "r"
        ):

            if i + 2 >= len(tokens):

                raise ValueError(
                    f"{cmd} requires: "
                    f"{cmd} <seconds> <repeats>"
                )


            seconds = float(
                tokens[i + 1]
            )

            repeats = int(
                tokens[i + 2]
            )


            if not (
                0
                < seconds
                <= MAX_PULSE_SEC
            ):

                raise ValueError(
                    f"{cmd}: seconds must be "
                    f">0 and <= {MAX_PULSE_SEC}"
                )


            if not (
                1
                <= repeats
                <= MAX_REPEATS
            ):

                raise ValueError(
                    f"{cmd}: repeats must be "
                    f"1..{MAX_REPEATS}"
                )


            program.append({

                "cmd": cmd,

                "seconds": seconds,

                "repeats": repeats,

            })


            i += 3

            continue


        # ----------------------------------------------------
        # ALTERNATE
        # ----------------------------------------------------

        if cmd == "a":

            if i + 3 >= len(tokens):

                raise ValueError(
                    "a requires: "
                    "a <left_sec> "
                    "<right_sec> "
                    "<repeats>"
                )


            left_sec = float(
                tokens[i + 1]
            )

            right_sec = float(
                tokens[i + 2]
            )

            repeats = int(
                tokens[i + 3]
            )


            if not (
                0
                < left_sec
                <= MAX_PULSE_SEC
            ):

                raise ValueError(
                    "a: left seconds out of range."
                )


            if not (
                0
                < right_sec
                <= MAX_PULSE_SEC
            ):

                raise ValueError(
                    "a: right seconds out of range."
                )


            if not (
                1
                <= repeats
                <= MAX_REPEATS
            ):

                raise ValueError(
                    f"a: repeats must be "
                    f"1..{MAX_REPEATS}"
                )


            program.append({

                "cmd": "a",

                "left_sec": left_sec,

                "right_sec": right_sec,

                "repeats": repeats,

            })


            i += 4

            continue


        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        if cmd == "p":

            if i + 1 >= len(tokens):

                raise ValueError(
                    "p requires: p <seconds>"
                )


            seconds = float(
                tokens[i + 1]
            )


            if not (
                0
                < seconds
                <= MAX_PAUSE_SEC
            ):

                raise ValueError(
                    "p: seconds must be "
                    f">0 and <= {MAX_PAUSE_SEC}"
                )


            program.append({

                "cmd": "p",

                "seconds": seconds,

            })


            i += 2

            continue


        raise ValueError(
            f"Unknown command '{cmd}'."
        )


    if not program:

        raise ValueError(
            "Program is empty."
        )


    return program


# ============================================================
# EXECUTION
# ============================================================

def run_program(program):

    for step_number, step in enumerate(
        program,
        start=1
    ):

        cmd = step["cmd"]


        # ----------------------------------------------------
        # PAUSE
        # ----------------------------------------------------

        if cmd == "p":

            stop()

            print(
                f"STEP {step_number}: "
                f"PAUSE "
                f"{step['seconds']:.2f}s"
            )

            sleep(
                step["seconds"]
            )

            continue


        # ----------------------------------------------------
        # LEFT / RIGHT
        # ----------------------------------------------------

        if cmd in (
            "l",
            "r"
        ):

            for cycle in range(
                1,
                step["repeats"] + 1
            ):

                seconds = step[
                    "seconds"
                ]


                if cmd == "l":

                    print(
                        f"STEP {step_number} "
                        f"{cycle}/"
                        f"{step['repeats']}: "
                        f"LEFT "
                        f"{seconds:.2f}s",
                        flush=True
                    )

                    set_left(
                        True
                    )

                    sleep(
                        seconds
                    )

                    stop()


                else:

                    print(
                        f"STEP {step_number} "
                        f"{cycle}/"
                        f"{step['repeats']}: "
                        f"RIGHT "
                        f"{seconds:.2f}s",
                        flush=True
                    )

                    set_right(
                        True
                    )

                    sleep(
                        seconds
                    )

                    stop()


                if cycle < step[
                    "repeats"
                ]:

                    sleep(
                        GAP_SEC
                    )


            continue


        # ----------------------------------------------------
        # ALTERNATE
        # ----------------------------------------------------

        if cmd == "a":

            for cycle in range(
                1,
                step["repeats"] + 1
            ):

                print(
                    f"STEP {step_number} "
                    f"{cycle}/"
                    f"{step['repeats']}: "
                    f"ALT LEFT "
                    f"{step['left_sec']:.2f}s",
                    flush=True
                )

                set_left(
                    True
                )

                sleep(
                    step[
                        "left_sec"
                    ]
                )

                stop()

                sleep(
                    GAP_SEC
                )


                print(
                    f"STEP {step_number} "
                    f"{cycle}/"
                    f"{step['repeats']}: "
                    f"ALT RIGHT "
                    f"{step['right_sec']:.2f}s",
                    flush=True
                )

                set_right(
                    True
                )

                sleep(
                    step[
                        "right_sec"
                    ]
                )

                stop()


                if cycle < step[
                    "repeats"
                ]:

                    sleep(
                        GAP_SEC
                    )


    stop()


# ============================================================
# HELP
# ============================================================

def show_help():

    print("""
------------------------------------------------------------
Commands
------------------------------------------------------------

l <sec> <repeat>

    physical LEFT pump

    example:
        l 1 3


r <sec> <repeat>

    physical RIGHT pump

    example:
        r 1 2


a <left_sec> <right_sec> <repeat>

    alternate LEFT then RIGHT

    example:
        a 1.6 1 5


p <sec>

    pumps OFF / coast

    example:
        p 3


st <commands> en <repeat>

    repeat an entire sequence

    example:
        st l 1.6 1 r 1 1 en 5


Multiple normal commands may also be entered:

    l 1.6 1 r 1 1 p 2 r 1 2


Special commands:

    help
    stop
    quit


Ctrl-C DURING MOVEMENT:

    immediately stops pumps
    and returns to voyager> prompt.

------------------------------------------------------------
""")


# ============================================================
# INTERACTIVE CONSOLE
# ============================================================

def main():

    stop()

    print("""
============================================================
 OCEAN VOYAGER - INTERACTIVE POOL CONTROL
============================================================

Enter one maneuver at a time.

Examples:

    l 1 3

    r 1 2

    l 1.6 1 r 1 1

    st l 1.6 1 r 1 1 en 5

Type:

    help
    stop
    quit

Ctrl-C while moving = immediate pumps OFF.

============================================================
""")


    while True:

        try:

            text = input(
                "voyager> "
            ).strip()


        except EOFError:

            stop()

            print(
                "\nPUMPS OFF - EXIT"
            )

            break


        except KeyboardInterrupt:

            stop()

            print(
                "\nPUMPS OFF"
            )

            continue


        if not text:

            continue


        lowered = text.lower()


        if lowered in (
            "q",
            "quit",
            "exit"
        ):

            stop()

            print(
                "PUMPS OFF - EXIT"
            )

            break


        if lowered in (
            "h",
            "help",
            "?"
        ):

            show_help()

            continue


        if lowered in (
            "s",
            "stop"
        ):

            stop()

            print(
                "PUMPS OFF"
            )

            continue


        try:

            program = parse_program(
                text
            )

            print(
                f"Executing: {text}"
            )

            run_program(
                program
            )

            print(
                ">>> COMPLETE - PUMPS OFF <<<"
            )


        except KeyboardInterrupt:

            stop()

            print()
            print(
                ">>> MANEUVER ABORTED "
                "- PUMPS OFF <<<"
            )


        except Exception as exc:

            stop()

            print(
                f"ERROR: {exc}"
            )


if __name__ == "__main__":

    try:

        main()

    finally:

        stop()
