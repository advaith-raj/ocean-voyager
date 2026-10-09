#!/usr/bin/env python3

from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from pathlib import Path
from datetime import datetime
import math
import csv


# ============================================================
# OCEAN VOYAGER - ADAPTIVE POOL EXPLORER
#
# COORDINATE CONVENTION
#
#        (0,0) ----------------------> +Y / RIGHT
#          |
#          |
#          v
#         +X / DOWN
#
# Pool:
#       X = 0 .. 5 ft
#       Y = 0 .. 8 ft
#
# Heading:
#       R =   0 deg = +Y
#       D =  90 deg = +X
#       L = 180 deg = -Y
#       U = 270 deg = -X
#
# Default exploration:
#       Long purposeful sweeps along Y.
#
# Ctrl-C ALWAYS switches pumps OFF.
# ============================================================


# ============================================================
# GPIO
# ============================================================

L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)

R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)


# ============================================================
# POOL GEOMETRY
# ============================================================

POOL_X_MAX = 5.0
POOL_Y_MAX = 8.0

# Keep normal planned routes away from physical walls.
MARGIN = 0.5

X_MIN_SAFE = MARGIN
X_MAX_SAFE = POOL_X_MAX - MARGIN

Y_MIN_SAFE = MARGIN
Y_MAX_SAFE = POOL_Y_MAX - MARGIN


# ============================================================
# Y-AXIS SWEEP PLAN
#
# Long traverses happen along Y.
#
# Approx lanes:
#
# x=.5  -> Y sweep
# x=1.5 -> reverse Y sweep
# x=2.5 -> reverse again
# etc.
#
# We are NOT obsessive about following these exact lines.
# They are navigation guides.
# ============================================================

PREFERRED_AXIS = "Y"

LANE_SPACING_FT = 1.0

LANE_X = []

xv = X_MIN_SAFE

while xv <= X_MAX_SAFE + 0.001:
    LANE_X.append(round(xv, 2))
    xv += LANE_SPACING_FT


# ============================================================
# PROPULSION CALIBRATION
# ============================================================

# Straight calibration from pool:
#
# LEFT  = 1.6
# RIGHT = 1.0

STRAIGHT_RATIO = 1.6

# Short forward action.
#
# We deliberately use short bursts because actual movement varies
# with wind, battery, water motion, etc.

FORWARD_RIGHT_SEC = 0.40
FORWARD_LEFT_SEC = FORWARD_RIGHT_SEC * STRAIGHT_RATIO

# Rough measured displacement model.
#
# Your ~6 balanced cycles nearly crossed the ~6.5 ft short side,
# so ~1 ft/full cycle is a reasonable initial estimate.
#
# A short action is therefore estimated proportionally.

FORWARD_FT_PER_FULL_CYCLE = 1.0


# ============================================================
# TURN MODEL
#
# Current assumption from pool calibration:
#
# Physical RIGHT pump:
#       ~2 x 1 sec ≈ logical LEFT 90 deg
#
# Physical LEFT pump:
#       ~3 x 1 sec ≈ logical RIGHT 90 deg
#
# These are only initial estimates.
# ============================================================

LOGICAL_LEFT_DEG_PER_SEC = 45.0
LOGICAL_RIGHT_DEG_PER_SEC = 30.0

SMALL_TURN_SEC = 0.25
MEDIUM_TURN_SEC = 0.45
LARGE_TURN_SEC = 0.70

GAP_SEC = 0.05


# ============================================================
# NAVIGATION TUNING
# ============================================================

# Ask for your position approximately every N motor actions.
MANUAL_CHECK_EVERY = 3

# Do not make silly left-right corrections for small errors.
HEADING_DEADBAND_DEG = 25.0

# Near enough to consider a waypoint reached.
SWEEP_GOAL_RADIUS_FT = 0.65
SHIFT_GOAL_RADIUS_FT = 0.45

# If corrected far from intended lane, planner can re-anchor
# gently rather than forcing boat back to an obsolete location.
LANE_RESYNC_DISTANCE_FT = 1.35

# Adaptation.
FORWARD_LEARN_RATE = 0.12
DRIFT_LEARN_RATE = 0.25

MIN_FORWARD_SCALE = 0.50
MAX_FORWARD_SCALE = 1.60

MAX_DRIFT_FT_SEC = 0.35


# ============================================================
# LOGGING
# ============================================================

LOG_DIR = Path("outputs/adaptive_pool_explorer")


# ============================================================
# MATH
# ============================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def wrap_heading(h):
    return h % 360.0


def heading_error(target, current):
    """
    Signed shortest heading difference.

    Positive = logical LEFT
    Negative = logical RIGHT
    """
    return ((target - current + 180.0) % 360.0) - 180.0


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


def heading_to(x, y, tx, ty):
    """
    Our heading convention:

        0 deg   = +Y
        90 deg  = +X
        180 deg = -Y
        270 deg = -X
    """

    dx = tx - x
    dy = ty - y

    return wrap_heading(
        math.degrees(
            math.atan2(dx, dy)
        )
    )


def heading_vector(h):

    r = math.radians(h)

    dx = math.sin(r)
    dy = math.cos(r)

    return dx, dy


def heading_name(h):

    h = wrap_heading(h)

    choices = [
        (0, "R"),
        (90, "D"),
        (180, "L"),
        (270, "U"),
    ]

    best = min(
        choices,
        key=lambda item:
            abs(heading_error(item[0], h))
    )

    return best[1]


# ============================================================
# PUMP CONTROL
# ============================================================

def stop():

    L1.off()
    L2.off()

    R1.off()
    R2.off()


def left_on():

    R1.off()
    R2.off()

    L1.on()
    L2.off()


def right_on():

    L1.off()
    L2.off()

    R1.on()
    R2.off()


def pulse_left(sec):

    left_on()

    sleep(sec)

    stop()


def pulse_right(sec):

    right_on()

    sleep(sec)

    stop()


# ============================================================
# STATE
# ============================================================

class State:

    def __init__(self, x, y, heading):

        self.x = x
        self.y = y

        self.heading = wrap_heading(heading)

        self.confidence = 0.75

        # Slowly learned physical forward effectiveness.
        self.forward_scale = 1.0

        # Faster-changing wind/current estimate.
        self.drift_x = 0.0
        self.drift_y = 0.0

        # Last human/vision-confirmed state.
        self.confirm_x = x
        self.confirm_y = y

        self.confirm_heading = self.heading

        self.confirm_time = monotonic()

        self.nominal_forward_since_confirm = 0.0

        self.actions_since_check = 0


# ============================================================
# SWEEP PLANNER
# ============================================================

class SweepPlanner:

    def __init__(self, state):

        # Start on lane closest to current x.
        self.lane_index = self.nearest_lane_index(state.x)

        # Determine sensible initial Y direction.
        #
        # Heading R normally means +Y.
        # Heading L normally means -Y.
        #
        # Otherwise choose the farther end.

        if abs(heading_error(0, state.heading)) <= 60:
            self.y_direction = +1

        elif abs(heading_error(180, state.heading)) <= 60:
            self.y_direction = -1

        else:
            distance_to_high = abs(Y_MAX_SAFE - state.y)
            distance_to_low = abs(state.y - Y_MIN_SAFE)

            self.y_direction = (
                +1
                if distance_to_high >= distance_to_low
                else -1
            )

        # Which direction we move through the lane list.
        self.lane_step_direction = +1

        self.mode = "SWEEP"

        self.goal = self.make_sweep_goal()


    def nearest_lane_index(self, x):

        return min(
            range(len(LANE_X)),
            key=lambda i:
                abs(LANE_X[i] - x)
        )


    def make_sweep_goal(self):

        lane_x = LANE_X[self.lane_index]

        if self.y_direction > 0:
            target_y = Y_MAX_SAFE
        else:
            target_y = Y_MIN_SAFE

        return (
            lane_x,
            target_y
        )


    def next_lane_index(self):

        candidate = (
            self.lane_index
            + self.lane_step_direction
        )

        # Reached bottom-most lane.
        if candidate >= len(LANE_X):

            self.lane_step_direction = -1

            candidate = (
                self.lane_index
                - 1
            )

        # Reached top-most lane.
        elif candidate < 0:

            self.lane_step_direction = +1

            candidate = (
                self.lane_index
                + 1
            )

        return candidate


    def begin_shift(self):

        self.mode = "SHIFT"

        self.next_lane = (
            self.next_lane_index()
        )

        # At the end of a long sweep, shift sideways while
        # remaining near the same Y boundary.

        self.goal = (
            LANE_X[self.next_lane],
            Y_MAX_SAFE
            if self.y_direction > 0
            else Y_MIN_SAFE
        )


    def finish_shift(self):

        self.lane_index = self.next_lane

        # Reverse the long-axis direction.
        self.y_direction *= -1

        self.mode = "SWEEP"

        self.goal = self.make_sweep_goal()


    def reached_goal(self, state):

        radius = (
            SWEEP_GOAL_RADIUS_FT
            if self.mode == "SWEEP"
            else SHIFT_GOAL_RADIUS_FT
        )

        return (
            distance(
                state.x,
                state.y,
                self.goal[0],
                self.goal[1]
            )
            <= radius
        )


    def advance_if_reached(self, state):

        if not self.reached_goal(state):
            return False

        if self.mode == "SWEEP":

            print()
            print(">>> END OF LONG SWEEP REACHED <<<")

            self.begin_shift()

            print(
                "Next: gentle lane transition "
                f"toward x={self.goal[0]:.2f}"
            )

        else:

            print()
            print(">>> LANE TRANSITION COMPLETE <<<")

            self.finish_shift()

            direction = (
                "+Y / RIGHT"
                if self.y_direction > 0
                else "-Y / LEFT"
            )

            print(
                "Next long sweep: "
                f"{direction}"
            )

        return True


    def resync_after_measurement(self, state):
        """
        If manual/vision correction reveals that the boat is
        far from the intended lane, don't force it backward.

        Anchor the sweep to the nearest sensible lane.
        """

        intended_x = LANE_X[self.lane_index]

        off_lane = abs(
            state.x - intended_x
        )

        if (
            self.mode == "SWEEP"
            and off_lane > LANE_RESYNC_DISTANCE_FT
        ):

            old_lane = self.lane_index

            self.lane_index = (
                self.nearest_lane_index(
                    state.x
                )
            )

            self.goal = (
                self.make_sweep_goal()
            )

            print()
            print(
                "TRAJECTORY RESYNC:"
            )

            print(
                f"old lane x="
                f"{LANE_X[old_lane]:.2f}"
            )

            print(
                f"new lane x="
                f"{LANE_X[self.lane_index]:.2f}"
            )

            print(
                "Continuing toward current "
                "long-axis destination."
            )


# ============================================================
# LOGGING
# ============================================================

def open_log():

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    path = (
        LOG_DIR
        / f"adaptive_explorer_{stamp}.csv"
    )

    fh = path.open(
        "w",
        newline=""
    )

    writer = csv.writer(fh)

    writer.writerow([
        "elapsed",
        "event",
        "x",
        "y",
        "heading",
        "confidence",
        "forward_scale",
        "drift_x",
        "drift_y",
        "planner_mode",
        "goal_x",
        "goal_y",
        "notes",
    ])

    fh.flush()

    return path, fh, writer


def log_state(
    writer,
    fh,
    start_time,
    event,
    state,
    planner,
    notes=""
):

    writer.writerow([
        f"{monotonic()-start_time:.3f}",
        event,
        f"{state.x:.3f}",
        f"{state.y:.3f}",
        f"{state.heading:.2f}",
        f"{state.confidence:.3f}",
        f"{state.forward_scale:.3f}",
        f"{state.drift_x:.3f}",
        f"{state.drift_y:.3f}",
        planner.mode,
        f"{planner.goal[0]:.3f}",
        f"{planner.goal[1]:.3f}",
        notes,
    ])

    fh.flush()


# ============================================================
# DEAD RECKONING
# ============================================================

def apply_drift(state, duration):

    state.x += (
        state.drift_x
        * duration
    )

    state.y += (
        state.drift_y
        * duration
    )


def predict_forward(state):

    nominal_distance = (
        FORWARD_FT_PER_FULL_CYCLE
        * FORWARD_RIGHT_SEC
    )

    actual_estimate = (
        nominal_distance
        * state.forward_scale
    )

    dx, dy = heading_vector(
        state.heading
    )

    state.x += (
        actual_estimate
        * dx
    )

    state.y += (
        actual_estimate
        * dy
    )

    state.nominal_forward_since_confirm += (
        nominal_distance
    )

    state.confidence = max(
        0.1,
        state.confidence - 0.025
    )


def predict_turn(
    state,
    direction,
    seconds
):

    if direction == "left":

        state.heading = wrap_heading(
            state.heading
            + LOGICAL_LEFT_DEG_PER_SEC
            * seconds
        )

    else:

        state.heading = wrap_heading(
            state.heading
            - LOGICAL_RIGHT_DEG_PER_SEC
            * seconds
        )

    state.confidence = max(
        0.1,
        state.confidence - 0.025
    )


# ============================================================
# PHYSICAL ACTIONS
# ============================================================

def move_forward(
    state,
    planner,
    writer,
    fh,
    start_time
):

    print(
        f"ACTION: FORWARD "
        f"L={FORWARD_LEFT_SEC:.2f}s "
        f"R={FORWARD_RIGHT_SEC:.2f}s"
    )

    t0 = monotonic()

    pulse_left(
        FORWARD_LEFT_SEC
    )

    sleep(
        GAP_SEC
    )

    pulse_right(
        FORWARD_RIGHT_SEC
    )

    duration = (
        monotonic() - t0
    )

    predict_forward(
        state
    )

    apply_drift(
        state,
        duration
    )

    state.actions_since_check += 1

    log_state(
        writer,
        fh,
        start_time,
        "FORWARD",
        state,
        planner
    )


def turn(
    state,
    planner,
    direction,
    seconds,
    writer,
    fh,
    start_time
):

    print(
        f"ACTION: TURN "
        f"{direction.upper()} "
        f"{seconds:.2f}s"
    )

    t0 = monotonic()

    if direction == "left":

        # Logical LEFT currently uses physical RIGHT pump.
        pulse_right(
            seconds
        )

    else:

        # Logical RIGHT currently uses physical LEFT pump.
        pulse_left(
            seconds
        )

    duration = (
        monotonic() - t0
    )

    predict_turn(
        state,
        direction,
        seconds
    )

    apply_drift(
        state,
        duration
    )

    state.actions_since_check += 1

    log_state(
        writer,
        fh,
        start_time,
        "TURN_" + direction.upper(),
        state,
        planner
    )


# ============================================================
# MANUAL POSE
# ============================================================

def parse_heading(value):

    lookup = {

        "r": 0,
        "right": 0,

        "d": 90,
        "down": 90,

        "l": 180,
        "left": 180,

        "u": 270,
        "up": 270,
    }

    v = value.lower()

    if v in lookup:
        return lookup[v]

    return wrap_heading(
        float(value)
    )


def manual_pose_prompt(state):

    print()
    print("========================================")
    print("WHERE AM I?")
    print("========================================")

    print(
        f"Estimate: "
        f"x={state.x:.2f} "
        f"y={state.y:.2f} "
        f"h={state.heading:.0f}° "
        f"({heading_name(state.heading)})"
    )

    print()
    print("ENTER       accept estimate")
    print("x y         correct position")
    print("x y h       correct position + heading")
    print("top         x=0")
    print("bottom      x=5")
    print("left        y=0")
    print("right       y=8")
    print("h R/D/L/U   correct heading only")
    print()

    while True:

        raw = input(
            "pose> "
        ).strip()

        if raw == "":
            return None

        low = raw.lower()

        if low == "top":

            return (
                0.0,
                state.y,
                None
            )

        if low == "bottom":

            return (
                POOL_X_MAX,
                state.y,
                None
            )

        if low == "left":

            return (
                state.x,
                0.0,
                None
            )

        if low == "right":

            return (
                state.x,
                POOL_Y_MAX,
                None
            )

        parts = raw.split()

        if (
            len(parts) == 2
            and parts[0].lower() == "h"
        ):

            return (
                state.x,
                state.y,
                parse_heading(
                    parts[1]
                )
            )

        if len(parts) in (2, 3):

            x = float(
                parts[0]
            )

            y = float(
                parts[1]
            )

            if not (
                0 <= x <= POOL_X_MAX
            ):

                print(
                    "x must be between 0 and 5"
                )

                continue

            if not (
                0 <= y <= POOL_Y_MAX
            ):

                print(
                    "y must be between 0 and 8"
                )

                continue

            h = None

            if len(parts) == 3:

                h = parse_heading(
                    parts[2]
                )

            return (
                x,
                y,
                h
            )

        print(
            "Could not parse that input."
        )


# ============================================================
# ADAPTIVE CORRECTION
# ============================================================

def apply_measurement(
    state,
    measured_x,
    measured_y,
    measured_heading=None,
    confidence=1.0
):

    now = monotonic()

    dt = max(
        0.25,
        now - state.confirm_time
    )

    predicted_x = state.x
    predicted_y = state.y

    error_x = (
        measured_x
        - predicted_x
    )

    error_y = (
        measured_y
        - predicted_y
    )

    error_distance = math.hypot(
        error_x,
        error_y
    )

    print()
    print(
        "LOCALIZATION CORRECTION:"
    )

    print(
        f"dx={error_x:+.2f} ft  "
        f"dy={error_y:+.2f} ft  "
        f"error={error_distance:.2f} ft"
    )


    # --------------------------------------------------------
    # Slowly learn how effective forward motion is.
    # --------------------------------------------------------

    if (
        state.nominal_forward_since_confirm
        > 0.20
    ):

        hx, hy = heading_vector(
            state.confirm_heading
        )

        actual_dx = (
            measured_x
            - state.confirm_x
        )

        actual_dy = (
            measured_y
            - state.confirm_y
        )

        projected_forward = (
            actual_dx * hx
            + actual_dy * hy
        )

        observed_scale = (
            projected_forward
            / state.nominal_forward_since_confirm
        )

        if (
            0.25
            < observed_scale
            < 2.0
        ):

            old = (
                state.forward_scale
            )

            state.forward_scale = clamp(

                old
                * (
                    1.0
                    - FORWARD_LEARN_RATE
                )

                + observed_scale
                * FORWARD_LEARN_RATE,

                MIN_FORWARD_SCALE,
                MAX_FORWARD_SCALE
            )

            print(
                "Forward model: "
                f"{old:.2f} -> "
                f"{state.forward_scale:.2f}"
            )


    # --------------------------------------------------------
    # Wind/current estimate changes faster.
    # --------------------------------------------------------

    observed_drift_x = (
        error_x / dt
    )

    observed_drift_y = (
        error_y / dt
    )

    state.drift_x = clamp(

        state.drift_x
        * (
            1.0
            - DRIFT_LEARN_RATE
        )

        + observed_drift_x
        * DRIFT_LEARN_RATE,

        -MAX_DRIFT_FT_SEC,
        MAX_DRIFT_FT_SEC
    )

    state.drift_y = clamp(

        state.drift_y
        * (
            1.0
            - DRIFT_LEARN_RATE
        )

        + observed_drift_y
        * DRIFT_LEARN_RATE,

        -MAX_DRIFT_FT_SEC,
        MAX_DRIFT_FT_SEC
    )


    # --------------------------------------------------------
    # Human/vision correction becomes current truth.
    # --------------------------------------------------------

    state.x = measured_x
    state.y = measured_y

    if measured_heading is not None:

        state.heading = wrap_heading(
            measured_heading
        )

    state.confidence = confidence

    state.confirm_x = state.x
    state.confirm_y = state.y

    state.confirm_heading = (
        state.heading
    )

    state.confirm_time = now

    state.nominal_forward_since_confirm = 0.0

    state.actions_since_check = 0


# ============================================================
# FUTURE VISUAL LOCALIZATION HOOK
# ============================================================

def try_visual_pose():
    """
    Later create:

        vision_pose_provider.py

    containing:

        def get_pose():
            return {
                "x": ...,
                "y": ...,
                "heading": ...,
                "confidence": ...
            }

    Return None when vision is uncertain.

    No navigation rewrite will be required.
    """

    try:

        import vision_pose_provider

        result = (
            vision_pose_provider
            .get_pose()
        )

    except Exception:

        return None

    if not result:
        return None

    return result


# ============================================================
# FUTURE TARGET DETECTION HOOK
# ============================================================

def check_target():
    """
    Later thermal/RGB integration can provide:

        target_event_provider.py

    with:

        def check_target():
            return {
                "detected": True,
                ...
            }

    For now this returns None.
    """

    try:

        import target_event_provider

        return (
            target_event_provider
            .check_target()
        )

    except Exception:

        return None


# ============================================================
# TRAJECTORY CONTROLLER
# ============================================================

def choose_action(
    state,
    goal
):

    desired_heading = heading_to(
        state.x,
        state.y,
        goal[0],
        goal[1]
    )

    error = heading_error(
        desired_heading,
        state.heading
    )

    magnitude = abs(
        error
    )


    # --------------------------------------------------------
    # Small error:
    # Don't zig-zag. Continue forward.
    # --------------------------------------------------------

    if magnitude <= HEADING_DEADBAND_DEG:

        return (
            "forward",
            0.0,
            desired_heading,
            error
        )


    # --------------------------------------------------------
    # Moderate error:
    # gentle correction.
    # --------------------------------------------------------

    if magnitude < 55:

        turn_seconds = (
            SMALL_TURN_SEC
        )


    # --------------------------------------------------------
    # Larger error.
    # --------------------------------------------------------

    elif magnitude < 100:

        turn_seconds = (
            MEDIUM_TURN_SEC
        )


    else:

        turn_seconds = (
            LARGE_TURN_SEC
        )


    if error > 0:

        return (
            "left",
            turn_seconds,
            desired_heading,
            error
        )

    else:

        return (
            "right",
            turn_seconds,
            desired_heading,
            error
        )


# ============================================================
# INITIAL POSE
# ============================================================

def initial_pose():

    print("""
Coordinate system:

        (0,0) ----------------------> +Y / R
          |
          |
          v
         +X / D

Pool:

        x = 0..5
        y = 0..8

Default exploration:

        long sweeps along Y

You said you will physically orient Voyager
along +Y before starting.
""")

    raw = input(
        "Initial x y [0.5 0.5]: "
    ).strip()

    if raw == "":

        x = 0.5
        y = 0.5

    else:

        parts = raw.split()

        if len(parts) != 2:

            raise ValueError(
                "Initial position requires x y"
            )

        x = float(
            parts[0]
        )

        y = float(
            parts[1]
        )


    hraw = input(
        "Initial heading R/D/L/U [R]: "
    ).strip()

    if hraw == "":

        heading = 0.0

    else:

        heading = parse_heading(
            hraw
        )


    return State(
        x,
        y,
        heading
    )


# ============================================================
# MAIN
# ============================================================

def main():

    stop()

    print("""
========================================================
 OCEAN VOYAGER - ADAPTIVE Y-SWEEP EXPLORER
========================================================

Mission:

    Explore continuously until Ctrl-C.

Behavior:

    Prefer long smooth sweeps along Y.

    Shift gradually across X between sweeps.

    Accept wind/drift rather than fighting every inch.

    Use manual position corrections now.

    Learn gradually from those corrections.

    Later use visual localization through the same
    pose interface.

    Thermal/RGB target detection will interrupt
    exploration later.

Ctrl-C ALWAYS switches pumps OFF.

========================================================
""")


    state = initial_pose()

    planner = SweepPlanner(
        state
    )

    log_path, log_fh, log_writer = (
        open_log()
    )

    start_time = monotonic()


    print()
    print(
        "Initial sweep goal:"
    )

    print(
        f"x={planner.goal[0]:.2f} "
        f"y={planner.goal[1]:.2f}"
    )

    print()
    print(
        f"Log: {log_path}"
    )

    print()
    print(
        "Press ENTER to start autonomous exploration."
    )

    input()


    try:

        while True:


            # ------------------------------------------------
            # Future target detection.
            # ------------------------------------------------

            target = check_target()

            if (
                target
                and target.get(
                    "detected",
                    False
                )
            ):

                stop()

                print()
                print(
                    ">>> TARGET DETECTED <<<"
                )

                print(
                    target
                )

                print(
                    "Target pursuit module will "
                    "take control here."
                )

                input(
                    "ENTER to resume exploration: "
                )


            # ------------------------------------------------
            # Future visual pose provider.
            # ------------------------------------------------

            visual = try_visual_pose()

            if visual:

                confidence = float(
                    visual.get(
                        "confidence",
                        0.0
                    )
                )

                if confidence >= 0.60:

                    apply_measurement(

                        state,

                        float(
                            visual["x"]
                        ),

                        float(
                            visual["y"]
                        ),

                        float(
                            visual["heading"]
                        ),

                        confidence
                    )

                    planner.resync_after_measurement(
                        state
                    )


            # ------------------------------------------------
            # Manual localization.
            # ------------------------------------------------

            if (
                state.actions_since_check
                >= MANUAL_CHECK_EVERY
            ):

                stop()

                correction = (
                    manual_pose_prompt(
                        state
                    )
                )

                if correction is None:

                    state.actions_since_check = 0

                else:

                    x, y, h = correction

                    apply_measurement(
                        state,
                        x,
                        y,
                        h,
                        1.0
                    )

                    planner.resync_after_measurement(
                        state
                    )

                    log_state(
                        log_writer,
                        log_fh,
                        start_time,
                        "MANUAL_CORRECTION",
                        state,
                        planner
                    )


            # ------------------------------------------------
            # Advance sweep state if goal reached.
            # ------------------------------------------------

            planner.advance_if_reached(
                state
            )


            # ------------------------------------------------
            # Safety if estimate says boat escaped safe area.
            #
            # Do NOT make violent correction.
            # Aim toward nearest safe interior point.
            # ------------------------------------------------

            if (
                state.x < 0
                or state.x > POOL_X_MAX
                or state.y < 0
                or state.y > POOL_Y_MAX
            ):

                stop()

                print()
                print(
                    ">>> POSITION ESTIMATE OUTSIDE POOL <<<"
                )

                correction = (
                    manual_pose_prompt(
                        state
                    )
                )

                if correction:

                    x, y, h = correction

                    apply_measurement(
                        state,
                        x,
                        y,
                        h,
                        1.0
                    )

                    planner.resync_after_measurement(
                        state
                    )

                else:

                    print(
                        "Need a position correction "
                        "before continuing."
                    )

                    continue


            # ------------------------------------------------
            # Status.
            # ------------------------------------------------

            desired = heading_to(
                state.x,
                state.y,
                planner.goal[0],
                planner.goal[1]
            )

            error = heading_error(
                desired,
                state.heading
            )

            print()
            print(
                "----------------------------------------"
            )

            print(
                f"POSE: "
                f"x={state.x:.2f} "
                f"y={state.y:.2f} "
                f"h={state.heading:.0f}° "
                f"({heading_name(state.heading)})"
            )

            print(
                f"MODE: {planner.mode}"
            )

            print(
                f"GOAL: "
                f"x={planner.goal[0]:.2f} "
                f"y={planner.goal[1]:.2f}"
            )

            print(
                f"COURSE: "
                f"desired={desired:.0f}° "
                f"error={error:+.0f}°"
            )

            print(
                f"MODEL: "
                f"forward={state.forward_scale:.2f} "
                f"drift="
                f"({state.drift_x:+.2f},"
                f"{state.drift_y:+.2f})"
            )

            print(
                "----------------------------------------"
            )


            # ------------------------------------------------
            # Choose motion.
            # ------------------------------------------------

            (
                action,
                amount,
                desired,
                error
            ) = choose_action(
                state,
                planner.goal
            )


            if action == "forward":

                move_forward(
                    state,
                    planner,
                    log_writer,
                    log_fh,
                    start_time
                )


            elif action == "left":

                turn(
                    state,
                    planner,
                    "left",
                    amount,
                    log_writer,
                    log_fh,
                    start_time
                )


            else:

                turn(
                    state,
                    planner,
                    "right",
                    amount,
                    log_writer,
                    log_fh,
                    start_time
                )


            sleep(
                0.10
            )


    except KeyboardInterrupt:

        stop()

        print()
        print(
            ">>> MISSION STOPPED - PUMPS OFF <<<"
        )


    finally:

        stop()

        log_fh.close()

        print()
        print(
            f"Log saved to:"
        )

        print(
            log_path
        )


if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        stop()

        print()
        print(
            ">>> MISSION STOPPED - PUMPS OFF <<<"
        )

    except Exception as exc:

        stop()

        print()
        print(
            f"ERROR: {exc}"
        )
