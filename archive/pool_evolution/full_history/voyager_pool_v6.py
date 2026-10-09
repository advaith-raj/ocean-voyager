from gpiozero import DigitalOutputDevice
from time import sleep
from pathlib import Path
from datetime import datetime
from collections import deque
import subprocess
import cv2
import numpy as np
import csv
import signal
import sys
import argparse

# ============================================================
# OCEAN VOYAGER - CONTINUOUS VISUAL NAVIGATION v6
#
# v6 changes from v5:
#   - VERY_NEAR no longer oscillates after short 0.65 s pivots.
#   - Wall release uses committed 2.8 s pivots in one direction.
#   - It tries the same direction up to 3 times before switching.
#   - If known, initial wall-release direction opposes the last steering direction.
#   - Recovery requires meaningful visual structure to reappear (hysteresis).
#   - All v5 CLI tuning remains optional; no parameters uses v6 defaults.
#   - GPIO mapping / physical pump semantics are unchanged.
#
# Ctrl-C always stops motors.
# All output goes to /tmp/voyager_v5.
# ============================================================

# ---------- CLI / TUNING ----------
p = argparse.ArgumentParser(description="Ocean Voyager pool navigation v5")

# Motion. Defaults are used when nothing is typed.
p.add_argument("--steer-sec", type=float, default=0.35,
               help="ordinary LEFT/RIGHT correction seconds (default: 0.35)")
p.add_argument("--escape-turn-sec", type=float, default=3.20,
               help="committed counter-turn seconds during ESCAPE (default: 3.20)")
p.add_argument("--escape-forward-bursts", type=int, default=2,
               help="forward bursts immediately after ESCAPE turn (default: 2)")
p.add_argument("--post-escape-forward-cycles", type=int, default=2,
               help="additional forced FORWARD decision cycles after ESCAPE (default: 2)")
p.add_argument("--repeat-turn-limit", type=int, default=4,
               help="same-direction steering actions before ESCAPE (default: 4)")

# Existing forward calibration, exposed for later trim without new scripts.
p.add_argument("--forward-left-sec", type=float, default=0.48,
               help="physical left pump pulse during FORWARD (default: 0.48)")
p.add_argument("--forward-right-sec", type=float, default=0.30,
               help="physical right pump pulse during FORWARD (default: 0.30)")
p.add_argument("--cautious-left-sec", type=float, default=0.32,
               help="left pump pulse after AVOID (default: 0.32)")
p.add_argument("--cautious-right-sec", type=float, default=0.20,
               help="right pump pulse after AVOID (default: 0.20)")
p.add_argument("--recovery-left-sec", type=float, default=0.24,
               help="left pump pulse after VERY_NEAR clears (default: 0.24)")
p.add_argument("--recovery-right-sec", type=float, default=0.15,
               help="right pump pulse after VERY_NEAR clears (default: 0.15)")
p.add_argument("--gap-sec", type=float, default=0.03,
               help="gap between sequential pump pulses (default: 0.03)")

# Vision / behavior thresholds.
p.add_argument("--side-dominance", type=float, default=1.60,
               help="side texture dominance threshold (default: 1.60)")
p.add_argument("--side-abs-difference", type=float, default=0.020,
               help="minimum side texture difference (default: 0.020)")
p.add_argument("--near-scale-threshold", type=float, default=8.0,
               help="large-pattern near-wall threshold (default: 8.0)")
p.add_argument("--min-visual-change", type=float, default=5.0,
               help="below this, scene counts as stagnant (default: 5.0)")
p.add_argument("--max-stagnant", type=int, default=3,
               help="stagnant observations before ESCAPE (default: 3)")
p.add_argument("--very-low-edge", type=float, default=0.004,
               help="feature-collapse edge threshold (default: 0.004)")
p.add_argument("--invalid-scale", type=float, default=1000.0,
               help="saturated/invalid pattern scale threshold (default: 1000)")
p.add_argument("--very-near-turn-sec", type=float, default=2.80,
               help="committed VERY_NEAR pivot seconds (default: 2.80)")
p.add_argument("--max-same-near-turns", type=int, default=3,
               help="committed VERY_NEAR attempts before trying opposite direction (default: 3)")
p.add_argument("--near-release-edge", type=float, default=0.010,
               help="minimum recovered edge structure to consider wall view opened (default: 0.010)")
p.add_argument("--near-release-scale", type=float, default=500.0,
               help="scale must fall below this to consider wall view opened (default: 500)")

# Camera.
p.add_argument("--camera-timeout-ms", type=int, default=350,
               help="rpicam-still timeout ms (default: 350)")
args = p.parse_args()

WIDTH = 1280
HEIGHT = 960
CAMERA_TIMEOUT_MS = args.camera_timeout_ms

STEER_SEC = args.steer_sec
ESCAPE_TURN_SEC = args.escape_turn_sec
ESCAPE_FORWARD_BURSTS = max(0, args.escape_forward_bursts)
POST_ESCAPE_FORWARD_CYCLES = max(0, args.post_escape_forward_cycles)
MAX_CONSECUTIVE_TURNS = max(1, args.repeat_turn_limit)

FWD_LEFT_SEC = args.forward_left_sec
FWD_RIGHT_SEC = args.forward_right_sec
CAUTIOUS_LEFT_SEC = args.cautious_left_sec
CAUTIOUS_RIGHT_SEC = args.cautious_right_sec
RECOVERY_LEFT_SEC = args.recovery_left_sec
RECOVERY_RIGHT_SEC = args.recovery_right_sec
GAP_SEC = args.gap_sec

SIDE_DOMINANCE = args.side_dominance
SIDE_ABS_DIFFERENCE = args.side_abs_difference
NEAR_SCALE_THRESHOLD = args.near_scale_threshold
MIN_VISUAL_CHANGE = args.min_visual_change
MAX_STAGNANT = max(1, args.max_stagnant)

VERY_LOW_EDGE = args.very_low_edge
INVALID_SCALE = args.invalid_scale
VERY_NEAR_CONFIRM_FRAMES = 1
VERY_NEAR_TURN_SEC = args.very_near_turn_sec
MAX_SAME_ESCAPE_TURNS = max(1, args.max_same_near_turns)
NEAR_RELEASE_EDGE = args.near_release_edge
NEAR_RELEASE_SCALE = args.near_release_scale

HISTORY_LEN = 10

# ---------- WORKING PHYSICAL GPIO MAPPING (unchanged from v4) ----------
L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)
R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

OUT = Path("/tmp/voyager_v6")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# MOTOR CONTROL -- v4 semantics preserved
# ============================================================

def stop():
    L1.off(); L2.off(); R1.off(); R2.off()

def physical_left(sec):
    stop()
    L1.on(); L2.off()
    sleep(sec)
    stop()

def physical_right(sec):
    stop()
    R1.on(); R2.off()
    sleep(sec)
    stop()

def forward(left_sec=None, right_sec=None):
    if left_sec is None:
        left_sec = FWD_LEFT_SEC
    if right_sec is None:
        right_sec = FWD_RIGHT_SEC

    stop()
    L1.on(); L2.off()
    sleep(left_sec)
    stop()
    sleep(GAP_SEC)
    R1.on(); R2.off()
    sleep(right_sec)
    stop()

def turn_left(sec=None):
    # Physical RIGHT pump rotates Voyager LEFT.
    physical_right(STEER_SEC if sec is None else sec)

def turn_right(sec=None):
    # Physical LEFT pump rotates Voyager RIGHT.
    physical_left(STEER_SEC if sec is None else sec)

# ============================================================
# SAFETY / SHUTDOWN
# ============================================================

def shutdown(signum=None, frame=None):
    stop()
    print("\n*** VOYAGER STOPPED - MOTORS OFF ***")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ============================================================
# CAMERA
# ============================================================

def capture(n):
    raw = Path("/tmp") / f"voyager_v6_raw_{n:05d}.jpg"
    final = OUT / f"frame_{n:05d}.jpg"

    cmd = [
        "rpicam-still",
        "-o", str(raw),
        "--timeout", str(CAMERA_TIMEOUT_MS),
        "--nopreview",
        "--width", str(WIDTH),
        "--height", str(HEIGHT),
    ]

    subprocess.run(
        cmd, check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    img = cv2.imread(str(raw))
    if img is None:
        raise RuntimeError("Camera capture failed")

    img = cv2.rotate(img, cv2.ROTATE_180)

    if not cv2.imwrite(str(final), img):
        raise RuntimeError("Could not save frame")

    try:
        raw.unlink()
    except Exception:
        pass

    return img, final

# ============================================================
# VISUAL FEATURES -- preserved from v4
# ============================================================

def texture_score(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    return float(np.count_nonzero(edges)) / float(edges.size)

def pattern_scale(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    inv = 255 - edges
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    if vals.size == 0:
        return 0.0
    return float(np.percentile(vals, 85))

def visual_change(prev_img, img):
    if prev_img is None:
        return 999.0
    a = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    a = cv2.resize(a, (320, 240))
    b = cv2.resize(b, (320, 240))
    return float(cv2.absdiff(a, b).mean())

def analyze(img):
    h, w = img.shape[:2]
    y1 = int(h * 0.42)
    y2 = int(h * 0.86)
    nav = img[y1:y2, :]
    gray = cv2.cvtColor(nav, cv2.COLOR_BGR2GRAY)
    third = gray.shape[1] // 3
    regions = {
        "L": gray[:, :third],
        "C": gray[:, third:2*third],
        "R": gray[:, 2*third:]
    }
    texture = {}
    scale = {}
    for k, roi in regions.items():
        texture[k] = texture_score(roi)
        scale[k] = pattern_scale(roi)
    return texture, scale, y1, y2

def very_near_wall(texture, scale):
    # Critical v4 lesson: all-blue / lost hexagons + saturated scale is
    # an EXTREME close-wall signature, not open water.
    all_edges_gone = all(texture[k] <= VERY_LOW_EDGE for k in ("L", "C", "R"))
    scale_saturated = any(
        (not np.isfinite(scale[k])) or scale[k] >= INVALID_SCALE
        for k in ("L", "C", "R")
    )
    return all_edges_gone and scale_saturated

def wall_view_opened(texture, scale):
    """Hysteretic release test: require real structure to reappear."""
    recovered_edge = max(texture[k] for k in ("L", "C", "R")) >= NEAR_RELEASE_EDGE
    recovered_scale = all(
        np.isfinite(scale[k]) and scale[k] < NEAR_RELEASE_SCALE
        for k in ("L", "C", "R")
    )
    return recovered_edge and recovered_scale

# ============================================================
# STATE / HISTORY
# ============================================================

actions = deque(maxlen=HISTORY_LEN)
consecutive_turns = 0
stagnant_count = 0
escape_count = 0
last_img = None
last_turn_direction = None

very_near_count = 0
very_near_escape_direction = None
very_near_escape_attempts = 0
recovering_from_wall = False

# v5 latch: after a committed escape, force translation before normal
# visual steering can reacquire the same wall.
post_escape_forward_remaining = 0

def recent_same_turn():
    recent = [a for a in list(actions)[-MAX_CONSECUTIVE_TURNS:]
              if a in ("LEFT", "RIGHT")]
    if len(recent) != MAX_CONSECUTIVE_TURNS:
        return None
    if all(a == "LEFT" for a in recent):
        return "LEFT"
    if all(a == "RIGHT" for a in recent):
        return "RIGHT"
    return None

def repeated_turning():
    if consecutive_turns >= MAX_CONSECUTIVE_TURNS:
        return last_turn_direction
    return recent_same_turn()

# ============================================================
# NAVIGATION
# ============================================================

def choose_normal_action(texture, scale):
    L, C, R = texture["L"], texture["C"], texture["R"]
    LS, CS, RS = scale["L"], scale["C"], scale["R"]

    left_near = (
        LS > NEAR_SCALE_THRESHOLD
        and L > R * SIDE_DOMINANCE
        and (L - R) > SIDE_ABS_DIFFERENCE
    )
    right_near = (
        RS > NEAR_SCALE_THRESHOLD
        and R > L * SIDE_DOMINANCE
        and (R - L) > SIDE_ABS_DIFFERENCE
    )

    if left_near and not right_near:
        return "RIGHT", "AVOID", "strong near-wall cue on left"

    if right_near and not left_near:
        return "LEFT", "AVOID", "strong near-wall cue on right"

    side_best_scale = min(LS, RS)
    if CS > NEAR_SCALE_THRESHOLD and CS > side_best_scale * 1.20:
        if LS < RS:
            return "LEFT", "AVOID", "center constrained; left appears farther"
        return "RIGHT", "AVOID", "center constrained; right appears farther"

    return "FORWARD", "VOYAGE", "no strong collision cue"

# ============================================================
# v5 ESCAPE
# ============================================================

def choose_escape_direction(texture, scale, repeated_direction=None):
    global escape_count

    # PRIMARY v5 rule:
    # if the controller has been repeatedly steering one way,
    # escape decisively in the OPPOSITE direction.
    if repeated_direction == "RIGHT":
        return "LEFT"
    if repeated_direction == "LEFT":
        return "RIGHT"

    # For stagnation without repeated steering, retain v4 visual preference.
    LS, RS = scale["L"], scale["R"]
    if abs(LS - RS) > 1.0:
        return "LEFT" if LS < RS else "RIGHT"

    return "LEFT" if escape_count % 2 == 0 else "RIGHT"

def execute_escape(direction):
    global escape_count

    print(
        f"       ESCAPE COMMIT: {direction} for {ESCAPE_TURN_SEC:.2f}s"
        f" + {ESCAPE_FORWARD_BURSTS} forward burst(s)"
    )

    if direction == "LEFT":
        turn_left(ESCAPE_TURN_SEC)
    else:
        turn_right(ESCAPE_TURN_SEC)

    # Unlike v4, don't merely pivot and instantly allow the same avoidance
    # command to take control again. Physically depart after the counter-turn.
    for _ in range(ESCAPE_FORWARD_BURSTS):
        forward()

    escape_count += 1

def execute_very_near_escape(direction):
    # v6: committed wall-release pivot. Still re-observe after every
    # maneuver, but do not oscillate directions after tiny pulses.
    print(
        f"       VERY_NEAR RELEASE: committed {direction} "
        f"for {VERY_NEAR_TURN_SEC:.2f}s, then RE-OBSERVE"
    )
    if direction == "LEFT":
        turn_left(VERY_NEAR_TURN_SEC)
    else:
        turn_right(VERY_NEAR_TURN_SEC)

def execute_recovery_forward():
    print("       RECOVERY: cautious forward pulse")
    forward(RECOVERY_LEFT_SEC, RECOVERY_RIGHT_SEC)

def execute(action, state):
    if action == "FORWARD":
        forward()
    elif action == "LEFT":
        turn_left()
        if state == "AVOID":
            forward(CAUTIOUS_LEFT_SEC, CAUTIOUS_RIGHT_SEC)
    elif action == "RIGHT":
        turn_right()
        if state == "AVOID":
            forward(CAUTIOUS_LEFT_SEC, CAUTIOUS_RIGHT_SEC)
    else:
        stop()

# ============================================================
# DIAGNOSTIC IMAGE
# ============================================================

def annotate(img, texture, scale, state, action, reason, change, y1, y2):
    out = img.copy()
    h, w = out.shape[:2]
    third = w // 3

    cv2.rectangle(out, (0, y1), (w-1, y2), (255,255,255), 2)
    cv2.line(out, (third, y1), (third, y2), (255,255,255), 2)
    cv2.line(out, (2*third, y1), (2*third, y2), (255,255,255), 2)

    cv2.putText(out, f"STATE {state}   ACTION {action}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255,255,255), 2)
    cv2.putText(out,
                f"EDGE L {texture['L']:.3f} C {texture['C']:.3f} R {texture['R']:.3f}",
                (20,75), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(out,
                f"SCALE L {scale['L']:.1f} C {scale['C']:.1f} R {scale['R']:.1f}",
                (20,105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(out, f"visual change {change:.1f}", (20,135),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    cv2.putText(out, reason[:90], (20,165),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return out

# ============================================================
# STARTUP / PARAMETER LOGGING
# ============================================================

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
logfile = OUT / f"run_{stamp}.csv"
settings_file = OUT / f"settings_{stamp}.txt"

settings = {
    "steer_sec": STEER_SEC,
    "escape_turn_sec": ESCAPE_TURN_SEC,
    "escape_forward_bursts": ESCAPE_FORWARD_BURSTS,
    "post_escape_forward_cycles": POST_ESCAPE_FORWARD_CYCLES,
    "repeat_turn_limit": MAX_CONSECUTIVE_TURNS,
    "forward_left_sec": FWD_LEFT_SEC,
    "forward_right_sec": FWD_RIGHT_SEC,
    "cautious_left_sec": CAUTIOUS_LEFT_SEC,
    "cautious_right_sec": CAUTIOUS_RIGHT_SEC,
    "recovery_left_sec": RECOVERY_LEFT_SEC,
    "recovery_right_sec": RECOVERY_RIGHT_SEC,
    "gap_sec": GAP_SEC,
    "side_dominance": SIDE_DOMINANCE,
    "side_abs_difference": SIDE_ABS_DIFFERENCE,
    "near_scale_threshold": NEAR_SCALE_THRESHOLD,
    "min_visual_change": MIN_VISUAL_CHANGE,
    "max_stagnant": MAX_STAGNANT,
    "very_low_edge": VERY_LOW_EDGE,
    "invalid_scale": INVALID_SCALE,
    "very_near_turn_sec": VERY_NEAR_TURN_SEC,
    "max_same_near_turns": MAX_SAME_ESCAPE_TURNS,
    "near_release_edge": NEAR_RELEASE_EDGE,
    "near_release_scale": NEAR_RELEASE_SCALE,
    "camera_timeout_ms": CAMERA_TIMEOUT_MS,
}

with open(settings_file, "w") as sf:
    for k, v in settings.items():
        sf.write(f"{k}={v}\n")

print()
print("================================================")
print(" OCEAN VOYAGER CONTINUOUS NAVIGATION v6")
print(" VOYAGE -> AVOID -> ESCAPE/VERY_NEAR -> DEPART")
print("================================================")
print()
print("Output :", OUT)
print("Log    :", logfile)
print("Settings:", settings_file)
print()
print("v6 ACTIVE DEFAULTS (no CLI parameters required)")
print(f"  ordinary steer             {STEER_SEC:.2f} s")
print(f"  ESCAPE counter-turn        {ESCAPE_TURN_SEC:.2f} s")
print(f"  ESCAPE forward bursts      {ESCAPE_FORWARD_BURSTS}")
print(f"  post-escape forward cycles {POST_ESCAPE_FORWARD_CYCLES}")
print(f"  repeated-turn limit        {MAX_CONSECUTIVE_TURNS}")
print(f"  VERY_NEAR committed turn   {VERY_NEAR_TURN_SEC:.2f} s x up to {MAX_SAME_ESCAPE_TURNS}")
print(f"  normal forward L/R         {FWD_LEFT_SEC:.2f} / {FWD_RIGHT_SEC:.2f} s")
print()
print("*** CTRL-C = IMMEDIATE MOTOR STOP ***")
print("Starting in 5 seconds...")
print()

stop()
sleep(5)

cycle = 0

try:
    with open(logfile, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cycle","state","action","reason",
            "edge_left","edge_center","edge_right",
            "scale_left","scale_center","scale_right",
            "visual_change","consecutive_turns","stagnant_count",
            "escape_count","very_near","very_near_attempts",
            "post_escape_forward_remaining",
            "steer_sec","escape_turn_sec","escape_forward_bursts",
            "post_escape_forward_cycles","repeat_turn_limit",
            "forward_left_sec","forward_right_sec",
            "frame"
        ])

        while True:
            cycle += 1
            stop()

            img, frame_path = capture(cycle)
            texture, scale, y1, y2 = analyze(img)
            change = visual_change(last_img, img)
            close_now = very_near_wall(texture, scale)

            if close_now:
                very_near_count += 1
            else:
                very_near_count = 0

            if last_img is not None and change < MIN_VISUAL_CHANGE:
                stagnant_count += 1
            else:
                stagnant_count = 0

            repeated_direction = repeated_turning()

            # 1. Actual feature-collapse/contact signature always has priority.
            if very_near_count >= VERY_NEAR_CONFIRM_FRAMES:
                state = "VERY_NEAR"

                if very_near_escape_direction is None:
                    # If we know which way we were steering into the wall,
                    # start by committing to the opposite direction.
                    if last_turn_direction == "RIGHT":
                        very_near_escape_direction = "LEFT"
                    elif last_turn_direction == "LEFT":
                        very_near_escape_direction = "RIGHT"
                    else:
                        very_near_escape_direction = (
                            "LEFT" if escape_count % 2 == 0 else "RIGHT"
                        )
                    very_near_escape_attempts = 0

                if very_near_escape_attempts >= MAX_SAME_ESCAPE_TURNS:
                    very_near_escape_direction = (
                        "RIGHT" if very_near_escape_direction == "LEFT" else "LEFT"
                    )
                    very_near_escape_attempts = 0

                action = very_near_escape_direction
                reason = "feature collapse / saturated scale: wall extremely near"
                recovering_from_wall = True
                post_escape_forward_remaining = 0

            # 2. VERY_NEAR has cleared: cautious recovery first.
            elif recovering_from_wall and wall_view_opened(texture, scale):
                state = "RECOVERY"
                action = "FORWARD"
                reason = "wall structure recovered; cautious translation away"

            elif recovering_from_wall:
                # Exact saturation may have cleared, but the view is not open
                # enough yet. Keep the same committed release direction.
                state = "VERY_NEAR"
                action = very_near_escape_direction
                reason = "wall view not yet open; continue committed release"

            # 3. v5 committed departure latch after normal ESCAPE.
            elif post_escape_forward_remaining > 0:
                state = "DEPART"
                action = "FORWARD"
                reason = (
                    "post-escape committed forward departure "
                    f"({post_escape_forward_remaining} remaining)"
                )

            else:
                stuck = (
                    repeated_direction is not None
                    or stagnant_count >= MAX_STAGNANT
                )

                if stuck:
                    state = "ESCAPE"
                    action = choose_escape_direction(
                        texture, scale, repeated_direction
                    )

                    if repeated_direction is not None:
                        reason = (
                            f"repeated {repeated_direction} steering; "
                            f"committed opposite escape {action}"
                        )
                    else:
                        reason = "insufficient visual progress; committed escape"
                else:
                    action, state, reason = choose_normal_action(texture, scale)

            diagnostic = annotate(
                img, texture, scale, state, action, reason, change, y1, y2
            )
            diag_path = OUT / f"decision_{cycle:05d}.jpg"
            cv2.imwrite(str(diag_path), diagnostic)

            print()
            print(
                f"{cycle:04d} | {state:<9} | {action:<7} | "
                f"E {texture['L']:.3f}/{texture['C']:.3f}/{texture['R']:.3f} | "
                f"S {scale['L']:.1f}/{scale['C']:.1f}/{scale['R']:.1f} | "
                f"d={change:.1f}"
            )
            print("       ", reason)

            writer.writerow([
                cycle,state,action,reason,
                texture["L"],texture["C"],texture["R"],
                scale["L"],scale["C"],scale["R"],
                change,consecutive_turns,stagnant_count,
                escape_count,int(close_now),very_near_escape_attempts,
                post_escape_forward_remaining,
                STEER_SEC,ESCAPE_TURN_SEC,ESCAPE_FORWARD_BURSTS,
                POST_ESCAPE_FORWARD_CYCLES,MAX_CONSECUTIVE_TURNS,
                FWD_LEFT_SEC,FWD_RIGHT_SEC,
                str(frame_path)
            ])
            f.flush()

            # ---------------- ACT ----------------
            if state == "VERY_NEAR":
                execute_very_near_escape(action)
                very_near_escape_attempts += 1
                actions.append("VERY_NEAR_" + action)
                consecutive_turns = 0
                stagnant_count = 0

            elif state == "RECOVERY":
                execute_recovery_forward()
                actions.append("RECOVERY_FORWARD")
                recovering_from_wall = False
                very_near_escape_direction = None
                very_near_escape_attempts = 0
                consecutive_turns = 0
                stagnant_count = 0

            elif state == "ESCAPE":
                execute_escape(action)
                actions.append("ESCAPE_" + action)
                consecutive_turns = 0
                stagnant_count = 0
                # These are ADDITIONAL observed/captured forward cycles,
                # after execute_escape's immediate forward bursts.
                post_escape_forward_remaining = POST_ESCAPE_FORWARD_CYCLES

            elif state == "DEPART":
                forward()
                actions.append("DEPART_FORWARD")
                post_escape_forward_remaining -= 1
                consecutive_turns = 0
                stagnant_count = 0

            else:
                execute(action, state)
                actions.append(action)

                if action in ("LEFT", "RIGHT"):
                    consecutive_turns += 1
                    last_turn_direction = action
                else:
                    consecutive_turns = 0

            last_img = img

except KeyboardInterrupt:
    stop()

finally:
    stop()

print()
print("Voyager stopped.")
print("Motors OFF.")
print("Data:", OUT)
