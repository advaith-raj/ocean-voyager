from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from pathlib import Path
from datetime import datetime
import subprocess
import threading
import cv2
import numpy as np
import csv
import signal
import sys
import argparse

# ============================================================
# OCEAN VOYAGER - CONTINUOUS VISUAL NAVIGATION v12
#
# SIMPLE CLOSED-LOOP ARCHITECTURE
#
#   FORWARD_SCAN:
#       calibrated forward propulsion runs continuously
#       camera streams continuously
#       if view becomes too close -> TURN_SCAN
#
#   TURN_SCAN:
#       ONE turn direction is latched
#       turning thrust remains active
#       camera keeps streaming/analyzing concurrently
#       first sufficiently open heading -> FORWARD_SCAN immediately
#
# Pool-calibrated OPEN plus persistent multi-frame APPROACH detection.
# Feature collapse is an immediate emergency override.
# After OPEN, forward is committed briefly to exploit the runway.
# No timed 2.8 s turns.
# No automatic left/right toggling while scanning.
# No stop-and-take-picture control loop.
# Motors and vision run concurrently.
#
# Camera frames are processed and discarded. Images are only saved
# periodically / on state changes, with a hard disk-image cap.
# ============================================================

p = argparse.ArgumentParser(description="Ocean Voyager continuous visual navigation v9")

# ---------- Forward calibration ----------
p.add_argument("--forward-left-sec", type=float, default=0.50,
               help="left pump pulse in continuous forward cycle (default: 0.48)")
p.add_argument("--forward-right-sec", type=float, default=0.70,
               help="right pump pulse in continuous forward cycle (default: 0.30)")
p.add_argument("--forward-gap-sec", type=float, default=0.03,
               help="gap between forward pump pulses (default: 0.03)")
p.add_argument("--calibrate-forward", action="store_true",
               help="run forward continuously for a fixed calibration interval, then stop")
p.add_argument("--calibrate-seconds", type=float, default=8.0,
               help="forward calibration duration (default: 8 s)")
p.add_argument("--calibrate-vision", action="store_true",
               help="motors OFF; continuously print live L/C/R vision metrics")

# ---------- Continuous camera ----------
p.add_argument("--width", type=int, default=640)
p.add_argument("--height", type=int, default=480)
p.add_argument("--fps", type=int, default=8,
               help="analysis stream target FPS (default: 8)")

# ---------- Wall/open thresholds ----------
p.add_argument("--close-scale", type=float, default=105.0,
               help="center scale at/below this confirms CLOSE (default: 105)")
p.add_argument("--open-scale", type=float, default=114.0,
               help="center scale at/above this can confirm OPEN (default: 114)")
p.add_argument("--open-edge-min", type=float, default=0.010,
               help="center edge density must exceed this for OPEN; <= means feature collapse (default: 0.010)")
p.add_argument("--very-low-edge", type=float, default=0.004,
               help="all-region feature-collapse threshold (default: 0.004)")
p.add_argument("--invalid-scale", type=float, default=1000.0,
               help="saturated distance-transform scale threshold (default: 1000)")
p.add_argument("--open-confirm-frames", type=int, default=2,
               help="consecutive OPEN frames required while turning (default: 2)")
p.add_argument("--close-confirm-frames", type=int, default=6,
               help="persistent low-scale frames required before turning (default: 6)")
p.add_argument("--approach-drop", type=float, default=4.0,
               help="minimum center-scale drop across the recent window to confirm approach (default: 4)")
p.add_argument("--forward-commit-sec", type=float, default=1.75,
               help="after OPEN, keep forward unless emergency feature collapse occurs (default: 1.75)")

# ---------- Turn choice ----------
p.add_argument("--default-turn", choices=("left", "right"), default="right",
               help="direction used only when image has no directional information (default: right)")
p.add_argument("--turn-pulse-sec", type=float, default=0.60,
               help="single same-direction turn pulse during TURN_SCAN (default: 0.60)")
p.add_argument("--turn-scan-sec", type=float, default=0.75,
               help="motor-off scan/coast interval after each turn pulse (default: 0.75)")

# ---------- Bounded diagnostics ----------
p.add_argument("--save-every-sec", type=float, default=5.0,
               help="periodic diagnostic image interval; 0 disables (default: 5)")
p.add_argument("--max-saved-images", type=int, default=200,
               help="hard cap on diagnostic JPEGs; oldest are deleted (default: 200)")

args = p.parse_args()

FWD_LEFT_SEC = max(0.01, args.forward_left_sec)
FWD_RIGHT_SEC = max(0.01, args.forward_right_sec)
FWD_GAP_SEC = max(0.0, args.forward_gap_sec)

CLOSE_SCALE = args.close_scale
OPEN_SCALE = args.open_scale
OPEN_EDGE_MIN = args.open_edge_min
VERY_LOW_EDGE = args.very_low_edge
INVALID_SCALE = args.invalid_scale
OPEN_CONFIRM_FRAMES = max(1, args.open_confirm_frames)
CLOSE_CONFIRM_FRAMES = max(2, args.close_confirm_frames)
APPROACH_DROP = max(0.0, args.approach_drop)
FORWARD_COMMIT_SEC = max(0.0, args.forward_commit_sec)

OUT = Path("/tmp/voyager_v12")
OUT.mkdir(parents=True, exist_ok=True)

# ---------- Exact working GPIO mapping ----------
L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)
R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

def motors_off():
    L1.off(); L2.off(); R1.off(); R2.off()

def left_pump_on():
    R1.off(); R2.off()
    L1.on(); L2.off()

def right_pump_on():
    L1.off(); L2.off()
    R1.on(); R2.off()

# ============================================================
# MOTOR WORKER
# Keeps propulsion running while the main thread analyzes camera frames.
# ============================================================

class MotorController:
    def __init__(self):
        self._mode = "OFF"
        self._lock = threading.Lock()
        self._quit = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def set_mode(self, mode):
        if mode not in ("OFF", "FORWARD", "TURN_LEFT", "TURN_RIGHT"):
            raise ValueError(mode)
        with self._lock:
            self._mode = mode

    def mode(self):
        with self._lock:
            return self._mode

    def _same_mode(self, expected):
        return (not self._quit.is_set()) and self.mode() == expected

    def _interruptible_wait(self, seconds, expected):
        end = monotonic() + seconds
        while monotonic() < end:
            if not self._same_mode(expected):
                return False
            sleep(min(0.01, max(0.0, end - monotonic())))
        return True

    def _run(self):
        while not self._quit.is_set():
            mode = self.mode()

            if mode == "OFF":
                motors_off()
                sleep(0.01)

            elif mode == "TURN_LEFT":
                # Physical RIGHT pump rotates Voyager LEFT.
                L1.off(); L2.off()
                R1.on(); R2.off()
                sleep(0.01)

            elif mode == "TURN_RIGHT":
                # Physical LEFT pump rotates Voyager RIGHT.
                R1.off(); R2.off()
                L1.on(); L2.off()
                sleep(0.01)

            elif mode == "FORWARD":
                # Preserve the proven sequential forward calibration, but run
                # it continuously in this worker while vision runs concurrently.
                motors_off()
                L1.on(); L2.off()
                if not self._interruptible_wait(FWD_LEFT_SEC, "FORWARD"):
                    motors_off()
                    continue

                motors_off()
                if FWD_GAP_SEC and not self._interruptible_wait(FWD_GAP_SEC, "FORWARD"):
                    continue

                R1.on(); R2.off()
                if not self._interruptible_wait(FWD_RIGHT_SEC, "FORWARD"):
                    motors_off()
                    continue

                motors_off()
                if FWD_GAP_SEC:
                    self._interruptible_wait(FWD_GAP_SEC, "FORWARD")

        motors_off()

    def close(self):
        self.set_mode("OFF")
        self._quit.set()
        self._thread.join(timeout=1.0)
        motors_off()

motor = MotorController()
camera_proc = None

def shutdown(signum=None, frame=None):
    global camera_proc
    try:
        motor.close()
    except Exception:
        motors_off()
    if camera_proc is not None:
        try:
            camera_proc.terminate()
        except Exception:
            pass
    motors_off()
    print("\n*** VOYAGER STOPPED - MOTORS OFF ***")
    sys.exit(0)

signal.signal(signal.SIGINT, shutdown)
signal.signal(signal.SIGTERM, shutdown)

# ============================================================
# FORWARD CALIBRATION MODE
# Same script; no new calibration script required.
# ============================================================

def run_forward_calibration():
    print()
    print("================================================")
    print(" V9 FORWARD CALIBRATION")
    print("================================================")
    print(f" LEFT pulse : {FWD_LEFT_SEC:.3f} s")
    print(f" RIGHT pulse: {FWD_RIGHT_SEC:.3f} s")
    print(f" GAP        : {FWD_GAP_SEC:.3f} s")
    print(f" RUN        : {args.calibrate_seconds:.1f} s")
    print()
    print("Starting in 3 seconds. Watch the boat's overall path.")
    for n in (3, 2, 1):
        print(n)
        sleep(1)

    motor.set_mode("FORWARD")
    start = monotonic()
    try:
        while monotonic() - start < args.calibrate_seconds:
            sleep(0.05)
    finally:
        motor.set_mode("OFF")
        sleep(0.05)
        motors_off()

    print("Calibration run complete; motors OFF.")
    print()
    print("Tune and rerun the SAME script, for example:")
    print("  --forward-left-sec 0.44 --forward-right-sec 0.30")
    print("or")
    print("  --forward-left-sec 0.48 --forward-right-sec 0.34")

# ============================================================
# CONTINUOUS MJPEG CAMERA
# One rpicam-vid process for the whole run; frames are not accumulated.
# ============================================================

class MJPEGCamera:
    def __init__(self, width, height, fps):
        global camera_proc
        cmd = [
            "rpicam-vid",
            "--codec", "mjpeg",
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--timeout", "0",
            "--nopreview",
            "-o", "-"
        ]
        self.proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0
        )
        camera_proc = self.proc
        self.buf = bytearray()

    def read(self):
        # Keep only enough bytes to find the next complete JPEG.
        while True:
            soi = self.buf.find(b"\xff\xd8")
            if soi >= 0:
                eoi = self.buf.find(b"\xff\xd9", soi + 2)
                if eoi >= 0:
                    jpg = bytes(self.buf[soi:eoi + 2])
                    del self.buf[:eoi + 2]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        # Physical camera is mounted inverted.
                        return cv2.rotate(img, cv2.ROTATE_180)

            chunk = self.proc.stdout.read(8192)
            if not chunk:
                raise RuntimeError("Continuous camera stream ended")

            self.buf.extend(chunk)

            # Defensive bound: never allow a malformed stream to grow RAM.
            if len(self.buf) > 4_000_000:
                last_soi = self.buf.rfind(b"\xff\xd8")
                if last_soi > 0:
                    del self.buf[:last_soi]
                else:
                    self.buf.clear()

    def close(self):
        global camera_proc
        try:
            self.proc.terminate()
            self.proc.wait(timeout=1)
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
        camera_proc = None

# ============================================================
# FAST VISUAL ANALYSIS
# ============================================================

def region_metrics(gray):
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    inv = 255 - edges
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    scale = float(np.percentile(vals, 85)) if vals.size else 0.0
    return edge_density, scale

def analyze(img):
    h, w = img.shape[:2]
    # Same navigation band concept as prior versions.
    y1 = int(h * 0.42)
    y2 = int(h * 0.86)
    gray = cv2.cvtColor(img[y1:y2, :], cv2.COLOR_BGR2GRAY)
    third = gray.shape[1] // 3

    rois = {
        "L": gray[:, :third],
        "C": gray[:, third:2*third],
        "R": gray[:, 2*third:]
    }

    edge = {}
    scale = {}
    for k, roi in rois.items():
        edge[k], scale[k] = region_metrics(roi)

    all_featureless = all(edge[k] <= VERY_LOW_EDGE for k in ("L", "C", "R"))
    saturated = any(
        (not np.isfinite(scale[k])) or scale[k] >= INVALID_SCALE
        for k in ("L", "C", "R")
    )
    very_near = all_featureless and saturated

    # Pool-calibrated interpretation:
    #   ~0.5 ft: center edge collapses to roughly 0.004-0.008 and scale is unstable.
    #   ~1 ft:   center scale is roughly 93-96.
    #   open:    center scale is typically 114+.
    #
    # Therefore feature collapse ALWAYS vetoes OPEN, and scale is interpreted
    # in the measured direction: lower = closer, higher = more open.
    center_feature_collapse = edge["C"] <= OPEN_EDGE_MIN

    close = (
        very_near
        or center_feature_collapse
        or (
            np.isfinite(scale["C"])
            and scale["C"] < INVALID_SCALE
            and scale["C"] <= CLOSE_SCALE
        )
    )

    open_view = (
        not very_near
        and not center_feature_collapse
        and np.isfinite(scale["C"])
        and scale["C"] < INVALID_SCALE
        and scale["C"] >= OPEN_SCALE
    )

    return edge, scale, very_near, close, open_view, y1, y2

def choose_turn_direction(edge, scale, very_near):
    # If the frame is completely featureless/saturated, it has no directional
    # information. Pick the configured direction ONCE; TURN_SCAN then latches it.
    if very_near:
        return args.default_turn.upper()

    # Turn toward the side that appears farther/open: lower finite pattern scale.
    ls = scale["L"]
    rs = scale["R"]

    l_ok = np.isfinite(ls) and ls < INVALID_SCALE
    r_ok = np.isfinite(rs) and rs < INVALID_SCALE

    if l_ok and r_ok:
        if ls < rs:
            return "LEFT"
        if rs < ls:
            return "RIGHT"

    # If only one side has a valid scale, turn toward that known side.
    if l_ok and not r_ok:
        return "LEFT"
    if r_ok and not l_ok:
        return "RIGHT"

    # If neither side has trustworthy distance information, do NOT infer
    # direction from a saturated/featureless region. Use the configured
    # default once and latch it for the entire TURN_SCAN episode.
    return args.default_turn.upper()

# ============================================================
# BOUNDED DIAGNOSTICS
# ============================================================

saved_images = []

def save_diag(img, frame_no, state, edge, scale, reason):
    if args.max_saved_images <= 0:
        return

    out = img.copy()
    cv2.putText(out, f"{state}  {reason}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255,255,255), 2)
    cv2.putText(out,
                f"E {edge['L']:.3f}/{edge['C']:.3f}/{edge['R']:.3f}",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)
    cv2.putText(out,
                f"S {scale['L']:.1f}/{scale['C']:.1f}/{scale['R']:.1f}",
                (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255,255,255), 2)

    path = OUT / f"diag_{frame_no:07d}.jpg"
    cv2.imwrite(str(path), out)
    saved_images.append(path)

    while len(saved_images) > args.max_saved_images:
        old = saved_images.pop(0)
        try:
            old.unlink()
        except FileNotFoundError:
            pass

def run_vision_calibration():
    print()
    print("================================================")
    print(" V12 LIVE VISION CALIBRATION - MOTORS OFF")
    print("================================================")
    print("Point Voyager at: OPEN/FAR, then MEDIUM, then CLOSE.")
    print("Metrics print continuously. Ctrl-C exits.")
    motors_off()
    cam = MJPEGCamera(args.width, args.height, args.fps)
    frame_no = 0
    try:
        while True:
            img = cam.read()
            frame_no += 1
            edge, scale, very_near, close, open_view, _, _ = analyze(img)
            print(
                f"{frame_no:06d} | "
                f"E {edge['L']:.3f}/{edge['C']:.3f}/{edge['R']:.3f} | "
                f"S {scale['L']:.1f}/{scale['C']:.1f}/{scale['R']:.1f} | "
                f"VN={int(very_near)} CLOSE={int(close)} OPEN={int(open_view)}"
            )
    finally:
        motors_off()
        cam.close()

# ============================================================
# MAIN CONTINUOUS CONTROL LOOP
# ============================================================

def autonomous():
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = OUT / f"run_{stamp}.csv"
    settings = OUT / f"settings_{stamp}.txt"

    with open(settings, "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}={v}\n")

    print()
    print("================================================")
    print(" OCEAN VOYAGER v12 - CONTINUOUS VISION CONTROL")
    print("================================================")
    print(f"camera           : {args.width}x{args.height} @ {args.fps} fps target")
    print(f"forward L/R      : {FWD_LEFT_SEC:.3f}s / {FWD_RIGHT_SEC:.3f}s")
    print(f"turn pulse/scan   : {args.turn_pulse_sec:.2f}s / {args.turn_scan_sec:.2f}s")
    print(f"close/open scale : <= {CLOSE_SCALE:.1f} / >= {OPEN_SCALE:.1f}")
    print(f"feature collapse : E_center <= {OPEN_EDGE_MIN:.3f} => CLOSE / never OPEN")
    print(f"confirm frames   : approach={CLOSE_CONFIRM_FRAMES}, open={OPEN_CONFIRM_FRAMES}")
    print(f"approach drop    : >= {APPROACH_DROP:.1f} scale units across window")
    print(f"forward commit   : {FORWARD_COMMIT_SEC:.2f} s after OPEN")
    print("emergency        : E_center <= threshold interrupts immediately")
    print("control           : FORWARD_SCAN <-> TURN_SCAN")
    print("TURN_SCAN         : short pulse -> motor OFF scan -> SAME direction pulse until OPEN")
    print("memory            : streaming; frames discarded after analysis")
    print(f"saved images cap  : {args.max_saved_images}")
    print()
    print("*** CTRL-C = MOTORS OFF ***")
    print("Starting in 5 seconds...")
    sleep(5)

    cam = MJPEGCamera(args.width, args.height, args.fps)

    state = "FORWARD_SCAN"
    turn_direction = None
    close_count = 0
    open_count = 0
    frame_no = 0
    last_save = monotonic()
    center_scale_history = []
    center_edge_history = []
    forward_commit_until = 0.0
    next_turn_pulse_at = 0.0
    turn_pulse_until = 0.0
    scanning_after_pulse = False

    motor.set_mode("FORWARD")

    try:
        with open(logfile, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "time_s","frame","state","motor_mode","turn_direction",
                "edge_L","edge_C","edge_R",
                "scale_L","scale_C","scale_R",
                "very_near","close","open_view",
                "close_count","open_count","reason"
            ])
            t0 = monotonic()

            while True:
                img = cam.read()
                frame_no += 1

                edge, scale, very_near, close, open_view, y1, y2 = analyze(img)

                center_scale_history.append(scale["C"])
                center_edge_history.append(edge["C"])
                history_max = max(CLOSE_CONFIRM_FRAMES, 6)
                if len(center_scale_history) > history_max:
                    center_scale_history.pop(0)
                    center_edge_history.pop(0)

                reason = ""

                if state == "FORWARD_SCAN":
                    open_count = 0
                    now_forward = monotonic()

                    # Emergency override: the measured half-foot signature
                    # (center edge collapse) can interrupt even committed forward.
                    emergency_close = very_near or edge["C"] <= OPEN_EDGE_MIN

                    # Ordinary CLOSE is deliberately conservative. A single low
                    # scale is not enough; require a sustained low-scale window
                    # AND evidence that the view is worsening/approaching.
                    low_scale_now = (
                        np.isfinite(scale["C"])
                        and scale["C"] < INVALID_SCALE
                        and scale["C"] <= CLOSE_SCALE
                    )

                    if low_scale_now:
                        close_count += 1
                    else:
                        close_count = 0

                    approach_confirmed = False
                    if (
                        close_count >= CLOSE_CONFIRM_FRAMES
                        and len(center_scale_history) >= CLOSE_CONFIRM_FRAMES
                    ):
                        recent = center_scale_history[-CLOSE_CONFIRM_FRAMES:]
                        valid_recent = [
                            x for x in recent
                            if np.isfinite(x) and x < INVALID_SCALE
                        ]
                        if len(valid_recent) == CLOSE_CONFIRM_FRAMES:
                            first_half = valid_recent[:max(1, len(valid_recent)//2)]
                            second_half = valid_recent[len(valid_recent)//2:]
                            early = float(np.median(first_half))
                            late = float(np.median(second_half))
                            approach_confirmed = (early - late) >= APPROACH_DROP

                    committed = now_forward < forward_commit_until

                    should_turn = emergency_close or (
                        (not committed) and approach_confirmed
                    )

                    if should_turn:
                        turn_direction = choose_turn_direction(edge, scale, very_near)
                        state = "TURN_SCAN"
                        close_count = 0
                        open_count = 0

                        # Direction is latched. Apply one short pulse,
                        # then motors OFF while the live camera keeps scanning.
                        if turn_direction == "LEFT":
                            motor.set_mode("TURN_LEFT")
                        else:
                            motor.set_mode("TURN_RIGHT")
                        turn_pulse_until = monotonic() + max(0.05, args.turn_pulse_sec)
                        next_turn_pulse_at = turn_pulse_until + max(0.05, args.turn_scan_sec)
                        scanning_after_pulse = False

                        trigger = "EMERGENCY" if emergency_close else "APPROACH"
                        reason = f"{trigger} -> latch {turn_direction}; pulse+scan"
                        print(
                            f"{frame_no:06d} {trigger} | "
                            f"E_C={edge['C']:.3f} "
                            f"S {scale['L']:.1f}/{scale['C']:.1f}/{scale['R']:.1f} "
                            f"-> TURN_{turn_direction}"
                        )
                        save_diag(img, frame_no, state, edge, scale, reason)

                else:  # TURN_SCAN
                    close_count = 0
                    now_turn = monotonic()

                    # End the active turn pulse, but keep scanning while the
                    # boat naturally drifts/rotates.
                    if motor.mode() in ("TURN_LEFT", "TURN_RIGHT") and now_turn >= turn_pulse_until:
                        motor.set_mode("OFF")
                        scanning_after_pulse = True

                    if open_view:
                        # Require consecutive open observations. If we have two
                        # frames, reject an apparent OPEN sequence that is rapidly
                        # collapsing back toward the wall.
                        trend_ok = True
                        if len(center_scale_history) >= 2:
                            prev_s, cur_s = center_scale_history[-2], center_scale_history[-1]
                            # Small fluctuations are fine; a large one-frame drop
                            # is not accepted as a stable open heading.
                            if np.isfinite(prev_s) and np.isfinite(cur_s):
                                trend_ok = cur_s >= (prev_s - 12.0)

                        if trend_ok:
                            open_count += 1
                        else:
                            open_count = 0
                    else:
                        open_count = 0

                    if open_count >= OPEN_CONFIRM_FRAMES:
                        state = "FORWARD_SCAN"
                        motor.set_mode("FORWARD")
                        forward_commit_until = monotonic() + FORWARD_COMMIT_SEC
                        reason = (
                            f"OPEN -> FORWARD; commit {FORWARD_COMMIT_SEC:.2f}s "
                            f"from latched {turn_direction}"
                        )
                        print(
                            f"{frame_no:06d} OPEN  | "
                            f"S {scale['L']:.1f}/{scale['C']:.1f}/{scale['R']:.1f} "
                            f"-> FORWARD"
                        )
                        save_diag(img, frame_no, state, edge, scale, reason)
                        turn_direction = None
                        open_count = 0
                        close_count = 0
                        center_scale_history.clear()
                        center_edge_history.clear()
                        scanning_after_pulse = False

                    elif now_turn >= next_turn_pulse_at:
                        # Still not open: pulse the SAME latched direction.
                        # Never automatically toggle LEFT/RIGHT.
                        if turn_direction == "LEFT":
                            motor.set_mode("TURN_LEFT")
                        else:
                            motor.set_mode("TURN_RIGHT")
                        turn_pulse_until = now_turn + max(0.05, args.turn_pulse_sec)
                        next_turn_pulse_at = turn_pulse_until + max(0.05, args.turn_scan_sec)
                        scanning_after_pulse = False
                        reason = f"not open -> another {turn_direction} pulse"

                now = monotonic()
                if args.save_every_sec > 0 and now - last_save >= args.save_every_sec:
                    save_diag(img, frame_no, state, edge, scale, "periodic")
                    last_save = now

                writer.writerow([
                    now - t0, frame_no, state, motor.mode(), turn_direction or "",
                    edge["L"], edge["C"], edge["R"],
                    scale["L"], scale["C"], scale["R"],
                    int(very_near), int(close), int(open_view),
                    close_count, open_count, reason
                ])

                # Keep CSV useful without forcing a disk sync every frame.
                if frame_no % max(1, args.fps) == 0:
                    f.flush()

    finally:
        motor.set_mode("OFF")
        motors_off()
        cam.close()

if __name__ == "__main__":
    try:
        if args.calibrate_forward:
            run_forward_calibration()
        elif args.calibrate_vision:
            run_vision_calibration()
        else:
            autonomous()
    finally:
        try:
            motor.close()
        except Exception:
            motors_off()
        motors_off()
