from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from pathlib import Path
from datetime import datetime
import subprocess, threading, signal, sys, argparse, csv
from collections import deque
import cv2
import numpy as np

# ============================================================
# OCEAN VOYAGER v28
# Latest-frame asynchronous visual control
#
# Threads:
#   1) CAMERA: continuously decodes MJPEG and overwrites ONE latest-frame slot.
#   2) CONTROL: analyzes only fresh/new frames; never consumes a backlog.
#   3) MOTOR: executes the requested propulsion state independently.
#
# Key behavior:
#   - More looking, less turning.
#   - FORWARD is dominant; normal avoidance uses one bounded decisive turn.
#   - Collision recovery: OFF/rebound -> bounded turn -> brief OFF observation.
#   - Recovery rotations are never chained blindly; vision evaluates between them.
#   - No low-edge "emergency" by itself (open water can also have low edges).
#   - BUT persistent E_C~0 + saturated S_C is a confirmed collision signature:
#     3 fresh frames trigger escape (~0.2-0.3 s at 10 fps).
#   - Forward->turn requires persistent low scale + worsening trend.
#   - Predictive APPROACH uses a temporal trend before hard collision.
#   - Confirmed COLLISION keeps one direction patiently; no rapid L/R flip.
#   - Automatic reverse is disabled until reverse is physically calibrated.
#   - APPROACH->COLLISION promotion respects the current pulse/scan cadence.
#   - Visual yaw-rate is estimated during FORWARD and LOGGED ONLY in v15.
#   - Every frame and control event is timestamped.
# ============================================================

ap = argparse.ArgumentParser()

# Known forward calibration
ap.add_argument("--forward-left-sec", type=float, default=0.50)
ap.add_argument("--forward-right-sec", type=float, default=0.70)
ap.add_argument("--forward-gap-sec", type=float, default=0.03)

# Camera
ap.add_argument("--width", type=int, default=640)
ap.add_argument("--height", type=int, default=480)
ap.add_argument("--fps", type=int, default=10,
                help="requested camera FPS (default 10)")
ap.add_argument("--max-frame-age", type=float, default=0.40,
                help="discard control decisions from frames older than this many seconds")

# Measured pool thresholds
ap.add_argument("--close-scale", type=float, default=105.0)
ap.add_argument("--open-scale", type=float, default=114.0)
ap.add_argument("--open-edge-min", type=float, default=0.010,
                help="OPEN requires some center structure; low edge alone is NOT close")

# Temporal decisions
ap.add_argument("--approach-frames", type=int, default=12,
                help="recent frames used to confirm sustained approach")
ap.add_argument("--approach-drop", type=float, default=4.0,
                help="required early->late median scale drop")
ap.add_argument("--approach-late-max", type=float, default=118.0,
                help="predictive approach: late-window median scale must be <= this")
ap.add_argument("--open-confirm-frames", type=int, default=2)
ap.add_argument("--forward-commit-sec", type=float, default=5.0)
ap.add_argument("--collision-edge-max", type=float, default=0.002,
                help="collision saturation: center edge <= this")
ap.add_argument("--collision-scale-min", type=float, default=1000.0,
                help="collision saturation: center scale >= this")
ap.add_argument("--collision-confirm-frames", type=int, default=3,
                help="consecutive fresh saturated frames required (default 3)")

# Less movement / more vision
ap.add_argument("--turn-pulse-sec", type=float, default=0.25)
ap.add_argument("--avoid-min-turn-sec", type=float, default=0.50,
                help="minimum continuous turn time for pre-collision avoidance")
ap.add_argument("--collision-min-turn-sec", type=float, default=1.00,
                help="minimum continuous turn time after confirmed collision")
ap.add_argument("--avoid-max-turn-sec", type=float, default=2.00,
                help="maximum continuous turn time for pre-collision avoidance")
ap.add_argument("--collision-max-turn-sec", type=float, default=4.00,
                help="maximum continuous turn time after confirmed collision")
ap.add_argument("--stuck-reverse-sec", type=float, default=1.00,
                help="reverse duration when a saturated collision turn makes no progress")
ap.add_argument("--stuck-max-reverses", type=int, default=3,
                help="maximum consecutive reverse attempts while vision remains saturated")
ap.add_argument("--stuck-scan-sec", type=float, default=1.00,
                help="motor-off scan time after each stuck reverse attempt")
ap.add_argument("--turn-scan-sec", type=float, default=1.00)
ap.add_argument("--default-turn", choices=("left","right"), default="right")
ap.add_argument("--avoid-flip-pulses", type=int, default=6,
                help="ordinary avoidance: flip latched turn after this many unsuccessful pulses")
ap.add_argument("--collision-flip-pulses", type=int, default=18,
                help="patient same-direction collision pulse block; no auto flip in v19")
ap.add_argument("--reverse-sec", type=float, default=0.50,
                help="reverse duration after both collision turn directions fail")
ap.add_argument("--reverse-scan-sec", type=float, default=1.00,
                help="motor-off observation time after reverse")
ap.add_argument("--reverse-test", action="store_true",
                help="run one reverse pulse with motors otherwise off")
ap.add_argument("--reverse-test-sec", type=float, default=0.50,
                help="duration for --reverse-test")
# v27 history-aware supervisory policy
ap.add_argument("--history-sec", type=float, default=12.0,
                help="rolling action-history window used by supervisor")
ap.add_argument("--min-useful-forward-sec", type=float, default=2.0,
                help="forward progress desired between ordinary avoidance turns")
ap.add_argument("--forward-starve-sec", type=float, default=4.0,
                help="if no useful forward for this long, bias strongly to FORWARD unless unsafe")
ap.add_argument("--avoid-turn-sec", type=float, default=0.80,
                help="single decisive pre-collision turn duration")
ap.add_argument("--rebound-turn-sec", type=float, default=0.90,
                help="single turn after passive collision rebound/clearance")
ap.add_argument("--recovery-coast-sec", type=float, default=0.50,
                help="motors-OFF observation pause after a collision-recovery turn")
ap.add_argument("--recovery-coast", action=argparse.BooleanOptionalAction, default=True,
                help="pause OFF after each collision-recovery turn before resuming FORWARD")
ap.add_argument("--collision-coast-max-sec", type=float, default=1.50,
                help="maximum motors-OFF passive rebound/watch interval before active escape")
ap.add_argument("--stuck-hold-release-frames", type=int, default=5,
                help="fresh non-hard frames required to release STUCK_HOLD back to FORWARD")
ap.add_argument("--stuck-hold-max-sec", type=float, default=8.0,
                help="maximum STUCK_HOLD observation time before a bounded active escape turn")
ap.add_argument("--collision-escape-turn-sec", type=float, default=1.20,
                help="decisive active turn duration after hard-contact coast/hold fails")
ap.add_argument("--escape-clear-edge-min", type=float, default=0.040,
                help="center edge required before collision recovery may return to FORWARD")
ap.add_argument("--escape-clear-frames", type=int, default=5,
                help="consecutive fresh clearance frames required before collision recovery returns FORWARD")
ap.add_argument("--post-turn-cooldown-sec", type=float, default=2.0,
                help="ordinary same-direction turn penalty window")
ap.add_argument("--max-turn-sec-window", type=float, default=2.4,
                help="soft maximum cumulative turning in history window")
ap.add_argument("--approach-edge-window", type=int, default=7,
                help="fresh frames used by v27 center-edge approach trend")
ap.add_argument("--approach-edge-drop", type=float, default=0.010,
                help="required median E_C drop across approach window")
ap.add_argument("--approach-edge-late-max", type=float, default=0.020,
                help="late E_C must be at/below this for ordinary approach confirmation")
ap.add_argument("--approach-consistency", type=float, default=0.65,
                help="fraction of E_C steps that should worsen for trend confirmation")
ap.add_argument("--imminent-edge-max", type=float, default=0.008,
                help="very-close E_C threshold; requires consecutive frames")
ap.add_argument("--imminent-confirm-frames", type=int, default=2)

# v27 independently switchable supervisory features.  BooleanOptionalAction
# creates both --feature and --no-feature forms.
ap.add_argument("--history-supervisor", action=argparse.BooleanOptionalAction, default=True,
                help="enable/disable all history-based action supervision")
ap.add_argument("--loop-guard", action=argparse.BooleanOptionalAction, default=True,
                help="time/progress based self-loop detection and strategy escalation")
ap.add_argument("--turn-budget-guard", action=argparse.BooleanOptionalAction, default=True,
                help="penalize/exclude excessive cumulative turning")
ap.add_argument("--repeat-turn-guard", action=argparse.BooleanOptionalAction, default=True,
                help="penalize repeated same-direction turns without progress")
ap.add_argument("--oscillation-guard", action=argparse.BooleanOptionalAction, default=True,
                help="penalize rapid LEFT/RIGHT alternation")
ap.add_argument("--forward-progress-guard", action=argparse.BooleanOptionalAction, default=True,
                help="penalize ordinary turns when little useful FORWARD occurred")
ap.add_argument("--approach-trend", action=argparse.BooleanOptionalAction, default=True,
                help="enable temporal E_C pre-collision approach confirmation")
ap.add_argument("--imminent-guard", action=argparse.BooleanOptionalAction, default=True,
                help="enable very-close E_C immediate avoidance evidence")
ap.add_argument("--passive-rebound", action=argparse.BooleanOptionalAction, default=True,
                help="after hard contact, motors OFF and watch for passive rebound")
ap.add_argument("--opposite-after-collision", action=argparse.BooleanOptionalAction, default=True,
                help="after rebound, prefer opposite of the avoidance turn that led to contact")

# Loop definition: primarily time/progress based; action patterns are supporting evidence only.
ap.add_argument("--loop-no-progress-sec", type=float, default=6.0,
                help="seconds without a meaningful FORWARD run before loop pressure rises")
ap.add_argument("--loop-forward-ratio-max", type=float, default=0.25,
                help="recent FORWARD fraction at/below which loop pressure rises")
ap.add_argument("--loop-pressure-threshold", type=float, default=0.65,
                help="0..1 loop pressure at which repeating the current strategy is strongly discouraged")

ap.add_argument("--yaw-window", type=int, default=15,
                help="number of visual yaw-rate samples for smoothed diagnostic")
ap.add_argument("--yaw-min-features", type=int, default=12,
                help="minimum tracked features required for a yaw estimate")
ap.add_argument("--yaw-print-sec", type=float, default=1.0,
                help="print smoothed visual yaw diagnostic at most this often")

# Passive MLX90640 telemetry only — NEVER controls navigation in v27.
ap.add_argument("--thermal-log", action=argparse.BooleanOptionalAction, default=True,
                help="passively log MLX90640 telemetry; failure never blocks RGB navigation")
ap.add_argument("--thermal-fps", type=float, default=2.0,
                help="target passive thermal logging rate (default 2 Hz)")
ap.add_argument("--thermal-full-frame", action=argparse.BooleanOptionalAction, default=True,
                help="write all 32x24 thermal pixels to a separate CSV")
ap.add_argument("--thermal-refresh-hz", type=float, default=2.0,
                choices=(1.0,2.0,4.0,8.0,16.0,32.0,64.0),
                help="MLX90640 sensor refresh rate; logging can be slower")
ap.add_argument("--thermal-retries", type=int, default=2,
                help="frame-read retries before skipping a thermal sample")

# Diagnostics
ap.add_argument("--frame-print-every", type=int, default=10,
                help="print periodic frame timing every N analyzed frames; 0 disables")
ap.add_argument("--save-every-sec", type=float, default=5.0)
ap.add_argument("--max-saved-images", type=int, default=150)
ap.add_argument("--calibrate-forward", action="store_true")
ap.add_argument("--calibrate-seconds", type=float, default=8.0)
ap.add_argument("--calibrate-vision", action="store_true")
args = ap.parse_args()

OUT = Path("/tmp/voyager_v28")
OUT.mkdir(parents=True, exist_ok=True)

# GPIO mapping used in prior pool runs
L1 = DigitalOutputDevice(17, initial_value=False)
L2 = DigitalOutputDevice(27, initial_value=False)
R1 = DigitalOutputDevice(22, initial_value=False)
R2 = DigitalOutputDevice(23, initial_value=False)

def motors_off():
    L1.off(); L2.off(); R1.off(); R2.off()

# ---------- global run clock / event timing ----------
RUN_T0 = monotonic()
_event_lock = threading.Lock()
_last_event_t = RUN_T0

def event(msg):
    global _last_event_t
    now = monotonic()
    with _event_lock:
        elapsed = now - RUN_T0
        delta = now - _last_event_t
        _last_event_t = now
        print(f"[{elapsed:8.3f}s | event Δ {delta:6.3f}s] {msg}", flush=True)

# ---------- Motor thread ----------
class MotorController:
    def __init__(self):
        self._lock = threading.Lock()
        self._quit = threading.Event()
        self._mode = "OFF"
        self._epoch = 0
        self._changed_at = monotonic()
        self._thread = threading.Thread(target=self._run, name="motor", daemon=True)
        self._thread.start()

    def snapshot(self):
        with self._lock:
            return self._mode, self._epoch, self._changed_at

    def set_mode(self, mode, announce=True):
        if mode not in ("OFF","FORWARD","TURN_LEFT","TURN_RIGHT","REVERSE"):
            raise ValueError(mode)
        with self._lock:
            if mode == self._mode:
                return self._epoch
            self._mode = mode
            self._epoch += 1
            self._changed_at = monotonic()
            ep = self._epoch
        if announce:
            event(f"MOTOR epoch={ep} -> {mode}")
        return ep

    def _same(self, mode):
        with self._lock:
            return (not self._quit.is_set()) and self._mode == mode

    def _wait(self, sec, mode):
        end = monotonic() + sec
        while monotonic() < end:
            if not self._same(mode):
                return False
            sleep(min(0.005, max(0, end-monotonic())))
        return True

    def _run(self):
        while not self._quit.is_set():
            mode, _, _ = self.snapshot()
            if mode == "OFF":
                motors_off(); sleep(0.005)
            elif mode == "TURN_LEFT":
                # Physical RIGHT pump -> left rotation
                L1.off(); L2.off(); R1.on(); R2.off()
                sleep(0.005)
            elif mode == "TURN_RIGHT":
                # Physical LEFT pump -> right rotation
                R1.off(); R2.off(); L1.on(); L2.off()
                sleep(0.005)
            elif mode == "REVERSE":
                # Reverse polarity on both channels. This is deliberately used
                # only as a collision recovery maneuver.
                L1.off(); L2.on(); R1.off(); R2.on()
                sleep(0.005)
            elif mode == "FORWARD":
                motors_off(); L1.on(); L2.off()
                if not self._wait(args.forward_left_sec, "FORWARD"):
                    motors_off(); continue
                motors_off()
                if args.forward_gap_sec and not self._wait(args.forward_gap_sec, "FORWARD"):
                    continue
                R1.on(); R2.off()
                if not self._wait(args.forward_right_sec, "FORWARD"):
                    motors_off(); continue
                motors_off()
                if args.forward_gap_sec:
                    self._wait(args.forward_gap_sec, "FORWARD")
        motors_off()

    def close(self):
        self.set_mode("OFF", announce=False)
        self._quit.set()
        self._thread.join(timeout=1)
        motors_off()

motor = MotorController()

# ---------- Latest-frame slot ----------
class LatestFrame:
    def __init__(self):
        self.cond = threading.Condition()
        self.frame_id = 0
        self.captured_at = 0.0
        self.image = None
        self.closed = False

    def publish(self, frame_id, captured_at, image):
        # ONE slot only. Old image reference is replaced and becomes collectible.
        with self.cond:
            self.frame_id = frame_id
            self.captured_at = captured_at
            self.image = image
            self.cond.notify_all()

    def wait_newer_than(self, last_id, timeout=1.0):
        with self.cond:
            self.cond.wait_for(
                lambda: self.closed or self.frame_id > last_id,
                timeout=timeout
            )
            if self.closed:
                return None
            if self.frame_id <= last_id or self.image is None:
                return None
            # image reference is immutable for our purposes after publication
            return self.frame_id, self.captured_at, self.image

    def close(self):
        with self.cond:
            self.closed = True
            self.cond.notify_all()

latest = LatestFrame()

# ---------- Camera acquisition thread ----------
class CameraThread:
    def __init__(self):
        self.quit = threading.Event()
        self.proc = None
        self.thread = threading.Thread(target=self._run, name="camera", daemon=True)
        self.thread.start()

    def _run(self):
        cmd = [
            "rpicam-vid", "--codec","mjpeg",
            "--width",str(args.width), "--height",str(args.height),
            "--framerate",str(args.fps), "--timeout","0",
            "--nopreview", "-o","-"
        ]
        try:
            self.proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0
            )
            buf = bytearray()
            fid = 0
            while not self.quit.is_set():
                chunk = self.proc.stdout.read(8192)
                if not chunk:
                    break
                buf.extend(chunk)

                # Decode every complete JPEG currently available. Publishing is
                # latest-only, so control can never build a stale frame queue.
                while True:
                    soi = buf.find(b"\xff\xd8")
                    if soi < 0: break
                    eoi = buf.find(b"\xff\xd9", soi+2)
                    if eoi < 0: break
                    jpg = bytes(buf[soi:eoi+2])
                    del buf[:eoi+2]
                    arr = np.frombuffer(jpg, dtype=np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        img = cv2.rotate(img, cv2.ROTATE_180)
                        fid += 1
                        latest.publish(fid, monotonic(), img)

                # hard RAM bound on malformed stream
                if len(buf) > 4_000_000:
                    pos = buf.rfind(b"\xff\xd8")
                    if pos > 0: del buf[:pos]
                    else: buf.clear()
        finally:
            latest.close()
            if self.proc:
                try: self.proc.terminate()
                except Exception: pass

    def close(self):
        self.quit.set()
        if self.proc:
            try: self.proc.terminate()
            except Exception: pass
        self.thread.join(timeout=1)

camera = None

# ---------- Passive thermal telemetry (NO control authority) ----------
class ThermalLogger:
    """Independent MLX90640 logger.

    Navigation never waits on this thread and never reads thermal data to choose
    a motor action in v27.  Sensor/import/read failures are logged and RGB
    navigation continues unchanged.
    """
    def __init__(self, stamp):
        self.enabled=bool(args.thermal_log)
        self.quit=threading.Event()
        self.lock=threading.Lock()
        self.latest=None
        self.samples=0
        self.errors=0
        self.summary_path=OUT/f"thermal_summary_{stamp}.csv"
        self.frame_path=OUT/f"thermal_frames_{stamp}.csv"
        self.thread=None
        if self.enabled:
            self.thread=threading.Thread(target=self._run,name="thermal",daemon=True)
            self.thread.start()

    def snapshot(self):
        with self.lock:
            return None if self.latest is None else dict(self.latest)

    def _run(self):
        try:
            import board, busio, adafruit_mlx90640
        except Exception as e:
            event(f"THERMAL disabled: import/setup modules unavailable: {e}")
            self.enabled=False
            return
        try:
            i2c=busio.I2C(board.SCL,board.SDA)
            mlx=adafruit_mlx90640.MLX90640(i2c)
            refresh_map={
                1.0:adafruit_mlx90640.RefreshRate.REFRESH_1_HZ,
                2.0:adafruit_mlx90640.RefreshRate.REFRESH_2_HZ,
                4.0:adafruit_mlx90640.RefreshRate.REFRESH_4_HZ,
                8.0:adafruit_mlx90640.RefreshRate.REFRESH_8_HZ,
                16.0:adafruit_mlx90640.RefreshRate.REFRESH_16_HZ,
                32.0:adafruit_mlx90640.RefreshRate.REFRESH_32_HZ,
                64.0:adafruit_mlx90640.RefreshRate.REFRESH_64_HZ,
            }
            mlx.refresh_rate=refresh_map[float(args.thermal_refresh_hz)]
        except Exception as e:
            event(f"THERMAL disabled: MLX90640 init failed: {e}")
            self.enabled=False
            return

        period=1.0/max(0.1,float(args.thermal_fps))
        next_t=monotonic()
        pixel_header=[f"p{i:03d}" for i in range(768)]
        try:
            with open(self.summary_path,"w",newline="") as sf:
                sw=csv.writer(sf)
                sw.writerow(["elapsed","sample","min_c","max_c","mean_c","range_c",
                             "left_mean_c","center_mean_c","right_mean_c",
                             "hot_x","hot_y","cold_x","cold_y"])
                ff=None; fw=None
                try:
                    if args.thermal_full_frame:
                        ff=open(self.frame_path,"w",newline="")
                        fw=csv.writer(ff)
                        fw.writerow(["elapsed","sample"]+pixel_header)
                    frame=[0.0]*768
                    event(f"THERMAL logging started at {args.thermal_fps:.1f} Hz "
                          f"(sensor refresh {args.thermal_refresh_hz:.0f} Hz); control weight=ZERO")
                    while not self.quit.is_set():
                        now=monotonic()
                        if now<next_t:
                            self.quit.wait(min(0.05,next_t-now)); continue
                        next_t=max(next_t+period,now)
                        ok=False; last_err=None
                        for _ in range(max(1,args.thermal_retries+1)):
                            try:
                                mlx.getFrame(frame); ok=True; break
                            except Exception as e:
                                last_err=e
                                if self.quit.wait(0.02): break
                        if not ok:
                            self.errors+=1
                            if self.errors<=3 or self.errors%20==0:
                                event(f"THERMAL read skipped (errors={self.errors}): {last_err}")
                            continue
                        t=monotonic(); a=np.asarray(frame,dtype=np.float32).reshape(24,32)
                        # MLX90640 can occasionally expose sentinel/invalid pixels near -273.15 C.
                        # Exclude physically implausible samples from summary statistics/locations.
                        valid=np.isfinite(a) & (a > -100.0) & (a < 200.0)
                        if not valid.any():
                            self.errors+=1; continue
                        clean=np.where(valid,a,np.nan)
                        vals=clean[np.isfinite(clean)]
                        mn=float(np.min(vals)); mx=float(np.max(vals)); mean=float(np.mean(vals))
                        # 32 columns -> L 0:10, C 10:22, R 22:32
                        lm=float(np.nanmean(clean[:,0:10])); cm=float(np.nanmean(clean[:,10:22])); rm=float(np.nanmean(clean[:,22:32]))
                        hi=int(np.nanargmax(clean)); lo=int(np.nanargmin(clean))
                        hy,hx=divmod(hi,32); cy,cx=divmod(lo,32)
                        self.samples+=1
                        rec={"elapsed":t-RUN_T0,"sample":self.samples,"min":mn,"max":mx,
                             "mean":mean,"range":mx-mn,"L":lm,"C":cm,"R":rm,
                             "hot_x":hx,"hot_y":hy,"cold_x":cx,"cold_y":cy,"captured_at":t}
                        with self.lock: self.latest=rec
                        sw.writerow([rec["elapsed"],self.samples,mn,mx,mean,mx-mn,lm,cm,rm,hx,hy,cx,cy])
                        if fw is not None:
                            fw.writerow([rec["elapsed"],self.samples]+[(float(x) if np.isfinite(x) else "") for x in clean.ravel()])
                        if self.samples%10==0:
                            sf.flush()
                            if ff is not None: ff.flush()
                finally:
                    if ff is not None: ff.close()
        except Exception as e:
            self.errors+=1
            event(f"THERMAL logger stopped after error: {e}")
        finally:
            try:
                if 'i2c' in locals(): i2c.deinit()
            except Exception: pass

    def close(self):
        self.quit.set()
        if self.thread:
            self.thread.join(timeout=2.0)

# ---------- Vision ----------
def region_metrics(gray):
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blur, 60, 150)
    edge = float(np.count_nonzero(edges))/float(edges.size)
    inv = 255 - edges
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3)
    vals = dist[dist > 0]
    scale = float(np.percentile(vals,85)) if vals.size else 0.0
    return edge, scale

def analyze(img):
    h,w = img.shape[:2]
    band = cv2.cvtColor(img[int(h*.42):int(h*.86),:], cv2.COLOR_BGR2GRAY)
    third = band.shape[1]//3
    rois = {"L":band[:,:third], "C":band[:,third:2*third], "R":band[:,2*third:]}
    E,S = {},{}
    for k,r in rois.items():
        E[k],S[k] = region_metrics(r)
    return E,S

def valid_scale(x):
    return np.isfinite(x) and x < 1000.0

def choose_turn(E,S):
    lv,rv = valid_scale(S["L"]),valid_scale(S["R"])
    # Higher scale was empirically associated with farther/more-open views.
    if lv and rv:
        if S["L"] > S["R"]: return "LEFT"
        if S["R"] > S["L"]: return "RIGHT"
    if lv and not rv: return "LEFT"
    if rv and not lv: return "RIGHT"
    return args.default_turn.upper()

def is_open(E,S):
    return (
        valid_scale(S["C"])
        and S["C"] >= args.open_scale
        and E["C"] > args.open_edge_min
    )


# ---------- Visual yaw-rate diagnostic ----------
# Diagnostic only in v15: it DOES NOT alter propulsion.
class VisualYawEstimator:
    def __init__(self):
        self.prev_gray = None
        self.prev_t = None
        self.samples = []
        self.last_print = 0.0

    def reset(self):
        self.prev_gray = None
        self.prev_t = None
        self.samples.clear()

    def update(self, img, captured_at):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_t is None:
            self.prev_gray = gray
            self.prev_t = captured_at
            return None

        dt = captured_at - self.prev_t
        prev = self.prev_gray
        self.prev_gray = gray
        self.prev_t = captured_at
        if dt <= 0.02 or dt > 0.50:
            return None

        pts0 = cv2.goodFeaturesToTrack(
            prev, maxCorners=120, qualityLevel=0.01,
            minDistance=8, blockSize=7
        )
        if pts0 is None or len(pts0) < args.yaw_min_features:
            return None

        pts1, status, err = cv2.calcOpticalFlowPyrLK(
            prev, gray, pts0, None,
            winSize=(21,21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03)
        )
        if pts1 is None or status is None:
            return None

        good = status.reshape(-1) == 1
        p0 = pts0.reshape(-1,2)[good]
        p1 = pts1.reshape(-1,2)[good]
        if len(p0) < args.yaw_min_features:
            return None

        M, inliers = cv2.estimateAffinePartial2D(
            p0, p1, method=cv2.RANSAC,
            ransacReprojThreshold=2.5, maxIters=1000, confidence=0.99
        )
        if M is None:
            return None
        if inliers is not None and int(inliers.sum()) < args.yaw_min_features:
            return None

        # Image rotation. Camera is fixed to the boat, so scene rotation is
        # opposite boat yaw. Positive returned value = boat clockwise/right.
        scene_angle_deg = float(np.degrees(np.arctan2(M[1,0], M[0,0])))
        boat_yaw_rate = -scene_angle_deg / dt

        # Reject implausible one-frame fits.
        if not np.isfinite(boat_yaw_rate) or abs(boat_yaw_rate) > 120.0:
            return None

        self.samples.append(boat_yaw_rate)
        if len(self.samples) > max(3, args.yaw_window):
            self.samples.pop(0)

        smooth = float(np.median(self.samples))
        return boat_yaw_rate, smooth, len(p0)

yaw_est = VisualYawEstimator()

# ---------- bounded diagnostic JPEGs ----------
saved = []
last_save_t = 0.0

def save_diag(img, fid, label):
    global last_save_t
    if args.max_saved_images <= 0: return
    p = OUT / f"diag_{fid:07d}_{label}.jpg"
    cv2.imwrite(str(p), img)
    saved.append(p)
    while len(saved) > args.max_saved_images:
        q=saved.pop(0)
        try:q.unlink()
        except FileNotFoundError:pass
    last_save_t = monotonic()

# ---------- calibration ----------
def forward_calibration():
    event(f"FORWARD calibration L={args.forward_left_sec:.3f} R={args.forward_right_sec:.3f}")
    sleep(3)
    motor.set_mode("FORWARD")
    sleep(args.calibrate_seconds)
    motor.set_mode("OFF")
    event("calibration complete")

def reverse_test():
    event(f"REVERSE TEST starts in 3s; duration={args.reverse_test_sec:.2f}s")
    sleep(3)
    motor.set_mode("REVERSE")
    sleep(args.reverse_test_sec)
    motor.set_mode("OFF")
    event("REVERSE TEST complete; motors OFF")

def vision_calibration():
    global camera
    camera=CameraThread()
    motor.set_mode("OFF", announce=False)
    last=0; prev_t=None
    event("VISION calibration; motors OFF")
    try:
        while True:
            item=latest.wait_newer_than(last,1)
            if item is None: continue
            fid,cap,img=item; last=fid
            now=monotonic(); E,S=analyze(img)
            dt=0 if prev_t is None else cap-prev_t
            prev_t=cap
            age=now-cap
            print(f"[{now-RUN_T0:8.3f}s] frame={fid:06d} dt={dt:.3f}s "
                  f"age={age*1000:.0f}ms E={E['L']:.3f}/{E['C']:.3f}/{E['R']:.3f} "
                  f"S={S['L']:.1f}/{S['C']:.1f}/{S['R']:.1f}",flush=True)
    finally:
        camera.close()

# ---------- autonomous control ----------
def autonomous():
    """v29: vision proposes; recent-history supervisor decides.

    Design goals:
      * FORWARD is the dominant/default navigation action.
      * Pre-collision avoidance is temporal and decisive, not twitchy.
      * One avoidance decision -> one bounded turn -> evaluate outcome.
      * Recent repeated turns / L-R oscillation are penalized.
      * Hard collision never launches an endless turn. Motors go OFF and
        continuous vision watches for passive rebound/clearance first.
      * Automatic reverse remains disabled until physically validated.
    """
    global camera,last_save_t
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    csvpath=OUT/f"run_{stamp}.csv"
    camera=CameraThread()
    thermal=ThermalLogger(stamp)

    state="FORWARD"
    last_fid=0
    prev_cap=None
    analyzed=0
    collision_count=0
    imminent_count=0
    edge_hist=deque(maxlen=max(3,args.approach_edge_window))
    action_hist=deque()   # (time, action, duration-or-0)
    turn_events=deque()   # (time, dir, duration, reason)
    last_turn_dir=None
    last_turn_t=-1e9
    last_turn_reason=None
    last_good_forward_t=monotonic()
    forward_started=monotonic()
    turn_end=0.0
    turn_dir=None
    turn_reason=None
    collision_started=0.0
    recovery_coast_end=0.0
    clear_count=0
    stuck_hold_started=0.0
    stuck_clear_count=0
    escape_clear_count=0
    failed_escape_dir=None
    latest_E=None
    latest_S=None
    last_policy_print=0.0
    last_action_change=monotonic()
    last_mode="FORWARD"

    def set_action(mode, why=""):
        nonlocal last_action_change,last_mode
        now=monotonic()
        if mode != last_mode:
            action_hist.append((last_action_change,last_mode,max(0.0,now-last_action_change)))
            last_action_change=now
            last_mode=mode
        motor.set_mode(mode)
        if why:
            event(f"POLICY {mode}: {why}")

    def prune(now):
        cutoff=now-args.history_sec
        while action_hist and action_hist[0][0]+action_hist[0][2] < cutoff:
            action_hist.popleft()
        while turn_events and turn_events[0][0] < cutoff:
            turn_events.popleft()

    def history_stats(now):
        prune(now)
        totals={"FORWARD":0.0,"TURN_LEFT":0.0,"TURN_RIGHT":0.0,"OFF":0.0,"REVERSE":0.0}
        cutoff=now-args.history_sec
        for t,m,d in action_hist:
            a=max(t,cutoff); b=min(t+d,now)
            if b>a: totals[m]=totals.get(m,0.0)+(b-a)
        # include current action
        a=max(last_action_change,cutoff)
        if now>a: totals[last_mode]=totals.get(last_mode,0.0)+(now-a)
        total=max(1e-6,sum(totals.values()))
        turn_total=totals.get("TURN_LEFT",0)+totals.get("TURN_RIGHT",0)
        dirs=[x[1] for x in turn_events]
        alternations=sum(1 for a,b in zip(dirs,dirs[1:]) if a!=b)
        same_run=0
        if dirs:
            same_run=1
            for d in reversed(dirs[:-1]):
                if d==dirs[-1]: same_run+=1
                else: break
        return totals, totals.get("FORWARD",0)/total, turn_total, alternations, same_run

    def approach_signal():
        """Return (confirmed, risk, explanation) from recent center E_C trajectory.
        Exact distance is intentionally NOT inferred.  We detect a sustained
        worsening trajectory, plus a separate very-close signature.
        """
        vals=[x[1] for x in edge_hist if np.isfinite(x[1])]
        if len(vals)<max(3,args.approach_edge_window):
            return False,0.0,"warming"
        n=len(vals); h=max(1,n//2)
        early=float(np.median(vals[:h])); late=float(np.median(vals[-h:]))
        drop=early-late
        diffs=np.diff(vals)
        consistency=float(np.mean(diffs<0)) if len(diffs) else 0.0
        # 0..1 diagnostic risk, not a rigid classifier.
        r_drop=min(1.0,max(0.0,drop/max(args.approach_edge_drop,1e-6)))
        r_close=min(1.0,max(0.0,(args.approach_edge_late_max-late)/max(args.approach_edge_late_max,1e-6)+0.5))
        risk=0.55*r_drop+0.30*consistency+0.15*r_close
        confirmed=(late<=args.approach_edge_late_max and
                   drop>=args.approach_edge_drop and
                   consistency>=args.approach_consistency)
        return confirmed,risk,f"E {early:.3f}->{late:.3f} drop={drop:.3f} agree={consistency:.2f}"

    def proposed_turn(E,S):
        # Preserve visual side choice, but history can veto an unproductive repeat.
        return choose_turn(E,S)

    def loop_pressure(now):
        """Return (0..1 pressure, explanation).

        v25 deliberately defines a loop primarily as TIME WITHOUT PRODUCTIVE
        FORWARD PROGRESS.  R/R/R or L/R/L/R patterns only add supporting
        evidence; they do not declare a loop by themselves.
        """
        if not (args.history_supervisor and args.loop_guard):
            return 0.0, "loop_guard=off"
        totals,fwd_ratio,turn_total,alternations,same_run=history_stats(now)
        # Deliberate STUCK_HOLD observation is not failed navigation progress.
        # Otherwise OFF time feeds the loop detector and can create a self-locking diagnosis.
        no_progress=0.0 if state=="STUCK_HOLD" else max(0.0, now-last_good_forward_t)
        p_time=min(1.0, no_progress/max(args.loop_no_progress_sec,1e-6))
        if args.loop_forward_ratio_max <= 0:
            p_fwd=0.0
        else:
            p_fwd=min(1.0,max(0.0,(args.loop_forward_ratio_max-fwd_ratio)/args.loop_forward_ratio_max))
        # Time/progress dominate. Patterns only support the diagnosis.
        p_pattern=min(1.0, 0.18*max(0,same_run-1) + 0.12*alternations)
        pressure=min(1.0,0.60*p_time + 0.30*p_fwd + 0.10*p_pattern)
        return pressure,(f"noFWD={no_progress:.1f}s fwd={fwd_ratio:.2f} "
                         f"same={same_run} alt={alternations}")

    def turn_allowed(direction, urgency, now):
        """Soft constraints. Safety can override ordinary navigation preferences."""
        if not args.history_supervisor:
            return True,"history_supervisor=off"
        totals,fwd_ratio,turn_total,alternations,same_run=history_stats(now)
        since_turn=now-last_turn_t
        forward_since_turn=max(0.0, now-max(last_turn_t,forward_started)) if last_turn_t>-1e8 else now-forward_started
        lp,lpwhy=loop_pressure(now)
        reasons=[]
        score=1.0
        if args.turn_budget_guard and turn_total>=args.max_turn_sec_window:
            score-=0.9; reasons.append(f"turn_budget={turn_total:.1f}s")
        if args.repeat_turn_guard and direction==last_turn_dir and since_turn<args.post_turn_cooldown_sec:
            score-=0.8; reasons.append("recent_same_turn")
        if args.repeat_turn_guard and same_run>=2:
            score-=0.5; reasons.append(f"same_turn_run={same_run}")
        if args.oscillation_guard and alternations>=2:
            score-=0.7; reasons.append(f"LR_oscillation={alternations}")
        if args.forward_progress_guard and forward_since_turn<args.min_useful_forward_sec and last_turn_t>-1e8:
            score-=0.45; reasons.append(f"little_forward={forward_since_turn:.1f}s")
        # A loop means the current strategy has consumed time without buying useful FWD.
        # Do NOT force FWD blindly: instead strongly reject repeating the same strategy.
        if args.loop_guard and lp>=args.loop_pressure_threshold:
            reasons.append(f"LOOP_PRESSURE={lp:.2f}({lpwhy})")
            if direction==last_turn_dir:
                score-=1.0
            else:
                score-=0.20  # alternative strategy remains possible
        if urgency=="IMMINENT": score+=1.0
        if args.turn_budget_guard and turn_total>=args.max_turn_sec_window*1.5:
            return False,"hard turn-budget guard; " + ",".join(reasons)
        return score>0.0,("allow" if not reasons else ",".join(reasons))+f" score={score:.2f}"

    def begin_turn(direction, duration, reason):
        nonlocal state,turn_end,turn_dir,turn_reason,last_turn_dir,last_turn_t,last_turn_reason
        now=monotonic()
        state="TURN"
        turn_dir=direction
        turn_reason=reason
        turn_end=now+duration
        last_turn_dir=direction
        last_turn_t=now
        last_turn_reason=reason
        turn_events.append((now,direction,duration,reason))
        set_action("TURN_LEFT" if direction=="LEFT" else "TURN_RIGHT",
                   f"{reason}; one decisive {direction} for {duration:.2f}s")

    event("START v29 collision-escape supervisor; waiting for live frames")
    set_action("FORWARD","default navigation; vision remains live")

    with open(csvpath,"w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow(["elapsed","frame","frame_dt","frame_age","state","motor","epoch",
                     "E_L","E_C","E_R","S_L","S_C","S_R","risk","approach",
                     "forward_ratio","turn_sec_window","alternations","same_turn_run",
                     "loop_pressure","last_turn","decision",
                     "thermal_sample","thermal_age","thermal_min_c","thermal_max_c",
                     "thermal_mean_c","thermal_range_c","thermal_L_c","thermal_C_c","thermal_R_c"])
        try:
            while True:
                item=latest.wait_newer_than(last_fid,0.10)
                now=monotonic()

                # Motor/state timing never waits for a camera frame.
                if state=="TURN" and now>=turn_end:
                    completed_reason=turn_reason
                    if completed_reason in ("REBOUND_CLEAR","STUCK_ESCAPE") and args.recovery_coast:
                        set_action("OFF",f"rebound turn complete -> observe/coast {args.recovery_coast_sec:.2f}s")
                        state="RECOVERY_COAST"
                        recovery_coast_end=now+max(0.0,args.recovery_coast_sec)
                    else:
                        set_action("FORWARD",f"{completed_reason} turn complete -> resume navigation")
                        state="FORWARD"
                        forward_started=now
                    edge_hist.clear()
                    collision_count=0
                    imminent_count=0
                    turn_dir=None
                    turn_reason=None

                if state=="RECOVERY_COAST" and now>=recovery_coast_end:
                    # Do not chain recovery rotations blindly. Observe the result first.
                    # A fresh hard-contact frame sends us back to passive rebound; otherwise
                    # restore the dominant FORWARD navigation state immediately.
                    if latest_E is not None and latest_S is not None:
                        coast_hard=(latest_E["C"]<=args.collision_edge_max and
                                    np.isfinite(latest_S["C"]) and
                                    latest_S["C"]>=args.collision_scale_min)
                    else:
                        coast_hard=False
                    # v29: a non-saturated frame is NOT enough to declare escape.
                    # Stay OFF and let fresh-frame logic below prove geometric clearance.
                    if coast_hard and args.passive_rebound:
                        set_action("OFF","recovery coast still sees hard contact -> rebound watch")
                        state="COLLISION_COAST"
                        collision_started=now
                        clear_count=0
                        escape_clear_count=0
                    else:
                        set_action("OFF","recovery turn complete -> verify wall clearance before FORWARD")
                        state="STUCK_HOLD"
                        stuck_hold_started=now
                        stuck_clear_count=0
                        escape_clear_count=0

                if state=="COLLISION_COAST":
                    # Passive rebound first. No turn while the image remains hard-saturated.
                    if now-collision_started>=args.collision_coast_max_sec:
                        # v29: passive rebound gets a short chance; if still pinned, actively escape.
                        direction=proposed_turn(latest_E,latest_S) if (latest_E is not None and latest_S is not None) else args.default_turn.upper()
                        if failed_escape_dir==direction:
                            direction="LEFT" if direction=="RIGHT" else "RIGHT"
                        begin_turn(direction,args.collision_escape_turn_sec,"STUCK_ESCAPE")
                        failed_escape_dir=direction
                        clear_count=0
                        escape_clear_count=0
                        event(f"COLLISION_COAST timeout: active {direction} escape; FORWARD locked until clearance")

                if state=="STUCK_HOLD":
                    # Observation state: keep propulsion OFF, but fresh frames below can release it.
                    set_action("OFF")

                if item is None:
                    continue

                fid,cap,img=item
                skipped=max(0,fid-last_fid-1) if last_fid else 0
                last_fid=fid
                frame_dt=0.0 if prev_cap is None else cap-prev_cap
                prev_cap=cap
                age=monotonic()-cap
                if age>args.max_frame_age:
                    event(f"STALE frame={fid} age={age:.3f}s ignored")
                    continue

                E,S=analyze(img); latest_E=E; latest_S=S; analyzed+=1
                now=monotonic(); decision=""
                hard=(E["C"]<=args.collision_edge_max and valid_scale(S["C"]) is False and
                      np.isfinite(S["C"]) and S["C"]>=args.collision_scale_min)
                # Equivalent but explicit hard signature; valid_scale is false at saturation.
                very_close=(np.isfinite(E["C"]) and E["C"]<=args.imminent_edge_max and not hard)

                if hard: collision_count+=1
                else: collision_count=0
                if very_close: imminent_count+=1
                else: imminent_count=0

                if state=="FORWARD":
                    edge_hist.append((cap,float(E["C"]),float(S["C"])))
                else:
                    # Do not mix turning/rebound imagery into forward approach trajectory.
                    if state!="COLLISION_COAST": edge_hist.clear()

                approach,risk,trend=approach_signal() if (state=="FORWARD" and args.approach_trend) else (False,0.0,"approach_trend=off")
                imminent=(args.imminent_guard and imminent_count>=args.imminent_confirm_frames)

                # HARD CONTACT has highest priority.
                if state in ("FORWARD","TURN","RECOVERY_COAST") and collision_count>=args.collision_confirm_frames:
                    set_action("OFF","hard collision -> motors OFF; continuous vision stays live")
                    state="COLLISION_COAST" if args.passive_rebound else "STUCK_HOLD"
                    collision_started=now
                    if state=="STUCK_HOLD":
                        stuck_hold_started=now
                        stuck_clear_count=0
                    clear_count=0
                    edge_hist.clear()
                    decision="COLLISION -> OFF/REBOUND" if args.passive_rebound else "COLLISION -> OFF/HOLD (rebound disabled)"
                    save_diag(img,fid,"collision")

                elif state=="COLLISION_COAST":
                    # A usable non-saturated view means rebound/drift created some clearance.
                    if hard:
                        clear_count=0
                    else:
                        clear_count+=1
                    if clear_count>=args.open_confirm_frames:
                        direction=proposed_turn(E,S)
                        # If the last avoidance turn drove us into contact, prefer the other
                        # direction once clearance appears. Otherwise use current side evidence.
                        if (args.opposite_after_collision and
                            last_turn_reason in ("APPROACH","IMMINENT") and last_turn_dir):
                            direction="LEFT" if last_turn_dir=="RIGHT" else "RIGHT"
                        begin_turn(direction,args.collision_escape_turn_sec,"REBOUND_CLEAR")
                        failed_escape_dir=direction
                        clear_count=0
                        collision_count=0
                        decision=f"REBOUND CLEAR -> {direction}"
                        save_diag(img,fid,"rebound_clear")

                elif state=="STUCK_HOLD":
                    # v29: after hard contact, do not resume FORWARD merely because saturation
                    # disappeared.  Require a visibly open center for several consecutive frames.
                    clear_now=(not hard and np.isfinite(E["C"]) and
                               E["C"]>=args.escape_clear_edge_min and
                               valid_scale(S["C"]))
                    if clear_now:
                        escape_clear_count+=1
                    else:
                        escape_clear_count=0
                    if escape_clear_count>=max(1,args.escape_clear_frames):
                        set_action("FORWARD","collision escape verified: sustained center clearance -> resume navigation")
                        state="FORWARD"
                        forward_started=now
                        last_good_forward_t=now
                        edge_hist.clear()
                        collision_count=0
                        imminent_count=0
                        clear_count=0
                        stuck_clear_count=0
                        escape_clear_count=0
                        failed_escape_dir=None
                        decision="COLLISION ESCAPE VERIFIED -> FORWARD"
                        event("COLLISION escape verified; loop episode reset")
                    elif now-stuck_hold_started>=args.stuck_hold_max_sec:
                        direction=proposed_turn(E,S)
                        if failed_escape_dir==direction:
                            direction="LEFT" if direction=="RIGHT" else "RIGHT"
                        begin_turn(direction,args.collision_escape_turn_sec,"STUCK_ESCAPE")
                        failed_escape_dir=direction
                        stuck_clear_count=0
                        escape_clear_count=0
                        decision=f"STUCK_HOLD uncleared -> active {direction} escape; FORWARD remains locked"

                elif state=="FORWARD":
                    # Vision proposes avoidance; supervisor validates it against recent history.
                    urgency=None
                    if imminent: urgency="IMMINENT"
                    elif approach: urgency="APPROACH"

                    if urgency:
                        direction=proposed_turn(E,S)
                        allowed,why=turn_allowed(direction,urgency,now)
                        if allowed:
                            begin_turn(direction,args.avoid_turn_sec,urgency)
                            decision=f"{urgency} -> {direction}; {why}"
                            save_diag(img,fid,urgency.lower())
                        else:
                            # If strong evidence says forward is unsafe but history says another
                            # turn would be pathological, OFF is better than another 360 or impact.
                            if urgency=="IMMINENT":
                                set_action("OFF",f"IMMINENT but turn rejected ({why}) -> brief safety hold")
                                state="SAFETY_HOLD"
                                turn_end=now+0.50
                                decision=f"IMMINENT -> OFF; {why}"
                            else:
                                decision=f"APPROACH proposal rejected; keep FORWARD; {why}"
                                event(decision)

                    # Forward-starvation is a bias, not a rigid rule. Since we are already
                    # FORWARD it simply documents that progress is being restored.
                    totals,fwd_ratio,turn_total,alternations,same_run=history_stats(now)
                    if now-last_good_forward_t>=args.forward_starve_sec and not urgency:
                        last_good_forward_t=now
                        event(f"FORWARD BIAS satisfied: continuing safe forward; ratio={fwd_ratio:.2f}")
                    if now-forward_started>=args.min_useful_forward_sec:
                        last_good_forward_t=now

                elif state=="SAFETY_HOLD":
                    # Half-second OFF breaks pathological turn sequences without committing
                    # to reverse. Then use fresh vision: collision stays OFF; otherwise FORWARD.
                    if now>=turn_end:
                        if hard:
                            state="COLLISION_COAST"; collision_started=now; clear_count=0
                            decision="SAFETY_HOLD -> COLLISION_COAST"
                        else:
                            set_action("FORWARD","safety hold complete; restore forward bias")
                            state="FORWARD"; forward_started=now; edge_hist.clear()
                            decision="SAFETY_HOLD -> FORWARD"

                # Yaw remains diagnostic only.
                yaw_raw=yaw_smooth=None
                if state=="FORWARD" and motor.snapshot()[0]=="FORWARD":
                    y=yaw_est.update(img,cap)
                    if y is not None:
                        yaw_raw,yaw_smooth,yaw_features=y
                else:
                    yaw_est.reset()

                totals,fwd_ratio,turn_total,alternations,same_run=history_stats(now)
                lp,lpwhy=loop_pressure(now)
                if now-last_policy_print>=1.0:
                    mode,ep,_=motor.snapshot()
                    fps=(1.0/frame_dt) if frame_dt>0 else 0.0
                    print(f"[{now-RUN_T0:8.3f}s] NAV frame={fid:06d} ~{fps:4.1f}fps "
                          f"state={state} motor={mode} E_C={E['C']:.3f} S_C={S['C']:.1f} "
                          f"risk={risk:.2f} FWD={fwd_ratio*100:4.0f}% TURN={turn_total:.1f}s "
                          f"loop={lp:.2f} alt={alternations} same={same_run} | {trend}",flush=True)
                    last_policy_print=now

                if args.save_every_sec>0 and now-last_save_t>=args.save_every_sec:
                    save_diag(img,fid,"periodic")

                mode,ep,_=motor.snapshot()
                wr.writerow([now-RUN_T0,fid,frame_dt,age,state,mode,ep,
                             E["L"],E["C"],E["R"],S["L"],S["C"],S["R"],
                             risk,int(approach or imminent),fwd_ratio,turn_total,
                             alternations,same_run,lp,last_turn_dir or "",decision,
                             *((lambda th: (
                                 ["","", "", "", "", "", "", "", ""] if th is None else
                                 [th["sample"],max(0.0,now-th["captured_at"]),th["min"],th["max"],
                                  th["mean"],th["range"],th["L"],th["C"],th["R"]]
                             ))(thermal.snapshot()))])
                if analyzed%10==0: f.flush()
        finally:
            motor.set_mode("OFF")
            motors_off()
            thermal.close()
            camera.close()

def shutdown(*_):
    try:motor.close()
    except Exception:motors_off()
    if camera:
        try:camera.close()
        except Exception:pass
    motors_off()
    print("\n*** CTRL-C: MOTORS OFF ***",flush=True)
    raise SystemExit

signal.signal(signal.SIGINT,shutdown)
signal.signal(signal.SIGTERM,shutdown)

if __name__=="__main__":
    try:
        if args.reverse_test:
            reverse_test()
        elif args.calibrate_forward:
            forward_calibration()
        elif args.calibrate_vision:
            vision_calibration()
        else:
            print("================================================")
            print(" OCEAN VOYAGER v28 - HISTORY-AWARE VISUAL SUPERVISOR")
            print("================================================")
            print(f"camera target      : {args.fps} fps; latest-frame only")
            print(f"thermal telemetry  : {args.thermal_log} at {args.thermal_fps:.1f} Hz "
                  f"(full_frame={args.thermal_full_frame}); NAVIGATION WEIGHT=0")
            print(f"forward L/R        : {args.forward_left_sec:.2f}s / {args.forward_right_sec:.2f}s")
            print(f"history window     : {args.history_sec:.1f}s")
            print(f"loop guard         : {args.loop_guard} (no-progress {args.loop_no_progress_sec:.1f}s, "
                  f"FWD ratio <= {args.loop_forward_ratio_max:.2f}, pressure >= {args.loop_pressure_threshold:.2f})")
            print(f"feature switches   : history={args.history_supervisor} turn_budget={args.turn_budget_guard} "
                  f"repeat={args.repeat_turn_guard} oscillation={args.oscillation_guard} "
                  f"forward_progress={args.forward_progress_guard}")
            print(f"vision switches    : approach={args.approach_trend} imminent={args.imminent_guard} "
                  f"rebound={args.passive_rebound} opposite_after_collision={args.opposite_after_collision}")
            print(f"approach E trend   : {args.approach_edge_window} frames, "
                  f"drop>={args.approach_edge_drop:.3f}, late<={args.approach_edge_late_max:.3f}, "
                  f"agree>={args.approach_consistency:.2f}")
            print(f"imminent E         : <= {args.imminent_edge_max:.3f} for "
                  f"{args.imminent_confirm_frames} frames")
            print(f"avoid maneuver     : ONE {args.avoid_turn_sec:.2f}s decisive turn -> FORWARD")
            print(f"turn guard         : soft budget {args.max_turn_sec_window:.2f}s / "
                  f"{args.history_sec:.1f}s + repeat/oscillation penalties")
            print(f"collision signature: {args.collision_confirm_frames} frames, "
                  f"E_C<={args.collision_edge_max:.3f}, S_C>={args.collision_scale_min:.0f}")
            print(f"collision recovery : MOTORS OFF + passive rebound watch up to "
                  f"{args.collision_coast_max_sec:.1f}s")
            print(f"rebound maneuver   : ONE {args.rebound_turn_sec:.2f}s turn when usable view returns")
            print(f"stuck hold recovery: {args.stuck_hold_release_frames} clear frames -> FORWARD; "
                  f"max {args.stuck_hold_max_sec:.1f}s -> one bounded escape turn")
            print(f"post-rebound turn : {'OFF '+format(args.recovery_coast_sec,'.2f')+'s observation' if args.recovery_coast else 'disabled'}")
            print("automatic reverse  : DISABLED (reverse-test remains available)")
            print("yaw correction     : diagnostic only; not allowed to disturb forward travel")
            print("*** CTRL-C = MOTORS OFF ***")
            print("Starting in 5 seconds...")
            sleep(5)
            autonomous()
    finally:
        try:motor.close()
        except Exception:motors_off()
        motors_off()
