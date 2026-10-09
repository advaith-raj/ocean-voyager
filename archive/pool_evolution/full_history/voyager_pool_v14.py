from gpiozero import DigitalOutputDevice
from time import sleep, monotonic
from pathlib import Path
from datetime import datetime
import subprocess, threading, signal, sys, argparse, csv
import cv2
import numpy as np

# ============================================================
# OCEAN VOYAGER v14
# Latest-frame asynchronous visual control
#
# Threads:
#   1) CAMERA: continuously decodes MJPEG and overwrites ONE latest-frame slot.
#   2) CONTROL: analyzes only fresh/new frames; never consumes a backlog.
#   3) MOTOR: executes the requested propulsion state independently.
#
# Key behavior:
#   - More looking, less turning.
#   - Turn pulse = short intervention; then motor OFF while vision observes.
#   - Chosen turn direction stays latched until OPEN is found.
#   - No automatic LEFT/RIGHT toggling.
#   - No low-edge "emergency" by itself (open water can also have low edges).
#   - BUT persistent E_C~0 + saturated S_C is a confirmed collision signature:
#     3 fresh frames trigger escape (~0.2-0.3 s at 10 fps).
#   - Forward->turn requires persistent low scale + worsening trend.
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
ap.add_argument("--open-confirm-frames", type=int, default=2)
ap.add_argument("--forward-commit-sec", type=float, default=2.0)
ap.add_argument("--collision-edge-max", type=float, default=0.002,
                help="collision saturation: center edge <= this")
ap.add_argument("--collision-scale-min", type=float, default=1000.0,
                help="collision saturation: center scale >= this")
ap.add_argument("--collision-confirm-frames", type=int, default=3,
                help="consecutive fresh saturated frames required (default 3)")

# Less movement / more vision
ap.add_argument("--turn-pulse-sec", type=float, default=0.25)
ap.add_argument("--turn-scan-sec", type=float, default=1.00)
ap.add_argument("--default-turn", choices=("left","right"), default="right")

# Diagnostics
ap.add_argument("--frame-print-every", type=int, default=10,
                help="print periodic frame timing every N analyzed frames; 0 disables")
ap.add_argument("--save-every-sec", type=float, default=5.0)
ap.add_argument("--max-saved-images", type=int, default=150)
ap.add_argument("--calibrate-forward", action="store_true")
ap.add_argument("--calibrate-seconds", type=float, default=8.0)
ap.add_argument("--calibrate-vision", action="store_true")
args = ap.parse_args()

OUT = Path("/tmp/voyager_v14")
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
        if mode not in ("OFF","FORWARD","TURN_LEFT","TURN_RIGHT"):
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
    global camera,last_save_t
    stamp=datetime.now().strftime("%Y%m%d_%H%M%S")
    csvpath=OUT/f"run_{stamp}.csv"
    camera=CameraThread()

    state="FORWARD"
    turn_dir=None
    last_fid=0
    prev_cap=None
    analyzed=0
    scale_hist=[]
    open_count=0
    collision_count=0
    forward_commit_until=monotonic()+args.forward_commit_sec
    turn_pulse_end=0.0
    next_turn_pulse=0.0

    event("START autonomous; waiting for live frames")
    motor.set_mode("FORWARD")

    with open(csvpath,"w",newline="") as f:
        wr=csv.writer(f)
        wr.writerow(["elapsed","frame","frame_dt","frame_age","state","motor","epoch",
                     "E_L","E_C","E_R","S_L","S_C","S_R","open_count","event"])
        try:
            while True:
                item=latest.wait_newer_than(last_fid,0.20)
                now=monotonic()

                # Timed motor transition is independent of camera arrival.
                if state=="TURN" and motor.snapshot()[0] in ("TURN_LEFT","TURN_RIGHT"):
                    if now >= turn_pulse_end:
                        motor.set_mode("OFF")
                        event(f"{turn_dir} pulse ended -> COAST_SCAN")

                # If observation window expires without OPEN, make another tiny
                # pulse in the SAME latched direction.
                if state=="TURN" and now >= next_turn_pulse:
                    if motor.snapshot()[0]=="OFF":
                        m="TURN_LEFT" if turn_dir=="LEFT" else "TURN_RIGHT"
                        motor.set_mode(m)
                        turn_pulse_end=now+args.turn_pulse_sec
                        next_turn_pulse=turn_pulse_end+args.turn_scan_sec
                        event(f"not OPEN -> short {turn_dir} pulse {args.turn_pulse_sec:.2f}s")

                if item is None:
                    continue

                fid,cap,img=item
                # latest-only: if camera outran control, skipped IDs are GOOD.
                skipped=max(0,fid-last_fid-1) if last_fid else 0
                last_fid=fid
                frame_dt=0.0 if prev_cap is None else cap-prev_cap
                prev_cap=cap
                age=monotonic()-cap

                # Never steer from stale visual data.
                if age > args.max_frame_age:
                    event(f"STALE frame={fid} age={age:.3f}s ignored")
                    continue

                E,S=analyze(img)
                analyzed+=1
                decision=""

                if args.frame_print_every and analyzed%args.frame_print_every==0:
                    fps=(1.0/frame_dt) if frame_dt>0 else 0
                    mode,ep,_=motor.snapshot()
                    print(f"[{monotonic()-RUN_T0:8.3f}s] FRAME {fid:06d} "
                          f"dt={frame_dt:.3f}s (~{fps:.1f}fps) age={age*1000:.0f}ms "
                          f"skip={skipped} state={state} motor={mode} ep={ep} "
                          f"E_C={E['C']:.3f} S_C={S['C']:.1f}",flush=True)

                if state=="FORWARD":
                    # Narrow collision signature learned from the pool:
                    # actual stuck collision repeatedly produced E_C=0.000 and
                    # S_C=65534.6. This is deliberately MUCH narrower than the
                    # old "low edge = emergency" rule, because open water can
                    # also have low edge density.
                    saturated_collision = (
                        E["C"] <= args.collision_edge_max
                        and np.isfinite(S["C"])
                        and S["C"] >= args.collision_scale_min
                    )

                    if saturated_collision:
                        collision_count += 1
                        if collision_count == 1:
                            event(
                                f"COLLISION candidate 1/{args.collision_confirm_frames} "
                                f"frame={fid} E_C={E['C']:.4f} S_C={S['C']:.1f}"
                            )
                    else:
                        collision_count = 0

                    if collision_count >= args.collision_confirm_frames:
                        turn_dir=choose_turn(E,S)
                        state="TURN"; open_count=0; scale_hist.clear()
                        collision_count=0
                        m="TURN_LEFT" if turn_dir=="LEFT" else "TURN_RIGHT"
                        motor.set_mode(m)
                        t=monotonic()
                        turn_pulse_end=t+args.turn_pulse_sec
                        next_turn_pulse=turn_pulse_end+args.turn_scan_sec
                        decision=f"COLLISION -> {turn_dir}"
                        event(
                            f"COLLISION confirmed E_C={E['C']:.4f} "
                            f"S_C={S['C']:.1f} -> {turn_dir} pulse "
                            f"{args.turn_pulse_sec:.2f}s"
                        )
                        save_diag(img,fid,"collision")

                    # Only run ordinary approach detection if collision handling
                    # did not already transition us into TURN.
                    elif valid_scale(S["C"]):
                        scale_hist.append(S["C"])
                        if len(scale_hist)>args.approach_frames:
                            scale_hist.pop(0)
                    else:
                        scale_hist.clear()

                    if (
                        state=="FORWARD"
                        and monotonic() >= forward_commit_until
                        and len(scale_hist)==args.approach_frames
                    ):
                        early=float(np.median(scale_hist[:len(scale_hist)//2]))
                        late=float(np.median(scale_hist[len(scale_hist)//2:]))
                        all_low=sum(x<=args.close_scale for x in scale_hist) >= int(.75*len(scale_hist))
                        worsening=(early-late)>=args.approach_drop
                        if all_low and worsening:
                            turn_dir=choose_turn(E,S)
                            state="TURN"; open_count=0; scale_hist.clear()
                            m="TURN_LEFT" if turn_dir=="LEFT" else "TURN_RIGHT"
                            motor.set_mode(m)
                            t=monotonic()
                            turn_pulse_end=t+args.turn_pulse_sec
                            next_turn_pulse=turn_pulse_end+args.turn_scan_sec
                            decision=f"APPROACH -> {turn_dir}"
                            event(f"APPROACH early={early:.1f} late={late:.1f} -> "
                                  f"{turn_dir} pulse {args.turn_pulse_sec:.2f}s")
                            save_diag(img,fid,"approach")

                else: # TURN
                    # Only frames captured after the current navigation episode
                    # are used; latest-only slot prevents backlog.
                    if is_open(E,S):
                        open_count+=1
                        if open_count==1:
                            event(f"OPEN candidate 1/{args.open_confirm_frames} "
                                  f"frame={fid} S_C={S['C']:.1f}")
                    else:
                        open_count=0

                    if open_count>=args.open_confirm_frames:
                        motor.set_mode("FORWARD")
                        state="FORWARD"; turn_dir=None; scale_hist.clear(); open_count=0
                        collision_count=0
                        forward_commit_until=monotonic()+args.forward_commit_sec
                        decision="OPEN -> FORWARD"
                        event(f"OPEN confirmed -> FORWARD; commit {args.forward_commit_sec:.2f}s")
                        save_diag(img,fid,"open")

                if args.save_every_sec>0 and monotonic()-last_save_t>=args.save_every_sec:
                    save_diag(img,fid,"periodic")

                mode,ep,_=motor.snapshot()
                wr.writerow([monotonic()-RUN_T0,fid,frame_dt,age,state,mode,ep,
                             E["L"],E["C"],E["R"],S["L"],S["C"],S["R"],
                             open_count,decision])
                if analyzed%10==0:f.flush()
        finally:
            motor.set_mode("OFF")
            motors_off()
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
        if args.calibrate_forward:
            forward_calibration()
        elif args.calibrate_vision:
            vision_calibration()
        else:
            print("================================================")
            print(" OCEAN VOYAGER v14 - ASYNC LATEST-FRAME CONTROL")
            print("================================================")
            print(f"camera target     : {args.fps} fps")
            print(f"turn pulse/scan   : {args.turn_pulse_sec:.2f}s / {args.turn_scan_sec:.2f}s")
            print(f"approach window   : {args.approach_frames} fresh frames")
            print(f"collision confirm : {args.collision_confirm_frames} frames "
                  f"(E_C<={args.collision_edge_max:.3f}, S_C>={args.collision_scale_min:.0f})")
            print(f"forward L/R       : {args.forward_left_sec:.2f}s / {args.forward_right_sec:.2f}s")
            print("camera buffer     : ONE latest frame only")
            print("timestamps        : monotonic frame + motor epochs + event deltas")
            print("*** CTRL-C = MOTORS OFF ***")
            print("Starting in 5 seconds...")
            sleep(5)
            autonomous()
    finally:
        try:motor.close()
        except Exception:motors_off()
        motors_off()
