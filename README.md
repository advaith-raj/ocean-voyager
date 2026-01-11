# Controller (All-in-One)

This repository provides a single-file controller program for a Raspberry Pi–based robot using:

- **MLX90640** thermal camera over **I2C**
- **PCA9685** 16‑channel PWM driver over **I2C** to command:
  - two **ESC-driven thrusters**
  - one **pan servo**
- Optional **dry-run mode** (no hardware required)

The program includes:
- Hardware Abstraction Layer (HAL) for **Pi hardware** and **dry-run simulation**
- Explicit finite-state machine (FSM): **SCAN → COARSE_ALIGN → FINE_ALIGN → APPROACH → VERIFY → DONE/ABORT**
- Concurrency threads:
  - Thermal capture thread (latest-frame buffer)
  - Verification worker thread (async verify requests)
  - Watchdog thread (failsafe stop if controller stalls)
- Logging + recording:
  - `events.jsonl` (structured event stream)
  - optional thermal frame dumps (`.npz`) for debugging/analysis

---

## Files

- `controller_all_in_one.py` — single-file controller implementation.

---

## Quick start

### 1) Install dependencies (Raspberry Pi)

Enable I2C:
- `raspi-config` → **Interface Options** → **I2C** → Enable

Install Python packages:
```bash
pip3 install numpy adafruit-blinka adafruit-circuitpython-pca9685 adafruit-circuitpython-mlx90640
```

### 2) Generate a config template

```bash
python3 controller_all_in_one.py --write-config-template controller_config.json
```

### 3) Edit the config

Open `controller_config.json` and replace **TBD** values. Minimum required fields:
- `hal` → `pi_pca9685` or `dryrun`
- `pca_address` → usually `0x40` (leave default unless yours is different)
- `left_thruster_ch` → PCA9685 channel number (0–15)
- `right_thruster_ch` → PCA9685 channel number (0–15)
- `servo_ch` → PCA9685 channel number (0–15)

Notes:
- PCA9685 channels are printed on the board **0–15**.
- ESC pulse defaults (`esc_min_us`, `esc_neutral_us`, `esc_max_us`) are common starting points; adjust if needed.

### 4) Run

#### Dry-run (no hardware)
```bash
python3 controller_all_in_one.py --config controller_config.json --hal dryrun
```

#### Raspberry Pi + PCA9685 + MLX90640
```bash
python3 controller_all_in_one.py --config controller_config.json --hal pi_pca9685
```

---

## Output (logs/recording)

Each run creates a folder:
```
runs/<timestamp>/
  events.jsonl
  thermal_*.npz   (optional, if enabled)
```

- `events.jsonl` contains one JSON object per line (state transitions, target acquisition, verify votes, watchdog trips, shutdown, etc.).
- Thermal `.npz` dumps are enabled by:
  - `save_npz: true`
  - `npz_every_n: <N>`

---

## Hardware notes (PCA9685)

- **VCC** (logic) should be **3.3V** on Raspberry Pi (so I2C pullups are 3.3V).
- **SDA/SCL** connect to Pi I2C pins (GPIO2/GPIO3).
- **V+** powers the servo/ESC signal headers (use an appropriate external supply/BEC).
- Grounds must be common between **Pi**, **PCA9685**, and **motor/servo power**.

---

## Customizing verification (AI camera)

`Verifier.verify()` is currently a stub that always returns `(False, 0.0)` in the Pi hardware mode.
To enable camera/model verification:
- Implement a verifier class that captures an image and runs your model.
- Return `(is_target: bool, confidence: float)`.

The controller performs multiple verification “votes” (`verify_votes`) and accepts when
`verify_accept_count` positives are reached.

---

## Safety

- The controller always attempts a safe shutdown: **thrusters neutral** + **servo parked**.
- A watchdog thread will stop thrusters if the controller heartbeat stalls longer than `watchdog_timeout_s`.


## Self-test (bring-up)

Run a basic bring-up checklist (thermal read, servo sweep, thrusters neutral):
```bash
python3 controller_all_in_one.py --config controller_config.json --self-test
```

Optional: enable small motion pulses (use extreme caution, keep props clear):
- Set `self_test_motion: true` in the config.
- Adjust `self_test_forward_cmd`, `self_test_yaw_cmd`, and `self_test_pulse_s` as needed.


---

## Pool mode

If you are testing in a swimming pool **without GPS/IMU/encoders/obstacle sensors**, you cannot truly avoid walls in software.
What you *can* do is reduce risk by preventing long straight runs and adding frequent stops.

Enable pool mode in the config:
- `pool_mode: true`

Pool mode changes behavior mainly in **APPROACH**:
- Limits forward thrust (`pool_forward_cmd_limit`)
- Limits yaw command (`pool_max_yaw_cmd`)
- Uses **forward bursts** (`pool_forward_burst_s`) followed by **pauses** (`pool_pause_s`)
- If the robot has been moving continuously too long, it stops (and can optionally reverse briefly)

### Recommended pool safety setup (strongly recommended)
- Use a **tether line** (prevents runaway).
- Add **soft bumpers** (pool noodles/foam around the hull).
- Keep thrust low (`forward_cmd` ~ 0.15–0.25).
- Use `--self-test` first.

### Why this doesn’t “guarantee” wall avoidance
Without a sensor that can detect the wall (camera/sonar/bump switch) and without a heading/position estimate,
the robot cannot know it is approaching the pool boundary. Pool mode only reduces the likelihood of impact by:
- limiting continuous motion
- inserting frequent stop/re-align opportunities


### Pool search pattern

When `pool_mode: true`, the controller starts in a **SEARCH_PATTERN** state before scanning/alignment.
This state runs an open-loop motion pattern intended to reduce wall impacts in a pool:

- Rotate in place for `pool_search_rotate_s` seconds (small yaw command)
- Pause for `pool_search_pause_s` seconds (stop and observe)
- Rotate the other direction, pause
- Every `pool_search_cycles_per_forward` rotate cycles, do a short forward step (`pool_search_forward_s`)
- At any time, if a stable target is detected, it immediately transitions to **COARSE_ALIGN**

Config knobs:
- `pool_search_timeout_s`
- `pool_search_rotate_cmd`, `pool_search_rotate_s`
- `pool_search_pause_s`
- `pool_search_forward_cmd`, `pool_search_forward_s`
- `pool_search_cycles_per_forward`

This still does not “sense” walls; it only limits continuous motion and inserts frequent stop windows.


---

## Wall guard (pool boundary safety)

Since your boat has **front / left / right** visual cameras, you can enable a vision-based “wall guard” reflex for pool tests.

When enabled:
- The controller reads frames from the 3 cameras.
- It computes a “closeness” score per camera.
- If the score exceeds a threshold **during forward motion phases** (APPROACH forward bursts and SEARCH_PATTERN forward steps), it enters **WALL_AVOID**:
  - stop → short reverse + yaw away from the closest side
  - then resumes `SEARCH_PATTERN` (pool mode) or `SCAN`

### Enable (USB cameras example)

```json
"camera_enabled": true,
"camera_front_device": 0,
"camera_left_device": 1,
"camera_right_device": 2,
"wall_guard_enabled": true,
"wall_guard_method": "flow",
"wall_guard_threshold": 1.8
```

### Method options
- `wall_guard_method: "flow"` (recommended): optical flow magnitude
- `wall_guard_method: "edges"`: edge density fallback

### Wall guard debug overlay (optional)

To save annotated frames for tuning:
```json
"wall_guard_debug_overlay": true,
"wall_guard_debug_trigger_only": true,
"wall_guard_debug_max_hz": 2.0
```

Frames are written under your run directory:
- `runs/<run-id>/wallguard_<ms>_<camera>.jpg`

### Dependency
OpenCV is required for camera + wall guard:
```bash
pip3 install opencv-python
```
On Raspberry Pi, you may prefer `python3-opencv` (apt) or `opencv-python-headless` (pip).

---

## Pool workflow (pictorial)

![Pool workflow diagram](pool_workflow_diagram_v3.png)

Download the diagram: `pool_workflow_diagram_v3.png`

### Pool workflow in plain English

This is what the controller does in a swimming pool when `pool_mode: true` and the wall-guard is enabled:

1) **SEARCH_PATTERN (safe reacquire)**
   - The boat *does not* drive continuously.
   - It alternates **rotate → pause**, and every few cycles it performs a **short forward step**.
   - Goal: re-acquire a target while keeping motion bounded in a confined pool.

2) **SCAN (thermal/primary detection)**
   - The sensor scan runs while the boat is rotating/sweeping.
   - It looks for a **stable target candidate** (e.g., a hot/cold blob above threshold for multiple frames).

3) **ALIGN (coarse → fine)**
   - The boat yaws to **center the target**.
   - Coarse alignment makes larger corrections; fine alignment uses smaller corrections and a deadband to avoid oscillation.

4) **APPROACH (pulsed forward motion)**
   - In pool mode, approach is **forward burst → pause** with a conservative thrust limit.
   - The pauses create frequent “checkpoints” to re-evaluate the target and keep the boat from building up speed.

5) **VERIFY (camera-in-the-loop confirmation)**
   - When the boat is close enough (target appears large), it captures images from the **front / left / right cameras**.
   - A classifier/heuristic verifies whether it is **debris** (or a false detection).
   - If rejected, the controller returns to safe search (SEARCH_PATTERN / SCAN).

6) **WALL GUARD (always-on safety overlay during forward phases)**
   - Independently of the main state machine, the system reads front/left/right camera frames during *forward* motion phases.
   - It computes a “wall closeness” score (default: optical flow magnitude).
   - If triggered, it enters **WALL_AVOID**: **stop → reverse + yaw away → cooldown**, then resumes SEARCH_PATTERN/SCAN.

**Key idea:** In a pool you don’t have true navigation sensors, so safety comes from (a) keeping forward motion short and reversible, and (b) using the three cameras to detect “getting too close” and back away.

