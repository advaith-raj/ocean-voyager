# Ocean Voyager Simulation + Hardware-in-the-Loop Platform

**Ocean Voyager** is an autonomous surface vehicle (ASV) software platform for detecting floating plastic waste before it fragments into microplastics. The robot uses a coverage/navigation loop, a thermal sensor to identify areas of interest, RGB cameras with a YOLO verifier to confirm waste, and differential-thrust motion control to patrol a swimming-pool-sized test environment.

This repository is organized so the same high-level autonomy logic can run in simulation, stationary HIL, and later on Raspberry Pi hardware by swapping hardware adapters.

Current package: **Build 0.5.9**  
Focus of this build: **staged Raspberry Pi hardware validation: mocked scenario sweep, thermal-only bring-up, camera-only bring-up, and stationary real-sensor HIL.**

---

## What this repository contains

This repo contains a runnable simulation/HIL codebase for Ocean Voyager:

```text
main.py                         Main runner for config + scenario execution
config/                         Runtime configs: full sim, YOLO, HIL, hardware-style configs
config/hardware_profiles/       MLX90640, camera array, PCA9685/T200-style profiles
scenarios/                      Pool layouts, object coordinates, mission goals
models/best.pt                  YOLO model used as the waste verifier
ocean_voyager/                  Main Python package
  adapters/                     Simulated and real-hardware-facing adapters
  core/                         Mission controller, startup manager, config loader, logging, types
  device_bus/                   File-based command/response bus layer
  interfaces/                   Abstract interfaces for sensors/actuators/models
  perception/                   Thermal and vision-model perception logic
  planning/                     Coverage planner, coverage grid, approach, wall avoidance
  simulation/                   World model, robot dynamics, rendering/replay
scripts/                        Utility scripts, model-eval tools, hardware bring-up and staged HIL tools
tests/                          Unit tests
docs/                           Design documentation
outputs_review_build_0_5_6/     Example annotated outputs from Build 0.5.6
```

---

## High-level mission logic

The autonomy loop is designed around a realistic pool test mission:

```text
START at home = (0,0)
↓
COVERAGE_SWEEP using lawnmower/serpentine path
↓
periodic THERMAL_SCAN
↓
if no thermal signature:
    continue coverage
↓
if thermal signature found:
    estimate bearing only, not distance
    enter APPROACH_TARGET
↓
APPROACH_TARGET:
    move one short step toward thermal bearing
    take RGB images
    run YOLO
    rescan thermal
    repeat until visual confirms/rejects/lost
↓
VISION_VERIFY:
    confirmed bottle/plastic → store estimated waste coordinate
    seaweed/glare/other → reject and mark processed
↓
RETURN_TO_COVERAGE:
    resume lawnmower patrol
↓
TIME_BOXED_PATROL or FULL_SURVEY completion
↓
RETURN_HOME to (0,0)
↓
MISSION_COMPLETE
```

The important correction in recent builds is that **thermal does not provide distance**. In hardware-faithful mode, thermal provides a 32×24 frame-like reading, blob centroid, bearing, confidence, and apparent strength. The robot must approach in steps and use RGB/YOLO verification rather than driving directly to a thermal-provided coordinate.

---

## Sensor roles

| Subsystem | Role in Ocean Voyager |
|---|---|
| Thermal sensor | Local area-of-interest / bearing detector |
| RGB cameras | Visual verification of plastic/waste versus seaweed/glare/other |
| YOLO model | Classifies camera frames for waste/plastic |
| Coverage planner | Moves Voyager through the pool in a structured lawnmower path |
| Coverage grid | Tracks physical path coverage, thermal-scanned area, camera coverage, and processed object zones |
| Motor/thruster model | Converts movement commands to PWM-style left/right thrust and simulated motion |
| Device bus | Logs CPU-to-device and device-to-CPU commands/responses for debugging and real-hardware integration |

---

## Simulation versus real hardware

The simulator is designed to use the **same interfaces** as real hardware:

```text
ThermalInterface
CameraArrayInterface
MotorExecutorInterface
PanServoInterface
WallGuardInterface
VisionModelInterface
```

In simulation, these interfaces are backed by simulated or image-playback adapters. On the robot, they can be backed by real Raspberry Pi / MLX90640 / camera / PCA9685 / ESC / thruster services.

Important difference:

| Topic | Simulation | Real hardware |
|---|---|---|
| Pose | True pose is known by simulator | Pose must be estimated/dead-reckoned unless overhead tracking is added |
| Object coordinates | Ground truth exists for scenario evaluation | Coordinates are estimated from robot pose + sensor observations |
| Thermal | MLX90640-like frame and bearing-only candidate | MLX90640 32×24 thermal readings |
| Camera | Sim/image playback/YOLO pipeline | Real camera capture on Raspberry Pi |
| Motors | PWM-level simulated motion | PCA9685 → ESC → thrusters |
| Coverage % | Computed from true simulated pose | Estimated from dead reckoning or external tracking |

See [`docs/SIM_VS_REAL_HARDWARE.md`](docs/SIM_VS_REAL_HARDWARE.md) for details.

---

## Quick start

Install dependencies:

```bash
pip install -r requirements.txt
```

### What the quick-start commands do

The quick-start commands exercise different layers of the same Ocean Voyager system.

| Command | What it tests | What it does not test |
|---|---|---|
| Basic simulation | Main autonomy loop with simple simulated sensors/motors | Realistic thermal behavior, YOLO image performance, real hardware |
| Hardware-faithful pool simulation | Main mission behavior with pool patrol, bearing-only thermal, stepwise approach, camera verification, coverage, return-home, and startup checks | Real physical sensor noise, real motor drift, waterproof hardware behavior |
| Model evaluation | YOLO waste detector on image folders | Robot navigation, thermal scanning, coverage, return-home |
| Hardware bring-up | Peripheral readiness on Raspberry Pi: thermal, cameras, motors/log-only safety, bus, status files | Full autonomous water mission |
| Tests | Automated checks that key modules still work | Real-world performance or scientific validation |

### What is common across the modes

All modes use the same basic repository structure and many of the same contracts:

```text
config files
scenario files
mission controller
startup manager
thermal/camera/motor interfaces
vision-model interface
device-bus logging
output folders
coverage and object-status concepts
```

The design idea is that autonomy talks to interfaces, not directly to hardware. That allows the same high-level logic to run with simulated devices, image-playback HIL, or real Raspberry Pi hardware adapters.

### 1. Basic simulation


Run the basic simulation:

```bash
python main.py \
  --config config/sim_full.yaml \
  --scenario scenarios/02_single_bottle_center.yaml
```

This is the fastest sanity check. It verifies scenario loading, the mission controller, simulated sensors, simulated motors, startup status files, state transitions, and output logging. It is not a real hardware or model-performance test.

### 2. Hardware-faithful pool simulation

Run the hardware-faithful pool simulator:

```bash
python main.py \
  --config config/sim_hardware_faithful_pool.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name scenario_36_review
```

This is the main engineering demo mode. It still runs in simulation, but it follows realistic assumptions: thermal provides bearing/confidence only, not object distance; Voyager approaches step-by-step; RGB cameras verify candidates; the coverage grid tracks survey progress; and the mission returns home after the objective.

### 3. Model evaluation

Run model evaluation on image folders:

```bash
python scripts/run_model_test_images.py \
  --model models/best.pt \
  --images assets/test_images \
  --output outputs/model_eval \
  --conf 0.35
```

This does not run the robot. It only evaluates the YOLO model on image folders. A good model score does not prove navigation works; a good navigation simulation does not prove the model is robust. Both layers are needed.


### 4. Raspberry Pi hardware bring-up

For first hardware loading this weekend, start with sensors and motors disabled/log-only:

```bash
python scripts/hardware_bringup.py \
  --config config/pi_hardware_sensors_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name pi_bringup_day1
```

To also attempt one thermal scan and one frame per configured camera:

```bash
python scripts/hardware_bringup.py \
  --config config/pi_hardware_sensors_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name pi_bringup_day1_probe \
  --probe
```

This writes:

```text
outputs/hardware_bringup/pi_bringup_day1/
  startup/startup_log.txt
  startup/startup_status.json
  startup/peripheral_status.json
  device_bus/traces/startup_trace.jsonl
  summary.json
```

Do not enable real motor PWM until neutral PWM, emergency stop, waterproofing, and tether/retrieval safety are verified.

### 5. Staged Raspberry Pi hardware validation

Build 0.5.9 adds a more complete staged path.

#### Stage 1 — run mocked/simulated scenarios on Raspberry Pi

```bash
python scripts/run_all_scenarios.py \
  --config config/sim_full.yaml \
  --preset quick \
  --steps 80
```

This confirms that the Raspberry Pi can run the repo, execute scenarios, write startup files, and produce logs without needing real sensors.

#### Stage 2 — thermal-only bring-up

```bash
python scripts/thermal_bringup.py \
  --config config/thermal_only_bringup.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name thermal_day1 \
  --samples 20 \
  --interactive
```

This logs real MLX90640 readings as Ocean Voyager `ThermalReading` objects: candidate, bearing, confidence, delta, blob pixel count, and frame path. Thermal remains bearing-only; it does not provide distance.

#### Stage 3 — camera-only bring-up

```bash
python scripts/camera_bringup.py \
  --config config/camera_only_bringup.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name camera_day1 \
  --samples 3
```

With YOLO:

```bash
python scripts/camera_bringup.py \
  --config config/stationary_real_thermal_real_camera_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name camera_yolo_day1 \
  --samples 3 \
  --run-vision
```

#### Stage 4 — stationary real-sensor HIL

```bash
python scripts/stationary_real_sensor_hil.py \
  --config config/stationary_real_thermal_real_camera_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name stationary_hil_day1 \
  --interactive
```

In this mode the robot does not physically move. The software logs a simulated approach step, and the operator manually moves the target closer to the stationary sensors.

### 6. Tests

Run tests:

```bash
pytest
```

Some tests that load the YOLO model may be skipped unless optional packages are installed and the environment variable is enabled.

---

## Important scenarios

The scenario files define the pool layout, object locations, and mission objective. Coordinates are relative to the starting position:

```text
home/start = (0,0)
units = meters
```

Interesting scenarios include:

```text
28_delayed_bottle_detection.yaml
29_delayed_bottle_and_seaweed.yaml
33_three_bottles_edges_full_survey.yaml
35_middle_and_edge_bottles_full_survey.yaml
36_time_boxed_three_bottles_full_pool_patrol.yaml
37_time_boxed_four_bottles_plus_seaweed_patrol.yaml
```

Scenario 36 is a good portfolio-style demo: Voyager patrols the pool, finds multiple bottles, continues the lawnmower sweep, and returns home after the time-boxed patrol.

See [`docs/SCENARIO_GUIDE.md`](docs/SCENARIO_GUIDE.md).

---

## Output files

Each run writes a run folder under `outputs/`, typically containing:

```text
trajectory.csv                  Robot position and state over time
object_coordinate_log.csv        Ground truth + estimated object locations/status
ground_truth_objects.csv         Object coordinates from scenario
summary.json                     Mission outcome and coverage metrics
coverage_grid_final.csv          Per-cell coverage status
coverage_grid_summary.json       Coverage percentages
coverage_grid_final.png          Static coverage visualization
world_replay.png                 Static mission replay
mission_animation.gif            Animated replay
thermal_scans.json               Thermal scan events
camera_capture_log.csv           Camera capture events
device_bus/traces/bus_trace.jsonl Command/response trace log
```

Build 0.5.6/0.5.7/0.5.8/0.5.9 review outputs also include:

```text
thermal_event_log.csv
visual_event_log.csv
recognition_timeline.csv
```

See [`docs/OUTPUTS_AND_ANIMATIONS.md`](docs/OUTPUTS_AND_ANIMATIONS.md).

---

## Animation legend

Recent enhanced animations use this convention:

```text
black line                 Voyager trajectory
blue area/circles           physical/path coverage
hollow yellow circle         weak thermal recognition
hollow orange circle         medium thermal recognition
hollow red circle            strong thermal recognition
filled purple diamond        visual first-seen / camera recognition
green check                  confirmed waste/plastic
red cross                    rejected object / seaweed / non-waste
bottom timeline              thermal, visual, and final-decision event timing
```

---

## Documentation map

```text
docs/ARCHITECTURE.md              Autonomy flow and software architecture
docs/SIM_VS_REAL_HARDWARE.md      What is the same/different in sim and hardware
docs/DEVICE_BUS.md                File-based command/response bus design
docs/SCENARIO_GUIDE.md            How scenarios are structured and how to add new ones
docs/OUTPUTS_AND_ANIMATIONS.md    Output files, metrics, color/shape legend
docs/HARDWARE_CALIBRATION_PLAN.md How to calibrate thermal range, camera, and motion model
docs/STARTUP_AND_PERIPHERAL_INIT.md Real-hardware-style boot/readiness sequence
docs/RASPBERRY_PI_BRINGUP.md Weekend hardware bring-up checklist
docs/RASPBERRY_PI_STAGED_VALIDATION.md Staged Pi validation path
docs/STATIONARY_REAL_SENSOR_HIL.md Real thermal + real camera stationary HIL
docs/DEVELOPMENT_HISTORY.md       Build history from 0.1 through 0.5.9
```

---

## Current limitations

This is a serious simulation/HIL platform, but it is not yet a perfect physical digital twin.

Current limitations include:

- Real pool localization is still approximate without GPS/overhead tracking/AprilTags/IMU calibration.
- Thermal sensor behavior is modeled as MLX90640-like, but real temperature contrast must be calibrated in the pool.
- Thruster motion uses configurable PWM/motion parameters, but real distance-per-command must be calibrated.
- Enhanced animation rendering exists in outputs and docs, but should next be formalized into a reusable script for any new scenario.

---

## Recommended next steps

1. Add a reusable enhanced animation renderer script:
   ```text
   scripts/render_enhanced_animation.py
   ```
2. Run real living-room HIL with real camera frames and motors disabled.
3. Calibrate thermal detection envelope in pool at 1 m, 2 m, 3 m, 4 m.
4. Calibrate movement:
   ```text
   PWM + duration → distance traveled / turn angle
   ```
5. Compare simulator and real hardware using device-bus traces.


### Raspberry Pi AI Camera / IMX500 bring-up

Build 0.5.10 adds a `rpicam-still` backend for the Raspberry Pi AI Camera. After `rpicam-hello --list-cameras` shows `imx500` and `rpicam-still` creates a test JPEG, run:

```bash
python scripts/camera_bringup.py \
  --config config/camera_only_ai_camera_bringup.yaml \
  --camera-ids front \
  --samples 5
```

This route does not use OpenCV `/dev/video0`; it captures through Raspberry Pi's `rpicam` / libcamera stack. Motors remain log-only. See `docs/RASPBERRY_PI_AI_CAMERA_BRINGUP.md`.
