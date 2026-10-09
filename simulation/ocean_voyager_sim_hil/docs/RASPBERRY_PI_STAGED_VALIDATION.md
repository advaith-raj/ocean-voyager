# Raspberry Pi Staged Hardware Validation

Build 0.5.9 adds a staged workflow for moving from pure simulation to real sensors.

The goal is to avoid jumping directly from a laptop/server simulation into a full water mission. Instead, each hardware layer is tested separately and logged.

---

## Stage 1 — mocked/simulated peripherals on Raspberry Pi

Run the same simulator on the Raspberry Pi with all peripherals mocked/simulated.

Purpose:

```text
verify Pi Python environment
verify repo installs/runs
verify scenario/config loading
verify startup manager
verify output logging
verify device-bus trace generation
verify mission controller
```

Command:

```bash
python scripts/run_all_scenarios.py \
  --config config/sim_full.yaml \
  --preset quick \
  --steps 80
```

Outputs:

```text
outputs/pi_mocked_scenario_sweep/
  scenario_sweep_summary.csv
  scenario_sweep_summary.json
```

This does not require real sensors.

---

## Stage 2 — thermal-only bring-up

Connect the MLX90640 thermal sensor to the Raspberry Pi.

Purpose:

```text
verify I2C / MLX90640 communication
save raw thermal frames
save preview images
log thermal delta / blob pixels / confidence / bearing
map real thermal readings into Ocean Voyager ThermalReading objects
```

Command:

```bash
python scripts/thermal_bringup.py \
  --config config/thermal_only_bringup.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name thermal_day1 \
  --samples 20 \
  --interactive
```

Outputs:

```text
outputs/hardware_bringup/thermal_day1/
  startup/startup_log.txt
  thermal_frames/*.npy
  thermal_preview/*.png
  thermal_readings.csv
  thermal_summary.json
```

What to test:

```text
empty water
water + object centered
object left of center
object right of center
object closer / farther
warmer object / cooler object if available
```

Important mapping to autonomy logic:

```text
candidate = true
confidence >= mission.thermal_min_confidence
bearing_deg = direction to investigate
range_known = false
```

Thermal still does not provide distance.

---

## Stage 3 — camera-only bring-up

Connect RGB camera(s).

Purpose:

```text
verify camera IDs
capture real frames
save images
optionally run YOLO
log detections
```

Command without YOLO:

```bash
python scripts/camera_bringup.py \
  --config config/camera_only_bringup.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name camera_day1 \
  --samples 3
```

Command with configured vision model:

```bash
python scripts/camera_bringup.py \
  --config config/stationary_real_thermal_real_camera_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name camera_yolo_day1 \
  --samples 3 \
  --run-vision
```

Outputs:

```text
camera_frames.csv
vision_detections.json
real_camera_frames/
```

---

## Stage 4 — stationary real-sensor HIL

Use real thermal and real camera, but keep motors log-only.

The robot physically stays still. The software simulates an approach command. The operator manually moves the target closer/aligned with the thermal bearing.

Command:

```bash
python scripts/stationary_real_sensor_hil.py \
  --config config/stationary_real_thermal_real_camera_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name stationary_hil_day1 \
  --interactive
```

Dry-run without real sensors:

```bash
python scripts/stationary_real_sensor_hil.py \
  --config config/stationary_real_thermal_real_camera_mock_vision.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name stationary_hil_dry_run \
  --use-sim \
  --mock-vision
```

Outputs:

```text
stationary_hil_events.csv
stationary_hil_detections.json
startup/startup_status.json
device_bus/traces/bus_trace.jsonl
summary.json
```

---

## Stage 5 — later real motion

Only after stages 1–4 are stable:

```text
calibrate neutral PWM
verify emergency stop
verify tether/retrieval
verify waterproofing
calibrate movement distance/turn rate
```

Then real motor movement can be considered.

Do not begin with real motor movement.
