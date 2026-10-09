# Ocean Voyager Build 0.5.9 Report

Build 0.5.9 focuses on the staged path for running Ocean Voyager on Raspberry Pi with progressively more real hardware.

## Purpose

The intended weekend workflow is:

```text
1. Run all software on Raspberry Pi with mocked/simulated peripherals.
2. Bring up the MLX90640 thermal sensor by itself.
3. Bring up the RGB camera/cameras by themselves.
4. Run stationary real-sensor HIL:
   - real thermal
   - real camera
   - YOLO if available
   - motors log-only
   - operator manually moves the target closer
```

## New scripts

```text
scripts/run_all_scenarios.py
scripts/thermal_bringup.py
scripts/camera_bringup.py
scripts/stationary_real_sensor_hil.py
```

### run_all_scenarios.py

Runs a quick or full scenario sweep and writes:

```text
scenario_sweep_summary.csv
scenario_sweep_summary.json
```

### thermal_bringup.py

Captures thermal readings and writes:

```text
thermal_frames/*.npy
thermal_preview/*.png
thermal_readings.csv
thermal_summary.json
```

### camera_bringup.py

Captures camera frames and optionally runs the vision model.

Outputs:

```text
camera_frames.csv
vision_detections.json
real_camera_frames/
```

### stationary_real_sensor_hil.py

Runs the real thermal + real camera stationary HIL loop. Motors remain log-only.

Outputs:

```text
stationary_hil_events.csv
stationary_hil_detections.json
summary.json
startup/
device_bus/
```

## New configs

```text
config/thermal_only_bringup.yaml
config/camera_only_bringup.yaml
config/stationary_real_thermal_real_camera_log_only.yaml
config/stationary_real_thermal_real_camera_mock_vision.yaml
```

## New docs

```text
docs/RASPBERRY_PI_STAGED_VALIDATION.md
docs/STATIONARY_REAL_SENSOR_HIL.md
```

## Vision-model startup behavior

The YOLO wrapper now supports health-check style behavior. If `ultralytics`, `torch`, or `models/best.pt` are unavailable, the adapter reports failed readiness instead of crashing at construction.

## Safety

Real motor movement is still intentionally out of scope for this build. Motors are log-only in all new hardware bring-up configs.
