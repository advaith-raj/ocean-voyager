# Ocean Voyager Simulation/HIL Build 0.5 Report

## Build theme

Build 0.5 upgrades the simulator from a workflow-only simulator toward a hardware-faithful, coordinate-based mission replay system.

The key goals were:

1. Use swimming-pool layouts with coordinates relative to the robot start point `(0,0)`.
2. Place plastic bottles, seaweed, and glare/false-positive objects at scenario-specific coordinates.
3. Track the Ocean Voyager path over time.
4. Store confirmed waste coordinates and object status.
5. Return to `(0,0)` after a mission timer.
6. Produce static replay images and animated GIFs.
7. Log CPU-to-peripheral and peripheral-to-CPU command/response traffic through a file-based device bus.
8. Add hardware-faithful configuration profiles for MLX90640-like thermal, camera array, PCA9685, and T200-style thrusters.

## Important new files

```text
config/sim_hardware_faithful_pool.yaml
config/hardware_profiles/mlx90640.yaml
config/hardware_profiles/camera_array.yaml
config/hardware_profiles/pca9685_t200_thrusters.yaml

ocean_voyager/device_bus/file_bus.py
ocean_voyager/simulation/replay.py

scripts/run_build_0_5_pool_suite.py
scripts/compare_bus_traces.py
scripts/peripheral_service_template.py

scenarios/20_pool_empty_return_home.yaml
scenarios/21_pool_single_bottle_return_home.yaml
scenarios/22_pool_bottle_and_seaweed_return_home.yaml
scenarios/23_pool_bottle_near_wall_return_home.yaml
scenarios/24_pool_multiple_objects_return_home.yaml
scenarios/25_pool_start_near_corner_return_home.yaml
scenarios/26_pool_low_thermal_delta_return_home.yaml
scenarios/27_pool_false_positive_only_return_home.yaml
```

## New output files per run

Each Build 0.5 pool run now produces:

```text
trajectory.csv
object_coordinate_log.csv
world_replay.png
mission_animation.gif
mission_log.jsonl
detections.json
recommended_actions.json
summary.json
thermal_frames/*.npy
frames/*.png
device_bus/commands/processed/*.json
device_bus/responses/processed/*.json
device_bus/traces/bus_trace.jsonl
```

## Device bus design

The simulated main CPU/peripheral communication is now logged through a file-based device bus.

Each command is written as a JSON file under:

```text
device_bus/commands/processed/
```

Each response is written as a JSON file under:

```text
device_bus/responses/processed/
```

A combined command/response trace is written to:

```text
device_bus/traces/bus_trace.jsonl
```

Example command types include:

```text
pan_servo.set_angle
thermal.capture_frame
camera_array.capture
motors.set_pwm_for_duration
motors.motion_update
motors.stop
```

This makes it easier to compare simulator logs with future real hardware logs.

## Hardware-faithful simulation improvements

### Thermal

The thermal simulator now produces MLX90640-like 24 x 32 thermal frame files:

```text
thermal_frames/thermal_000001_pan_+00.0.npy
```

It logs thermal frame metadata, including:

```text
frame shape
pan angle
background temperature
max temperature
max delta
peak pixel
visible objects
```

### Camera

The simulated camera array logs camera capture commands and responses. It uses configurable:

```text
front / left / right cameras
frame width / height
FOV
mount angles
latency config
```

### Motors / thrusters

The motor simulator now converts motion commands into PWM-like outputs and logs:

```text
left_pwm_us
right_pwm_us
duration_sec
normalized left/right thruster values
estimated start/end pose
estimated distance
estimated speed
yaw rate
```

This is not a fully calibrated hydrodynamic model yet, but it is much closer to the real command path.

## Validation summary

Test suite:

```text
9 passed, 1 skipped
```

The skipped test is the YOLO model load test when `ultralytics` is not installed in the runtime.

Build 0.5 sample pool scenarios were run and output animations were generated.

| Scenario | Confirmed | Rejected | Returned home | Notes |
|---|---:|---:|---:|---|
| 20 empty pool | 0 | 0 | yes | path + home return only |
| 21 single bottle | 1 | 0 | yes | bottle confirmed |
| 22 bottle + seaweed | 1 | 1 | yes | bottle confirmed, seaweed rejected |
| 23 bottle near wall | 1 | 0 | yes | near-wall bottle confirmed |
| 24 multiple objects | 2 | 1 | yes | two bottles confirmed, seaweed rejected |
| 25 start near corner | 1 | 1 | yes | bottle confirmed, seaweed rejected |
| 26 low thermal delta | 0 | 0 | yes | low thermal-contrast bottle missed, as expected |
| 27 false-positive only | 0 | 1 | yes | seaweed rejected |

## Commands

Run one scenario:

```bash
python main.py \
  --config config/sim_hardware_faithful_pool.yaml \
  --scenario scenarios/22_pool_bottle_and_seaweed_return_home.yaml \
  --run-name my_pool_test
```

Run the Build 0.5 pool suite:

```bash
python scripts/run_build_0_5_pool_suite.py
```

Compare future simulator-vs-real traces:

```bash
python scripts/compare_bus_traces.py \
  --sim outputs/build_0_5_pool_runs/22_pool_bottle_and_seaweed_return_home/device_bus/traces/bus_trace.jsonl \
  --real outputs/real_pool_test/device_bus/traces/bus_trace.jsonl \
  --output bus_trace_comparison.json
```

## Known limitations

1. Real hardware calibration is still needed for actual pool movement distance per PWM/duration.
2. Real MLX90640 field-of-view variant should be confirmed: 55 x 35 or 110 x 75.
3. Real RGB camera model/lens/FOV should be confirmed.
4. File bus is currently simulated in-process; the command/response schema is ready for separate real peripheral services.
5. Real return-to-home will require dead reckoning at first and may drift until calibrated.
