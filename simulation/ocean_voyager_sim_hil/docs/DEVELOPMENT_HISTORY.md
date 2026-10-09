# Development History

## Build 0.1

Initial simulation/HIL skeleton:

- YAML config/scenario loading
- simulated pool world
- robot pose and object coordinates
- differential-thrust movement
- simulated thermal/camera/motor adapters
- mission controller
- stationary HIL mode
- output logs

## Build 0.2

Added trained YOLO model integration:

- `models/best.pt`
- model evaluation harness
- target label mapping
- YOLO wrapper

## Build 0.3

Added real uploaded test images:

- positives: plastic/bottle images
- negatives: seaweed/glare
- threshold sweep
- annotated model outputs

## Build 0.4

Added image-playback stationary HIL:

- real image assets feed into the mission pipeline
- YOLO verification inside HIL flow
- motors disabled/log-only

## Build 0.5

Added coordinate mission replay:

- start/home = (0,0)
- object coordinate logging
- return-home behavior
- animation output
- file-based device-bus logs
- hardware profile layer

## Build 0.5.1

Added coordinate labels into outputs:

- `ground_truth_objects.csv`
- `object_coordinates.json`
- coordinate labels in replay images/animations

## Build 0.5.2

Added delayed thermal scenarios and coverage grid:

- delayed bottle detection
- object outside range
- low-delta object becomes detectable when closer
- coverage grid outputs

## Build 0.5.3

Corrected thermal realism:

- thermal no longer gives distance
- thermal provides bearing/blob/confidence only
- approach becomes stepwise with RGB captures
- physical path coverage is separated from thermal scan coverage

## Build 0.5.4

Added time-boxed patrol mode:

- repeat lawnmower passes until time expires
- return home after timer
- pass-equivalent coverage metric
- scenarios 36 and 37

## Build 0.5.5

Added recognition visualization:

- thermal first-seen markers
- visual first-seen markers
- green check for confirmed waste
- red cross for rejected object

## Build 0.5.6

Improved visual convention:

- thermal = hollow circles
- visual = filled purple diamonds
- bottom event timeline strip

## Build 0.5.7

Documentation cleanup:

- polished README
- architecture documentation
- simulation-vs-real explanation
- device-bus documentation
- scenario guide
- output/animation guide
- hardware calibration plan


## Build 0.5.8

Hardware bring-up and startup sequence:

- Added formal startup manager.
- Added peripheral status schema.
- Added startup output files:
  - `startup/startup_log.txt`
  - `startup/startup_status.json`
  - `startup/peripheral_status.json`
  - `device_bus/traces/startup_trace.jsonl`
- Added safe Raspberry Pi hardware bring-up script:
  - `scripts/hardware_bringup.py`
- Added safe Pi config:
  - `config/pi_hardware_sensors_log_only.yaml`
- Real hardware adapters no longer fail at construction; they report readiness through health checks.
- README quick-start section now explains how basic simulation, hardware-faithful simulation, model evaluation, hardware bring-up, and tests differ.
- Added docs:
  - `STARTUP_AND_PERIPHERAL_INIT.md`
  - `RASPBERRY_PI_BRINGUP.md`


## Build 0.5.9

Staged Raspberry Pi hardware validation:

- Added `scripts/run_all_scenarios.py` for mocked/simulated scenario sweeps on Raspberry Pi.
- Added `scripts/thermal_bringup.py` for MLX90640 thermal-only bring-up.
- Added `scripts/camera_bringup.py` for real camera bring-up and optional vision-model testing.
- Added `scripts/stationary_real_sensor_hil.py` for real thermal + real camera + log-only motor stationary HIL.
- Added configs:
  - `config/thermal_only_bringup.yaml`
  - `config/camera_only_bringup.yaml`
  - `config/stationary_real_thermal_real_camera_log_only.yaml`
  - `config/stationary_real_thermal_real_camera_mock_vision.yaml`
- Updated YOLO wrapper to report health-check failure cleanly instead of crashing during construction if dependencies/model are unavailable.
- Added docs:
  - `docs/RASPBERRY_PI_STAGED_VALIDATION.md`
  - `docs/STATIONARY_REAL_SENSOR_HIL.md`
- README now includes a staged Pi validation workflow.
