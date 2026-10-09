# Ocean Voyager Simulation/HIL Build 0.5.2 Report

## Build theme

Build 0.5.2 focuses on delayed thermal discovery and coverage-memory behavior.

The main additions are:

1. New delayed thermal scenarios where the robot scans, initially finds no thermal signature, moves on the lawnmower path, then detects the object only after it enters thermal range.
2. A coverage grid that remembers path-covered, thermally scanned, and object-processed areas.
3. Processed-object zones around confirmed/rejected targets, preventing repeated chasing of the same object.
4. Coverage waypoint skipping when a waypoint neighborhood has already been explored.
5. Thermal scan logs that explicitly record no-candidate scans and candidate-hit scans.
6. Updated replay PNG/GIF animations showing coverage grid, thermal scan history, candidate estimates, object coordinates, and final object status.

## Important new files

```text
ocean_voyager/planning/coverage_grid.py

scenarios/28_delayed_bottle_detection.yaml
scenarios/29_delayed_bottle_and_seaweed.yaml
scenarios/30_object_outside_thermal_range.yaml
scenarios/31_object_enters_range_after_coverage_move.yaml
scenarios/32_low_delta_becomes_detectable_when_close.yaml
```

## Updated files

```text
main.py
ocean_voyager/core/mission_controller.py
ocean_voyager/core/logger.py
ocean_voyager/planning/coverage.py
ocean_voyager/simulation/replay.py
config/sim_hardware_faithful_pool.yaml
scripts/run_build_0_5_pool_suite.py
tests/test_yolo_model.py
```

## New output files per run

Build 0.5.2 adds these coverage and scan artifacts:

```text
coverage_grid_final.csv
coverage_grid_summary.json
coverage_grid_timeline.jsonl
coverage_grid_final.png
thermal_scans.json
```

The original mission artifacts remain:

```text
trajectory.csv
object_coordinate_log.csv
ground_truth_objects.csv
object_coordinates.json
world_replay.png
mission_animation.gif
mission_log.jsonl
device_bus/traces/bus_trace.jsonl
summary.json
```

## Coverage grid behavior

The pool is divided into cells. Each cell can record:

```text
wall_margin
path_covered
thermal_scanned
vision_checked
processed_object_id
processed_status
explored
```

A cell is treated as explored if the robot has physically passed near it, the thermal scan sector covered it, or it is inside a processed object zone.

This allows the robot to return to the original lawnmower waypoint after a detour while still remembering which areas were already covered or scanned.

## Delayed thermal behavior

The new delayed scenarios demonstrate that thermal can initially miss an object and later detect it after coverage movement.

| Scenario | Purpose | Expected behavior |
|---|---|---|
| 28 | Delayed bottle detection | Many no-candidate scans, then bottle candidate, confirm, return home |
| 29 | Delayed bottle + seaweed | Bottle confirmed, seaweed rejected, return home |
| 30 | Object outside thermal range | No thermal candidate before return-home timer |
| 31 | Object enters range after coverage move | Bottle enters reduced thermal range and is confirmed |
| 32 | Low thermal delta becomes detectable when close | Weak thermal signature crosses threshold only after robot gets closer |

## Sample run summary from generated outputs

```text
28 delayed bottle:                first thermal hit at t=25.5s after 187 no-candidate scan records
29 delayed bottle + seaweed:      first thermal hit at t=21.0s; bottle confirmed and seaweed rejected
30 outside thermal range:         no candidate detected; returned home
31 enters range after movement:   first thermal hit at t=21.0s after initial no-candidate scans
32 low delta close detection:     first thermal hit at t=9.0s after initial no-candidate scans
```

## Animation changes

The replay animations now show:

```text
Blue-ish grid cells: path-covered area
Yellow grid cells: thermally scanned area
Green/red processed zones: confirmed/rejected object neighborhoods
Gray dots: thermal scans with no candidate
Orange diamonds: thermal candidate estimates
Black path line: Voyager path
Object labels: ID, type, and coordinates relative to start/home (0,0)
```

## Validation

Test suite:

```text
9 passed, 1 skipped
```

The skipped test is the optional YOLO model load test. Set `RUN_YOLO_LOAD_TEST=1` locally to run it.

## Commands

Run a delayed thermal scenario:

```bash
python main.py \
  --config config/sim_hardware_faithful_pool.yaml \
  --scenario scenarios/28_delayed_bottle_detection.yaml \
  --run-name 28_delayed_bottle_detection
```

Run all Build 0.5/0.5.2 pool scenarios:

```bash
python scripts/run_build_0_5_pool_suite.py
```

Inspect key outputs:

```text
outputs/build_0_5_pool_runs/<scenario>/mission_animation.gif
outputs/build_0_5_pool_runs/<scenario>/world_replay.png
outputs/build_0_5_pool_runs/<scenario>/coverage_grid_final.png
outputs/build_0_5_pool_runs/<scenario>/thermal_scans.json
outputs/build_0_5_pool_runs/<scenario>/coverage_grid_summary.json
outputs/build_0_5_pool_runs/<scenario>/object_coordinate_log.csv
```

## Known limitations

1. The coverage grid is a planner-memory model, not a proof of actual physical coverage until real localization is available.
2. Thermal range and delta thresholds still need real MLX90640 pool calibration.
3. The simulator uses true pose; real hardware will initially use estimated/dead-reckoned pose unless overhead tracking, AprilTags, or IMU integration is added.
4. Real T200 movement per PWM/duration still needs pool calibration.
