# Scenario Guide

Scenarios define the pool, start position, object positions, mission objective, and expected outcomes.

---

## Coordinate frame

All pool-object coordinates are relative to the starting position:

```text
start/home = (0,0)
units = meters
```

This makes it easy to set up real pool tests.

Example:

```yaml
robot:
  start_x: 0.0
  start_y: 0.0
  start_heading_deg: 0.0

objects:
  - id: bottle_001
    type: bottle
    x: 3.2
    y: 1.1
```

---

## Object types

Supported object types include:

```text
bottle
seaweed
glare
wall
unknown
```

Typical expected outcomes:

```text
confirm_waste
reject_false_positive
ignore_or_miss
```

---

## Mission objectives

Recommended objective types:

### 1. find_and_return

Find the first confirmed object and return home.

Useful for quick detection demos.

### 2. full_pool_survey

Continue until a required physical/path coverage percentage is reached.

Example:

```yaml
mission:
  objective: full_pool_survey
  required_path_coverage_percent: 90
  continue_after_confirming_objects: true
```

### 3. time_boxed_patrol

Keep surveying until mission time expires, then return home.

Example:

```yaml
mission:
  objective: time_boxed_patrol
  max_time_sec: 420
  return_home_trigger: time_expired
  continue_after_confirming_objects: true
```

---

## Interesting scenarios

```text
28_delayed_bottle_detection.yaml
  Demonstrates no initial thermal recognition, then delayed detection.

29_delayed_bottle_and_seaweed.yaml
  Demonstrates bottle confirmation and seaweed rejection.

30_object_outside_thermal_range.yaml
  Demonstrates object not detected before return-home.

31_object_enters_range_after_coverage_move.yaml
  Demonstrates detection only after Voyager moves closer.

32_low_delta_becomes_detectable_when_close.yaml
  Demonstrates weak thermal contrast becoming useful at closer range.

33_three_bottles_edges_full_survey.yaml
  Demonstrates multiple bottles around the pool.

36_time_boxed_three_bottles_full_pool_patrol.yaml
  Recommended main demo: multiple bottles + full pool patrol.

37_time_boxed_four_bottles_plus_seaweed_patrol.yaml
  Recommended advanced demo: bottles + rejected seaweed.
```

---

## Adding a new scenario

1. Copy a similar YAML file from `scenarios/`.
2. Change object positions.
3. Choose mission objective.
4. Run:

```bash
python main.py \
  --config config/sim_hardware_faithful_pool.yaml \
  --scenario scenarios/YOUR_SCENARIO.yaml \
  --run-name your_run_name
```

5. Inspect:

```text
outputs/your_run_name/summary.json
outputs/your_run_name/mission_animation.gif
outputs/your_run_name/object_coordinate_log.csv
```
