# Stationary Real-Sensor HIL

Stationary real-sensor HIL is the bridge between simulation and a physical moving robot.

The key idea:

```text
real thermal sensor
real RGB camera
real YOLO model if available
motors log-only
robot does not physically move
operator manually moves target object closer
```

This lets Ocean Voyager test the real sensing/decision pipeline before water motion is attempted.

---

## Why this mode exists

In normal simulation, the robot moves through the pool.

In stationary HIL, the robot does not move. Instead:

```text
software decides it would move toward a thermal bearing
operator moves the object closer to the stationary sensors
software captures thermal/camera again
software verifies with vision
```

This preserves the autonomy logic while avoiding physical movement risk.

---

## Command

```bash
python scripts/stationary_real_sensor_hil.py \
  --config config/stationary_real_thermal_real_camera_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name stationary_hil_day1 \
  --interactive
```

---

## Operator flow

Example terminal flow:

```text
[Cycle 1] THERMAL weak candidate: bearing=+18.0°, conf=0.31
[Cycle 1] Simulated Voyager would move about 0.25 m toward bearing +18.0°.
Move the target about that much closer/aligned with the bearing, then press Enter...

[Cycle 1] camera=front image=...
[Cycle 1] DECISION: thermal candidate seen, vision not confirmed yet.

[Cycle 2] THERMAL medium candidate: bearing=+10.0°, conf=0.48
Move target closer...

[Cycle 2] camera=front image=...
[Cycle 2] DECISION: confirmed waste ✓
```

---

## Outputs

```text
stationary_hil_events.csv
stationary_hil_detections.json
startup/startup_log.txt
startup/startup_status.json
startup/peripheral_status.json
device_bus/traces/bus_trace.jsonl
summary.json
```

### stationary_hil_events.csv

Each row represents a thermal/camera/decision cycle:

```text
cycle
thermal_candidate
thermal_bearing_deg
thermal_confidence
thermal_strength
thermal_delta_c
thermal_blob_pixel_count
simulated_action
vision_confirmed
vision_best_label
vision_best_confidence
decision
```

### stationary_hil_detections.json

Raw detection records per camera and cycle.

---

## Important interpretation

A thermal candidate means:

```text
candidate = true
bearing_deg = direction of interest
confidence = how strong the cue is
range_known = false
```

The robot should not claim that thermal gave object distance.

A vision confirmation means:

```text
camera captured image
YOLO/mock model detected a target label
confidence >= vision threshold
```

Only after vision confirmation should the system mark the object as confirmed waste.

---

## Configs

Main real-sensor config:

```text
config/stationary_real_thermal_real_camera_log_only.yaml
```

Dry-run/mock vision config:

```text
config/stationary_real_thermal_real_camera_mock_vision.yaml
```

The real-sensor config uses YOLO:

```yaml
vision_model:
  type: yolo
  model_path: models/best.pt
  confidence_threshold: 0.35
```

If YOLO dependencies are not installed, startup will report the vision model as failed rather than crashing silently.

---

## Safety

Motors remain log-only:

```yaml
sources:
  motors: log_only
```

The script may log simulated approach commands, but it does not move the robot.
