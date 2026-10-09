# Hardware Calibration Plan

The simulator is only useful if it is calibrated against real hardware.

This document lists the calibration tests needed before claiming real-world accuracy.

---

## 1. Thermal detection envelope

Test the real thermal sensor at known distances.

Suggested distances:

```text
1 m
2 m
3 m
4 m
```

For each object:

```text
plastic bottle
seaweed / sargassum substitute
glare / reflection
empty water
```

Record:

```text
distance
bearing
thermal max/min
thermal delta
blob pixel count
confidence
lighting condition
water temperature
object temperature if available
```

Goal:

```text
Determine when thermal is weak, medium, strong, or absent.
```

---

## 2. Camera / YOLO detection envelope

At the same distances, capture RGB frames.

Record:

```text
distance
camera id
lighting condition
YOLO label
YOLO confidence
bounding box size
true positive / false positive / false negative
```

Goal:

```text
Determine practical visual confirmation range.
```

---

## 3. Motion calibration

Use short motor pulses in pool.

Example table:

```text
left_pwm right_pwm duration measured_distance measured_turn_angle
1550     1550      1.0      ...
1600     1600      1.0      ...
1550     1650      1.0      ...
1450     1550      1.0      ...
```

Goal:

```text
PWM + duration → distance traveled / turn angle
```

Update:

```text
config/hardware_profiles/pca9685_t200_thrusters.yaml
motion_model:
  max_forward_speed_mps
  turn_rate_gain_deg_per_s
  acceleration_lag_s
  drag coefficients
```

---

## 4. Return-home calibration

Run a time-boxed patrol and compare:

```text
dead-reckoned final pose
observed final pose
actual distance from home
```

If possible, record overhead video to measure drift.

---

## 5. Device-bus comparison

Run the same test in simulation and real hardware/log-only mode.

Compare:

```text
bus_trace.jsonl from simulation
bus_trace.jsonl from real hardware
```

Look for differences in:

```text
sensor latency
thermal bearing
vision confidence
motor command execution
estimated movement
missed/false detections
```

---

## 6. Safety

Real motor testing should always include:

```text
adult supervision
tether or retrieval line
emergency stop
motors neutral on startup
motors neutral on timeout
no exposed electronics near water
low PWM limits for first pool tests
```
