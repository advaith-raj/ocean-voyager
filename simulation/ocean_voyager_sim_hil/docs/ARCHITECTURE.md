# Ocean Voyager Architecture

This document explains how Ocean Voyager is structured internally.

---

## Core design principle

The main autonomy code should not care whether sensors and actuators are simulated or real. It talks to abstract interfaces:

```text
ThermalInterface
CameraArrayInterface
MotorExecutorInterface
PanServoInterface
WallGuardInterface
VisionModelInterface
```

Simulation, HIL, and real hardware are selected through config files and adapter classes.

---

## Main control flow

```text
main.py
  ↓
config_loader.py
  ↓
factory.py builds:
  - world
  - thermal adapter
  - camera adapter
  - motor adapter
  - wall guard
  - vision model
  - logger
  ↓
MissionController.run()
```

The mission controller owns the main state machine.

---

## Mission state machine

Typical pool survey mission:

```text
INIT
  ↓
COVERAGE_SWEEP
  ↓
THERMAL_CANDIDATE_FOUND
  ↓
APPROACH_TARGET
  ↓
VISION_VERIFY
  ↓
RETURN_TO_COVERAGE
  ↓
COVERAGE_SWEEP
  ↓
RETURN_HOME
  ↓
MISSION_COMPLETE
```

Some scenarios return after first object. Time-boxed patrol scenarios continue surveying until the mission timer expires.

---

## Coverage planning

Ocean Voyager uses a lawnmower/serpentine pattern.

```text
→ → → → →
          ↓
← ← ← ← ←
↓
→ → → → →
```

The coverage planner generates waypoints inside the safe pool region. The coverage grid tracks what has actually been physically covered.

Important rule:

```text
thermal-scanned area is not the same as physical/path coverage
```

Thermal coverage is logged as evidence, but physical/path coverage is the main metric for full-pool survey completion.

---

## Thermal behavior

In hardware-faithful mode, thermal does not provide distance.

The simulated thermal sensor approximates an MLX90640-style frame and extracts:

```text
blob centroid
bearing
thermal delta
pixel count / apparent blob size
confidence
```

It does not return:

```text
exact object distance
exact object coordinate
```

The robot therefore uses thermal as a bearing/area-of-interest cue.

---

## Approach behavior

When a thermal candidate is found:

```text
1. Turn toward the thermal bearing.
2. Move one short step.
3. Capture RGB frames.
4. Run YOLO.
5. Rescan thermal.
6. Repeat until confirm/reject/lost.
```

This avoids giving the simulator unrealistic knowledge that the real thermal sensor would not have.

---

## Vision verification

RGB cameras and YOLO are used to decide whether the candidate is waste.

```text
thermal cue → stepwise approach → camera frames → YOLO → confirm/reject
```

For current demos:

```text
confirmed waste = green check
rejected object/seaweed = red cross
```

---

## Coverage grid

The grid tracks per-cell status:

```text
path_covered
thermal_scanned
vision_checked
processed_object
confirmed_waste_zone
rejected_object_zone
wall_margin
```

This supports:

- path coverage %
- camera coverage %
- thermal scan coverage %
- processed object zones
- repeat patrol / pass-equivalent coverage

---

## Return home

Home is the start coordinate:

```text
home = (0,0)
```

Return-home behavior is triggered by:

```text
time expired
coverage complete
first object found, if scenario is find-and-return
```

In simulation, pose is known exactly. In real pool testing, return-home requires estimated pose from dead reckoning or external localization.

---

## Important modules

```text
ocean_voyager/core/mission_controller.py
  Mission state machine

ocean_voyager/planning/coverage.py
  Lawnmower/serpentine waypoints

ocean_voyager/planning/coverage_grid.py
  Grid-level coverage metrics

ocean_voyager/planning/target_approach.py
  Bearing/target approach commands

ocean_voyager/perception/thermal_detector.py
  Thermal candidate extraction

ocean_voyager/perception/vision_model.py
  Mock/YOLO vision model wrapper

ocean_voyager/simulation/robot_dynamics.py
  Differential-thrust motion model

ocean_voyager/device_bus/file_bus.py
  File-based command/response tracing
```
