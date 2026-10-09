# Outputs and Animations

Ocean Voyager generates logs and visual artifacts for each mission run.

---

## Main output files

```text
trajectory.csv
  Robot pose and state over time.

object_coordinate_log.csv
  Ground truth and estimated object coordinates.

ground_truth_objects.csv
  Object positions from the scenario file.

summary.json
  Mission outcome, coverage metrics, path length, return-home result.

coverage_grid_final.csv
  Per-cell coverage grid status.

coverage_grid_summary.json
  Coverage percentages and visit counts.

coverage_grid_final.png
  Static coverage map.

world_replay.png
  Static mission replay.

mission_animation.gif
  Animated mission replay.

thermal_scans.json
  Thermal scan events.

camera_capture_log.csv
  Camera capture events.

device_bus/traces/bus_trace.jsonl
  CPU-to-device and device-to-CPU command trace.
```

---

## Recognition event logs

Enhanced review outputs include:

```text
thermal_event_log.csv
visual_event_log.csv
recognition_timeline.csv
```

These identify when each object was first seen by thermal, when vision first saw it, and when it was confirmed or rejected.

---

## Animation color and shape legend

Build 0.5.6/0.5.7 convention:

```text
black line                 Voyager trajectory
blue area/circles           physical/path coverage
hollow yellow circle         weak thermal recognition
hollow orange circle         medium thermal recognition
hollow red circle            strong thermal recognition
filled purple diamond        visual first-seen / camera recognition
green check                  confirmed waste/plastic
red cross                    rejected object / seaweed / non-waste
bottom timeline              timing of thermal, visual, and final decisions
```

---

## Coverage metrics

### Path coverage

Physical/path coverage is the percentage of surveyable grid cells that the robot passed near.

```text
path coverage % =
  path-covered cells / valid survey cells × 100
```

### Thermal coverage

Thermal coverage is the percentage of cells included in thermal scan cones.

This is useful evidence, but it should not replace physical path coverage for full-pool survey completion.

### Camera coverage

Camera coverage is the percentage of grid cells seen by RGB camera snapshots/captures.

### Pass-equivalent coverage

Unique path coverage cannot exceed 100%. Pass-equivalent/cumulative coverage can exceed 100%.

Example:

```text
100% = pool covered once
150% = about one and a half passes
200% = about two passes
```

---

## Interpreting the animation

A strong demo should show:

```text
1. Voyager starts at home.
2. It follows a lawnmower patrol.
3. Thermal hollow markers begin appearing.
4. Visual purple diamond appears after approach.
5. Green check or red cross marks final decision.
6. Voyager resumes patrol.
7. It returns home when objective/time is complete.
```
