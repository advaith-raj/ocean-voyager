# Thermal Bearing-Only Design

A real MLX90640-like thermal sensor does not provide object distance. It provides a low-resolution temperature image. Ocean Voyager should therefore use thermal as a directional cue, not as a rangefinder.

## Sensor output used by autonomy

The autonomy loop may use:

- pan angle
- thermal frame path
- blob centroid pixel
- blob pixel count
- max thermal delta
- confidence
- bearing estimate

The autonomy loop must not use:

- true object range
- true object ID
- true object coordinate

## Approach loop

```text
thermal blob bearing found
→ capture RGB frames
→ run vision
→ rescan thermal
→ turn/move one short step toward latest bearing
→ repeat until RGB confirms/rejects or candidate is lost
```

## Coordinate logging

Simulation still stores scenario ground-truth coordinates for evaluation and animation labels. The robot's estimated coordinate is logged separately and is based on the confirmation pose, not thermal range.
