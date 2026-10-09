# Coordinate Output Patch for Build 0.5

This patch makes object coordinates explicit in every coordinate-pool run.

For each scenario output folder, the simulator now writes:

- `ground_truth_objects.csv` — simple list of every bottle/seaweed/glare coordinate relative to start/home `(0,0)`.
- `object_coordinates.json` — machine-readable coordinate frame plus ground-truth and detection-result coordinates.
- `object_coordinate_log.csv` — ground-truth coordinate, final status, detected/confirmed flag, and estimated coordinate when seen.
- `world_replay.png` — static replay now labels each object with `(x, y)` meters.
- `mission_animation.gif` — animation labels each object with `(x, y)` meters.

Coordinate convention:

- Origin: robot starting position / home coordinate.
- Home: `(0,0)`.
- Units: meters.
- Positive x: initial forward direction.
- Positive y: left/right relative to the chosen scenario coordinate frame.
