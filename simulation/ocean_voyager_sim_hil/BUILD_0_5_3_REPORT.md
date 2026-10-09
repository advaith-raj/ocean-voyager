# Ocean Voyager Build 0.5.3 Report

## Main correction

Build 0.5.3 removes the unrealistic assumption that the thermal sensor provides distance.

The simulated MLX90640-style thermal sensor now returns hardware-faithful outputs:

- 24 x 32 thermal frame file
- pan/servo angle
- thermal blob centroid pixel
- blob pixel count
- thermal max delta
- confidence
- estimated bearing
- `range_known: false`
- `range_source: bearing_only_no_range`

It does **not** return object range or exact object coordinates to the main autonomy loop.

## Updated approach behavior

After thermal detects an area of interest:

1. Voyager creates a bearing-only candidate.
2. Voyager takes RGB pictures at every approach step.
3. YOLO/mock vision runs every approach step.
4. Voyager rescans thermal every approach step.
5. Voyager moves in short bearing-guided increments.
6. If RGB confirms plastic/bottle, the object is marked confirmed.
7. If RGB identifies seaweed/glare, the object is marked rejected.
8. If repeated approach steps do not confirm anything, the candidate is marked lost.

## Coverage correction

Build 0.5.2 counted thermal-scanned cells as explored for waypoint skipping. That was too generous.

Build 0.5.3 separates:

- physical/path coverage
- thermal-scanned coverage
- RGB camera coverage
- processed object zones

Mission completion and waypoint skipping now use physical/path coverage plus processed object zones. Thermal-scanned cells remain important evidence but do not replace physical survey coverage.

## Animation legend

- Blue: physical path coverage
- Yellow: thermal scanned area
- Purple: RGB camera capture coverage
- Orange diamond/ray endpoint: thermal bearing-only candidate indicator
- Green: confirmed waste / processed zone
- Red: rejected false-positive / processed zone
- Black line: Voyager trajectory

## New/updated scenarios

- 28_delayed_bottle_detection
- 29_delayed_bottle_and_seaweed
- 30_object_outside_thermal_range
- 31_object_enters_range_after_coverage_move
- 32_low_delta_becomes_detectable_when_close
- 33_three_bottles_edges_full_survey
- 34_delayed_bottle_full_physical_coverage
- 35_middle_and_edges_full_survey

## Test status

`pytest` result: 9 passed, 1 skipped.

The skipped test is the optional YOLO model-load test unless `RUN_YOLO_LOAD_TEST=1` is set and `ultralytics` is installed.
