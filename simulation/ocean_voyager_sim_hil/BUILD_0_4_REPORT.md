# Ocean Voyager Build 0.4 Report

## Purpose
Build 0.4 connects real image assets into the stationary HIL pipeline. The robot remains fixed and motors are disabled, but the perception and decision pipeline runs as if the robot were evaluating a thermal candidate in front of it.

## Pipeline

```text
Simulated thermal candidate
  -> image-playback camera frame
  -> YOLO model (`models/best.pt`)
  -> confirm / reject / lost decision
  -> recommended next movement
  -> log-only motor executor
```

## Key files

- `config/stationary_hil_yolo_image_playback.yaml`
- `ocean_voyager/adapters/sim/image_playback_camera.py`
- `scenarios/16_hil_image_bottle_confirm.yaml`
- `scenarios/17_hil_image_seaweed_reject.yaml`
- `scenarios/18_hil_image_glare_reject.yaml`
- `scenarios/19_hil_image_known_failure_cases.yaml`

## Validated runs

### Bottle confirm
Command:

```bash
python main.py --config config/stationary_hil_yolo_image_playback.yaml --scenario scenarios/16_hil_image_bottle_confirm.yaml --run-name hil_bottle_0_4 --steps 3
```

Result: PASS 3 / FAIL 0.

### Seaweed reject
Command:

```bash
python main.py --config config/stationary_hil_yolo_image_playback.yaml --scenario scenarios/17_hil_image_seaweed_reject.yaml --run-name 17_hil_image_seaweed_reject_0_4 --steps 3
```

Result: PASS 3 / FAIL 0.

### Glare reject
Command:

```bash
python main.py --config config/stationary_hil_yolo_image_playback.yaml --scenario scenarios/18_hil_image_glare_reject.yaml --run-name 18_hil_image_glare_reject_0_4 --steps 3
```

Result: PASS 3 / FAIL 0.

### Known failure cases
Command:

```bash
python main.py --config config/stationary_hil_yolo_image_playback.yaml --scenario scenarios/19_hil_image_known_failure_cases.yaml --run-name 19_hil_image_known_failure_cases_0_4 --steps 3
```

Result: PASS 1 / FAIL 2. This is expected because it includes one previously missed bottle image and one previously false-positive seaweed image.

## Output to inspect
Each run creates:

- `hil_step_results.csv`
- `detections.json`
- `recommended_actions.json`
- `summary.json`
- copied playback frames under `frames/`

## Interpretation
This build is now suitable for stationary/living-room perception testing. It does not test physical movement yet. It tests whether the robot would confirm or reject a candidate and what movement it would recommend if motors were enabled.
