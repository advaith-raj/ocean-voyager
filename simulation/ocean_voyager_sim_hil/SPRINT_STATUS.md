# Sprint Status — Build 0.2

## Added in Build 0.2

- Copied uploaded trained model to `models/best.pt`.
- Confirmed model loads with Ultralytics YOLO.
- Confirmed model class names: `{0: 'waste'}`.
- Added `config/sim_full_yolo.yaml`.
- Added `scripts/run_model_test_images.py` for positive/negative model evaluation.
- Added `assets/test_images/positives/` and `assets/test_images/negatives/` folders with README instructions.
- Updated `YoloVisionModel` to accept target labels.
- Updated mission classifier to treat `waste` as a target label when configured.
- Added YOLO model load unit test.

## Validated here

```bash
python -m pytest -q
# 7 passed

python scripts/run_model_test_images.py --model models/best.pt --images assets/test_images --output outputs/model_eval_empty --conf 0.25
# model loads; 0 images evaluated because no real test images are bundled yet

python main.py --config config/sim_full_yolo.yaml --scenario scenarios/02_single_bottle_center.yaml --run-name yolo_smoke_02
# run completes; synthetic frames are not expected to provide meaningful detection accuracy
```

## Next build recommendation

Build 0.3 should add real image assets:

- bottle/plastic positive images
- seaweed negative images
- glare/reflection negative images
- pool wall/edge negative images
- empty water negative images

Then run:

```bash
python scripts/run_model_test_images.py --model models/best.pt --images assets/test_images --output outputs/model_eval_real --conf 0.25
```

Based on false positives/false negatives, tune:

- `confidence_threshold`
- target labels
- verification logic
- low-confidence retry/reject behavior


## Build 0.4 status

Added image-playback stationary HIL mode:

- New config: `config/stationary_hil_yolo_image_playback.yaml`
- New camera adapter: `ImagePlaybackCameraArray`
- New scenarios:
  - `16_hil_image_bottle_confirm.yaml`
  - `17_hil_image_seaweed_reject.yaml`
  - `18_hil_image_glare_reject.yaml`
  - `19_hil_image_known_failure_cases.yaml`
- New output: `hil_step_results.csv`
- Motors remain disabled/log-only.

Validation runs completed:

| Scenario | Expected | Result |
|---|---|---|
| bottle confirm | 3 target frames confirmed | PASS 3 / FAIL 0 |
| seaweed reject | 3 non-target frames rejected/not confirmed | PASS 3 / FAIL 0 |
| glare reject | 3 non-target frames rejected/not confirmed | PASS 3 / FAIL 0 |
| known failure cases | intentionally includes hard FN/FP examples | PASS 1 / FAIL 2 |

The known failure scenario is deliberately included for future threshold tuning/retraining.
