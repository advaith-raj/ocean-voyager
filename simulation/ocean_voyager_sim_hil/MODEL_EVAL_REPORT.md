# Build 0.3 Model Evaluation Report

This build adds a first real-image evaluation set for the uploaded YOLO model `best.pt`.

## Image set used
- Positives (plastic / bottle / waste): 8 images
- Negatives (seaweed): 7 images
- Negatives (glare / reflections): 5 images
- Total: 20 images

## Class mapping
The uploaded model exposes a single class:
- `0: waste`

The simulator and test harness treat the following labels as target labels:
- `waste`
- `bottle`
- `plastic`
- `plastic_bottle`

## Results by confidence threshold
| Confidence threshold | TP | TN | FP | FN | Accuracy |
|---|---:|---:|---:|---:|---:|
| 0.25 | 6 | 10 | 2 | 2 | 0.80 |
| 0.35 | 6 | 11 | 1 | 2 | 0.85 |
| 0.50 | 5 | 11 | 1 | 3 | 0.80 |
| 0.60 | 4 | 11 | 1 | 4 | 0.75 |
| 0.70 | 4 | 12 | 0 | 4 | 0.80 |

## Recommended current threshold
Based on this small test set, the best current operating point is:
- `confidence_threshold: 0.35`

That is why `config/sim_full_yolo.yaml` now defaults to `0.35`.

## Error cases at 0.35
### False negatives
- `plastic_004.jpg` — bottle missed
- `plastic_006.jpg` — blurred / difficult bottle missed

### False positives
- `seaweed_007.jpg` — seaweed incorrectly detected as `waste` with confidence `0.6219`

## Notes
- This is a small test set and should be treated as an initial calibration, not a final benchmark.
- The model is already behaving reasonably against glare: all 5 glare images were correctly rejected at `0.35`.
- The main remaining issue in this batch is one seaweed false positive and two missed bottle examples.
- Additional far-distance, blurry, partially submerged bottle images and more dense seaweed images will help refine the threshold and guide future retraining.
