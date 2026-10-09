# Trained vision model

`models/best.pt` is the uploaded trained YOLO/Ultralytics model.

Validated in Build 0.2:
- Loads with Ultralytics YOLO.
- Model class names: `{0: 'waste'}`.
- Target labels used by the simulator/model-test scripts: `waste`, `bottle`, `plastic`, `plastic_bottle`.

For meaningful model evaluation, add real images under:

```text
assets/test_images/positives/   # images containing target waste/plastic/bottle
assets/test_images/negatives/   # seaweed, glare, wall/edge, empty water, leaves, etc.
```

Then run:

```bash
python scripts/run_model_test_images.py --model models/best.pt --images assets/test_images --output outputs/model_eval --conf 0.25
```
