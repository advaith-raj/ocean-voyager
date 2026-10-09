# Test images

Place real evaluation images here for Build 0.2 model testing.

Recommended structure:

```text
assets/test_images/
  positives/
    bottle_001.jpg
    bottle_002.jpg
  negatives/
    seaweed_001.jpg
    glare_001.jpg
    wall_001.jpg
    empty_water_001.jpg
```

The included `scripts/run_model_test_images.py` script will evaluate all images in these folders.
Positive images are expected to contain the target object. Negative images are expected not to contain the target object.
