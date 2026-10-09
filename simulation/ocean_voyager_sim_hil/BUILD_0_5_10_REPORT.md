# Ocean Voyager Build 0.5.10 Report

Build 0.5.10 is a small hardware-bring-up update layered directly on Build 0.5.9.

## Main purpose

Add first-pass support for the Raspberry Pi AI Camera / IMX500, which is detected
and captured on the Raspberry Pi through `rpicam-hello` / `rpicam-still` rather
than OpenCV `VideoCapture`.

## Added

- `ocean_voyager/adapters/hardware/rpicam_camera_array.py`
  - New `RpicamCameraArray` adapter.
  - Uses `rpicam-still` to capture real CSI camera frames.
  - Health check can run `rpicam-still --list-cameras` and capture a probe frame.
  - Writes frames to `rpicam_frames/` and logs captures through the device bus.
- `config/camera_only_ai_camera_bringup.yaml`
  - Vision-only AI Camera bring-up.
  - One logical camera: `front`.
  - Motors remain log-only.
  - Vision model remains mock for safe hardware validation.
- `config/stationary_ai_camera_mock_vision.yaml`
  - Stationary HIL config using simulated thermal + AI Camera + mock vision + log-only motors.
- `docs/RASPBERRY_PI_AI_CAMERA_BRINGUP.md`
  - Commands for validating `rpicam-hello`, `rpicam-still`, and Ocean Voyager AI Camera capture.

## Changed

- `ocean_voyager/core/factory.py`
  - Recognizes camera source values:
    - `rpicam`
    - `rpicam_still`
    - `ai_camera`
    - `rpicam_ai_camera`
    - `raspberry_pi_ai_camera`
  - These build the new `RpicamCameraArray` adapter.

## Safety

- No motor behavior changed.
- New bring-up configs use `motors: log_only`.
- This build is still sensor-only / stationary validation oriented.

## Recommended Pi command

```bash
cd ~/robot/ocean_voyager_sim_hil_build_0_5_10
source .venv/bin/activate
python scripts/camera_bringup.py \
  --config config/camera_only_ai_camera_bringup.yaml \
  --camera-ids front \
  --samples 5
```

Expected pass condition:

- `frames_ok` equals `5`.
- `frames_failed` equals `0`.
- Startup status is `ready`.
