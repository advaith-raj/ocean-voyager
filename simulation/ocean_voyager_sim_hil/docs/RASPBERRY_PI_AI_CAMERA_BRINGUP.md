# Raspberry Pi AI Camera Bring-Up (Build 0.5.10)

Build 0.5.10 adds a `rpicam-still` camera backend for Raspberry Pi CSI cameras,
including the Raspberry Pi AI Camera / IMX500. This is layered on top of Build
0.5.9 and keeps motors log-only.

## Confirm the Pi sees the camera

```bash
rpicam-hello --list-cameras
```

Expected for the AI Camera:

```text
0 : imx500 [4056x3040 10-bit RGGB] (.../imx500@1a)
```

## Confirm raw capture works

```bash
rpicam-still -o ~/ai_camera_test.jpg --timeout 2000 --nopreview
ls -lh ~/ai_camera_test.jpg
```

A nonzero JPEG size means the hardware path is working.

## Ocean Voyager AI Camera bring-up

From the project root:

```bash
source .venv/bin/activate
python scripts/camera_bringup.py \
  --config config/camera_only_ai_camera_bringup.yaml \
  --camera-ids front \
  --samples 5
```

Check:

```bash
cat outputs/hardware_bringup/camera_bringup/summary.json
cat outputs/hardware_bringup/camera_bringup/camera_frames.csv
ls -lh outputs/hardware_bringup/camera_bringup/rpicam_frames
```

Expected summary:

```text
frames_ok = 5
frames_failed = 0
startup.overall_status = ready
```

## Why this build is different from 0.5.9

Build 0.5.9's real camera adapter uses OpenCV `VideoCapture` and `/dev/video*`.
The Raspberry Pi AI Camera was validated through the Raspberry Pi `rpicam` /
libcamera stack, so 0.5.10 adds `RpicamCameraArray`, which captures images using
`rpicam-still` instead of OpenCV.

## Supported first-pass setup

- One physical AI Camera.
- Logical camera id: `front`.
- Motors: log-only.
- Vision model: mock by default in the bring-up config.

After this passes, the next upgrade can connect these captured images to the
trained YOLO verifier or an IMX500 on-camera model pipeline.
