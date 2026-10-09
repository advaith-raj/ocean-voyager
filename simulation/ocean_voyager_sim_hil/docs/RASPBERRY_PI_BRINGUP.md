# Raspberry Pi Bring-Up Guide

This guide is for loading Ocean Voyager onto the real Raspberry Pi with sensors attached.

The goal of the first hardware session is not a full autonomous water mission. The goal is to verify that the code can see the peripherals and write useful startup/status files.

---

## Recommended first command

Start with the safe sensors/log-only profile:

```bash
python scripts/hardware_bringup.py \
  --config config/pi_hardware_sensors_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name pi_bringup_day1
```

This checks:

```text
config loading
scenario loading
device bus creation
thermal adapter initialization
camera adapter initialization
vision model initialization
motor safety/log-only state
startup safety checks
```

It does not start a full mission.

---

## Probe sensors

After basic startup works, run:

```bash
python scripts/hardware_bringup.py \
  --config config/pi_hardware_sensors_log_only.yaml \
  --scenario scenarios/36_time_boxed_three_bottles_full_pool_patrol.yaml \
  --run-name pi_bringup_day1_probe \
  --probe
```

This attempts:

```text
one MLX90640 thermal frame
one image from each configured camera
one motor stop/log-only command
```

---

## Files to inspect

Look in:

```text
outputs/hardware_bringup/pi_bringup_day1/
```

Important files:

```text
startup/startup_log.txt
startup/startup_status.json
startup/peripheral_status.json
device_bus/traces/startup_trace.jsonl
summary.json
```

If `--probe` is used, also inspect:

```text
thermal_frames/
real_camera_frames/
```

---

## Interpreting results

### All ready

Good:

```text
thermal: ready
camera_array: ready
motors: ready or log_only
vision_model: ready
safety: ready
```

### Thermal failed

Likely causes:

```text
MLX90640 not wired correctly
I2C not enabled on Raspberry Pi
wrong voltage/power
missing adafruit_mlx90640 library
wrong I2C address / bus issue
```

Check:

```bash
sudo raspi-config
i2cdetect -y 1
```

### Camera failed

Likely causes:

```text
camera not enabled
wrong device number
OpenCV not installed
USB camera not detected
Pi camera needs Picamera2 backend
```

Check:

```bash
ls /dev/video*
python -c "import cv2; print(cv2.__version__)"
```

### Motors degraded/log-only

This is expected for first bring-up.

Do not arm motors until neutral PWM and emergency stop are verified.

---

## Hardware assumptions

Current real adapters are safe first-pass adapters:

| Adapter | Backend |
|---|---|
| RealMLX90640Thermal | Adafruit MLX90640 stack |
| RealCameraArray | OpenCV VideoCapture |
| RealPCA9685MotorExecutor | Adafruit PCA9685, but log-only unless armed |

The camera adapter may need adjustment if using Picamera2 or an Arducam multiplexer.

---

## Installing likely dependencies on Raspberry Pi

Exact packages depend on the Pi image and hardware, but likely candidates are:

```bash
pip install adafruit-circuitpython-mlx90640
pip install adafruit-circuitpython-pca9685
pip install opencv-python
pip install numpy pyyaml pillow
```

For Pi Camera modules, Picamera2 may be installed through apt rather than pip.

---

## Safety checklist before any real motor test

```text
[ ] boat is out of hands / propellers clear
[ ] thrusters are in water if they require water cooling/lubrication
[ ] electronics are protected from water
[ ] battery is secure
[ ] tether or retrieval line attached
[ ] neutral PWM verified
[ ] emergency stop verified
[ ] motors.arm_real_motors intentionally set true
[ ] adult supervision
```

For now, use:

```yaml
sources:
  motors: log_only
```


---

## Build 0.5.9 staged scripts

For a more complete staged workflow, see:

```text
docs/RASPBERRY_PI_STAGED_VALIDATION.md
docs/STATIONARY_REAL_SENSOR_HIL.md
```

Recommended sequence:

```bash
python scripts/run_all_scenarios.py --config config/sim_full.yaml --preset quick --steps 80
python scripts/thermal_bringup.py --config config/thermal_only_bringup.yaml --run-name thermal_day1 --samples 20 --interactive
python scripts/camera_bringup.py --config config/camera_only_bringup.yaml --run-name camera_day1 --samples 3
python scripts/stationary_real_sensor_hil.py --config config/stationary_real_thermal_real_camera_log_only.yaml --run-name stationary_hil_day1 --interactive
```
