# Pool Controller Dependencies

The v45 physical controller uses both ordinary Python packages and
Raspberry Pi hardware/software interfaces.

## Base Python Packages

Install:

    pip install -r requirements/pool_base.txt

This includes:

- NumPy
- Ultralytics YOLO

The trained model is:

    models/best.pt

## Raspberry Pi Packages

Install the Raspberry Pi hardware dependencies:

    pip install -r requirements/pool_pi.txt

The working experimental Raspberry Pi environment included:

- gpiozero 2.0.1
- Picamera2 0.3.36
- OpenCV 4.10.0
- NumPy 2.2.4
- Adafruit Blinka 9.1.0
- adafruit-circuitpython-mlx90640 1.3.9
- Ultralytics 8.4.95

These versions document the known working environment. They are not all
strict version requirements.

## Picamera2 and OpenCV

On Raspberry Pi OS, Picamera2 and OpenCV may be supplied through the
operating system rather than installed into an isolated pip environment.

Verify them with:

    python3 -c "import cv2; print('OpenCV', cv2.__version__)"
    python3 -c "from picamera2 import Picamera2; print('Picamera2 OK')"

## YOLO

Verify Ultralytics with:

    python3 -c "from ultralytics import YOLO; print('Ultralytics OK')"

Then verify the model:

    python3 - <<'PY'
    from ultralytics import YOLO
    model = YOLO("models/best.pt")
    print("Model loaded")
    print("Classes:", model.names)
    PY

The expected model class in the archived working configuration is:

    {0: 'waste'}

## Thermal Sensor

Verify the MLX90640 Python library with:

    python3 -c "import adafruit_mlx90640; print('MLX90640 library OK')"

I2C must also be enabled on the Raspberry Pi.

## GPIO

Verify gpiozero with:

    python3 -c "from gpiozero import DigitalOutputDevice; print('gpiozero OK')"

Do not enable propulsion merely to test whether the Python package imports.

## Historical Environment Freeze

The simulation/HIL directory contains historical Raspberry Pi environment
freezes:

    simulation/ocean_voyager_sim_hil/requirements_pi_cpu.txt
    simulation/ocean_voyager_sim_hil/requirements_pi_freeze.txt

Those files preserve the experimental machine environment and contain many
packages unrelated to Ocean Voyager itself. They should not be treated as
the recommended installation list for a new system.
