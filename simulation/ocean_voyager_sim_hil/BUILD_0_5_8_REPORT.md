# Ocean Voyager Build 0.5.8 Report

Build 0.5.8 focuses on making the repository closer to real Raspberry Pi hardware bring-up.

## Main changes

### 1. README quick-start clarification

The README now explains the difference between:

- basic simulation
- hardware-faithful pool simulation
- model evaluation
- Raspberry Pi hardware bring-up
- tests

It also explains what is common across modes.

### 2. Formal startup/peripheral initialization

Added:

```text
ocean_voyager/core/startup.py
```

Startup sequence:

```text
SYSTEM_BOOT
LOAD_CONFIG
START_DEVICE_BUS
INIT_PERIPHERALS
HEALTH_CHECK
CALIBRATION_CHECK
ARM_MISSION
MISSION_READY
```

Startup outputs:

```text
startup/startup_log.txt
startup/startup_status.json
startup/peripheral_status.json
device_bus/traces/startup_trace.jsonl
```

### 3. Runtime bundle

Updated factory behavior with a `RuntimeAdapters` bundle so startup can inspect all adapters and the device bus before mission execution.

### 4. Safe Raspberry Pi hardware bring-up

Added:

```text
scripts/hardware_bringup.py
config/pi_hardware_sensors_log_only.yaml
```

The first Pi bring-up should use real thermal/camera sources and log-only motors.

### 5. Real adapter bring-up behavior

Updated hardware adapters so they do not fail immediately at construction. They now report readiness through `health_check()`:

```text
RealMLX90640Thermal
RealCameraArray
RealPCA9685MotorExecutor
```

If libraries or devices are missing, startup reports failure/degraded status instead of crashing during import.

### 6. New docs

Added:

```text
docs/STARTUP_AND_PERIPHERAL_INIT.md
docs/RASPBERRY_PI_BRINGUP.md
```

## Safety note

Real motor PWM remains disabled/log-only by default. Do not enable real motor PWM until neutral PWM, emergency stop, waterproofing, and tether/retrieval safety are verified.
