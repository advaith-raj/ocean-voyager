# Startup and Peripheral Initialization

Build 0.5.8 adds a formal startup/readiness sequence so Ocean Voyager behaves more like a real robotics system.

The mission should not begin immediately after loading a config. It should first bring up the bus, create peripherals, run health checks, verify safety assumptions, and only then arm the mission.

---

## Startup sequence

The startup manager follows this sequence:

```text
SYSTEM_BOOT
↓
LOAD_CONFIG
↓
START_DEVICE_BUS
↓
INIT_PERIPHERALS
↓
HEALTH_CHECK
↓
CALIBRATION_CHECK
↓
ARM_MISSION
↓
MISSION_READY
```

If a required device fails, the mission is not armed unless the config explicitly allows degraded mode.

---

## Devices checked

The startup manager checks these devices:

```text
pan_servo
thermal
camera_array
vision_model
motors
wall_guard
safety
```

Each device reports a standard status object:

```json
{
  "device": "thermal",
  "adapter": "RealMLX90640Thermal",
  "mode": "stationary_hil",
  "source": "real_mlx90640",
  "required": true,
  "status": "ready",
  "initialized": true,
  "self_test": "pass",
  "capabilities": {
    "sensor": "MLX90640",
    "frame_shape": [24, 32],
    "bearing_only": true,
    "range_known": false
  },
  "last_error": null,
  "warnings": []
}
```

---

## Status meanings

```text
ready
  Device initialized and self-test passed.

degraded
  Device is usable in a limited/safe mode.
  Example: real motor adapter loaded but motors are not armed.

failed
  Required device is not ready. Mission should not start.

skipped
  Optional device was not checked or not required.
```

---

## Output files

Every normal run and hardware bring-up run can produce:

```text
startup/startup_log.txt
startup/startup_status.json
startup/peripheral_status.json
device_bus/traces/startup_trace.jsonl
```

### startup_log.txt

Human-readable boot log.

Example:

```text
[BOOT] SYSTEM_BOOT
[BUS] START_DEVICE_BUS
[PERIPHERALS] INIT_PERIPHERALS
[HEALTH] HEALTH_CHECK
[THERMAL] READY adapter=SimThermalSensor required=True self_test=pass
[CAMERA_ARRAY] READY adapter=SimCameraArray required=True self_test=pass
[MOTORS] READY adapter=SimMotorExecutor required=True self_test=pass
[MISSION] ARM_MISSION
[MISSION] MISSION_READY
```

### peripheral_status.json

Machine-readable list of device statuses.

### startup_status.json

Overall mission arming result.

### startup_trace.jsonl

One JSON record per startup event. Useful for comparing simulation and hardware startup behavior.

---

## Startup config

Example:

```yaml
startup:
  enabled: true
  abort_on_failure: true
  allow_degraded_mode: false
  require_devices:
    - thermal
    - camera_array
    - motors
    - vision_model
    - safety
  require_emergency_stop_clear: true
  emergency_stop_clear: true
```

### abort_on_failure

If true, mission stops before running when a required device fails.

### allow_degraded_mode

If true, degraded required devices can still allow the run to proceed. This is useful for special diagnostics, but not recommended for real water tests.

### require_devices

List of devices that must be ready.

### emergency_stop_clear

Software flag used by the startup safety check. A real hardware emergency-stop input can be wired into this later.

---

## Command lifecycle

After startup, device commands follow this conceptual lifecycle:

```text
command_created
↓
command_sent
↓
device_acknowledged
↓
device_executed
↓
response_received
↓
logged
```

The file bus records this in:

```text
device_bus/commands/processed/
device_bus/responses/processed/
device_bus/traces/bus_trace.jsonl
```

---

## Real-hardware safety rule

For Raspberry Pi hardware bring-up:

```text
real sensors are okay to probe
real motors should remain log-only until:
  neutral PWM is verified
  emergency stop is verified
  thrusters are in water
  electronics are protected from water
  the boat is tethered or retrievable
```

The safe bring-up config uses:

```yaml
sources:
  motors: log_only
```

and the motor config uses:

```yaml
motors:
  arm_real_motors: false
```
