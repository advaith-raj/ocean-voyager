# File-Based Device Bus

The device bus is a debugging and integration layer that logs commands between the main CPU logic and peripheral-like services.

---

## Purpose

The bus makes simulation and hardware integration cleaner.

Instead of tightly coupling the main program to a sensor or actuator implementation, the main program can write a command and receive a response.

Conceptually:

```text
Main autonomy program
  → command file / bus trace
Peripheral process, simulated or real
  → response file / bus trace
Main autonomy program
  → continues mission logic
```

---

## Why this matters

It makes it easier to compare:

```text
same command in simulation
same command on real hardware
```

Example comparison:

| Command | Simulation response | Real response |
|---|---|---|
| servo set angle 30° | settled in 5 ms | settled in 280 ms |
| thermal capture | candidate bearing +18° | candidate bearing +22° |
| camera capture | frame path available | frame path available |
| motor PWM 1580/1620 for 0.8 s | moved 0.28 m | measured/estimated 0.20 m |

This lets us calibrate the simulator.

---

## Typical command schema

```json
{
  "request_id": "motor_000142",
  "tick": 381,
  "time_sec": 95.25,
  "device": "motors",
  "command": "set_pwm_for_duration",
  "params": {
    "left_pwm_us": 1580,
    "right_pwm_us": 1645,
    "duration_sec": 0.8,
    "reason": "approach candidate"
  }
}
```

Typical response:

```json
{
  "request_id": "motor_000142",
  "tick": 381,
  "time_sec": 96.05,
  "device": "motors",
  "status": "ok",
  "latency_ms": 32,
  "payload": {
    "executed": "simulated",
    "distance_m": 0.27,
    "avg_speed_mps": 0.34
  }
}
```

---

## Bus trace output

Each run can produce:

```text
device_bus/traces/bus_trace.jsonl
```

Every command/response should include:

```text
request_id
tick
time_sec
direction
device
command
params
status
payload
latency_ms
run_mode
```

---

## Simulation-to-real swap

Simulation mode:

```text
main program writes motor command
sim motor process estimates movement
sim motor process writes response
```

Real hardware mode:

```text
main program writes motor command
real motor service sends PWM to PCA9685/ESC
real motor service writes response
```

The main mission controller should not need major changes.

---

## Safety hooks for real motors

The real motor service should include:

```text
neutral PWM on startup
clamp PWM limits
timeout-to-neutral
emergency_stop.flag
log every PWM command
reject commands outside safe bounds
```

For real pool tests, motors should initially run in log-only mode.
