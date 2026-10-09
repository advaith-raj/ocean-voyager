# Simulation vs Real Hardware

Ocean Voyager is built so simulation and real hardware use the same software contracts wherever possible.

---

## What is the same?

The mission controller always interacts with interfaces:

```text
thermal.scan()
camera.get_frame()
vision_model.predict()
motors.execute()
wall_guard.check()
```

So the high-level autonomy logic can remain stable while sensor/actuator implementations change.

---

## What changes between simulation and hardware?

| Subsystem | Simulation | Real hardware |
|---|---|---|
| Thermal | MLX90640-like frame model and bearing extraction | MLX90640 32×24 readings over I2C |
| RGB camera | synthetic render, image playback, or simulated frame | Pi/USB/Arducam camera frame |
| Vision | Mock model or YOLO model on image paths | YOLO model on real captured frames |
| Motors | PWM-level motion model | PCA9685 PWM → ESC → thrusters |
| Position | True simulated pose | Estimated pose / dead reckoning unless external localization is added |
| Object coordinates | Ground truth exists in scenario | Estimated at confirmation time |
| Wall guard | Known pool geometry | camera/vision/distance-sensor logic needed |
| Coverage | exact from simulation pose | estimated from dead-reckoned/observed pose |

---

## Thermal limitation

The real thermal sensor does not provide distance.

It provides a low-resolution thermal frame. The software can estimate:

```text
bearing
blob centroid
blob size
thermal contrast
confidence
```

It should not assume:

```text
distance to object
exact object coordinate
```

Therefore the simulator should not cheat by giving exact distance to the autonomy layer.

---

## Coordinate truthfulness

Simulation can log both:

```text
ground_truth_x, ground_truth_y
estimated_x, estimated_y
```

Real hardware should only claim estimated coordinates unless external localization is available.

Recommended terminology:

```text
ground_truth_coordinate        available in simulation only
thermal_bearing_estimate       from thermal blob centroid + servo/pan angle
vision_confirmed_estimate      from robot pose + visual bearing / close approach
dead_reckoned_pose             from motor commands and motion calibration
```

---

## Real pool localization

A regular pool is a good controlled testing environment, but accurate return-to-home requires localization.

Possible approaches:

```text
1. Dead reckoning from motor commands
2. IMU/compass heading correction
3. Overhead phone/camera tracking
4. AprilTag markers around pool
5. Visual wall/boundary localization
```

Build 0.5.7 is ready for dead-reckoning-style estimated pose, but real-world drift must be measured.

---

## Why keep simulation?

Simulation is still valuable because it tests:

- mission logic
- state transitions
- coverage behavior
- sensor/actuator command flow
- object logging
- return-home behavior
- failure cases
- animation/replay outputs

But calibrated hardware tests are required before claiming real performance.
