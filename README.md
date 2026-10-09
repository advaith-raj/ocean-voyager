# Ocean Voyager

**Ocean Voyager** is an experimental autonomous surface vehicle (ASV) for low-cost marine monitoring, with a particular focus on detecting floating macro-plastic debris before it fragments into smaller particles.

The project progressed from a fully simulated autonomous mission, through hardware-in-the-loop (HIL) testing, to a Raspberry Pi-based physical catamaran operating autonomously in a swimming pool.

The current physical prototype integrates:

- Raspberry Pi 4
- RGB camera
- MLX90640 thermal camera
- YOLO-based visual object detection
- dual water-jet propulsion
- DRV8833 motor control
- visual wall/collision detection
- stochastic exploration
- independent RGB and thermal detection
- corroborated RGB/thermal sensor fusion
- bounded target pursuit
- automatic return to exploration
- trajectory, sensor, and event logging

This repository intentionally preserves both the **current implementation** and the **engineering progression that led to it**.

---

## 1. Project Goal

Floating plastic is easier to identify and potentially recover while it remains macro-scale debris.

Ocean Voyager investigates whether a relatively inexpensive autonomous surface vehicle can:

1. autonomously explore a bounded water environment;
2. detect candidate floating debris;
3. combine complementary sensing modalities;
4. investigate confirmed targets;
5. recover from walls and collisions; and
6. continue exploration without continuous human control.

The current system is an experimental research prototype rather than a production marine vehicle.

---

## 2. Development Path

Ocean Voyager was developed progressively:

    simulation
        |
        v
    scenario / robustness testing
        |
        v
    Raspberry Pi execution
        |
        v
    stationary HIL
        |
        v
    RGB and thermal bring-up
        |
        v
    propulsion calibration
        |
        v
    closed-loop pool navigation
        |
        v
    collision / recovery experiments
        |
        v
    asynchronous latest-frame control
        |
        v
    sustained autonomous exploration
        |
        v
    RGB + thermal integration
        |
        v
    corroborated sensor fusion
        |
        v
    stochastic exploration
        |
        v
    v45 physical autonomy

See [`docs/experiments/`](docs/experiments/) for the detailed experimental progression.

---

## 3. Two Software Generations

The repository contains two related but distinct software systems.

### 3.1 Simulation / HIL Platform

Located at:

    simulation/ocean_voyager_sim_hil/

This framework supports combinations of simulated and real components, including:

- fully simulated missions
- configurable environments
- coverage planning
- simulated thermal and RGB sensing
- target detection and approach
- voting and sensor-fusion experiments
- return-to-home behavior
- stationary HIL
- real thermal + simulated vision
- simulated thermal + real camera
- real sensors + log-only motors
- Raspberry Pi hardware adapters
- automated scenario execution
- pytest-based regression testing
- replay and animation tools

The framework preserves the earlier general-purpose architecture used before the physical pool controller became more specialized.

### 3.2 Physical Pool Autonomy

The current physical controller is:

    src/pool/voyager_pool_v45.py

The final lightweight prototype uses:

    Raspberry Pi 4
          |
          v
       DRV8833
       /     \
      v       v
    Pump L   Pump R

Perception is provided by:

    RGB Camera ----+
                   |
                   v
              Fusion Logic ----> Target Behavior
                   ^
                   |
    MLX90640 ------+

The physical system uses experimentally calibrated differential thrust rather than assuming identical left and right propulsion.

---

## 4. Real-Time Architecture

Physical testing exposed an important issue: processing stale camera frames can cause navigation decisions to describe where the boat **was**, rather than where it is now.

The pool software therefore evolved toward an asynchronous latest-frame architecture.

    CAMERA THREAD
          |
          | continuously replace
          v
    LATEST FRAME
          |
          v
    CONTROL / PERCEPTION
          |
          v
    REQUESTED MOTION
          |
          v
    MOTOR EXECUTION

Only the newest useful observation is needed by control; an ever-growing camera backlog is avoided.

See [`docs/architecture/`](docs/architecture/) for more detail.

---

## 5. v45 Stochastic Exploration

v45 replaces rigid repeated search behavior with configurable stochastic navigation.

Each action specification has the form:

    probability,min_duration,max_duration

A representative normal-navigation configuration is:

    --normal-forward "0.70,2.0,12.0"
    --normal-left "0.07,1,6.0"
    --normal-right "0.17,1,6.0"
    --normal-pause "0.06,0.2,4"

This strongly favors meaningful forward traversal while allowing variable turns and pauses.

Collision conditions use a separate distribution:

    --collision-left "0.15,0.5,6.0"
    --collision-right "0.45,1,6.0"
    --collision-pause "0.4,0.5,3.0"

The result is **constrained stochastic exploration**, not unconstrained random motor commands.

A random seed can be supplied when reproducible action sampling is desired.

---

## 6. RGB + Thermal Perception

The two sensing modalities operate independently.

### RGB

The RGB pipeline uses a trained YOLO model to identify candidate debris.

The repository model is:

    models/best.pt

### Thermal

The MLX90640 produces a 24 x 32 thermal array.

Thermal processing searches for temperature anomalies relative to the observed thermal background.

### Fusion

Later controllers compare RGB and thermal evidence rather than allowing every isolated sensor event to interrupt navigation.

Conceptually:

    RGB DETECTION --------+
                          |
                          v
                     CORROBORATE
                          |
                          v
                     FUSED TARGET
                          ^
                          |
    THERMAL DETECTION ----+

Fusion can consider timing and estimated bearing agreement.

This also makes the decision process auditable after a run.

---

## 7. Bounded Target Pursuit

A sufficiently corroborated target can temporarily interrupt normal exploration.

    EXPLORE
       |
       v
    OBSERVE
       |
       v
    RGB / THERMAL EVIDENCE
       |
       v
    CORROBORATE
       |
       v
    TARGET PURSUIT
       |
       | bounded duration
       v
    COOLDOWN
       |
       v
    EXPLORE

This prevents repeated detections of the same target from permanently taking over navigation.

---

## 8. Repository Structure

    ocean-voyager/
    |
    +-- README.md
    +-- .gitignore
    |
    +-- src/
    |   +-- pool/
    |       +-- voyager_pool_v45.py
    |
    +-- simulation/
    |   +-- ocean_voyager_sim_hil/
    |
    +-- scripts/
    |   +-- hardware/
    |   +-- evaluation/
    |   +-- visualization/
    |
    +-- models/
    |   +-- best.pt
    |   +-- MODEL_NOTES.md
    |
    +-- docs/
    |   +-- architecture/
    |   +-- experiments/
    |   +-- figures/
    |
    +-- examples/
    |   +-- pool_run_04/
    |
    +-- archive/
    |   +-- pool_evolution/
    |       +-- milestones/
    |       +-- full_history/
    |
    +-- configs/
    +-- tests/

The historical pool controllers are intentionally separated from the current implementation.

See [`archive/pool_evolution/`](archive/pool_evolution/) for the version history.

---

## 9. Installation

The project was developed primarily with Python on Raspberry Pi 4.

The historical simulation/HIL package includes environment specifications at:

    simulation/ocean_voyager_sim_hil/requirements.txt
    simulation/ocean_voyager_sim_hil/requirements_pi_cpu.txt
    simulation/ocean_voyager_sim_hil/requirements_pi_hil.txt

### Simulation Environment

From the repository root:

    cd simulation/ocean_voyager_sim_hil

    python3 -m venv .venv
    source .venv/bin/activate

    python -m pip install --upgrade pip
    pip install -r requirements.txt

The exact Raspberry Pi environment depends on whether real camera, thermal, GPIO, or inference hardware is being used.

---

## 10. Reproducing the Development Progression

A useful way to explore the repository is to follow approximately the same sequence used during development.

### Stage A — Simulation

    cd simulation/ocean_voyager_sim_hil

Inspect the available scenarios:

    ls scenarios

The simulation package contains scenario runners and configuration files for exercising mission logic without physical hardware.

### Stage B — Regression Tests

Within the simulation/HIL environment:

    pytest -q

The retained test suite covers areas including mission execution, configuration/world behavior, startup, sensor visibility, camera playback, Raspberry Pi camera configuration, and coordinate/pool behavior.

### Stage C — Scenario Suite

Relevant utilities include:

    scripts/run_all_scenarios.py
    scripts/run_build_0_5_pool_suite.py

These exercise larger groups of configurations than individual unit/regression tests.

### Stage D — Stationary HIL

Useful programs include:

    scripts/stationary_real_sensor_hil.py
    scripts/manual_navigation_hil.py

Stationary HIL allows real perception to be exercised before physical propulsion is enabled.

### Stage E — Sensor Bring-Up

Public-facing bring-up utilities are collected in:

    scripts/hardware/

including:

    camera_bringup.py
    thermal_bringup.py
    hardware_bringup.py
    vision_burst_test.py
    combined_thermal_vision_burst_test.py
    thermal_anomaly_test_v2.py

### Stage F — Propulsion Testing

Propulsion utilities include:

    pump_gpio_test.py
    pump_bathtub_test.py
    pump_pool_manual_pulse.py
    pump_pool_sequence_calibration.py
    pump_pool_sequence_sensor_capture.py
    pump_pool_sequence_visual_only.py

The recommended progression is:

    GPIO
      |
      v
    individual pump
      |
      v
    manual pulses
      |
      v
    straight-line calibration
      |
      v
    turn calibration
      |
      v
    repeated pool sequences
      |
      v
    autonomous navigation

Motor timings should be recalibrated for any new physical build.

### Stage G — Physical Controller History

Selected milestone controllers are under:

    archive/pool_evolution/milestones/

The broader surviving source history is under:

    archive/pool_evolution/full_history/

These files show how physical observations changed the algorithms over time.

### Stage H — Current Physical Controller

The final physical controller is:

    src/pool/voyager_pool_v45.py

Before enabling active motors, verify the camera, thermal sensor, motor wiring, propulsion direction, and emergency-stop procedure on the actual platform.

---

## 11. Representative v45 Pool Command

A successful experimental configuration used:

    python src/pool/voyager_pool_v45.py \
      --vision-mode active \
      --thermal-mode active \
      --model models/best.pt \
      --normal-forward "0.70,2.0,12.0" \
      --normal-left "0.07,1,6.0" \
      --normal-right "0.17,1,6.0" \
      --normal-pause "0.06,0.2,4" \
      --collision-left "0.15,0.5,6.0" \
      --collision-right "0.45,1,6.0" \
      --collision-pause "0.4,0.5,3.0" \
      --target-pursuit-max-sec 5 \
      --target-cooldown-sec 30 \
      --archive-rgb-every-frames 2 \
      --output outputs/pool_v45_random_02

These are experimental parameters for one physical prototype and should not be treated as universal calibration values.

Paths must also be adjusted to the local installation.

---

## 12. Physical Demo / Replay

A curated demonstration location is provided at:

    examples/pool_run_04/

The v45 visualization utility is:

    scripts/visualization/render_pool_v45_sensor_animation_v1.py

The replay/renderer is intended to make the autonomous behavior observable after a physical run by combining trajectory, sensor, and target-event information.

---

## 13. Historical Hardware Note

The earlier simulation/HIL platform contains a PCA9685 / T200-style propulsion abstraction.

That code is intentionally preserved because it documents an earlier phase of the architecture.

It is **not** the propulsion system used by the final lightweight physical pool prototype.

The final prototype uses miniature pumps through a DRV8833.

---

## 14. Testing Philosophy

Ocean Voyager uses progressive validation.

Instead of enabling the entire autonomy stack immediately, development moved through:

- simulation
- automated scenarios
- Raspberry Pi execution
- stationary HIL
- thermal-only testing
- RGB-only testing
- combined sensor testing
- log-only motor operation
- manual propulsion calibration
- controlled pool navigation
- collision/recovery testing
- autonomous exploration
- target detection
- multimodal fusion
- bounded target pursuit

This approach helps separate failures in perception, control, electronics, mechanics, and navigation.

---

## 15. Current Experimental Status

The physical pool prototype has demonstrated:

- autonomous movement
- sustained exploration
- visual boundary interaction
- collision/recovery behavior
- RGB target detection
- thermal target detection
- RGB/thermal fusion
- bounded target pursuit
- cooldown
- return to exploration
- trajectory and sensor replay

The swimming pool remains a controlled environment.

Open-water operation introduces additional challenges including:

- wind
- waves
- currents
- sunlight and reflections
- larger localization uncertainty
- waterproofing
- communication range
- power endurance
- marine durability

These remain future development areas.

---

## 16. Research Direction

Ocean Voyager provides an experimental platform for studying:

- autonomous surface robotics
- environmental monitoring
- marine plastic detection
- edge AI
- multimodal perception
- RGB/thermal sensor fusion
- uncertainty-aware target confirmation
- adaptive exploration
- resource-constrained autonomous systems
- multi-vehicle monitoring

Potential future work includes larger-scale testing, improved localization, environmental robustness, cooperative exploration, and deployment in more realistic marine environments.

---

## 17. Documentation

Additional documentation:

- [`docs/architecture/`](docs/architecture/) — software architecture
- [`docs/experiments/`](docs/experiments/) — experimental progression
- [`archive/pool_evolution/`](archive/pool_evolution/) — physical-controller history
- [`examples/pool_run_04/`](examples/pool_run_04/) — curated physical-run example
- `simulation/ocean_voyager_sim_hil/` — simulation/HIL documentation, scenarios, build reports, and tests

---

## Disclaimer

Ocean Voyager is an experimental research prototype.

Hardware configurations, motor timings, thresholds, perception parameters, and navigation settings were developed for a particular prototype and controlled test environment. They should be independently validated and recalibrated before use on different hardware or in different environments.
