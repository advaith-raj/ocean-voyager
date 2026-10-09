# Ocean Voyager Software Architecture

Ocean Voyager contains two related but distinct software generations:

1. the general simulation / hardware-in-the-loop platform; and
2. the later physical pool-autonomy controller.

This distinction is important because the physical prototype evolved substantially as real-world propulsion, camera latency, drift, wall interaction, and sensor behavior were characterized.

## 1. Simulation / HIL Architecture

The simulation/HIL platform was designed around replaceable adapters so simulated and real components could be mixed.

Conceptually:

    Mission Controller
           |
           +---- Coverage / Navigation
           |
           +---- Perception
           |       |
           |       +---- Thermal
           |       +---- RGB
           |       +---- Fusion
           |
           +---- Target Approach
           |
           +---- Hardware Abstraction
                   |
                   +---- Simulated adapters
                   +---- Raspberry Pi adapters

This allowed configurations such as:

    simulated thermal + simulated camera + simulated motors

    real thermal + simulated camera + log-only motors

    simulated thermal + real camera + log-only motors

    real thermal + real camera + log-only motors

The implementation is preserved under:

    simulation/ocean_voyager_sim_hil/

The framework also contains scenarios, tests, configuration files, Raspberry Pi adapters, replay utilities, and historical build documentation.

## 2. Physical Pool Architecture

The later pool controller became a specialized real-time autonomy program for the Raspberry Pi prototype.

A major architectural change was the adoption of an asynchronous latest-frame design.

Conceptually:

    CAMERA THREAD
          |
          | continuously replace latest frame
          v
    LATEST FRAME SLOT
          |
          v
    CONTROL / PERCEPTION
          |
          v
    REQUESTED MOTION
          |
          v
    MOTOR EXECUTION

The important property is that control operates on the newest available image rather than processing an ever-growing queue of stale camera frames.

This became particularly important once physical motor actions were long enough for the scene to change significantly between observations.

## 3. Navigation State Logic

The physical controllers evolved through states representing behaviors such as:

    FORWARD_SCAN
         |
         v
    TURN_SCAN
         |
         v
    APPROACH / COLLISION
         |
         v
    RECOVERY
         |
         v
    resume exploration

Later controllers moved away from relying exclusively on deterministic state sequences and introduced higher-level exploration actions and stochastic selection.

## 4. RGB Perception

The physical system uses an RGB camera for two distinct purposes:

1. navigation / wall-interaction information; and
2. target recognition.

Target recognition uses a trained YOLO model.

The final repository model is stored at:

    models/best.pt

RGB detections can produce:

- target confidence,
- target location,
- approximate bearing,
- timestamped detection evidence.

## 5. Thermal Perception

The MLX90640 provides a 24 x 32 thermal array.

Thermal processing searches for temperature anomalies relative to the observed thermal background.

Thermal evidence is maintained independently from RGB evidence so that the two modalities can later be compared rather than treating one as a substitute for the other.

## 6. Sensor Fusion

Later physical controllers combine RGB and thermal observations.

Conceptually:

    RGB evidence --------+
                         |
                         v
                    FUSION LOGIC ----> fused target
                         ^
                         |
    THERMAL evidence ----+

Corroboration can consider factors such as:

- temporal proximity,
- estimated target bearing,
- detection validity,
- configurable fusion thresholds.

This allows target pursuit to be triggered by stronger multimodal evidence rather than every individual sensor event.

## 7. Target Pursuit

A confirmed target can temporarily interrupt normal exploration.

Conceptually:

    EXPLORE
       |
       v
    sensor evidence
       |
       v
    corroboration
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

Both pursuit duration and cooldown are bounded.

This prevents repeated detections of one target from permanently taking control of the mission.

## 8. Stochastic Exploration

v45 introduces configurable stochastic navigation.

Each navigation option is represented as:

    probability,min_duration,max_duration

Normal exploration and collision/recovery conditions use separate distributions.

This produces behavior that is variable without being unconstrained.

For example, a forward-biased distribution can encourage meaningful traversal while occasional turns alter the search trajectory.

A random seed can be supplied when repeatable behavior is required.

## 9. Motor Layer

The final physical prototype uses:

    Raspberry Pi GPIO
            |
            v
         DRV8833
        /       \
       v         v
    left pump  right pump

The two miniature pumps do not produce identical thrust.

The controller therefore uses experimentally calibrated timing rather than assuming perfectly symmetric motors.

## 10. Historical Hardware Abstraction

The earlier simulation/HIL framework contains support for a PCA9685 / T200-style propulsion abstraction.

That code is intentionally retained as part of the project's development history.

It should not be interpreted as the propulsion architecture of the final lightweight physical prototype.

## 11. Current Implementation

The current physical controller is:

    src/pool/voyager_pool_v45.py

The general simulation/HIL implementation is:

    simulation/ocean_voyager_sim_hil/

The historical physical-controller progression is documented in:

    archive/pool_evolution/
