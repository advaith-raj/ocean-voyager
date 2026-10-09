# Ocean Voyager Experimental Progression

Ocean Voyager was developed through progressive validation rather than by immediately placing the complete autonomy stack on the water.

The objective was to isolate software, perception, electronics, propulsion, and physical-navigation failures before combining them.

## Phase 1 - Software Simulation

The initial system was developed as a configurable autonomous-mission simulator.

The simulation framework exercised:

- exploration and coverage,
- debris detection,
- target approach,
- sensor behavior,
- voting and confirmation,
- return-to-home behavior,
- failure and retry conditions,
- mission logging.

Simulation made it possible to exercise large numbers of runs before physical hardware was introduced.

## Phase 2 - Scenario and Robustness Testing

A scenario library was developed to test variations in:

- world geometry,
- target placement,
- coverage behavior,
- thermal visibility,
- visual visibility,
- voting thresholds,
- retries,
- false positives,
- sensor failures,
- correlated sensor behavior.

Automated tests and scenario runners were used to catch regressions as the software evolved.

## Phase 3 - Raspberry Pi Execution

The software was moved onto the Raspberry Pi before physical propulsion was enabled.

This allowed verification of:

- Python environment,
- package dependencies,
- configuration loading,
- mission execution,
- Raspberry Pi adapters,
- CPU-only execution,
- sensor interfaces.

## Phase 4 - Stationary Hardware-in-the-Loop

Real sensors were introduced while the vehicle remained stationary.

Configurations included:

    real thermal + simulated vision

    simulated thermal + real camera

    real thermal + real camera

    real sensors + log-only motors

This separated perception problems from propulsion problems.

## Phase 5 - Sensor Bring-Up

The RGB and thermal sensors were tested independently.

RGB testing included:

- camera discovery,
- frame acquisition,
- orientation,
- image capture,
- YOLO inference.

Thermal testing included:

- I2C communication,
- frame acquisition,
- temperature validity,
- thermal visualization,
- anomaly detection.

The sensors were then tested together.

## Phase 6 - Propulsion Bring-Up

The final lightweight prototype uses two miniature water pumps controlled by a DRV8833.

Testing progressed through:

    GPIO test
       |
       v
    individual pump test
       |
       v
    manual pulse test
       |
       v
    forward calibration
       |
       v
    turn calibration
       |
       v
    repeated pool sequences

Physical testing showed that the two propulsion channels were asymmetric.

The controller therefore uses experimentally calibrated timing rather than assuming identical left and right thrust.

## Phase 7 - Structural Iteration

Pool testing also changed the physical vehicle.

The hull/electronics arrangement was revised to improve:

- buoyancy,
- stability,
- electronics protection,
- weight distribution,
- resistance to trapped water and imbalance.

Mechanical changes and software calibration therefore evolved together.

## Phase 8 - Closed-Loop Visual Navigation

The early physical autonomy controllers used visual measurements to alternate between forward movement and scanning/turning.

The software progressively added:

- CLOSE/OPEN scene interpretation,
- feature-collapse handling,
- persistent approach detection,
- forward-commit intervals,
- collision signatures,
- visual-yaw diagnostics.

## Phase 9 - Recovery Behavior

Physical wall interactions required increasingly explicit recovery logic.

Experiments introduced:

- reverse capability,
- minimum turn duration,
- maximum turn duration,
- STUCK detection,
- bounded reverse attempts,
- observation between recovery actions.

These changes reduced repeated short or ineffective correction commands.

## Phase 10 - Latest-Frame Architecture

Physical testing exposed an important real-time problem: stale camera frames.

The software therefore moved to an asynchronous design in which:

- the camera continuously captures,
- only the latest frame is retained,
- control consumes fresh observations,
- motor execution operates independently.

This prevented navigation decisions from being made from an accumulated queue of old images.

## Phase 11 - Sustained Exploration

Short movement pulses often produced behavior that looked hesitant and covered little water.

Later testing increased useful action durations and explored different movement patterns.

The objective shifted from simply demonstrating motor response to producing recognizable autonomous exploration.

## Phase 12 - Unified Pool Autonomy

The later controller generation integrated:

- exploration,
- wall/collision recovery,
- RGB sensing,
- thermal sensing,
- target handling,
- trajectory/event logging,
- approximate replay pose.

This represented the transition from isolated pool-navigation experiments to an integrated autonomous system.

## Phase 13 - Multimodal Detection

RGB and thermal detection were maintained as independent evidence streams.

This allowed the system to distinguish:

- RGB-only detections,
- thermal-only detections,
- corroborated detections.

The later controllers increasingly required corroboration before allowing a candidate target to interrupt normal search.

## Phase 14 - Auditable Fusion

Fusion behavior was instrumented so that target events could be reconstructed after a run.

Logs and archived sensor frames made it possible to examine:

- which modality detected first,
- detection timestamps,
- estimated bearings,
- whether fusion occurred,
- why pursuit was triggered,
- what the vehicle did afterward.

## Phase 15 - Stochastic Exploration

v45 introduced configurable stochastic navigation.

Instead of repeatedly executing one deterministic search sequence, actions are sampled from probability and duration distributions.

Normal exploration and collision conditions have separate distributions.

A representative configuration strongly favors forward movement while allowing occasional turns and pauses.

This creates varied exploration while preserving explicit behavioral bounds.

## Phase 16 - Bounded Target Pursuit

When sufficiently corroborated target evidence is available, exploration can temporarily yield to target pursuit.

Pursuit is time-limited.

After pursuit, a cooldown prevents immediate repeated takeover by the same target.

The controller then returns to autonomous exploration.

The resulting high-level loop is:

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
       v
    COOLDOWN
       |
       v
    EXPLORE

## Current Experimental Status

The physical pool prototype has demonstrated:

- autonomous movement,
- sustained exploration,
- wall interaction and recovery,
- RGB detection,
- thermal detection,
- multimodal fusion,
- bounded target pursuit,
- cooldown,
- return to exploration,
- post-run trajectory and sensor replay.

The pool remains a controlled environment.

Open-water operation introduces additional challenges including:

- wind,
- waves,
- currents,
- sunlight and reflections,
- larger localization error,
- communication range,
- waterproofing,
- power endurance,
- marine durability.

Those remain future development areas.
