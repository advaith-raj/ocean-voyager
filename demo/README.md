# Physical Pool Demo

This directory is reserved for a curated example of a successful Ocean Voyager v45 physical pool run.

The purpose of the example is to show the complete autonomy loop rather than preserve the entire raw experimental dataset.

## Demonstrated Behaviors

The run demonstrates interaction between:

- stochastic autonomous exploration
- wall/collision handling
- RGB target detection
- MLX90640 thermal detection
- RGB/thermal corroboration
- bounded target pursuit
- target cooldown
- return to autonomous exploration
- post-run trajectory and sensor reconstruction

## Controller Configuration

The v45 configuration used for this experiment was:

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

Paths should be changed for other installations.

## Navigation Distribution

Each stochastic navigation specification uses:

    probability,min_duration,max_duration

For example:

    --normal-forward "0.70,2.0,12.0"

means that FORWARD has probability 0.70 in the normal-action distribution and, when selected, its duration is sampled between 2 and 12 seconds.

Normal and collision conditions use separate action distributions.

This allows collision behavior to be biased differently from normal exploration.

## Target Behavior

The high-level target sequence is:

    EXPLORE
       |
       v
    RGB / thermal observations
       |
       v
    corroborated evidence
       |
       v
    TARGET PURSUIT
       |
       | maximum 5 seconds
       v
    COOLDOWN
       |
       | 30 seconds
       v
    EXPLORE

This prevents a detected target from permanently taking over the mission.

## Archived RGB Frames

The option:

    --archive-rgb-every-frames 2

archives RGB observations at a regular interval so that navigation and target events can be inspected after the run.

## Curated Evidence

Only a small representative subset of the physical-run data should be stored in GitHub.

Suggested contents:

    README.md

    events.csv
    trajectory.csv

    representative_rgb/
    representative_thermal/
    fusion_events/

    demo.mp4

Large raw sensor datasets and routine experimental captures should remain outside the Git repository.

## Replaying the Run

The visualization utility is located at:

    scripts/visualization/render_pool_v45_sensor_animation_v1.py

The renderer can combine navigation, trajectory, RGB, thermal, and fusion information into a synchronized demonstration of what the autonomous system was doing during the physical run.

## Interpretation

This run is evidence of operation in a controlled swimming-pool environment.

It should not be interpreted as evidence of open-ocean readiness.

The value of the experiment is that the complete software/sensor/propulsion loop was exercised on the physical platform rather than only in simulation.
