# Physical Pool Controller Evolution

This directory preserves the experimental development history of the Ocean Voyager physical pool controller.

Two views are provided:

- `milestones/` — selected versions representing important engineering transitions.
- `full_history/` — the surviving historical controller versions.

The historical files are retained for provenance and experimentation. They should not all be interpreted as supported releases.

## Selected Milestones

| Version | Development milestone |
|---|---|
| v5 | Early physical pool-control generation |
| v10 | Closed-loop visual navigation baseline |
| v11 | Refined CLOSE/OPEN visual interpretation and feature-collapse handling |
| v12 | Persistent approach detection and forward-commit behavior |
| v14 | Explicit multi-frame collision signature |
| v15 | Optical-flow / visual-yaw diagnostics |
| v16 | Reverse/recovery capability introduced |
| v17 | Three-thread latest-frame architecture: camera, control, and motor execution |
| v19 | Predictive APPROACH behavior and refined recovery logic |
| v20 | Minimum continuous turn duration |
| v21 | Maximum/bounded turn duration |
| v22 | Explicit STUCK recovery with bounded reverse attempts |
| v27 | Later asynchronous visual-navigation/recovery generation |
| v30 | Physical propulsion and navigation calibration refinement |
| v33 | Sustained-motion physical navigation generation |
| v36 | Unified pool autonomy: exploration sequences, wall recovery, RGB + thermal sensing, target behavior, and replay pose |
| v42 | Late-stage sensor/fusion and target-takeover refinement |
| v43 | Corroborated RGB + thermal target takeover |
| v44 | Auditable RGB/thermal fusion and target pursuit |
| v45 | Stochastic exploration + sensor fusion + bounded target pursuit |

## Major Architectural Progression

The physical software evolved approximately as:

    basic motor control
          |
          v
    visual navigation
          |
          v
    approach / collision detection
          |
          v
    reverse and recovery
          |
          v
    asynchronous latest-frame architecture
          |
          v
    bounded turn / STUCK recovery
          |
          v
    sustained physical exploration
          |
          v
    unified RGB + thermal autonomy
          |
          v
    corroborated sensor fusion
          |
          v
    stochastic exploration
          |
          v
    v45

The final architecture was therefore the result of repeated physical testing and calibration rather than a single controller design.

## Why Preserve Historical Versions?

Real-world pool testing exposed behaviors that were difficult to reproduce perfectly in simulation:

- asymmetric propulsion
- drift
- inertia
- stale camera frames
- reflections and changing visual texture
- wall approaches
- collision recovery
- overly short motor actions
- repeated deterministic movement patterns
- sensor false positives

The historical controllers document how these observations changed the software.

## Historical Gaps

The surviving source history is not numerically continuous.

`voyager_pool_v23.py` was found to consist entirely of NUL bytes in the original Raspberry Pi copy. It was excluded rather than reconstructed.

Versions v24-v26 and v35 were not present in the recovered source set.

No attempt has been made to fabricate missing historical source.

## Current Physical Controller

The current physical pool controller is:

    ../../src/pool/voyager_pool_v45.py

The versions in this directory are primarily intended to document the engineering progression leading to that implementation.
