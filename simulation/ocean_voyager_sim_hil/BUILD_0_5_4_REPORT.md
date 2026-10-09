# Ocean Voyager Build 0.5.4 Report

Build 0.5.4 adds explicit **time-boxed patrol** behavior and cumulative coverage tracking.

## Major additions

- Mission modes:
  - `find_and_return`
  - `full_pool_survey`
  - `time_boxed_patrol`
- Time-boxed patrol repeats the lawnmower route until the configured mission timer expires.
- Return-home is triggered by time expiration rather than first object or unique coverage completion.
- The coverage grid now tracks both:
  - `path_covered_percent` = unique physical/path coverage, capped at 100%.
  - `cumulative_path_coverage_percent` = repeated coverage/pass metric, can exceed 100%.
- Coverage CSV now includes `path_visit_count` for each cell.
- New scenarios:
  - `36_time_boxed_three_bottles_full_pool_patrol.yaml`
  - `37_time_boxed_four_bottles_plus_seaweed_patrol.yaml`

## Intended demo behavior

Scenario 36 is the clean demo requested: three bottles placed at different pool locations, Voyager follows the lawnmower path, confirms all bottles when encountered, continues patrolling instead of immediately returning, and returns home when the mission time expires.

