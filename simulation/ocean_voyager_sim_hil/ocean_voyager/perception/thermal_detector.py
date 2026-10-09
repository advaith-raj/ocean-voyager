from __future__ import annotations
from itertools import count
from ocean_voyager.core.types import Pose, TargetCandidate, ThermalReading, project_from_pose

_candidate_counter = count(1)


def candidates_from_readings(
    pose: Pose,
    readings: list[ThermalReading],
    min_confidence: float = 0.15,
    hardware_faithful: bool = True,
    uncertainty_ray_m: float = 2.0,
) -> list[TargetCandidate]:
    """Convert thermal blobs to candidates.

    Hardware-faithful mode deliberately does not use thermal range. The real
    MLX90640-style sensor gives a temperature image, not distance, so the
    candidate stores only a bearing. estimated_x/y are just the end of an
    uncertainty ray used for visualization/logging and must not be used as a
    precise target coordinate.
    """
    candidates: list[TargetCandidate] = []
    for r in readings:
        if not r.candidate or r.confidence < min_confidence:
            continue
        if hardware_faithful or not r.range_known or r.distance_m < 0:
            d = -1.0
            ray = float(uncertainty_ray_m)
            ex, ey = project_from_pose(pose, r.bearing_deg, ray)
            range_known = False
            range_source = "bearing_only_no_range"
        else:
            d = r.distance_m
            ex, ey = project_from_pose(pose, r.bearing_deg, r.distance_m)
            range_known = True
            range_source = r.range_source or "simulated_range"
        candidates.append(
            TargetCandidate(
                id=f"cand_{next(_candidate_counter):04d}",
                bearing_deg=r.bearing_deg,
                distance_m=d,
                thermal_delta_c=r.observed_delta_c,
                confidence=r.confidence,
                estimated_x=ex,
                estimated_y=ey,
                source_object_id=r.object_id if not hardware_faithful else None,
                range_known=range_known,
                range_source=range_source,
                first_seen_x=pose.x,
                first_seen_y=pose.y,
            )
        )
    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates
