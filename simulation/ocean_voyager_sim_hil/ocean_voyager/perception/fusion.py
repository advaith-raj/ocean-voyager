from __future__ import annotations
from ocean_voyager.core.types import DetectionResult, ObjectType, TargetCandidate


def classify_candidate(candidate: TargetCandidate, detections: list[DetectionResult], bottle_labels: set[str] | None = None, reject_labels: set[str] | None = None, min_conf: float = 0.50) -> tuple[str, DetectionResult | None]:
    bottle_labels = bottle_labels or {"bottle", "plastic_bottle", "plastic"}
    reject_labels = reject_labels or {"seaweed", "glare", "wall", "unknown"}
    if not detections:
        return "lost", None
    best = max(detections, key=lambda d: d.confidence)
    label = best.label.lower()
    if best.confidence < min_conf:
        return "uncertain", best
    if label in bottle_labels:
        return "confirmed", best
    if label in reject_labels:
        return "rejected", best
    return "uncertain", best
