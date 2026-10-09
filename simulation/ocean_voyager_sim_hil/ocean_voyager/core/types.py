from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import math


class ObjectType(str, Enum):
    BOTTLE = "bottle"
    SEAWEED = "seaweed"
    GLARE = "glare"
    WALL = "wall"
    UNKNOWN = "unknown"


class ObjectStatus(str, Enum):
    UNVISITED = "unvisited"
    CANDIDATE = "candidate"
    APPROACHING = "approaching"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    LOST = "lost"


class MissionState(str, Enum):
    INIT = "INIT"
    COVERAGE_SWEEP = "COVERAGE_SWEEP"
    EDGE_AVOIDANCE = "EDGE_AVOIDANCE"
    THERMAL_CANDIDATE_FOUND = "THERMAL_CANDIDATE_FOUND"
    APPROACH_TARGET = "APPROACH_TARGET"
    VISION_VERIFY = "VISION_VERIFY"
    RETURN_TO_COVERAGE = "RETURN_TO_COVERAGE"
    RETURN_HOME = "RETURN_HOME"
    MISSION_COMPLETE = "MISSION_COMPLETE"
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass
class Pose:
    x: float
    y: float
    heading_deg: float

    def copy(self) -> "Pose":
        return Pose(self.x, self.y, self.heading_deg)

    def as_dict(self) -> Dict[str, float]:
        return {"x": self.x, "y": self.y, "heading_deg": normalize_angle_deg(self.heading_deg)}


@dataclass
class Pool:
    """Rectangle pool in a local mission coordinate frame.

    Build 0.5 supports two coordinate styles:
    - legacy: lower-left is (0, 0), length_m/width_m define the opposite corner
    - mission-relative: x_min/y_min can be negative, so the robot can start at (0, 0)
      even when the pool extends behind or to the side of the start point.
    """

    length_m: float
    width_m: float
    wall_margin_m: float = 0.35
    x_min: float = 0.0
    y_min: float = 0.0

    @property
    def x_max(self) -> float:
        return self.x_min + self.length_m

    @property
    def y_max(self) -> float:
        return self.y_min + self.width_m

    def contains(self, x: float, y: float, margin: Optional[float] = None) -> bool:
        m = self.wall_margin_m if margin is None else margin
        return self.x_min + m <= x <= self.x_max - m and self.y_min + m <= y <= self.y_max - m

    def clamp(self, pose: Pose, margin: Optional[float] = None) -> Pose:
        m = self.wall_margin_m if margin is None else margin
        pose.x = min(max(pose.x, self.x_min + m), self.x_max - m)
        pose.y = min(max(pose.y, self.y_min + m), self.y_max - m)
        pose.heading_deg = normalize_angle_deg(pose.heading_deg)
        return pose

    def as_dict(self) -> Dict[str, float]:
        return {
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "length_m": self.length_m,
            "width_m": self.width_m,
            "wall_margin_m": self.wall_margin_m,
        }


@dataclass
class WorldObject:
    id: str
    type: ObjectType
    x: float
    y: float
    thermal_delta_c: float = 0.0
    radius_m: float = 0.15
    image: Optional[str] = None
    orientation_deg: float = 0.0
    status: ObjectStatus = ObjectStatus.UNVISITED
    expected: str = ""

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        d["status"] = self.status.value
        return d


@dataclass
class ThermalReading:
    # Hardware-faithful fields. A real MLX90640-like sensor does NOT provide range.
    object_id: Optional[str]
    bearing_deg: float                 # relative bearing estimate from pan angle + blob centroid
    distance_m: float                  # -1.0 when range is not known; retained for legacy compatibility
    observed_delta_c: float
    confidence: float
    pan_angle_deg: float
    candidate: bool
    frame_path: Optional[str] = None
    peak_pixel: Optional[Tuple[int, int]] = None
    blob_centroid_pixel: Optional[Tuple[float, float]] = None
    blob_pixel_count: int = 0
    range_known: bool = False
    range_source: str = "bearing_only_no_range"

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CameraVisibleObject:
    object_id: str
    type: ObjectType
    distance_m: float
    bearing_deg: float
    rel_x: float
    rel_y: float
    apparent_size: float

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["type"] = self.type.value
        return d


@dataclass
class CameraFrame:
    camera_id: str
    image_path: Optional[str]
    visible_objects: List[CameraVisibleObject]
    wall_observation: Optional[Dict[str, Any]] = None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "image_path": self.image_path,
            "visible_objects": [o.as_dict() for o in self.visible_objects],
            "wall_observation": self.wall_observation,
        }


@dataclass
class DetectionResult:
    label: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    object_id: Optional[str] = None
    camera_id: Optional[str] = None

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TargetCandidate:
    id: str
    bearing_deg: float
    distance_m: float                  # -1.0 in hardware-faithful bearing-only mode
    thermal_delta_c: float
    confidence: float
    estimated_x: float                 # uncertainty-ray endpoint for visualization only when range_known=False
    estimated_y: float
    source_object_id: Optional[str] = None
    status: ObjectStatus = ObjectStatus.CANDIDATE
    range_known: bool = False
    range_source: str = "bearing_only_no_range"
    approach_steps: int = 0
    first_seen_x: float = 0.0
    first_seen_y: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        return d


@dataclass
class MotionCommand:
    yaw: float
    forward: float
    duration_s: float
    reason: str = ""
    action: str = "MOVE"

    def clamp(self, max_abs: float = 1.0) -> "MotionCommand":
        self.yaw = max(-max_abs, min(max_abs, self.yaw))
        self.forward = max(-max_abs, min(max_abs, self.forward))
        self.duration_s = max(0.0, self.duration_s)
        return self

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class WallObservation:
    front_close: bool
    left_close: bool
    right_close: bool
    closest_side: Optional[str]
    closest_distance_m: Optional[float]
    score: float

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MissionSummary:
    scenario_name: str
    outcome: str
    steps: int
    objects_confirmed: int
    objects_rejected: int
    thermal_candidates: int
    wall_avoidance_events: int
    coverage_waypoints_completed: int
    output_dir: str
    notes: List[str] = field(default_factory=list)
    returned_home: bool = False
    path_length_m: float = 0.0
    final_distance_to_home_m: float = 0.0
    mission_time_s: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


def normalize_angle_deg(angle: float) -> float:
    """Normalize to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def heading_vector(heading_deg: float) -> Tuple[float, float]:
    rad = math.radians(heading_deg)
    return math.cos(rad), math.sin(rad)


def distance(a_x: float, a_y: float, b_x: float, b_y: float) -> float:
    return math.hypot(b_x - a_x, b_y - a_y)


def bearing_from_to(pose: Pose, x: float, y: float) -> float:
    absolute = math.degrees(math.atan2(y - pose.y, x - pose.x))
    return normalize_angle_deg(absolute - pose.heading_deg)


def project_from_pose(pose: Pose, bearing_deg: float, distance_m: float) -> Tuple[float, float]:
    absolute = math.radians(pose.heading_deg + bearing_deg)
    return pose.x + distance_m * math.cos(absolute), pose.y + distance_m * math.sin(absolute)
