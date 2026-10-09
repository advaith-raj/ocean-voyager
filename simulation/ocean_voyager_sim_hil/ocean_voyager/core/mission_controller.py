from __future__ import annotations
from typing import Any

from ocean_voyager.core.types import MissionState, MissionSummary, ObjectStatus, TargetCandidate, distance, project_from_pose
from ocean_voyager.interfaces.camera import CameraArrayInterface
from ocean_voyager.interfaces.motors import MotorExecutorInterface
from ocean_voyager.interfaces.thermal import ThermalInterface
from ocean_voyager.interfaces.vision_model import VisionModelInterface
from ocean_voyager.interfaces.wall_guard import WallGuardInterface
from ocean_voyager.perception.fusion import classify_candidate
from ocean_voyager.perception.thermal_detector import candidates_from_readings
from ocean_voyager.planning.coverage import LawnmowerCoveragePlanner
from ocean_voyager.planning.coverage_grid import CoverageGrid
from ocean_voyager.planning.target_approach import command_toward_bearing, command_toward_candidate, command_toward_point
from ocean_voyager.planning.wall_avoidance import avoidance_command
from ocean_voyager.simulation.world import SimWorld
from ocean_voyager.core.logger import MissionLogger


class MissionController:
    """Mission state machine.

    Build 0.5.3 changes the thermal behavior to hardware-faithful bearing-only:
    thermal can create a direction of interest, but it cannot provide range. During
    approach, Voyager repeatedly captures RGB frames, runs vision, rescans thermal,
    and moves in short steps toward the latest bearing.
    """

    def __init__(
        self,
        world: SimWorld,
        thermal: ThermalInterface,
        camera: CameraArrayInterface,
        motors: MotorExecutorInterface,
        wall_guard: WallGuardInterface,
        vision_model: VisionModelInterface,
        logger: MissionLogger,
        cfg: dict[str, Any],
    ):
        self.world = world
        self.thermal = thermal
        self.camera = camera
        self.motors = motors
        self.wall_guard = wall_guard
        self.vision_model = vision_model
        self.logger = logger
        self.cfg = cfg
        mission_cfg = cfg.get("mission", {})
        self.max_steps = int(mission_cfg.get("max_steps", 350))
        self.max_confirmed = int(mission_cfg.get("max_confirmed", 99))
        self.scan_every_steps = int(mission_cfg.get("scan_every_steps", 4))
        self.wall_guard_enabled = bool(mission_cfg.get("wall_guard_enabled", True))
        self.verify_distance_m = float(mission_cfg.get("verify_distance_m", 0.75))  # retained for legacy coordinate mode
        self.coverage = LawnmowerCoveragePlanner(world.pool, float(mission_cfg.get("lane_spacing_m", 1.0)), float(mission_cfg.get("coverage_safety_offset_m", 0.55)))
        self.coverage_grid = CoverageGrid.from_config(world.pool, cfg)
        self.state = MissionState.INIT
        self.active_candidate: TargetCandidate | None = None
        self.all_candidates: list[TargetCandidate] = []
        self.confirmed = 0
        self.rejected = 0
        self.wall_events = 0
        self.thermal_candidate_events = 0
        self.return_home_enabled = bool(mission_cfg.get("return_home_enabled", False))
        self.return_home_after_time_s = float(mission_cfg.get("return_home_after_time_sec", mission_cfg.get("max_time_sec", 999999)))
        self.home_x = float(mission_cfg.get("home_x", 0.0))
        self.home_y = float(mission_cfg.get("home_y", 0.0))
        self.home_radius_m = float(mission_cfg.get("home_radius_m", 0.35))
        self.returned_home = False
        self._return_home_started = False
        self.hardware_faithful_thermal = bool(mission_cfg.get("hardware_faithful_thermal", True))
        self.thermal_uncertainty_ray_m = float(mission_cfg.get("thermal_uncertainty_ray_m", 2.0))
        self.approach_forward = float(mission_cfg.get("bearing_approach_forward", 0.22))
        self.approach_duration_s = float(mission_cfg.get("bearing_approach_duration_s", 0.45))
        self.max_approach_steps = int(mission_cfg.get("max_approach_steps", 18))
        self.active_approach_steps = 0
        self.required_path_coverage_percent = float(mission_cfg.get("required_path_coverage_percent", 0.0))
        self.cumulative_path_coverage_target_percent = float(mission_cfg.get("cumulative_path_coverage_target_percent", 0.0))
        self.continue_after_confirming_objects = bool(mission_cfg.get("continue_after_confirming_objects", True))
        self.objective = str(mission_cfg.get("objective", "full_pool_survey")).lower()
        self.keep_patrolling_until_time_expires = bool(mission_cfg.get("keep_patrolling_until_time_expires", self.objective == "time_boxed_patrol"))
        self.repeat_lawnmower_after_complete = bool(mission_cfg.get("repeat_lawnmower_after_complete", self.objective == "time_boxed_patrol"))
        self.skip_covered_waypoints = bool(mission_cfg.get("skip_covered_waypoints", self.objective != "time_boxed_patrol"))
        if self.objective == "time_boxed_patrol" and mission_cfg.get("return_home_after_time_sec") is None:
            self.return_home_after_time_s = float(mission_cfg.get("max_time_sec", self.return_home_after_time_s))
        self.coverage_pass_count = 0

    def _start_return_home(self, notes: list[str], reason: str) -> None:
        if self.return_home_enabled and self.state not in {MissionState.RETURN_HOME, MissionState.MISSION_COMPLETE}:
            self.active_candidate = None
            self.state = MissionState.RETURN_HOME
            self._return_home_started = True
            notes.append(reason)
            self.logger.event("return_home_triggered", {"time_s": self.world.time_s, "reason": reason, "home": {"x": self.home_x, "y": self.home_y}})

    def _mark_source_object(self, object_id: str | None, status: ObjectStatus) -> None:
        if object_id:
            obj = self.world.get_object(object_id)
            if obj:
                obj.status = status

    def _mark_processed_object_zone(self, candidate: TargetCandidate, status: ObjectStatus, step: int) -> None:
        # Hardware-faithful: mark processed zone from estimated robot/camera pose, not from
        # hidden ground-truth object coordinates. Ground truth is still shown separately.
        x, y = candidate.estimated_x, candidate.estimated_y
        obj_id = candidate.source_object_id or candidate.id
        self.coverage_grid.mark_object_processed(obj_id, x, y, status.value, step=step, time_s=self.world.time_s)

    def _log_thermal_scan(self, step: int, readings: list, candidates: list[TargetCandidate], min_conf: float) -> None:
        angles = sorted({float(r.pan_angle_deg) for r in readings})
        by_angle = {round(float(c.bearing_deg), 3): c for c in candidates}
        for angle in angles:
            rows = [r for r in readings if abs(float(r.pan_angle_deg) - angle) < 1e-6]
            best = max(rows, key=lambda r: r.confidence) if rows else None
            angle_candidates = [r for r in rows if r.candidate and r.confidence >= min_conf]
            candidate_found = bool(angle_candidates)
            ray_x = ray_y = None
            bearing = None
            blob_pixel_count = 0
            if candidate_found:
                br = max(angle_candidates, key=lambda r: r.confidence)
                bearing = float(br.bearing_deg)
                ray_x, ray_y = project_from_pose(self.world.robot_pose, bearing, self.thermal_uncertainty_ray_m)
                blob_pixel_count = int(getattr(br, "blob_pixel_count", 0))
            self.coverage_grid.mark_thermal_scan(
                self.world.robot_pose,
                pan_angle_deg=angle,
                step=step,
                time_s=self.world.time_s,
                candidate_found=candidate_found,
                max_range_m=float(self.cfg.get("sensors", {}).get("thermal", {}).get("max_range_m", self.cfg.get("sensors", {}).get("thermal", {}).get("max_detection_range_m", 3.5))),
                fov_deg=float(self.cfg.get("sensors", {}).get("thermal", {}).get("fov_deg", self.cfg.get("sensors", {}).get("thermal", {}).get("fov_x_deg", 55.0))),
                best_delta_c=float(best.observed_delta_c) if best else None,
                best_confidence=float(best.confidence) if best else None,
                candidate_x=ray_x,
                candidate_y=ray_y,
            )
            self.logger.thermal_scan({
                "step": step,
                "time_s": self.world.time_s,
                "pose": self.world.robot_pose.as_dict(),
                "pan_angle_deg": angle,
                "candidate_found": candidate_found,
                "thermal_output_mode": "bearing_only_no_range",
                "range_known": False,
                "candidate_bearing_deg": bearing,
                "candidate_ray_x": ray_x,
                "candidate_ray_y": ray_y,
                "candidate_estimated_x": ray_x,
                "candidate_estimated_y": ray_y,
                "blob_pixel_count": blob_pixel_count,
                "best_delta_c": float(best.observed_delta_c) if best else None,
                "best_confidence": float(best.confidence) if best else None,
                "reading_count": len(rows),
                "readings": [r.as_dict() for r in rows],
            })

    def _scan_for_candidates(self, step: int) -> list[TargetCandidate]:
        readings = self.thermal.scan()
        min_conf = float(self.cfg.get("mission", {}).get("thermal_min_confidence", 0.12))
        candidates = candidates_from_readings(
            self.world.robot_pose,
            readings,
            min_conf,
            hardware_faithful=self.hardware_faithful_thermal,
            uncertainty_ray_m=self.thermal_uncertainty_ray_m,
        )
        self._log_thermal_scan(step, readings, candidates, min_conf)
        return candidates

    def _mark_camera_coverage(self, step: int) -> None:
        camera_cfg = self.cfg.get("sensors", {}).get("camera", {})
        mount_angles = camera_cfg.get("mount_angles_deg", {"front": 0.0, "left": 75.0, "right": -75.0})
        fov = float(camera_cfg.get("fov_deg", 70.0))
        rng = float(camera_cfg.get("max_range_m", camera_cfg.get("max_detection_range_m", 3.5)))
        for camera_id, mount in mount_angles.items():
            self.coverage_grid.mark_camera_capture(self.world.robot_pose, str(camera_id), float(mount), fov, rng, step=step, time_s=self.world.time_s)
            self.logger.event("camera_coverage", {
                "step": step,
                "time_s": self.world.time_s,
                "pose": self.world.robot_pose.as_dict(),
                "camera_id": str(camera_id),
                "mount_angle_deg": float(mount),
                "fov_deg": fov,
                "range_m": rng,
            })

    def _capture_and_classify(self, step: int, reason: str) -> tuple[str, Any, list[Any]]:
        frames = self.camera.get_all_frames()
        self._mark_camera_coverage(step)
        detections = []
        for frame in frames.values():
            detections.extend(self.vision_model.predict(frame))
        # Ignore already processed simulated objects; the real robot would use the
        # processed-zone map to avoid re-confirming the same physical target.
        filtered = []
        for det in detections:
            obj_id = getattr(det, "object_id", None)
            obj = self.world.get_object(obj_id) if obj_id else None
            if obj and obj.status in {ObjectStatus.CONFIRMED, ObjectStatus.REJECTED, ObjectStatus.LOST}:
                continue
            filtered.append(det)
        detections = filtered
        if self.active_candidate is None:
            return "lost", None, detections
        target_labels = set(x.lower() for x in self.cfg.get("vision_model", {}).get("target_labels", ["bottle", "plastic", "plastic_bottle", "waste"]))
        outcome, best = classify_candidate(self.active_candidate, detections, bottle_labels=target_labels, min_conf=float(self.cfg.get("mission", {}).get("vision_min_confidence", 0.50)))
        self.logger.detection({
            "step": step,
            "time_s": self.world.time_s,
            "state": self.state.value,
            "reason": reason,
            "candidate": self.active_candidate.as_dict(),
            "detections": [d.as_dict() for d in detections],
            "outcome": outcome,
            "best_detection": best.as_dict() if best else None,
            "note": "RGB was captured during bearing-only approach; thermal did not provide range.",
        })
        return outcome, best, detections

    def _finish_candidate_from_vision(self, step: int, outcome: str, best: Any | None) -> bool:
        if self.active_candidate is None or outcome not in {"confirmed", "rejected"}:
            return False
        # In real hardware, coordinate is estimated from robot pose at the time of confirmation.
        # If a camera model later provides bearing/box-size ranging, this estimate can improve.
        self.active_candidate.source_object_id = getattr(best, "object_id", None) or self.active_candidate.source_object_id
        self.active_candidate.estimated_x = self.world.robot_pose.x
        self.active_candidate.estimated_y = self.world.robot_pose.y
        self.active_candidate.range_known = False
        self.active_candidate.range_source = "vision_confirmed_at_robot_pose_estimate"
        if outcome == "confirmed":
            self.active_candidate.status = ObjectStatus.CONFIRMED
            self._mark_source_object(self.active_candidate.source_object_id, ObjectStatus.CONFIRMED)
            self.confirmed += 1
        else:
            self.active_candidate.status = ObjectStatus.REJECTED
            self._mark_source_object(self.active_candidate.source_object_id, ObjectStatus.REJECTED)
            self.rejected += 1
        self._mark_processed_object_zone(self.active_candidate, self.active_candidate.status, step)
        final_payload = {
            "step": step,
            "time_s": self.world.time_s,
            "state": self.state.value,
            "outcome": outcome,
            "candidate": self.active_candidate.as_dict(),
            "best_detection": best.as_dict() if hasattr(best, "as_dict") else None,
            "ground_truth_object": self.active_candidate.source_object_id,
            "coordinate_source": self.active_candidate.range_source,
            "note": "Final coordinate is estimated from confirmation pose; thermal did not provide distance.",
        }
        self.logger.detection(final_payload)
        self.logger.event("candidate_finished", final_payload)
        return True

    def _physical_coverage_reached(self) -> bool:
        if self.required_path_coverage_percent <= 0:
            return False
        return float(self.coverage_grid.summary().get("path_covered_percent", 0.0)) >= self.required_path_coverage_percent

    def run(self) -> MissionSummary:
        self.state = MissionState.COVERAGE_SWEEP
        notes: list[str] = []
        step = -1
        try:
            for step in range(self.max_steps):
                self.logger.pose(step, self.world.time_s, self.state.value, self.world.robot_pose.as_dict())
                self.coverage_grid.mark_path_covered(self.world.robot_pose, step=step, time_s=self.world.time_s, state=self.state.value)
                if self.return_home_enabled and not self._return_home_started and self.world.time_s >= self.return_home_after_time_s:
                    self._start_return_home(notes, f"return home after {self.return_home_after_time_s:.1f}s mission timer")

                if self.state == MissionState.COVERAGE_SWEEP:
                    if self.objective != "time_boxed_patrol" and self._physical_coverage_reached():
                        if self.return_home_enabled:
                            self._start_return_home(notes, f"required path coverage reached: {self.coverage_grid.summary().get('path_covered_percent')}%")
                        else:
                            self.state = MissionState.MISSION_COMPLETE
                            notes.append("required physical path coverage reached")
                            break
                    elif self.coverage.completed():
                        if self.keep_patrolling_until_time_expires and self.repeat_lawnmower_after_complete and not self._return_home_started:
                            self.coverage_pass_count += 1
                            self.coverage.reset_for_new_pass()
                            self.logger.event("coverage_pass_completed", {
                                "step": step,
                                "time_s": self.world.time_s,
                                "coverage_pass_count": self.coverage_pass_count,
                                "coverage_grid": self.coverage_grid.summary(),
                                "note": "time-boxed patrol repeats lawnmower passes until timer expires",
                            })
                        elif self.return_home_enabled:
                            self._start_return_home(notes, "coverage waypoints completed; returning home")
                        else:
                            self.state = MissionState.MISSION_COMPLETE
                            notes.append("coverage waypoints completed")
                            break
                    if self.state == MissionState.RETURN_HOME:
                        continue
                    if self.wall_guard_enabled:
                        wall = self.wall_guard.check()
                        if wall.front_close or wall.left_close or wall.right_close:
                            self.wall_events += 1
                            cmd = avoidance_command(wall)
                            self.logger.action({"step": step, "time_s": self.world.time_s, "state": self.state.value, "command": cmd.as_dict(), "wall": wall.as_dict()})
                            self.motors.execute(cmd)
                            continue
                    if step % self.scan_every_steps == 0:
                        candidates = self._scan_for_candidates(step)
                        if candidates:
                            self.thermal_candidate_events += len(candidates)
                            self.all_candidates.extend(candidates)
                            self.active_candidate = candidates[0]
                            self.active_candidate.status = ObjectStatus.APPROACHING
                            self.active_approach_steps = 0
                            self.state = MissionState.APPROACH_TARGET
                            self.logger.detection({"step": step, "time_s": self.world.time_s, "state": "THERMAL_CANDIDATE_FOUND", "candidate": self.active_candidate.as_dict(), "note": "bearing-only thermal candidate; no thermal range"})
                            continue
                    cmd = self.coverage.next_command(self.world.robot_pose, coverage_grid=self.coverage_grid, skip_covered=self.skip_covered_waypoints)
                    self.logger.action({"step": step, "time_s": self.world.time_s, "state": self.state.value, "command": cmd.as_dict()})
                    self.motors.execute(cmd)

                elif self.state == MissionState.APPROACH_TARGET:
                    assert self.active_candidate is not None
                    if self.wall_guard_enabled:
                        wall = self.wall_guard.check()
                        if wall.front_close or wall.left_close or wall.right_close:
                            self.wall_events += 1
                            cmd = avoidance_command(wall)
                            self.logger.action({"step": step, "time_s": self.world.time_s, "state": self.state.value, "command": cmd.as_dict(), "wall": wall.as_dict()})
                            self.motors.execute(cmd)
                            self.state = MissionState.RETURN_TO_COVERAGE
                            continue
                    # Take RGB images at every approach step, not only at a guessed verify distance.
                    outcome, best, _detections = self._capture_and_classify(step, reason="approach_step_rgb_check")
                    if self._finish_candidate_from_vision(step, outcome, best):
                        if self.confirmed >= self.max_confirmed and not self.continue_after_confirming_objects:
                            if self.return_home_enabled:
                                self._start_return_home(notes, "max confirmed reached; returning home")
                            else:
                                self.state = MissionState.MISSION_COMPLETE
                                notes.append("max confirmed reached")
                                break
                        else:
                            self.active_candidate = None
                            self.state = MissionState.RETURN_TO_COVERAGE
                        continue
                    # Rescan thermal and update only the bearing/blob confidence.
                    candidates = self._scan_for_candidates(step)
                    if candidates:
                        updated = candidates[0]
                        self.active_candidate.bearing_deg = updated.bearing_deg
                        self.active_candidate.thermal_delta_c = updated.thermal_delta_c
                        self.active_candidate.confidence = updated.confidence
                        self.active_candidate.estimated_x = updated.estimated_x
                        self.active_candidate.estimated_y = updated.estimated_y
                        self.active_candidate.range_known = False
                        self.active_candidate.range_source = "bearing_only_no_range"
                        self.active_candidate.approach_steps += 1
                    else:
                        self.active_candidate.approach_steps += 1
                        self.logger.event("thermal_candidate_temporarily_lost", {"step": step, "time_s": self.world.time_s, "candidate": self.active_candidate.as_dict()})
                    if self.active_candidate.approach_steps >= self.max_approach_steps:
                        self.active_candidate.status = ObjectStatus.LOST
                        self.active_candidate.estimated_x = self.world.robot_pose.x
                        self.active_candidate.estimated_y = self.world.robot_pose.y
                        self._mark_processed_object_zone(self.active_candidate, ObjectStatus.LOST, step)
                        self.logger.event("candidate_lost", {"step": step, "time_s": self.world.time_s, "candidate": self.active_candidate.as_dict(), "reason": "max bearing-approach steps without RGB confirmation"})
                        self.active_candidate = None
                        self.state = MissionState.RETURN_TO_COVERAGE
                        continue
                    cmd = command_toward_bearing(self.active_candidate.bearing_deg, forward=self.approach_forward, duration_s=self.approach_duration_s, action="BEARING_APPROACH", reason="thermal bearing only; step and rescan")
                    self.logger.action({"step": step, "time_s": self.world.time_s, "state": self.state.value, "command": cmd.as_dict(), "candidate": self.active_candidate.as_dict()})
                    self.motors.execute(cmd)

                elif self.state == MissionState.VISION_VERIFY:
                    # Legacy compatibility: in 0.5.3 normal approach verifies every step.
                    outcome, best, _detections = self._capture_and_classify(step, reason="legacy_verify")
                    if self._finish_candidate_from_vision(step, outcome, best):
                        self.active_candidate = None
                    self.state = MissionState.RETURN_TO_COVERAGE

                elif self.state == MissionState.RETURN_TO_COVERAGE:
                    self.state = MissionState.COVERAGE_SWEEP
                    continue

                elif self.state == MissionState.RETURN_HOME:
                    d_home = distance(self.world.robot_pose.x, self.world.robot_pose.y, self.home_x, self.home_y)
                    if d_home <= self.home_radius_m:
                        self.returned_home = True
                        self.state = MissionState.MISSION_COMPLETE
                        notes.append("home reached")
                        self.logger.event("home_reached", {"step": step, "time_s": self.world.time_s, "distance_to_home_m": d_home})
                        break
                    cmd = command_toward_point(self.world.robot_pose, self.home_x, self.home_y, stop_distance_m=self.home_radius_m, action="RETURN_HOME", reason="return to mission origin")
                    self.logger.action({"step": step, "time_s": self.world.time_s, "state": self.state.value, "command": cmd.as_dict(), "home": {"x": self.home_x, "y": self.home_y}, "distance_to_home_m": d_home})
                    self.motors.execute(cmd)

                elif self.state == MissionState.MISSION_COMPLETE:
                    break
            else:
                notes.append("max steps reached")
        finally:
            self.motors.stop()
        final_d_home = distance(self.world.robot_pose.x, self.world.robot_pose.y, self.home_x, self.home_y)
        self.logger.event("mission_metrics", {"objective": self.objective, "coverage_pass_count": self.coverage_pass_count, "coverage_grid": self.coverage_grid.summary()})
        outcome = "success" if (self.confirmed > 0 or self.coverage.completed() or self.returned_home or self._physical_coverage_reached()) else "partial"
        return MissionSummary(
            scenario_name=str(self.cfg.get("scenario_name", "unknown")),
            outcome=outcome,
            steps=step + 1 if step >= 0 else 0,
            objects_confirmed=self.confirmed,
            objects_rejected=self.rejected,
            thermal_candidates=self.thermal_candidate_events,
            wall_avoidance_events=self.wall_events,
            coverage_waypoints_completed=self.coverage.index,
            output_dir=str(self.logger.output_dir),
            notes=notes,
            returned_home=self.returned_home,
            path_length_m=round(self.logger.path_length_m, 4),
            final_distance_to_home_m=round(final_d_home, 4),
            mission_time_s=round(self.world.time_s, 4),
        )
