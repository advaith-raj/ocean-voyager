from __future__ import annotations

from pathlib import Path
from typing import Any
import csv

from ocean_voyager.core.logger import MissionLogger
from ocean_voyager.core.types import MotionCommand
from ocean_voyager.interfaces.camera import CameraArrayInterface
from ocean_voyager.interfaces.thermal import ThermalInterface
from ocean_voyager.interfaces.vision_model import VisionModelInterface
from ocean_voyager.perception.fusion import classify_candidate
from ocean_voyager.perception.thermal_detector import candidates_from_readings
from ocean_voyager.planning.target_approach import command_toward_candidate
from ocean_voyager.simulation.world import SimWorld


class StationaryHILRunner:
    """Perception-only runner. It computes next movement but never executes motors.

    This supports both pure simulation and image-playback HIL. In image-playback
    mode, real image assets are fed to the vision model while the robot remains
    stationary and motor execution is disabled/log-only.
    """

    def __init__(self, world: SimWorld, thermal: ThermalInterface, camera: CameraArrayInterface, vision_model: VisionModelInterface, logger: MissionLogger, cfg: dict[str, Any]):
        self.world = world
        self.thermal = thermal
        self.camera = camera
        self.vision_model = vision_model
        self.logger = logger
        self.cfg = cfg
        self.step_results: list[dict[str, Any]] = []
        self.target_labels = set(x.lower() for x in cfg.get("vision_model", {}).get("target_labels", ["bottle", "plastic", "plastic_bottle", "waste"]))

    @staticmethod
    def _expected_from_frames(frames: dict) -> dict[str, Any]:
        for frame in frames.values():
            meta = frame.wall_observation or {}
            if meta.get("source_image") or meta.get("instruction") or meta.get("expected"):
                return meta
        return {"expected": "unknown"}

    @staticmethod
    def _step_pass_fail(expected: str, outcome: str) -> str:
        expected = (expected or "unknown").lower()
        if expected in {"target", "positive", "bottle", "plastic", "waste"}:
            return "PASS" if outcome == "confirmed" else "FAIL"
        if expected in {"non_target", "negative", "seaweed", "glare", "empty", "wall"}:
            return "PASS" if outcome != "confirmed" else "FAIL"
        return "NA"

    def _write_step_results_csv(self) -> None:
        path = self.logger.output_dir / "hil_step_results.csv"
        fieldnames = [
            "step",
            "expected",
            "result",
            "outcome",
            "best_label",
            "best_confidence",
            "recommended_action",
            "yaw",
            "forward",
            "duration_s",
            "source_image",
            "instruction",
        ]
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.step_results:
                writer.writerow({k: row.get(k, "") for k in fieldnames})

    def run(self, steps: int = 5) -> dict[str, Any]:
        for step in range(steps):
            self.logger.pose(step, self.world.time_s, "STATIONARY_HIL", self.world.robot_pose.as_dict())
            readings = self.thermal.scan()
            candidates = candidates_from_readings(self.world.robot_pose, readings, self.cfg.get("mission", {}).get("thermal_min_confidence", 0.12))
            frames = self.camera.get_all_frames()
            frame_meta = self._expected_from_frames(frames)
            detections = []
            for frame in frames.values():
                detections.extend(self.vision_model.predict(frame))
            selected = candidates[0] if candidates else None
            if selected:
                outcome, best = classify_candidate(
                    selected,
                    detections,
                    bottle_labels=self.target_labels,
                    min_conf=self.cfg.get("mission", {}).get("vision_min_confidence", 0.50),
                )
                if outcome == "confirmed":
                    command = command_toward_candidate(self.world.robot_pose, selected, self.cfg.get("mission", {}).get("verify_distance_m", 0.75))
                elif outcome == "uncertain":
                    command = MotionCommand(0, 0, 0, action="HOLD_RESCAN", reason="thermal candidate but vision confidence is uncertain")
                elif outcome == "rejected":
                    command = MotionCommand(0, 0, 0, action="REJECT_TARGET", reason="vision classified candidate as non-target")
                else:
                    command = MotionCommand(0, 0, 0, action="NO_VISUAL_CONFIRMATION", reason="thermal candidate was not confirmed by vision")
            else:
                outcome, best = "no_candidate", None
                command = MotionCommand(0, 0, 0, action="NO_TARGET", reason="no thermal candidate")
            expected = str(frame_meta.get("expected", "unknown"))
            result = self._step_pass_fail(expected, outcome)
            row = {
                "step": step,
                "expected": expected,
                "result": result,
                "outcome": outcome,
                "best_label": best.label if best else "none",
                "best_confidence": round(float(best.confidence), 4) if best else 0.0,
                "recommended_action": command.action,
                "yaw": command.yaw,
                "forward": command.forward,
                "duration_s": command.duration_s,
                "source_image": frame_meta.get("source_image"),
                "instruction": frame_meta.get("instruction", ""),
            }
            self.step_results.append(row)
            self.logger.detection({
                "step": step,
                "mode": "stationary_hil",
                "playback": frame_meta,
                "selected_candidate": selected.as_dict() if selected else None,
                "detections": [d.as_dict() for d in detections],
                "outcome": outcome,
                "expected": expected,
                "step_result": result,
                "best_detection": best.as_dict() if best else None,
            })
            self.logger.action({"step": step, "mode": "stationary_hil", "recommended_command": command.as_dict(), "motor_execution": "disabled_log_only", "step_result": result})
            self.world.step_time(0.5)
        pass_count = sum(1 for r in self.step_results if r.get("result") == "PASS")
        fail_count = sum(1 for r in self.step_results if r.get("result") == "FAIL")
        self._write_step_results_csv()
        summary = {
            "mode": "stationary_hil",
            "steps": steps,
            "pass_count": pass_count,
            "fail_count": fail_count,
            "output_dir": str(self.logger.output_dir),
            "note": "No motors were executed. Move object/wall cues manually between steps on real hardware.",
        }
        self.logger.finalize(summary)
        return summary
