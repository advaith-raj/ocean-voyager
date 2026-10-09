from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Any
import csv
import json
import math
import shutil


class MissionLogger:
    def __init__(self, base_output_dir: str | Path = "outputs", run_name: str | None = None):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(base_output_dir) / (run_name or f"run_{ts}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "mission_log.jsonl"
        self.trajectory_path = self.output_dir / "trajectory.csv"
        self.detections_path = self.output_dir / "detections.json"
        self.actions_path = self.output_dir / "recommended_actions.json"
        self.summary_path = self.output_dir / "summary.json"
        self.thermal_scan_path = self.output_dir / "thermal_scans.json"
        self.object_coordinate_path = self.output_dir / "object_coordinate_log.csv"
        self.ground_truth_object_path = self.output_dir / "ground_truth_objects.csv"
        self.object_coordinate_json_path = self.output_dir / "object_coordinates.json"
        self.motor_command_path = self.output_dir / "motor_command_log.json"
        self.detections: list[dict[str, Any]] = []
        self.actions: list[dict[str, Any]] = []
        self.thermal_scans: list[dict[str, Any]] = []
        self.trajectory_initialized = False
        self._last_pose: dict[str, Any] | None = None
        self.path_length_m = 0.0

    def event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"event_type": event_type, **payload}
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def pose(self, step: int, time_s: float, state: str, pose: dict[str, Any]) -> None:
        if not self.trajectory_initialized:
            with self.trajectory_path.open("w", encoding="utf-8") as f:
                f.write("step,time_s,state,x,y,heading_deg,path_length_m\n")
            self.trajectory_initialized = True
        if self._last_pose is not None:
            self.path_length_m += math.hypot(float(pose["x"]) - float(self._last_pose["x"]), float(pose["y"]) - float(self._last_pose["y"]))
        self._last_pose = dict(pose)
        with self.trajectory_path.open("a", encoding="utf-8") as f:
            f.write(f"{step},{time_s:.3f},{state},{pose['x']:.4f},{pose['y']:.4f},{pose['heading_deg']:.3f},{self.path_length_m:.4f}\n")


    def thermal_scan(self, payload: dict[str, Any]) -> None:
        self.thermal_scans.append(payload)
        self.event("thermal_scan", payload)

    def detection(self, payload: dict[str, Any]) -> None:
        self.detections.append(payload)
        self.event("detection", payload)

    def action(self, payload: dict[str, Any]) -> None:
        self.actions.append(payload)
        self.event("action", payload)

    def save_static_inputs(self, config_path: str | Path | None = None, scenario_path: str | Path | None = None) -> None:
        if config_path:
            try:
                shutil.copy2(config_path, self.output_dir / "config_used.yaml")
            except Exception:
                pass
        if scenario_path:
            try:
                shutil.copy2(scenario_path, self.output_dir / "scenario_used.yaml")
            except Exception:
                pass

    def write_object_coordinates(self, objects: list[Any], detections: list[dict[str, Any]] | None = None) -> None:
        """Write object locations in the mission coordinate frame.

        Coordinates are relative to HOME/START = (0, 0).
        - ground_truth_* are the scenario coordinates.
        - estimated_* are what the robot inferred from the thermal candidate when seen.
        """
        det_lookup: dict[str, dict[str, Any]] = {}
        for d in detections or self.detections:
            obj_id = d.get("ground_truth_object") or (d.get("candidate") or {}).get("source_object_id")
            if obj_id and obj_id not in det_lookup:
                det_lookup[str(obj_id)] = d

        ground_truth_rows: list[dict[str, Any]] = []
        object_rows: list[dict[str, Any]] = []
        for obj in objects:
            det = det_lookup.get(obj.id, {})
            cand = det.get("candidate") or {}
            status = getattr(obj.status, "value", str(obj.status))
            obj_type = getattr(obj.type, "value", str(obj.type))
            ground_truth_rows.append({
                "object_id": obj.id,
                "type": obj_type,
                "x_m_from_start": round(float(obj.x), 4),
                "y_m_from_start": round(float(obj.y), 4),
                "expected": getattr(obj, "expected", ""),
            })
            object_rows.append({
                "object_id": obj.id,
                "type": obj_type,
                "ground_truth_x": f"{obj.x:.4f}",
                "ground_truth_y": f"{obj.y:.4f}",
                "expected": getattr(obj, "expected", ""),
                "final_status": status,
                "detected": bool(det),
                "confirmed": status == "confirmed",
                "estimated_x": f"{float(cand.get('estimated_x', 0.0)):.4f}" if cand else "",
                "estimated_y": f"{float(cand.get('estimated_y', 0.0)):.4f}" if cand else "",
                "first_seen_step": det.get("step", ""),
                "first_seen_time_s": det.get("time_s", ""),
            })

        with self.ground_truth_object_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["object_id", "type", "x_m_from_start", "y_m_from_start", "expected"])
            writer.writeheader()
            writer.writerows(ground_truth_rows)

        with self.object_coordinate_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "object_id", "type", "ground_truth_x", "ground_truth_y", "expected", "final_status",
                "detected", "confirmed", "estimated_x", "estimated_y", "first_seen_step", "first_seen_time_s",
            ])
            writer.writeheader()
            writer.writerows(object_rows)

        payload = {
            "coordinate_frame": {
                "origin": "HOME / START",
                "home_x_m": 0.0,
                "home_y_m": 0.0,
                "units": "meters",
                "note": "All object coordinates are relative to the starting position (0, 0).",
            },
            "ground_truth_objects": ground_truth_rows,
            "object_detection_results": object_rows,
        }
        with self.object_coordinate_json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

    def finalize(self, summary: dict[str, Any]) -> None:
        with self.detections_path.open("w", encoding="utf-8") as f:
            json.dump(self.detections, f, indent=2, default=str)
        with self.thermal_scan_path.open("w", encoding="utf-8") as f:
            json.dump(self.thermal_scans, f, indent=2, default=str)
        with self.actions_path.open("w", encoding="utf-8") as f:
            json.dump(self.actions, f, indent=2, default=str)
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)
