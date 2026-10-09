from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import csv
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ocean_voyager.core.types import Pool, Pose, normalize_angle_deg, distance


@dataclass
class CoverageCell:
    ix: int
    iy: int
    center_x: float
    center_y: float
    wall_margin: bool = False
    path_covered: bool = False
    thermal_scanned: bool = False
    vision_checked: bool = False
    first_vision_time_s: float | None = None
    processed_object_id: str = ""
    processed_status: str = ""
    first_path_time_s: float | None = None
    first_thermal_time_s: float | None = None
    path_visit_count: int = 0

    @property
    def explored(self) -> bool:
        # For mission completion/waypoint skipping, explored means physical path or
        # an explicitly processed object zone. Thermal-scanned area is logged as
        # sensor evidence but does not by itself prove physical coverage.
        return self.path_covered or bool(self.processed_object_id)


@dataclass
class CoverageGrid:
    """Grid-based coverage map in the mission coordinate frame.

    This lets the lawnmower planner return to the old waypoint after a detour while
    still remembering what areas have already been traversed or thermally scanned.
    """

    pool: Pool
    cell_size_m: float = 0.35
    path_coverage_radius_m: float = 0.55
    thermal_coverage_range_m: float = 3.5
    thermal_fov_deg: float = 55.0
    processed_object_radius_m: float = 0.70
    enabled: bool = True
    events: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.cell_size_m = max(0.10, float(self.cell_size_m))
        self.nx = int(math.ceil(self.pool.length_m / self.cell_size_m))
        self.ny = int(math.ceil(self.pool.width_m / self.cell_size_m))
        self.cells: dict[tuple[int, int], CoverageCell] = {}
        for ix in range(self.nx):
            for iy in range(self.ny):
                cx = self.pool.x_min + (ix + 0.5) * self.cell_size_m
                cy = self.pool.y_min + (iy + 0.5) * self.cell_size_m
                wall_margin = not self.pool.contains(cx, cy, margin=self.pool.wall_margin_m)
                self.cells[(ix, iy)] = CoverageCell(ix, iy, cx, cy, wall_margin=wall_margin)

    @classmethod
    def from_config(cls, pool: Pool, cfg: dict[str, Any]) -> "CoverageGrid":
        c = cfg.get("coverage_grid", {})
        sensors = cfg.get("sensors", {})
        thermal = sensors.get("thermal", {})
        mission = cfg.get("mission", {})
        return cls(
            pool=pool,
            enabled=bool(c.get("enabled", True)),
            cell_size_m=float(c.get("cell_size_m", 0.35)),
            path_coverage_radius_m=float(c.get("path_coverage_radius_m", 0.55)),
            thermal_coverage_range_m=float(c.get("thermal_coverage_range_m", thermal.get("max_range_m", thermal.get("max_detection_range_m", 3.5)))),
            thermal_fov_deg=float(c.get("thermal_fov_deg", thermal.get("fov_deg", thermal.get("fov_x_deg", 55.0)))),
            processed_object_radius_m=float(c.get("processed_object_radius_m", mission.get("verify_distance_m", 0.72))),
        )

    def _iter_cells_within_radius(self, x: float, y: float, radius_m: float):
        r = max(0.0, radius_m)
        ix0 = max(0, int(math.floor((x - r - self.pool.x_min) / self.cell_size_m)))
        ix1 = min(self.nx - 1, int(math.floor((x + r - self.pool.x_min) / self.cell_size_m)))
        iy0 = max(0, int(math.floor((y - r - self.pool.y_min) / self.cell_size_m)))
        iy1 = min(self.ny - 1, int(math.floor((y + r - self.pool.y_min) / self.cell_size_m)))
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                cell = self.cells[(ix, iy)]
                if distance(x, y, cell.center_x, cell.center_y) <= r:
                    yield cell

    def mark_path_covered(self, pose: Pose, step: int, time_s: float, state: str = "") -> None:
        if not self.enabled:
            return
        count = 0
        for cell in self._iter_cells_within_radius(pose.x, pose.y, self.path_coverage_radius_m):
            if cell.wall_margin:
                continue
            if not cell.path_covered:
                cell.first_path_time_s = time_s
                count += 1
            cell.path_covered = True
            cell.path_visit_count += 1
        self.events.append({"event": "path_covered", "step": step, "time_s": time_s, "x": pose.x, "y": pose.y, "state": state, "cells_marked": count})

    def mark_thermal_scan(self, pose: Pose, pan_angle_deg: float, step: int, time_s: float, candidate_found: bool = False, max_range_m: float | None = None, fov_deg: float | None = None, best_delta_c: float | None = None, best_confidence: float | None = None, candidate_x: float | None = None, candidate_y: float | None = None) -> None:
        if not self.enabled:
            return
        max_range = float(max_range_m if max_range_m is not None else self.thermal_coverage_range_m)
        fov = float(fov_deg if fov_deg is not None else self.thermal_fov_deg)
        center_abs = normalize_angle_deg(pose.heading_deg + pan_angle_deg)
        count = 0
        for cell in self._iter_cells_within_radius(pose.x, pose.y, max_range):
            if cell.wall_margin:
                continue
            abs_bearing = math.degrees(math.atan2(cell.center_y - pose.y, cell.center_x - pose.x))
            rel = normalize_angle_deg(abs_bearing - center_abs)
            if abs(rel) <= fov / 2.0:
                if not cell.thermal_scanned:
                    cell.first_thermal_time_s = time_s
                    count += 1
                cell.thermal_scanned = True
        self.events.append({
            "event": "thermal_scan",
            "step": step,
            "time_s": time_s,
            "pose_x": pose.x,
            "pose_y": pose.y,
            "heading_deg": pose.heading_deg,
            "pan_angle_deg": pan_angle_deg,
            "scan_center_abs_deg": center_abs,
            "range_m": max_range,
            "fov_deg": fov,
            "candidate_found": candidate_found,
            "best_delta_c": best_delta_c,
            "best_confidence": best_confidence,
            "candidate_x": candidate_x,
            "candidate_y": candidate_y,
            "cells_marked": count,
        })

    def mark_camera_capture(self, pose: Pose, camera_id: str, mount_angle_deg: float, fov_deg: float, max_range_m: float, step: int, time_s: float) -> None:
        if not self.enabled:
            return
        center_abs = normalize_angle_deg(pose.heading_deg + mount_angle_deg)
        count = 0
        for cell in self._iter_cells_within_radius(pose.x, pose.y, max_range_m):
            if cell.wall_margin:
                continue
            abs_bearing = math.degrees(math.atan2(cell.center_y - pose.y, cell.center_x - pose.x))
            rel = normalize_angle_deg(abs_bearing - center_abs)
            if abs(rel) <= fov_deg / 2.0:
                if not cell.vision_checked:
                    cell.first_vision_time_s = time_s
                    count += 1
                cell.vision_checked = True
        self.events.append({
            "event": "camera_capture",
            "step": step,
            "time_s": time_s,
            "pose_x": pose.x,
            "pose_y": pose.y,
            "camera_id": camera_id,
            "mount_angle_deg": mount_angle_deg,
            "fov_deg": fov_deg,
            "range_m": max_range_m,
            "cells_marked": count,
        })

    def mark_object_processed(self, object_id: str, x: float, y: float, status: str, step: int, time_s: float) -> None:
        if not self.enabled:
            return
        count = 0
        for cell in self._iter_cells_within_radius(x, y, self.processed_object_radius_m):
            if cell.wall_margin:
                continue
            cell.processed_object_id = object_id
            cell.processed_status = status
            count += 1
        self.events.append({"event": "object_processed", "step": step, "time_s": time_s, "object_id": object_id, "x": x, "y": y, "status": status, "cells_marked": count})

    def is_point_explored(self, x: float, y: float, radius_m: float | None = None, min_fraction: float = 0.55) -> bool:
        if not self.enabled:
            return False
        radius = radius_m if radius_m is not None else self.path_coverage_radius_m
        cells = [c for c in self._iter_cells_within_radius(x, y, radius) if not c.wall_margin]
        if not cells:
            return False
        explored = sum(1 for c in cells if c.explored)
        return (explored / len(cells)) >= min_fraction

    def summary(self) -> dict[str, Any]:
        valid = [c for c in self.cells.values() if not c.wall_margin]
        def pct(n: int) -> float:
            return round(100.0 * n / max(len(valid), 1), 2)
        path = sum(1 for c in valid if c.path_covered)
        cumulative_path_visits = sum(int(c.path_visit_count) for c in valid)
        thermal = sum(1 for c in valid if c.thermal_scanned)
        explored = sum(1 for c in valid if c.explored)
        vision = sum(1 for c in valid if c.vision_checked)
        processed = sum(1 for c in valid if c.processed_object_id)
        return {
            "enabled": self.enabled,
            "cell_size_m": self.cell_size_m,
            "valid_cells": len(valid),
            "path_covered_cells": path,
            "raw_path_cell_visit_count": cumulative_path_visits,
            "thermal_scanned_cells": thermal,
            "explored_cells": explored,
            "vision_checked_cells": vision,
            "processed_object_cells": processed,
            "path_covered_percent": pct(path),
            "raw_path_cell_visit_percent": round(100.0 * cumulative_path_visits / max(len(valid), 1), 2),
            "average_path_visit_count": round(cumulative_path_visits / max(len(valid), 1), 3),
            "thermal_scanned_percent": pct(thermal),
            "vision_checked_percent": pct(vision),
            "explored_percent": pct(explored),
        }

    def write_outputs(self, output_dir: str | Path) -> dict[str, str]:
        out = Path(output_dir)
        csv_path = out / "coverage_grid_final.csv"
        events_path = out / "coverage_grid_timeline.jsonl"
        camera_log_path = out / "camera_capture_log.csv"
        thermal_coverage_log_path = out / "thermal_scan_coverage_log.csv"
        summary_path = out / "coverage_grid_summary.json"
        png_path = out / "coverage_grid_final.png"

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "ix", "iy", "center_x", "center_y", "wall_margin", "path_covered", "thermal_scanned", "vision_checked",
                "processed_object_id", "processed_status", "explored", "path_visit_count", "first_path_time_s", "first_thermal_time_s", "first_vision_time_s",
            ])
            writer.writeheader()
            for cell in sorted(self.cells.values(), key=lambda c: (c.iy, c.ix)):
                writer.writerow({
                    "ix": cell.ix,
                    "iy": cell.iy,
                    "center_x": f"{cell.center_x:.4f}",
                    "center_y": f"{cell.center_y:.4f}",
                    "wall_margin": cell.wall_margin,
                    "path_covered": cell.path_covered,
                    "thermal_scanned": cell.thermal_scanned,
                    "vision_checked": cell.vision_checked,
                    "processed_object_id": cell.processed_object_id,
                    "processed_status": cell.processed_status,
                    "explored": cell.explored,
                    "path_visit_count": cell.path_visit_count,
                    "first_path_time_s": "" if cell.first_path_time_s is None else f"{cell.first_path_time_s:.3f}",
                    "first_thermal_time_s": "" if cell.first_thermal_time_s is None else f"{cell.first_thermal_time_s:.3f}",
                    "first_vision_time_s": "" if cell.first_vision_time_s is None else f"{cell.first_vision_time_s:.3f}",
                })
        with events_path.open("w", encoding="utf-8") as f:
            for e in self.events:
                f.write(json.dumps(e, default=str) + "\n")

        camera_events = [e for e in self.events if e.get("event") == "camera_capture"]
        with camera_log_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["step", "time_s", "camera_id", "pose_x", "pose_y", "mount_angle_deg", "fov_deg", "range_m", "cells_marked"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in camera_events:
                writer.writerow({k: e.get(k, "") for k in fieldnames})

        thermal_events = [e for e in self.events if e.get("event") == "thermal_scan"]
        with thermal_coverage_log_path.open("w", newline="", encoding="utf-8") as f:
            fieldnames = ["step", "time_s", "pose_x", "pose_y", "heading_deg", "pan_angle_deg", "scan_center_abs_deg", "range_m", "fov_deg", "candidate_found", "best_delta_c", "best_confidence", "candidate_x", "candidate_y", "cells_marked"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for e in thermal_events:
                writer.writerow({k: e.get(k, "") for k in fieldnames})

        summary = self.summary()
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        self.write_png(png_path)
        return {
            "coverage_grid_csv": str(csv_path),
            "coverage_grid_timeline": str(events_path),
            "camera_capture_log": str(camera_log_path),
            "thermal_scan_coverage_log": str(thermal_coverage_log_path),
            "coverage_grid_summary": str(summary_path),
            "coverage_grid_png": str(png_path),
        }

    def write_png(self, output_path: str | Path) -> None:
        out = Path(output_path)
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.set_title("Coverage Grid Final State")
        ax.set_xlabel("x relative to start (m)")
        ax.set_ylabel("y relative to start (m)")
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlim(self.pool.x_min - 0.2, self.pool.x_max + 0.2)
        ax.set_ylim(self.pool.y_min - 0.2, self.pool.y_max + 0.2)
        size = max(8, 120 * self.cell_size_m)
        for cell in self.cells.values():
            if cell.wall_margin:
                marker = "s"; alpha = 0.16
            elif cell.processed_object_id:
                marker = "*"; alpha = 0.80
            elif cell.path_covered and cell.vision_checked:
                marker = "s"; alpha = 0.72
            elif cell.vision_checked:
                marker = "s"; alpha = 0.45
            elif cell.path_covered and cell.thermal_scanned:
                marker = "s"; alpha = 0.65
            elif cell.thermal_scanned:
                marker = "s"; alpha = 0.30
            elif cell.path_covered:
                marker = "s"; alpha = 0.25
            else:
                continue
            ax.scatter([cell.center_x], [cell.center_y], marker=marker, s=size, alpha=alpha)
        ax.add_patch(plt.Rectangle((self.pool.x_min, self.pool.y_min), self.pool.length_m, self.pool.width_m, fill=False, linewidth=2))
        ax.scatter([0], [0], marker="*", s=120)
        ax.text(0.05, 0.05, "HOME (0,0)", fontsize=8)
        fig.tight_layout()
        fig.savefig(out, dpi=150)
        plt.close(fig)
