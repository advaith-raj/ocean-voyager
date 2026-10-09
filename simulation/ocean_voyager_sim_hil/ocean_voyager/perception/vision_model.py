from __future__ import annotations
from pathlib import Path
from typing import Any
from ocean_voyager.core.types import CameraFrame, DetectionResult
from ocean_voyager.interfaces.vision_model import VisionModelInterface


class MockVisionModel(VisionModelInterface):
    """Deterministic classifier for simulator/HIL pipeline tests."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg or {}
        self.confidence_by_type = self.cfg.get("confidence_by_type", {"bottle": 0.90, "seaweed": 0.78, "glare": 0.70})

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_type": "mock",
            "target": "deterministic simulator-visible objects",
            "labels": list(self.confidence_by_type.keys()),
        }

    def health_check(self) -> dict[str, Any]:
        return {
            "status": "ready",
            "self_test": "pass",
            "capabilities": self.capabilities(),
            "last_error": None,
            "warnings": ["Mock vision is for simulator/HIL plumbing only; it does not classify real images."],
        }

    def predict(self, frame: CameraFrame) -> list[DetectionResult]:
        results: list[DetectionResult] = []
        for obj in frame.visible_objects:
            label = obj.type.value
            conf = float(self.confidence_by_type.get(label, 0.45))
            if obj.distance_m > 3.0:
                conf *= 0.75
            if obj.distance_m < 0.7:
                conf *= 0.95
            results.append(DetectionResult(label=label, confidence=round(conf, 3), bbox=(100, 80, 260, 220), object_id=obj.object_id, camera_id=frame.camera_id))
        results.sort(key=lambda r: r.confidence, reverse=True)
        return results


class YoloVisionModel(VisionModelInterface):
    """YOLO wrapper with safe startup behavior.

    The adapter no longer crashes during construction when ultralytics/model files
    are missing. Instead, `health_check()` reports failed status and `predict()`
    raises a clear RuntimeError if used before the model is ready.
    """

    def __init__(self, model_path: str, confidence_threshold: float = 0.25, target_labels: list[str] | None = None):
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.target_labels = {x.lower() for x in (target_labels or ["waste", "bottle", "plastic", "plastic_bottle"])}
        self.model = None
        self.available = False
        self.last_error: str | None = None
        self.warnings: list[str] = []
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"model path does not exist: {model_path}")
            from ultralytics import YOLO  # type: ignore
            self.model = YOLO(model_path)
            self.available = True
        except Exception as exc:  # pragma: no cover - depends on local install
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.warnings.append("YOLO unavailable. Install ultralytics/torch and verify models/best.pt before real vision integration.")

    def capabilities(self) -> dict[str, Any]:
        return {
            "model_type": "yolo",
            "model_path": self.model_path,
            "confidence_threshold": self.confidence_threshold,
            "target_labels": sorted(self.target_labels),
            "available": self.available,
        }

    def health_check(self) -> dict[str, Any]:
        if self.available:
            names = {}
            try:
                names = dict(getattr(self.model, "names", {}) or {})
            except Exception:
                names = {}
            return {
                "status": "ready",
                "self_test": "pass",
                "capabilities": self.capabilities() | {"model_names": names},
                "last_error": None,
                "warnings": self.warnings,
            }
        return {
            "status": "failed",
            "self_test": "fail",
            "capabilities": self.capabilities(),
            "last_error": self.last_error,
            "warnings": self.warnings,
        }

    def predict(self, frame: CameraFrame) -> list[DetectionResult]:  # pragma: no cover - optional dependency
        if not self.available or self.model is None:
            raise RuntimeError(f"YOLO model is not ready: {self.last_error}")
        if not frame.image_path:
            return []
        preds = self.model(frame.image_path, verbose=False)
        out: list[DetectionResult] = []
        for pred in preds:
            names = pred.names
            for box in pred.boxes:
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                cls_id = int(box.cls[0])
                xyxy = box.xyxy[0].tolist()
                out.append(DetectionResult(label=str(names.get(cls_id, cls_id)), confidence=conf, bbox=tuple(int(v) for v in xyxy), camera_id=frame.camera_id))
        out.sort(key=lambda r: r.confidence, reverse=True)
        return out
