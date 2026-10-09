from __future__ import annotations
from abc import ABC, abstractmethod
from ocean_voyager.core.types import CameraFrame, DetectionResult


class VisionModelInterface(ABC):
    @abstractmethod
    def predict(self, frame: CameraFrame) -> list[DetectionResult]:
        raise NotImplementedError
