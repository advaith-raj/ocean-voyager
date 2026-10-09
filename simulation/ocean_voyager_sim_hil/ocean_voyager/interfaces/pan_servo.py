from __future__ import annotations
from abc import ABC, abstractmethod


class PanServoInterface(ABC):
    @abstractmethod
    def rotate_to(self, angle_deg: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def current_angle(self) -> float:
        raise NotImplementedError
