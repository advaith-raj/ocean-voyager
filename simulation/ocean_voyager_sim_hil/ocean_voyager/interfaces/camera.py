from __future__ import annotations
from abc import ABC, abstractmethod
from ocean_voyager.core.types import CameraFrame


class CameraArrayInterface(ABC):
    @abstractmethod
    def get_frame(self, camera_id: str) -> CameraFrame:
        raise NotImplementedError

    def get_all_frames(self) -> dict[str, CameraFrame]:
        return {cam: self.get_frame(cam) for cam in ("front", "left", "right")}
