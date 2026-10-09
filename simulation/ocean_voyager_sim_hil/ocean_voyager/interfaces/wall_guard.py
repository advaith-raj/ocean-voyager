from __future__ import annotations
from abc import ABC, abstractmethod
from ocean_voyager.core.types import WallObservation


class WallGuardInterface(ABC):
    @abstractmethod
    def check(self) -> WallObservation:
        raise NotImplementedError
