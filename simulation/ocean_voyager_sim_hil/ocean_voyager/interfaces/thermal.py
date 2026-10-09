from __future__ import annotations
from abc import ABC, abstractmethod
from ocean_voyager.core.types import ThermalReading


class ThermalInterface(ABC):
    @abstractmethod
    def scan(self) -> list[ThermalReading]:
        raise NotImplementedError
