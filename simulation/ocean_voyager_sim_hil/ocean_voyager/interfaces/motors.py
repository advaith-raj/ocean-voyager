from __future__ import annotations
from abc import ABC, abstractmethod
from ocean_voyager.core.types import MotionCommand


class MotorExecutorInterface(ABC):
    @abstractmethod
    def execute(self, command: MotionCommand) -> None:
        raise NotImplementedError

    @abstractmethod
    def stop(self) -> None:
        raise NotImplementedError
