from __future__ import annotations
from typing import Any
from ocean_voyager.interfaces.motors import MotorExecutorInterface
from ocean_voyager.core.types import MotionCommand
from ocean_voyager.simulation.robot_dynamics import DifferentialThrustDynamics
from ocean_voyager.device_bus.file_bus import NullDeviceBus


class SimMotorExecutor(MotorExecutorInterface):
    def __init__(self, dynamics: DifferentialThrustDynamics, command_log: list[dict] | None = None, bus: Any | None = None):
        self.dynamics = dynamics
        self.command_log = command_log if command_log is not None else []
        self.bus = bus or NullDeviceBus()

    def execute(self, command: MotionCommand) -> None:
        command.clamp()
        pwm = self.dynamics.command_to_pwm(command)
        self.bus.transact(
            device="motors",
            command="set_pwm_for_duration",
            params={**pwm, "duration_sec": command.duration_s, "source_command": command.as_dict()},
            response_payload={"ack": True, "planned_execution": "simulated"},
            time_sec=self.dynamics.world.time_s,
        )
        motion = self.dynamics.apply(command)
        self.command_log.append(command.as_dict() | {"executor": "sim", **motion})
        self.bus.transact(
            device="motors",
            command="motion_update",
            params={"request_source": "set_pwm_for_duration"},
            response_payload=motion,
            time_sec=self.dynamics.world.time_s,
        )

    def stop(self) -> None:
        cmd = MotionCommand(0, 0, 0, action="STOP", reason="sim stop")
        pwm = self.dynamics.command_to_pwm(cmd)
        self.command_log.append(cmd.as_dict() | {"executor": "sim", "pwm_output": pwm})
        self.bus.transact(device="motors", command="stop", params=pwm, response_payload={"ack": True, "neutral_pwm_sent": True}, time_sec=self.dynamics.world.time_s)


class LogOnlyMotorExecutor(MotorExecutorInterface):
    def __init__(self, command_log: list[dict] | None = None, bus: Any | None = None):
        self.command_log = command_log if command_log is not None else []
        self.bus = bus or NullDeviceBus()

    def execute(self, command: MotionCommand) -> None:
        self.command_log.append(command.as_dict() | {"executor": "log_only", "executed": False})
        self.bus.transact(device="motors", command="log_only_command", params=command.as_dict(), response_payload={"executed": False, "reason": "motors disabled/log-only"})

    def stop(self) -> None:
        cmd = MotionCommand(0, 0, 0, action="STOP", reason="log-only stop")
        self.command_log.append(cmd.as_dict() | {"executor": "log_only", "executed": False})
        self.bus.transact(device="motors", command="log_only_stop", params=cmd.as_dict(), response_payload={"executed": False})
