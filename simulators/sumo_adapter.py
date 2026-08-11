from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import shutil
from typing import Mapping


class SUMOUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SUMOConfig:
    configuration_file: Path
    step_length_seconds: float = 1.0
    seed: int = 0
    binary: str = "sumo"


class SUMOControllerBridge:
    """Apply frozen route assignments through the minimal TraCI vehicle API."""

    @staticmethod
    def apply_routes(traci_module, assignments: Mapping[str, tuple[str, ...]]) -> None:
        active = set(traci_module.vehicle.getIDList())
        for vehicle_id, route in assignments.items():
            if vehicle_id in active:
                traci_module.vehicle.setRoute(vehicle_id, list(route))


class SUMOAdapter:
    def __init__(self, config: SUMOConfig) -> None:
        self.config = config

    def availability(self) -> tuple[bool, str]:
        binary = shutil.which(self.config.binary)
        traci = importlib.util.find_spec("traci")
        if binary is None:
            return False, f"SUMO binary {self.config.binary!r} is not installed"
        if traci is None:
            return False, "Python package 'traci' is not installed; install the [sumo] extra"
        if not self.config.configuration_file.exists():
            return False, f"SUMO configuration does not exist: {self.config.configuration_file}"
        return True, "available"

    def command(self) -> list[str]:
        return [
            self.config.binary,
            "-c",
            str(self.config.configuration_file),
            "--step-length",
            str(self.config.step_length_seconds),
            "--seed",
            str(self.config.seed),
            "--no-step-log",
            "true",
        ]

    def run(self, frozen_controller, maximum_steps: int) -> dict[str, float]:
        available, reason = self.availability()
        if not available:
            raise SUMOUnavailable(reason)
        import traci  # type: ignore

        traci.start(self.command())
        steps = 0
        departed = 0
        arrived = 0
        try:
            while steps < maximum_steps and traci.simulation.getMinExpectedNumber() > 0:
                traci.simulationStep()
                observations = frozen_controller.observe(traci)
                assignments = frozen_controller.act(observations)
                SUMOControllerBridge.apply_routes(traci, assignments)
                departed += traci.simulation.getDepartedNumber()
                arrived += traci.simulation.getArrivedNumber()
                steps += 1
        finally:
            traci.close()
        return {"steps": float(steps), "departed": float(departed), "arrived": float(arrived)}

