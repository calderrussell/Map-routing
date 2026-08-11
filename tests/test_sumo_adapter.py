from pathlib import Path

from simulators.sumo_adapter import SUMOAdapter, SUMOConfig, SUMOControllerBridge


class _Vehicle:
    def __init__(self):
        self.routes = {}

    def getIDList(self):
        return ["veh0"]

    def setRoute(self, vehicle_id, route):
        self.routes[vehicle_id] = route


class _TraCI:
    def __init__(self):
        self.vehicle = _Vehicle()


def test_sumo_bridge_and_explicit_availability_status(tmp_path: Path) -> None:
    traci = _TraCI()
    SUMOControllerBridge.apply_routes(
        traci, {"veh0": ("edge0", "edge1"), "not_departed": ("edge2",)}
    )
    assert traci.vehicle.routes == {"veh0": ["edge0", "edge1"]}
    adapter = SUMOAdapter(SUMOConfig(tmp_path / "missing.sumocfg"))
    available, reason = adapter.availability()
    assert not available
    assert reason

