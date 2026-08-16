from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

from data_processed.scenarios import Incident, Scenario, dynamic_demand
from oracle.network import CellNetwork
from simulators.ctm import CTMSimulator


ROAD_CLASS = {
    "service": 0.0,
    "residential": 1.0,
    "tertiary": 2.0,
    "secondary": 3.0,
    "primary": 4.0,
    "trunk": 5.0,
    "motorway": 5.0,
}

DEFAULT_ROAD_ATTRIBUTES = {
    "capacity": 7.0,
    "storage": 35.0,
    "free_speed": 0.8,
    "wave_speed": 0.35,
    "free_time": 1.0,
    "lanes": 1.0,
    "road_class": 2.0,
}


@dataclass(frozen=True)
class GeoNode:
    id: str
    lat: float
    lon: float
    label: str


@dataclass(frozen=True)
class RoadSegment:
    id: str
    name: str
    source: str
    target: str
    geometry: tuple[tuple[float, float], ...]
    bidirectional: bool
    capacity: float
    storage: float
    free_speed: float
    wave_speed: float
    free_time: float
    lanes: float
    road_class: float


@dataclass(frozen=True)
class DirectedRoadCell:
    id: str
    road_id: str
    name: str
    source: str
    target: str
    geometry: tuple[tuple[float, float], ...]
    direction: str
    capacity: float
    storage: float
    free_speed: float
    wave_speed: float
    free_time: float
    lanes: float
    road_class: float


@dataclass(frozen=True)
class ODPair:
    id: str
    origin_node: str
    destination_node: str
    origin_cell: str
    destination_cell: str
    base_demand: float


@dataclass(frozen=True)
class GeoRoadGraph:
    name: str
    attribution: str
    source_path: Path
    nodes: tuple[GeoNode, ...]
    roads: tuple[RoadSegment, ...]
    cells: tuple[DirectedRoadCell, ...]
    od_pairs: tuple[ODPair, ...]

    @property
    def node_by_id(self) -> dict[str, GeoNode]:
        return {node.id: node for node in self.nodes}

    @property
    def cell_by_id(self) -> dict[str, DirectedRoadCell]:
        return {cell.id: cell for cell in self.cells}


@dataclass(frozen=True)
class GeoGraphProblem:
    graph: GeoRoadGraph
    network: CellNetwork
    simulator: CTMSimulator
    origins: np.ndarray
    destinations: np.ndarray
    base_demand: np.ndarray
    scenario: Scenario


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number") from exc
    if not math.isfinite(result) or (positive and result <= 0):
        qualifier = "a positive finite number" if positive else "finite"
        raise ValueError(f"{label} must be {qualifier}")
    return result


def _bearing(geometry: tuple[tuple[float, float], ...]) -> float:
    lat1, lon1 = geometry[0]
    lat2, lon2 = geometry[-1]
    east = math.radians(lon2 - lon1) * math.cos(math.radians((lat1 + lat2) / 2.0))
    north = math.radians(lat2 - lat1)
    return math.atan2(east, north)


def _parse_nodes(payload: dict[str, Any]) -> tuple[GeoNode, ...]:
    records = payload.get("nodes")
    if not isinstance(records, list) or not records:
        raise ValueError("graph JSON must contain a non-empty 'nodes' list")
    nodes: list[GeoNode] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"nodes[{index}] must be an object")
        node_id = str(record.get("id", "")).strip()
        if not node_id or node_id in seen:
            raise ValueError(f"nodes[{index}].id must be non-empty and unique")
        lat = _finite_number(record.get("lat"), f"nodes[{index}].lat")
        lon = _finite_number(record.get("lon"), f"nodes[{index}].lon")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"nodes[{index}] is outside WGS84 latitude/longitude bounds")
        seen.add(node_id)
        nodes.append(GeoNode(node_id, lat, lon, str(record.get("label", node_id))))
    return tuple(nodes)


def _parse_geometry(
    record: dict[str, Any], source: GeoNode, target: GeoNode, label: str
) -> tuple[tuple[float, float], ...]:
    raw = record.get("geometry", [[source.lat, source.lon], [target.lat, target.lon]])
    if not isinstance(raw, list) or len(raw) < 2:
        raise ValueError(f"{label}.geometry must contain at least two [lat, lon] points")
    points: list[tuple[float, float]] = []
    for point_index, point in enumerate(raw):
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"{label}.geometry[{point_index}] must be [lat, lon]")
        lat = _finite_number(point[0], f"{label}.geometry[{point_index}][0]")
        lon = _finite_number(point[1], f"{label}.geometry[{point_index}][1]")
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            raise ValueError(f"{label}.geometry[{point_index}] is outside WGS84 bounds")
        points.append((lat, lon))
    endpoint_tolerance = 2e-4
    if max(abs(points[0][0] - source.lat), abs(points[0][1] - source.lon)) > endpoint_tolerance:
        raise ValueError(f"{label}.geometry must start at its source node")
    if max(abs(points[-1][0] - target.lat), abs(points[-1][1] - target.lon)) > endpoint_tolerance:
        raise ValueError(f"{label}.geometry must end at its target node")
    return tuple(points)


def _parse_roads(
    payload: dict[str, Any], node_by_id: dict[str, GeoNode]
) -> tuple[RoadSegment, ...]:
    records = payload.get("roads")
    if not isinstance(records, list) or not records:
        raise ValueError("graph JSON must contain a non-empty 'roads' list")
    supplied_defaults = payload.get("defaults", {})
    if not isinstance(supplied_defaults, dict):
        raise ValueError("defaults must be an object")
    defaults = {**DEFAULT_ROAD_ATTRIBUTES, **supplied_defaults}
    roads: list[RoadSegment] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"roads[{index}] must be an object")
        label = f"roads[{index}]"
        road_id = str(record.get("id", "")).strip()
        if not road_id or road_id in seen:
            raise ValueError(f"{label}.id must be non-empty and unique")
        source_id, target_id = str(record.get("source", "")), str(record.get("target", ""))
        if source_id not in node_by_id or target_id not in node_by_id:
            raise ValueError(f"{label} references an unknown source or target node")
        if source_id == target_id:
            raise ValueError(f"{label} cannot be a self-loop")
        values = {**defaults, **record}
        bidirectional = record.get("bidirectional", True)
        if not isinstance(bidirectional, bool):
            raise ValueError(f"{label}.bidirectional must be true or false")
        road_class_raw = values["road_class"]
        if isinstance(road_class_raw, str):
            if road_class_raw not in ROAD_CLASS:
                raise ValueError(f"{label}.road_class is not recognized")
            road_class = ROAD_CLASS[road_class_raw]
        else:
            road_class = _finite_number(road_class_raw, f"{label}.road_class")
        if not 0.0 <= road_class <= 5.0:
            raise ValueError(f"{label}.road_class must lie in [0, 5]")
        roads.append(
            RoadSegment(
                id=road_id,
                name=str(record.get("name", road_id)),
                source=source_id,
                target=target_id,
                geometry=_parse_geometry(
                    record, node_by_id[source_id], node_by_id[target_id], label
                ),
                bidirectional=bidirectional,
                capacity=_finite_number(values["capacity"], f"{label}.capacity", positive=True),
                storage=_finite_number(values["storage"], f"{label}.storage", positive=True),
                free_speed=_finite_number(
                    values["free_speed"], f"{label}.free_speed", positive=True
                ),
                wave_speed=_finite_number(
                    values["wave_speed"], f"{label}.wave_speed", positive=True
                ),
                free_time=_finite_number(
                    values["free_time"], f"{label}.free_time", positive=True
                ),
                lanes=_finite_number(values["lanes"], f"{label}.lanes", positive=True),
                road_class=road_class,
            )
        )
        seen.add(road_id)
    return tuple(roads)


def _directed_cells(roads: tuple[RoadSegment, ...]) -> tuple[DirectedRoadCell, ...]:
    cells: list[DirectedRoadCell] = []
    for road in roads:
        for direction, source, target, geometry in (
            ("forward", road.source, road.target, road.geometry),
            ("reverse", road.target, road.source, tuple(reversed(road.geometry))),
        ):
            if direction == "reverse" and not road.bidirectional:
                continue
            cells.append(
                DirectedRoadCell(
                    id=f"{road.id}:{direction}",
                    road_id=road.id,
                    name=road.name,
                    source=source,
                    target=target,
                    geometry=geometry,
                    direction=direction,
                    capacity=road.capacity,
                    storage=road.storage,
                    free_speed=road.free_speed,
                    wave_speed=road.wave_speed,
                    free_time=road.free_time,
                    lanes=road.lanes,
                    road_class=road.road_class,
                )
            )
    return tuple(cells)


def _resolve_od_pairs(
    payload: dict[str, Any], cells: tuple[DirectedRoadCell, ...], node_ids: set[str]
) -> tuple[ODPair, ...]:
    records = payload.get("od_pairs")
    if not isinstance(records, list) or not records:
        raise ValueError("graph JSON must contain a non-empty 'od_pairs' list")
    cell_by_id = {cell.id: cell for cell in cells}
    node_graph = nx.DiGraph()
    edge_cell: dict[tuple[str, str], str] = {}
    for cell in cells:
        current = node_graph.get_edge_data(cell.source, cell.target)
        if current is None or cell.free_time < current["weight"]:
            node_graph.add_edge(cell.source, cell.target, weight=cell.free_time)
            edge_cell[(cell.source, cell.target)] = cell.id
    pairs: list[ODPair] = []
    seen: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"od_pairs[{index}] must be an object")
        label = f"od_pairs[{index}]"
        pair_id = str(record.get("id", f"od_{index}"))
        if pair_id in seen:
            raise ValueError(f"{label}.id must be unique")
        origin_node = str(record.get("origin_node", ""))
        destination_node = str(record.get("destination_node", ""))
        if origin_node not in node_ids or destination_node not in node_ids:
            raise ValueError(f"{label} references an unknown origin or destination node")
        origin_cell = record.get("origin_cell")
        destination_cell = record.get("destination_cell")
        if origin_cell is None or destination_cell is None:
            try:
                path = nx.shortest_path(
                    node_graph, origin_node, destination_node, weight="weight"
                )
            except (nx.NetworkXNoPath, nx.NodeNotFound) as exc:
                raise ValueError(f"{label} has no directed road path") from exc
            if len(path) < 2:
                raise ValueError(f"{label} origin and destination must differ")
            path_cells = [edge_cell[(source, target)] for source, target in zip(path, path[1:])]
            origin_cell = path_cells[0] if origin_cell is None else str(origin_cell)
            destination_cell = path_cells[-1] if destination_cell is None else str(destination_cell)
        origin_cell, destination_cell = str(origin_cell), str(destination_cell)
        if origin_cell not in cell_by_id or destination_cell not in cell_by_id:
            raise ValueError(f"{label} references an unknown directed cell")
        if cell_by_id[origin_cell].source != origin_node:
            raise ValueError(f"{label}.origin_cell must leave origin_node")
        if cell_by_id[destination_cell].target != destination_node:
            raise ValueError(f"{label}.destination_cell must enter destination_node")
        pairs.append(
            ODPair(
                id=pair_id,
                origin_node=origin_node,
                destination_node=destination_node,
                origin_cell=origin_cell,
                destination_cell=destination_cell,
                base_demand=_finite_number(
                    record.get("base_demand"), f"{label}.base_demand", positive=True
                ),
            )
        )
        seen.add(pair_id)
    return tuple(pairs)


def _cell_network(graph_name: str, cells: tuple[DirectedRoadCell, ...]) -> CellNetwork:
    movements: list[tuple[int, int]] = []
    movement_capacity: list[float] = []
    for source_index, source in enumerate(cells):
        for target_index, target in enumerate(cells):
            if source.target == target.source and source.source != target.target:
                movements.append((source_index, target_index))
                movement_capacity.append(min(source.capacity, target.capacity))
    if not movements:
        raise ValueError("road graph produces no legal non-U-turn CTM movements")
    return CellNetwork(
        name=graph_name,
        cell_ids=tuple(cell.id for cell in cells),
        movements=tuple(movements),
        storage=np.asarray([cell.storage for cell in cells]),
        capacity=np.asarray([cell.capacity for cell in cells]),
        free_speed=np.asarray([cell.free_speed for cell in cells]),
        wave_speed=np.asarray([cell.wave_speed for cell in cells]),
        free_time=np.asarray([cell.free_time for cell in cells]),
        movement_capacity=np.asarray(movement_capacity),
        lanes=np.asarray([cell.lanes for cell in cells]),
        road_class=np.asarray([cell.road_class for cell in cells]),
        bearing=np.asarray([_bearing(cell.geometry) for cell in cells]),
    )


def _scenario(
    payload: dict[str, Any], graph: GeoRoadGraph, base_demand: np.ndarray, *, seed: int | None,
    horizon: int | None,
) -> Scenario:
    spec = payload.get("scenario", {})
    if not isinstance(spec, dict):
        raise ValueError("scenario must be an object")
    chosen_seed = int(spec.get("seed", 47) if seed is None else seed)
    chosen_horizon = int(spec.get("horizon", 18) if horizon is None else horizon)
    if chosen_horizon < 2:
        raise ValueError("scenario.horizon must be at least 2")
    regime = str(spec.get("regime", "event"))
    if regime not in {"low", "near_capacity", "oversaturated", "event"}:
        raise ValueError("scenario.regime is not recognized")
    forecast_noise = _finite_number(spec.get("forecast_noise", 0.12), "forecast_noise")
    if forecast_noise < 0:
        raise ValueError("scenario.forecast_noise must be nonnegative")
    realized, forecast = dynamic_demand(
        base_demand,
        horizon=chosen_horizon,
        seed=chosen_seed,
        regime=regime,  # type: ignore[arg-type]
        forecast_noise=forecast_noise,
    )
    cell_index = {cell.id: index for index, cell in enumerate(graph.cells)}
    road_cells: dict[str, list[str]] = {}
    for cell in graph.cells:
        road_cells.setdefault(cell.road_id, []).append(cell.id)
    incidents: list[Incident] = []
    incident_records = spec.get("incidents", [])
    if not isinstance(incident_records, list):
        raise ValueError("scenario.incidents must be a list")
    for index, record in enumerate(incident_records):
        if not isinstance(record, dict):
            raise ValueError(f"scenario.incidents[{index}] must be an object")
        ids = [str(value) for value in record.get("cell_ids", [])]
        for road_id in record.get("road_ids", []):
            if str(road_id) not in road_cells:
                raise ValueError(f"scenario.incidents[{index}] references unknown road {road_id}")
            ids.extend(road_cells[str(road_id)])
        if not ids or any(cell_id not in cell_index for cell_id in ids):
            raise ValueError(f"scenario.incidents[{index}] must reference known roads or cells")
        start = int(record.get("start", 0))
        duration = int(record.get("duration", 1))
        observation_delay = int(record.get("observation_delay", 0))
        capacity_multiplier = _finite_number(
            record.get("capacity_multiplier", 0.3), "capacity_multiplier"
        )
        speed_multiplier = _finite_number(
            record.get("speed_multiplier", 0.5), "speed_multiplier"
        )
        kind = str(record.get("kind", "unfamiliar_location"))
        if start < 0 or duration < 1 or observation_delay < 0:
            raise ValueError(
                f"scenario.incidents[{index}] needs start >= 0, duration >= 1, and observation_delay >= 0"
            )
        if not 0.0 <= capacity_multiplier <= 1.0 or not 0.0 < speed_multiplier <= 1.0:
            raise ValueError(
                f"scenario.incidents[{index}] multipliers must describe a capacity/speed reduction"
            )
        if kind not in {
            "unfamiliar_location",
            "outside_severity",
            "multiple",
            "closure",
            "noisy_duration",
        }:
            raise ValueError(f"scenario.incidents[{index}].kind is not recognized")
        incidents.append(
            Incident(
                affected_cells=tuple(sorted({cell_index[cell_id] for cell_id in ids})),
                start=start,
                duration=duration,
                capacity_multiplier=capacity_multiplier,
                speed_multiplier=speed_multiplier,
                observation_delay=observation_delay,
                kind=kind,  # type: ignore[arg-type]
            )
        )
    return Scenario(
        scenario_id=str(spec.get("id", f"{graph.name}_visual_test")),
        topology=graph.name,
        seed=chosen_seed,
        realized_demand=realized,
        forecast_demand=forecast,
        incidents=tuple(incidents),
        regime=regime,  # type: ignore[arg-type]
    )


def load_graph_problem(
    path: str | Path, *, seed: int | None = None, horizon: int | None = None
) -> GeoGraphProblem:
    """Load and validate a geographic road graph, then create its CTM test problem."""

    source_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid graph JSON in {source_path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("graph JSON must be an object with schema_version 1")
    graph_name = str(payload.get("name", source_path.stem)).strip()
    if not graph_name:
        raise ValueError("graph name must be non-empty")
    nodes = _parse_nodes(payload)
    node_by_id = {node.id: node for node in nodes}
    roads = _parse_roads(payload, node_by_id)
    cells = _directed_cells(roads)
    od_pairs = _resolve_od_pairs(payload, cells, set(node_by_id))
    graph = GeoRoadGraph(
        name=graph_name,
        attribution=str(payload.get("attribution", "User-supplied geographic graph")),
        source_path=source_path,
        nodes=nodes,
        roads=roads,
        cells=cells,
        od_pairs=od_pairs,
    )
    network = _cell_network(graph.name, graph.cells)
    cell_index = {cell_id: index for index, cell_id in enumerate(network.cell_ids)}
    origins = np.asarray([cell_index[pair.origin_cell] for pair in od_pairs], dtype=np.int64)
    destinations = np.asarray(
        [cell_index[pair.destination_cell] for pair in od_pairs], dtype=np.int64
    )
    for pair, origin, destination in zip(od_pairs, origins, destinations):
        if destination not in nx.descendants(network.graph(), int(origin)) | {int(origin)}:
            raise ValueError(f"OD pair {pair.id} is disconnected in the CTM movement graph")
    simulator = CTMSimulator(network, origins, destinations)
    base_demand = np.asarray([pair.base_demand for pair in od_pairs], dtype=float)
    scenario = _scenario(payload, graph, base_demand, seed=seed, horizon=horizon)
    return GeoGraphProblem(
        graph=graph,
        network=network,
        simulator=simulator,
        origins=origins,
        destinations=destinations,
        base_demand=base_demand,
        scenario=scenario,
    )
