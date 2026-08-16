from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import html
import json
import math
from pathlib import Path

import folium
from branca.element import Element
from folium.plugins import TimestampedGeoJson
import numpy as np

from visualization.graph_io import DirectedRoadCell, GeoGraphProblem
from visualization.trace import ControllerTrace


CONGESTION_COLORS = (
    (0.25, "#2c7bb6"),
    (0.50, "#00a6ca"),
    (0.75, "#00ccbc"),
    (1.00, "#f9d057"),
    (math.inf, "#d7191c"),
)


def _congestion_color(ratio: float) -> str:
    return next(color for threshold, color in CONGESTION_COLORS if ratio <= threshold)


def _display_geometry(cell: DirectedRoadCell, amount: float = 0.000012) -> list[list[float]]:
    """Offset opposing directed cells to opposite sides of the road centerline."""

    points = np.asarray(cell.geometry, dtype=float)
    delta = points[-1] - points[0]
    normal = np.asarray([-delta[1], delta[0]], dtype=float)
    norm = float(np.linalg.norm(normal))
    if norm > 0:
        points = points + amount * normal / norm
    return points.tolist()


def _trace_arrays(problem: GeoGraphProblem, trace: ControllerTrace) -> dict[str, np.ndarray]:
    network = problem.network
    sources = network.movement_sources
    occupancy = np.stack(
        [np.sum(step.result.state.occupancy, axis=0) for step in trace.steps]
    )
    ratios = occupancy / network.storage[None, :]
    outflow = np.zeros_like(occupancy)
    for time_index, step in enumerate(trace.steps):
        movement_totals = np.sum(step.result.movement_flow, axis=0)
        np.add.at(outflow[time_index], sources, movement_totals)
        for commodity, destination in enumerate(problem.destinations):
            outflow[time_index, destination] += step.result.exit_flow[commodity]
    return {
        "occupancy": occupancy,
        "ratios": ratios,
        "outflow": outflow,
        "total_flow": np.sum(outflow, axis=0),
        "max_ratio": np.max(ratios, axis=0),
    }


def _tooltip(
    cell: DirectedRoadCell,
    max_ratio: float,
    total_flow: float,
    *,
    prefix: str,
) -> str:
    return (
        f"<b>{html.escape(prefix)} — {html.escape(cell.name)}</b><br>"
        f"{html.escape(cell.source)} → {html.escape(cell.target)}<br>"
        f"Directed cell: {html.escape(cell.id)}<br>"
        f"Maximum occupancy: {100.0 * max_ratio:.1f}% of storage<br>"
        f"Vehicles routed through cell: {total_flow:.1f}<br>"
        f"Per-step capacity: {cell.capacity:.1f}"
    )


def _add_trace_layer(
    map_object: folium.Map,
    problem: GeoGraphProblem,
    trace: ControllerTrace,
    *,
    layer_name: str,
    show: bool,
) -> dict[str, np.ndarray]:
    arrays = _trace_arrays(problem, trace)
    maximum_flow = max(float(np.max(arrays["total_flow"])), 1e-9)
    group = folium.FeatureGroup(name=layer_name, show=show)
    for cell_index, cell in enumerate(problem.graph.cells):
        ratio = float(arrays["max_ratio"][cell_index])
        volume = float(arrays["total_flow"][cell_index])
        folium.PolyLine(
            _display_geometry(cell),
            color=_congestion_color(ratio),
            weight=3.0 + 7.0 * math.sqrt(volume / maximum_flow),
            opacity=0.82,
            tooltip=folium.Tooltip(
                _tooltip(cell, ratio, volume, prefix=layer_name), sticky=True
            ),
        ).add_to(group)
    group.add_to(map_object)
    return arrays


def _animation_geojson(
    problem: GeoGraphProblem, trace: ControllerTrace, arrays: dict[str, np.ndarray]
) -> dict[str, object]:
    features: list[dict[str, object]] = []
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for time_index, step in enumerate(trace.steps):
        stamp = (start + timedelta(minutes=time_index)).isoformat().replace("+00:00", "Z")
        for cell_index, cell in enumerate(problem.graph.cells):
            ratio = float(arrays["ratios"][time_index, cell_index])
            flow = float(arrays["outflow"][time_index, cell_index])
            coordinates = [
                [lon, lat] for lat, lon in _display_geometry(cell, amount=0.000018)
            ]
            incident = step.capacity_multiplier[cell_index] < 1.0
            tooltip = (
                f"t={time_index} · {cell.name} · {cell.source}→{cell.target} · "
                f"occupancy {100.0 * ratio:.1f}% · flow {flow:.2f}"
                + (" · INCIDENT ACTIVE" if incident else "")
            )
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": coordinates},
                    "properties": {
                        "times": [stamp] * len(coordinates),
                        "style": {
                            "color": "#7f0000" if incident else _congestion_color(ratio),
                            "weight": 8 if incident else 5,
                            "opacity": 0.85,
                        },
                        "tooltip": tooltip,
                    },
                }
            )
    return {"type": "FeatureCollection", "features": features}


def _add_network_and_markers(map_object: folium.Map, problem: GeoGraphProblem) -> None:
    network_group = folium.FeatureGroup(name="Road centerlines", show=False)
    for road in problem.graph.roads:
        folium.PolyLine(
            [list(point) for point in road.geometry],
            color="#556270",
            weight=3,
            opacity=0.65,
            tooltip=f"{html.escape(road.name)} ({html.escape(road.id)})",
        ).add_to(network_group)
    network_group.add_to(map_object)

    incident_cells = {
        cell_index
        for incident in problem.scenario.incidents
        for cell_index in incident.affected_cells
    }
    if incident_cells:
        incident_group = folium.FeatureGroup(name="Incident locations", show=True)
        seen_roads: set[str] = set()
        for cell_index in sorted(incident_cells):
            cell = problem.graph.cells[cell_index]
            if cell.road_id in seen_roads:
                continue
            seen_roads.add(cell.road_id)
            folium.PolyLine(
                [list(point) for point in cell.geometry],
                color="#7f0000",
                weight=11,
                opacity=0.5,
                dash_array="8 8",
                tooltip=f"Incident: {html.escape(cell.name)}",
            ).add_to(incident_group)
        incident_group.add_to(map_object)

    node_by_id = problem.graph.node_by_id
    od_group = folium.FeatureGroup(name="OD endpoints", show=True)
    for pair_index, pair in enumerate(problem.graph.od_pairs, start=1):
        origin = node_by_id[pair.origin_node]
        destination = node_by_id[pair.destination_node]
        folium.Marker(
            [origin.lat, origin.lon],
            tooltip=f"OD {pair_index} origin: {html.escape(origin.label)}",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(od_group)
        folium.Marker(
            [destination.lat, destination.lon],
            tooltip=f"OD {pair_index} destination: {html.escape(destination.label)}",
            icon=folium.Icon(color="red", icon="flag-checkered", prefix="fa"),
        ).add_to(od_group)
    od_group.add_to(map_object)


def _panel_html(
    problem: GeoGraphProblem,
    model_trace: ControllerTrace,
    baseline_trace: ControllerTrace | None,
    title: str,
) -> str:
    model = model_trace.evaluation.metrics
    baseline_row = ""
    if baseline_trace is not None:
        baseline = baseline_trace.evaluation.metrics
        delta = model.total_system_travel_time - baseline.total_system_travel_time
        baseline_row = (
            f"<div><b>Baseline TSTT:</b> {baseline.total_system_travel_time:.2f}</div>"
            f"<div><b>GNN − baseline:</b> {delta:+.2f}</div>"
        )
    return f"""
    <div id="gnn-routing-summary" style="position:fixed;top:12px;right:12px;z-index:9999;
      width:300px;background:rgba(255,255,255,.96);border:1px solid #8b949e;border-radius:8px;
      box-shadow:0 2px 9px rgba(0,0,0,.25);padding:12px;font:13px/1.45 Arial,sans-serif;">
      <div style="font-size:16px;font-weight:700;margin-bottom:6px;">{html.escape(title)}</div>
      <div><b>Graph:</b> {html.escape(problem.graph.name)}</div>
      <div><b>Scenario:</b> {html.escape(problem.scenario.scenario_id)}</div>
      <div><b>GNN TSTT:</b> {model.total_system_travel_time:.2f}</div>
      <div><b>Throughput:</b> {model.throughput:.2f}</div>
      <div><b>Unfinished vehicles:</b> {model.unfinished_vehicles:.2f}</div>
      <div><b>Conservation residual:</b> {model.conservation_residual:.2e}</div>
      {baseline_row}
      <div style="margin-top:7px;color:#57606a;">Use the time slider to replay congestion.
      Toggle layers to compare the GNN with dynamic shortest path.</div>
    </div>
    """


def _legend_html(attribution: str) -> str:
    swatches = "".join(
        f'<span style="display:inline-block;width:18px;height:8px;background:{color};"></span>'
        f" {label}<br>"
        for color, label in (
            ("#2c7bb6", "≤25% occupied"),
            ("#00a6ca", "≤50%"),
            ("#00ccbc", "≤75%"),
            ("#f9d057", "≤100%"),
            ("#d7191c", ">100% (violation)"),
        )
    )
    return f"""
    <div id="gnn-routing-legend" style="position:fixed;bottom:24px;right:12px;z-index:9998;
      background:rgba(255,255,255,.94);border:1px solid #8b949e;border-radius:6px;
      padding:9px;font:11px/1.45 Arial,sans-serif;max-width:285px;">
      <b>Occupancy / storage</b><br>{swatches}
      Line width = total routed volume<br>
      <span style="color:#57606a;">{html.escape(attribution)}</span>
    </div>
    """


def render_folium_result(
    problem: GeoGraphProblem,
    model_trace: ControllerTrace,
    output_path: str | Path,
    *,
    baseline_trace: ControllerTrace | None = None,
    title: str = "GNN traffic-routing result",
) -> Path:
    """Write an interactive Folium map for one traced model evaluation."""

    if not model_trace.steps:
        raise ValueError("cannot render an empty controller trace")
    coordinates = [(node.lat, node.lon) for node in problem.graph.nodes]
    center = [
        float(np.mean([point[0] for point in coordinates])),
        float(np.mean([point[1] for point in coordinates])),
    ]
    map_object = folium.Map(
        location=center,
        zoom_start=15,
        tiles="CartoDB positron",
        control_scale=True,
        prefer_canvas=True,
    )
    map_object.get_root().header.add_child(Element(f"<title>{html.escape(title)}</title>"))
    _add_network_and_markers(map_object, problem)
    model_arrays = _add_trace_layer(
        map_object,
        problem,
        model_trace,
        layer_name="GNN: maximum congestion + routed volume",
        show=True,
    )
    if baseline_trace is not None:
        _add_trace_layer(
            map_object,
            problem,
            baseline_trace,
            layer_name="Baseline: maximum congestion + routed volume",
            show=False,
        )
    TimestampedGeoJson(
        _animation_geojson(problem, model_trace, model_arrays),
        period="PT1M",
        duration="PT1M",
        transition_time=350,
        auto_play=False,
        loop=False,
        add_last_point=False,
        date_options="[Simulation step] m",
        time_slider_drag_update=True,
    ).add_to(map_object)
    folium.LayerControl(collapsed=False, position="topleft").add_to(map_object)
    map_object.fit_bounds(coordinates, padding=(28, 28))
    map_object.get_root().html.add_child(
        Element(_panel_html(problem, model_trace, baseline_trace, title))
    )
    map_object.get_root().html.add_child(Element(_legend_html(problem.graph.attribution)))
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    map_object.save(destination)
    return destination


def trace_summary(
    problem: GeoGraphProblem,
    model_trace: ControllerTrace,
    *,
    baseline_trace: ControllerTrace | None = None,
) -> dict[str, object]:
    """Return the JSON-safe companion summary used by the CLI."""

    payload: dict[str, object] = {
        "graph": problem.graph.name,
        "graph_file": str(problem.graph.source_path),
        "scenario": problem.scenario.scenario_id,
        "cells": problem.network.n_cells,
        "movements": problem.network.n_movements,
        "od_pairs": len(problem.graph.od_pairs),
        "horizon": len(problem.scenario.realized_demand),
        "controller": model_trace.evaluation.controller,
        "metrics": asdict(model_trace.evaluation.metrics),
        "latency_seconds": model_trace.evaluation.latency,
    }
    if baseline_trace is not None:
        payload["baseline"] = {
            "controller": baseline_trace.evaluation.controller,
            "metrics": asdict(baseline_trace.evaluation.metrics),
            "latency_seconds": baseline_trace.evaluation.latency,
        }
        payload["gnn_minus_baseline_tstt"] = (
            model_trace.evaluation.metrics.total_system_travel_time
            - baseline_trace.evaluation.metrics.total_system_travel_time
        )
    return json.loads(json.dumps(payload, allow_nan=False))
