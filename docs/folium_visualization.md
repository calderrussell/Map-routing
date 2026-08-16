# Visual model testing with Folium

The `gnn-dso-map` command turns one geographic road graph into a complete, inspectable
model test:

1. validate the graph and convert every directed road segment into a CTM cell;
2. resolve each origin/destination pair onto its first and last road cells;
3. load a heterogeneous GNN checkpoint or train a small local teaching model;
4. run the model in closed loop under the graph's demand and incidents;
5. run dynamic shortest path on the identical scenario for context;
6. write an interactive Folium HTML map and a machine-readable JSON summary.

## Run the bundled Cambridge example

After `pip install -e '.[dev]'`:

```bash
.venv/bin/gnn-dso-map --train-demo \
  --graph examples/graphs/cambridge_mit.json \
  --output artifacts/runs/cambridge_gnn_map.html \
  --save-checkpoint artifacts/runs/cambridge_gnn.pt
```

The quick training settings are intentionally small. They prove that the full path
works; they are not evidence that the resulting policy is near-optimal. The JSON
summary reports the GNN-minus-baseline TSTT directly so an underperforming model cannot
be mistaken for a successful research result.

To evaluate a saved topology-general checkpoint:

```bash
.venv/bin/gnn-dso-map \
  --graph path/to/your_graph.json \
  --checkpoint path/to/model.pt \
  --output artifacts/runs/my_graph_result.html
```

Use `--controlled-fraction 0.5` to test partial compliance, `--horizon 30` to override
the scenario horizon, or `--no-baseline` to skip the comparison run. A checkpoint must
contain either a GNN state dictionary or a dictionary with `state_dict`; model width
and layer count are inferred when `model_config` is absent.

## What the map means

- Color is the maximum observed occupancy divided by physical cell storage: blue is
  light, yellow is near storage, and red would indicate a storage violation.
- Line width is the total physically realized vehicle flow through that directed cell.
  The two directions are slightly offset so opposing flows remain visible.
- The time slider replays per-step occupancy, realized flow, and active incidents.
- Green play markers are origins; red flags are destinations.
- The layer control switches between the GNN result, the dynamic-shortest-path result,
  incident locations, OD endpoints, and unstyled road centerlines.
- The fixed panel reports TSTT, throughput, unfinished vehicles, and conservation.

The output is one HTML result file, but Folium's JavaScript libraries and the CartoDB
base tiles are fetched by the browser, so viewing the complete basemap requires an
internet connection.

## Geographic graph JSON schema

The loader accepts schema version 1. Coordinates and geometries use WGS84 `[lat, lon]`
order. A minimal file is:

```json
{
  "schema_version": 1,
  "name": "two_route_example",
  "attribution": "Describe the source and license of the geographic data",
  "defaults": {
    "capacity": 7.0,
    "storage": 35.0,
    "free_speed": 0.8,
    "wave_speed": 0.35,
    "free_time": 1.0,
    "lanes": 1,
    "road_class": "tertiary"
  },
  "nodes": [
    {"id": "a", "lat": 42.0, "lon": -71.0, "label": "Origin"},
    {"id": "b", "lat": 42.001, "lon": -70.999, "label": "Junction"},
    {"id": "c", "lat": 42.002, "lon": -70.998, "label": "Destination"}
  ],
  "roads": [
    {"id": "a_b", "name": "First Road", "source": "a", "target": "b", "bidirectional": true},
    {"id": "b_c", "name": "Second Road", "source": "b", "target": "c", "bidirectional": true}
  ],
  "od_pairs": [
    {"id": "a_to_c", "origin_node": "a", "destination_node": "c", "base_demand": 3.0}
  ],
  "scenario": {
    "id": "held_out_event",
    "horizon": 18,
    "seed": 47,
    "regime": "event",
    "forecast_noise": 0.12,
    "incidents": [
      {"road_ids": ["b_c"], "start": 6, "duration": 4,
       "capacity_multiplier": 0.3, "speed_multiplier": 0.5,
       "kind": "unfamiliar_location"}
    ]
  }
}
```

Roads default to bidirectional and expand to IDs such as `a_b:forward` and
`a_b:reverse`. Set `bidirectional` to `false` for a one-way road. A road may include a
polyline in `geometry`; otherwise its endpoints form a straight segment. Per-road
attributes override `defaults`. `road_class` accepts a number or one of `service`,
`residential`, `tertiary`, `secondary`, `primary`, `trunk`, and `motorway`.

OD road cells are normally selected from the free-time shortest node path. For full
control, set `origin_cell` and `destination_cell` explicitly; the origin cell must
leave the origin node and the destination cell must enter the destination node.
Incidents can reference expanded `cell_ids` or `road_ids` (which affects both
directions).

The loader rejects duplicate IDs, invalid coordinates, disconnected OD pairs,
non-positive physical attributes, malformed geometries, and unknown incident targets
before the model is run.
