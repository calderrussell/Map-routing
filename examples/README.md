# Sioux Falls: an end-to-end real-network example

This example uses the canonical Sioux Falls transportation benchmark: a stylized model
of a real US city's road network with 24 intersections and 76 directed links. It is not
a live, lane-accurate OpenStreetMap extract; it is used because it is small,
reproducible, and standard in traffic-assignment research.

Run it from the repository root:

```bash
.venv/bin/python examples/sioux_falls_gnn.py
```

The default quick run takes roughly 10–20 seconds on a laptop. A larger demonstration
set and model are available with `--full`.

```bash
.venv/bin/python examples/sioux_falls_gnn.py --full
```

## What each step does

1. **Load the road graph.** `sioux_falls_static()` creates the 24-node/76-link
   benchmark. `static_links_to_cells()` makes every directed road link a CTM cell and
   connects consecutive road links with admissible movement edges.
2. **Define traffic commodities.** Three OD zone pairs—1→20, 7→13, and 24→1—are
   attached to reproducible outbound and inbound link cells.
3. **Generate labels.** `gradient_oracle_demonstrations()` samples states and demand,
   solves a receding-horizon CTM routing problem, and stores only the first action plus
   solver diagnostics. These local solutions are explicitly marked uncertified.
4. **Train the GNN.** `HeteroSpatioTemporalGNN` encodes road cells, intersection
   entities, OD tokens, movement relations, and global context. The decoder produces a
   masked split probability for every active OD commodity and legal road movement.
5. **Evaluate closed loop.** The trained recurrent policy and dynamic shortest paths
   face the same held-out event demand and the same 75% capacity loss on link 10→15.
6. **Inspect honest metrics.** The JSON compares TSTT, throughput, unfinished vehicles,
   source queues, physical violations, and end-to-end latency. The quick example is a
   usage demonstration, not a trained publication model; it may underperform a strong
   baseline because it has only six oracle demonstrations and no DAgger rounds.

Outputs are written to:

- `artifacts/runs/sioux_falls_example.json`
- `artifacts/runs/sioux_falls_gnn.pt`

Both are gitignored runtime artifacts. For research claims, replace quick mode with the
frozen E0–E7 protocol, train only on assigned training topologies, tune only on
validation topologies, and leave Anaheim/Winnipeg untouched until final testing.

## What the quick run actually showed

With seed 31, six oracle demonstrations, and 60 imitation epochs:

| Controller | TSTT | Throughput | Unfinished vehicles |
|---|---:|---:|---:|
| Heterogeneous GNN | 2054.20 | 72.94 | 219.94 |
| Dynamic shortest paths | 1840.75 | 116.17 | 176.72 |

The GNN was physically valid—zero capacity violation and a maximum conservation
residual of `5.68e-14`—but it did **not** beat the baseline. That is the useful lesson:
a low imitation loss (`0.00765` → `0.0000152`) does not guarantee good closed-loop
travel time. The policy encounters states outside its six oracle demonstrations and
errors compound over time. The next research step is to run DAgger on model-visited
states, decision-focused CTM fine-tuning, incident augmentation, and the full topology
split rather than presenting this teaching run as a performance result.

## Visual geographic example

The separate [`graphs/cambridge_mit.json`](graphs/cambridge_mit.json) fixture contains
a small, deliberately simplified network of real Cambridge street names and WGS84
coordinates. It is designed for the Folium result viewer rather than as a calibrated
traffic model. From the repository root:

```bash
.venv/bin/gnn-dso-map --train-demo \
  --graph examples/graphs/cambridge_mit.json \
  --output artifacts/runs/cambridge_gnn_map.html
```

See [`../docs/folium_visualization.md`](../docs/folium_visualization.md) to use a
checkpoint or substitute your own coordinate-bearing road graph.
