# Quantum Computing Applications for Flight Trajectory Optimization

A small research/engineering project (university Quantum Computing course)
that models a simplified flight-trajectory shortest-path problem, solves it
classically (brute force + Dijkstra), formulates it as a QUBO, and solves
the QUBO with QAOA on a local quantum simulator (Qiskit + Aer) -- then
compares the two honestly, including when QAOA does *not* find the optimum.

**This does not demonstrate quantum advantage.** It demonstrates the
methodology (problem -> QUBO -> Ising -> QAOA -> decode -> validate -> compare)
on toy-sized instances, and documents where and why it stops scaling. See
`docs/experimental_methodology.md`.

## 1. Project motivation

Trajectory/route optimization is a canonical combinatorial optimization
problem, and shortest-path-style QUBOs are one of the more approachable
ways to see how a real (if simplified) engineering objective turns into a
cost Hamiltonian a quantum computer could in principle optimize. This
project builds that pipeline end to end, with a classical ground truth at
every step so the quantum result can actually be checked rather than taken
on faith.

## 2. Problem statement

Given a directed graph of waypoints with per-edge distance, fuel cost, time
cost and weather penalty, find the minimum-combined-cost simple path from a
given origin to a given destination. See `docs/problem_formulation.md` for
the full mathematical statement, including why the four cost components are
normalized before being combined.

## 3. Architecture

```
src/
  models/         FlightNetwork, Node, Edge, objective/cost evaluation
  data/           deterministic scenario generator
  classical/      brute-force ground truth, Dijkstra baseline
  optimization/   QUBO derivation (pure Python, no quantum SDK), metrics
  quantum/        QUBO -> Ising, QAOA solver, bitstring decoder/validator
  experiments/    CLI: generate benchmark scenarios
  visualization/  network + route-comparison plots (PNG, shared by CLI and web app)
  webapp/         local Flask research dashboard (this is the "web app")
  utils/          typed config (pydantic), structured logging
  main.py         end-to-end CLI demonstration
tests/            62 tests: unit + property-style + one real end-to-end QAOA run
docs/             architecture, math formulation, QUBO derivation, QAOA method, methodology
configs/          (reserved for saved experiment configs)
results/          scenarios/, benchmarks/ (plots), benchmark_results.csv (appended per run)
```

See `docs/architecture.md` for the reasoning behind the module boundaries
(in particular why QUBO construction and the Ising/QAOA layer are split
into two modules with the Qiskit dependency isolated to one file).

## 4. Installation

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12 and the exact pinned versions in
`requirements.txt` (Qiskit 2.5.2, qiskit-aer 0.17.2, qiskit-optimization
0.7.0). Qiskit's APIs move fast; other versions are not guaranteed to work
unmodified (see the transpiler-bug note in `docs/experimental_methodology.md`).

## 5. Running a basic experiment (CLI)

```bash
python -m src.main
```

Runs the full pipeline (scenario -> brute force -> classical -> QUBO ->
QAOA -> decode -> compare) on a small (5-node) scenario and prints a
summary, saves a route-comparison plot to `results/benchmarks/`, and
appends a row to `results/benchmark_results.csv`.

Useful flags:

```bash
python -m src.main --num-nodes 6 --seed 7 --reps 3 --shots 4096
python -m src.main --alpha 1 --beta 2 --gamma 1 --delta 0.5   # reweight the objective
python -m src.main --no-plot --no-save                          # quick, no side effects
```

`--num-nodes` defaults to 5 deliberately: qubits = directed edges in the
network, and statevector QAOA training time grows roughly exponentially
with qubit count (measured; see `docs/experimental_methodology.md`). 8+
nodes at the default connectivity can take several minutes; going further
will hit a `QuantumSimulatorError` safety guard rather than exhausting
memory.

## 6. Running the web dashboard

```bash
python -m src.webapp
# then open http://127.0.0.1:5050
```

A small local, single-user dashboard (Flask) wrapping the same pipeline
`src/main.py` uses: set scenario parameters and objective weights, generate
a network, tune QAOA settings, run the optimization, and see the classical
vs. quantum routes plotted side by side plus every reported metric. It
holds one scenario/result in memory (not a per-browser-session store) --
it's a research tool for one person at a time, not a production web
service. `--host` / `--port` / `--debug` flags are available.

## 7. Generating benchmark scenarios

```bash
python -m src.experiments.generate_scenarios --seed 42
python -m src.experiments.generate_scenarios --seed 42 --tier medium
python -m src.experiments.generate_scenarios --seed 42 --num-nodes 12
```

Writes deterministic scenario JSON files to `results/scenarios/`.

## 8. Running tests

```bash
python -m pytest -q
```

62 tests, ~3-13s depending on machine (one test runs a real QAOA
optimization on a 12-qubit instance). Notably:

- `tests/test_qubo.py::test_qubo_global_minimum_matches_brute_force_shortest_path`
  exhaustively sweeps all 2^12 bit assignments of a hand-built QUBO and
  confirms the minimum-energy assignment is exactly the brute-force-verified
  shortest path -- the central correctness proof for the QUBO derivation.
- `tests/test_dijkstra.py` cross-validates Dijkstra against brute force on
  five independently seeded scenarios.
- `tests/test_end_to_end.py` runs the real QAOA solver (not mocked) and
  checks every invariant that must hold regardless of whether QAOA finds
  the optimum (a valid decoded path can never cost less than the true
  optimum; metrics are computed consistently).

## 9. Generating plots

Plots are produced automatically by `python -m src.main` (saved under
`results/benchmarks/`) and by the web dashboard (rendered directly in the
browser). To regenerate a plot for a specific saved scenario/result
combination programmatically, see `src/visualization/plot_network.py`
(`plot_network_only`, `plot_route_comparison`) -- both return PNG bytes and
optionally also write to a path.

## 10. Example results

From an actual run (`python -m src.main --seed 42`, 5 nodes, 16 qubits):

```
Brute-force ground truth (8 simple paths examined): WP0 -> WP4, cost 4.3203
Classical (Dijkstra):                                WP0 -> WP4, cost 4.3203
Cross-validation (brute force == classical): PASS
QUBO: 16 qubits, penalty_lambda=128.0
QAOA (reps=3, COBYLA, 4096 shots): 80 optimizer evaluations, converged
Quantum route:  WP0 -> WP4, cost 4.3203 (matched the optimum this run)
Optimality gap: 0.0000%   Approximation ratio: 1.0000
```

From a different seed (`--num-nodes 6 --seed 7`), QAOA did **not** find a
valid path at all at reps=3/120 iterations -- reported honestly as "no
valid bitstring decoded to a path", not hidden or padded. Both outcomes are
real, unedited runs; see `results/benchmark_results.csv` for the full log
this repo ships with.

## 11. Limitations

- **Toy-scale only.** Nodes represent abstract waypoints, not real airports;
  distance/fuel/time/weather are synthetic per-edge numbers with plausible
  but invented randomization ranges, not real aircraft performance or
  weather data.
- **Statevector simulation caps qubit count hard.** Memory need is
  2^qubits x 16 bytes; this project measured that ceiling directly (see
  `docs/experimental_methodology.md`) and enforces a configurable
  `max_qubits` safety guard (`QAOAConfig.max_qubits`, default 24) rather
  than letting the simulator crash.
- **QAOA at shallow depth on a penalty-heavy landscape is genuinely hard to
  train.** It sometimes reaches the classical optimum and sometimes
  doesn't, even on identical-size instances with a different seed -- this
  project reports both outcomes rather than cherry-picking successful runs.
- **The "large" benchmark tier (15-25 nodes) is intentionally beyond QAOA
  feasibility** at this project's default connectivity; it is meant to
  demonstrate *where* the wall is, per the project's Problem Scaling
  requirement, not to be run through QAOA end to end.
- **Real flight trajectory optimization** involves aircraft performance
  envelopes, live wind/weather, air traffic control and airspace
  regulation, airport procedures, safety separation, and altitude-dependent
  fuel burn -- none of which this project models. It is a methodology
  demonstration, not an operational aviation tool.
- A Qiskit 2.5.2 transpiler bug (a Rust panic in the 1-qubit-gate
  optimization pass, triggered by `QAOAAnsatz` circuits above roughly 30
  qubits when run through the preset pass manager) was discovered during
  development and worked around by decomposing the ansatz directly instead
  of transpiling it for a hardware target it was never going to run on. See
  `src/quantum/qaoa_solver.py` and `docs/experimental_methodology.md`.

## 12. Future work

- Try alternative QUBO penalty schedules / adaptive lambda tuning.
- Warm-start QAOA parameters from the classical solution's structure.
- Compare against simulated annealing / quantum annealing formulations.
- Multi-objective (Pareto) comparison instead of a single scalarized weight vector.
- If real hardware access is available: run the trained circuit on actual
  IBM Quantum hardware for the smallest instances and compare shot noise
  against the noiseless-simulator baseline this project uses.
