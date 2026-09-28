# Experimental Methodology

## Benchmark design

Scenarios are generated deterministically (`src/data/scenario_generator.py`,
seeded via `numpy.random.default_rng`) at three nominal sizes (Section 7 of
the project brief):

| Tier   | Nodes      |
|--------|------------|
| small  | 5, 6, 7, 8 |
| medium | 10, 12, 15 |
| large  | 18, 20, 25 |

These are simplified research benchmarks, not operational aviation
networks -- see `docs/problem_formulation.md` for what is and isn't
modeled.

## Classical baseline and ground truth

- **Brute force** (`src/classical/brute_force.py`): exhaustively enumerates
  every simple path via `networkx.all_simple_paths` and takes the minimum.
  Used as ground truth for instances with `<= 10` nodes
  (`BRUTE_FORCE_NODE_LIMIT` in `src/main.py`); beyond that the number of
  simple paths grows too fast to enumerate in reasonable time on a
  moderately connected graph.
- **Classical baseline** (`src/classical/dijkstra.py`): `networkx.dijkstra_path`
  on the combined-cost graph. Valid because every edge cost is `>= 0` by
  construction (see `docs/problem_formulation.md`), which is Dijkstra's
  correctness precondition.
- Every pipeline run cross-validates the two when both are available
  (`abs(classical_cost - brute_force_cost) < 1e-6`) before the quantum layer
  is even built -- see the "Cross-validation (brute force == classical)"
  line in `src/main.py`'s output.

## Metrics

For each run: number of nodes/edges/qubits, classical cost, quantum cost,
optimal cost (brute force when available, else classical), execution time
(scenario generation, QAOA circuit build/train/sample separately), shots,
QAOA optimizer evaluation count and convergence flag, and the derived
metrics below (`src/optimization/metrics.py`):

```
approximation_ratio = quantum_cost / optimal_cost      (1.0 = optimal; >1.0 = worse)
optimality_gap       = (quantum_cost - optimal_cost) / optimal_cost   (0.0 = optimal)
```

Both handle `optimal_cost == 0` explicitly (return `1.0`/`0.0` if
`quantum_cost` is also `0`, else `+inf`) rather than raising
`ZeroDivisionError` -- see that module's tests. Every run appends one row
to `results/benchmark_results.csv`, including software versions and a
timestamp, so results are reproducible and comparable across runs
(Section 18 of the project brief).

## Two limitations discovered by actually running this project

Both of the following were found empirically during development -- by
running the pipeline and watching it fail -- not anticipated in advance.
They are recorded here because they materially shape what this project can
and cannot demonstrate, and because "actually execute it, don't assume it
works" is a stated project requirement.

### 1. A Qiskit 2.5.2 transpiler bug at higher qubit counts

Running `QAOAAnsatz` circuits above roughly 30 qubits through Qiskit's
preset transpiler pass manager (`generate_preset_pass_manager(...).run(ansatz)`,
targeting an `AerSimulator` backend) crashed with a Rust panic inside the
transpiler's 1-qubit-gate-optimization pass (`optimize_1q_gates_decomposition`),
at *every* `optimization_level` (0 through 3):

```
thread '<unnamed>' panicked at .../optimize_1q_gates_decomposition.rs:377:36:
index out of bounds: the len is 27 but the index is 33
```

This is a genuine bug/limitation in the pinned Qiskit version, not a
mistake in this project's circuit construction (the same ansatz, at 12
qubits, transpiled and ran without incident). The fix used here
(`src/quantum/qaoa_solver.py::_fully_decompose`) sidesteps the bug rather
than patching around it: basis-gate translation for a specific hardware
target is only needed to run on real hardware, and this project only ever
targets a local Aer simulation, so the ansatz is lowered to primitive gates
via repeated `.decompose()` calls instead, which never invokes the crashing
pass at all.

### 2. Statevector memory scales as 2^qubits, and qubits scale fast with connectivity

Measured directly (statevector size = `2^qubits x 16 bytes`, complex128):

| nodes | k_nearest | extra_edge_fraction | qubits | statevector memory |
|------:|----------:|---------------------:|-------:|--------------------:|
| 5     | 1         | 0.0                   | 8      | ~0 MB               |
| 8     | 2         | 0.1                   | 24     | 256 MB              |
| 10    | 2         | 0.1                   | 30     | 16,384 MB           |
| 12    | 3         | 0.0                   | 42     | 67,108,864 MB       |
| 15    | 3         | 0.0                   | 54     | 274,877,906,944 MB  |

(Full table generated during development covered nodes in `{5,6,7,8,10,12,15}`
x `k_nearest in {1,2,3}` x `extra_edge_fraction in {0,0.1}`; the rows above
are representative.) On this project's development machine (a container
reporting ~4000 MB available to Aer), attempting to hold a statevector
above roughly 26-28 qubits fails outright:

```
ERROR: Insufficient memory to run circuit QAOA using the statevector
simulator. Required memory: 262144M, max memory: 4000M
```

Two consequences, both handled explicitly rather than left to crash:

- `QAOAConfig.max_qubits` (default 24) is checked **before** any circuit is
  built (`src/quantum/qaoa_solver.py::_check_qubit_feasibility`), raising a
  clear `QuantumSimulatorError` naming the actual memory requirement if
  exceeded, instead of letting Aer's own OOM error surface deep inside a
  primitive call (or, worse, exhausting the host's memory).
- `ScenarioConfig`'s default connectivity (`k_nearest=2, extra_edge_fraction=0.1`)
  and `src/main.py`'s default `--num-nodes 5` were chosen, after this
  measurement, specifically to keep the out-of-the-box CLI/web-app
  experience within a comfortably feasible qubit range (16 qubits by
  default) -- earlier defaults (`k_nearest=3`) produced 34 qubits for an
  8-node scenario, which is the exact configuration that triggered the
  transpiler crash above during development and would in any case have
  required 256 GB to simulate.
- Beyond ~20 qubits, per-evaluation training cost also grows fast in wall
  time (not just memory): a single noiseless expectation-value evaluation
  measured at 16 qubits (~0.1s) vs. 20 qubits (~1.3-1.6s) on the same
  hardware -- roughly consistent with the expected exponential scaling.
  This is why `src/main.py`'s default is a 5-node scenario (a full QAOA
  training run completes in well under 15 seconds) rather than the
  8-node default used earlier in development.
- The "large" benchmark tier (15-25 nodes) is **expected** to exceed
  `max_qubits` at this project's default connectivity. That is by design,
  per the project brief's own framing of the large tier as being "subject
  to simulator feasibility" -- it demonstrates where the wall is, which is
  itself one of this project's required deliverables (Section 21: document
  limitations honestly).

## What this project does and does not claim

- It does **not** claim quantum advantage, and reports every QAOA run's
  actual outcome, including outright failures to find any valid path
  (observed at, e.g., 6 nodes/16 qubits/reps=3/120 iterations, seed 7 --
  see `results/benchmark_results.csv` for the logged run).
- Where QAOA does find the classical optimum (also observed, e.g., the
  5-node/16-qubit default scenario at seed 42), that is reported as a
  genuine, unedited result -- not evidence of anything beyond "QAOA found
  the optimum on this particular small instance."
- Distinctions maintained throughout: measured experimental results (this
  document's tables and `results/benchmark_results.csv`) vs. theoretical
  expectations (`docs/quantum_method.md`'s description of what QAOA is
  supposed to do) vs. explicit assumptions/simplifications
  (`docs/problem_formulation.md`) vs. future possibilities
  (`README.md`'s Future Work section).
