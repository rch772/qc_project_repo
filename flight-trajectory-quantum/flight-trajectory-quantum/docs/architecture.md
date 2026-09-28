# Architecture

## Module boundaries and why they're drawn this way

```
src/models/        FlightNetwork, Node, Edge, cost evaluation (objective.py)
src/data/           deterministic scenario generation
src/classical/      brute-force ground truth, Dijkstra baseline
src/optimization/   QUBO derivation (qubo.py) + comparison metrics (metrics.py)
src/quantum/        Ising conversion (ising.py), QAOA solver (qaoa_solver.py), decoder
src/experiments/    CLI: benchmark scenario generation
src/visualization/  plotting (shared by CLI and web app)
src/webapp/          Flask dashboard
src/utils/          typed config (pydantic), logging
src/main.py          end-to-end CLI demonstration
```

**`optimization/` vs `quantum/` is a deliberate dependency boundary, not
just a naming choice.** `src/optimization/qubo.py` builds the QUBO
(variables, linear/quadratic coefficients, penalty terms) using only plain
Python and the project's own data model -- it does not import Qiskit at
all, and can be tested (and was tested, exhaustively, for a 12-qubit
instance) with nothing but `itertools.product`. Everything that touches a
quantum SDK -- converting to an Ising Hamiltonian, building and training
the QAOA ansatz, running the simulator -- lives in `src/quantum/`, and in
fact only `src/quantum/ising.py` imports `qiskit_optimization` at all. This
means: (a) the mathematical correctness of the QUBO derivation can be
verified independently of whether Qiskit is even installed or working, and
(b) when a Qiskit upgrade changes an API (which happened during
development -- see `docs/experimental_methodology.md`), the blast radius is
contained to `src/quantum/`.

**`classical/` mirrors `optimization/quantum/` in spirit:** it holds the
two solvers whose job is to produce a trustworthy reference cost
(`brute_force.py` for small instances, `dijkstra.py` as the practical
baseline), so that whatever the QUBO/QAOA layer produces has something
independent to be checked against.

**Why edges are excluded from the graph rather than penalized as
"restricted".** Section 5 of the project brief asks for restricted-airspace
edges. Two designs were possible: (1) keep every candidate edge in the
graph and add a large penalty for using a restricted one, or (2) never add
restricted edges to the graph at all. This project uses (2)
(`src/data/scenario_generator.py`): a restricted edge is excluded from `E`
entirely, after first computing a spanning tree of the candidate edge set
and only ever excluding edges *outside* that tree, so the network can never
be disconnected by restriction. The payoff is that every downstream
consumer (brute force, Dijkstra, the QUBO) only ever has to reason about
"is this edge in the graph or not" -- there is no separate "is this edge
allowed" check scattered through the codebase, and a path found by any
solver is airspace-compliant by construction. The honest trade-off (real
restrictions are time- and altitude-dependent, not static) is stated, not
hidden -- see `docs/problem_formulation.md`.

**Why the web app is a thin Flask layer, not a rewrite.** `src/webapp/app.py`
calls the exact same functions `src/main.py` calls
(`generate_scenario`, `brute_force_solution`, `classical_shortest_path`,
`build_qubo`, `solve_qaoa`, `decode_all_counts`, ...). The dashboard and the
CLI are two front ends over one tested pipeline; neither reimplements or
approximates the pipeline's logic. State (the current scenario and last
result) is kept in a single module-level object rather than a database or
per-session store, which is an intentional simplification appropriate for a
local, single-user research tool (see the brief's explicit instruction:
"a simple research dashboard rather than a flashy website"), not an
oversight.

**Why plotting returns bytes, not just files.** `src/visualization/plot_network.py`'s
two functions (`plot_network_only`, `plot_route_comparison`) always return
PNG bytes and *optionally* also write to disk if given a path. This lets
`src/main.py` save a file for a CLI user and `src/webapp/app.py` stream the
same bytes directly over HTTP, from the same drawing code, with no
duplication.

## Data flow (one full run)

```
ScenarioConfig --> generate_scenario() --> FlightNetwork
FlightNetwork + ObjectiveWeights --> brute_force_solution() --> PathResult (ground truth, small n only)
FlightNetwork + ObjectiveWeights --> classical_shortest_path() --> PathResult (Dijkstra)
FlightNetwork + ObjectiveWeights --> build_qubo() --> QUBOResult (pure Python)
QUBOResult --> qubo_to_ising() --> (SparsePauliOp, offset)   [only file importing qiskit_optimization]
(SparsePauliOp, QAOAConfig) --> solve_qaoa() --> QAOAResult (trained params, measurement counts)
QUBOResult + QAOAResult.counts --> decode_all_counts() --> list[DecodedSolution]
DecodedSolution list --> best_valid_solution(), success_probability()
PathResult (classical) + DecodedSolution (best) --> approximation_ratio(), optimality_gap()
FlightNetwork + both paths --> plot_route_comparison() --> PNG bytes
```

Every arrow above is an actual function call exercised by
`tests/test_end_to_end.py`, not just a design diagram.
