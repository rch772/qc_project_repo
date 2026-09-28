# QUBO Derivation

Implementation: `src/optimization/qubo.py::build_qubo`. This document
derives exactly what that function computes; nothing here is aspirational
or approximate.

## Variables

One binary variable per **directed edge**: for `e = (u, v) in E`,

```
x_e in {0, 1},   x_e = 1  iff  edge e is part of the selected trajectory
```

If the network has `m` directed edges, the QUBO has `m` variables, i.e.
`m` qubits once converted to Ising form (`src/quantum/ising.py`).

## Objective term

```
C(x) = sum_{e in E} cost(e) * x_e
```

where `cost(e)` is the same normalized, weighted per-edge cost defined in
`docs/problem_formulation.md`.

## Feasibility: flow conservation

Selecting an edge subset is only a valid trajectory if it forms a single
simple path from `origin` to `destination`. This project encodes that with
one flow-conservation constraint per node:

```
outflow(n) - inflow(n) = b_n     for every n in V

outflow(n) = sum_{e = (n, *)} x_e     (edges leaving n)
inflow(n)  = sum_{e = (*, n)} x_e     (edges entering n)

b_n = { +1  if n == origin
      { -1  if n == destination
      {  0  otherwise
```

Intuition: at the origin, exactly one more edge must leave than arrive (net
outflow 1); at the destination, exactly one more edge must arrive than
leave (net inflow 1); everywhere else, whatever comes in must also go out
(nothing is picked up or dropped mid-route).

**Why this doesn't need an explicit "no cycles" constraint.** A set of
edges satisfying flow conservation everywhere could, in principle, include
the origin-to-destination path *plus* an unrelated disjoint cycle
elsewhere (a cycle has net flow 0 at every one of its nodes, so it doesn't
violate any `b_n`). Two things handle this without extra QUBO terms: (1)
since every edge cost is `>= 0`, adding a gratuitous cycle can only
increase `C(x)`, so it is never part of a *minimum*-cost feasible solution
-- the QUBO's own objective already discourages it; (2) as a belt-and-braces
check, `src/quantum/decoder.py` explicitly walks the selected edges from
origin to destination and flags a "disjoint cycle or dangling segment" as
invalid if any selected edges are left over after the walk reaches the
destination, rather than assuming flow conservation alone guarantees a
clean path (see `tests/test_decoder.py::test_decode_disjoint_cycle_is_invalid`,
which constructs exactly this pathological case and confirms it's rejected).

## Penalty formulation

Each constraint is squared and added with coefficient `lambda`:

```
H(x) = C(x) + lambda * sum_{n in V} (outflow(n) - inflow(n) - b_n)^2
```

Expanding the square for a single node `n`, and writing `s_{n,e} in {+1, -1, 0}`
for edge `e`'s contribution to node `n` (`+1` if `e` leaves `n`, `-1` if `e`
enters `n`, `0` otherwise):

```
g_n(x) = sum_e s_{n,e} x_e - b_n

g_n(x)^2 = sum_e sum_f s_{n,e} s_{n,f} x_e x_f  -  2 b_n sum_e s_{n,e} x_e  +  b_n^2
```

Using `x_e^2 = x_e` for binary variables (so the `e == f` diagonal terms
fold into linear coefficients) and combining the `(e,f)`/`(f,e)` off-diagonal
pairs (each unordered pair `{e,f}` with `e < f` appears twice in the double
sum, each time contributing `s_{n,e} s_{n,f} x_e x_f`), node `n` contributes:

```
linear[e]      += lambda * (s_{n,e}^2 - 2 b_n s_{n,e})      for every edge e incident to n
quadratic[e,f] += lambda * 2 * s_{n,e} * s_{n,f}             for every pair e < f both incident to n
constant       += lambda * b_n^2
```

`build_qubo` computes exactly these three updates, looping over each node's
incident edges only (not all `m^2` edge pairs), then adds the objective's
linear term `cost(e)` on top. The result is a `QUBOResult` holding
`linear: dict[int, float]`, `quadratic: dict[(int,int), float]`, and
`constant: float`, such that for any bit assignment `x`:

```
H(x) = constant + sum_i linear[i]*x_i + sum_{i<j} quadratic[i,j]*x_i*x_j
```

(`QUBOResult.energy(x)` evaluates exactly this expression -- it is the
function both the correctness test below and the quantum decoder use to
score a candidate bitstring.)

## Choosing lambda

Any single constraint violation changes some `g_n(x)^2` by an integer
`>= 1` (since `s_{n,e}` and `b_n` are integers). For the penalty to
guarantee that *no* infeasible assignment beats *every* feasible one, it
suffices that `lambda` exceed the largest possible objective **saving** an
infeasible assignment could offer over a feasible one -- and that saving is
bounded above by the sum of all edge costs (dropping every edge would save
at most that much, and can't save more). `build_qubo`'s default is

```
lambda = 2 * sum_e cost(e)      (2x safety margin), or 1.0 if all costs are 0
```

`penalty_lambda` can be overridden explicitly (`build_qubo(..., penalty_lambda=...)`,
also exposed as `QAOAConfig.penalty_lambda` and as a CLI/web app field) --
useful for studying how the penalty scale affects QAOA trainability (a
larger `lambda` guarantees feasibility-optimum-equals-global-optimum more
robustly but also stretches the energy landscape, which was observed during
development to make QAOA *harder* to train at fixed circuit depth; see
`docs/experimental_methodology.md`).

## Validation

This derivation was checked, not just written down. On a hand-built 5-node,
12-edge network with a deliberately expensive direct shortcut (so a wrong
formulation could plausibly pick it):

```python
# tests/test_qubo.py::test_qubo_global_minimum_matches_brute_force_shortest_path
for bits in itertools.product([0, 1], repeat=12):   # all 4096 assignments
    energy = qubo.energy(bits)
    # track the minimum
assert minimum_energy_bitstring == bitstring_for(brute_force_optimal_path)
```

The exhaustive minimum over all `2^12` bit assignments is exactly the
brute-force-verified shortest path, with the QUBO's minimum energy equal to
the path's true cost to within floating-point tolerance. This test runs as
part of `python -m pytest` and is the project's central correctness proof
for the QUBO formulation -- everything downstream (Ising conversion, QAOA)
is only meaningful because this holds.

Two further tests target the penalty mechanism specifically:
`test_all_zero_assignment_is_infeasible_and_penalized` (the empty edge set,
which violates the origin's flow constraint, scores worse than the true
optimum) and `test_branching_assignment_is_penalized_above_valid_path` (a
valid path plus one dangling extra edge scores worse than the path alone).
