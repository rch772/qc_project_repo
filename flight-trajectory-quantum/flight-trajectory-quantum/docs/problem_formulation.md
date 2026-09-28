# Problem Formulation

## Graph

A flight network is a directed graph `G = (V, E)`:

- `V = {0, 1, ..., n-1}`: waypoints, each with 2D coordinates `(x_i, y_i)`
  (`src/models/network.py::Node`).
- `E subseteq V x V`: directed edges (flight legs). Direction matters:
  edge `(u, v)` and edge `(v, u)` are two independent edges with
  independently sampled cost factors, because a leg can genuinely cost
  differently in each direction (headwind vs. tailwind -- see below).
  Self-loops are disallowed.

Each edge `(u, v) in E` carries four nonnegative raw quantities
(`src/models/network.py::Edge`):

- `distance(u,v)`: Euclidean distance between the two waypoints' coordinates.
- `fuel_cost(u,v)`: `distance x base_fuel_rate x fuel_multiplier`, where
  `fuel_multiplier ~ Uniform(0.8, 1.3)` is sampled independently per
  directed edge (simulating aircraft load / altitude variation -- so fuel
  is *not* simply proportional to distance, deliberately, per the project
  brief's caution against that assumption).
- `time_cost(u,v)`: `distance / (base_speed x wind_factor)`, where
  `wind_factor ~ Uniform(0.75, 1.25)` is sampled independently per directed
  edge (a `wind_factor > 1` in one direction models a tailwind; the
  opposite direction of the same physical leg typically gets a different,
  independently sampled factor, modeling that a headwind one way is a
  tailwind the other way -- though the sampling is independent, not
  perfectly anti-correlated, since real winds also vary with the specific
  route/altitude flown).
- `weather_penalty(u,v)`: `0` for most edges; for a randomly chosen
  fraction of edges, `Uniform(0.2, 0.6) x distance` (a storm cell or other
  weather hazard along that leg).

All four quantities are `>= 0` by construction
(`Edge.__post_init__` raises if any is negative).

## Restricted airspace

A fraction of candidate edges are excluded from `E` entirely at generation
time (not kept in the graph with a penalty) -- see `docs/architecture.md`
for why, and the guarantee (via a protected spanning tree) that this can
never disconnect the network. This means every path any solver in this
project can find is airspace-compliant by construction; there is no
separate "is this edge allowed" check anywhere downstream.

**Simplification, stated plainly:** real airspace restrictions are
time-varying and altitude-dependent (a corridor might be closed for an hour,
or only above/below a certain flight level); this project's restrictions
are static properties of an edge, decided once at scenario-generation time.

## Objective

For a simple path `P = (v_0 = origin, v_1, ..., v_k = destination)` where
every consecutive pair is an edge in `E`, the cost of the path is

```
cost(P) = sum_{l=0}^{k-1} edge_cost(v_l, v_{l+1})
```

where the per-edge combined cost is a weighted sum of the four normalized
components (`src/models/network.py::FlightNetwork.edge_cost`):

```
edge_cost(u,v) = alpha * distance(u,v)/D̄  +  beta * fuel_cost(u,v)/F̄
               + gamma * time_cost(u,v)/T̄  +  delta * weather_penalty(u,v)/W̄
```

`D̄, F̄, T̄, W̄` are the mean value of each raw component across *all* edges
in the network (`FlightNetwork.normalization_factors`; a component with
mean 0, e.g. no edge carries a weather penalty, uses 1.0 instead to avoid
division by zero). `alpha, beta, gamma, delta >= 0` are user-configurable
weights (`ObjectiveWeights`).

**Why normalize at all?** Distance, fuel, time and weather penalty live on
different raw numeric scales (this project's synthetic scenarios put
distance in the hundreds, fuel roughly 2-4x distance, time roughly
distance/8, weather penalty roughly 0.2-0.6x distance when present at all).
Without normalization, setting `alpha = beta = gamma = delta = 1` would not
mean "weigh all four factors equally" -- it would silently mean "let
whichever raw quantity happens to be numerically largest dominate the
route choice", which is exactly the "minimizing distance is not the same
as minimizing fuel" trap the project brief warns against. Normalizing by
each component's mean puts all four on a comparable scale first, so the
weights actually control relative importance. `normalize_components=True`
is the default; it can be turned off (`ObjectiveWeights(normalize_components=False)`)
to use raw units directly, which is occasionally useful for constructing
hand-verifiable test cases (see `tests/`).

## Optimization problem

```
minimize    cost(P)
over        simple paths P from origin to destination in G
```

"Simple" (no repeated nodes) is enforced by every solver in this project
(`FlightNetwork.validate_path` rejects a path that revisits a node; the
QUBO's flow-conservation constraints make a solution containing a
node-repeating structure infeasible except in the disjoint-cycle sense
handled explicitly by the decoder -- see `docs/qubo_derivation.md` and
`docs/quantum_method.md`).

Because every edge cost is `>= 0`, this is an ordinary nonnegative-weight
shortest-path problem for the classical solver
(`src/classical/dijkstra.py` uses this fact directly: Dijkstra's optimality
guarantee requires nonnegative weights, which holds here by construction),
and it is what `src/classical/brute_force.py` verifies exhaustively for
small instances by enumerating every simple path.

## What this project does NOT model

Real flight trajectory optimization additionally involves: aircraft
performance envelopes and weight/balance; live, evolving weather (not a
static per-edge penalty decided once); air traffic control clearances and
separation requirements; airspace class and procedural restrictions;
airport-specific departure/arrival procedures; altitude selection and
altitude-dependent fuel burn; and safety margins and contingency fuel. This
project abstracts all of that into four static per-edge numbers so the
combinatorial structure (choosing a path) can be studied in isolation --
it is a methodology demonstration, not an operational aviation system.
