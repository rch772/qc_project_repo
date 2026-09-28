from __future__ import annotations

import networkx as nx
import numpy as np

from src.models.network import Edge, FlightNetwork, Node
from src.utils.config import ScenarioConfig
from src.utils.logging_config import get_logger

logger = get_logger(__name__)

# Sampling ranges for the physical realism factors. Kept as module constants
# (rather than buried magic numbers) so they're easy to find and cite in
# docs/problem_formulation.md.
FUEL_MULTIPLIER_RANGE = (0.8, 1.3)   # simulates aircraft load / altitude variation
WIND_FACTOR_RANGE = (0.75, 1.25)     # >1 = tailwind (faster), <1 = headwind (slower)
WEATHER_PENALTY_FRACTION_OF_DISTANCE = (0.2, 0.6)


def _euclidean(a: Node, b: Node) -> float:
    return float(np.hypot(a.x - b.x, a.y - b.y))


def _knn_pairs(nodes: list[Node], k: int) -> set[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    for i, ni in enumerate(nodes):
        dists = sorted(
            ((j, _euclidean(ni, nj)) for j, nj in enumerate(nodes) if j != i),
            key=lambda item: (item[1], item[0]),
        )
        for j, _ in dists[:k]:
            pairs.add((min(ni.id, nodes[j].id), max(ni.id, nodes[j].id)))
    return pairs


def _extra_random_pairs(
    nodes: list[Node], existing: set[tuple[int, int]], fraction: float, rng: np.random.Generator
) -> set[tuple[int, int]]:
    n = len(nodes)
    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if (i, j) not in existing]
    all_pairs.sort()
    if not all_pairs:
        return set()
    num_extra = round(fraction * len(existing))
    num_extra = min(num_extra, len(all_pairs))
    if num_extra == 0:
        return set()
    idx = rng.choice(len(all_pairs), size=num_extra, replace=False)
    return {all_pairs[i] for i in sorted(idx.tolist())}


def _repair_connectivity(nodes: list[Node], pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
    """Deterministically bridge connected components (nearest node pair
    between the two lowest-indexed components) until the graph is one
    connected component."""
    pairs = set(pairs)
    g = nx.Graph()
    g.add_nodes_from(n.id for n in nodes)
    g.add_edges_from(pairs)
    by_id = {n.id: n for n in nodes}

    while True:
        comps = sorted((sorted(c) for c in nx.connected_components(g)), key=lambda c: c[0])
        if len(comps) <= 1:
            break
        comp_a, comp_b = comps[0], comps[1]
        best_pair = None
        best_dist = float("inf")
        for i in comp_a:
            for j in comp_b:
                d = _euclidean(by_id[i], by_id[j])
                if d < best_dist:
                    best_dist = d
                    best_pair = (min(i, j), max(i, j))
        pairs.add(best_pair)
        g.add_edge(*best_pair)
        logger.info(f"Bridging disconnected components with edge {best_pair} (dist={best_dist:.2f})")
    return pairs


def _protected_spanning_tree_pairs(nodes: list[Node], pairs: set[tuple[int, int]]) -> set[tuple[int, int]]:
    g = nx.Graph()
    g.add_nodes_from(n.id for n in nodes)
    g.add_edges_from(pairs)
    tree = nx.minimum_spanning_tree(g)  # unweighted here -> deterministic given fixed edge order
    return {(min(u, v), max(u, v)) for u, v in tree.edges()}


def generate_scenario(config: ScenarioConfig) -> FlightNetwork:
    rng = np.random.default_rng(config.seed)

    xs = rng.uniform(0.0, config.area_size, size=config.num_nodes)
    ys = rng.uniform(0.0, config.area_size, size=config.num_nodes)
    nodes = [Node(id=i, name=f"WP{i}", x=float(xs[i]), y=float(ys[i])) for i in range(config.num_nodes)]

    knn_pairs = _knn_pairs(nodes, config.k_nearest)
    extra_pairs = _extra_random_pairs(nodes, knn_pairs, config.extra_edge_fraction, rng)
    candidate_pairs = knn_pairs | extra_pairs
    candidate_pairs = _repair_connectivity(nodes, candidate_pairs)

    protected = _protected_spanning_tree_pairs(nodes, candidate_pairs)
    removable = sorted(candidate_pairs - protected)
    num_restricted = min(round(config.restricted_fraction * len(removable)), len(removable))
    restricted: set[tuple[int, int]] = set()
    if num_restricted > 0 and removable:
        idx = rng.choice(len(removable), size=num_restricted, replace=False)
        restricted = {removable[i] for i in sorted(idx.tolist())}

    final_pairs = sorted(candidate_pairs - restricted)

    # Build directed edges (there and back) with independently sampled factors.
    directed_edges: list[tuple[int, int]] = []
    for (i, j) in final_pairs:
        directed_edges.append((i, j))
        directed_edges.append((j, i))

    num_weather = round(config.weather_fraction * len(directed_edges))
    num_weather = min(num_weather, len(directed_edges))
    weather_idx: set[int] = set()
    if num_weather > 0:
        idx = rng.choice(len(directed_edges), size=num_weather, replace=False)
        weather_idx = set(idx.tolist())

    by_id = {n.id: n for n in nodes}
    network = FlightNetwork(
        metadata={
            "seed": config.seed,
            "num_nodes": config.num_nodes,
            "area_size": config.area_size,
            "k_nearest": config.k_nearest,
            "extra_edge_fraction": config.extra_edge_fraction,
            "restricted_fraction": config.restricted_fraction,
            "weather_fraction": config.weather_fraction,
            "base_fuel_rate": config.base_fuel_rate,
            "base_speed": config.base_speed,
            "num_restricted_pairs_excluded": len(restricted),
        }
    )
    for n in nodes:
        network.add_node(n)

    for k, (u, v) in enumerate(directed_edges):
        distance = _euclidean(by_id[u], by_id[v])
        fuel_mult = rng.uniform(*FUEL_MULTIPLIER_RANGE)
        wind_factor = rng.uniform(*WIND_FACTOR_RANGE)
        fuel_cost = distance * config.base_fuel_rate * fuel_mult
        speed = config.base_speed * wind_factor
        time_cost = distance / speed
        weather_penalty = 0.0
        if k in weather_idx:
            weather_penalty = distance * rng.uniform(*WEATHER_PENALTY_FRACTION_OF_DISTANCE)
        network.add_edge(
            Edge(
                u=u,
                v=v,
                distance=distance,
                fuel_cost=fuel_cost,
                time_cost=time_cost,
                weather_penalty=weather_penalty,
            )
        )

    origin = config.origin if config.origin is not None else 0
    destination = config.destination if config.destination is not None else config.num_nodes - 1
    network.metadata["origin"] = origin
    network.metadata["destination"] = destination

    if not network.is_reachable(origin, destination):
        # Should not happen given the connectivity repair above, but fail
        # loudly rather than silently returning an unusable scenario.
        raise RuntimeError(
            f"Generated scenario (seed={config.seed}) has no path from "
            f"{origin} to {destination}; this indicates a bug in the "
            f"connectivity-repair step."
        )

    logger.info(
        f"Generated scenario: {network.num_nodes} nodes, {network.num_edges} directed edges, "
        f"{len(restricted)} pairs excluded as restricted, seed={config.seed}"
    )
    return network
