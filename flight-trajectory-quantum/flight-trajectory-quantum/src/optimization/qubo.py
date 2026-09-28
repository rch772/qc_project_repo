from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

from src.models.network import FlightNetwork, InvalidNodeError
from src.utils.config import ObjectiveWeights
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class QUBOResult:
    """A QUBO instance: minimize x^T Q x (upper-triangular form) i.e.

        H(x) = constant + sum_i linear[i] * x_i
                         + sum_{(i,j), i<j} quadratic[(i,j)] * x_i * x_j
    """

    edges: list[tuple[int, int]]                       # variable index -> directed edge (u, v)
    edge_index: dict[tuple[int, int], int]              # directed edge -> variable index
    linear: dict[int, float]
    quadratic: dict[tuple[int, int], float]
    constant: float
    penalty_lambda: float
    origin: int
    destination: int
    edge_costs: dict[int, float] = field(default_factory=dict)  # raw (unpenalized) per-edge cost, for reporting

    @property
    def num_variables(self) -> int:
        return len(self.edges)

    def energy(self, x: Sequence[int]) -> float:
        """H(x) for a given 0/1 assignment. Used both by the quantum decoder
        (to score candidate measurement outcomes) and by tests that
        brute-force sweep all 2^n assignments to confirm the QUBO minimum
        matches the known shortest path."""
        if len(x) != self.num_variables:
            raise ValueError(f"Expected {self.num_variables} bits, got {len(x)}.")
        total = self.constant
        for i, coeff in self.linear.items():
            total += coeff * x[i]
        for (i, j), coeff in self.quadratic.items():
            total += coeff * x[i] * x[j]
        return total

    def decode_edges(self, x: Sequence[int]) -> list[tuple[int, int]]:
        """Directed edges selected (x_e == 1) by a bit assignment, in
        variable-index order (NOT necessarily a valid ordered path -- see
        src/quantum/decoder.py for turning this edge set into a path)."""
        return [self.edges[i] for i, bit in enumerate(x) if bit == 1]


def build_qubo(
    network: FlightNetwork,
    origin: int,
    destination: int,
    weights: ObjectiveWeights,
    penalty_lambda: float | None = None,
) -> QUBOResult:
    if origin not in network.nodes:
        raise InvalidNodeError(f"Origin node {origin} not in the network.")
    if destination not in network.nodes:
        raise InvalidNodeError(f"Destination node {destination} not in the network.")
    if origin == destination:
        raise ValueError("Origin and destination must differ.")

    edges = sorted(network.edges.keys())
    edge_index = {e: i for i, e in enumerate(edges)}

    norm_factors = network.normalization_factors() if weights.normalize_components else None
    costs = [network.edge_cost(network.edges[e], weights, norm_factors) for e in edges]

    if penalty_lambda is None:
        total_cost = sum(costs)
        # See docs/qubo_derivation.md: any single constraint violation moves
        # g_n(x)^2 by an integer amount >= 1, so lambda need only exceed the
        # largest possible objective *saving* from violating a constraint,
        # which is bounded above by the sum of all edge costs. A factor of 2
        # is kept as a safety margin.
        penalty_lambda = 2.0 * total_cost if total_cost > 0 else 1.0

    linear: dict[int, float] = defaultdict(float)
    quadratic: dict[tuple[int, int], float] = defaultdict(float)
    constant = 0.0

    for i, c in enumerate(costs):
        linear[i] += c

    incident_by_node: dict[int, list[tuple[int, int]]] = {n: [] for n in network.nodes}
    for idx, (u, v) in enumerate(edges):
        incident_by_node[u].append((idx, 1))    # leaving u contributes +1 to outflow(u)
        incident_by_node[v].append((idx, -1))   # entering v contributes -1 (i.e. +1 to inflow(v))

    for n in network.nodes:
        b_n = 1 if n == origin else (-1 if n == destination else 0)
        incident = incident_by_node[n]
        for idx, s in incident:
            # s*s == 1 always (s is +-1); kept explicit to mirror the derivation.
            linear[idx] += penalty_lambda * (s * s - 2 * b_n * s)
        for a in range(len(incident)):
            idx_a, s_a = incident[a]
            for b in range(a + 1, len(incident)):
                idx_b, s_b = incident[b]
                i, j = (idx_a, idx_b) if idx_a < idx_b else (idx_b, idx_a)
                quadratic[(i, j)] += penalty_lambda * 2 * s_a * s_b
        constant += penalty_lambda * (b_n ** 2)

    logger.info(
        f"Built QUBO: {len(edges)} variables (qubits), penalty_lambda={penalty_lambda:.4f}, "
        f"origin={origin}, destination={destination}"
    )

    return QUBOResult(
        edges=edges,
        edge_index=edge_index,
        linear=dict(linear),
        quadratic=dict(quadratic),
        constant=constant,
        penalty_lambda=penalty_lambda,
        origin=origin,
        destination=destination,
        edge_costs={i: c for i, c in enumerate(costs)},
    )
