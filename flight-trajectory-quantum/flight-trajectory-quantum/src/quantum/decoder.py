from __future__ import annotations

from dataclasses import dataclass

from src.optimization.qubo import QUBOResult
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class DecodedSolution:
    bitstring: str                     # variable-order "x0 x1 ... x_{n-1}"
    count: int
    probability: float
    selected_edges: tuple[tuple[int, int], ...]
    qubo_energy: float
    is_valid_path: bool
    path: tuple[int, ...] | None        # None if not a valid path
    validation_error: str | None        # None if valid


def _walk_path(qubo: QUBOResult, selected_edges: list[tuple[int, int]]) -> tuple[tuple[int, ...] | None, str | None]:
    """Attempt to reconstruct a simple origin->destination path using
    exactly the selected edges. Returns (path, None) on success or
    (None, reason) on failure."""
    if not selected_edges:
        return None, "no edges selected"

    out_deg: dict[int, list[int]] = {}
    in_deg_count: dict[int, int] = {}
    for u, v in selected_edges:
        out_deg.setdefault(u, []).append(v)
        in_deg_count[v] = in_deg_count.get(v, 0) + 1

    for node, targets in out_deg.items():
        if len(targets) > 1:
            return None, f"branching: node {node} has {len(targets)} outgoing selected edges"
    for node, count in in_deg_count.items():
        if count > 1:
            return None, f"merging: node {node} has {count} incoming selected edges"

    origin, destination = qubo.origin, qubo.destination
    if origin not in out_deg:
        return None, f"origin {origin} has no outgoing selected edge"
    if in_deg_count.get(origin, 0) != 0:
        return None, f"origin {origin} has an incoming selected edge (not a valid path source)"
    if destination in out_deg:
        return None, f"destination {destination} has an outgoing selected edge (not a valid path sink)"
    if in_deg_count.get(destination, 0) != 1:
        return None, f"destination {destination} has no incoming selected edge"

    path = [origin]
    visited_edges = 0
    current = origin
    while current != destination:
        if current not in out_deg:
            return None, f"path breaks at node {current} before reaching destination {destination}"
        nxt = out_deg[current][0]
        path.append(nxt)
        visited_edges += 1
        current = nxt
        if visited_edges > len(selected_edges):
            return None, "cycle detected while walking the path"

    if visited_edges != len(selected_edges):
        return None, (
            f"{len(selected_edges) - visited_edges} selected edge(s) are not part of the "
            f"origin->destination path (disjoint cycle or dangling segment)"
        )
    if len(set(path)) != len(path):
        return None, "path revisits a node"

    return tuple(path), None


def decode_bitstring(qubo: QUBOResult, bitstring: str, count: int, shots: int) -> DecodedSolution:
    bits = [int(b) for b in bitstring]
    selected_edges = qubo.decode_edges(bits)
    energy = qubo.energy(bits)
    path, error = _walk_path(qubo, selected_edges)
    return DecodedSolution(
        bitstring=bitstring,
        count=count,
        probability=count / shots if shots else 0.0,
        selected_edges=tuple(selected_edges),
        qubo_energy=energy,
        is_valid_path=path is not None,
        path=path,
        validation_error=error,
    )


def decode_all_counts(qubo: QUBOResult, counts: dict[str, int], shots: int) -> list[DecodedSolution]:
    """Decode every distinct measured bitstring, sorted by measured
    probability (descending)."""
    decoded = [decode_bitstring(qubo, bs, c, shots) for bs, c in counts.items()]
    decoded.sort(key=lambda d: d.count, reverse=True)
    return decoded


def best_valid_solution(decoded: list[DecodedSolution]) -> DecodedSolution | None:
    """Among all decoded outcomes, the valid-path one with the lowest QUBO
    energy (i.e. lowest trajectory cost), or None if no measured bitstring
    decoded to a valid path."""
    valid = [d for d in decoded if d.is_valid_path]
    if not valid:
        return None
    return min(valid, key=lambda d: d.qubo_energy)


def success_probability(decoded: list[DecodedSolution]) -> float:
    """Total measured probability mass that decoded to ANY valid
    origin->destination path (not just the best one)."""
    return sum(d.probability for d in decoded if d.is_valid_path)
