import pytest

from src.models.network import Edge, FlightNetwork, Node
from src.optimization.qubo import build_qubo
from src.quantum.decoder import (
    best_valid_solution,
    decode_all_counts,
    decode_bitstring,
    success_probability,
)
from src.utils.config import ObjectiveWeights


def _diamond_network() -> FlightNetwork:
    net = FlightNetwork()
    for i, (x, y) in enumerate([(0, 0), (1, 1), (1, -1), (2, 0), (3, 0)]):
        net.add_node(Node(id=i, name=f"N{i}", x=x, y=y))
    raw_edges = [
        (0, 1, 1.5), (1, 0, 1.6),
        (0, 2, 1.4), (2, 0, 1.5),
        (1, 3, 1.5), (3, 1, 1.5),
        (2, 3, 1.7), (3, 2, 1.7),
        (0, 3, 5.0), (3, 0, 5.0),
        (3, 4, 1.0), (4, 3, 1.0),
    ]
    for (u, v, d) in raw_edges:
        net.add_edge(Edge(u=u, v=v, distance=d, fuel_cost=d * 2, time_cost=d * 0.5, weather_penalty=0.0))
    return net


def _bits_for(qubo, edges):
    return "".join("1" if qubo.edges[i] in edges else "0" for i in range(qubo.num_variables))


def test_decode_valid_path_bitstring():
    net = _diamond_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    bits = _bits_for(qubo, [(0, 1), (1, 3), (3, 4)])
    decoded = decode_bitstring(qubo, bits, count=500, shots=1000)
    assert decoded.is_valid_path
    assert decoded.path == (0, 1, 3, 4)
    assert decoded.validation_error is None
    assert decoded.probability == pytest.approx(0.5)
    assert decoded.qubo_energy == pytest.approx(4.0)


def test_decode_empty_bitstring_is_invalid():
    net = _diamond_network()
    qubo = build_qubo(net, 0, 4, ObjectiveWeights(normalize_components=False))
    bits = "0" * qubo.num_variables
    decoded = decode_bitstring(qubo, bits, count=10, shots=100)
    assert not decoded.is_valid_path
    assert "no edges selected" in decoded.validation_error


def test_decode_branching_bitstring_is_invalid():
    net = _diamond_network()
    weights = ObjectiveWeights(normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    # origin (0) has two outgoing selected edges -> branching, not a path.
    bits = _bits_for(qubo, [(0, 1), (0, 2), (1, 3), (3, 4)])
    decoded = decode_bitstring(qubo, bits, count=10, shots=100)
    assert not decoded.is_valid_path
    assert "branching" in decoded.validation_error


def test_decode_disjoint_cycle_is_invalid():
    """A bitstring can satisfy per-node flow conservation everywhere (so it
    is not caught by a branching/merging check) while still not being a
    single path, if it contains a separate cycle disjoint from the
    origin->destination walk. The decoder must catch this via the
    edge-count check, not just per-node degree."""
    net = FlightNetwork()
    for i in range(5):
        net.add_node(Node(id=i, name=f"N{i}", x=float(i), y=0.0))
    net.add_edge(Edge(u=0, v=1, distance=1.0, fuel_cost=0.0, time_cost=0.0))
    net.add_edge(Edge(u=1, v=2, distance=1.0, fuel_cost=0.0, time_cost=0.0))
    net.add_edge(Edge(u=3, v=4, distance=1.0, fuel_cost=0.0, time_cost=0.0))
    net.add_edge(Edge(u=4, v=3, distance=1.0, fuel_cost=0.0, time_cost=0.0))

    weights = ObjectiveWeights(normalize_components=False)
    qubo = build_qubo(net, 0, 2, weights)
    bits = _bits_for(qubo, [(0, 1), (1, 2), (3, 4), (4, 3)])
    decoded = decode_bitstring(qubo, bits, count=10, shots=100)
    assert not decoded.is_valid_path
    assert "disjoint cycle" in decoded.validation_error or "dangling" in decoded.validation_error


def test_decode_all_counts_sorted_by_count_desc():
    net = _diamond_network()
    weights = ObjectiveWeights(normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    counts = {
        _bits_for(qubo, [(0, 1), (1, 3), (3, 4)]): 300,
        _bits_for(qubo, [(0, 2), (2, 3), (3, 4)]): 700,
        "0" * qubo.num_variables: 100,
    }
    decoded = decode_all_counts(qubo, counts, shots=1100)
    assert [d.count for d in decoded] == [700, 300, 100]


def test_best_valid_solution_picks_lowest_energy_among_valid():
    net = _diamond_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    counts = {
        _bits_for(qubo, [(0, 1), (1, 3), (3, 4)]): 100,   # cost 4.0, optimal
        _bits_for(qubo, [(0, 2), (2, 3), (3, 4)]): 400,   # cost 4.1, higher count but worse cost
        "0" * qubo.num_variables: 500,                     # invalid, highest count
    }
    decoded = decode_all_counts(qubo, counts, shots=1000)
    best = best_valid_solution(decoded)
    assert best is not None
    assert best.path == (0, 1, 3, 4)
    assert best.qubo_energy == pytest.approx(4.0)


def test_best_valid_solution_none_when_nothing_valid():
    net = _diamond_network()
    qubo = build_qubo(net, 0, 4, ObjectiveWeights(normalize_components=False))
    counts = {"0" * qubo.num_variables: 1000}
    decoded = decode_all_counts(qubo, counts, shots=1000)
    assert best_valid_solution(decoded) is None


def test_success_probability_sums_only_valid_paths():
    net = _diamond_network()
    weights = ObjectiveWeights(normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    counts = {
        _bits_for(qubo, [(0, 1), (1, 3), (3, 4)]): 150,
        _bits_for(qubo, [(0, 2), (2, 3), (3, 4)]): 250,
        "0" * qubo.num_variables: 600,
    }
    decoded = decode_all_counts(qubo, counts, shots=1000)
    assert success_probability(decoded) == pytest.approx(0.4)
