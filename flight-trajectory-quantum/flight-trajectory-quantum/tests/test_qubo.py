import itertools

import pytest

from src.classical.brute_force import brute_force_solution
from src.models.network import Edge, FlightNetwork, Node
from src.optimization.qubo import build_qubo
from src.utils.config import ObjectiveWeights


def _diamond_network() -> FlightNetwork:
    """5 nodes, 12 directed edges, with one deliberately expensive
    direct shortcut (0->3, 3->0) that a correct solver must avoid."""
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


def test_qubo_variable_count_equals_num_edges():
    net = _diamond_network()
    weights = ObjectiveWeights(normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    assert qubo.num_variables == net.num_edges == 12


def test_qubo_variable_edge_mapping_is_consistent():
    net = _diamond_network()
    weights = ObjectiveWeights(normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    for edge, idx in qubo.edge_index.items():
        assert qubo.edges[idx] == edge


def test_all_zero_assignment_is_infeasible_and_penalized():
    """The empty edge set violates the origin's flow constraint, so its
    QUBO energy must include a nonzero penalty contribution and therefore
    exceed the energy of any valid path."""
    net = _diamond_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    zero_bits = [0] * qubo.num_variables
    valid_edges = [(0, 1), (1, 3), (3, 4)]
    valid_bits = [1 if qubo.edges[i] in valid_edges else 0 for i in range(qubo.num_variables)]

    zero_energy = qubo.energy(zero_bits)
    valid_energy = qubo.energy(valid_bits)
    assert zero_energy > valid_energy


def test_branching_assignment_is_penalized_above_valid_path():
    """Selecting a valid path PLUS one extra dangling edge should cost more
    than the valid path alone (extra edge adds both raw cost and, because
    it breaks flow conservation at its endpoints, constraint penalty)."""
    net = _diamond_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    qubo = build_qubo(net, 0, 4, weights)
    valid_edges = [(0, 1), (1, 3), (3, 4)]
    valid_bits = [1 if qubo.edges[i] in valid_edges else 0 for i in range(qubo.num_variables)]
    branching_edges = valid_edges + [(0, 2)]  # extra outgoing edge from origin
    branching_bits = [1 if qubo.edges[i] in branching_edges else 0 for i in range(qubo.num_variables)]

    assert qubo.energy(branching_bits) > qubo.energy(valid_bits)


def test_qubo_global_minimum_matches_brute_force_shortest_path():
    """Exhaustively sweep all 2^12 assignments and confirm the minimum
    -energy assignment decodes to exactly the brute-force-verified shortest
    path. This is the central correctness proof for the QUBO derivation."""
    net = _diamond_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    origin, destination = 0, 4

    bf_result, _ = brute_force_solution(net, origin, destination, weights)
    qubo = build_qubo(net, origin, destination, weights)

    n = qubo.num_variables
    best_energy = None
    best_bits = None
    for bits in itertools.product([0, 1], repeat=n):
        e = qubo.energy(bits)
        if best_energy is None or e < best_energy:
            best_energy = e
            best_bits = bits

    expected_edges = list(zip(bf_result.path[:-1], bf_result.path[1:]))
    expected_bits = tuple(1 if qubo.edges[i] in expected_edges else 0 for i in range(n))

    assert best_bits == expected_bits
    assert best_energy == pytest.approx(bf_result.total_cost, abs=1e-9)


def test_penalty_lambda_can_be_overridden():
    net = _diamond_network()
    weights = ObjectiveWeights(normalize_components=False)
    qubo_default = build_qubo(net, 0, 4, weights)
    qubo_custom = build_qubo(net, 0, 4, weights, penalty_lambda=1000.0)
    assert qubo_custom.penalty_lambda == 1000.0
    assert qubo_custom.penalty_lambda != qubo_default.penalty_lambda


def test_build_qubo_rejects_same_origin_and_destination():
    net = _diamond_network()
    with pytest.raises(ValueError):
        build_qubo(net, 0, 0, ObjectiveWeights())
