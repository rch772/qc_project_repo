import pytest

from src.classical.brute_force import brute_force_solution
from src.classical.dijkstra import classical_shortest_path
from src.models.network import Edge, FlightNetwork, Node
from src.optimization.metrics import approximation_ratio, optimality_gap
from src.optimization.qubo import build_qubo
from src.quantum.decoder import best_valid_solution, decode_all_counts, success_probability
from src.quantum.qaoa_solver import solve_qaoa
from src.utils.config import ObjectiveWeights, QAOAConfig


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


def test_full_pipeline_runs_and_produces_consistent_metrics():
    net = _diamond_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    origin, destination = 0, 4

    bf_result, _ = brute_force_solution(net, origin, destination, weights)
    classical_result = classical_shortest_path(net, origin, destination, weights)
    # Classical Dijkstra must agree with brute-force ground truth (Section 15
    # cross-validation requirement) before quantum is even in the picture.
    assert classical_result.total_cost == pytest.approx(bf_result.total_cost)

    qubo = build_qubo(net, origin, destination, weights)
    assert qubo.num_variables == net.num_edges

    qaoa_config = QAOAConfig(reps=2, optimizer="COBYLA", max_iter=80, shots=2000, seed=11)
    qaoa_result = solve_qaoa(qubo, qaoa_config)
    assert qaoa_result.num_qubits == qubo.num_variables
    assert sum(qaoa_result.counts.values()) == qaoa_config.shots

    decoded = decode_all_counts(qubo, qaoa_result.counts, qaoa_result.shots)
    assert len(decoded) > 0
    success_p = success_probability(decoded)
    assert 0.0 <= success_p <= 1.0

    best = best_valid_solution(decoded)
    if best is not None:
        # Any valid decoded trajectory can never beat the true optimum
        # (brute force enumerates every simple path), only match or exceed it.
        assert best.qubo_energy >= bf_result.total_cost - 1e-6
        net.validate_path(list(best.path))
        assert best.path[0] == origin
        assert best.path[-1] == destination

        gap = optimality_gap(best.qubo_energy, bf_result.total_cost)
        ratio = approximation_ratio(best.qubo_energy, bf_result.total_cost)
        assert gap >= -1e-9
        assert ratio >= 1.0 - 1e-9
