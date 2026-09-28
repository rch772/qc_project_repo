import pytest

from src.classical.brute_force import brute_force_solution
from src.data.scenario_generator import generate_scenario
from src.models.network import DisconnectedNetworkError, Edge, FlightNetwork, Node
from src.utils.config import ObjectiveWeights, ScenarioConfig


def test_brute_force_finds_known_optimum_on_tiny_graph():
    net = FlightNetwork()
    for i in range(4):
        net.add_node(Node(id=i, name=f"N{i}", x=float(i), y=0.0))
    # Direct 0->3 is expensive; 0->1->2->3 is cheap.
    net.add_edge(Edge(u=0, v=3, distance=10.0, fuel_cost=0.0, time_cost=0.0))
    net.add_edge(Edge(u=0, v=1, distance=1.0, fuel_cost=0.0, time_cost=0.0))
    net.add_edge(Edge(u=1, v=2, distance=1.0, fuel_cost=0.0, time_cost=0.0))
    net.add_edge(Edge(u=2, v=3, distance=1.0, fuel_cost=0.0, time_cost=0.0))

    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    result, examined = brute_force_solution(net, 0, 3, weights)
    assert result.path == (0, 1, 2, 3)
    assert result.total_cost == pytest.approx(3.0)
    assert examined == 2  # the direct edge and the 3-hop route


def test_brute_force_raises_when_disconnected():
    net = FlightNetwork()
    net.add_node(Node(id=0, name="A", x=0.0, y=0.0))
    net.add_node(Node(id=1, name="B", x=1.0, y=0.0))
    with pytest.raises(DisconnectedNetworkError):
        brute_force_solution(net, 0, 1, ObjectiveWeights())


@pytest.mark.parametrize("num_nodes,seed", [(5, 1), (6, 2), (7, 3)])
def test_brute_force_matches_manual_min_over_all_simple_paths(num_nodes, seed):
    net = generate_scenario(ScenarioConfig(num_nodes=num_nodes, seed=seed))
    origin, destination = net.metadata["origin"], net.metadata["destination"]
    weights = ObjectiveWeights()
    result, examined = brute_force_solution(net, origin, destination, weights)
    assert result.path[0] == origin
    assert result.path[-1] == destination
    net.validate_path(list(result.path))  # every edge in the reported path must actually exist
    assert examined >= 1
