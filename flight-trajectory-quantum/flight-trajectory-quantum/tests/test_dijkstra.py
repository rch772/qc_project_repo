import pytest

from src.classical.brute_force import brute_force_solution
from src.classical.dijkstra import classical_shortest_path
from src.data.scenario_generator import generate_scenario
from src.models.network import DisconnectedNetworkError, FlightNetwork, Node
from src.utils.config import ObjectiveWeights, ScenarioConfig


@pytest.mark.parametrize("num_nodes,seed", [(5, 1), (6, 2), (7, 3), (8, 4), (10, 5)])
def test_dijkstra_matches_brute_force_ground_truth(num_nodes, seed):
    net = generate_scenario(ScenarioConfig(num_nodes=num_nodes, seed=seed))
    origin, destination = net.metadata["origin"], net.metadata["destination"]
    weights = ObjectiveWeights()

    bf_result, _ = brute_force_solution(net, origin, destination, weights)
    dijkstra_result = classical_shortest_path(net, origin, destination, weights)

    assert dijkstra_result.total_cost == pytest.approx(bf_result.total_cost, rel=1e-9)


def test_dijkstra_raises_when_disconnected():
    net = FlightNetwork()
    net.add_node(Node(id=0, name="A", x=0.0, y=0.0))
    net.add_node(Node(id=1, name="B", x=1.0, y=0.0))
    with pytest.raises(DisconnectedNetworkError):
        classical_shortest_path(net, 0, 1, ObjectiveWeights())


def test_dijkstra_result_path_is_valid_in_network():
    net = generate_scenario(ScenarioConfig(num_nodes=10, seed=99))
    origin, destination = net.metadata["origin"], net.metadata["destination"]
    result = classical_shortest_path(net, origin, destination, ObjectiveWeights())
    net.validate_path(list(result.path))
    assert result.path[0] == origin
    assert result.path[-1] == destination
