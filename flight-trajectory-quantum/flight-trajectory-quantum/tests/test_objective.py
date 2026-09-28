import pytest

from src.models.network import Edge, FlightNetwork, InvalidPathError, Node
from src.models.objective import evaluate_path
from src.utils.config import ObjectiveWeights


def _network() -> FlightNetwork:
    net = FlightNetwork()
    net.add_node(Node(id=0, name="A", x=0.0, y=0.0))
    net.add_node(Node(id=1, name="B", x=1.0, y=0.0))
    net.add_node(Node(id=2, name="C", x=2.0, y=0.0))
    net.add_edge(Edge(u=0, v=1, distance=1.0, fuel_cost=3.0, time_cost=0.2, weather_penalty=0.0))
    net.add_edge(Edge(u=1, v=2, distance=2.0, fuel_cost=5.0, time_cost=0.4, weather_penalty=1.0))
    return net


def test_evaluate_path_component_breakdown_unweighted_sums():
    net = _network()
    weights = ObjectiveWeights(alpha=1.0, beta=1.0, gamma=1.0, delta=1.0, normalize_components=False)
    result = evaluate_path(net, [0, 1, 2], weights)
    assert result.distance == pytest.approx(3.0)
    assert result.fuel_cost == pytest.approx(8.0)
    assert result.time_cost == pytest.approx(0.6)
    assert result.weather_penalty == pytest.approx(1.0)
    assert result.total_cost == pytest.approx(3.0 + 8.0 + 0.6 + 1.0)


def test_evaluate_path_respects_weights():
    net = _network()
    distance_only = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    result = evaluate_path(net, [0, 1, 2], distance_only)
    assert result.total_cost == pytest.approx(3.0)


def test_evaluate_path_num_edges():
    net = _network()
    weights = ObjectiveWeights(normalize_components=False)
    result = evaluate_path(net, [0, 1, 2], weights)
    assert result.num_edges == 2


def test_evaluate_invalid_path_raises():
    net = _network()
    weights = ObjectiveWeights()
    with pytest.raises(InvalidPathError):
        evaluate_path(net, [0, 2], weights)  # no direct edge
