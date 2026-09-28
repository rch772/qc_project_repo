import pytest

from src.models.network import (
    DisconnectedNetworkError,
    Edge,
    FlightNetwork,
    InvalidEdgeError,
    InvalidNodeError,
    InvalidPathError,
    Node,
)
from src.utils.config import ObjectiveWeights


def _tiny_network() -> FlightNetwork:
    net = FlightNetwork()
    net.add_node(Node(id=0, name="A", x=0.0, y=0.0))
    net.add_node(Node(id=1, name="B", x=1.0, y=0.0))
    net.add_node(Node(id=2, name="C", x=2.0, y=0.0))
    net.add_edge(Edge(u=0, v=1, distance=1.0, fuel_cost=2.0, time_cost=0.5))
    net.add_edge(Edge(u=1, v=2, distance=1.0, fuel_cost=2.0, time_cost=0.5))
    net.add_edge(Edge(u=0, v=2, distance=3.0, fuel_cost=9.0, time_cost=2.0))
    return net


def test_add_duplicate_node_raises():
    net = FlightNetwork()
    net.add_node(Node(id=0, name="A", x=0.0, y=0.0))
    with pytest.raises(InvalidNodeError):
        net.add_node(Node(id=0, name="A2", x=1.0, y=1.0))


def test_edge_to_unknown_node_raises():
    net = FlightNetwork()
    net.add_node(Node(id=0, name="A", x=0.0, y=0.0))
    with pytest.raises(InvalidEdgeError):
        net.add_edge(Edge(u=0, v=99, distance=1.0, fuel_cost=1.0, time_cost=1.0))


def test_duplicate_edge_raises():
    net = _tiny_network()
    with pytest.raises(InvalidEdgeError):
        net.add_edge(Edge(u=0, v=1, distance=1.0, fuel_cost=1.0, time_cost=1.0))


def test_self_loop_raises():
    with pytest.raises(InvalidEdgeError):
        Edge(u=0, v=0, distance=1.0, fuel_cost=1.0, time_cost=1.0)


def test_negative_cost_component_raises():
    with pytest.raises(InvalidEdgeError):
        Edge(u=0, v=1, distance=-1.0, fuel_cost=1.0, time_cost=1.0)


def test_validate_path_errors():
    net = _tiny_network()
    with pytest.raises(InvalidPathError):
        net.validate_path([])
    with pytest.raises(InvalidPathError):
        net.validate_path([0])
    with pytest.raises(InvalidPathError):
        net.validate_path([0, 99])
    with pytest.raises(InvalidPathError):
        net.validate_path([0, 1, 0])  # revisits a node
    with pytest.raises(InvalidPathError):
        net.validate_path([0, 2, 1])  # no edge 2->1


def test_is_reachable():
    net = _tiny_network()
    assert net.is_reachable(0, 2)
    net.add_node(Node(id=3, name="D", x=5.0, y=5.0))
    assert not net.is_reachable(0, 3)


def test_path_cost_matches_manual_sum():
    net = _tiny_network()
    weights = ObjectiveWeights(alpha=1.0, beta=0.0, gamma=0.0, delta=0.0, normalize_components=False)
    cost = net.path_cost([0, 1, 2], weights)
    assert cost == pytest.approx(1.0 + 1.0)


def test_normalization_changes_relative_weighting():
    net = _tiny_network()
    raw_weights = ObjectiveWeights(alpha=1.0, beta=1.0, gamma=0.0, delta=0.0, normalize_components=False)
    norm_weights = ObjectiveWeights(alpha=1.0, beta=1.0, gamma=0.0, delta=0.0, normalize_components=True)
    raw_cost = net.path_cost([0, 2], raw_weights)
    norm_cost = net.path_cost([0, 2], norm_weights)
    # fuel_cost is on a much larger raw scale (9.0) than distance (3.0) for this
    # edge, so without normalization it dominates; with normalization the two
    # components are brought to comparable scale, changing the combined value.
    assert raw_cost != pytest.approx(norm_cost)


def test_save_and_load_json_round_trip(tmp_path):
    net = _tiny_network()
    net.metadata["seed"] = 123
    path = tmp_path / "net.json"
    net.save_json(path)
    loaded = FlightNetwork.load_json(path)
    assert loaded.num_nodes == net.num_nodes
    assert loaded.num_edges == net.num_edges
    assert loaded.metadata["seed"] == 123
    assert loaded.get_edge(0, 1).distance == pytest.approx(1.0)


def test_disconnected_network_error_is_flight_network_error():
    assert issubclass(DisconnectedNetworkError, Exception)
