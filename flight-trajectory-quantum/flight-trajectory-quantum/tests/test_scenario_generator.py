import pytest

from src.data.scenario_generator import generate_scenario
from src.utils.config import ScenarioConfig


@pytest.mark.parametrize("num_nodes", [5, 8, 10, 15])
def test_generated_scenario_is_connected(num_nodes):
    config = ScenarioConfig(num_nodes=num_nodes, seed=42)
    net = generate_scenario(config)
    origin = net.metadata["origin"]
    destination = net.metadata["destination"]
    assert net.is_reachable(origin, destination)


def test_reproducibility_same_seed_gives_identical_network():
    config = ScenarioConfig(num_nodes=10, seed=7)
    net_a = generate_scenario(config)
    net_b = generate_scenario(config)
    assert net_a.to_dict() == net_b.to_dict()


def test_different_seed_gives_different_network():
    net_a = generate_scenario(ScenarioConfig(num_nodes=10, seed=1))
    net_b = generate_scenario(ScenarioConfig(num_nodes=10, seed=2))
    assert net_a.to_dict() != net_b.to_dict()


def test_all_edge_costs_are_nonnegative():
    net = generate_scenario(ScenarioConfig(num_nodes=12, seed=3))
    for edge in net.edges.values():
        assert edge.distance >= 0
        assert edge.fuel_cost >= 0
        assert edge.time_cost >= 0
        assert edge.weather_penalty >= 0


def test_edges_are_directed_both_ways():
    net = generate_scenario(ScenarioConfig(num_nodes=8, seed=5))
    for (u, v) in net.edges:
        assert net.has_edge(v, u), f"edge {(u, v)} exists without its reverse direction"


def test_zero_restricted_fraction_still_connected():
    net = generate_scenario(ScenarioConfig(num_nodes=8, seed=9, restricted_fraction=0.0))
    assert net.is_reachable(0, 7)


def test_high_restricted_fraction_still_connected():
    # Restricted edges are only ever sampled from the non-spanning-tree
    # pairs, so even a high restricted_fraction must not disconnect the graph.
    net = generate_scenario(ScenarioConfig(num_nodes=10, seed=11, restricted_fraction=0.9, extra_edge_fraction=0.5))
    assert net.is_reachable(0, 9)


def test_num_nodes_matches_config():
    net = generate_scenario(ScenarioConfig(num_nodes=6, seed=1))
    assert net.num_nodes == 6
