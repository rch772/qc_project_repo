from __future__ import annotations

import networkx as nx

from src.models.network import DisconnectedNetworkError, FlightNetwork, InvalidNodeError
from src.models.objective import PathResult, evaluate_path
from src.utils.config import ObjectiveWeights
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def brute_force_solution(
    network: FlightNetwork, origin: int, destination: int, weights: ObjectiveWeights
) -> tuple[PathResult, int]:
    """Return (best PathResult, number of simple paths examined).

    Raises InvalidNodeError if origin/destination are not in the network,
    and DisconnectedNetworkError if no simple path connects them.
    """
    if origin not in network.nodes:
        raise InvalidNodeError(f"Origin node {origin} not in the network.")
    if destination not in network.nodes:
        raise InvalidNodeError(f"Destination node {destination} not in the network.")

    topology = nx.DiGraph()
    topology.add_nodes_from(network.nodes.keys())
    topology.add_edges_from(network.edges.keys())

    best: PathResult | None = None
    examined = 0
    for path in nx.all_simple_paths(topology, origin, destination):
        examined += 1
        result = evaluate_path(network, path, weights)
        if best is None or result.total_cost < best.total_cost:
            best = result

    if best is None:
        raise DisconnectedNetworkError(
            f"No simple path from {origin} to {destination} in this network."
        )

    logger.info(
        f"Brute force examined {examined} simple paths; optimum cost={best.total_cost:.4f} "
        f"over path {best.path}"
    )
    return best, examined
