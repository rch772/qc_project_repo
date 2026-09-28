from __future__ import annotations

import networkx as nx

from src.models.network import DisconnectedNetworkError, FlightNetwork, InvalidNodeError
from src.models.objective import PathResult, evaluate_path
from src.utils.config import ObjectiveWeights
from src.utils.logging_config import get_logger

logger = get_logger(__name__)


def classical_shortest_path(
    network: FlightNetwork, origin: int, destination: int, weights: ObjectiveWeights
) -> PathResult:
    if origin not in network.nodes:
        raise InvalidNodeError(f"Origin node {origin} not in the network.")
    if destination not in network.nodes:
        raise InvalidNodeError(f"Destination node {destination} not in the network.")

    g = network.to_networkx(weights)
    try:
        path = nx.dijkstra_path(g, origin, destination, weight="cost")
    except nx.NetworkXNoPath as exc:
        raise DisconnectedNetworkError(
            f"No path from {origin} to {destination} in this network."
        ) from exc

    result = evaluate_path(network, path, weights)
    logger.info(f"Classical Dijkstra solution cost={result.total_cost:.4f} over path {result.path}")
    return result
