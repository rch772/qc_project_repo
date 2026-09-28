from __future__ import annotations

from dataclasses import dataclass

from src.models.network import FlightNetwork
from src.utils.config import ObjectiveWeights


@dataclass(frozen=True)
class PathResult:
    path: tuple[int, ...]
    total_cost: float
    distance: float
    fuel_cost: float
    time_cost: float
    weather_penalty: float

    @property
    def num_edges(self) -> int:
        return len(self.path) - 1


def evaluate_path(network: FlightNetwork, path: list[int], weights: ObjectiveWeights) -> PathResult:
    """Validate `path` in `network` and return its total combined cost plus
    the raw (unnormalized, unweighted) component sums."""
    edges = network.path_edges(path)  # raises InvalidPathError if invalid
    norm_factors = network.normalization_factors() if weights.normalize_components else None
    total = 0.0
    distance = fuel = time_ = weather = 0.0
    for e in edges:
        total += network.edge_cost(e, weights, norm_factors)
        distance += e.distance
        fuel += e.fuel_cost
        time_ += e.time_cost
        weather += e.weather_penalty
    return PathResult(
        path=tuple(path),
        total_cost=total,
        distance=distance,
        fuel_cost=fuel,
        time_cost=time_,
        weather_penalty=weather,
    )
