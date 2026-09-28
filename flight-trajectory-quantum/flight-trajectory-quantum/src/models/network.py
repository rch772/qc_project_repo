from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from src.utils.config import ObjectiveWeights


class FlightNetworkError(Exception):
    """Base class for all flight-network errors."""


class InvalidNodeError(FlightNetworkError):
    pass


class InvalidEdgeError(FlightNetworkError):
    pass


class InvalidPathError(FlightNetworkError):
    pass


class DisconnectedNetworkError(FlightNetworkError):
    pass


@dataclass(frozen=True)
class Node:
    id: int
    name: str
    x: float
    y: float


@dataclass(frozen=True)
class Edge:
    u: int
    v: int
    distance: float
    fuel_cost: float
    time_cost: float
    weather_penalty: float = 0.0

    def __post_init__(self) -> None:
        for field_name in ("distance", "fuel_cost", "time_cost", "weather_penalty"):
            if getattr(self, field_name) < 0:
                raise InvalidEdgeError(
                    f"Edge ({self.u}->{self.v}): {field_name} must be >= 0, "
                    f"got {getattr(self, field_name)}"
                )
        if self.u == self.v:
            raise InvalidEdgeError(f"Self-loop edge ({self.u}->{self.v}) is not allowed.")


@dataclass
class FlightNetwork:
    nodes: dict[int, Node] = field(default_factory=dict)
    edges: dict[tuple[int, int], Edge] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- construction -----------------------------------------------------

    def add_node(self, node: Node) -> None:
        if node.id in self.nodes:
            raise InvalidNodeError(f"Node id {node.id} already exists.")
        self.nodes[node.id] = node

    def add_edge(self, edge: Edge) -> None:
        if edge.u not in self.nodes or edge.v not in self.nodes:
            raise InvalidEdgeError(
                f"Edge ({edge.u}->{edge.v}) references a node id not in the network."
            )
        if (edge.u, edge.v) in self.edges:
            raise InvalidEdgeError(f"Edge ({edge.u}->{edge.v}) already exists.")
        self.edges[(edge.u, edge.v)] = edge

    # -- queries ------------------------------------------------------------

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def has_edge(self, u: int, v: int) -> bool:
        return (u, v) in self.edges

    def get_edge(self, u: int, v: int) -> Edge:
        try:
            return self.edges[(u, v)]
        except KeyError as exc:
            raise InvalidEdgeError(f"No edge ({u}->{v}) in the network.") from exc

    def out_neighbors(self, node_id: int) -> list[int]:
        if node_id not in self.nodes:
            raise InvalidNodeError(f"Node id {node_id} not in the network.")
        return [v for (u, v) in self.edges if u == node_id]

    def in_neighbors(self, node_id: int) -> list[int]:
        if node_id not in self.nodes:
            raise InvalidNodeError(f"Node id {node_id} not in the network.")
        return [u for (u, v) in self.edges if v == node_id]

    def is_reachable(self, origin: int, destination: int) -> bool:
        if origin not in self.nodes:
            raise InvalidNodeError(f"Origin node {origin} not in the network.")
        if destination not in self.nodes:
            raise InvalidNodeError(f"Destination node {destination} not in the network.")
        return nx.has_path(self._raw_digraph(), origin, destination)

    def validate_path(self, path: list[int]) -> None:
        """Raise InvalidPathError if `path` is not a valid simple walk in
        this network (nonempty, all node ids known, consecutive nodes
        connected by an existing edge)."""
        if not path or len(path) < 2:
            raise InvalidPathError("Path must contain at least an origin and a destination.")
        for node_id in path:
            if node_id not in self.nodes:
                raise InvalidPathError(f"Path references unknown node id {node_id}.")
        if len(set(path)) != len(path):
            raise InvalidPathError("Path revisits a node; only simple paths are supported.")
        for u, v in zip(path[:-1], path[1:]):
            if not self.has_edge(u, v):
                raise InvalidPathError(f"Path uses nonexistent edge ({u}->{v}).")

    def path_edges(self, path: list[int]) -> list[Edge]:
        self.validate_path(path)
        return [self.get_edge(u, v) for u, v in zip(path[:-1], path[1:])]

    # -- cost normalization / conversion ------------------------------------

    def normalization_factors(self) -> dict[str, float]:
        """Mean value of each raw cost component across all edges, used to
        bring distance/fuel/time/weather onto comparable scales before
        combining them with (alpha, beta, gamma, delta). See
        docs/problem_formulation.md. Falls back to 1.0 for a component whose
        mean is 0 (e.g. no edges carry a weather penalty) to avoid division
        by zero."""
        if not self.edges:
            return {"distance": 1.0, "fuel_cost": 1.0, "time_cost": 1.0, "weather_penalty": 1.0}
        n = len(self.edges)
        sums = {"distance": 0.0, "fuel_cost": 0.0, "time_cost": 0.0, "weather_penalty": 0.0}
        for e in self.edges.values():
            sums["distance"] += e.distance
            sums["fuel_cost"] += e.fuel_cost
            sums["time_cost"] += e.time_cost
            sums["weather_penalty"] += e.weather_penalty
        return {k: (v / n if v > 0 else 1.0) for k, v in sums.items()}

    def edge_cost(
        self, edge: Edge, weights: ObjectiveWeights, norm_factors: dict[str, float] | None = None
    ) -> float:
        """Combined scalar cost of a single edge under `weights`."""
        if norm_factors is None:
            norm_factors = self.normalization_factors() if weights.normalize_components else {
                "distance": 1.0, "fuel_cost": 1.0, "time_cost": 1.0, "weather_penalty": 1.0
            }
        d = edge.distance / norm_factors["distance"]
        f = edge.fuel_cost / norm_factors["fuel_cost"]
        t = edge.time_cost / norm_factors["time_cost"]
        w = edge.weather_penalty / norm_factors["weather_penalty"]
        return weights.alpha * d + weights.beta * f + weights.gamma * t + weights.delta * w

    def path_cost(self, path: list[int], weights: ObjectiveWeights) -> float:
        edges = self.path_edges(path)
        norm_factors = self.normalization_factors() if weights.normalize_components else None
        return sum(self.edge_cost(e, weights, norm_factors) for e in edges)

    def _raw_digraph(self) -> nx.DiGraph:
        """Topology-only networkx graph (no cost attribute), used for
        reachability checks that don't depend on objective weights."""
        g = nx.DiGraph()
        g.add_nodes_from(self.nodes.keys())
        g.add_edges_from(self.edges.keys())
        return g

    def to_networkx(self, weights: ObjectiveWeights) -> nx.DiGraph:
        """Directed graph with a 'cost' edge attribute equal to the combined
        objective under `weights`, plus the raw components for inspection."""
        g = nx.DiGraph()
        for node in self.nodes.values():
            g.add_node(node.id, name=node.name, x=node.x, y=node.y)
        norm_factors = self.normalization_factors() if weights.normalize_components else None
        for (u, v), e in self.edges.items():
            g.add_edge(
                u,
                v,
                cost=self.edge_cost(e, weights, norm_factors),
                distance=e.distance,
                fuel_cost=e.fuel_cost,
                time_cost=e.time_cost,
                weather_penalty=e.weather_penalty,
            )
        return g

    # -- serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [asdict(n) for n in self.nodes.values()],
            "edges": [asdict(e) for e in self.edges.values()],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FlightNetwork":
        net = cls(metadata=data.get("metadata", {}))
        for n in data["nodes"]:
            net.add_node(Node(**n))
        for e in data["edges"]:
            net.add_edge(Edge(**e))
        return net

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: str | Path) -> "FlightNetwork":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
