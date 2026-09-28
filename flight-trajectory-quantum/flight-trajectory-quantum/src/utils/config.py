from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class ObjectiveWeights(BaseModel):
    """Weights (alpha, beta, gamma, delta) combining the four edge-cost
    components into a single scalar cost, plus the normalization switch.

    cost(edge) = alpha * norm_distance(edge)
               + beta  * norm_fuel(edge)
               + gamma * norm_time(edge)
               + delta * norm_weather(edge)

    See docs/problem_formulation.md for why normalization is necessary
    (distance, fuel, time and weather penalties live on different scales,
    so raw weighted sums would be dominated by whichever unit is numerically
    largest, not by what the weights actually intend).
    """

    alpha: float = Field(default=1.0, ge=0.0, description="Weight on distance")
    beta: float = Field(default=1.0, ge=0.0, description="Weight on fuel cost")
    gamma: float = Field(default=1.0, ge=0.0, description="Weight on time cost")
    delta: float = Field(default=1.0, ge=0.0, description="Weight on weather/airspace penalty")
    normalize_components: bool = Field(
        default=True,
        description="If True, divide each cost component by its mean value "
        "across all edges before applying weights, so alpha/beta/gamma/delta "
        "are comparable regardless of raw units.",
    )

    @model_validator(mode="after")
    def _check_nonzero(self) -> "ObjectiveWeights":
        if self.alpha == self.beta == self.gamma == self.delta == 0.0:
            raise ValueError("At least one objective weight must be > 0.")
        return self


class ScenarioConfig(BaseModel):
    """Deterministic scenario-generation parameters."""

    num_nodes: int = Field(ge=2, description="Number of waypoints in the network")
    seed: int = Field(default=42, description="Random seed for reproducibility")
    area_size: float = Field(default=500.0, gt=0, description="Side length of the square area (nmi) nodes are placed in")
    k_nearest: int = Field(default=2, ge=1, description="Each node connects to its k nearest neighbors (directed both ways)")
    extra_edge_fraction: float = Field(default=0.1, ge=0.0, le=1.0, description="Fraction of additional random long-range edges to add on top of the kNN graph")
    restricted_fraction: float = Field(default=0.08, ge=0.0, le=0.9, description="Fraction of candidate edges excluded entirely (simulated restricted airspace)")
    weather_fraction: float = Field(default=0.15, ge=0.0, le=1.0, description="Fraction of edges assigned a nonzero weather penalty")
    base_fuel_rate: float = Field(default=3.0, gt=0, description="Base fuel units burned per unit distance")
    base_speed: float = Field(default=8.0, gt=0, description="Base cruise speed (distance units per time unit)")
    origin: int | None = Field(default=None, description="Origin node id; defaults to node 0 if None")
    destination: int | None = Field(default=None, description="Destination node id; defaults to the last node id if None")


class QAOAConfig(BaseModel):
    """Quantum (QAOA) simulation hyperparameters."""

    reps: int = Field(default=2, ge=1, description="QAOA circuit depth p (number of cost/mixer layer repetitions)")
    optimizer: str = Field(default="COBYLA", description="scipy.optimize.minimize method name for the classical outer loop")
    max_iter: int = Field(default=200, ge=1, description="Maximum classical optimizer iterations")
    shots: int = Field(default=4096, ge=1, description="Number of measurement shots per circuit evaluation")
    seed: int = Field(default=42, description="Seed for simulator and initial-parameter RNG")
    penalty_lambda: float | None = Field(
        default=None,
        description="Constraint penalty coefficient lambda for the QUBO. If "
        "None, it is auto-derived from the edge-cost scale (see "
        "docs/qubo_derivation.md).",
    )
    max_qubits: int = Field(
        default=26,
        ge=1,
        description="Safety cap on statevector simulation size. A QUBO "
        "needing more qubits than this is refused with a clear error "
        "instead of attempting an allocation that would exhaust the "
        "machine's memory (statevector size is 2^num_qubits complex128 "
        "amplitudes = 16 bytes each; 26 qubits ~= 1 GB, discovered "
        "empirically to be a safe ceiling on a machine with a few GB of "
        "RAM -- see docs/experimental_methodology.md).",
    )


class ExperimentConfig(BaseModel):
    """Top-level config bundling everything one experiment run needs."""

    scenario: ScenarioConfig
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    qaoa: QAOAConfig = Field(default_factory=QAOAConfig)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.model_validate(data)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
