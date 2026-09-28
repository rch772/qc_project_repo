from __future__ import annotations

import math


def approximation_ratio(candidate_cost: float, optimal_cost: float) -> float:
    """candidate_cost / optimal_cost (minimization convention: 1.0 is
    optimal, >1.0 means the candidate is worse). If optimal_cost == 0, the
    ratio is only meaningful when candidate_cost is also 0 (both describe a
    free trajectory); otherwise it is undefined and reported as +inf rather
    than raising a ZeroDivisionError."""
    if optimal_cost == 0:
        return 1.0 if candidate_cost == 0 else math.inf
    return candidate_cost / optimal_cost


def optimality_gap(candidate_cost: float, optimal_cost: float) -> float:
    """(candidate_cost - optimal_cost) / optimal_cost. 0.0 means optimal;
    positive means worse than optimal (candidate costs more). Same
    zero-cost handling as approximation_ratio."""
    if optimal_cost == 0:
        return 0.0 if candidate_cost == 0 else math.inf
    return (candidate_cost - optimal_cost) / optimal_cost
