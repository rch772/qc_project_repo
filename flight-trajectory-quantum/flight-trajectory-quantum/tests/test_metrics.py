import math

import pytest

from src.optimization.metrics import approximation_ratio, optimality_gap


def test_approximation_ratio_optimal_case():
    assert approximation_ratio(10.0, 10.0) == pytest.approx(1.0)


def test_approximation_ratio_worse_case():
    assert approximation_ratio(12.0, 10.0) == pytest.approx(1.2)


def test_approximation_ratio_zero_optimal_zero_candidate():
    assert approximation_ratio(0.0, 0.0) == pytest.approx(1.0)


def test_approximation_ratio_zero_optimal_nonzero_candidate_is_inf():
    assert approximation_ratio(5.0, 0.0) == math.inf


def test_optimality_gap_optimal_case():
    assert optimality_gap(10.0, 10.0) == pytest.approx(0.0)


def test_optimality_gap_worse_case():
    assert optimality_gap(12.0, 10.0) == pytest.approx(0.2)


def test_optimality_gap_zero_optimal_zero_candidate():
    assert optimality_gap(0.0, 0.0) == pytest.approx(0.0)


def test_optimality_gap_zero_optimal_nonzero_candidate_is_inf():
    assert optimality_gap(5.0, 0.0) == math.inf
