from __future__ import annotations

import pytest

from susy_mp_bootstrap.figure8_matrix_quadratic import (
    Figure8QuadraticReducer,
    figure8_bosonic_basis,
    figure8_fermion_minus_basis,
    figure8_fermion_plus_basis,
    figure8_ground_basis,
    figure8_x2_exact,
    figure8_x4_exact,
    solve_figure8_bound,
)


def test_figure8_exact_lines_match_known_formulas() -> None:
    assert figure8_x2_exact(3) == pytest.approx(9 / 2)
    assert figure8_x4_exact(3) == pytest.approx(3 * 19 / 4)


def test_figure8_reduced_basis_sizes() -> None:
    assert len(figure8_bosonic_basis()) == 7
    assert len(figure8_fermion_minus_basis()) == 3
    assert len(figure8_fermion_plus_basis()) == 3
    assert len(figure8_ground_basis()) == 5


def test_figure8_vacuum_constraints_reproduce_x2_bound_at_n1() -> None:
    result = solve_figure8_bound(n=1, observable=("X", "X"))
    assert result.feasible
    assert result.objective_value == pytest.approx(0.5, rel=1e-5, abs=1e-6)


def test_figure8_x4_bound_is_conservative_at_n1() -> None:
    result = solve_figure8_bound(n=1, observable=("X", "X", "X", "X"))
    assert result.feasible
    assert result.objective_value == pytest.approx(0.25, rel=1e-5, abs=1e-6)
    assert result.objective_value < figure8_x4_exact(1)


def test_figure8_ground_entry_closes_quadratically() -> None:
    reducer = Figure8QuadraticReducer(n=2)
    expr = reducer.ground_entry_expr(("X",), ("P",))
    # <X [H, P]> = i <X^2> in the quadratic model.
    assert expr == pytest.approx({reducer.word_to_index[("X", "X")]: 1.0j})
