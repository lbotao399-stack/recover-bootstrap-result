from __future__ import annotations

from fractions import Fraction

import pytest

from susy_mp_bootstrap.figure10_matrix_cubic_largeg import (
    Figure10Reducer,
    build_figure10_problem,
)


def test_figure10_block_dimensions_match_paper_counting() -> None:
    reducer = Figure10Reducer(n=100, universe_source_level=Fraction(8, 1))
    assert reducer.block_dimensions() == {-2: 4, -1: 8, 0: 20, 1: 8, 2: 4}


def test_figure10_bosonic_cut_and_join_px_rule() -> None:
    reducer = Figure10Reducer(n=100, universe_source_level=Fraction(8, 1))
    expr = reducer.moment_expr_from_raw_word(("P", "X"))
    xp_index = reducer.monomial_to_index[(("X", "P"),)]
    assert expr[0] == pytest.approx(-1.0j * 10000)
    assert expr[xp_index] == pytest.approx(1.0j)


def test_figure10_fermionic_cut_and_join_rule() -> None:
    reducer = Figure10Reducer(n=100, universe_source_level=Fraction(8, 1))
    expr = reducer.moment_expr_from_raw_word(("PSI", "PSIDAG"))
    canonical_index = reducer.monomial_to_index[(("PSIDAG", "PSI"),)]
    assert expr[0] == pytest.approx(10000)
    assert expr[canonical_index] == pytest.approx(-1.0)


def test_figure10_problem_builds_with_practical_seed_levels() -> None:
    reducer = Figure10Reducer(n=100, universe_source_level=Fraction(8, 1))
    problem = build_figure10_problem(
        reducer,
        gauge_seed_level=Fraction(5, 1),
        eom_seed_level=Fraction(7, 1),
        reality_seed_level=Fraction(5, 1),
    )
    assert len(reducer.canonical_universe) == 389
    assert len(problem.equality_exprs) == 278
