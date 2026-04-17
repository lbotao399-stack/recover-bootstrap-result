from __future__ import annotations

import pytest
import numpy as np

from susy_mp_bootstrap.figure4_cubic import (
    BASIS_B5,
    BASIS_C4,
    FIGURE4_RR_DIMENSION_DEFAULT,
    FIGURE4_WKB_COEFFICIENT,
    Figure4Reducer,
    figure4_ground_basis,
    figure4_operator_basis,
    figure4_rr_classical_basis,
    figure4_rr_full_energy,
    figure4_rr_leading_eigenvalue,
    figure4_wkb_curve,
)


def _expr_to_dict(expr: tuple[tuple[int, complex], ...]) -> dict[int, complex]:
    return dict(expr)


def test_basis_sizes_match_theory() -> None:
    assert len(BASIS_B5) == 20
    assert len(BASIS_C4) == 11
    assert len(figure4_operator_basis()) == 12
    assert len(figure4_ground_basis()) == 8
    assert () not in figure4_ground_basis()


def test_moment_low_order_relations_match_theory() -> None:
    reducer = Figure4Reducer(energy_hat=1.0, lambda_value=1.0, epsilon=-1)

    m3 = _expr_to_dict(reducer.moment_expr(3))
    m4 = _expr_to_dict(reducer.moment_expr(4))

    assert m3 == pytest.approx({0: 0.5 + 0.0j, 1: 0.25 + 0.0j})
    assert m4 == pytest.approx({0: 2.0 / 3.0 + 0.0j, 1: 1.0 + 0.0j, 2: 1.0 / 3.0 + 0.0j})


def test_normal_order_of_zq() -> None:
    reducer = Figure4Reducer(energy_hat=1.0, lambda_value=1.0, epsilon=-1)
    normal = dict(reducer.normal_order(("z", "q")))
    assert normal == pytest.approx({(1, 1): 1.0 + 0.0j, (0, 0): 1.0j})


def test_ground_entry_zz_is_half() -> None:
    reducer = Figure4Reducer(energy_hat=1.0, lambda_value=1.0, epsilon=-1)
    constant, expr = reducer.ground_entry_expr(("z",), ("z",))
    assert expr == {}
    assert constant == pytest.approx(0.5 + 0.0j)


def test_ground_matrix_entry_is_hermitian_under_swap() -> None:
    reducer = Figure4Reducer(energy_hat=1.25, lambda_value=0.5, epsilon=-1)
    entry_12 = reducer.ground_entry_expr(("z",), ("q", "q"))
    entry_21 = reducer.ground_entry_expr(("q", "q"), ("z",))

    const_12, expr_12 = entry_12
    const_21, expr_21 = entry_21
    assert const_12 == pytest.approx(const_21.conjugate())
    assert expr_12 == pytest.approx({index: value.conjugate() for index, value in expr_21.items()})


def test_operator_basis_levels_do_not_exceed_five() -> None:
    def level(word: tuple[str, ...]) -> int:
        return sum(2 if token == "q" else 1 for token in word)

    assert max(level(word) for word in figure4_operator_basis()) == 5
    assert max(level(word) for word in figure4_ground_basis()) == 4


def test_wkb_curve_matches_large_g_formula() -> None:
    g_values = np.array([0.0, 1.0, 8.0], dtype=float)
    energies = figure4_wkb_curve(g_values)
    assert energies == pytest.approx(
        np.array([0.0, FIGURE4_WKB_COEFFICIENT, 4.0 * FIGURE4_WKB_COEFFICIENT], dtype=float)
    )


def test_rr_leading_coefficient_matches_theory_value() -> None:
    coefficient = figure4_rr_leading_eigenvalue(dimension=FIGURE4_RR_DIMENSION_DEFAULT)
    assert coefficient == pytest.approx(0.28106780538565, abs=2e-12)


def test_rr_leading_variational_sequence_is_monotone() -> None:
    values = [
        figure4_rr_leading_eigenvalue(dimension=dimension)
        for dimension in (20, 30, 40, 50, 60)
    ]
    assert values[1] <= values[0]
    assert values[2] <= values[1]
    assert values[3] <= values[2]
    assert values[4] <= values[3]


def test_rr_full_energy_uses_positive_frequency_classical_basis() -> None:
    xi, omega = figure4_rr_classical_basis(1.0)
    assert omega > 0.0
    energy = figure4_rr_full_energy(5.0)
    assert np.isfinite(energy)
    assert energy > 0.0
