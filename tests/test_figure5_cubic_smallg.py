from __future__ import annotations

import pytest

from susy_mp_bootstrap.figure5_cubic_smallg import (
    Figure5CubicReducer,
    figure5_ambient_basis_size,
    figure5_basis_size,
    figure5_canonical_basis,
)


def test_figure5_basis_sizes_match_theory() -> None:
    assert figure5_basis_size(7) == 20
    assert figure5_basis_size(8) == 25
    assert figure5_basis_size(9) == 30
    assert figure5_basis_size(10) == 36
    assert figure5_ambient_basis_size(7) == 9
    assert figure5_ambient_basis_size(8) == 9
    assert figure5_ambient_basis_size(9) == 12
    assert figure5_ambient_basis_size(10) == 12


def test_figure5_basis_respects_level_cutoff() -> None:
    for level in (7, 8, 9, 10):
        for p_power, x_power in figure5_canonical_basis(level):
            assert 2 * p_power + x_power <= level


def test_figure5_low_order_pure_moment_relations_match_guidance() -> None:
    reducer = Figure5CubicReducer(g=1.0, energy=0.2)
    relation3 = reducer.pure_moment_relation(3)
    relation4 = reducer.pure_moment_relation(4)
    assert relation3 == pytest.approx({0: 4.0 + 0.0j, 1: -4.0 + 0.0j, 2: -12.0 + 0.0j, 3: -8.0 + 0.0j})
    assert relation4 == pytest.approx({0: 5.6 + 0.0j, 1: 12.0 + 0.0j, 2: -8.0 + 0.0j, 3: -20.0 + 0.0j, 4: -12.0 + 0.0j})


def test_figure5_matrix_entry_is_hermitian_under_swap() -> None:
    reducer = Figure5CubicReducer(g=0.7, energy=0.1)
    left = reducer.matrix_entry_expr((1, 2), (0, 3))
    right = reducer.matrix_entry_expr((0, 3), (1, 2))
    assert left == pytest.approx({index: value.conjugate() for index, value in right.items()})
