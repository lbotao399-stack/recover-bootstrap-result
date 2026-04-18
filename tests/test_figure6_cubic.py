from __future__ import annotations

import pytest

from susy_mp_bootstrap.figure6_cubic import (
    Figure6CubicReducer,
    figure6_ground_basis,
    figure6_ground_basis_size,
    figure6_operator_basis,
    figure6_operator_basis_size,
)


def test_figure6_basis_sizes_match_theory() -> None:
    assert figure6_operator_basis_size() == 12
    assert figure6_ground_basis_size() == 8
    assert (0, 0) not in figure6_ground_basis()
    assert figure6_operator_basis_size(7) == 6
    assert figure6_ground_basis_size(7) == 3


def test_figure6_basis_levels_match_theory() -> None:
    assert max(2 * p + x for p, x in figure6_operator_basis()) == 5
    assert max(2 * p + x for p, x in figure6_ground_basis()) == 4
    assert max(2 * p + x for p, x in figure6_operator_basis(7)) == 3
    assert max(2 * p + x for p, x in figure6_ground_basis(7)) == 2


def test_figure6_ground_entry_matches_simple_commutator_example() -> None:
    reducer = Figure6CubicReducer(g=0.5, energy=0.05)
    # <x [H, x]> = -i <x p> = 1/2 in an energy eigenstate.
    assert reducer.ground_entry_expr((0, 1), (0, 1)) == pytest.approx({0: 0.5 + 0.0j})
