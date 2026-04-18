from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pytest

from susy_mp_bootstrap.figure9_potential import (
    critical_coupling,
    figure9_potential,
    figure9_source_formula_typo,
    figure9_stationary_discriminant,
    figure9_stationary_points,
)


def test_critical_coupling_matches_closed_form() -> None:
    assert critical_coupling() == pytest.approx((6.0 * 3.0**0.5) ** (-0.5))


def test_corrected_potential_has_expected_discriminant_sign_change() -> None:
    gc = critical_coupling()
    assert figure9_stationary_discriminant(gc) == pytest.approx(0.0, abs=1e-12)
    assert figure9_stationary_discriminant(0.2) > 0.0
    assert figure9_stationary_discriminant(0.5) < 0.0


def test_stationary_point_count_changes_across_critical_coupling() -> None:
    assert len(figure9_stationary_points(0.2)) == 3
    assert len(figure9_stationary_points(critical_coupling())) == 2
    assert len(figure9_stationary_points(0.5)) == 1


def test_source_typo_cannot_reproduce_critical_merge() -> None:
    gc = critical_coupling()
    for x in (-5.0, -2.0, 0.0, 1.0):
        second_derivative = 1.0 + 2.0 * gc + 6.0 * gc * gc * x * x
        assert second_derivative > 0.0
    assert figure9_source_formula_typo(0.0, gc) == pytest.approx(-0.5)


def test_corrected_potential_normalization_at_origin() -> None:
    for g in (0.0, 0.2, critical_coupling(), 0.5):
        assert figure9_potential(0.0, g) == pytest.approx(-0.5)
