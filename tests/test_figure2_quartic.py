from __future__ import annotations

from pathlib import Path
import math
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from susy_mp_bootstrap.figure2_quartic import (
    Figure2EuConfig,
    Figure2ExConfig,
    quartic_affine_mean_x_pencil,
    quartic_hankel_matrix,
    quartic_moments,
    quartic_projected_feasible_u,
    scan_figure2_eu,
    scan_figure2_ex,
)


def test_quartic_moments_match_closed_forms_epsilon_minus_one() -> None:
    energy = 0.4
    a = 0.1
    u = 0.8
    sqrt2 = math.sqrt(2.0)
    moments = quartic_moments(5, energy=energy, mean_x=a, u=u, epsilon=-1, g=1.0)

    assert moments[3] == pytest.approx(-a + 1.0 / sqrt2)
    assert moments[4] == pytest.approx((4.0 * energy - 1.0 - 4.0 * u) / 3.0 + sqrt2 * a)
    assert moments[5] == pytest.approx((2.0 * energy + 1.0) * a + (sqrt2 / 4.0) * (5.0 * u - 3.0))
    assert moments[6] == pytest.approx((36.0 * energy * u - 32.0 * energy + 23.0 * u + 38.0) / 15.0 - 3.0 * sqrt2 * a)
    assert moments[7] == pytest.approx((4.0 - 6.0 * energy) * a + (sqrt2 / 12.0) * (40.0 * energy - 49.0 * u + 5.0))
    assert moments[8] == pytest.approx(
        (800.0 * energy * energy - 1664.0 * energy * u + 368.0 * energy + 1373.0 * u - 1357.0) / 210.0
        + 6.0 * sqrt2 * (energy + 1.0) * a
    )


def test_quartic_moments_match_closed_forms_epsilon_plus_one() -> None:
    energy = 0.4
    a = 0.1
    u = 0.8
    sqrt2 = math.sqrt(2.0)
    moments = quartic_moments(5, energy=energy, mean_x=a, u=u, epsilon=1, g=1.0)

    assert moments[3] == pytest.approx(-a - 1.0 / sqrt2)
    assert moments[4] == pytest.approx((4.0 * energy - 1.0 - 4.0 * u) / 3.0 - sqrt2 * a)
    assert moments[5] == pytest.approx((2.0 * energy + 1.0) * a - (sqrt2 / 4.0) * (5.0 * u - 3.0))
    assert moments[6] == pytest.approx((36.0 * energy * u - 32.0 * energy + 23.0 * u + 38.0) / 15.0 + 3.0 * sqrt2 * a)
    assert moments[7] == pytest.approx((4.0 - 6.0 * energy) * a - (sqrt2 / 12.0) * (40.0 * energy - 49.0 * u + 5.0))
    assert moments[8] == pytest.approx(
        (800.0 * energy * energy - 1664.0 * energy * u + 368.0 * energy + 1373.0 * u - 1357.0) / 210.0
        - 6.0 * sqrt2 * (energy + 1.0) * a
    )


def test_quartic_moments_respect_z2_symmetry() -> None:
    energy = 0.7
    a = 0.23
    u = 0.91
    left = quartic_moments(6, energy=energy, mean_x=a, u=u, epsilon=-1, g=1.0)
    right = quartic_moments(6, energy=energy, mean_x=-a, u=u, epsilon=1, g=1.0)
    for order in range(0, 11):
        assert left[order] == pytest.approx(((-1) ** order) * right[order])


def test_figure2_scan_is_nested_and_symmetric() -> None:
    config = Figure2ExConfig(
        levels=(5, 6),
        x_min=-0.3,
        x_max=0.3,
        e_min=0.0,
        e_max=1.5,
        num_x=31,
        num_e=31,
        search_iterations=16,
        u_max=10.0,
    )
    _, _, masks, _ = scan_figure2_ex(config)
    assert masks[6].sum() <= masks[5].sum()
    assert (masks[5] == masks[5][:, ::-1]).all()
    assert (masks[6] == masks[6][:, ::-1]).all()


def test_quartic_affine_mean_x_pencil_reconstructs_matrix() -> None:
    energy = 0.6
    mean_x = 0.2
    u = 0.9
    base, slope = quartic_affine_mean_x_pencil(5, energy=energy, u=u, epsilon=-1, g=1.0)
    reconstructed = base + mean_x * slope
    direct = quartic_hankel_matrix(5, energy=energy, mean_x=mean_x, u=u, epsilon=-1, g=1.0)
    assert np.allclose(reconstructed, direct)


def test_quartic_projected_u_region_is_epsilon_independent() -> None:
    energy = 1.0
    u = 0.8
    feasible_minus, _, _ = quartic_projected_feasible_u(5, energy=energy, u=u, epsilon=-1, g=1.0)
    feasible_plus, _, _ = quartic_projected_feasible_u(5, energy=energy, u=u, epsilon=1, g=1.0)
    assert feasible_minus == feasible_plus


def test_figure2_eu_scan_is_nested() -> None:
    config = Figure2EuConfig(
        levels=(5, 6),
        u_min=0.0,
        u_max=1.5,
        e_min=0.0,
        e_max=1.5,
        num_u=21,
        num_e=21,
        search_iterations=16,
    )
    _, _, masks, _ = scan_figure2_eu(config)
    assert masks[6].sum() <= masks[5].sum()
