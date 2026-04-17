from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from susy_mp_bootstrap.figure1_quadratic import (
    Figure1Config,
    quadratic_feasible,
    quadratic_hankel_matrix,
    quadratic_sho_even_moments,
    scan_figure1_region,
)


def test_quadratic_even_moments_match_closed_forms() -> None:
    energy = 0.0
    u = 0.5
    a = 2.0 * energy + 1.0
    moments = quadratic_sho_even_moments(7, energy=energy, u=u)
    assert moments[4] == pytest.approx((3.0 / 8.0) * (2.0 * a * u + 1.0))
    assert moments[6] == pytest.approx((5.0 / 16.0) * (2.0 * a * a * u + a + 8.0 * u))
    assert moments[8] == pytest.approx((35.0 / 128.0) * (2.0 * a**3 * u + a * a + 26.0 * a * u + 9.0))
    assert moments[10] == pytest.approx(
        (63.0 / 256.0) * (2.0 * a**4 * u + a**3 + 58.0 * a * a * u + 25.0 * a + 128.0 * u)
    )


def test_quadratic_hankel_ground_state_is_psd() -> None:
    matrix = quadratic_hankel_matrix(6, energy=0.0, u=0.5)
    assert np.min(np.linalg.eigvalsh(matrix)) >= -1e-10
    assert quadratic_feasible(6, energy=0.0, u=0.5)


def test_quadratic_infeasible_point_rejected() -> None:
    assert not quadratic_feasible(4, energy=0.0, u=2.0)


def test_small_scan_is_nested() -> None:
    config = Figure1Config(levels=(4, 5, 6, 7), u_min=0.0, u_max=2.0, e_min=0.0, e_max=2.0, num_u=31, num_e=31)
    _, _, masks = scan_figure1_region(config)
    assert np.count_nonzero(masks[7]) <= np.count_nonzero(masks[6]) <= np.count_nonzero(masks[5]) <= np.count_nonzero(
        masks[4]
    )

