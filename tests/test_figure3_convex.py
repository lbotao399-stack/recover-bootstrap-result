from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from susy_mp_bootstrap.figure3_convex import (
    Figure3Config,
    QuarticConvexReducer,
    figure3_k2_analytic_bound,
    figure3_perturbation_curve,
    scan_figure3,
    solve_figure3_point,
)


def test_figure3_perturbation_curve_matches_formula() -> None:
    g_values = np.array([0.5, 1.0])
    expected = 0.25 / (g_values**2) + 0.5 - (g_values**2) / 16.0 + 27.0 * (g_values**4) / 128.0
    assert np.allclose(figure3_perturbation_curve(g_values), expected)


def test_k2_bootstrap_matches_analytic_bound() -> None:
    for g in (0.5, 1.0, 1.5):
        status, value = solve_figure3_point(g=g, level=2, epsilon=-1, solver_eps=1e-7, solver_max_iters=50000)
        assert status in {"optimal", "optimal_inaccurate"}
        assert value is not None
        expected = float(figure3_k2_analytic_bound(np.array([g]))[0])
        assert value == pytest.approx(expected, abs=5e-5)


def test_quartic_convex_reducer_low_order_relations() -> None:
    reducer = QuarticConvexReducer(g=1.0, epsilon=-1)
    assert dict(reducer.p_expr(1, 0)) == {}
    assert dict(reducer.p_expr(1, 1)) == {0: -0.5j}
    assert dict(reducer.p_expr(2, 0)) == {1: -1 / np.sqrt(2), 2: 1.0, 4: 1.0}
    assert dict(reducer.pure_p_constraint(1)) == {0: -1j / np.sqrt(2), 1: 1j, 3: 1j}


def test_scan_figure3_returns_finite_k2_values() -> None:
    config = Figure3Config(levels=(2,), g_min=0.5, g_max=1.5, num_g=3, solver_eps=1e-7, solver_max_iters=50000)
    _, bounds, statuses = scan_figure3(config)
    assert np.all(np.isfinite(bounds[2]))
    assert all(status in {"optimal", "optimal_inaccurate"} for status in statuses[2])
