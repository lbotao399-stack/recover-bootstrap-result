from __future__ import annotations

import math

import pytest

from susy_mp_bootstrap.figure11_matrix_cubic_smallg import (
    Figure11Config,
    figure11_critical_gc,
    figure11_default_g_grid,
    figure11_kink_g0,
)


def test_figure11_critical_points_match_guidance() -> None:
    gc = figure11_critical_gc()
    g0 = figure11_kink_g0()
    assert gc == pytest.approx(0.31020161970069987)
    assert g0 == pytest.approx(math.sqrt(2.0) * gc)
    assert g0 == pytest.approx(0.43869133765083085)


def test_figure11_default_grid_contains_kink_point_once() -> None:
    grid = figure11_default_g_grid()
    g0 = figure11_kink_g0()
    matches = [value for value in grid if math.isclose(value, g0, rel_tol=0.0, abs_tol=1e-12)]
    assert len(matches) == 1
    assert grid == tuple(sorted(grid))


def test_figure11_default_config_uses_small_g_scan() -> None:
    config = Figure11Config()
    assert config.n == 100
    assert config.solver == "SCS"
    assert min(config.g_values) >= 0.2
    assert max(config.g_values) <= 0.8
