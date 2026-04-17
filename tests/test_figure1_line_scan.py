from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from susy_mp_bootstrap.figure1_quadratic import (
    Figure1LineConfig,
    line_mask_to_intervals,
    quadratic_line_u,
    scan_figure1_line,
)


def test_quadratic_line_relation() -> None:
    assert quadratic_line_u(0.0) == 0.5
    assert quadratic_line_u(-0.5) == 0.0


def test_line_mask_to_intervals() -> None:
    e_values = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
    mask = np.array([False, True, True, False, True])
    assert line_mask_to_intervals(e_values, mask) == [(0.1, 0.2, 2), (0.4, 0.4, 1)]


def test_line_scan_is_monotone() -> None:
    config = Figure1LineConfig(min_level=4, max_level=12, e_min=-0.5, e_max=3.0, num_e=401)
    _, masks = scan_figure1_line(config)
    levels = sorted(masks)
    for left, right in zip(levels, levels[1:], strict=False):
        assert np.count_nonzero(masks[right]) <= np.count_nonzero(masks[left])

