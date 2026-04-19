from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest

from susy_mp_bootstrap.toy_figure4_x2y2 import (
    build_toy_figure4_reduction,
    toy_free_moment_keys,
    toy_moment_keys,
    toy_operator_parity_blocks,
    toy_shell_counts,
    toy_words,
)


def test_toy_word_counts_match_guidance() -> None:
    assert len(toy_words(12)) == 588
    assert len(toy_words(8)) == 175
    assert len(toy_words(6)) == 80
    assert toy_shell_counts(12) == [1, 2, 5, 8, 14, 20, 30, 40, 55, 70, 91, 112, 140]


def test_toy_reduced_moment_count_matches_parity_and_swap() -> None:
    moment_keys = toy_moment_keys(12)
    assert len(moment_keys) == 82
    assert toy_free_moment_keys(12) == ((2, 0, 0, 0), (4, 0, 0, 0), (6, 0, 0, 0), (8, 0, 0, 0), (10, 0, 0, 0), (12, 0, 0, 0))


def test_toy_operator_blocks_match_guidance() -> None:
    blocks = toy_operator_parity_blocks(6)
    assert {parity: len(words) for parity, words in blocks.items()} == {
        (0, 0): 23,
        (0, 1): 20,
        (1, 0): 20,
        (1, 1): 17,
    }


def test_toy_level4_regressions_are_built_in() -> None:
    energy = 1.5
    reduction = build_toy_figure4_reduction(energy, tolerance=1e-8)
    zeros = np.zeros(len(reduction.free_keys), dtype=float)

    px = reduction.moment_expr((0, 0, 1, 0)).evaluate(zeros)
    py = reduction.moment_expr((0, 0, 0, 1)).evaluate(zeros)
    xpx = reduction.moment_expr((1, 0, 1, 0)).evaluate(zeros)
    ypy = reduction.moment_expr((0, 1, 0, 1)).evaluate(zeros)
    px2 = reduction.moment_expr((0, 0, 2, 0)).evaluate(zeros)
    py2 = reduction.moment_expr((0, 0, 0, 2)).evaluate(zeros)
    x2y2 = reduction.moment_expr((2, 2, 0, 0)).evaluate(zeros)

    assert abs(px) < 1e-10
    assert abs(py) < 1e-10
    assert xpx == pytest.approx(0.5j, abs=1e-9)
    assert ypy == pytest.approx(0.5j, abs=1e-9)
    assert px2 == pytest.approx(2.0 * energy / 3.0, abs=1e-8)
    assert py2 == pytest.approx(2.0 * energy / 3.0, abs=1e-8)
    assert x2y2 == pytest.approx(2.0 * energy / 3.0, abs=1e-8)


def test_toy_reduction_keeps_x2_as_a_free_direction_when_possible() -> None:
    reduction = build_toy_figure4_reduction(1.5, tolerance=1e-8)
    assert reduction.free_keys[:4] == (
        (2, 0, 0, 0),
        (4, 0, 0, 0),
        (6, 0, 0, 0),
        (8, 0, 0, 0),
    )
    x2 = reduction.moment_expr((2, 0, 0, 0))
    assert x2.constant == pytest.approx(0.0, abs=1e-12)
    assert x2.coefficients == {0: pytest.approx(1.0, abs=1e-12)}
