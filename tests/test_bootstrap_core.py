from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pytest
import sympy as sp

from susy_mp_bootstrap.matrix_bootstrap import (
    cubic_level8_block_dimensions,
    large_n_harmonic_oscillator_benchmarks,
    quadratic_minimal_psd_matrix,
    quadratic_x2_lower_bound,
)
from susy_mp_bootstrap.matrix_words import PSI, PSIDAG, P, X, TracedWord
from susy_mp_bootstrap.models_1d import (
    cubic_mp_superpotential,
    harmonic_oscillator_energy,
    harmonic_oscillator_superpotential,
)
from susy_mp_bootstrap.moments_1d import (
    harmonic_oscillator_recursion_coefficients,
    recursion_coefficients,
)
from susy_mp_bootstrap.sdp_core import (
    build_fixed_energy_problem,
    export_sdpb_json,
    solve_fixed_energy_sdp,
)


def test_zero_mode_normalizability() -> None:
    sho = harmonic_oscillator_superpotential(1)
    cubic = cubic_mp_superpotential(sp.Symbol("g", positive=True))
    assert sho.normalizable_zero_mode_sector() == -1
    assert cubic.normalizable_zero_mode_sector() is None


def test_general_recursion_matches_sho_specialization() -> None:
    omega = sp.Integer(3)
    energy = sp.Symbol("E")
    epsilon = -1
    superpotential = harmonic_oscillator_superpotential(omega)
    for t in (2, 4, 6):
        assert recursion_coefficients(
            superpotential,
            t=t - 1,
            energy=energy,
            epsilon=epsilon,
        ) == harmonic_oscillator_recursion_coefficients(
            t=t,
            energy=energy,
            epsilon=epsilon,
            omega=omega,
        )


def test_fixed_energy_sdp_solves_sho_ground_state(tmp_path: Path) -> None:
    pytest.importorskip("cvxpy", reason="cvxpy is required for solver smoke")
    problem = build_fixed_energy_problem(
        harmonic_oscillator_superpotential(1),
        epsilon=-1,
        energy=0.0,
        matrix_size=3,
        moment_cutoff=6,
        t_values=(1, 3, 5),
    )
    result = solve_fixed_energy_sdp(
        problem,
        solver="SCS",
        solver_kwargs={"eps": 1e-8, "max_iters": 100000},
    )
    assert result.feasible
    assert result.moments is not None
    assert result.moments[2] == pytest.approx(0.5, abs=1e-4)
    assert result.moments[4] == pytest.approx(0.75, abs=1e-4)

    export_path = tmp_path / "sho_problem.json"
    export_sdpb_json(problem, export_path)
    exported = json.loads(export_path.read_text(encoding="utf-8"))
    assert exported["matrix_size"] == 3
    assert exported["constraints"][0]["label"] == "normalization"


def test_matrix_warmup_benchmarks() -> None:
    lower_bound = quadratic_x2_lower_bound(4, 2.0)
    assert lower_bound == pytest.approx(4.0)

    matrix = quadratic_minimal_psd_matrix(4, 2.0, lower_bound)
    eigenvalues = sorted(float(value.real) for value in np.linalg.eigvals(matrix))
    assert eigenvalues[0] >= -1e-9

    benchmarks = large_n_harmonic_oscillator_benchmarks(5)
    assert benchmarks["E"] == pytest.approx(12.5)
    assert benchmarks["X2"] == pytest.approx(12.5)
    assert benchmarks["X4"] == pytest.approx(5 * (2 * 25 + 1) / 4)
    assert benchmarks["X6"] == pytest.approx(5 * 25 * 27 / 8)
    assert benchmarks["PX"] == complex(0.0, -12.5)


def test_traced_word_metadata() -> None:
    word = TracedWord.from_letters(P, X, PSIDAG, PSI)
    assert word.level() == sp.Rational(6, 1)
    assert word.fermion_number() == 0
    assert word.reality_sign() == -1
    assert word.dagger() == TracedWord.from_letters(PSIDAG, PSI, X, P)
    assert TracedWord.from_letters(X, P, X).cyclic_canonical() == TracedWord.from_letters(P, X, X)


def test_cubic_level8_block_counting() -> None:
    dimensions = cubic_level8_block_dimensions()
    assert dimensions == {0: 20, 1: 8, -1: 8, 2: 4, -2: 4}
    assert sum(dimensions.values()) == 44
