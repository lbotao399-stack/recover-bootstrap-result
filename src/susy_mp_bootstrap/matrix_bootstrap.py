from __future__ import annotations

from typing import Final

import numpy as np

from .matrix_words import PSI, PSIDAG, P, X, TracedWord


def quadratic_x2_lower_bound(n: int, a: float) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    return (n**2) / (2 * a)


def quadratic_minimal_psd_matrix(n: int, a: float, x2_expectation: float) -> np.ndarray:
    return np.array(
        [
            [n, 0.0, 0.0],
            [0.0, x2_expectation, 0.5j * n**2],
            [0.0, -0.5j * n**2, (a**2) * x2_expectation],
        ],
        dtype=complex,
    )


def quadratic_operator_basis() -> tuple[TracedWord, ...]:
    return (
        TracedWord.from_letters(),
        TracedWord.from_letters(X),
        TracedWord.from_letters(P),
        TracedWord.from_letters(PSI),
        TracedWord.from_letters(PSIDAG),
        TracedWord.from_letters(X, X),
        TracedWord.from_letters(P, X),
        TracedWord.from_letters(P, P),
        TracedWord.from_letters(PSI, PSIDAG),
        TracedWord.from_letters(PSI, X),
        TracedWord.from_letters(PSI, P),
        TracedWord.from_letters(PSIDAG, X),
        TracedWord.from_letters(PSIDAG, P),
    )


def large_n_harmonic_oscillator_benchmarks(n: int) -> dict[str, float | complex]:
    if n <= 0:
        raise ValueError("n must be positive")
    return {
        "E": n**2 / 2,
        "X2": n**2 / 2,
        "X4": n * (2 * n**2 + 1) / 4,
        "X6": 5 * n**2 * (n**2 + 2) / 8,
        "PX": -0.5j * n**2,
    }


_CUBIC_LEVEL8_BLOCKS: Final[dict[int, tuple[TracedWord, ...]]] = {
    0: (
        TracedWord.from_letters(),
        TracedWord.from_letters(X),
        TracedWord.from_letters(P),
        TracedWord.from_letters(X, X),
        TracedWord.from_letters(X, P),
        TracedWord.from_letters(P, P),
        TracedWord.from_letters(PSIDAG, PSI),
        TracedWord.from_letters(PSI, PSIDAG),
        TracedWord.from_letters(X, X, X),
        TracedWord.from_letters(X, X, P),
        TracedWord.from_letters(X, P, X),
        TracedWord.from_letters(P, X, X),
        TracedWord.from_letters(X, X, X, X),
        TracedWord.from_letters(PSIDAG, PSI, X),
        TracedWord.from_letters(PSI, PSIDAG, X),
        TracedWord.from_letters(X, PSIDAG, PSI),
        TracedWord.from_letters(X, PSI, PSIDAG),
        TracedWord.from_letters(PSIDAG, X, PSI),
        TracedWord.from_letters(PSI, X, PSIDAG),
        TracedWord.from_letters(X, X, PSIDAG, PSI),
    ),
    1: (
        TracedWord.from_letters(PSIDAG),
        TracedWord.from_letters(X, PSIDAG),
        TracedWord.from_letters(PSIDAG, X),
        TracedWord.from_letters(P, PSIDAG),
        TracedWord.from_letters(PSIDAG, P),
        TracedWord.from_letters(X, X, PSIDAG),
        TracedWord.from_letters(X, PSIDAG, X),
        TracedWord.from_letters(PSIDAG, X, X),
    ),
    -1: (
        TracedWord.from_letters(PSI),
        TracedWord.from_letters(X, PSI),
        TracedWord.from_letters(PSI, X),
        TracedWord.from_letters(P, PSI),
        TracedWord.from_letters(PSI, P),
        TracedWord.from_letters(X, X, PSI),
        TracedWord.from_letters(X, PSI, X),
        TracedWord.from_letters(PSI, X, X),
    ),
    2: (
        TracedWord.from_letters(PSIDAG, PSIDAG),
        TracedWord.from_letters(X, PSIDAG, PSIDAG),
        TracedWord.from_letters(PSIDAG, X, PSIDAG),
        TracedWord.from_letters(PSIDAG, PSIDAG, X),
    ),
    -2: (
        TracedWord.from_letters(PSI, PSI),
        TracedWord.from_letters(X, PSI, PSI),
        TracedWord.from_letters(PSI, X, PSI),
        TracedWord.from_letters(PSI, PSI, X),
    ),
}


def cubic_level8_basis_blocks() -> dict[int, tuple[TracedWord, ...]]:
    return _CUBIC_LEVEL8_BLOCKS


def cubic_level8_block_dimensions() -> dict[int, int]:
    return {fermion_number: len(words) for fermion_number, words in _CUBIC_LEVEL8_BLOCKS.items()}

