from __future__ import annotations

from dataclasses import dataclass, field
from math import comb
from pathlib import Path
from typing import Any
import csv
import json
import warnings

import numpy as np


ToyWord = tuple[int, int, int, int]


def _import_cvxpy():
    import cvxpy as cp

    return cp


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def toy_word_level(word: ToyWord) -> int:
    m, n, s, t = word
    return m + n + 2 * s + 2 * t


def _falling_factorial(value: int, order: int) -> int:
    if order < 0:
        raise ValueError("order must be non-negative")
    result = 1
    for shift in range(order):
        result *= value - shift
    return result


def _parity_signature(word: ToyWord) -> tuple[int, int]:
    m, n, s, t = word
    return ((m + s) % 2, (n + t) % 2)


def _swap_word(word: ToyWord) -> ToyWord:
    m, n, s, t = word
    return (n, m, t, s)


def _canonical_moment_key(word: ToyWord) -> ToyWord | None:
    if min(word) < 0:
        return None
    if _parity_signature(word) != (0, 0):
        return None
    m, n, s, t = word
    swapped = _swap_word(word)
    if m < n:
        return swapped
    if m > n:
        return word
    if s < t:
        return swapped
    return word


def toy_words(max_level: int, *, exact_level: int | None = None) -> list[ToyWord]:
    words: list[ToyWord] = []
    min_level = 0 if exact_level is None else exact_level
    max_shell = max_level if exact_level is None else exact_level
    for s in range(max_shell // 2 + 1):
        for t in range(max_shell // 2 + 1):
            residual = max_shell - 2 * s - 2 * t
            if residual < 0:
                continue
            for m in range(residual + 1):
                for n in range(residual - m + 1):
                    word = (m, n, s, t)
                    level = toy_word_level(word)
                    if level < min_level or level > max_level:
                        continue
                    words.append(word)
    return sorted(words, key=lambda word: (toy_word_level(word), word))


def toy_shell_counts(max_level: int) -> list[int]:
    return [len(toy_words(level, exact_level=level)) for level in range(max_level + 1)]


def toy_moment_keys(max_level: int) -> tuple[ToyWord, ...]:
    representatives: set[ToyWord] = set()
    for word in toy_words(max_level):
        canonical = _canonical_moment_key(word)
        if canonical is not None:
            representatives.add(canonical)
    return tuple(sorted(representatives, key=lambda word: (toy_word_level(word), word)))


def toy_free_moment_keys(max_level: int) -> tuple[ToyWord, ...]:
    return tuple((degree, 0, 0, 0) for degree in range(2, max_level + 1, 2))


def toy_operator_basis(max_level: int = 6) -> tuple[ToyWord, ...]:
    return tuple(toy_words(max_level))


def toy_operator_parity_blocks(max_level: int = 6) -> dict[tuple[int, int], tuple[ToyWord, ...]]:
    groups: dict[tuple[int, int], list[ToyWord]] = {(0, 0): [], (0, 1): [], (1, 0): [], (1, 1): []}
    for word in toy_operator_basis(max_level):
        groups[_parity_signature(word)].append(word)
    return {parity: tuple(words) for parity, words in groups.items()}


def toy_heisenberg_x2_lower_bound(energy: float) -> float:
    if energy <= 0.0:
        return np.inf
    return 3.0 / (8.0 * float(energy))


def toy_level6_cubic_value(energy: float, x2: float) -> float:
    energy = float(energy)
    x2 = float(x2)
    return 216.0 * x2**3 - 864.0 * energy**2 * x2**2 - 180.0 * energy * x2 + 256.0 * energy**3 + 81.0


def toy_level6_cubic_satisfied(energy: float, x2: float, *, tolerance: float = 1e-8) -> bool:
    return toy_level6_cubic_value(energy, x2) <= tolerance


def _left_equation_terms(word: ToyWord, energy: float) -> dict[ToyWord, complex]:
    m, n, s, t = word
    coefficients: dict[ToyWord, complex] = {}

    def add(term: ToyWord, coefficient: complex) -> None:
        coefficients[term] = coefficients.get(term, 0.0 + 0.0j) + coefficient

    add((m, n, s + 2, t), 0.5)
    add((m, n, s, t + 2), 0.5)
    add((m + 2, n + 2, s, t), 0.5)
    add((m, n, s, t), -float(energy))
    if m >= 1:
        add((m - 1, n, s + 1, t), -1.0j * m)
    if m >= 2:
        add((m - 2, n, s, t), -0.5 * m * (m - 1))
    if n >= 1:
        add((m, n - 1, s, t + 1), -1.0j * n)
    if n >= 2:
        add((m, n - 2, s, t), -0.5 * n * (n - 1))
    return coefficients


def _right_equation_terms(word: ToyWord, energy: float) -> dict[ToyWord, complex]:
    m, n, s, t = word
    coefficients: dict[ToyWord, complex] = {}

    def add(term: ToyWord, coefficient: complex) -> None:
        coefficients[term] = coefficients.get(term, 0.0 + 0.0j) + coefficient

    add((m, n, s + 2, t), 0.5)
    add((m, n, s, t + 2), 0.5)
    add((m, n, s, t), -float(energy))

    fall2 = {0: 1, 1: 2, 2: 2}
    for a in range(min(s, 2) + 1):
        for b in range(min(t, 2) + 1):
            coefficient = 0.5 * ((-1.0j) ** (a + b)) * comb(s, a) * comb(t, b) * fall2[a] * fall2[b]
            add((m + 2 - a, n + 2 - b, s - a, t - b), coefficient)
    return coefficients


def _commutator_equation_terms(word: ToyWord, energy: float) -> dict[ToyWord, complex]:
    coefficients = dict(_left_equation_terms(word, energy))
    for term, coefficient in _right_equation_terms(word, energy).items():
        coefficients[term] = coefficients.get(term, 0.0 + 0.0j) - coefficient
    return coefficients


def _hermiticity_equation_terms(word: ToyWord) -> dict[ToyWord, complex]:
    m, n, s, t = word
    coefficients: dict[ToyWord, complex] = {word: float((-1) ** (s + t))}

    def add(term: ToyWord, coefficient: complex) -> None:
        coefficients[term] = coefficients.get(term, 0.0 + 0.0j) + coefficient

    for a in range(min(s, m) + 1):
        for b in range(min(t, n) + 1):
            coefficient = -comb(s, a) * comb(t, b) * ((-1.0j) ** (a + b)) * _falling_factorial(m, a) * _falling_factorial(n, b)
            add((m - a, n - b, s - a, t - b), coefficient)
    return coefficients


def _complex_row_to_real_rows(
    coefficients: dict[int, complex],
    *,
    rhs: complex = 0.0 + 0.0j,
    size: int,
    tolerance: float = 1e-12,
) -> tuple[list[np.ndarray], list[float]]:
    row = np.zeros(size, dtype=np.complex128)
    for index, coefficient in coefficients.items():
        row[index] += coefficient
    real_rows: list[np.ndarray] = []
    rhs_values: list[float] = []
    if np.max(np.abs(np.real(row))) > tolerance or abs(np.real(rhs)) > tolerance:
        real_rows.append(np.real(row))
        rhs_values.append(float(np.real(rhs)))
    if np.max(np.abs(np.imag(row))) > tolerance or abs(np.imag(rhs)) > tolerance:
        real_rows.append(np.imag(row))
        rhs_values.append(float(np.imag(rhs)))
    return real_rows, rhs_values


@dataclass
class AffineExpr:
    constant: complex = 0.0 + 0.0j
    coefficients: dict[int, complex] = field(default_factory=dict)

    def add_scaled(self, other: "AffineExpr", scale: complex) -> None:
        if scale == 0:
            return
        self.constant += scale * other.constant
        for index, coefficient in other.coefficients.items():
            updated = self.coefficients.get(index, 0.0 + 0.0j) + scale * coefficient
            if abs(updated) < 1e-13:
                self.coefficients.pop(index, None)
            else:
                self.coefficients[index] = updated

    def scale(self, factor: complex) -> "AffineExpr":
        scaled = AffineExpr(constant=self.constant * factor)
        for index, coefficient in self.coefficients.items():
            scaled.coefficients[index] = coefficient * factor
        return scaled

    def conjugate(self) -> "AffineExpr":
        conj = AffineExpr(constant=np.conjugate(self.constant))
        for index, coefficient in self.coefficients.items():
            conj.coefficients[index] = np.conjugate(coefficient)
        return conj

    def evaluate(self, values: np.ndarray) -> complex:
        total = complex(self.constant)
        for index, coefficient in self.coefficients.items():
            total += coefficient * values[index]
        return total


@dataclass(frozen=True)
class ToyFigure4Reduction:
    energy: float
    moment_level: int
    positivity_level: int
    free_keys: tuple[ToyWord, ...]
    moment_keys: tuple[ToyWord, ...]
    expressions: dict[ToyWord, AffineExpr]
    max_residual: float

    def moment_expr(self, word: ToyWord) -> AffineExpr:
        if min(word) < 0 or toy_word_level(word) > self.moment_level:
            return AffineExpr()
        if _parity_signature(word) != (0, 0):
            return AffineExpr()
        expr = self.expressions.get(word)
        if expr is None:
            swapped = _swap_word(word)
            expr = self.expressions.get(swapped)
        if expr is None:
            raise KeyError(f"moment {word} was not solved by the toy Figure 4 recursion")
        copied = AffineExpr(constant=expr.constant)
        copied.coefficients.update(expr.coefficients)
        return copied


@dataclass(frozen=True)
class ToyFigure4SliceResult:
    energy: float
    status_min: str
    status_max: str
    feasible: bool
    accepted: bool
    analytic_rejected: bool
    x2_min: float | None
    x2_max: float | None
    linear_residual_min: float | None
    linear_residual_max: float | None
    gram_min_eig_min: float | None
    gram_min_eig_max: float | None
    heisenberg_margin_min: float | None
    heisenberg_margin_max: float | None
    level6_cubic_min: float | None
    level6_cubic_max: float | None


@dataclass(frozen=True)
class ToyFigure4Config:
    moment_level: int = 12
    positivity_level: int = 6
    right_level: int = 8
    commutator_level: int = 11
    energy_min: float = 0.0
    energy_max: float = 12.0
    num_energy: int = 121
    solver: str = "SCS"
    solver_eps: float = 1e-6
    solver_max_iters: int = 40000
    elimination_tolerance: float = 1e-9

    def to_json(self) -> dict[str, Any]:
        return {
            "moment_level": self.moment_level,
            "positivity_level": self.positivity_level,
            "right_level": self.right_level,
            "commutator_level": self.commutator_level,
            "energy_min": self.energy_min,
            "energy_max": self.energy_max,
            "num_energy": self.num_energy,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "elimination_tolerance": self.elimination_tolerance,
        }


def _copy_expr(expr: AffineExpr) -> AffineExpr:
    copied = AffineExpr(constant=expr.constant)
    copied.coefficients.update(expr.coefficients)
    return copied


def _free_expr(index: int) -> AffineExpr:
    expr = AffineExpr()
    expr.coefficients[index] = 1.0 + 0.0j
    return expr


def _known_moment_expr(expressions: dict[ToyWord, AffineExpr], word: ToyWord, *, moment_level: int) -> AffineExpr:
    if min(word) < 0 or toy_word_level(word) > moment_level:
        return AffineExpr()
    if _parity_signature(word) != (0, 0):
        return AffineExpr()
    if word in expressions:
        return _copy_expr(expressions[word])
    swapped = _swap_word(word)
    if swapped in expressions:
        return _copy_expr(expressions[swapped])
    raise KeyError(f"moment {word} is not available yet during toy Figure 4 elimination")


def _solve_linear_moment(
    *,
    expressions: dict[ToyWord, AffineExpr],
    terms: dict[ToyWord, complex],
    target: ToyWord,
    moment_level: int,
) -> AffineExpr:
    coefficient = terms.get(target, 0.0 + 0.0j)
    if abs(coefficient) < 1e-12:
        raise RuntimeError(f"target {target} has vanishing coefficient in the selected recursion equation")
    expr = AffineExpr()
    for word, term_coefficient in terms.items():
        if word == target or abs(term_coefficient) < 1e-12:
            continue
        expr.add_scaled(_known_moment_expr(expressions, word, moment_level=moment_level), -term_coefficient / coefficient)
    return expr


def _has_known_moment(expressions: dict[ToyWord, AffineExpr], word: ToyWord, *, moment_level: int) -> bool:
    if min(word) < 0 or toy_word_level(word) > moment_level:
        return True
    if _parity_signature(word) != (0, 0):
        return True
    return word in expressions or _swap_word(word) in expressions


def _assign_swap_symmetric(
    expressions: dict[ToyWord, AffineExpr],
    word: ToyWord,
    expr: AffineExpr,
    *,
    moment_level: int,
) -> None:
    if min(word) < 0 or toy_word_level(word) > moment_level:
        return
    if _parity_signature(word) != (0, 0):
        return
    expressions[word] = _copy_expr(expr)
    swapped = _swap_word(word)
    if _parity_signature(swapped) == (0, 0) and toy_word_level(swapped) <= moment_level:
        expressions[swapped] = _copy_expr(expr)


def _evaluate_hat_moment_vector(
    reduction: ToyFigure4Reduction,
    moment_keys: tuple[ToyWord, ...],
    values: np.ndarray,
) -> np.ndarray:
    hat_values = []
    for word in moment_keys:
        moment_value = reduction.moment_expr(word).evaluate(values)
        hat_value = ((-1.0j) ** (word[2] + word[3])) * moment_value
        hat_values.append(float(np.real_if_close(hat_value, tol=1000).real))
    return np.asarray(hat_values, dtype=float)


def _max_reduction_residual(
    reduction: ToyFigure4Reduction,
    *,
    right_level: int,
    commutator_level: int,
) -> float:
    matrix, rhs, full_moment_keys = _build_full_linear_system(
        energy=reduction.energy,
        moment_level=reduction.moment_level,
        right_level=right_level,
        commutator_level=commutator_level,
    )
    samples = [np.zeros(len(reduction.free_keys), dtype=float)]
    for index in range(len(reduction.free_keys)):
        basis = np.zeros(len(reduction.free_keys), dtype=float)
        basis[index] = 1.0
        samples.append(basis)

    max_residual = 0.0
    for sample in samples:
        hat_values = _evaluate_hat_moment_vector(reduction, full_moment_keys, sample)
        residual = matrix @ hat_values - rhs
        if residual.size:
            max_residual = max(max_residual, float(np.max(np.abs(residual))))
    return max_residual


def _build_full_linear_system(
    *,
    energy: float,
    moment_level: int,
    right_level: int,
    commutator_level: int,
) -> tuple[np.ndarray, np.ndarray, tuple[ToyWord, ...]]:
    moment_keys = tuple(word for word in toy_words(moment_level) if _parity_signature(word) == (0, 0))
    index_of = {word: index for index, word in enumerate(moment_keys)}
    rows: list[np.ndarray] = []
    rhs_values: list[float] = []

    def add_equation(term_coefficients: dict[ToyWord, complex], rhs: complex = 0.0 + 0.0j) -> None:
        coefficients: dict[int, complex] = {}
        for word, coefficient in term_coefficients.items():
            if abs(coefficient) < 1e-12:
                continue
            if min(word) < 0 or _parity_signature(word) != (0, 0):
                continue
            if toy_word_level(word) > moment_level:
                continue
            coefficients[index_of[word]] = coefficients.get(index_of[word], 0.0 + 0.0j) + coefficient * ((-1.0j) ** (word[2] + word[3]))
        real_rows, real_rhs = _complex_row_to_real_rows(coefficients, rhs=rhs, size=len(moment_keys))
        rows.extend(real_rows)
        rhs_values.extend(real_rhs)

    for word in toy_words(commutator_level):
        left = _left_equation_terms(word, energy)
        right = _right_equation_terms(word, energy)
        diff = dict(left)
        for term, coefficient in right.items():
            diff[term] = diff.get(term, 0.0 + 0.0j) - coefficient
        add_equation(diff)

    for word in toy_words(right_level):
        add_equation(_right_equation_terms(word, energy))

    for word in toy_words(moment_level):
        add_equation(_hermiticity_equation_terms(word))

    for word in moment_keys:
        swapped = _swap_word(word)
        if swapped != word:
            add_equation({word: 1.0 + 0.0j, swapped: -1.0 + 0.0j})

    normalization = {index_of[(0, 0, 0, 0)]: 1.0 + 0.0j}
    real_rows, real_rhs = _complex_row_to_real_rows(normalization, rhs=1.0 + 0.0j, size=len(moment_keys))
    rows.extend(real_rows)
    rhs_values.extend(real_rhs)

    matrix = np.stack(rows, axis=0)
    rhs_vector = np.asarray(rhs_values, dtype=float)
    return matrix, rhs_vector, moment_keys


def _choose_preferred_free_indices(matrix: np.ndarray, moment_keys: tuple[ToyWord, ...]) -> list[int]:
    from scipy.linalg import qr

    rank = int(np.linalg.matrix_rank(matrix))
    nullity = matrix.shape[1] - rank
    if nullity <= 0:
        return []

    key_to_index = {word: index for index, word in enumerate(moment_keys)}
    max_degree = max((word[0] for word in moment_keys if word[1:] == (0, 0, 0)), default=0)
    preferred_words = [word for word in toy_free_moment_keys(max_degree) if word in key_to_index]
    free_indices = [key_to_index[word] for word in preferred_words]
    all_columns = list(range(matrix.shape[1]))

    def dependent_rank(candidate_free: list[int]) -> tuple[int, int]:
        dependent = [column for column in all_columns if column not in candidate_free]
        return int(np.linalg.matrix_rank(matrix[:, dependent])), len(dependent)

    current_rank, dependent_count = dependent_rank(free_indices)
    if current_rank > dependent_count:
        raise RuntimeError("toy Figure 4 preferred free basis over-constrained the dependent block")

    _, _, pivot_columns = qr(matrix, pivoting=True, mode="economic")
    for column in pivot_columns[rank:]:
        if int(column) in free_indices:
            continue
        free_indices = sorted(free_indices + [int(column)])
        if len(free_indices) == nullity:
            break

    if len(free_indices) < nullity:
        for column in all_columns:
            if column in free_indices:
                continue
            free_indices = sorted(free_indices + [column])
            if len(free_indices) == nullity:
                break

    final_rank, final_dependent = dependent_rank(free_indices)
    if len(free_indices) != nullity or final_rank < final_dependent:
        raise RuntimeError("toy Figure 4 could not build a preferred free basis containing the pure x moments")
    return free_indices


def _choose_automatic_free_indices(matrix: np.ndarray) -> list[int]:
    from scipy.linalg import qr

    rank = int(np.linalg.matrix_rank(matrix))
    _, _, pivot_columns = qr(matrix, pivoting=True, mode="economic")
    return sorted(int(index) for index in pivot_columns[rank:])


def build_toy_figure4_reduction(
    energy: float,
    *,
    moment_level: int = 12,
    positivity_level: int = 6,
    right_level: int = 8,
    commutator_level: int = 11,
    tolerance: float = 1e-9,
) -> ToyFigure4Reduction:
    free_keys = toy_free_moment_keys(moment_level)
    expressions: dict[ToyWord, AffineExpr] = {(0, 0, 0, 0): AffineExpr(constant=1.0 + 0.0j)}
    for free_index, word in enumerate(free_keys):
        expressions[word] = _free_expr(free_index)

    all_moment_keys = toy_moment_keys(moment_level)

    for shell in range(moment_level + 1):
        shell_keys = [word for word in all_moment_keys if toy_word_level(word) == shell]
        if not shell_keys:
            continue

        free_word = (shell, 0, 0, 0) if (shell, 0, 0, 0) in shell_keys and shell > 0 else None
        dependent_words = [word for word in shell_keys if word not in expressions]
        if not dependent_words:
            continue

        equations_by_target: dict[ToyWord, dict[ToyWord, complex]] = {}

        for q in range(1, shell // 2 + 1):
            residual_degree = shell - 2 * q

            for s in range(1, q + 1):
                t = q - s
                for m in range(residual_degree + 1):
                    n = residual_degree - m
                    target = _canonical_moment_key((m, n, s, t))
                    if target is None or toy_word_level(target) != shell:
                        continue
                    if target in expressions or target in equations_by_target:
                        continue
                    equations_by_target[target] = _commutator_equation_terms((m + 1, n, s - 1, t), energy)

            for m in range(residual_degree + 1):
                n = residual_degree - m
                target = _canonical_moment_key((m, n, 0, q))
                if target is None or toy_word_level(target) != shell:
                    continue
                if target in expressions or target in equations_by_target:
                    continue
                equations_by_target[target] = _commutator_equation_terms((m, n + 1, 0, q - 1), energy)

        for m in range(shell + 1):
            n = shell - m
            target = _canonical_moment_key((m, n, 0, 0))
            if target is None or n == 0 or toy_word_level(target) != shell:
                continue
            if target in expressions or target in equations_by_target:
                continue
            if m == 0:
                expressions[target] = _known_moment_expr(expressions, (n, 0, 0, 0), moment_level=moment_level)
                continue
            equations_by_target[target] = _right_equation_terms((m - 2, n - 2, 0, 0), energy)

        if set(equations_by_target) != set(dependent_words):
            missing_targets = sorted(set(dependent_words) - set(equations_by_target), key=lambda word: (toy_word_level(word), word))
            extra_targets = sorted(set(equations_by_target) - set(dependent_words), key=lambda word: (toy_word_level(word), word))
            raise RuntimeError(
                "toy Figure 4 shell recursion target mismatch: "
                f"missing={missing_targets[:6]}, extra={extra_targets[:6]}"
            )

        rows: list[list[complex]] = []
        rhs_affine: list[AffineExpr] = []
        for target in dependent_words:
            terms = equations_by_target[target]
            row = [0.0 + 0.0j for _ in dependent_words]
            known = AffineExpr()
            for word, coefficient in terms.items():
                if abs(coefficient) < 1e-12:
                    continue
                if min(word) < 0 or toy_word_level(word) > moment_level:
                    continue
                if _parity_signature(word) != (0, 0):
                    continue
                level = toy_word_level(word)
                if level < shell:
                    known.add_scaled(_known_moment_expr(expressions, word, moment_level=moment_level), coefficient)
                    continue
                if level > shell:
                    raise RuntimeError(f"toy Figure 4 shell recursion raised level at shell {shell}: {word}")
                key = _canonical_moment_key(word)
                if key is None:
                    continue
                if free_word is not None and key == free_word:
                    known.add_scaled(_known_moment_expr(expressions, key, moment_level=moment_level), coefficient)
                    continue
                try:
                    row[dependent_words.index(key)] += coefficient
                except ValueError as error:
                    raise RuntimeError(f"unexpected shell variable {key} while solving shell {shell}") from error
            rows.append(row)
            rhs_affine.append(known.scale(-1.0))

        matrix = np.asarray(rows, dtype=np.complex128)
        rank = int(np.linalg.matrix_rank(matrix))
        if rank < len(dependent_words):
            raise RuntimeError(
                f"toy Figure 4 shell recursion produced rank-deficient shell matrix at shell {shell}: "
                f"rank {rank} < {len(dependent_words)}"
            )

        const_rhs = np.asarray([expr.constant for expr in rhs_affine], dtype=np.complex128)
        const_solution, _, _, _ = np.linalg.lstsq(matrix, const_rhs, rcond=None)
        const_residual = matrix @ const_solution - const_rhs
        max_shell_residual = float(np.max(np.abs(const_residual))) if const_residual.size else 0.0

        solved_exprs = [AffineExpr(constant=complex(const_solution[index])) for index in range(len(dependent_words))]
        for free_index in range(len(free_keys)):
            coeff_rhs = np.asarray([expr.coefficients.get(free_index, 0.0 + 0.0j) for expr in rhs_affine], dtype=np.complex128)
            coeff_solution, _, _, _ = np.linalg.lstsq(matrix, coeff_rhs, rcond=None)
            coeff_residual = matrix @ coeff_solution - coeff_rhs
            if coeff_residual.size:
                max_shell_residual = max(max_shell_residual, float(np.max(np.abs(coeff_residual))))
            for dep_index, coefficient in enumerate(coeff_solution):
                if abs(coefficient) >= 1e-13:
                    solved_exprs[dep_index].coefficients[free_index] = complex(coefficient)

        if max_shell_residual > tolerance:
            raise RuntimeError(
                f"toy Figure 4 shell recursion residual too large at shell {shell}: {max_shell_residual:.3e}"
            )

        for word, expr in zip(dependent_words, solved_exprs, strict=True):
            expressions[word] = expr

    moment_keys = toy_moment_keys(moment_level)
    missing = [word for word in moment_keys if not _has_known_moment(expressions, word, moment_level=moment_level)]
    if missing:
        missing_preview = ", ".join(str(word) for word in missing[:8])
        raise RuntimeError(f"toy Figure 4 shell recursion left unsolved moments: {missing_preview}")

    reduction = ToyFigure4Reduction(
        energy=float(energy),
        moment_level=moment_level,
        positivity_level=positivity_level,
        free_keys=free_keys,
        moment_keys=moment_keys,
        expressions=expressions,
        max_residual=0.0,
    )
    max_residual = _max_reduction_residual(
        reduction,
        right_level=right_level,
        commutator_level=commutator_level,
    )
    if max_residual > tolerance:
        raise RuntimeError(f"toy Figure 4 shell reduction residual too large: {max_residual:.3e}")
    return ToyFigure4Reduction(
        energy=float(energy),
        moment_level=moment_level,
        positivity_level=positivity_level,
        free_keys=free_keys,
        moment_keys=moment_keys,
        expressions=expressions,
        max_residual=max_residual,
    )


def _gram_entry_expr(reduction: ToyFigure4Reduction, left: ToyWord, right: ToyWord) -> AffineExpr:
    m_a, n_a, s_a, t_a = left
    m_b, n_b, s_b, t_b = right
    m_total = m_a + m_b
    n_total = n_a + n_b
    expr = AffineExpr()
    for a in range(min(s_a, m_total) + 1):
        for b in range(min(t_a, n_total) + 1):
            coefficient = (
                comb(s_a, a)
                * comb(t_a, b)
                * ((-1.0j) ** (a + b))
                * _falling_factorial(m_total, a)
                * _falling_factorial(n_total, b)
            )
            moment = (m_total - a, n_total - b, s_a + s_b - a, t_a + t_b - b)
            expr.add_scaled(reduction.moment_expr(moment), coefficient)
    return expr


def build_toy_figure4_gram_blocks(reduction: ToyFigure4Reduction) -> dict[tuple[int, int], list[list[AffineExpr]]]:
    blocks: dict[tuple[int, int], list[list[AffineExpr]]] = {}
    for parity, basis in toy_operator_parity_blocks(reduction.positivity_level).items():
        size = len(basis)
        matrix = [[AffineExpr() for _ in range(size)] for _ in range(size)]
        for i, left in enumerate(basis):
            for j, right in enumerate(basis):
                matrix[i][j] = _gram_entry_expr(reduction, left, right)

        for i in range(size):
            for j in range(i, size):
                symmetrized = AffineExpr()
                symmetrized.add_scaled(matrix[i][j], 0.5)
                symmetrized.add_scaled(matrix[j][i].conjugate(), 0.5)
                matrix[i][j] = symmetrized
                matrix[j][i] = symmetrized.conjugate()
        blocks[parity] = matrix
    return blocks


def _evaluate_block_matrix(block: list[list[AffineExpr]], values: np.ndarray) -> np.ndarray:
    matrix = np.asarray([[expr.evaluate(values) for expr in row] for row in block], dtype=np.complex128)
    return 0.5 * (matrix + matrix.conjugate().T)


def _gram_min_eigenvalue(blocks: dict[tuple[int, int], list[list[AffineExpr]]], values: np.ndarray) -> float:
    min_eigenvalue = np.inf
    for block in blocks.values():
        matrix = _evaluate_block_matrix(block, values)
        eigenvalues = np.linalg.eigvalsh(matrix)
        min_eigenvalue = min(min_eigenvalue, float(np.min(np.real(eigenvalues))))
    return float(min_eigenvalue)


def _linear_residual_inf(
    reduction: ToyFigure4Reduction,
    values: np.ndarray,
    *,
    right_level: int,
    commutator_level: int,
) -> float:
    matrix, rhs, full_moment_keys = _build_full_linear_system(
        energy=reduction.energy,
        moment_level=reduction.moment_level,
        right_level=right_level,
        commutator_level=commutator_level,
    )
    hat_values = _evaluate_hat_moment_vector(reduction, full_moment_keys, values)
    residual = matrix @ hat_values - rhs
    return 0.0 if residual.size == 0 else float(np.max(np.abs(residual)))


def _affine_real(expr: AffineExpr, variables) -> Any:
    result = float(np.real(expr.constant))
    for index, coefficient in expr.coefficients.items():
        result = result + float(np.real(coefficient)) * variables[index]
    return result


def _affine_imag(expr: AffineExpr, variables) -> Any:
    result = float(np.imag(expr.constant))
    for index, coefficient in expr.coefficients.items():
        result = result + float(np.imag(coefficient)) * variables[index]
    return result


def _solve_toy_figure4_objective(
    blocks: dict[tuple[int, int], list[list[AffineExpr]]],
    *,
    target: AffineExpr,
    objective: str,
    solver: str,
    solver_eps: float,
    solver_max_iters: int,
) -> tuple[str, np.ndarray | None]:
    cp = _import_cvxpy()
    if objective not in {"min", "max"}:
        raise ValueError("objective must be 'min' or 'max'")

    any_block = next(iter(blocks.values()))
    free_dim = 0
    for row in any_block:
        for expr in row:
            for index in expr.coefficients:
                free_dim = max(free_dim, index + 1)
    variables = cp.Variable(free_dim)
    constraints = []
    for matrix in blocks.values():
        real_rows = []
        imag_rows = []
        for row in matrix:
            real_rows.append([_affine_real(expr, variables) for expr in row])
            imag_rows.append([_affine_imag(expr, variables) for expr in row])
        real_part = cp.bmat(real_rows)
        imag_part = cp.bmat(imag_rows)
        constraints.append(cp.bmat([[real_part, -imag_part], [imag_part, real_part]]) >> 0)

    objective_expr = _affine_real(target, variables)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Constraint #.*contains too many subexpressions.*")
        problem = cp.Problem(cp.Minimize(objective_expr) if objective == "min" else cp.Maximize(objective_expr), constraints)
    solve_kwargs: dict[str, Any] = {"warm_start": True}
    if solver == "SCS":
        solve_kwargs["eps"] = solver_eps
        solve_kwargs["max_iters"] = solver_max_iters
    elif solver == "CLARABEL":
        solve_kwargs["max_iter"] = solver_max_iters
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Constraint #.*contains too many subexpressions.*")
        warnings.filterwarnings("ignore", message="Solution may be inaccurate.*")
        problem.solve(solver=solver, **solve_kwargs)
    status = str(problem.status)
    feasible = status not in {"infeasible", "infeasible_inaccurate", "unbounded", "unbounded_inaccurate"}
    values = None
    if feasible and variables.value is not None:
        values = np.asarray(variables.value, dtype=float).reshape(-1)
    return status, values


def _certify_toy_figure4_point(
    reduction: ToyFigure4Reduction,
    blocks: dict[tuple[int, int], list[list[AffineExpr]]],
    values: np.ndarray | None,
    *,
    right_level: int,
    commutator_level: int,
) -> tuple[float | None, float | None, float | None, float | None]:
    if values is None:
        return None, None, None, None
    x2 = float(np.real(reduction.moment_expr((2, 0, 0, 0)).evaluate(values)))
    linear_residual = _linear_residual_inf(
        reduction,
        values,
        right_level=right_level,
        commutator_level=commutator_level,
    )
    gram_min_eig = _gram_min_eigenvalue(blocks, values)
    heisenberg_margin = x2 * (2.0 * reduction.energy / 3.0) - 0.25
    return linear_residual, gram_min_eig, heisenberg_margin, toy_level6_cubic_value(reduction.energy, x2)


def solve_toy_figure4_slice(
    energy: float,
    *,
    config: ToyFigure4Config | None = None,
) -> ToyFigure4SliceResult:
    resolved = ToyFigure4Config() if config is None else config
    reduction = build_toy_figure4_reduction(
        energy,
        moment_level=resolved.moment_level,
        positivity_level=resolved.positivity_level,
        right_level=resolved.right_level,
        commutator_level=resolved.commutator_level,
        tolerance=resolved.elimination_tolerance,
    )
    blocks = build_toy_figure4_gram_blocks(reduction)
    status_min, values_min = _solve_toy_figure4_objective(
        blocks,
        target=reduction.moment_expr((2, 0, 0, 0)),
        objective="min",
        solver=resolved.solver,
        solver_eps=resolved.solver_eps,
        solver_max_iters=resolved.solver_max_iters,
    )
    status_max, values_max = _solve_toy_figure4_objective(
        blocks,
        target=reduction.moment_expr((2, 0, 0, 0)),
        objective="max",
        solver=resolved.solver,
        solver_eps=resolved.solver_eps,
        solver_max_iters=resolved.solver_max_iters,
    )
    x2_expr = reduction.moment_expr((2, 0, 0, 0))
    feasible = values_min is not None and values_max is not None
    x2_min = None if values_min is None else float(np.real(x2_expr.evaluate(values_min)))
    x2_max = None if values_max is None else float(np.real(x2_expr.evaluate(values_max)))

    linear_residual_min, gram_min_eig_min, heisenberg_margin_min, level6_cubic_min = _certify_toy_figure4_point(
        reduction,
        blocks,
        values_min,
        right_level=resolved.right_level,
        commutator_level=resolved.commutator_level,
    )
    linear_residual_max, gram_min_eig_max, heisenberg_margin_max, level6_cubic_max = _certify_toy_figure4_point(
        reduction,
        blocks,
        values_max,
        right_level=resolved.right_level,
        commutator_level=resolved.commutator_level,
    )

    analytic_tolerance = 1e-8
    residual_tolerance = 1e-9
    eigen_tolerance = 1e-8
    analytic_rejected = any(
        condition
        for condition in (
            heisenberg_margin_min is not None and heisenberg_margin_min < -analytic_tolerance,
            heisenberg_margin_max is not None and heisenberg_margin_max < -analytic_tolerance,
            level6_cubic_min is not None and level6_cubic_min > analytic_tolerance,
            level6_cubic_max is not None and level6_cubic_max > analytic_tolerance,
        )
    )
    accepted = all(
        condition
        for condition in (
            feasible,
            status_min == "optimal",
            status_max == "optimal",
            linear_residual_min is not None and linear_residual_min <= residual_tolerance,
            linear_residual_max is not None and linear_residual_max <= residual_tolerance,
            gram_min_eig_min is not None and gram_min_eig_min >= -eigen_tolerance,
            gram_min_eig_max is not None and gram_min_eig_max >= -eigen_tolerance,
            heisenberg_margin_min is not None and heisenberg_margin_min >= -analytic_tolerance,
            heisenberg_margin_max is not None and heisenberg_margin_max >= -analytic_tolerance,
            level6_cubic_min is not None and level6_cubic_min <= analytic_tolerance,
            level6_cubic_max is not None and level6_cubic_max <= analytic_tolerance,
            x2_min is not None and x2_max is not None and x2_min <= x2_max + analytic_tolerance,
        )
    )

    return ToyFigure4SliceResult(
        energy=float(energy),
        status_min=status_min,
        status_max=status_max,
        feasible=feasible,
        accepted=accepted,
        analytic_rejected=analytic_rejected,
        x2_min=x2_min,
        x2_max=x2_max,
        linear_residual_min=linear_residual_min,
        linear_residual_max=linear_residual_max,
        gram_min_eig_min=gram_min_eig_min,
        gram_min_eig_max=gram_min_eig_max,
        heisenberg_margin_min=heisenberg_margin_min,
        heisenberg_margin_max=heisenberg_margin_max,
        level6_cubic_min=level6_cubic_min,
        level6_cubic_max=level6_cubic_max,
    )


def scan_toy_figure4(config: ToyFigure4Config) -> list[ToyFigure4SliceResult]:
    energies = np.linspace(config.energy_min, config.energy_max, config.num_energy)
    return [solve_toy_figure4_slice(float(energy), config=config) for energy in energies]


def write_toy_figure4_bounds_csv(path: str | Path, results: list[ToyFigure4SliceResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "energy",
                "status_min",
                "status_max",
                "feasible",
                "accepted",
                "analytic_rejected",
                "x2_min",
                "x2_max",
                "linear_residual_min",
                "linear_residual_max",
                "gram_min_eig_min",
                "gram_min_eig_max",
                "heisenberg_margin_min",
                "heisenberg_margin_max",
                "level6_cubic_min",
                "level6_cubic_max",
            ],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "energy": result.energy,
                    "status_min": result.status_min,
                    "status_max": result.status_max,
                    "feasible": int(result.feasible),
                    "accepted": int(result.accepted),
                    "analytic_rejected": int(result.analytic_rejected),
                    "x2_min": np.nan if result.x2_min is None else result.x2_min,
                    "x2_max": np.nan if result.x2_max is None else result.x2_max,
                    "linear_residual_min": np.nan if result.linear_residual_min is None else result.linear_residual_min,
                    "linear_residual_max": np.nan if result.linear_residual_max is None else result.linear_residual_max,
                    "gram_min_eig_min": np.nan if result.gram_min_eig_min is None else result.gram_min_eig_min,
                    "gram_min_eig_max": np.nan if result.gram_min_eig_max is None else result.gram_min_eig_max,
                    "heisenberg_margin_min": np.nan if result.heisenberg_margin_min is None else result.heisenberg_margin_min,
                    "heisenberg_margin_max": np.nan if result.heisenberg_margin_max is None else result.heisenberg_margin_max,
                    "level6_cubic_min": np.nan if result.level6_cubic_min is None else result.level6_cubic_min,
                    "level6_cubic_max": np.nan if result.level6_cubic_max is None else result.level6_cubic_max,
                }
            )


def plot_toy_figure4(results: list[ToyFigure4SliceResult], *, out_path: str | Path) -> None:
    plt = _import_matplotlib()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for result in results:
        if result.x2_min is None or result.x2_max is None:
            continue
        if result.accepted:
            color = "#1b4f9c"
            alpha = 0.9
            linewidth = 2.0
        elif result.analytic_rejected:
            color = "#c94f3d"
            alpha = 0.9
            linewidth = 1.8
        else:
            color = "#7f7f7f"
            alpha = 0.8
            linewidth = 1.5
        ax.vlines(result.energy, result.x2_min, result.x2_max, color=color, alpha=alpha, linewidth=linewidth)
        ax.scatter([result.energy, result.energy], [result.x2_min, result.x2_max], color=color, alpha=alpha, s=10)

    positive_energies = np.array([result.energy for result in results if result.energy > 0.0], dtype=float)
    if positive_energies.size:
        ax.plot(
            positive_energies,
            [toy_heisenberg_x2_lower_bound(energy) for energy in positive_energies],
            color="#111111",
            linestyle="--",
            linewidth=1.1,
            label=r"$\langle x^2\rangle \ge 3/(8E)$",
        )
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\langle x^2 \rangle$")
    ax.set_title("Toy Figure 4 Certified Vertical Slices")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def run_toy_figure4_scan(*, out_dir: str | Path, config: ToyFigure4Config | None = None) -> Path:
    resolved = ToyFigure4Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(resolved.to_json(), indent=2), encoding="utf-8")

    results = scan_toy_figure4(resolved)
    write_toy_figure4_bounds_csv(output_dir / "bounds.csv", results)
    plot_toy_figure4(results, out_path=output_dir / "toy_figure4_x2y2.png")

    feasible = [result for result in results if result.feasible]
    accepted = [result for result in results if result.accepted]
    analytically_rejected = [result for result in results if result.analytic_rejected]
    summary_lines = [
        "# Toy Figure 4 Summary",
        "",
        f"- energy window: [{resolved.energy_min}, {resolved.energy_max}] with {resolved.num_energy} slices",
        f"- moment level: {resolved.moment_level}",
        f"- positivity level: {resolved.positivity_level}",
        f"- raw solver-returned slices: {len(feasible)} / {len(results)}",
        f"- certified accepted slices: {len(accepted)} / {len(results)}",
        f"- analytically rejected slices: {len(analytically_rejected)} / {len(results)}",
    ]
    if feasible:
        summary_lines.extend(
            [
                f"- first raw slice energy: {feasible[0].energy:.6f}",
                f"- last raw slice energy: {feasible[-1].energy:.6f}",
                f"- smallest raw lower bound on <x^2>: {min(result.x2_min for result in feasible if result.x2_min is not None):.6f}",
                f"- largest raw upper bound on <x^2>: {max(result.x2_max for result in feasible if result.x2_max is not None):.6f}",
            ]
        )
    if analytically_rejected:
        summary_lines.append(
            f"- first analytically rejected energy: {analytically_rejected[0].energy:.6f}"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_dir
