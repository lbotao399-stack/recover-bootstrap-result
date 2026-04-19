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
    x2_min: float | None
    x2_max: float | None


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


def build_toy_figure4_reduction(
    energy: float,
    *,
    moment_level: int = 12,
    positivity_level: int = 6,
    right_level: int = 8,
    commutator_level: int = 11,
    tolerance: float = 1e-9,
) -> ToyFigure4Reduction:
    from scipy.linalg import qr

    matrix, rhs, moment_keys = _build_full_linear_system(
        energy=energy,
        moment_level=moment_level,
        right_level=right_level,
        commutator_level=commutator_level,
    )
    rank = int(np.linalg.matrix_rank(matrix))
    _, _, pivot_columns = qr(matrix, pivoting=True, mode="economic")
    free_indices = sorted(int(index) for index in pivot_columns[rank:])
    dependent_indices = [index for index in range(len(moment_keys)) if index not in free_indices]
    free_keys = tuple(moment_keys[index] for index in free_indices)

    dependent_matrix = matrix[:, dependent_indices]
    free_matrix = matrix[:, free_indices]
    dependent_rank = int(np.linalg.matrix_rank(dependent_matrix))
    if dependent_rank < len(dependent_indices):
        raise RuntimeError("toy Figure 4 linear reduction is rank-deficient even after automatic pivot selection")

    constant_solution, _, _, _ = np.linalg.lstsq(dependent_matrix, rhs, rcond=None)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        constant_residual = dependent_matrix @ constant_solution - rhs
    max_residual = float(np.max(np.abs(constant_residual))) if constant_residual.size else 0.0

    hat_expressions: dict[ToyWord, AffineExpr] = {}
    for free_position, free_index in enumerate(free_indices):
        hat_expressions[moment_keys[free_index]] = _free_expr(free_position)
    for position, dependent_index in enumerate(dependent_indices):
        hat_expressions[moment_keys[dependent_index]] = AffineExpr(constant=complex(constant_solution[position]))

    for free_position in range(len(free_indices)):
        response = -free_matrix[:, free_position]
        solution, _, _, _ = np.linalg.lstsq(dependent_matrix, response, rcond=None)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            residual = dependent_matrix @ solution - response
        max_residual = max(max_residual, float(np.max(np.abs(residual))) if residual.size else 0.0)
        for position, dependent_index in enumerate(dependent_indices):
            coefficient = complex(solution[position])
            if abs(coefficient) >= 1e-13:
                hat_expressions[moment_keys[dependent_index]].coefficients[free_position] = coefficient

    if max_residual > tolerance:
        raise RuntimeError(f"toy Figure 4 linear reduction residual too large: {max_residual:.3e}")

    expressions: dict[ToyWord, AffineExpr] = {}
    for word, hat_expr in hat_expressions.items():
        expressions[word] = hat_expr.scale(((-1.0j) ** (word[2] + word[3])))

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
    feasible = values_min is not None and values_max is not None
    x2_expr = reduction.moment_expr((2, 0, 0, 0))
    return ToyFigure4SliceResult(
        energy=float(energy),
        status_min=status_min,
        status_max=status_max,
        feasible=feasible,
        x2_min=None if values_min is None else float(np.real(x2_expr.evaluate(values_min))),
        x2_max=None if values_max is None else float(np.real(x2_expr.evaluate(values_max))),
    )


def scan_toy_figure4(config: ToyFigure4Config) -> list[ToyFigure4SliceResult]:
    energies = np.linspace(config.energy_min, config.energy_max, config.num_energy)
    return [solve_toy_figure4_slice(float(energy), config=config) for energy in energies]


def write_toy_figure4_bounds_csv(path: str | Path, results: list[ToyFigure4SliceResult]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["energy", "status_min", "status_max", "feasible", "x2_min", "x2_max"])
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "energy": result.energy,
                    "status_min": result.status_min,
                    "status_max": result.status_max,
                    "feasible": int(result.feasible),
                    "x2_min": np.nan if result.x2_min is None else result.x2_min,
                    "x2_max": np.nan if result.x2_max is None else result.x2_max,
                }
            )


def plot_toy_figure4(results: list[ToyFigure4SliceResult], *, out_path: str | Path) -> None:
    plt = _import_matplotlib()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    energies = np.array([result.energy for result in results], dtype=float)
    lower = np.array([np.nan if result.x2_min is None else result.x2_min for result in results], dtype=float)
    upper = np.array([np.nan if result.x2_max is None else result.x2_max for result in results], dtype=float)
    mask = np.isfinite(lower) & np.isfinite(upper)

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    if np.any(mask):
        ax.fill_between(energies[mask], lower[mask], upper[mask], color="#5b84c4", alpha=0.35, linewidth=0.0)
        ax.plot(energies[mask], lower[mask], color="#1b4f9c", linewidth=1.2)
        ax.plot(energies[mask], upper[mask], color="#1b4f9c", linewidth=1.2)
    ax.set_xlabel(r"$E$")
    ax.set_ylabel(r"$\langle x^2 \rangle$")
    ax.set_title("Toy Figure 4 Archipelago")
    ax.grid(alpha=0.25)
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
    summary_lines = [
        "# Toy Figure 4 Summary",
        "",
        f"- energy window: [{resolved.energy_min}, {resolved.energy_max}] with {resolved.num_energy} slices",
        f"- moment level: {resolved.moment_level}",
        f"- positivity level: {resolved.positivity_level}",
        f"- feasible slices: {len(feasible)} / {len(results)}",
    ]
    if feasible:
        summary_lines.extend(
            [
                f"- first feasible energy: {feasible[0].energy:.6f}",
                f"- last feasible energy: {feasible[-1].energy:.6f}",
                f"- smallest lower bound on <x^2>: {min(result.x2_min for result in feasible if result.x2_min is not None):.6f}",
                f"- largest upper bound on <x^2>: {max(result.x2_max for result in feasible if result.x2_max is not None):.6f}",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_dir
