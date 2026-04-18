from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from .matrix_bootstrap import large_n_harmonic_oscillator_benchmarks, quadratic_x2_lower_bound
from .matrix_words import PSI, PSIDAG, P, X


RawWord = tuple[str, ...]
LinearExpr = dict[int, complex]
RAW_LETTERS = (X, P, PSI, PSIDAG)


def _import_cvxpy():
    import cvxpy as cp

    return cp


def _expr_add_scaled(target: LinearExpr, source: LinearExpr, scale: complex) -> None:
    if scale == 0:
        return
    for index, coefficient in source.items():
        updated = target.get(index, 0.0 + 0.0j) + scale * coefficient
        if abs(updated) < 1e-13:
            target.pop(index, None)
        else:
            target[index] = updated


def _expr_real(expr: LinearExpr, variables) -> Any:
    result = float(np.real(expr.get(0, 0.0 + 0.0j)))
    for index, coefficient in expr.items():
        if index == 0:
            continue
        result = result + float(np.real(coefficient)) * variables[index - 1]
    return result


def _expr_imag(expr: LinearExpr, variables) -> Any:
    result = float(np.imag(expr.get(0, 0.0 + 0.0j)))
    for index, coefficient in expr.items():
        if index == 0:
            continue
        result = result + float(np.imag(coefficient)) * variables[index - 1]
    return result


def _single_var_expr(index: int, coefficient: complex = 1.0) -> LinearExpr:
    return {index: coefficient}


def _build_real_psd_expression(cp, matrix_exprs: list[list[LinearExpr]], variables) -> Any:
    size = len(matrix_exprs)
    re_matrix = [[0 for _ in range(size)] for _ in range(size)]
    im_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            re_matrix[row][column] = _expr_real(matrix_exprs[row][column], variables)
            im_matrix[row][column] = _expr_imag(matrix_exprs[row][column], variables)
    big_rows = []
    for row in range(size):
        big_rows.append(re_matrix[row] + [-entry for entry in im_matrix[row]])
    for row in range(size):
        big_rows.append(im_matrix[row] + re_matrix[row])
    return cp.bmat(big_rows)


def raw_word_dagger(word: RawWord) -> RawWord:
    mapped = []
    for letter in reversed(word):
        if letter == PSI:
            mapped.append(PSIDAG)
        elif letter == PSIDAG:
            mapped.append(PSI)
        else:
            mapped.append(letter)
    return tuple(mapped)


def raw_word_p_count(word: RawWord) -> int:
    return sum(letter == P for letter in word)


def raw_word_charge(word: RawWord) -> int:
    charge = 0
    for letter in word:
        if letter == PSI:
            charge -= 1
        elif letter == PSIDAG:
            charge += 1
    return charge


def raw_word_is_bosonic(word: RawWord) -> bool:
    return all(letter in (X, P) for letter in word)


def all_raw_words(max_length: int) -> tuple[RawWord, ...]:
    words: list[RawWord] = [tuple()]
    frontier: list[RawWord] = [tuple()]
    for _ in range(max_length):
        new_frontier: list[RawWord] = []
        for word in frontier:
            for letter in RAW_LETTERS:
                new_frontier.append(word + (letter,))
        words.extend(new_frontier)
        frontier = new_frontier
    return tuple(words)


def all_charge_zero_words(max_length: int) -> tuple[RawWord, ...]:
    return tuple(word for word in all_raw_words(max_length) if raw_word_charge(word) == 0)


def all_bosonic_words(max_length: int) -> tuple[RawWord, ...]:
    words: list[RawWord] = [tuple()]
    frontier: list[RawWord] = [tuple()]
    for _ in range(max_length):
        new_frontier: list[RawWord] = []
        for word in frontier:
            new_frontier.append(word + (X,))
            new_frontier.append(word + (P,))
        words.extend(new_frontier)
        frontier = new_frontier
    return tuple(words)


def figure8_bosonic_basis() -> tuple[RawWord, ...]:
    return (
        tuple(),
        (X,),
        (P,),
        (X, X),
        (P, X),
        (P, P),
        (PSI, PSIDAG),
    )


def figure8_fermion_minus_basis() -> tuple[RawWord, ...]:
    return (
        (PSI,),
        (PSI, X),
        (PSI, P),
    )


def figure8_fermion_plus_basis() -> tuple[RawWord, ...]:
    return (
        (PSIDAG,),
        (PSIDAG, X),
        (PSIDAG, P),
    )


def figure8_ground_bosonic_basis() -> tuple[RawWord, ...]:
    return (
        (X,),
        (P,),
        (X, X),
        (P, X),
        (P, P),
    )


def figure8_ground_fermion_minus_basis() -> tuple[RawWord, ...]:
    return figure8_fermion_minus_basis()


def figure8_ground_fermion_plus_basis() -> tuple[RawWord, ...]:
    return figure8_fermion_plus_basis()


def figure8_ground_basis() -> tuple[RawWord, ...]:
    return figure8_ground_bosonic_basis()


def figure8_x2_exact(n: int, a: float = 1.0) -> float:
    return quadratic_x2_lower_bound(n, a)


def figure8_x4_exact(n: int, a: float = 1.0) -> float:
    if n <= 0:
        raise ValueError("n must be positive")
    if a <= 0:
        raise ValueError("a must be positive")
    return n * (2 * n**2 + 1) / (4 * a**2)


class Figure8QuadraticReducer:
    def __init__(self, *, n: int, a: float = 1.0, max_length: int = 4) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        if a <= 0:
            raise ValueError("a must be positive")
        self.n = int(n)
        self.a = float(a)
        self.max_length = int(max_length)
        self.words = all_charge_zero_words(max_length)
        self.word_to_index = {word: position for position, word in enumerate(self.words[1:], start=1)}

    def moment_expr(self, word: RawWord) -> LinearExpr:
        if len(word) > self.max_length:
            raise ValueError(f"word length {len(word)} exceeds max_length={self.max_length}")
        if raw_word_charge(word) != 0:
            return {}
        if not word:
            return {0: complex(self.n)}
        sign = 1.0j if raw_word_p_count(word) % 2 else 1.0 + 0.0j
        return {self.word_to_index[word]: sign}

    def apply_hamiltonian(self, word: RawWord) -> dict[RawWord, complex]:
        expr: dict[RawWord, complex] = {}
        for index, letter in enumerate(word):
            prefix = word[:index]
            suffix = word[index + 1 :]
            if letter == X:
                mapped = prefix + (P,) + suffix
                coefficient = -1.0j
            elif letter == P:
                mapped = prefix + (X,) + suffix
                coefficient = 1.0j * (self.a**2)
            elif letter == PSI:
                mapped = prefix + (PSI,) + suffix
                coefficient = -self.a
            elif letter == PSIDAG:
                mapped = prefix + (PSIDAG,) + suffix
                coefficient = self.a
            else:
                raise ValueError(f"unsupported letter {letter}")
            expr[mapped] = expr.get(mapped, 0.0 + 0.0j) + coefficient
        return {mapped: coefficient for mapped, coefficient in expr.items() if abs(coefficient) >= 1e-13}

    def ordinary_entry_expr(self, left: RawWord, right: RawWord) -> LinearExpr:
        return self.moment_expr(raw_word_dagger(left) + right)

    def ground_entry_expr(self, left: RawWord, right: RawWord) -> LinearExpr:
        expr: LinearExpr = {}
        for word, coefficient in self.apply_hamiltonian(right).items():
            _expr_add_scaled(expr, self.moment_expr(raw_word_dagger(left) + word), coefficient)
        return expr

    def dagger_constraint_expr(self, word: RawWord) -> LinearExpr:
        expr: LinearExpr = {}
        _expr_add_scaled(expr, self.moment_expr(raw_word_dagger(word)), 1.0)
        _expr_add_scaled(expr, self.moment_expr(word), -1.0 if raw_word_p_count(word) % 2 == 0 else 1.0)
        return expr

    def right_vacuum_expr(self, word: RawWord) -> LinearExpr:
        expr: LinearExpr = {}
        _expr_add_scaled(expr, self.moment_expr(word + (P,)), 1.0)
        _expr_add_scaled(expr, self.moment_expr(word + (X,)), -1.0j * self.a)
        return expr

    def left_vacuum_expr(self, word: RawWord) -> LinearExpr:
        expr: LinearExpr = {}
        _expr_add_scaled(expr, self.moment_expr((P,) + word), 1.0)
        _expr_add_scaled(expr, self.moment_expr((X,) + word), 1.0j * self.a)
        return expr

    def gauge_expr(self, word: RawWord) -> LinearExpr:
        return self.gauge_inserted_expr(tuple(), word)

    def gauge_inserted_expr(self, left: RawWord, right: RawWord) -> LinearExpr:
        expr: LinearExpr = {}
        _expr_add_scaled(expr, self.moment_expr(left + (X, P) + right), 1.0j)
        _expr_add_scaled(expr, self.moment_expr(left + (P, X) + right), -1.0j)
        _expr_add_scaled(expr, self.moment_expr(left + (PSI, PSIDAG) + right), -1.0)
        _expr_add_scaled(expr, self.moment_expr(left + (PSIDAG, PSI) + right), -1.0)
        _expr_add_scaled(expr, self.moment_expr(left + right), 2.0 * self.n)
        return expr

    def gauge_right_expr(self, word: RawWord) -> LinearExpr:
        return self.gauge_inserted_expr(word, tuple())

    def fermion_zero_left_expr(self, word: RawWord) -> LinearExpr:
        return self.fermion_zero_inserted_expr(tuple(), word)

    def fermion_zero_right_expr(self, word: RawWord) -> LinearExpr:
        return self.fermion_zero_inserted_expr(word, tuple())

    def fermion_zero_inserted_expr(self, left: RawWord, right: RawWord) -> LinearExpr:
        return self.moment_expr(left + (PSIDAG, PSI) + right)

    def fermion_full_left_expr(self, word: RawWord) -> LinearExpr:
        return self.fermion_full_inserted_expr(tuple(), word)

    def fermion_full_right_expr(self, word: RawWord) -> LinearExpr:
        return self.fermion_full_inserted_expr(word, tuple())

    def fermion_full_inserted_expr(self, left: RawWord, right: RawWord) -> LinearExpr:
        expr: LinearExpr = {}
        _expr_add_scaled(expr, self.moment_expr(left + (PSI, PSIDAG) + right), 1.0)
        _expr_add_scaled(expr, self.moment_expr(left + right), -float(self.n))
        return expr


@dataclass(frozen=True)
class Figure8Config:
    n_min: int = 1
    n_max: int = 10
    a: float = 1.0
    closure_length: int = 4
    solver: str = "AUTO"
    solver_eps: float = 1e-8
    solver_max_iters: int = 50000
    margin_tolerance: float = 1e-7
    psd_tolerance: float = 1e-7
    include_ground: bool = True

    def n_values(self) -> np.ndarray:
        return np.arange(self.n_min, self.n_max + 1, dtype=int)

    def to_json(self) -> dict[str, Any]:
        return {
            "n_min": self.n_min,
            "n_max": self.n_max,
            "a": self.a,
            "closure_length": self.closure_length,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "margin_tolerance": self.margin_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "include_ground": self.include_ground,
            "bosonic_basis_size": len(figure8_bosonic_basis()),
            "fermion_minus_basis_size": len(figure8_fermion_minus_basis()),
            "fermion_plus_basis_size": len(figure8_fermion_plus_basis()),
            "ground_basis_size": len(figure8_ground_basis()),
        }


@dataclass
class Figure8SolveResult:
    status: str
    objective_value: float | None
    feasible: bool
    solver_name: str | None
    psd_residual_main: float | None
    psd_residual_ground: float | None
    values: np.ndarray | None


def _solver_order(cp, solver: str) -> tuple[str, ...]:
    if solver != "AUTO":
        return (solver,)
    installed = set(cp.installed_solvers())
    order = []
    for candidate in ("CLARABEL", "CVXOPT", "SCS"):
        if candidate in installed:
            order.append(candidate)
    return tuple(order)


def _max_psd_residual(matrix_values: list[np.ndarray | None]) -> float | None:
    residuals: list[float] = []
    for matrix in matrix_values:
        if matrix is None:
            return None
        array = np.asarray(matrix, dtype=float)
        array = 0.5 * (array + array.T)
        norm = max(1.0, float(np.linalg.norm(array, ord=2)))
        minimum = float(np.linalg.eigvalsh(array).min())
        residuals.append(max(0.0, -minimum) / norm)
    return max(residuals, default=0.0)


def _quartic_bosonic_targets(
    reducer: Figure8QuadraticReducer,
    *,
    dxx_expr: LinearExpr,
) -> dict[RawWord, LinearExpr]:
    x2_expr = reducer.moment_expr((X, X))
    x4_expr = reducer.moment_expr((X, X, X, X))
    nx2 = {}
    _expr_add_scaled(nx2, x2_expr, float(reducer.n))
    half_dxx = {}
    _expr_add_scaled(half_dxx, dxx_expr, 0.5)

    nx2_plus_half_dxx: LinearExpr = {}
    _expr_add_scaled(nx2_plus_half_dxx, nx2, 1.0)
    _expr_add_scaled(nx2_plus_half_dxx, half_dxx, 1.0)

    x4_minus_nx2_minus_dxx: LinearExpr = {}
    _expr_add_scaled(x4_minus_nx2_minus_dxx, x4_expr, 1.0)
    _expr_add_scaled(x4_minus_nx2_minus_dxx, x2_expr, -float(reducer.n))
    _expr_add_scaled(x4_minus_nx2_minus_dxx, dxx_expr, -1.0)

    targets: dict[RawWord, LinearExpr] = {}
    targets[(X, X, X, X)] = x4_expr

    expr: LinearExpr = {}
    _expr_add_scaled(expr, nx2_plus_half_dxx, 1.0j)
    targets[(X, X, X, P)] = expr
    expr = {}
    _expr_add_scaled(expr, nx2_plus_half_dxx, -1.0j)
    targets[(P, X, X, X)] = expr

    expr = {}
    _expr_add_scaled(expr, dxx_expr, 0.5j)
    targets[(X, X, P, X)] = expr
    expr = {}
    _expr_add_scaled(expr, dxx_expr, -0.5j)
    targets[(X, P, X, X)] = expr

    targets[(X, X, P, P)] = x4_minus_nx2_minus_dxx
    targets[(P, P, X, X)] = x4_minus_nx2_minus_dxx

    expr = {}
    _expr_add_scaled(expr, dxx_expr, 0.5)
    targets[(X, P, X, P)] = expr
    targets[(P, X, P, X)] = expr

    targets[(X, P, P, X)] = x4_expr
    targets[(P, X, X, P)] = x4_expr

    expr = {}
    _expr_add_scaled(expr, x4_expr, 1.0j)
    targets[(X, P, P, P)] = expr
    expr = {}
    _expr_add_scaled(expr, dxx_expr, 0.5j)
    targets[(P, X, P, P)] = expr
    expr = {}
    _expr_add_scaled(expr, x4_minus_nx2_minus_dxx, 1.0j)
    targets[(P, P, X, P)] = expr
    expr = {}
    _expr_add_scaled(expr, x4_expr, -1.0j)
    targets[(P, P, P, X)] = expr
    targets[(P, P, P, P)] = x4_expr
    return targets


def solve_figure8_bound(
    *,
    n: int,
    observable: RawWord,
    a: float = 1.0,
    closure_length: int = 4,
    include_ground: bool = True,
    solver: str = "AUTO",
    solver_eps: float = 1e-8,
    solver_max_iters: int = 50000,
    margin_tolerance: float = 1e-7,
    psd_tolerance: float = 1e-7,
) -> Figure8SolveResult:
    cp = _import_cvxpy()
    reducer = Figure8QuadraticReducer(n=n, a=a, max_length=closure_length)
    dxx_index = len(reducer.words)
    variables = cp.Variable(len(reducer.words))
    dxx_expr = _single_var_expr(dxx_index)

    constraints = []
    for word in reducer.words:
        expr = reducer.dagger_constraint_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)

    for word in reducer.words:
        expr: LinearExpr = {}
        for mapped, coefficient in reducer.apply_hamiltonian(word).items():
            _expr_add_scaled(expr, reducer.moment_expr(mapped), coefficient)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)

    for word in all_bosonic_words(reducer.max_length - 1):
        expr = reducer.right_vacuum_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)
        expr = reducer.left_vacuum_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)

    for word in reducer.words:
        if len(word) <= reducer.max_length - 2:
            expr = reducer.gauge_expr(word)
            constraints.append(_expr_real(expr, variables) == 0.0)
            constraints.append(_expr_imag(expr, variables) == 0.0)
            expr = reducer.gauge_right_expr(word)
            constraints.append(_expr_real(expr, variables) == 0.0)
            constraints.append(_expr_imag(expr, variables) == 0.0)

    for word in all_bosonic_words(reducer.max_length - 2):
        expr = reducer.fermion_zero_left_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)
        expr = reducer.fermion_zero_right_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)
        expr = reducer.fermion_full_left_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)
        expr = reducer.fermion_full_right_expr(word)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)

    # Minimal split sector for Figure 8 quartic closure.
    expr = reducer.moment_expr((X,))
    constraints.append(_expr_real(expr, variables) == 0.0)
    constraints.append(_expr_imag(expr, variables) == 0.0)
    expr = reducer.moment_expr((P,))
    constraints.append(_expr_real(expr, variables) == 0.0)
    constraints.append(_expr_imag(expr, variables) == 0.0)

    x2_expr = reducer.moment_expr((X, X))
    x4_expr = reducer.moment_expr((X, X, X, X))
    expr = {}
    _expr_add_scaled(expr, x2_expr, 2.0 * reducer.a)
    _expr_add_scaled(expr, {0: -float(reducer.n**2)}, 1.0)
    constraints.append(_expr_real(expr, variables) == 0.0)
    constraints.append(_expr_imag(expr, variables) == 0.0)

    expr = {}
    _expr_add_scaled(expr, dxx_expr, 2.0 * reducer.a)
    _expr_add_scaled(expr, {0: -float(reducer.n)}, 1.0)
    constraints.append(_expr_real(expr, variables) == 0.0)
    constraints.append(_expr_imag(expr, variables) == 0.0)

    expr = {}
    _expr_add_scaled(expr, x4_expr, 4.0 * reducer.a)
    _expr_add_scaled(expr, x2_expr, -4.0 * float(reducer.n))
    _expr_add_scaled(expr, dxx_expr, -2.0)
    constraints.append(_expr_real(expr, variables) == 0.0)
    constraints.append(_expr_imag(expr, variables) == 0.0)

    quartic_targets = _quartic_bosonic_targets(reducer, dxx_expr=dxx_expr)
    for word, target_expr in quartic_targets.items():
        expr = {}
        _expr_add_scaled(expr, reducer.moment_expr(word), 1.0)
        _expr_add_scaled(expr, target_expr, -1.0)
        constraints.append(_expr_real(expr, variables) == 0.0)
        constraints.append(_expr_imag(expr, variables) == 0.0)

    bosonic_basis = figure8_bosonic_basis()
    fermion_minus_basis = figure8_fermion_minus_basis()
    fermion_plus_basis = figure8_fermion_plus_basis()

    ordinary_blocks = [
        [[reducer.ordinary_entry_expr(left, right) for right in bosonic_basis] for left in bosonic_basis],
        [[reducer.ordinary_entry_expr(left, right) for right in fermion_minus_basis] for left in fermion_minus_basis],
        [[reducer.ordinary_entry_expr(left, right) for right in fermion_plus_basis] for left in fermion_plus_basis],
    ]
    main_psd_blocks = [_build_real_psd_expression(cp, block, variables) for block in ordinary_blocks]

    scalar_exprs = [
        [{0: float(reducer.n)}, reducer.moment_expr((X,))],
        [reducer.moment_expr((X,)), dxx_expr],
    ]
    main_psd_blocks.append(_build_real_psd_expression(cp, scalar_exprs, variables))

    ground_psd_blocks: list[Any] = []
    if include_ground:
        ground_blocks = [
            figure8_ground_bosonic_basis(),
            figure8_ground_fermion_minus_basis(),
            figure8_ground_fermion_plus_basis(),
        ]
        for basis in ground_blocks:
            ground_exprs = [[reducer.ground_entry_expr(left, right) for right in basis] for left in basis]
            ground_psd_blocks.append(_build_real_psd_expression(cp, ground_exprs, variables))

    for block in main_psd_blocks:
        constraints.append(block >> 0)
    for block in ground_psd_blocks:
        constraints.append(block >> 0)

    objective = cp.Minimize(_expr_real(reducer.moment_expr(observable), variables))
    problem = cp.Problem(objective, constraints)

    best = Figure8SolveResult(
        status="solver_not_run",
        objective_value=None,
        feasible=False,
        solver_name=None,
        psd_residual_main=None,
        psd_residual_ground=None,
        values=None,
    )

    for solver_name in _solver_order(cp, solver):
        solve_kwargs: dict[str, Any] = {"solver": solver_name, "warm_start": True, "verbose": False}
        if solver_name == "SCS":
            solve_kwargs["eps"] = solver_eps
            solve_kwargs["max_iters"] = solver_max_iters
        elif solver_name == "CLARABEL":
            solve_kwargs["max_iter"] = solver_max_iters
        try:
            problem.solve(**solve_kwargs)
        except Exception:
            best = Figure8SolveResult(
                status=f"{solver_name.lower()}_failed",
                objective_value=None,
                feasible=False,
                solver_name=solver_name,
                psd_residual_main=None,
                psd_residual_ground=None,
                values=None,
            )
            continue

        status = str(problem.status)
        if status not in {"optimal", "optimal_inaccurate"} or variables.value is None:
            best = Figure8SolveResult(
                status=status,
                objective_value=None,
                feasible=False,
                solver_name=solver_name,
                psd_residual_main=None,
                psd_residual_ground=None,
                values=None,
            )
            continue

        main_residual = _max_psd_residual([block.value for block in main_psd_blocks])
        ground_residual = _max_psd_residual([block.value for block in ground_psd_blocks]) if include_ground else 0.0
        feasible = (
            main_residual is not None
            and ground_residual is not None
            and main_residual <= psd_tolerance
            and ground_residual <= psd_tolerance
        )
        result = Figure8SolveResult(
            status=status,
            objective_value=float(problem.value) if problem.value is not None else None,
            feasible=feasible,
            solver_name=solver_name,
            psd_residual_main=main_residual,
            psd_residual_ground=ground_residual if include_ground else None,
            values=np.asarray(variables.value, dtype=float) if feasible else None,
        )
        if feasible:
            return result
        best = result

    return best


def _plot_figure8(
    n_values: np.ndarray,
    x2_lower: np.ndarray,
    x2_exact: np.ndarray,
    x4_lower: np.ndarray,
    x4_exact: np.ndarray,
    *,
    out_path: str | Path,
) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axes = pyplot.subplots(1, 2, figsize=(10.8, 4.8))

    axes[0].plot(n_values, x2_exact, color="black", linewidth=1.8, label="exact")
    axes[0].plot(n_values, x2_lower, color="#1f78b4", marker="o", linewidth=2.1, label="lower bound")
    axes[0].set_xlabel(r"$N$")
    axes[0].set_ylabel(r"$\langle \mathrm{tr}\, X^2 \rangle$")
    axes[0].set_title("Figure 8: $\\langle \\mathrm{tr} X^2 \\rangle$")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].plot(n_values, x4_exact, color="black", linewidth=1.8, label="exact")
    axes[1].plot(n_values, x4_lower, color="#1f78b4", marker="o", linewidth=2.1, label="lower bound")
    axes[1].set_xlabel(r"$N$")
    axes[1].set_ylabel(r"$\langle \mathrm{tr}\, X^4 \rangle$")
    axes[1].set_title("Figure 8: $\\langle \\mathrm{tr} X^4 \\rangle$")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)


def run_figure8_scan(
    *,
    out_dir: str | Path,
    config: Figure8Config | None = None,
) -> dict[str, Any]:
    resolved = Figure8Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(resolved.to_json(), indent=2), encoding="utf-8")

    n_values = resolved.n_values()
    x2_lower = np.full(n_values.size, np.nan, dtype=float)
    x2_exact = np.array([figure8_x2_exact(int(n), resolved.a) for n in n_values], dtype=float)
    x4_lower = np.full(n_values.size, np.nan, dtype=float)
    x4_exact = np.array([figure8_x4_exact(int(n), resolved.a) for n in n_values], dtype=float)
    x2_statuses: list[str] = []
    x4_statuses: list[str] = []

    for index, n in enumerate(n_values):
        x2_result = solve_figure8_bound(
            n=int(n),
            observable=(X, X),
            a=resolved.a,
            closure_length=resolved.closure_length,
            include_ground=resolved.include_ground,
            solver=resolved.solver,
            solver_eps=resolved.solver_eps,
            solver_max_iters=resolved.solver_max_iters,
            margin_tolerance=resolved.margin_tolerance,
            psd_tolerance=resolved.psd_tolerance,
        )
        x2_statuses.append(
            f"{x2_result.solver_name.lower() if x2_result.solver_name else 'none'}:{x2_result.status}:{'feas' if x2_result.feasible else 'infeas'}"
        )
        if x2_result.feasible and x2_result.objective_value is not None:
            x2_lower[index] = x2_result.objective_value

        x4_result = solve_figure8_bound(
            n=int(n),
            observable=(X, X, X, X),
            a=resolved.a,
            closure_length=resolved.closure_length,
            include_ground=resolved.include_ground,
            solver=resolved.solver,
            solver_eps=resolved.solver_eps,
            solver_max_iters=resolved.solver_max_iters,
            margin_tolerance=resolved.margin_tolerance,
            psd_tolerance=resolved.psd_tolerance,
        )
        x4_statuses.append(
            f"{x4_result.solver_name.lower() if x4_result.solver_name else 'none'}:{x4_result.status}:{'feas' if x4_result.feasible else 'infeas'}"
        )
        if x4_result.feasible and x4_result.objective_value is not None:
            x4_lower[index] = x4_result.objective_value

    with (output_dir / "bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["N", "x2_lower", "x2_exact", "x2_status", "x4_lower", "x4_exact", "x4_status"],
        )
        writer.writeheader()
        for index, n in enumerate(n_values):
            writer.writerow(
                {
                    "N": int(n),
                    "x2_lower": float(x2_lower[index]) if np.isfinite(x2_lower[index]) else np.nan,
                    "x2_exact": float(x2_exact[index]),
                    "x2_status": x2_statuses[index],
                    "x4_lower": float(x4_lower[index]) if np.isfinite(x4_lower[index]) else np.nan,
                    "x4_exact": float(x4_exact[index]),
                    "x4_status": x4_statuses[index],
                }
            )

    _plot_figure8(n_values, x2_lower, x2_exact, x4_lower, x4_exact, out_path=output_dir / "figure8_bounds.png")

    summary_lines = [
        "# Figure 8 quadratic matrix warmup",
        "",
        "- Model: quadratic supersymmetric matrix QM with raw non-cyclic traced words.",
        "- Closure: Heisenberg + gauge + quadratic SUSY-vacuum constraints + minimal double-trace split sector.",
        "- Basis blocks: bosonic `7x7`, fermionic `3x3`, `3x3`.",
        f"- x2 feasible points: `{int(np.isfinite(x2_lower).sum())}/{len(n_values)}`",
        f"- x4 feasible points: `{int(np.isfinite(x4_lower).sum())}/{len(n_values)}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    return {
        "N": n_values,
        "x2_lower": x2_lower,
        "x2_exact": x2_exact,
        "x4_lower": x4_lower,
        "x4_exact": x4_exact,
        "x2_statuses": x2_statuses,
        "x4_statuses": x4_statuses,
        "out_dir": output_dir,
    }
