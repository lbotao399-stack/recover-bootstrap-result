from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np


FreeExpr = dict[int, complex]
WordExpr = dict[tuple[str, ...], complex]


def _import_cvxpy():
    import cvxpy as cp

    return cp


def _expr_add_scaled(target: FreeExpr, source: FreeExpr, scale: complex) -> None:
    if scale == 0:
        return
    for index, coefficient in source.items():
        updated = target.get(index, 0.0) + scale * coefficient
        if abs(updated) < 1e-13:
            target.pop(index, None)
        else:
            target[index] = updated


def _word_add_scaled(target: WordExpr, source: WordExpr, scale: complex) -> None:
    if scale == 0:
        return
    for word, coefficient in source.items():
        updated = target.get(word, 0.0) + scale * coefficient
        if abs(updated) < 1e-13:
            target.pop(word, None)
        else:
            target[word] = updated


def _expr_real(expr: FreeExpr, constant: float, variables) -> Any:
    result = float(np.real(constant))
    for index, coefficient in expr.items():
        result = result + float(np.real(coefficient)) * variables[index - 1]
    return result


def _expr_imag(expr: FreeExpr, constant: complex, variables) -> Any:
    result = float(np.imag(constant))
    for index, coefficient in expr.items():
        result = result + float(np.imag(coefficient)) * variables[index - 1]
    return result


def _evaluate_expr(constant: complex, expr: FreeExpr, values: np.ndarray) -> complex:
    total = complex(constant)
    for index, coefficient in expr.items():
        total += coefficient * values[index - 1]
    return total


def _word_level(word: tuple[str, ...]) -> int:
    return sum(2 if token == "q" else 1 for token in word)


def _reverse_word(word: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(reversed(word))


def _generate_words(max_level: int) -> list[tuple[str, ...]]:
    @lru_cache(maxsize=None)
    def words_of_level(level: int) -> tuple[tuple[str, ...], ...]:
        if level == 0:
            return (tuple(),)
        words: list[tuple[str, ...]] = []
        if level >= 1:
            for word in words_of_level(level - 1):
                words.append(word + ("z",))
        if level >= 2:
            for word in words_of_level(level - 2):
                words.append(word + ("q",))
        return tuple(words)

    basis: list[tuple[str, ...]] = [tuple()]
    for level in range(1, max_level + 1):
        basis.extend(words_of_level(level))
    return basis


BASIS_B5: tuple[tuple[str, ...], ...] = tuple(_generate_words(5))
BASIS_C4: tuple[tuple[str, ...], ...] = tuple(word for word in _generate_words(4) if word)
CANONICAL_BASIS_B5: tuple[tuple[str, ...], ...] = (
    tuple(),
    ("z",),
    ("q",),
    ("z", "z"),
    ("q", "z"),
    ("z", "z", "z"),
    ("q", "q"),
    ("q", "z", "z"),
    ("z", "z", "z", "z"),
    ("q", "q", "z"),
    ("q", "z", "z", "z"),
    ("z", "z", "z", "z", "z"),
)
CANONICAL_BASIS_C4: tuple[tuple[str, ...], ...] = (
    ("z",),
    ("z", "z"),
    ("z", "z", "z"),
    ("z", "z", "z", "z"),
    ("q",),
    ("q", "z"),
    ("q", "z", "z"),
    ("q", "q"),
)


class Figure4Reducer:
    def __init__(self, *, energy_hat: float, lambda_value: float, epsilon: int = -1) -> None:
        self.energy_hat = float(energy_hat)
        self.lambda_value = float(lambda_value)
        self.epsilon = int(epsilon)
        if self.epsilon not in (-1, 1):
            raise ValueError("epsilon must be +/- 1")

    @lru_cache(maxsize=None)
    def moment_expr(self, order: int) -> tuple[tuple[int, complex], ...]:
        if order < 0:
            return tuple()
        if order == 0:
            return ((0, 1.0 + 0.0j),)
        if order == 1:
            return ((1, 1.0 + 0.0j),)
        if order == 2:
            return ((2, 1.0 + 0.0j),)

        expr: FreeExpr = {}
        prefactor = 1.0 / (4.0 * (order - 1))
        _expr_add_scaled(
            expr,
            dict(self.moment_expr(order - 6)),
            prefactor * (order - 3) * (order - 4) * (order - 5),
        )
        _expr_add_scaled(
            expr,
            dict(self.moment_expr(order - 4)),
            prefactor * 8.0 * (order - 3) * self.energy_hat,
        )
        _expr_add_scaled(
            expr,
            dict(self.moment_expr(order - 2)),
            prefactor * 2.0 * self.lambda_value * (order - 2),
        )
        _expr_add_scaled(
            expr,
            dict(self.moment_expr(order - 3)),
            prefactor * (-4.0 * self.epsilon * (2 * order - 5)),
        )
        return tuple(sorted(expr.items()))

    @lru_cache(maxsize=None)
    def p_expr(self, q_power: int, z_power: int) -> tuple[tuple[int, complex], ...]:
        if q_power < 0 or z_power < 0:
            return tuple()
        if q_power == 0:
            return self.moment_expr(z_power)
        if q_power == 1:
            if z_power == 0:
                return tuple()
            expr = dict(self.moment_expr(z_power - 1))
            for index in list(expr):
                expr[index] *= -0.5j * z_power
            return tuple(sorted(expr.items()))

        expr: FreeExpr = {}
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 2, z_power)),
            2.0 * self.energy_hat,
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 2, z_power + 1)),
            -2.0 * self.epsilon,
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 2, z_power + 2)),
            0.5 * self.lambda_value,
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 2, z_power + 4)),
            -1.0,
        )
        prefactor = -2.0j * (q_power - 2)
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 3, z_power)),
            prefactor * self.epsilon,
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 3, z_power + 1)),
            prefactor * (-0.5 * self.lambda_value),
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 3, z_power + 3)),
            prefactor * 2.0,
        )
        prefactor2 = (q_power - 2) * (q_power - 3)
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 4, z_power)),
            prefactor2 * (-0.5 * self.lambda_value),
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 4, z_power + 2)),
            prefactor2 * 6.0,
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 5, z_power + 1)),
            4.0j * (q_power - 2) * (q_power - 3) * (q_power - 4),
        )
        _expr_add_scaled(
            expr,
            dict(self.p_expr(q_power - 6, z_power)),
            -(q_power - 2) * (q_power - 3) * (q_power - 4) * (q_power - 5),
        )
        return tuple(sorted(expr.items()))

    @lru_cache(maxsize=None)
    def normal_order(self, word: tuple[str, ...]) -> tuple[tuple[tuple[int, int], complex], ...]:
        for index in range(len(word) - 1):
            if word[index] == "z" and word[index + 1] == "q":
                swapped = word[:index] + ("q", "z") + word[index + 2 :]
                shortened = word[:index] + word[index + 2 :]
                expr: dict[tuple[int, int], complex] = {}
                _word_add_scaled(expr, dict(self.normal_order(swapped)), 1.0)
                _word_add_scaled(expr, dict(self.normal_order(shortened)), 1.0j)
                return tuple(sorted(expr.items()))
        q_power = sum(token == "q" for token in word)
        z_power = len(word) - q_power
        return (((q_power, z_power), 1.0 + 0.0j),)

    def expectation_word(self, word: tuple[str, ...]) -> tuple[tuple[int, complex], ...]:
        expr: FreeExpr = {}
        for (q_power, z_power), coefficient in dict(self.normal_order(word)).items():
            _expr_add_scaled(expr, dict(self.p_expr(q_power, z_power)), coefficient)
        return tuple(sorted(expr.items()))

    def commutator_atom(self, token: str) -> WordExpr:
        if token == "z":
            return {("q",): -1.0j}
        if token == "q":
            return {
                tuple(): 1.0j * self.epsilon,
                ("z",): -0.5j * self.lambda_value,
                ("z", "z", "z"): 2.0j,
            }
        raise ValueError(f"unknown token: {token}")

    @lru_cache(maxsize=None)
    def commutator_word(self, word: tuple[str, ...]) -> tuple[tuple[tuple[str, ...], complex], ...]:
        expr: WordExpr = {}
        for index, token in enumerate(word):
            prefix = word[:index]
            suffix = word[index + 1 :]
            for middle, coefficient in self.commutator_atom(token).items():
                expanded = prefix + middle + suffix
                _word_add_scaled(expr, {expanded: coefficient}, 1.0)
        return tuple(sorted(expr.items()))

    def matrix_entry_expr(self, left: tuple[str, ...], right: tuple[str, ...]) -> tuple[complex, FreeExpr]:
        word = _reverse_word(left) + right
        expr = dict(self.expectation_word(word))
        constant = expr.pop(0, 0.0 + 0.0j)
        return constant, expr

    def ground_entry_expr(self, left: tuple[str, ...], right: tuple[str, ...]) -> tuple[complex, FreeExpr]:
        expr: FreeExpr = {}
        constant = 0.0 + 0.0j
        left_dag = _reverse_word(left)
        for middle, coefficient in dict(self.commutator_word(right)).items():
            word = left_dag + middle
            term = dict(self.expectation_word(word))
            constant += coefficient * term.pop(0, 0.0 + 0.0j)
            _expr_add_scaled(expr, term, coefficient)
        return constant, expr


def figure4_operator_basis() -> tuple[tuple[str, ...], ...]:
    return CANONICAL_BASIS_B5


def figure4_ground_basis() -> tuple[tuple[str, ...], ...]:
    return CANONICAL_BASIS_C4


def figure4_string_operator_basis() -> tuple[tuple[str, ...], ...]:
    return BASIS_B5


def figure4_string_ground_basis() -> tuple[tuple[str, ...], ...]:
    return BASIS_C4


@dataclass(frozen=True)
class Figure4Config:
    g_min: float = 0.0
    g_max: float = 100.0
    num_g_small: int = 25
    num_g_large: int = 13
    epsilon: int = -1
    e_min: float = 0.0
    e_max: float = 7.0
    coarse_energy_steps: int = 60
    refine_steps: int = 10
    solver: str = "AUTO"
    solver_eps: float = 1e-7
    solver_max_iters: int = 50000
    psd_tolerance: float = 2e-6
    margin_tolerance: float = 1e-4
    guide_grid_size: int = 800
    guide_extent: float = 8.0
    seed_step: float = 0.002
    max_seed_tries: int = 8
    boundary_step_abs: float = 0.02
    boundary_step_rel: float = 0.05
    boundary_growth: float = 1.8

    def g_grid(self) -> np.ndarray:
        small = np.linspace(0.05, 5.0, self.num_g_small)
        large = np.linspace(5.0, self.g_max, self.num_g_large)
        return np.unique(np.concatenate(([0.0], small, large)))

    def to_json(self) -> dict[str, Any]:
        return {
            "g_min": self.g_min,
            "g_max": self.g_max,
            "num_g_small": self.num_g_small,
            "num_g_large": self.num_g_large,
            "epsilon": self.epsilon,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "coarse_energy_steps": self.coarse_energy_steps,
            "refine_steps": self.refine_steps,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "psd_tolerance": self.psd_tolerance,
            "margin_tolerance": self.margin_tolerance,
            "guide_grid_size": self.guide_grid_size,
            "guide_extent": self.guide_extent,
            "seed_step": self.seed_step,
            "max_seed_tries": self.max_seed_tries,
            "boundary_step_abs": self.boundary_step_abs,
            "boundary_step_rel": self.boundary_step_rel,
            "boundary_growth": self.boundary_growth,
            "basis_size": 12,
            "ground_basis_size": 8,
            "basis": "shifted-scaled cubic SUSY QM with exact canonical basis compression",
            "string_basis_size": 20,
            "string_ground_basis_size": 11,
        }


@dataclass
class Figure4FeasibilityResult:
    status: str
    feasible: bool
    m1: float | None
    m2: float | None
    margin: float | None
    psd_residual_main: float | None
    psd_residual_ground: float | None
    solver_name: str | None


@dataclass(frozen=True)
class Figure4GuideResult:
    energy: float
    energy_hat: float
    m1: float
    m2: float


def _build_real_psd_constraints(cp, matrix_exprs, variables) -> tuple[Any, Any]:
    size = len(matrix_exprs)
    re_matrix = [[0 for _ in range(size)] for _ in range(size)]
    im_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            constant, expr = matrix_exprs[row][column]
            re_matrix[row][column] = _expr_real(expr, constant, variables)
            im_matrix[row][column] = _expr_imag(expr, constant, variables)
    big_rows = []
    for row in range(size):
        big_rows.append(re_matrix[row] + [-entry for entry in im_matrix[row]])
    for row in range(size):
        big_rows.append(im_matrix[row] + re_matrix[row])
    return cp.bmat(big_rows), size


def _solver_order(cp, requested: str) -> list[str]:
    installed = set(cp.installed_solvers())
    solver = requested.upper()
    if solver == "AUTO":
        return [name for name in ("CLARABEL", "CVXOPT", "SCS") if name in installed]
    return [solver]


def figure4_feasibility(
    *,
    energy_hat: float,
    lambda_value: float,
    include_ground: bool,
    epsilon: int = -1,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    psd_tolerance: float = 1e-7,
    margin_tolerance: float = 1e-4,
    initial_guess: tuple[float, float] | None = None,
) -> Figure4FeasibilityResult:
    cp = _import_cvxpy()
    reducer = Figure4Reducer(energy_hat=energy_hat, lambda_value=lambda_value, epsilon=epsilon)
    variables = cp.Variable(2)
    if initial_guess is not None:
        variables.value = np.array(initial_guess, dtype=float)

    ordinary_basis = figure4_operator_basis()
    ground_basis = figure4_ground_basis()
    main_entries = [
        [reducer.matrix_entry_expr(left, right) for right in ordinary_basis]
        for left in ordinary_basis
    ]
    main_psd, _ = _build_real_psd_constraints(cp, main_entries, variables)
    margin = cp.Variable()
    constraints = [margin <= 1.0]
    constraints.append(main_psd - margin * np.eye(main_psd.shape[0]) >> 0)

    ground_psd = None
    if include_ground:
        ground_entries = [
            [reducer.ground_entry_expr(left, right) for right in ground_basis]
            for left in ground_basis
        ]
        ground_psd, _ = _build_real_psd_constraints(cp, ground_entries, variables)
        constraints.append(ground_psd - margin * np.eye(ground_psd.shape[0]) >> 0)

    problem = cp.Problem(cp.Maximize(margin), constraints)
    best = Figure4FeasibilityResult(
        status="solver_not_run",
        feasible=False,
        m1=None,
        m2=None,
        margin=None,
        psd_residual_main=None,
        psd_residual_ground=None,
        solver_name=None,
    )
    for solver_name in _solver_order(cp, solver):
        solve_kwargs: dict[str, Any] = {
            "solver": solver_name,
            "warm_start": True,
            "verbose": False,
        }
        if solver_name == "SCS":
            solve_kwargs["eps"] = solver_eps
            solve_kwargs["max_iters"] = solver_max_iters
        elif solver_name == "CLARABEL":
            solve_kwargs["max_iter"] = solver_max_iters
        try:
            problem.solve(**solve_kwargs)
        except Exception:
            best = Figure4FeasibilityResult(
                status=f"{solver_name.lower()}_failed",
                feasible=False,
                m1=None,
                m2=None,
                margin=None,
                psd_residual_main=None,
                psd_residual_ground=None,
                solver_name=solver_name,
            )
            continue

        status = str(problem.status)
        if status not in {"optimal", "optimal_inaccurate"} or variables.value is None or main_psd.value is None or margin.value is None:
            best = Figure4FeasibilityResult(
                status=status,
                feasible=False,
                m1=None,
                m2=None,
                margin=None,
                psd_residual_main=None,
                psd_residual_ground=None,
                solver_name=solver_name,
            )
            continue

        main_matrix = np.asarray(main_psd.value, dtype=float)
        main_matrix = 0.5 * (main_matrix + main_matrix.T)
        main_norm = max(1.0, float(np.linalg.norm(main_matrix, ord=2)))
        main_min = float(np.linalg.eigvalsh(main_matrix).min())
        main_residual = max(0.0, -main_min) / main_norm

        ground_residual = 0.0
        if ground_psd is not None:
            if ground_psd.value is None:
                best = Figure4FeasibilityResult(
                    status=status,
                    feasible=False,
                    m1=None,
                    m2=None,
                    margin=float(margin.value),
                    psd_residual_main=main_residual,
                    psd_residual_ground=None,
                    solver_name=solver_name,
                )
                continue
            ground_matrix = np.asarray(ground_psd.value, dtype=float)
            ground_matrix = 0.5 * (ground_matrix + ground_matrix.T)
            ground_norm = max(1.0, float(np.linalg.norm(ground_matrix, ord=2)))
            ground_min = float(np.linalg.eigvalsh(ground_matrix).min())
            ground_residual = max(0.0, -ground_min) / ground_norm

        feasible = (
            float(margin.value) >= -margin_tolerance
            and main_residual <= psd_tolerance
            and ground_residual <= psd_tolerance
        )
        result = Figure4FeasibilityResult(
            status=status,
            feasible=feasible,
            m1=float(variables.value[0]) if feasible else None,
            m2=float(variables.value[1]) if feasible else None,
            margin=float(margin.value),
            psd_residual_main=main_residual,
            psd_residual_ground=ground_residual if include_ground else None,
            solver_name=solver_name,
        )
        if feasible:
            return result
        best = result

    return best


def _energy_hat_from_physical(g: float, energy: float) -> float:
    return (energy - 1.0 / (32.0 * g * g)) / (g ** (2.0 / 3.0))


def _status_tag(result: Figure4FeasibilityResult) -> str:
    if result.solver_name is None:
        return result.status
    suffix = "feas" if result.feasible else "infeas"
    return f"{result.solver_name.lower()}:{result.status}:{suffix}"


def _scaled_fd_guide(
    *,
    g: float,
    epsilon: int,
    grid_size: int,
    extent: float,
) -> Figure4GuideResult:
    lambda_value = g ** (-4.0 / 3.0)
    z = np.linspace(-extent, extent, grid_size)
    dz = float(z[1] - z[0])
    diagonal = (1.0 / (dz * dz)) + 0.5 * z**4 + epsilon * z - 0.25 * lambda_value * z**2
    off_diagonal = np.full(grid_size - 1, -0.5 / (dz * dz), dtype=float)
    hamiltonian = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    eigenvalues, eigenvectors = np.linalg.eigh(hamiltonian)
    psi = eigenvectors[:, 0]
    density = np.abs(psi) ** 2
    density = density / float(np.sum(density) * dz)
    energy_hat = float(eigenvalues[0])
    return Figure4GuideResult(
        energy=(1.0 / (32.0 * g * g)) + (g ** (2.0 / 3.0)) * energy_hat,
        energy_hat=energy_hat,
        m1=float(np.sum(density * z) * dz),
        m2=float(np.sum(density * z * z) * dz),
    )


def _locate_seed(
    *,
    g: float,
    center_energy: float,
    include_ground: bool,
    config: Figure4Config,
    initial_guess: tuple[float, float] | None,
) -> tuple[float, Figure4FeasibilityResult] | None:
    tested: set[float] = set()
    offsets = [0.0]
    for radius in range(1, config.max_seed_tries):
        step = radius * config.seed_step
        offsets.extend((-step, step))

    for offset in offsets:
        energy = min(config.e_max, max(config.e_min, center_energy + offset))
        key = round(energy, 12)
        if key in tested:
            continue
        tested.add(key)
        result = figure4_feasibility(
            energy_hat=_energy_hat_from_physical(g, energy),
            lambda_value=g ** (-4.0 / 3.0),
            include_ground=include_ground,
            epsilon=config.epsilon,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            psd_tolerance=config.psd_tolerance,
            margin_tolerance=config.margin_tolerance,
            initial_guess=initial_guess,
        )
        if result.feasible:
            return energy, result
    return None


def _step_to_bracket(
    *,
    g: float,
    seed_energy: float,
    seed_result: Figure4FeasibilityResult,
    direction: float,
    include_ground: bool,
    config: Figure4Config,
) -> tuple[float, float, Figure4FeasibilityResult]:
    assert seed_result.feasible
    feasible_energy = seed_energy
    feasible_result = seed_result
    guess = (seed_result.m1, seed_result.m2)
    step = max(config.boundary_step_abs, config.boundary_step_rel * max(seed_energy, 1.0))

    while True:
        probe = feasible_energy + direction * step
        if probe <= config.e_min:
            return config.e_min, feasible_energy, feasible_result
        if probe >= config.e_max:
            return feasible_energy, config.e_max, feasible_result

        result = figure4_feasibility(
            energy_hat=_energy_hat_from_physical(g, probe),
            lambda_value=g ** (-4.0 / 3.0),
            include_ground=include_ground,
            epsilon=config.epsilon,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            psd_tolerance=config.psd_tolerance,
            margin_tolerance=config.margin_tolerance,
            initial_guess=guess,
        )
        if result.feasible:
            feasible_energy = probe
            feasible_result = result
            guess = (result.m1, result.m2)
            step *= config.boundary_growth
            continue

        if direction < 0:
            return probe, feasible_energy, feasible_result
        return feasible_energy, probe, feasible_result


def _find_first_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[0])


def _find_last_true(mask: np.ndarray) -> int | None:
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None
    return int(indices[-1])


def _refine_boundary(
    *,
    g: float,
    left: float,
    right: float,
    want_leftmost_feasible: bool,
    include_ground: bool,
    epsilon: int,
    solver: str,
    solver_eps: float,
    solver_max_iters: int,
    psd_tolerance: float,
    margin_tolerance: float,
    initial_guess: tuple[float, float] | None,
    refine_steps: int,
) -> tuple[float, Figure4FeasibilityResult]:
    low = float(left)
    high = float(right)
    last_feasible: Figure4FeasibilityResult | None = None
    for _ in range(refine_steps):
        middle = 0.5 * (low + high)
        result = figure4_feasibility(
            energy_hat=_energy_hat_from_physical(g, middle),
            lambda_value=g ** (-4.0 / 3.0),
            include_ground=include_ground,
            epsilon=epsilon,
            solver=solver,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
            psd_tolerance=psd_tolerance,
            margin_tolerance=margin_tolerance,
            initial_guess=initial_guess,
        )
        if result.feasible:
            last_feasible = result
            initial_guess = (result.m1, result.m2)
            if want_leftmost_feasible:
                high = middle
            else:
                low = middle
        else:
            if want_leftmost_feasible:
                low = middle
            else:
                high = middle
    if last_feasible is None:
        boundary = high if want_leftmost_feasible else low
        fallback = figure4_feasibility(
            energy_hat=_energy_hat_from_physical(g, boundary),
            lambda_value=g ** (-4.0 / 3.0),
            include_ground=include_ground,
            epsilon=epsilon,
            solver=solver,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
            psd_tolerance=psd_tolerance,
            margin_tolerance=margin_tolerance,
            initial_guess=initial_guess,
        )
        return boundary, fallback
    return (high if want_leftmost_feasible else low), last_feasible


def scan_figure4(config: Figure4Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    g_values = config.g_grid()
    lower = np.full(g_values.size, np.nan, dtype=float)
    upper = np.full(g_values.size, np.nan, dtype=float)
    lower_statuses: list[str] = []
    upper_statuses: list[str] = []

    for index, g in enumerate(g_values):
        if g == 0.0:
            lower[index] = 0.0
            upper[index] = 0.0
            lower_statuses.append("exact:g0")
            upper_statuses.append("exact:g0")
            continue

        guide = _scaled_fd_guide(
            g=float(g),
            epsilon=config.epsilon,
            grid_size=config.guide_grid_size,
            extent=config.guide_extent,
        )
        guide_guess = (guide.m1, guide.m2)

        ordinary_seed = _locate_seed(
            g=float(g),
            center_energy=guide.energy,
            include_ground=False,
            config=config,
            initial_guess=guide_guess,
        )
        if ordinary_seed is None:
            lower_statuses.append("ordinary:not_found")
        else:
            seed_energy, seed_result = ordinary_seed
            left, right, seed_for_refine = _step_to_bracket(
                g=float(g),
                seed_energy=seed_energy,
                seed_result=seed_result,
                direction=-1.0,
                include_ground=False,
                config=config,
            )
            boundary, refined = _refine_boundary(
                g=float(g),
                left=float(left),
                right=float(right),
                want_leftmost_feasible=True,
                include_ground=False,
                epsilon=config.epsilon,
                solver=config.solver,
                solver_eps=config.solver_eps,
                solver_max_iters=config.solver_max_iters,
                psd_tolerance=config.psd_tolerance,
                margin_tolerance=config.margin_tolerance,
                initial_guess=(seed_for_refine.m1, seed_for_refine.m2),
                refine_steps=config.refine_steps,
            )
            lower[index] = boundary
            lower_statuses.append(_status_tag(refined))

        ground_seed = _locate_seed(
            g=float(g),
            center_energy=guide.energy,
            include_ground=True,
            config=config,
            initial_guess=guide_guess,
        )
        if ground_seed is None:
            upper_statuses.append("ground:not_found")
        else:
            seed_energy, seed_result = ground_seed
            left, right, seed_for_refine = _step_to_bracket(
                g=float(g),
                seed_energy=seed_energy,
                seed_result=seed_result,
                direction=1.0,
                include_ground=True,
                config=config,
            )
            boundary, refined = _refine_boundary(
                g=float(g),
                left=float(left),
                right=float(right),
                want_leftmost_feasible=False,
                include_ground=True,
                epsilon=config.epsilon,
                solver=config.solver,
                solver_eps=config.solver_eps,
                solver_max_iters=config.solver_max_iters,
                psd_tolerance=config.psd_tolerance,
                margin_tolerance=config.margin_tolerance,
                initial_guess=(seed_for_refine.m1, seed_for_refine.m2),
                refine_steps=config.refine_steps,
            )
            upper[index] = boundary
            upper_statuses.append(_status_tag(refined))

    return g_values, lower, upper, lower_statuses, upper_statuses


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_figure4(
    g_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    out_path: str | Path,
) -> None:
    plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(9.4, 7.0))
    axis.scatter(
        g_values,
        upper,
        facecolors="none",
        edgecolors="#2ca25f",
        linewidths=1.4,
        s=40,
        label="upper",
        zorder=3,
    )
    axis.scatter(g_values, lower, color="#6a3d9a", s=14, label="lower", zorder=4)
    axis.set_xlim(0.0, 100.0)
    axis.set_ylim(0.0, 7.0)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$E$")
    axis.set_title("Figure 4 cubic SUSY QM ground-state energy bounds")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_figure4_scan(
    *,
    out_dir: str | Path,
    config: Figure4Config | None = None,
) -> dict[str, Any]:
    resolved_config = Figure4Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(resolved_config.to_json(), indent=2), encoding="utf-8")
    g_values, lower, upper, lower_statuses, upper_statuses = scan_figure4(resolved_config)

    fieldnames = ["g", "lower_energy", "lower_status", "upper_energy", "upper_status"]
    with (output_dir / "bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, g in enumerate(g_values):
            writer.writerow(
                {
                    "g": float(g),
                    "lower_energy": float(lower[index]) if np.isfinite(lower[index]) else np.nan,
                    "lower_status": lower_statuses[index],
                    "upper_energy": float(upper[index]) if np.isfinite(upper[index]) else np.nan,
                    "upper_status": upper_statuses[index],
                }
            )

    plot_figure4(g_values, lower, upper, out_path=output_dir / "figure4_bounds.png")

    summary_lines = [
        "# Figure 4 cubic SUSY QM bootstrap",
        "",
        "Model:",
        "- `W(x) = x^2 / 2 + g x^3 / 3`",
        "- shifted-scaled Hamiltonian `H = 1/(32 g^2) + g^{2/3} Hhat`",
        "- sector fixed to `epsilon = -1`",
        "",
        "Matrices:",
        "- ordinary bootstrap basis size: `12`",
        "- ground-state basis size: `8`",
        "- unreduced string counts kept for reference: `20` and `11`",
        "",
        "Status counts:",
        f"- lower: {', '.join(f'{name}={count}' for name, count in zip(*np.unique(np.asarray(lower_statuses, dtype=object), return_counts=True), strict=False))}",
        f"- upper: {', '.join(f'{name}={count}' for name, count in zip(*np.unique(np.asarray(upper_statuses, dtype=object), return_counts=True), strict=False))}",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "g_values": g_values,
        "lower": lower,
        "upper": upper,
        "lower_statuses": lower_statuses,
        "upper_statuses": upper_statuses,
    }
