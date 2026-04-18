from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from math import comb, exp, factorial, log
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from .figure4_cubic import figure4_rr_full_energy
from .figure5_cubic_smallg import (
    Figure5MomentRepresentation,
    _build_moment_representation,
    _evaluate_expr,
    _expr_imag,
    _expr_real,
    _solver_order,
    figure5_canonical_basis,
    figure5_energy_from_eta,
    figure5_eta_from_energy,
    figure5_instanton_energy_scale,
    Figure5CubicReducer,
)


MomentExpr = dict[int, complex]
WordExpr = dict[tuple[str, ...], complex]


def _import_cvxpy():
    import cvxpy as cp

    return cp


def _expr_add_scaled(target: MomentExpr, source: MomentExpr, scale: complex) -> None:
    if scale == 0:
        return
    for index, coefficient in source.items():
        updated = target.get(index, 0.0 + 0.0j) + scale * coefficient
        if abs(updated) < 1e-13:
            target.pop(index, None)
        else:
            target[index] = updated


def _figure6_status_label(result: "Figure6SolveResult") -> str:
    if result.solver_name is None:
        return result.status
    suffix = "feas" if result.feasible else "infeas"
    return f"{result.solver_name.lower()}:{result.status}:{suffix}"


def _double_factorial_odd(n: int) -> float:
    if n <= 0:
        return 1.0
    value = 1.0
    for factor in range(1, n + 1, 2):
        value *= factor
    return float(value)


def _ho_ground_moment(order: int) -> float:
    if order < 0:
        return 0.0
    if order % 2 == 1:
        return 0.0
    return float(_double_factorial_odd(order - 1) / (2 ** (order // 2)))


def figure6_smallg_mu(g: float) -> float:
    return float(-0.5 * g - (65.0 / 48.0) * g**3)


def figure6_smallg_sigma2(g: float) -> float:
    return float(0.5 + (53.0 / 48.0) * g**2)


def figure6_centered_gaussian_anchor(order: int, g: float) -> float:
    if order == 0:
        return 1.0
    mu = figure6_smallg_mu(g)
    sigma2 = figure6_smallg_sigma2(g)
    total = 0.0
    for j in range(order // 2 + 1):
        even = 2 * j
        coefficient = comb(order, even)
        central = _double_factorial_odd(2 * j - 1) * (sigma2**j)
        total += coefficient * central * (mu ** (order - even))
    return float(total)


def _figure6_smallg_scale(order: int, g: float) -> float:
    if order <= 0:
        return 1.0
    if order % 2 == 1:
        return float(max(abs(g), 1e-6))
    return float(max(g * g, 1e-6))


def _build_centered_moment_representation(cp, *, cutoff: int, g: float) -> Figure5MomentRepresentation:
    anchors = np.array([figure6_centered_gaussian_anchor(order, g) for order in range(cutoff + 1)], dtype=float)
    scales = np.array([1.0] + [_figure6_smallg_scale(order, g) for order in range(1, cutoff + 1)], dtype=float)
    delta = cp.Variable(cutoff)
    raw_moments: list[Any] = [1.0]
    for order in range(1, cutoff + 1):
        raw_moments.append(float(anchors[order]) + float(scales[order]) * delta[order - 1])
    return Figure5MomentRepresentation(
        raw_moments=raw_moments,
        decision_variable=delta,
        uses_smallg_parametrization=True,
        anchors=anchors,
        scales=scales,
    )


def figure6_operator_basis(level: int = 10) -> tuple[tuple[int, int], ...]:
    return figure5_canonical_basis(level // 2)


def figure6_ground_basis(level: int = 10) -> tuple[tuple[int, int], ...]:
    max_level = max(1, (level - 2) // 2)
    return tuple(entry for entry in figure5_canonical_basis(max_level) if entry != (0, 0))


def figure6_operator_basis_size(level: int = 10) -> int:
    return len(figure6_operator_basis(level))


def figure6_ground_basis_size(level: int = 10) -> int:
    return len(figure6_ground_basis(level))


def figure6_eta_value(g: float, energy: float | None) -> float:
    if energy is None or not np.isfinite(energy) or energy <= 0.0 or g <= 0.0:
        return float("nan")
    scale = figure5_instanton_energy_scale(g)
    if scale <= 0.0 or not np.isfinite(scale):
        return float("nan")
    return float(log(energy / scale))


def figure6_eta_array(g_values: np.ndarray, energies: np.ndarray) -> np.ndarray:
    return np.array([figure6_eta_value(float(g), float(energy)) if np.isfinite(energy) else np.nan for g, energy in zip(g_values, energies, strict=True)], dtype=float)


class Figure6CubicReducer(Figure5CubicReducer):
    @staticmethod
    def _basis_to_word(basis_element: tuple[int, int]) -> tuple[str, ...]:
        p_power, x_power = basis_element
        return ("p",) * p_power + ("x",) * x_power

    @lru_cache(maxsize=None)
    def potential_derivative_coeffs(self, order: int) -> tuple[tuple[int, complex], ...]:
        if order <= 0:
            return tuple()
        if order == 1:
            return (
                (0, -self.g + 0.0j),
                (1, 1.0 + 0.0j),
                (2, 3.0 * self.g + 0.0j),
                (3, 2.0 * (self.g**2) + 0.0j),
            )
        if order == 2:
            return (
                (0, 1.0 + 0.0j),
                (1, 6.0 * self.g + 0.0j),
                (2, 6.0 * (self.g**2) + 0.0j),
            )
        if order == 3:
            return (
                (0, 6.0 * self.g + 0.0j),
                (1, 12.0 * (self.g**2) + 0.0j),
            )
        if order == 4:
            return ((0, 12.0 * (self.g**2) + 0.0j),)
        return tuple()

    def expectation_x_p_x(self, *, x_left: int, p_total: int, x_right: int) -> MomentExpr:
        expr: MomentExpr = {}
        if x_left < 0 or p_total < 0 or x_right < 0:
            return expr
        for r in range(min(x_left, p_total) + 1):
            coefficient = (1.0j**r) * factorial(r) * comb(x_left, r) * comb(p_total, r)
            _expr_add_scaled(expr, dict(self.p_expr(p_total - r, x_left + x_right - r)), coefficient)
        return expr

    @lru_cache(maxsize=None)
    def normal_order_word(self, word: tuple[str, ...]) -> tuple[tuple[tuple[int, int], complex], ...]:
        for index in range(len(word) - 1):
            if word[index] == "x" and word[index + 1] == "p":
                swapped = word[:index] + ("p", "x") + word[index + 2 :]
                shortened = word[:index] + word[index + 2 :]
                expr: dict[tuple[int, int], complex] = {}
                for key, value in dict(self.normal_order_word(swapped)).items():
                    expr[key] = expr.get(key, 0.0 + 0.0j) + value
                for key, value in dict(self.normal_order_word(shortened)).items():
                    expr[key] = expr.get(key, 0.0 + 0.0j) + 1.0j * value
                return tuple(sorted((key, value) for key, value in expr.items() if abs(value) >= 1e-13))
        p_power = sum(token == "p" for token in word)
        x_power = len(word) - p_power
        return (((p_power, x_power), 1.0 + 0.0j),)

    def expectation_word(self, word: tuple[str, ...]) -> MomentExpr:
        expr: MomentExpr = {}
        for (p_power, x_power), coefficient in dict(self.normal_order_word(word)).items():
            _expr_add_scaled(expr, dict(self.p_expr(p_power, x_power)), coefficient)
        return expr

    def commutator_atom(self, token: str) -> WordExpr:
        if token == "x":
            return {("p",): -1.0j}
        if token == "p":
            return {
                tuple(): -1.0j * self.g,
                ("x",): 1.0j,
                ("x", "x"): 3.0j * self.g,
                ("x", "x", "x"): 2.0j * (self.g**2),
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
                expr[expanded] = expr.get(expanded, 0.0 + 0.0j) + coefficient
        return tuple(sorted((key, value) for key, value in expr.items() if abs(value) >= 1e-13))

    def ground_entry_expr(self, left: tuple[int, int], right: tuple[int, int]) -> MomentExpr:
        expr: MomentExpr = {}
        left_word = self._basis_to_word(left)
        right_word = self._basis_to_word(right)
        left_dag = tuple(reversed(left_word))
        for middle, coefficient in dict(self.commutator_word(right_word)).items():
            _expr_add_scaled(expr, self.expectation_word(left_dag + middle), coefficient)
        return expr


def _build_real_psd_expression(cp, matrix_exprs: list[list[MomentExpr]], moments) -> Any:
    size = len(matrix_exprs)
    re_matrix = [[0 for _ in range(size)] for _ in range(size)]
    im_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            re_matrix[row][column] = _expr_real(matrix_exprs[row][column], moments)
            im_matrix[row][column] = _expr_imag(matrix_exprs[row][column], moments)
    big_rows = []
    for row in range(size):
        big_rows.append(re_matrix[row] + [-entry for entry in im_matrix[row]])
    for row in range(size):
        big_rows.append(im_matrix[row] + re_matrix[row])
    return cp.bmat(big_rows)


def _matrix_values(matrix_exprs: list[list[MomentExpr]], moment_values: np.ndarray) -> np.ndarray:
    size = len(matrix_exprs)
    matrix = np.zeros((size, size), dtype=complex)
    for row in range(size):
        for column in range(size):
            matrix[row, column] = _evaluate_expr(matrix_exprs[row][column], moment_values)
    return 0.5 * (matrix + matrix.conj().T)


def _inverse_sqrt_hermitian(matrix: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    hermitian = 0.5 * (matrix + matrix.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(hermitian)
    clipped = np.maximum(eigenvalues, eps)
    inv_sqrt = np.diag(1.0 / np.sqrt(clipped))
    return eigenvectors @ inv_sqrt @ eigenvectors.conj().T


def _transform_expr_matrix(matrix_exprs: list[list[MomentExpr]], transform: np.ndarray) -> list[list[MomentExpr]]:
    size = len(matrix_exprs)
    transformed: list[list[MomentExpr]] = [[{} for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            expr: MomentExpr = {}
            for a in range(size):
                row_coef = transform[row, a]
                if abs(row_coef) < 1e-13:
                    continue
                for b in range(size):
                    coefficient = row_coef * np.conj(transform[column, b])
                    if abs(coefficient) < 1e-13:
                        continue
                    _expr_add_scaled(expr, matrix_exprs[a][b], coefficient)
            transformed[row][column] = expr
    return transformed


@dataclass(frozen=True)
class Figure6Config:
    level: int = 10
    g_min: float = 0.0
    g_max: float = 0.3
    num_g: int = 13
    g_values_override: tuple[float, ...] | None = None
    e_min: float = 0.0
    e_max: float = 2.0
    small_g_switch: float = 0.4
    use_smallg_eta_scan: bool = True
    smallg_eta_min: float = -6.0
    smallg_eta_max: float = 8.0
    use_smallg_parity_variables: bool = True
    use_centered_gaussian_anchor: bool = True
    use_ho_preconditioning: bool = True
    solver: str = "AUTO"
    solver_eps: float = 1e-7
    solver_max_iters: int = 50000
    margin_tolerance: float = 1e-3
    eq_tolerance: float = 5e-5
    psd_tolerance: float = 5e-4
    lower_bisection_steps: int = 10
    upper_bisection_steps: int = 10
    upper_probe_points: int = 9

    def g_grid(self) -> np.ndarray:
        if self.g_values_override is not None:
            return np.array(sorted(self.g_values_override, reverse=True), dtype=float)
        return np.linspace(self.g_max, self.g_min, self.num_g)

    def to_json(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "g_min": self.g_min,
            "g_max": self.g_max,
            "num_g": self.num_g,
            "g_values_override": None if self.g_values_override is None else list(self.g_values_override),
            "e_min": self.e_min,
            "e_max": self.e_max,
            "small_g_switch": self.small_g_switch,
            "use_smallg_eta_scan": self.use_smallg_eta_scan,
            "smallg_eta_min": self.smallg_eta_min,
            "smallg_eta_max": self.smallg_eta_max,
            "use_smallg_parity_variables": self.use_smallg_parity_variables,
            "use_centered_gaussian_anchor": self.use_centered_gaussian_anchor,
            "use_ho_preconditioning": self.use_ho_preconditioning,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "margin_tolerance": self.margin_tolerance,
            "eq_tolerance": self.eq_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "lower_bisection_steps": self.lower_bisection_steps,
            "upper_bisection_steps": self.upper_bisection_steps,
            "upper_probe_points": self.upper_probe_points,
            "ordinary_basis_size": figure6_operator_basis_size(self.level),
            "ground_basis_size": figure6_ground_basis_size(self.level),
            "note": f"Figure 6 level-{self.level} original-variable cubic ordinary+ground PSD with eta scan",
        }


@dataclass
class Figure6SolveResult:
    status: str
    feasible: bool
    objective_value: float | None
    moments: np.ndarray | None
    margin: float | None
    eq_residual: float | None
    psd_residual_main: float | None
    psd_residual_ground: float | None
    solver_name: str | None


@dataclass
class Figure6EnergyPoint:
    g: float
    energy: float | None
    status: str
    rr_upper: float
    moments: np.ndarray | None


@dataclass
class Figure6HierarchyLevelResult:
    level: int
    g_values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    lower_eta: np.ndarray
    upper_eta: np.ndarray
    lower_statuses: list[str]
    upper_statuses: list[str]


@lru_cache(maxsize=None)
def _figure6_reference_transforms(level: int) -> tuple[np.ndarray, np.ndarray]:
    reducer = Figure6CubicReducer(g=0.0, energy=0.0)
    reference_moments = np.array([_ho_ground_moment(order) for order in range(21)], dtype=float)
    ordinary_basis = figure6_operator_basis(level)
    ground_basis = figure6_ground_basis(level)
    ordinary_exprs = [[reducer.matrix_entry_expr(left, right) for right in ordinary_basis] for left in ordinary_basis]
    ground_exprs = [[reducer.ground_entry_expr(left, right) for right in ground_basis] for left in ground_basis]
    ordinary_matrix = _matrix_values(ordinary_exprs, reference_moments)
    ground_matrix = _matrix_values(ground_exprs, reference_moments)
    return _inverse_sqrt_hermitian(ordinary_matrix), _inverse_sqrt_hermitian(ground_matrix)


def figure6_feasibility(
    *,
    g: float,
    energy: float,
    include_ground: bool,
    level: int = 10,
    small_g_switch: float = 0.4,
    use_smallg_parity_variables: bool = True,
    use_centered_gaussian_anchor: bool = True,
    use_ho_preconditioning: bool = True,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    margin_tolerance: float = 1e-3,
    eq_tolerance: float = 5e-5,
    psd_tolerance: float = 5e-4,
    initial_moments: np.ndarray | None = None,
) -> Figure6SolveResult:
    cp = _import_cvxpy()
    reducer = Figure6CubicReducer(g=g, energy=energy)
    cutoff = 2 * level
    moment_repr = _build_moment_representation(
        cp,
        cutoff=cutoff,
        g=g,
        use_smallg_parametrization=False,
    )
    if use_smallg_parity_variables and use_centered_gaussian_anchor and g < small_g_switch:
        moment_repr = _build_centered_moment_representation(cp, cutoff=cutoff, g=g)
    elif use_smallg_parity_variables and g < small_g_switch:
        moment_repr = _build_moment_representation(
            cp,
            cutoff=cutoff,
            g=g,
            use_smallg_parametrization=True,
        )
    moment_repr.set_initial_guess(initial_moments)
    moments = moment_repr.raw_moments
    constraints = []
    if not moment_repr.uses_smallg_parametrization:
        constraints.append(moments[0] == 1.0)

    pure_relations = [reducer.pure_moment_relation(order) for order in range(3, cutoff + 1)]
    for expr in pure_relations:
        constraints.append(_expr_real(expr, moments) == 0.0)
        constraints.append(_expr_imag(expr, moments) == 0.0)

    ordinary_basis = figure6_operator_basis(level)
    ground_basis = figure6_ground_basis(level)
    ordinary_exprs = [[reducer.matrix_entry_expr(left, right) for right in ordinary_basis] for left in ordinary_basis]
    if use_ho_preconditioning:
        ordinary_transform, ground_transform = _figure6_reference_transforms(level)
        ordinary_exprs = _transform_expr_matrix(ordinary_exprs, ordinary_transform)
    main_psd = _build_real_psd_expression(cp, ordinary_exprs, moments)
    margin = cp.Variable()
    constraints.append(margin <= 1.0)
    constraints.append(margin >= -1.0)
    constraints.append(main_psd - margin * np.eye(main_psd.shape[0]) >> 0)

    ground_psd = None
    ground_exprs: list[list[MomentExpr]] | None = None
    if include_ground:
        ground_exprs = [[reducer.ground_entry_expr(left, right) for right in ground_basis] for left in ground_basis]
        if use_ho_preconditioning:
            if 'ground_transform' not in locals():
                _, ground_transform = _figure6_reference_transforms(level)
            ground_exprs = _transform_expr_matrix(ground_exprs, ground_transform)
        ground_psd = _build_real_psd_expression(cp, ground_exprs, moments)
        constraints.append(ground_psd - margin * np.eye(ground_psd.shape[0]) >> 0)

    problem = cp.Problem(cp.Maximize(margin), constraints)
    best = Figure6SolveResult(
        status="solver_not_run",
        feasible=False,
        objective_value=None,
        moments=None,
        margin=None,
        eq_residual=None,
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
            best = Figure6SolveResult(
                status=f"{solver_name.lower()}_failed",
                feasible=False,
                objective_value=None,
                moments=None,
                margin=None,
                eq_residual=None,
                psd_residual_main=None,
                psd_residual_ground=None,
                solver_name=solver_name,
            )
            continue

        status = str(problem.status)
        moment_values = moment_repr.extract_raw_values()
        if status not in {"optimal", "optimal_inaccurate"} or moment_values is None or margin.value is None or main_psd.value is None:
            best = Figure6SolveResult(
                status=status,
                feasible=False,
                objective_value=None,
                moments=None,
                margin=None,
                eq_residual=None,
                psd_residual_main=None,
                psd_residual_ground=None,
                solver_name=solver_name,
            )
            continue

        norm_m = max(1.0, float(np.max(np.abs(moment_values))))
        eq_residual = 0.0
        for expr in pure_relations:
            eq_residual = max(eq_residual, abs(_evaluate_expr(expr, moment_values)))
        eq_residual = float(eq_residual / norm_m)

        main_values = np.asarray(main_psd.value, dtype=float)
        main_values = 0.5 * (main_values + main_values.T)
        main_norm = max(1.0, float(np.linalg.norm(main_values, ord=2)))
        main_min = float(np.linalg.eigvalsh(main_values).min())
        main_residual = max(0.0, -main_min) / main_norm

        ground_residual = 0.0
        if include_ground:
            if ground_psd is None or ground_psd.value is None:
                best = Figure6SolveResult(
                    status=status,
                    feasible=False,
                    objective_value=None,
                    moments=None,
                    margin=float(margin.value),
                    eq_residual=eq_residual,
                    psd_residual_main=main_residual,
                    psd_residual_ground=None,
                    solver_name=solver_name,
                )
                continue
            ground_values = np.asarray(ground_psd.value, dtype=float)
            ground_values = 0.5 * (ground_values + ground_values.T)
            ground_norm = max(1.0, float(np.linalg.norm(ground_values, ord=2)))
            ground_min = float(np.linalg.eigvalsh(ground_values).min())
            ground_residual = max(0.0, -ground_min) / ground_norm

        feasible = (
            float(margin.value) >= -margin_tolerance
            and eq_residual <= eq_tolerance
            and main_residual <= psd_tolerance
            and ground_residual <= psd_tolerance
            and abs(moment_values[0] - 1.0) <= 5e-7
        )
        result = Figure6SolveResult(
            status=status,
            feasible=feasible,
            objective_value=float(margin.value),
            moments=moment_values if feasible else None,
            margin=float(margin.value),
            eq_residual=eq_residual,
            psd_residual_main=main_residual,
            psd_residual_ground=ground_residual if include_ground else None,
            solver_name=solver_name,
        )
        if feasible:
            return result
        best = result
    return best


def _eta_window(g: float, config: Figure6Config, rr_upper: float) -> tuple[float, float]:
    scale = figure5_instanton_energy_scale(g)
    if scale <= 0.0:
        return config.e_min, min(config.e_max, rr_upper)
    eta_low = config.smallg_eta_min
    eta_cap = min(config.smallg_eta_max, figure5_eta_from_energy(g, max(rr_upper, scale * exp(config.smallg_eta_min))))
    return eta_low, eta_cap


def _find_lower_bound(
    *,
    g: float,
    config: Figure6Config,
    initial_moments: np.ndarray | None,
    energy_floor_hint: float | None = None,
    energy_cap_hint: float | None = None,
) -> Figure6EnergyPoint:
    rr_upper = min(config.e_max, float(figure4_rr_full_energy(g)))
    if energy_cap_hint is not None and np.isfinite(energy_cap_hint):
        rr_upper = min(rr_upper, float(energy_cap_hint))
    use_eta_mode = config.use_smallg_eta_scan and g < config.small_g_switch and g > 0.0 and figure5_instanton_energy_scale(g) > 0.0
    guess = initial_moments

    if use_eta_mode:
        eta_low, eta_cap = _eta_window(g, config, rr_upper)
        if energy_floor_hint is not None and energy_floor_hint > 0.0:
            eta_low = max(eta_low, figure5_eta_from_energy(g, float(energy_floor_hint)))
        eta_cap = min(eta_cap, figure5_eta_from_energy(g, max(rr_upper, figure5_energy_from_eta(g, eta_low))))
        probe_parameters = np.unique(np.linspace(eta_low, eta_cap, config.upper_probe_points))
    else:
        low_energy = config.e_min if energy_floor_hint is None else max(config.e_min, float(energy_floor_hint))
        high_energy = rr_upper
        probe_parameters = np.unique(np.linspace(low_energy, high_energy, config.upper_probe_points))

    high_parameter = None
    high_result: Figure6SolveResult | None = None
    for parameter in probe_parameters:
        energy = figure5_energy_from_eta(g, float(parameter)) if use_eta_mode else float(parameter)
        result = figure6_feasibility(
            g=g,
            energy=float(energy),
            include_ground=False,
            level=config.level,
            small_g_switch=config.small_g_switch,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            use_centered_gaussian_anchor=config.use_centered_gaussian_anchor,
            use_ho_preconditioning=config.use_ho_preconditioning,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=guess,
        )
        if result.feasible:
            high_parameter = float(parameter)
            high_result = result
            guess = result.moments
            break
    if high_parameter is None or high_result is None:
        return Figure6EnergyPoint(g=g, energy=None, status="ordinary:not_found", rr_upper=rr_upper, moments=None)

    if use_eta_mode:
        low_parameter = config.smallg_eta_min
        low_energy = figure5_energy_from_eta(g, low_parameter)
        high_energy = figure5_energy_from_eta(g, high_parameter)
    else:
        low_parameter = config.e_min
        low_energy = low_parameter
        high_energy = float(high_parameter)

    low_result = figure6_feasibility(
        g=g,
        energy=float(low_energy),
        include_ground=False,
        level=config.level,
        small_g_switch=config.small_g_switch,
        use_smallg_parity_variables=config.use_smallg_parity_variables,
        use_centered_gaussian_anchor=config.use_centered_gaussian_anchor,
        use_ho_preconditioning=config.use_ho_preconditioning,
        solver=config.solver,
        solver_eps=config.solver_eps,
        solver_max_iters=config.solver_max_iters,
        margin_tolerance=config.margin_tolerance,
        eq_tolerance=config.eq_tolerance,
        psd_tolerance=config.psd_tolerance,
        initial_moments=guess,
    )
    if low_result.feasible:
        return Figure6EnergyPoint(g=g, energy=low_energy, status=_figure6_status_label(low_result), rr_upper=rr_upper, moments=low_result.moments)

    best_parameter = high_parameter
    best_result = high_result
    lower_infeasible = low_parameter
    for _ in range(config.lower_bisection_steps):
        middle = 0.5 * (lower_infeasible + best_parameter)
        energy = figure5_energy_from_eta(g, middle) if use_eta_mode else middle
        result = figure6_feasibility(
            g=g,
            energy=float(energy),
            include_ground=False,
            level=config.level,
            small_g_switch=config.small_g_switch,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            use_centered_gaussian_anchor=config.use_centered_gaussian_anchor,
            use_ho_preconditioning=config.use_ho_preconditioning,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=best_result.moments,
        )
        if result.feasible:
            best_parameter = middle
            best_result = result
        else:
            lower_infeasible = middle
    best_energy = figure5_energy_from_eta(g, best_parameter) if use_eta_mode else best_parameter
    return Figure6EnergyPoint(g=g, energy=float(best_energy), status=_figure6_status_label(best_result), rr_upper=rr_upper, moments=best_result.moments)


def _find_upper_bound(
    *,
    g: float,
    lower_point: Figure6EnergyPoint,
    config: Figure6Config,
    initial_moments: np.ndarray | None,
    previous_upper_energy: float | None,
    energy_cap_hint: float | None = None,
) -> Figure6EnergyPoint:
    if lower_point.energy is None:
        return Figure6EnergyPoint(g=g, energy=None, status="ordinary:not_found", rr_upper=lower_point.rr_upper, moments=None)

    rr_upper = lower_point.rr_upper
    if energy_cap_hint is not None and np.isfinite(energy_cap_hint):
        rr_upper = min(rr_upper, float(energy_cap_hint))
    use_eta_mode = config.use_smallg_eta_scan and g < config.small_g_switch and g > 0.0 and figure5_instanton_energy_scale(g) > 0.0
    guess = initial_moments if initial_moments is not None else lower_point.moments

    if use_eta_mode:
        lower_parameter = figure5_eta_from_energy(g, max(lower_point.energy, figure5_energy_from_eta(g, config.smallg_eta_min)))
        _, eta_cap = _eta_window(g, config, rr_upper)
        seeds = [lower_parameter, eta_cap]
        if previous_upper_energy is not None and previous_upper_energy > 0.0:
            try:
                seeds.append(figure5_eta_from_energy(g, previous_upper_energy))
            except ValueError:
                pass
        probe_parameters = np.unique(np.concatenate((np.array(seeds), np.linspace(lower_parameter, eta_cap, config.upper_probe_points))))
        probe_parameters = probe_parameters[(probe_parameters >= lower_parameter) & (probe_parameters <= eta_cap)]
    else:
        lower_parameter = lower_point.energy
        probe_parameters = np.unique(
            np.concatenate((np.array([lower_parameter, rr_upper]), np.linspace(lower_parameter, rr_upper, config.upper_probe_points)))
        )

    last_feasible_parameter = None
    last_feasible_result: Figure6SolveResult | None = None
    first_infeasible_above = None
    for parameter in probe_parameters:
        energy = figure5_energy_from_eta(g, float(parameter)) if use_eta_mode else float(parameter)
        result = figure6_feasibility(
            g=g,
            energy=float(energy),
            include_ground=True,
            level=config.level,
            small_g_switch=config.small_g_switch,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            use_centered_gaussian_anchor=config.use_centered_gaussian_anchor,
            use_ho_preconditioning=config.use_ho_preconditioning,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=guess,
        )
        if result.feasible:
            last_feasible_parameter = float(parameter)
            last_feasible_result = result
            guess = result.moments
        elif last_feasible_parameter is not None:
            first_infeasible_above = float(parameter)
            break

    if last_feasible_parameter is None or last_feasible_result is None:
        return Figure6EnergyPoint(g=g, energy=None, status="ground:not_found", rr_upper=rr_upper, moments=None)

    if first_infeasible_above is None:
        boundary_parameter = last_feasible_parameter
        boundary_energy = figure5_energy_from_eta(g, boundary_parameter) if use_eta_mode else boundary_parameter
        return Figure6EnergyPoint(g=g, energy=float(boundary_energy), status=_figure6_status_label(last_feasible_result), rr_upper=rr_upper, moments=last_feasible_result.moments)

    low = last_feasible_parameter
    high = first_infeasible_above
    best_parameter = last_feasible_parameter
    best_result = last_feasible_result
    for _ in range(config.upper_bisection_steps):
        middle = 0.5 * (low + high)
        energy = figure5_energy_from_eta(g, middle) if use_eta_mode else middle
        result = figure6_feasibility(
            g=g,
            energy=float(energy),
            include_ground=True,
            level=config.level,
            small_g_switch=config.small_g_switch,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            use_centered_gaussian_anchor=config.use_centered_gaussian_anchor,
            use_ho_preconditioning=config.use_ho_preconditioning,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=best_result.moments,
        )
        if result.feasible:
            low = middle
            best_parameter = middle
            best_result = result
        else:
            high = middle
    best_energy = figure5_energy_from_eta(g, best_parameter) if use_eta_mode else best_parameter
    return Figure6EnergyPoint(g=g, energy=float(best_energy), status=_figure6_status_label(best_result), rr_upper=rr_upper, moments=best_result.moments)


def scan_figure6(config: Figure6Config) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    g_values_desc = config.g_grid()
    lower = np.full(g_values_desc.size, np.nan, dtype=float)
    upper = np.full(g_values_desc.size, np.nan, dtype=float)
    lower_statuses: list[str] = []
    upper_statuses: list[str] = []
    previous_lower_moments: np.ndarray | None = None
    previous_upper_moments: np.ndarray | None = None
    previous_upper_energy: float | None = None

    for index, g in enumerate(g_values_desc):
        if g == 0.0:
            lower[index] = 0.0
            upper[index] = 0.0
            lower_statuses.append("exact:g0")
            upper_statuses.append("exact:g0")
            continue

        lower_point = _find_lower_bound(g=float(g), config=config, initial_moments=previous_lower_moments)
        lower_statuses.append(lower_point.status)
        if lower_point.energy is not None:
            lower[index] = lower_point.energy
            previous_lower_moments = lower_point.moments

        upper_point = _find_upper_bound(
            g=float(g),
            lower_point=lower_point,
            config=config,
            initial_moments=previous_upper_moments if previous_upper_moments is not None else lower_point.moments,
            previous_upper_energy=previous_upper_energy,
        )
        upper_statuses.append(upper_point.status)
        if upper_point.energy is not None:
            upper[index] = upper_point.energy
            previous_upper_moments = upper_point.moments
            previous_upper_energy = upper_point.energy

    return g_values_desc[::-1], lower[::-1], upper[::-1], lower_statuses[::-1], upper_statuses[::-1]


def scan_figure6_from_prior(
    config: Figure6Config,
    *,
    prior_g_values: np.ndarray,
    prior_lower: np.ndarray,
    prior_upper: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    prior_map: dict[float, tuple[float | None, float | None]] = {}
    for g, low, up in zip(prior_g_values, prior_lower, prior_upper, strict=True):
        low_value = float(low) if np.isfinite(low) else None
        up_value = float(up) if np.isfinite(up) else None
        prior_map[round(float(g), 12)] = (low_value, up_value)

    g_values_desc = config.g_grid()
    lower = np.full(g_values_desc.size, np.nan, dtype=float)
    upper = np.full(g_values_desc.size, np.nan, dtype=float)
    lower_statuses: list[str] = []
    upper_statuses: list[str] = []
    previous_lower_moments: np.ndarray | None = None
    previous_upper_moments: np.ndarray | None = None
    previous_upper_energy: float | None = None

    for index, g in enumerate(g_values_desc):
        if g == 0.0:
            lower[index] = 0.0
            upper[index] = 0.0
            lower_statuses.append("exact:g0")
            upper_statuses.append("exact:g0")
            continue

        prior_low, prior_up = prior_map.get(round(float(g), 12), (None, None))
        lower_point = _find_lower_bound(
            g=float(g),
            config=config,
            initial_moments=previous_lower_moments,
            energy_floor_hint=prior_low,
            energy_cap_hint=prior_up,
        )
        lower_statuses.append(lower_point.status)
        if lower_point.energy is not None:
            lower[index] = lower_point.energy
            previous_lower_moments = lower_point.moments

        upper_point = _find_upper_bound(
            g=float(g),
            lower_point=lower_point,
            config=config,
            initial_moments=previous_upper_moments if previous_upper_moments is not None else lower_point.moments,
            previous_upper_energy=previous_upper_energy,
            energy_cap_hint=prior_up,
        )
        upper_statuses.append(upper_point.status)
        if upper_point.energy is not None:
            upper[index] = upper_point.energy
            previous_upper_moments = upper_point.moments
            previous_upper_energy = upper_point.energy

    return g_values_desc[::-1], lower[::-1], upper[::-1], lower_statuses[::-1], upper_statuses[::-1]


def plot_figure6(
    g_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    out_path: str | Path,
    level: int,
    xlim: tuple[float, float] = (0.0, 0.3),
    ylim: tuple[float, float] = (0.0, 2.0),
) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(7.5, 5.2))
    lower_mask = np.isfinite(lower)
    upper_mask = np.isfinite(upper)
    axis.plot(g_values[lower_mask], lower[lower_mask], color="#1f78b4", linewidth=2.2, marker="o", markersize=4.2, label="lower")
    axis.plot(g_values[upper_mask], upper[upper_mask], color="#fdbf00", linewidth=2.2, marker="o", markersize=4.2, label="upper")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$E$")
    axis.set_title(f"Figure 6 level-{level} ground-state energy bounds")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)


def plot_figure6_eta(
    g_values: np.ndarray,
    lower_eta: np.ndarray,
    upper_eta: np.ndarray,
    *,
    out_path: str | Path,
    level: int,
    xlim: tuple[float, float] = (0.0, 0.3),
    ylim: tuple[float, float] = (-12.0, 12.0),
) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(7.5, 5.2))
    lower_mask = np.isfinite(lower_eta)
    upper_mask = np.isfinite(upper_eta)
    axis.plot(g_values[lower_mask], lower_eta[lower_mask], color="#1f78b4", linewidth=2.2, marker="o", markersize=4.2, label="lower")
    axis.plot(g_values[upper_mask], upper_eta[upper_mask], color="#fdbf00", linewidth=2.2, marker="o", markersize=4.2, label="upper")
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$\ln(E/E_{\rm inst})$")
    axis.set_title(f"Figure 6 level-{level} bounds in instanton units")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)


def _figure6_level_color(level: int) -> str:
    color_map = {
        7: "#e31a1c",
        8: "#6a3d9a",
        9: "#1f78b4",
        10: "#33a02c",
    }
    return color_map.get(level, "#444444")


def plot_figure6_eta_hierarchy(
    level_results: list[Figure6HierarchyLevelResult],
    *,
    out_path: str | Path,
    xlim: tuple[float, float] = (0.0, 0.3),
    ylim: tuple[float, float] = (-12.0, 12.0),
) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(8.2, 5.8))
    for result in level_results:
        color = _figure6_level_color(result.level)
        lower_mask = np.isfinite(result.lower_eta)
        upper_mask = np.isfinite(result.upper_eta)
        axis.plot(
            result.g_values[lower_mask],
            result.lower_eta[lower_mask],
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=3.8,
            label=f"L{result.level} lower",
        )
        axis.plot(
            result.g_values[upper_mask],
            result.upper_eta[upper_mask],
            color=color,
            linewidth=2.1,
            linestyle="--",
            marker="o",
            markersize=3.8,
            label=f"L{result.level} upper",
        )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$\ln(E/E_{\rm inst})$")
    axis.set_title("Figure 6 hierarchy in instanton units")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)


def plot_figure6_hierarchy(
    level_results: list[Figure6HierarchyLevelResult],
    *,
    out_path: str | Path,
    xlim: tuple[float, float] = (0.0, 0.3),
    ylim: tuple[float, float] = (0.0, 2.0),
) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as pyplot

    figure, axis = pyplot.subplots(figsize=(8.2, 5.8))
    for result in level_results:
        color = _figure6_level_color(result.level)
        lower_mask = np.isfinite(result.lower)
        upper_mask = np.isfinite(result.upper)
        axis.plot(
            result.g_values[lower_mask],
            result.lower[lower_mask],
            color=color,
            linewidth=2.1,
            marker="o",
            markersize=3.8,
            label=f"L{result.level} lower",
        )
        axis.plot(
            result.g_values[upper_mask],
            result.upper[upper_mask],
            color=color,
            linewidth=2.1,
            linestyle="--",
            marker="o",
            markersize=3.8,
            label=f"L{result.level} upper",
        )
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$E$")
    axis.set_title("Figure 6 hierarchy in original energy units")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=9)
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)


def _write_figure6_outputs(
    *,
    output_dir: Path,
    config: Figure6Config,
    g_values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    lower_statuses: list[str],
    upper_statuses: list[str],
    eta_ylim: tuple[float, float] = (-12.0, 12.0),
) -> tuple[np.ndarray, np.ndarray]:
    lower_eta = figure6_eta_array(g_values, lower)
    upper_eta = figure6_eta_array(g_values, upper)

    with (output_dir / "bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["g", "lower", "lower_eta", "lower_status", "upper", "upper_eta", "upper_status"],
        )
        writer.writeheader()
        for index, g in enumerate(g_values):
            writer.writerow(
                {
                    "g": float(g),
                    "lower": float(lower[index]) if np.isfinite(lower[index]) else np.nan,
                    "lower_eta": float(lower_eta[index]) if np.isfinite(lower_eta[index]) else np.nan,
                    "lower_status": lower_statuses[index],
                    "upper": float(upper[index]) if np.isfinite(upper[index]) else np.nan,
                    "upper_eta": float(upper_eta[index]) if np.isfinite(upper_eta[index]) else np.nan,
                    "upper_status": upper_statuses[index],
                }
            )

    plot_figure6(g_values, lower, upper, out_path=output_dir / "figure6_bounds.png", level=config.level)
    plot_figure6_eta(
        g_values,
        lower_eta,
        upper_eta,
        out_path=output_dir / "figure6_bounds_eta.png",
        level=config.level,
        ylim=eta_ylim,
    )

    summary_lines = [
        f"# Figure 6 level-{config.level} ground-state bounds",
        "",
        "- Model: cubic SUSY QM in original `(x,p)` variables.",
        f"- Ordinary block: canonical compressed basis `{figure6_operator_basis_size(config.level)}x{figure6_operator_basis_size(config.level)}`.",
        f"- Ground-state block: canonical compressed basis `{figure6_ground_basis_size(config.level)}x{figure6_ground_basis_size(config.level)}`.",
        "- Small-g scan variable: `E = E_inst(g) exp(eta)`.",
        "- Output also includes `eta = ln(E / E_inst)`.",
        "- Pure moment recursion imposed as linear equalities; no `m_n/g^2` forward elimination.",
        f"- Lower feasible points: `{int(np.isfinite(lower).sum())}/{len(g_values)}`",
        f"- Upper feasible points: `{int(np.isfinite(upper).sum())}/{len(g_values)}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return lower_eta, upper_eta


def run_figure6_scan(
    *,
    out_dir: str | Path,
    config: Figure6Config | None = None,
) -> dict[str, Any]:
    resolved_config = Figure6Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(resolved_config.to_json(), indent=2), encoding="utf-8")

    g_values, lower, upper, lower_statuses, upper_statuses = scan_figure6(resolved_config)
    lower_eta, upper_eta = _write_figure6_outputs(
        output_dir=output_dir,
        config=resolved_config,
        g_values=g_values,
        lower=lower,
        upper=upper,
        lower_statuses=lower_statuses,
        upper_statuses=upper_statuses,
    )

    return {
        "g_values": g_values,
        "lower": lower,
        "upper": upper,
        "lower_eta": lower_eta,
        "upper_eta": upper_eta,
        "lower_statuses": lower_statuses,
        "upper_statuses": upper_statuses,
        "out_dir": output_dir,
    }


def run_figure6_hierarchy(
    *,
    out_dir: str | Path,
    base_config: Figure6Config,
    levels: tuple[int, ...] = (7, 8, 9, 10),
    eta_ylim: tuple[float, float] = (-12.0, 12.0),
) -> dict[int, Figure6HierarchyLevelResult]:
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    level_results: list[Figure6HierarchyLevelResult] = []
    results_by_level: dict[int, Figure6HierarchyLevelResult] = {}
    prior_result: Figure6HierarchyLevelResult | None = None

    for level in levels:
        config = replace(base_config, level=level)
        level_dir = output_dir / f"level_{level}"
        level_dir.mkdir(parents=True, exist_ok=True)
        (level_dir / "config.json").write_text(json.dumps(config.to_json(), indent=2), encoding="utf-8")

        if prior_result is None:
            g_values, lower, upper, lower_statuses, upper_statuses = scan_figure6(config)
        else:
            g_values, lower, upper, lower_statuses, upper_statuses = scan_figure6_from_prior(
                config,
                prior_g_values=prior_result.g_values,
                prior_lower=prior_result.lower,
                prior_upper=prior_result.upper,
            )

        lower_eta, upper_eta = _write_figure6_outputs(
            output_dir=level_dir,
            config=config,
            g_values=g_values,
            lower=lower,
            upper=upper,
            lower_statuses=lower_statuses,
            upper_statuses=upper_statuses,
            eta_ylim=eta_ylim,
        )
        result = Figure6HierarchyLevelResult(
            level=level,
            g_values=g_values,
            lower=lower,
            upper=upper,
            lower_eta=lower_eta,
            upper_eta=upper_eta,
            lower_statuses=lower_statuses,
            upper_statuses=upper_statuses,
        )
        level_results.append(result)
        results_by_level[level] = result
        prior_result = result

    plot_figure6_hierarchy(level_results, out_path=output_dir / "figure6_hierarchy.png")
    plot_figure6_eta_hierarchy(level_results, out_path=output_dir / "figure6_hierarchy_eta.png", ylim=eta_ylim)

    summary_lines = [
        "# Figure 6 hierarchy",
        "",
        "- Main hierarchy plot uses the original energy coordinate `E`.",
        "- Vertical variable: `eta = ln(E / E_inst)`.",
        "- Levels scanned hierarchically: `" + ", ".join(str(level) for level in levels) + "`.",
        "- Level `L+1` uses level `L` lower/upper windows to shrink the scan domain.",
    ]
    for result in level_results:
        summary_lines.append(
            f"- Level {result.level}: lower `{int(np.isfinite(result.lower).sum())}/{len(result.g_values)}`, upper `{int(np.isfinite(result.upper).sum())}/{len(result.g_values)}`."
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return results_by_level
