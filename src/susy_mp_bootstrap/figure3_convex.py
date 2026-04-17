from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, sqrt
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np


MomentExpr = dict[int, complex]


def _import_cvxpy():
    import cvxpy as cp

    return cp


def _expr_add_scaled(target: MomentExpr, source: MomentExpr, scale: complex) -> None:
    if scale == 0:
        return
    for index, coefficient in source.items():
        updated = target.get(index, 0.0) + scale * coefficient
        if abs(updated) < 1e-13:
            target.pop(index, None)
        else:
            target[index] = updated


def _expr_conjugate(source: MomentExpr) -> MomentExpr:
    return {index: np.conjugate(coefficient) for index, coefficient in source.items()}


def _expr_real(expr: MomentExpr, variables) -> Any:
    result = 0
    for index, coefficient in expr.items():
        result = result + float(np.real(coefficient)) * variables[index]
    return result


def _expr_imag(expr: MomentExpr, variables) -> Any:
    result = 0
    for index, coefficient in expr.items():
        result = result + float(np.imag(coefficient)) * variables[index]
    return result


def _evaluate_expr(expr: MomentExpr, moment_values: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for index, coefficient in expr.items():
        total += coefficient * moment_values[index]
    return total


def _stationary_shift(g: float, epsilon: int) -> float:
    alpha = epsilon * g / sqrt(2.0)
    mu = -alpha
    for _ in range(64):
        value = g * g * mu * mu * mu + mu + alpha
        derivative = 3.0 * g * g * mu * mu + 1.0
        updated = mu - value / derivative
        if abs(updated - mu) < 1e-15:
            return float(updated)
        mu = updated
    return float(mu)


class PolynomialPotentialReducer:
    def __init__(
        self,
        *,
        derivatives: dict[int, dict[int, complex]],
        energy_terms: MomentExpr,
    ) -> None:
        self.v_derivatives = derivatives
        self._energy_terms = energy_terms
        self.max_derivative_order = max(derivatives)

    @lru_cache(maxsize=None)
    def p_expr(self, p_power: int, x_power: int) -> tuple[tuple[int, complex], ...]:
        if p_power < 0 or x_power < 0:
            return tuple()
        if p_power == 0:
            return ((x_power, 1.0 + 0.0j),)
        if p_power == 1:
            if x_power == 0:
                return tuple()
            return ((x_power - 1, -0.5j * x_power),)

        expr: MomentExpr = {}
        if x_power > 0:
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 1, x_power - 1)), -0.5j * x_power)
        for order in range(1, min(p_power - 1, self.max_derivative_order) + 1):
            scale = -(comb(p_power - 1, order) * (1j) ** (order + 1)) / (x_power + 1)
            for degree, coefficient in self.v_derivatives[order].items():
                _expr_add_scaled(
                    expr,
                    dict(self.p_expr(p_power - 1 - order, x_power + 1 + degree)),
                    scale * coefficient,
                )
        return tuple(sorted(expr.items()))

    @lru_cache(maxsize=None)
    def pure_p_constraint(self, order: int) -> tuple[tuple[int, complex], ...]:
        expr: MomentExpr = {}
        for derivative_order in range(1, min(order, self.max_derivative_order) + 1):
            scale = comb(order, derivative_order) * (1j) ** derivative_order
            for degree, coefficient in self.v_derivatives[derivative_order].items():
                _expr_add_scaled(
                    expr,
                    dict(self.p_expr(order - derivative_order, degree)),
                    scale * coefficient,
                )
        return tuple(sorted(expr.items()))

    def energy_expr(self) -> MomentExpr:
        return dict(self._energy_terms)


class QuarticConvexReducer(PolynomialPotentialReducer):
    def __init__(self, *, g: float, epsilon: int) -> None:
        if g <= 0:
            raise ValueError("g must be positive")
        if epsilon not in (-1, 1):
            raise ValueError("epsilon must be +/- 1")
        self.g = float(g)
        self.epsilon = int(epsilon)
        self.alpha = self.epsilon * self.g / sqrt(2.0)
        derivatives: dict[int, dict[int, complex]] = {
            1: {0: self.alpha, 1: 1.0, 3: self.g**2},
            2: {0: 1.0, 2: 3.0 * self.g**2},
            3: {1: 6.0 * self.g**2},
            4: {0: 6.0 * self.g**2},
        }
        energy_terms = {
            0: 1.0 / (4.0 * self.g**2),
            1: 1.5 * self.alpha,
            2: 1.0,
            4: 0.75 * self.g**2,
        }
        super().__init__(derivatives=derivatives, energy_terms=energy_terms)


class ShiftedScaledQuarticReducer(PolynomialPotentialReducer):
    def __init__(self, *, g: float, epsilon: int) -> None:
        if g <= 0:
            raise ValueError("g must be positive")
        if epsilon not in (-1, 1):
            raise ValueError("epsilon must be +/- 1")
        self.g = float(g)
        self.epsilon = int(epsilon)
        self.alpha = self.epsilon * self.g / sqrt(2.0)
        self.mu = _stationary_shift(self.g, self.epsilon)
        self.beta = (1.0 + 3.0 * self.g * self.g * self.mu * self.mu) / (2.0 * (self.g ** (4.0 / 3.0)))
        self.gamma = (self.g ** (1.0 / 3.0)) * self.mu
        constant_offset = 1.0 / (4.0 * self.g**2) + 0.5 * self.mu * self.mu + self.alpha * self.mu + 0.25 * self.g * self.g * self.mu**4
        scale = self.g ** (2.0 / 3.0)
        derivatives: dict[int, dict[int, complex]] = {
            1: {1: 2.0 * self.beta, 2: 3.0 * self.gamma, 3: 1.0},
            2: {0: 2.0 * self.beta, 1: 6.0 * self.gamma, 2: 3.0},
            3: {0: 6.0 * self.gamma, 1: 6.0},
            4: {0: 6.0},
        }
        energy_terms = {
            0: constant_offset,
            2: scale * (2.0 * self.beta),
            3: scale * (2.5 * self.gamma),
            4: scale * 0.75,
        }
        super().__init__(derivatives=derivatives, energy_terms=energy_terms)


@dataclass(frozen=True)
class Figure3Config:
    levels: tuple[int, ...] = (2, 3, 4, 5, 6)
    g_min: float = 0.05
    g_max: float = 5.0
    num_g: int = 61
    epsilon: int = -1
    pure_constraint_order_extra: int = 2
    solver: str = "AUTO"
    solver_eps: float = 1e-7
    solver_max_iters: int = 50000
    eq_tolerance: float = 1e-6
    psd_tolerance: float = 1e-6
    nesting_tolerance: float = 1e-7

    def to_json(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "g_min": self.g_min,
            "g_max": self.g_max,
            "num_g": self.num_g,
            "epsilon": self.epsilon,
            "pure_constraint_order_extra": self.pure_constraint_order_extra,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "eq_tolerance": self.eq_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "nesting_tolerance": self.nesting_tolerance,
            "formulation": "reduced shifted-scaled basis",
        }


@dataclass
class Figure3PointResult:
    status: str
    value: float | None
    certified: bool
    moments: np.ndarray | None
    eq_residual: float | None
    psd_residual: float | None
    solver_name: str | None


def figure3_perturbation_curve(g_values: np.ndarray) -> np.ndarray:
    g_values = np.asarray(g_values, dtype=float)
    return 0.25 / (g_values**2) + 0.5 - (g_values**2) / 16.0 + 27.0 * (g_values**4) / 128.0


def figure3_k2_analytic_bound(g_values: np.ndarray) -> np.ndarray:
    g_values = np.asarray(g_values, dtype=float)
    return 0.25 / (g_values**2) + sqrt(3.0) / 4.0 - 9.0 * (g_values**2) / 32.0


def _required_moment_cutoff(level: int, reducer: PolynomialPotentialReducer, pure_constraint_order: int) -> int:
    max_index = 4
    for i in range(1, level):
        for j in range(1, level):
            for expr in (
                dict(reducer.p_expr(i + j, 0)),
                dict(reducer.p_expr(i, j)),
                _expr_conjugate(dict(reducer.p_expr(j, i))),
                {i + j: 1.0 + 0.0j},
            ):
                if expr:
                    max_index = max(max_index, max(expr))
    for i in range(1, level):
        expr = dict(reducer.p_expr(i, 0))
        if expr:
            max_index = max(max_index, max(expr))
        max_index = max(max_index, i)
    for order in range(1, pure_constraint_order + 1):
        expr = dict(reducer.pure_p_constraint(order))
        if expr:
            max_index = max(max_index, max(expr))
    energy_expr = reducer.energy_expr()
    if energy_expr:
        max_index = max(max_index, max(energy_expr))
    return max_index


def _basis_expressions(level: int, reducer: PolynomialPotentialReducer) -> list[list[MomentExpr]]:
    basis: list[tuple[str, int]] = [("id", 0)]
    basis.extend(("p", power) for power in range(1, level))
    basis.extend(("x", power) for power in range(1, level))
    expressions: list[list[MomentExpr]] = []
    for kind_row, power_row in basis:
        row: list[MomentExpr] = []
        for kind_col, power_col in basis:
            if kind_row == "id" and kind_col == "id":
                expr = {0: 1.0 + 0.0j}
            elif kind_row == "id" and kind_col == "p":
                expr = dict(reducer.p_expr(power_col, 0))
            elif kind_row == "id" and kind_col == "x":
                expr = {power_col: 1.0 + 0.0j}
            elif kind_row == "p" and kind_col == "id":
                expr = _expr_conjugate(dict(reducer.p_expr(power_row, 0)))
            elif kind_row == "p" and kind_col == "p":
                expr = dict(reducer.p_expr(power_row + power_col, 0))
            elif kind_row == "p" and kind_col == "x":
                expr = dict(reducer.p_expr(power_row, power_col))
            elif kind_row == "x" and kind_col == "id":
                expr = {power_row: 1.0 + 0.0j}
            elif kind_row == "x" and kind_col == "p":
                expr = _expr_conjugate(dict(reducer.p_expr(power_col, power_row)))
            else:
                expr = {power_row + power_col: 1.0 + 0.0j}
            row.append(expr)
        expressions.append(row)
    return expressions


def _solver_order(cp, requested: str) -> list[str]:
    installed = set(cp.installed_solvers())
    solver = requested.upper()
    if solver == "AUTO":
        return [name for name in ("CLARABEL", "SCS") if name in installed]
    return [solver]


def _status_label(result: Figure3PointResult) -> str:
    if result.solver_name is None:
        return result.status
    suffix = "cert" if result.certified else "uncert"
    return f"{result.solver_name.lower()}:{result.status}:{suffix}"


def solve_figure3_point_result(
    *,
    g: float,
    level: int,
    epsilon: int,
    pure_constraint_order_extra: int = 2,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    eq_tolerance: float = 1e-6,
    psd_tolerance: float = 1e-6,
    initial_moments: np.ndarray | None = None,
) -> Figure3PointResult:
    cp = _import_cvxpy()
    reducer = ShiftedScaledQuarticReducer(g=g, epsilon=epsilon)
    pure_constraint_order = 2 * level + pure_constraint_order_extra
    moment_cutoff = _required_moment_cutoff(level, reducer, pure_constraint_order)
    moments = cp.Variable(moment_cutoff + 1)
    if initial_moments is not None and initial_moments.shape == (moment_cutoff + 1,):
        moments.value = initial_moments.copy()
    constraints = [moments[0] == 1.0]

    pure_constraints = [dict(reducer.pure_p_constraint(order)) for order in range(1, pure_constraint_order + 1)]
    for expr in pure_constraints:
        constraints.append(_expr_real(expr, moments) == 0.0)
        constraints.append(_expr_imag(expr, moments) == 0.0)

    matrix_exprs = _basis_expressions(level, reducer)
    size = 2 * level - 1
    re_matrix = [[0 for _ in range(size)] for _ in range(size)]
    im_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            expr = matrix_exprs[row][column]
            re_matrix[row][column] = _expr_real(expr, moments)
            im_matrix[row][column] = _expr_imag(expr, moments)

    big_psd_rows = []
    for row in range(size):
        big_psd_rows.append(re_matrix[row] + [-entry for entry in im_matrix[row]])
    for row in range(size):
        big_psd_rows.append(im_matrix[row] + re_matrix[row])
    big_psd = cp.bmat(big_psd_rows)
    constraints.append(big_psd >> 0)

    objective_expr = _expr_real(reducer.energy_expr(), moments)
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)

    best_result = Figure3PointResult(
        status="solver_not_run",
        value=None,
        certified=False,
        moments=None,
        eq_residual=None,
        psd_residual=None,
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
            best_result = Figure3PointResult(
                status=f"{solver_name.lower()}_failed",
                value=None,
                certified=False,
                moments=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue

        status = str(problem.status)
        if status in {"infeasible", "infeasible_inaccurate", "unbounded", "unbounded_inaccurate"}:
            best_result = Figure3PointResult(
                status=status,
                value=None,
                certified=False,
                moments=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue
        if problem.value is None or moments.value is None:
            best_result = Figure3PointResult(
                status=status,
                value=None,
                certified=False,
                moments=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue

        moment_values = np.asarray(moments.value, dtype=float).reshape(-1)
        norm_m = max(1.0, float(np.max(np.abs(moment_values))))
        eq_residual = 0.0
        for expr in pure_constraints:
            eq_residual = max(eq_residual, abs(_evaluate_expr(expr, moment_values)))
        eq_residual = float(eq_residual / norm_m)

        matrix_values = np.zeros((size, size), dtype=complex)
        for row in range(size):
            for column in range(size):
                matrix_values[row, column] = _evaluate_expr(matrix_exprs[row][column], moment_values)
        matrix_values = 0.5 * (matrix_values + matrix_values.conj().T)
        eigenvalues = np.linalg.eigvalsh(matrix_values)
        matrix_norm = max(1.0, float(np.linalg.norm(matrix_values, ord=2)))
        psd_residual = float(max(0.0, -float(np.min(eigenvalues))) / matrix_norm)

        certified = (
            np.isfinite(problem.value)
            and eq_residual <= eq_tolerance
            and psd_residual <= psd_tolerance
            and abs(moment_values[0] - 1.0) <= 5e-7
        )
        result = Figure3PointResult(
            status=status,
            value=float(problem.value) if certified else None,
            certified=certified,
            moments=moment_values if certified else None,
            eq_residual=eq_residual,
            psd_residual=psd_residual,
            solver_name=solver_name,
        )
        if certified:
            return result
        best_result = result

    return best_result


def solve_figure3_point(
    *,
    g: float,
    level: int,
    epsilon: int,
    pure_constraint_order_extra: int = 2,
    energy_floor: float | None = None,
    operator_scaling: bool = False,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
) -> tuple[str, float | None]:
    del energy_floor, operator_scaling
    result = solve_figure3_point_result(
        g=g,
        level=level,
        epsilon=epsilon,
        pure_constraint_order_extra=pure_constraint_order_extra,
        solver=solver,
        solver_eps=solver_eps,
        solver_max_iters=solver_max_iters,
    )
    return result.status, result.value


def solve_figure3_point_bisection(
    *,
    g: float,
    level: int,
    epsilon: int,
    lower_bound: float,
    upper_hint: float | None = None,
    pure_constraint_order_extra: int = 2,
    operator_scaling: bool = False,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    tolerance: float = 1e-4,
    max_bisection_steps: int = 24,
    max_upper_expansions: int = 12,
) -> tuple[str, float | None]:
    del operator_scaling, tolerance, max_bisection_steps, max_upper_expansions
    result = solve_figure3_point_result(
        g=g,
        level=level,
        epsilon=epsilon,
        pure_constraint_order_extra=pure_constraint_order_extra,
        solver=solver,
        solver_eps=solver_eps,
        solver_max_iters=solver_max_iters,
    )
    if result.value is None:
        return result.status, None
    if result.value + 1e-10 < lower_bound:
        return result.status, None
    if upper_hint is not None and result.value - 1e-10 > upper_hint:
        return result.status, None
    return result.status, result.value


class Figure3FeasibilityProblem:
    def __init__(
        self,
        *,
        g: float,
        level: int,
        epsilon: int,
        pure_constraint_order_extra: int = 2,
    ) -> None:
        self.g = float(g)
        self.level = int(level)
        self.epsilon = int(epsilon)
        self.pure_constraint_order_extra = int(pure_constraint_order_extra)

    def solve(
        self,
        energy_cap: float,
        *,
        solver: str = "AUTO",
        solver_eps: float = 1e-7,
        solver_max_iters: int = 50000,
    ) -> tuple[str, bool]:
        status, value = solve_figure3_point(
            g=self.g,
            level=self.level,
            epsilon=self.epsilon,
            pure_constraint_order_extra=self.pure_constraint_order_extra,
            solver=solver,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
        )
        feasible = value is not None and value <= energy_cap + 1e-9
        return status, feasible


def scan_figure3(config: Figure3Config) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, list[str]]]:
    g_values = np.linspace(config.g_min, config.g_max, config.num_g)
    bounds: dict[int, np.ndarray] = {}
    statuses: dict[int, list[str]] = {}
    for level in config.levels:
        level_bounds = np.full(config.num_g, np.nan, dtype=float)
        level_statuses: list[str] = []
        previous_moments: np.ndarray | None = None
        for index, g in enumerate(g_values):
            result = solve_figure3_point_result(
                g=float(g),
                level=level,
                epsilon=config.epsilon,
                pure_constraint_order_extra=config.pure_constraint_order_extra,
                solver=config.solver,
                solver_eps=config.solver_eps,
                solver_max_iters=config.solver_max_iters,
                eq_tolerance=config.eq_tolerance,
                psd_tolerance=config.psd_tolerance,
                initial_moments=previous_moments,
            )
            level_statuses.append(_status_label(result))
            if result.value is not None:
                level_bounds[index] = result.value
                previous_moments = result.moments
        bounds[level] = level_bounds
        statuses[level] = level_statuses
    return g_values, bounds, statuses


def refine_figure3_levels(
    *,
    g_values: np.ndarray,
    seed_bounds: dict[int, np.ndarray],
    levels: tuple[int, ...],
    epsilon: int,
    pure_constraint_order_extra: int = 2,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    tolerance: float = 1e-7,
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    del tolerance
    config = Figure3Config(
        levels=levels,
        g_min=float(g_values[0]),
        g_max=float(g_values[-1]),
        num_g=len(g_values),
        epsilon=epsilon,
        pure_constraint_order_extra=pure_constraint_order_extra,
        solver=solver,
        solver_eps=solver_eps,
        solver_max_iters=solver_max_iters,
    )
    bounds, statuses = dict(seed_bounds), {}
    _, fresh_bounds, fresh_statuses = scan_figure3(config)
    for level in levels:
        bounds[level] = fresh_bounds[level]
        statuses[level] = fresh_statuses[level]
    return bounds, statuses


def write_figure3_csv(path: str | Path, g_values: np.ndarray, bounds: dict[int, np.ndarray], statuses: dict[int, list[str]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["g"]
    for level in sorted(bounds):
        fieldnames.extend([f"level_{level}_energy", f"level_{level}_status"])
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, g in enumerate(g_values):
            row: dict[str, Any] = {"g": float(g)}
            for level in sorted(bounds):
                row[f"level_{level}_energy"] = float(bounds[level][index]) if np.isfinite(bounds[level][index]) else np.nan
                row[f"level_{level}_status"] = statuses[level][index]
            writer.writerow(row)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_figure3(
    g_values: np.ndarray,
    bounds: dict[int, np.ndarray],
    statuses: dict[int, list[str]],
    *,
    out_path: str | Path,
    nesting_tolerance: float = 1e-7,
) -> None:
    plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(9.2, 7.0))
    color_map = plt.get_cmap("viridis")
    sorted_levels = sorted(bounds)
    y_min, y_max = -1.0, 4.0

    cleaned: dict[int, np.ndarray] = {}
    previous: np.ndarray | None = None
    for level in sorted_levels:
        values = np.asarray(bounds[level], dtype=float).copy()
        for index, status in enumerate(statuses[level]):
            if ":cert" not in status:
                values[index] = np.nan
        values[(values < y_min) | (values > y_max)] = np.nan
        if previous is not None:
            for index in range(values.size):
                if np.isfinite(values[index]) and np.isfinite(previous[index]):
                    if values[index] + nesting_tolerance < previous[index]:
                        values[index] = np.nan
        cleaned[level] = values
        previous = values

    positions = np.linspace(0.1, 0.95, len(sorted_levels))
    for level, position in zip(sorted_levels, positions, strict=False):
        axis.plot(
            g_values,
            cleaned[level],
            color=color_map(position),
            linewidth=1.5,
            marker="o",
            markersize=2.8,
            label=f"K={level}",
        )
    axis.plot(
        g_values,
        figure3_perturbation_curve(g_values),
        color="black",
        linewidth=2.0,
        label="PT",
    )
    axis.set_xlim(0.0, 5.0)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$E_{\min}$")
    axis.set_title("Figure 3 quartic correction ground-state lower bounds")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2)
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_figure3_scan(
    *,
    out_dir: str | Path,
    config: Figure3Config | None = None,
) -> dict[str, Any]:
    resolved_config = Figure3Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(resolved_config.to_json(), indent=2), encoding="utf-8")
    g_values, bounds, statuses = scan_figure3(resolved_config)
    write_figure3_csv(output_dir / "bounds.csv", g_values, bounds, statuses)
    plot_figure3(
        g_values,
        bounds,
        statuses,
        out_path=output_dir / "figure3_bootstrap_vs_pt.png",
        nesting_tolerance=resolved_config.nesting_tolerance,
    )

    summary_lines = [
        "# Figure 3 convex bootstrap",
        "",
        "Model:",
        "- reduced basis: `(1, q, ..., q^{K-1}, y, ..., y^{K-1})`",
        "- shifted-scaled variables: `x = mu(g) + g^{-1/3} y`, `p = g^{1/3} q`",
        "- stationary point: `g^2 mu^3 + mu + epsilon g / sqrt(2) = 0`",
        f"- `epsilon = {resolved_config.epsilon}`",
        "",
        "Scan setup:",
        f"- levels: {', '.join(str(level) for level in resolved_config.levels)}",
        f"- g range: [{resolved_config.g_min}, {resolved_config.g_max}] with {resolved_config.num_g} points",
        f"- solver: {resolved_config.solver}",
        f"- pure-p constraint order: `2K + {resolved_config.pure_constraint_order_extra}`",
        "- continuation: warm-start by previous certified moments at the same level",
        "",
        "Status counts:",
    ]
    for level in sorted(bounds):
        unique, counts = np.unique(np.asarray(statuses[level], dtype=object), return_counts=True)
        text = ", ".join(f"{name}={int(count)}" for name, count in zip(unique.tolist(), counts.tolist(), strict=False))
        summary_lines.append(f"- K={level}: {text}")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "g_values": g_values,
        "bounds": bounds,
        "statuses": statuses,
    }
