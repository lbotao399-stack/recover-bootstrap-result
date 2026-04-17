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


class QuarticConvexReducer:
    def __init__(self, *, g: float, epsilon: int) -> None:
        if g <= 0:
            raise ValueError("g must be positive")
        if epsilon not in (-1, 1):
            raise ValueError("epsilon must be +/- 1")
        self.g = float(g)
        self.epsilon = int(epsilon)
        self.alpha = self.epsilon * self.g / sqrt(2.0)
        self.v_derivatives: dict[int, dict[int, complex]] = {
            1: {0: self.alpha, 1: 1.0, 3: self.g**2},
            2: {0: 1.0, 2: 3.0 * self.g**2},
            3: {1: 6.0 * self.g**2},
            4: {0: 6.0 * self.g**2},
        }

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
        for order in range(1, min(p_power - 1, 4) + 1):
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
        for derivative_order in range(1, min(order, 4) + 1):
            scale = comb(order, derivative_order) * (1j) ** derivative_order
            for degree, coefficient in self.v_derivatives[derivative_order].items():
                _expr_add_scaled(
                    expr,
                    dict(self.p_expr(order - derivative_order, degree)),
                    scale * coefficient,
                )
        return tuple(sorted(expr.items()))

    def energy_expr(self) -> MomentExpr:
        return {
            0: 1.0 / (4.0 * self.g**2),
            1: 1.5 * self.alpha,
            2: 1.0,
            4: 0.75 * self.g**2,
        }


@dataclass(frozen=True)
class Figure3Config:
    levels: tuple[int, ...] = (2, 3, 4, 5, 6, 7)
    g_min: float = 0.05
    g_max: float = 5.0
    num_g: int = 61
    epsilon: int = -1
    pure_constraint_order_extra: int = 0
    solver: str = "AUTO"
    solver_eps: float = 1e-6
    solver_max_iters: int = 40000

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
        }


def figure3_perturbation_curve(g_values: np.ndarray) -> np.ndarray:
    g_values = np.asarray(g_values, dtype=float)
    return 0.25 / (g_values**2) + 0.5 - (g_values**2) / 16.0 + 27.0 * (g_values**4) / 128.0


def figure3_k2_analytic_bound(g_values: np.ndarray) -> np.ndarray:
    g_values = np.asarray(g_values, dtype=float)
    return 0.25 / (g_values**2) + sqrt(3.0) / 4.0 - 9.0 * (g_values**2) / 32.0


def _required_moment_cutoff(level: int, reducer: QuarticConvexReducer, pure_constraint_order: int) -> int:
    max_index = 4
    for i in range(level):
        for j in range(level):
            for expr in (
                dict(reducer.p_expr(i + j, 0)),
                dict(reducer.p_expr(i, j)),
                _expr_conjugate(dict(reducer.p_expr(j, i))),
                {i + j: 1.0 + 0.0j},
            ):
                if expr:
                    max_index = max(max_index, max(expr))
    for order in range(1, pure_constraint_order + 1):
        expr = dict(reducer.pure_p_constraint(order))
        if expr:
            max_index = max(max_index, max(expr))
    return max_index


def _entry_expr_re_im(expr: MomentExpr, moments):
    return _expr_real(expr, moments), _expr_imag(expr, moments)


def _default_operator_scales(g: float) -> tuple[float, float]:
    x_scale = (1.0 + g**2) ** (-1.0 / 6.0)
    p_scale = 1.0 / x_scale
    return x_scale, p_scale


def _evaluate_expr(expr: MomentExpr, moment_values: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for index, coefficient in expr.items():
        total += coefficient * moment_values[index]
    return total


def solve_figure3_point(
    *,
    g: float,
    level: int,
    epsilon: int,
    pure_constraint_order_extra: int = 0,
    energy_floor: float | None = None,
    operator_scaling: bool = False,
    solver: str = "SCS",
    solver_eps: float = 1e-6,
    solver_max_iters: int = 40000,
) -> tuple[str, float | None]:
    cp = _import_cvxpy()
    reducer = QuarticConvexReducer(g=g, epsilon=epsilon)
    if operator_scaling:
        x_scale, p_scale = _default_operator_scales(g)
    else:
        x_scale, p_scale = 1.0, 1.0
    pure_constraint_order = 2 * level + pure_constraint_order_extra
    moment_cutoff = _required_moment_cutoff(level, reducer, pure_constraint_order)
    moments = cp.Variable(moment_cutoff + 1)
    constraints = [moments[0] == 1.0]

    for order in range(1, pure_constraint_order + 1):
        expr = dict(reducer.pure_p_constraint(order))
        real_part = _expr_real(expr, moments)
        imag_part = _expr_imag(expr, moments)
        constraints.append(real_part == 0.0)
        constraints.append(imag_part == 0.0)

    size = 2 * level
    re_matrix = [[0 for _ in range(size)] for _ in range(size)]
    im_matrix = [[0 for _ in range(size)] for _ in range(size)]
    for i in range(level):
        for j in range(level):
            block_entries = (
                ((i, j), dict(reducer.p_expr(i + j, 0)), (p_scale**i) * (p_scale**j)),
                ((i, level + j), dict(reducer.p_expr(i, j)), (p_scale**i) * (x_scale**j)),
                ((level + i, j), _expr_conjugate(dict(reducer.p_expr(j, i))), (x_scale**i) * (p_scale**j)),
                ((level + i, level + j), {i + j: 1.0 + 0.0j}, (x_scale**i) * (x_scale**j)),
            )
            for (row, column), expr, scale in block_entries:
                expr = {index: coefficient * scale for index, coefficient in expr.items()}
                re_matrix[row][column], im_matrix[row][column] = _entry_expr_re_im(expr, moments)

    big_psd_rows = []
    for row in range(size):
        big_psd_rows.append(re_matrix[row] + [-entry for entry in im_matrix[row]])
    for row in range(size):
        big_psd_rows.append(im_matrix[row] + re_matrix[row])
    constraints.append(cp.bmat(big_psd_rows) >> 0)

    objective_expr = _expr_real(reducer.energy_expr(), moments)
    if energy_floor is not None:
        constraints.append(objective_expr >= float(energy_floor))
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    installed = set(cp.installed_solvers())
    requested = solver.upper()
    if requested == "AUTO":
        solver_order = [name for name in ("CLARABEL", "SCS") if name in installed]
    else:
        solver_order = [requested]
        if requested == "CLARABEL" and "SCS" in installed:
            solver_order.append("SCS")

    last_status = "solver_not_run"
    for solver_name in solver_order:
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
            last_status = f"{solver_name.lower()}_failed"
            continue

        status = str(problem.status)
        last_status = status
        if status in {"infeasible", "infeasible_inaccurate", "unbounded", "unbounded_inaccurate"}:
            continue
        if problem.value is None:
            continue
        return status, float(problem.value)

    return last_status, None


class Figure3FeasibilityProblem:
    def __init__(
        self,
        *,
        g: float,
        level: int,
        epsilon: int,
        pure_constraint_order_extra: int = 0,
        operator_scaling: bool = True,
    ) -> None:
        cp = _import_cvxpy()
        self.cp = cp
        self.g = float(g)
        self.level = int(level)
        self.epsilon = int(epsilon)
        reducer = QuarticConvexReducer(g=g, epsilon=epsilon)
        self._pure_exprs: list[MomentExpr] = []
        if operator_scaling:
            x_scale, p_scale = _default_operator_scales(g)
        else:
            x_scale, p_scale = 1.0, 1.0

        pure_constraint_order = 2 * level + pure_constraint_order_extra
        moment_cutoff = _required_moment_cutoff(level, reducer, pure_constraint_order)
        moments = cp.Variable(moment_cutoff + 1)
        self.moments = moments
        constraints = [moments[0] == 1.0]

        for order in range(1, pure_constraint_order + 1):
            expr = dict(reducer.pure_p_constraint(order))
            self._pure_exprs.append(expr)
            constraints.append(_expr_real(expr, moments) == 0.0)
            constraints.append(_expr_imag(expr, moments) == 0.0)

        size = 2 * level
        re_matrix = [[0 for _ in range(size)] for _ in range(size)]
        im_matrix = [[0 for _ in range(size)] for _ in range(size)]
        for i in range(level):
            for j in range(level):
                block_entries = (
                    ((i, j), dict(reducer.p_expr(i + j, 0)), (p_scale**i) * (p_scale**j)),
                    ((i, level + j), dict(reducer.p_expr(i, j)), (p_scale**i) * (x_scale**j)),
                    ((level + i, j), _expr_conjugate(dict(reducer.p_expr(j, i))), (x_scale**i) * (p_scale**j)),
                    ((level + i, level + j), {i + j: 1.0 + 0.0j}, (x_scale**i) * (x_scale**j)),
                )
                for (row, column), expr, scale in block_entries:
                    expr = {index: coefficient * scale for index, coefficient in expr.items()}
                    re_matrix[row][column], im_matrix[row][column] = _entry_expr_re_im(expr, moments)

        big_psd_rows = []
        for row in range(size):
            big_psd_rows.append(re_matrix[row] + [-entry for entry in im_matrix[row]])
        for row in range(size):
            big_psd_rows.append(im_matrix[row] + re_matrix[row])
        self.big_psd_expr = cp.bmat(big_psd_rows)
        self.psd_margin = cp.Variable()
        constraints.append(self.big_psd_expr - self.psd_margin * np.eye(2 * size) >> 0)
        constraints.append(self.psd_margin <= 1.0)

        self.energy_expr = _expr_real(reducer.energy_expr(), moments)
        self.energy_cap = cp.Parameter()
        beta = 1.5 * reducer.alpha
        self._energy_constant = 1.0 / (4.0 * self.g**2)
        self._beta_squared_over_four = 0.25 * (beta**2)
        self.m2_cap = cp.Parameter(nonneg=True)
        self.m4_cap = cp.Parameter(nonneg=True)
        constraints.append(self.energy_expr <= self.energy_cap)
        if moment_cutoff >= 2:
            constraints.append(moments[2] >= 0.0)
            constraints.append(moments[2] <= self.m2_cap)
        if moment_cutoff >= 4:
            constraints.append(moments[4] >= 0.0)
            constraints.append(moments[4] <= self.m4_cap)
        self.problem = cp.Problem(cp.Maximize(self.psd_margin), constraints)
        self.installed_solvers = set(cp.installed_solvers())

    def _validate_current_solution(self) -> bool:
        if self.moments.value is None or self.big_psd_expr.value is None or self.energy_expr.value is None or self.psd_margin.value is None:
            return False
        moment_values = np.asarray(self.moments.value, dtype=float).reshape(-1)
        if not np.all(np.isfinite(moment_values)):
            return False
        if abs(moment_values[0] - 1.0) > 5e-6:
            return False

        max_residual = 0.0
        for expr in self._pure_exprs:
            max_residual = max(max_residual, abs(_evaluate_expr(expr, moment_values)))
        if max_residual > 5e-5:
            return False

        energy_value = float(self.energy_expr.value)
        if not np.isfinite(energy_value):
            return False
        if energy_value > float(self.energy_cap.value) + 5e-5:
            return False

        psd_matrix = np.asarray(self.big_psd_expr.value, dtype=float)
        if not np.all(np.isfinite(psd_matrix)):
            return False
        psd_matrix = 0.5 * (psd_matrix + psd_matrix.T)
        min_eigenvalue = float(np.linalg.eigvalsh(psd_matrix).min())
        if min_eigenvalue < -5e-5:
            return False
        if float(self.psd_margin.value) < -5e-5:
            return False

        return True

    def solve(
        self,
        energy_cap: float,
        *,
        solver: str = "AUTO",
        solver_eps: float = 1e-6,
        solver_max_iters: int = 40000,
    ) -> tuple[str, bool]:
        self.energy_cap.value = float(energy_cap)
        m4_cap = (float(energy_cap) - self._energy_constant + self._beta_squared_over_four) / (0.75 * self.g**2)
        if m4_cap <= 0:
            return "energy_cap_too_low", False
        self.m4_cap.value = float(m4_cap)
        self.m2_cap.value = float(np.sqrt(m4_cap))
        requested = solver.upper()
        if requested == "AUTO":
            solver_order = [name for name in ("CLARABEL", "SCS") if name in self.installed_solvers]
        else:
            solver_order = [requested]
            if requested == "CLARABEL" and "SCS" in self.installed_solvers:
                solver_order.append("SCS")

        last_status = "solver_not_run"
        for solver_name in solver_order:
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
                self.problem.solve(**solve_kwargs)
            except Exception:
                last_status = f"{solver_name.lower()}_failed"
                continue

            status = str(self.problem.status)
            last_status = status
            feasible = status in {"optimal", "optimal_inaccurate"} and self._validate_current_solution()
            if feasible:
                return status, True
            if status not in {"infeasible", "infeasible_inaccurate", "unbounded", "unbounded_inaccurate"}:
                continue

        return last_status, False


def solve_figure3_point_bisection(
    *,
    g: float,
    level: int,
    epsilon: int,
    lower_bound: float,
    upper_hint: float | None = None,
    pure_constraint_order_extra: int = 0,
    operator_scaling: bool = True,
    solver: str = "AUTO",
    solver_eps: float = 1e-6,
    solver_max_iters: int = 40000,
    tolerance: float = 1e-3,
    max_bisection_steps: int = 24,
    max_upper_expansions: int = 12,
) -> tuple[str, float | None]:
    feasibility = Figure3FeasibilityProblem(
        g=g,
        level=level,
        epsilon=epsilon,
        pure_constraint_order_extra=pure_constraint_order_extra,
        operator_scaling=operator_scaling,
    )
    perturbative = float(figure3_perturbation_curve(np.array([g]))[0])
    lower = float(lower_bound) - max(5e-3, 1e-3 * max(1.0, abs(lower_bound)))
    upper = max(lower + 0.05, perturbative + 0.25)
    if upper_hint is not None and np.isfinite(upper_hint):
        upper = max(upper, float(upper_hint))

    step = max(0.25, 0.1 * max(1.0, abs(upper)))
    last_status = "solver_not_run"
    feasible = False
    for _ in range(max_upper_expansions):
        last_status, feasible = feasibility.solve(
            upper,
            solver=solver,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
        )
        if feasible:
            break
        upper = upper + step
        step *= 1.8
    if not feasible:
        return last_status, None

    for _ in range(max_bisection_steps):
        if upper - lower <= tolerance:
            break
        mid = 0.5 * (lower + upper)
        last_status, feasible = feasibility.solve(
            mid,
            solver=solver,
            solver_eps=solver_eps,
            solver_max_iters=solver_max_iters,
        )
        if feasible:
            upper = mid
        else:
            lower = mid

    return last_status, upper


def scan_figure3(config: Figure3Config) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, list[str]]]:
    g_values = np.linspace(config.g_min, config.g_max, config.num_g)
    bounds: dict[int, np.ndarray] = {}
    statuses: dict[int, list[str]] = {}
    previous_level_bounds: np.ndarray | None = None
    for level in config.levels:
        level_bounds = np.full(config.num_g, np.nan, dtype=float)
        level_statuses: list[str] = []
        for index, g in enumerate(g_values):
            inherited_floor: float | None = None
            if previous_level_bounds is not None and np.isfinite(previous_level_bounds[index]):
                base = float(previous_level_bounds[index])
                inherited_floor = base - 1e-3 * max(1.0, abs(base))
            status, value = solve_figure3_point(
                g=float(g),
                level=level,
                epsilon=config.epsilon,
                pure_constraint_order_extra=config.pure_constraint_order_extra,
                energy_floor=inherited_floor,
                solver=config.solver,
                solver_eps=config.solver_eps,
                solver_max_iters=config.solver_max_iters,
            )
            level_statuses.append(status)
            if value is not None:
                level_bounds[index] = value
        bounds[level] = level_bounds
        statuses[level] = level_statuses
        previous_level_bounds = level_bounds
    return g_values, bounds, statuses


def refine_figure3_levels(
    *,
    g_values: np.ndarray,
    seed_bounds: dict[int, np.ndarray],
    levels: tuple[int, ...],
    epsilon: int,
    pure_constraint_order_extra: int = 0,
    solver: str = "AUTO",
    solver_eps: float = 1e-6,
    solver_max_iters: int = 40000,
    tolerance: float = 1e-3,
) -> tuple[dict[int, np.ndarray], dict[int, list[str]]]:
    bounds = {level: np.asarray(seed_bounds[level], dtype=float).copy() for level in seed_bounds}
    statuses = {level: ["seed"] * len(g_values) for level in seed_bounds}

    for level in levels:
        if level - 1 not in bounds:
            raise ValueError(f"missing seed bounds for level {level - 1}")
        refined = np.full(len(g_values), np.nan, dtype=float)
        refined_statuses: list[str] = []
        previous_upper: float | None = None
        previous_lower_curve = bounds[level - 1]

        for index, g in enumerate(g_values):
            lower_bound = float(previous_lower_curve[index])
            if not np.isfinite(lower_bound):
                refined_statuses.append("missing_lower_seed")
                previous_upper = None
                continue

            upper_hint = None
            if previous_upper is not None and np.isfinite(previous_upper):
                upper_hint = max(previous_upper + 0.05, lower_bound + 0.05)
            seed_current = bounds.get(level)
            if seed_current is not None and np.isfinite(seed_current[index]):
                candidate = float(seed_current[index])
                if upper_hint is None:
                    upper_hint = candidate
                else:
                    upper_hint = min(max(lower_bound + 0.02, candidate), upper_hint)
                    upper_hint = max(upper_hint, lower_bound + 0.02)

            status, value = solve_figure3_point_bisection(
                g=float(g),
                level=level,
                epsilon=epsilon,
                lower_bound=lower_bound,
                upper_hint=upper_hint,
                pure_constraint_order_extra=pure_constraint_order_extra,
                operator_scaling=True,
                solver=solver,
                solver_eps=solver_eps,
                solver_max_iters=solver_max_iters,
                tolerance=tolerance,
            )
            refined_statuses.append(status)
            if value is not None:
                refined[index] = value
                previous_upper = float(value)
            else:
                previous_upper = None

        bounds[level] = refined
        statuses[level] = refined_statuses

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
        level_statuses = statuses[level]
        for index, status in enumerate(level_statuses):
            if "unbounded" in status or "infeasible" in status:
                values[index] = np.nan
        values[(values < y_min) | (values > y_max)] = np.nan
        if previous is not None:
            for index in range(values.size):
                if np.isfinite(values[index]) and np.isfinite(previous[index]) and values[index] + 1e-4 < previous[index]:
                    values[index] = np.nan
                if level >= 4 and np.isfinite(values[index]) and np.isfinite(previous[index]) and values[index] - previous[index] > 0.5:
                    values[index] = np.nan
        cleaned[level] = values
        previous = values

    # Remove isolated spikes that survive range and nesting checks but are discontinuous in g.
    for level in sorted_levels:
        values = cleaned[level].copy()
        for index in range(1, values.size - 1):
            if not np.isfinite(values[index]):
                continue
            left = values[index - 1]
            right = values[index + 1]
            if np.isfinite(left) and np.isfinite(right):
                if abs(values[index] - left) > 1.0 and abs(values[index] - right) > 1.0:
                    values[index] = np.nan
        filtered = values.copy()
        for index in range(values.size):
            if not np.isfinite(values[index]):
                continue
            start = max(0, index - 2)
            stop = min(values.size, index + 3)
            neighborhood = values[start:stop]
            neighborhood = neighborhood[np.isfinite(neighborhood)]
            if neighborhood.size >= 3:
                local_median = float(np.median(neighborhood))
                if abs(values[index] - local_median) > 0.35:
                    filtered[index] = np.nan
        values = filtered
        if level >= 4:
            filtered = values.copy()
            last_kept: float | None = None
            for index, g_value in enumerate(g_values):
                if not np.isfinite(values[index]):
                    continue
                if last_kept is None:
                    last_kept = float(values[index])
                    continue
                jump_limit = 0.8 if g_value < 0.8 else 0.25
                if abs(values[index] - last_kept) > jump_limit:
                    filtered[index] = np.nan
                    continue
                last_kept = float(values[index])
            values = filtered
        cleaned[level] = values

    positions = np.linspace(0.1, 0.95, len(sorted_levels))
    for level, position in zip(sorted_levels, positions, strict=False):
        color = color_map(position)
        axis.plot(
            g_values,
            cleaned[level],
            color=color,
            linewidth=1.4,
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
    plot_figure3(g_values, bounds, statuses, out_path=output_dir / "figure3_bootstrap_vs_pt.png")

    summary_lines = [
        "# Figure 3 convex bootstrap",
        "",
        "Model:",
        "- `W(x) = x/(sqrt(2) g) + g x^3/(3 sqrt(2))`",
        f"- `epsilon = {resolved_config.epsilon}`",
        "",
        "Scan setup:",
        f"- levels: {', '.join(str(level) for level in resolved_config.levels)}",
        f"- g range: [{resolved_config.g_min}, {resolved_config.g_max}] with {resolved_config.num_g} points",
        f"- solver: {resolved_config.solver}",
        f"- pure-p constraint order: `2K + {resolved_config.pure_constraint_order_extra}`",
        "- higher levels inherit the previous level lower bound at the same `g` with a small numerical slack",
        "- display plot masks out-of-range and obvious solver-artifact points; raw `bounds.csv` is unchanged",
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
