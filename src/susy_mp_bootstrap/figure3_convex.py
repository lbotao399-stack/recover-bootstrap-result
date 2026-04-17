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


def solve_figure3_point(
    *,
    g: float,
    level: int,
    epsilon: int,
    pure_constraint_order_extra: int = 0,
    energy_floor: float | None = None,
    solver: str = "SCS",
    solver_eps: float = 1e-6,
    solver_max_iters: int = 40000,
) -> tuple[str, float | None]:
    cp = _import_cvxpy()
    reducer = QuarticConvexReducer(g=g, epsilon=epsilon)
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
                ((i, j), dict(reducer.p_expr(i + j, 0))),
                ((i, level + j), dict(reducer.p_expr(i, j))),
                ((level + i, j), _expr_conjugate(dict(reducer.p_expr(j, i)))),
                ((level + i, level + j), {i + j: 1.0 + 0.0j}),
            )
            for (row, column), expr in block_entries:
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
