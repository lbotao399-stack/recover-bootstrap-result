from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import comb, exp, factorial, log, sqrt
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np

from .figure4_cubic import figure4_rr_full_energy


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


def _expr_real(expr: MomentExpr, variables) -> Any:
    result = 0.0
    for index, coefficient in expr.items():
        result = result + float(np.real(coefficient)) * variables[index]
    return result


def _expr_complex(expr: MomentExpr, variables) -> Any:
    result = 0.0 + 0.0j
    for index, coefficient in expr.items():
        result = result + coefficient * variables[index]
    return result


def _expr_imag(expr: MomentExpr, variables) -> Any:
    result = 0.0
    for index, coefficient in expr.items():
        result = result + float(np.imag(coefficient)) * variables[index]
    return result


def _evaluate_expr(expr: MomentExpr, values: np.ndarray) -> complex:
    total = 0.0 + 0.0j
    for index, coefficient in expr.items():
        total += coefficient * values[index]
    return total


def _solver_order(cp, requested: str) -> list[str]:
    installed = set(cp.installed_solvers())
    solver = requested.upper()
    if solver == "AUTO":
        return [name for name in ("CLARABEL", "SCS") if name in installed]
    return [solver]


def figure5_canonical_basis(level: int) -> tuple[tuple[int, int], ...]:
    basis: list[tuple[int, int]] = []
    for p_power in range(level // 2 + 1):
        for x_power in range(level - 2 * p_power + 1):
            basis.append((p_power, x_power))
    return tuple(basis)


def figure5_instanton_energy_scale(g: float) -> float:
    if g <= 0.0:
        return 0.0
    return float((1.0 / (2.0 * np.pi)) * exp(-1.0 / (3.0 * g * g)))


def figure5_energy_from_eta(g: float, eta: float) -> float:
    return float(figure5_instanton_energy_scale(g) * exp(eta))


def figure5_eta_from_energy(g: float, energy: float) -> float:
    scale = figure5_instanton_energy_scale(g)
    if g <= 0.0 or scale <= 0.0 or energy <= 0.0:
        raise ValueError("eta is defined only for positive g and positive energy")
    return float(log(energy / scale))


def figure5_smallg_center_mu(g: float) -> float:
    return float(-0.5 * g - (65.0 / 48.0) * g**3)


def figure5_smallg_center_sigma2(g: float) -> float:
    return float(0.5 + (53.0 / 48.0) * g**2)


def figure5_smallg_center_sigma(g: float) -> float:
    return float(sqrt(max(figure5_smallg_center_sigma2(g), 1e-12)))


def figure5_ho_ground_moment(order: int) -> float:
    if order < 0:
        return 0.0
    if order % 2 == 1:
        return 0.0
    value = 1.0
    for factor in range(1, order, 2):
        value *= factor
    return float(value / (2 ** (order // 2)))


def figure5_smallg_anchor(order: int, g: float) -> float:
    if order == 0:
        return 1.0
    if order == 1:
        return float(-0.5 * g - (65.0 / 48.0) * g**3)
    if order == 2:
        return float(0.5 + (65.0 / 48.0) * g**2 + (19165.0 / 2304.0) * g**4)
    if order == 3:
        return float(-(5.0 / 4.0) * g - (745.0 / 96.0) * g**3)
    if order == 4:
        return float(0.75 + (205.0 / 32.0) * g**2)
    return figure5_ho_ground_moment(order)


def figure5_smallg_scale(order: int, g: float) -> float:
    if order <= 0:
        return 1.0
    if order % 2 == 1:
        return float(max(abs(g), 1e-6))
    return float(max(g * g, 1e-6))


@dataclass(frozen=True)
class Figure5Config:
    levels: tuple[int, ...] = (7, 8, 9, 10)
    g_min: float = 0.3
    g_max: float = 1.0
    num_g: int = 16
    g_values_override: tuple[float, ...] | None = None
    e_min: float = 0.0
    e_max: float = 0.3
    small_g_switch: float = 0.4
    use_smallg_eta_scan: bool = True
    smallg_eta_min: float = -6.0
    smallg_eta_max: float = 6.0
    use_smallg_centered_basis: bool = False
    use_smallg_parity_variables: bool = False
    use_smallg_nu_objective: bool = True
    solver: str = "AUTO"
    solver_eps: float = 1e-7
    solver_max_iters: int = 50000
    margin_tolerance: float = 5e-5
    eq_tolerance: float = 5e-6
    psd_tolerance: float = 5e-6
    bisection_steps: int = 14
    upper_probe_points: int = 5
    energy_floor_slack: float = 0.0
    x2_energy_scan_points: int = 7
    x2_edge_slack: float = 1e-4
    x2_lock_to_edge: bool = True
    use_completed_basis: bool = True

    def g_grid(self) -> np.ndarray:
        if self.g_values_override is not None:
            return np.array(sorted(self.g_values_override, reverse=True), dtype=float)
        return np.linspace(self.g_max, self.g_min, self.num_g)

    def to_json(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
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
            "use_smallg_centered_basis": self.use_smallg_centered_basis,
            "use_smallg_parity_variables": self.use_smallg_parity_variables,
            "use_smallg_nu_objective": self.use_smallg_nu_objective,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "margin_tolerance": self.margin_tolerance,
            "eq_tolerance": self.eq_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "bisection_steps": self.bisection_steps,
            "upper_probe_points": self.upper_probe_points,
            "energy_floor_slack": self.energy_floor_slack,
            "x2_energy_scan_points": self.x2_energy_scan_points,
            "x2_edge_slack": self.x2_edge_slack,
            "x2_lock_to_edge": self.x2_lock_to_edge,
            "use_completed_basis": self.use_completed_basis,
            "basis": "canonical p^a x^b basis in original (x,p) variables",
            "note": "small-g cubic ordinary bootstrap with hybrid eta scan and centered basis below the small-g switch",
        }


@dataclass
class Figure5SolveResult:
    status: str
    feasible: bool
    objective_value: float | None
    moments: np.ndarray | None
    margin: float | None
    eq_residual: float | None
    psd_residual: float | None
    solver_name: str | None


@dataclass
class Figure5MomentRepresentation:
    raw_moments: list[Any] | Any
    decision_variable: Any
    uses_smallg_parametrization: bool
    anchors: np.ndarray | None
    scales: np.ndarray | None

    def set_initial_guess(self, initial_moments: np.ndarray | None) -> None:
        if initial_moments is None:
            return
        if not self.uses_smallg_parametrization:
            if initial_moments.shape == self.decision_variable.shape:
                self.decision_variable.value = initial_moments.copy()
            return
        cutoff = len(self.anchors) - 1 if self.anchors is not None else 0
        if initial_moments.shape != (cutoff + 1,):
            return
        delta = np.zeros(cutoff, dtype=float)
        for order in range(1, cutoff + 1):
            scale = float(self.scales[order])
            if abs(scale) < 1e-15:
                continue
            delta[order - 1] = (float(initial_moments[order]) - float(self.anchors[order])) / scale
        self.decision_variable.value = delta

    def extract_raw_values(self) -> np.ndarray | None:
        if not self.uses_smallg_parametrization:
            if self.decision_variable.value is None:
                return None
            return np.asarray(self.decision_variable.value, dtype=float).reshape(-1)
        if self.decision_variable.value is None or self.anchors is None or self.scales is None:
            return None
        delta = np.asarray(self.decision_variable.value, dtype=float).reshape(-1)
        values = np.zeros(len(self.anchors), dtype=float)
        values[0] = 1.0
        for order in range(1, len(self.anchors)):
            values[order] = float(self.anchors[order] + self.scales[order] * delta[order - 1])
        return values


def _build_moment_representation(cp, *, cutoff: int, g: float, use_smallg_parametrization: bool) -> Figure5MomentRepresentation:
    if use_smallg_parametrization and g > 0.0:
        anchors = np.array([figure5_smallg_anchor(order, g) for order in range(cutoff + 1)], dtype=float)
        scales = np.array([1.0] + [figure5_smallg_scale(order, g) for order in range(1, cutoff + 1)], dtype=float)
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
    moments = cp.Variable(cutoff + 1)
    return Figure5MomentRepresentation(
        raw_moments=moments,
        decision_variable=moments,
        uses_smallg_parametrization=False,
        anchors=None,
        scales=None,
    )


class Figure5CubicReducer:
    def __init__(self, *, g: float, energy: float) -> None:
        if g < 0:
            raise ValueError("g must be nonnegative")
        self.g = float(g)
        self.energy = float(energy)

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
        prefactor = p_power - 2
        _expr_add_scaled(expr, dict(self.p_expr(p_power - 2, x_power)), 2.0 * self.energy + 1.0)
        _expr_add_scaled(expr, dict(self.p_expr(p_power - 2, x_power + 1)), 2.0 * self.g)
        _expr_add_scaled(expr, dict(self.p_expr(p_power - 2, x_power + 2)), -1.0)
        _expr_add_scaled(expr, dict(self.p_expr(p_power - 2, x_power + 3)), -2.0 * self.g)
        _expr_add_scaled(expr, dict(self.p_expr(p_power - 2, x_power + 4)), -(self.g**2))

        if prefactor >= 1:
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 3, x_power)), 2.0j * self.g * prefactor)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 3, x_power + 1)), -2.0j * prefactor)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 3, x_power + 2)), -6.0j * self.g * prefactor)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 3, x_power + 3)), -4.0j * (self.g**2) * prefactor)

        if prefactor >= 2:
            prefactor2 = prefactor * (prefactor - 1)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 4, x_power)), prefactor2)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 4, x_power + 1)), 6.0 * self.g * prefactor2)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 4, x_power + 2)), 6.0 * (self.g**2) * prefactor2)

        if prefactor >= 3:
            prefactor3 = prefactor * (prefactor - 1) * (prefactor - 2)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 5, x_power)), 2.0j * self.g * prefactor3)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 5, x_power + 1)), 4.0j * (self.g**2) * prefactor3)

        if prefactor >= 4:
            prefactor4 = prefactor * (prefactor - 1) * (prefactor - 2) * (prefactor - 3)
            _expr_add_scaled(expr, dict(self.p_expr(p_power - 6, x_power)), -(self.g**2) * prefactor4)
        return tuple(sorted(expr.items()))

    def matrix_entry_expr(self, left: tuple[int, int], right: tuple[int, int]) -> MomentExpr:
        p_left, x_left = left
        p_right, x_right = right
        expr: MomentExpr = {}
        total_p = p_left + p_right
        for r in range(min(x_left, total_p) + 1):
            coefficient = (1.0j**r) * factorial(r) * comb(x_left, r) * comb(total_p, r)
            _expr_add_scaled(expr, dict(self.p_expr(total_p - r, x_left + x_right - r)), coefficient)
        return expr

    def pure_moment_relation(self, order: int) -> MomentExpr:
        expr: MomentExpr = {}
        terms = [
            (order - 6, (order - 3) * (order - 4) * (order - 5)),
            (order - 4, 4.0 * (order - 3) * (2.0 * self.energy + 1.0)),
            (order - 3, 4.0 * self.g * (2 * order - 5)),
            (order - 2, -4.0 * (order - 2)),
            (order - 1, -4.0 * self.g * (2 * order - 3)),
            (order, -4.0 * (self.g**2) * (order - 1)),
        ]
        for index, coefficient in terms:
            if index < 0 or coefficient == 0:
                continue
            expr[index] = expr.get(index, 0.0 + 0.0j) + coefficient
        return expr

    @staticmethod
    def _shifted_power_coeffs(power: int, shift: float) -> dict[int, float]:
        coeffs: dict[int, float] = {}
        for degree in range(power + 1):
            coeffs[degree] = comb(power, degree) * ((-shift) ** (power - degree))
        return coeffs

    def centered_matrix_entry_expr(self, left: tuple[int, int], right: tuple[int, int], *, mu: float, sigma: float) -> MomentExpr:
        p_left, y_left = left
        p_right, y_right = right
        expr: MomentExpr = {}
        scale_prefactor = sigma ** (p_left + p_right - y_left - y_right)
        left_coeffs = self._shifted_power_coeffs(y_left, mu)
        right_coeffs = self._shifted_power_coeffs(y_right, mu)
        for x_left, coeff_left in left_coeffs.items():
            for x_right, coeff_right in right_coeffs.items():
                _expr_add_scaled(
                    expr,
                    self.matrix_entry_expr((p_left, x_left), (p_right, x_right)),
                    scale_prefactor * coeff_left * coeff_right,
                )
        return expr


def figure5_basis_size(level: int) -> int:
    return len(figure5_canonical_basis(level))


def figure5_ambient_level(level: int) -> int:
    return (level + 1) // 2


def figure5_ambient_basis(level: int) -> tuple[tuple[int, int], ...]:
    return figure5_canonical_basis(figure5_ambient_level(level))


def figure5_ambient_basis_size(level: int) -> int:
    return len(figure5_ambient_basis(level))


def _figure5_matrix_values(matrix_exprs: list[list[MomentExpr]], moment_values: np.ndarray) -> np.ndarray:
    size = len(matrix_exprs)
    matrix = np.zeros((size, size), dtype=complex)
    for row in range(size):
        for column in range(size):
            matrix[row, column] = _evaluate_expr(matrix_exprs[row][column], moment_values)
    return 0.5 * (matrix + matrix.conj().T)


def _figure5_matrix_exprs(
    reducer: Figure5CubicReducer,
    basis: tuple[tuple[int, int], ...],
    *,
    g: float,
    use_centered_basis: bool,
) -> list[list[MomentExpr]]:
    if use_centered_basis and g > 0.0:
        mu = figure5_smallg_center_mu(g)
        sigma = figure5_smallg_center_sigma(g)
        return [[reducer.centered_matrix_entry_expr(left, right, mu=mu, sigma=sigma) for right in basis] for left in basis]
    return [[reducer.matrix_entry_expr(left, right) for right in basis] for left in basis]


def _figure5_status_label(result: Figure5SolveResult) -> str:
    if result.solver_name is None:
        return result.status
    suffix = "feas" if result.feasible else "infeas"
    return f"{result.solver_name.lower()}:{result.status}:{suffix}"


def _figure5_smallg_energy_window(g: float, config: Figure5Config, rr_upper: float) -> tuple[float, float]:
    scale = figure5_instanton_energy_scale(g)
    if scale <= 0.0:
        return config.e_min, min(config.e_max, rr_upper)
    eta_low = config.smallg_eta_min
    eta_high = min(config.smallg_eta_max, figure5_eta_from_energy(g, max(scale * exp(config.smallg_eta_min), config.e_max)))
    rr_eta = min(eta_high, figure5_eta_from_energy(g, max(rr_upper, scale * exp(config.smallg_eta_min))))
    return eta_low, max(rr_eta, eta_low)


def figure5_feasibility(
    *,
    g: float,
    level: int,
    energy: float,
    small_g_switch: float = 0.4,
    use_smallg_centered_basis: bool = True,
    use_smallg_parity_variables: bool = True,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    margin_tolerance: float = 5e-5,
    eq_tolerance: float = 5e-6,
    psd_tolerance: float = 5e-6,
    initial_moments: np.ndarray | None = None,
    use_completed_basis: bool = True,
) -> Figure5SolveResult:
    cp = _import_cvxpy()
    reducer = Figure5CubicReducer(g=g, energy=energy)
    cutoff = 2 * level
    moment_repr = _build_moment_representation(
        cp,
        cutoff=cutoff,
        g=g,
        use_smallg_parametrization=use_smallg_parity_variables and g < small_g_switch,
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

    basis = figure5_ambient_basis(level) if use_completed_basis else figure5_canonical_basis(level)
    matrix_exprs = _figure5_matrix_exprs(
        reducer,
        basis,
        g=g,
        use_centered_basis=use_smallg_centered_basis and g < small_g_switch,
    )
    matrix = cp.Variable((len(basis), len(basis)), hermitian=True)
    margin = cp.Variable()
    constraints.append(margin <= 1.0)
    constraints.append(margin >= -1.0)
    for row, left in enumerate(basis):
        for column in range(row, len(basis)):
            right = basis[column]
            if (2 * left[0] + left[1]) + (2 * right[0] + right[1]) <= level or not use_completed_basis:
                constraints.append(matrix[row, column] == _expr_complex(matrix_exprs[row][column], moments))
    constraints.append(matrix - margin * np.eye(len(basis)) >> 0)
    problem = cp.Problem(cp.Maximize(margin), constraints)

    best = Figure5SolveResult(
        status="solver_not_run",
        feasible=False,
        objective_value=None,
        moments=None,
        margin=None,
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
            best = Figure5SolveResult(
                status=f"{solver_name.lower()}_failed",
                feasible=False,
                objective_value=None,
                moments=None,
                margin=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue

        status = str(problem.status)
        moment_values = moment_repr.extract_raw_values()
        if status not in {"optimal", "optimal_inaccurate"} or moment_values is None or margin.value is None or matrix.value is None:
            best = Figure5SolveResult(
                status=status,
                feasible=False,
                objective_value=None,
                moments=None,
                margin=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue

        norm_m = max(1.0, float(np.max(np.abs(moment_values))))
        eq_residual = 0.0
        for expr in pure_relations:
            eq_residual = max(eq_residual, abs(_evaluate_expr(expr, moment_values)))
        eq_residual = float(eq_residual / norm_m)

        matrix_values = np.asarray(matrix.value, dtype=complex)
        matrix_norm = max(1.0, float(np.linalg.norm(matrix_values, ord=2)))
        psd_residual = float(max(0.0, -float(np.linalg.eigvalsh(matrix_values).min())) / matrix_norm)
        feasible = (
            float(margin.value) >= -margin_tolerance
            and eq_residual <= eq_tolerance
            and psd_residual <= psd_tolerance
            and abs(moment_values[0] - 1.0) <= 5e-7
        )
        result = Figure5SolveResult(
            status=status,
            feasible=feasible,
            objective_value=float(margin.value),
            moments=moment_values if feasible else None,
            margin=float(margin.value),
            eq_residual=eq_residual,
            psd_residual=psd_residual,
            solver_name=solver_name,
        )
        if feasible:
            return result
        best = result
    return best


def figure5_minimize_x2(
    *,
    g: float,
    level: int,
    energy: float,
    small_g_switch: float = 0.4,
    use_smallg_centered_basis: bool = True,
    use_smallg_parity_variables: bool = True,
    use_smallg_nu_objective: bool = True,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 50000,
    margin_tolerance: float = 5e-5,
    eq_tolerance: float = 5e-6,
    psd_tolerance: float = 5e-6,
    initial_moments: np.ndarray | None = None,
    use_completed_basis: bool = True,
) -> Figure5SolveResult:
    cp = _import_cvxpy()
    reducer = Figure5CubicReducer(g=g, energy=energy)
    cutoff = 2 * level
    moment_repr = _build_moment_representation(
        cp,
        cutoff=cutoff,
        g=g,
        use_smallg_parametrization=use_smallg_parity_variables and g < small_g_switch,
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

    basis = figure5_ambient_basis(level) if use_completed_basis else figure5_canonical_basis(level)
    matrix_exprs = _figure5_matrix_exprs(
        reducer,
        basis,
        g=g,
        use_centered_basis=use_smallg_centered_basis and g < small_g_switch,
    )
    matrix = cp.Variable((len(basis), len(basis)), hermitian=True)
    for row, left in enumerate(basis):
        for column in range(row, len(basis)):
            right = basis[column]
            if (2 * left[0] + left[1]) + (2 * right[0] + right[1]) <= level or not use_completed_basis:
                constraints.append(matrix[row, column] == _expr_complex(matrix_exprs[row][column], moments))
    constraints.append(matrix >> 0)
    objective_expr = moments[2]
    if use_smallg_nu_objective and g < small_g_switch:
        objective_expr = (moments[2] - 0.5) / max(g * g, 1e-6)
    problem = cp.Problem(cp.Minimize(objective_expr), constraints)

    best = Figure5SolveResult(
        status="solver_not_run",
        feasible=False,
        objective_value=None,
        moments=None,
        margin=None,
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
            best = Figure5SolveResult(
                status=f"{solver_name.lower()}_failed",
                feasible=False,
                objective_value=None,
                moments=None,
                margin=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue

        status = str(problem.status)
        moment_values = moment_repr.extract_raw_values()
        if status not in {"optimal", "optimal_inaccurate"} or moment_values is None or matrix.value is None:
            best = Figure5SolveResult(
                status=status,
                feasible=False,
                objective_value=None,
                moments=None,
                margin=None,
                eq_residual=None,
                psd_residual=None,
                solver_name=solver_name,
            )
            continue

        norm_m = max(1.0, float(np.max(np.abs(moment_values))))
        eq_residual = 0.0
        for expr in pure_relations:
            eq_residual = max(eq_residual, abs(_evaluate_expr(expr, moment_values)))
        eq_residual = float(eq_residual / norm_m)
        matrix_values = np.asarray(matrix.value, dtype=complex)
        matrix_norm = max(1.0, float(np.linalg.norm(matrix_values, ord=2)))
        psd_residual = float(max(0.0, -float(np.linalg.eigvalsh(matrix_values).min())) / matrix_norm)
        feasible = (
            eq_residual <= eq_tolerance
            and psd_residual <= psd_tolerance
            and abs(moment_values[0] - 1.0) <= 5e-7
        )
        result = Figure5SolveResult(
            status=status,
            feasible=feasible,
            objective_value=float(moment_values[2]) if feasible else None,
            moments=moment_values if feasible else None,
            margin=None,
            eq_residual=eq_residual,
            psd_residual=psd_residual,
            solver_name=solver_name,
        )
        if feasible:
            return result
        best = result
    return best


@dataclass
class Figure5EnergyPoint:
    g: float
    energy: float | None
    status: str
    rr_upper: float
    moments: np.ndarray | None


@dataclass
class Figure5X2Point:
    g: float
    x2: float | None
    status: str
    energy_argmin: float | None
    moments: np.ndarray | None


def _find_energy_lower_bound(
    *,
    g: float,
    level: int,
    config: Figure5Config,
    initial_moments: np.ndarray | None,
    energy_floor: float | None = None,
) -> Figure5EnergyPoint:
    rr_upper = min(config.e_max, float(figure4_rr_full_energy(g)))
    use_eta_mode = config.use_smallg_eta_scan and g < config.small_g_switch and g > 0.0
    probes = np.linspace(rr_upper, config.e_max, config.upper_probe_points)
    high_parameter = None
    high_result: Figure5SolveResult | None = None
    guess = initial_moments
    if use_eta_mode:
        eta_low, eta_rr = _figure5_smallg_energy_window(g, config, rr_upper)
        eta_cap = min(config.smallg_eta_max, figure5_eta_from_energy(g, max(config.e_max, figure5_energy_from_eta(g, config.smallg_eta_min))))
        coarse = np.linspace(eta_rr, eta_cap, config.upper_probe_points)
        local_center = np.arange(np.ceil(max(eta_rr, -2.0)), np.floor(min(eta_cap, 2.0)) + 1.0)
        probe_parameters = np.unique(np.concatenate((coarse, local_center, np.array([0.0]))))
        probe_parameters = probe_parameters[(probe_parameters >= eta_rr) & (probe_parameters <= eta_cap)]
    else:
        probe_parameters = np.unique(probes)
    for parameter in probe_parameters:
        energy = figure5_energy_from_eta(g, float(parameter)) if use_eta_mode else float(parameter)
        result = figure5_feasibility(
            g=g,
            level=level,
            energy=float(energy),
            small_g_switch=config.small_g_switch,
            use_smallg_centered_basis=config.use_smallg_centered_basis,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=guess,
            use_completed_basis=config.use_completed_basis,
        )
        if result.feasible:
            high_parameter = float(parameter)
            high_result = result
            guess = result.moments
            break
    if high_parameter is None or high_result is None:
        return Figure5EnergyPoint(g=g, energy=None, status="upper:not_found", rr_upper=rr_upper, moments=None)

    if use_eta_mode:
        low_parameter = config.smallg_eta_min
        if energy_floor is not None and energy_floor > 0.0:
            low_parameter = max(low_parameter, figure5_eta_from_energy(g, energy_floor))
        low_energy_ref = figure5_energy_from_eta(g, low_parameter)
    else:
        low_energy_ref = config.e_min if energy_floor is None else max(config.e_min, energy_floor)
        low_parameter = low_energy_ref

    high_energy = figure5_energy_from_eta(g, high_parameter) if use_eta_mode else float(high_parameter)
    low_energy = low_energy_ref
    if low_energy >= high_energy:
        return Figure5EnergyPoint(
            g=g,
            energy=high_energy,
            status=_figure5_status_label(high_result),
            rr_upper=rr_upper,
            moments=high_result.moments,
        )
    low_result = figure5_feasibility(
        g=g,
        level=level,
        energy=low_energy,
        small_g_switch=config.small_g_switch,
        use_smallg_centered_basis=config.use_smallg_centered_basis,
        use_smallg_parity_variables=config.use_smallg_parity_variables,
        solver=config.solver,
        solver_eps=config.solver_eps,
        solver_max_iters=config.solver_max_iters,
        margin_tolerance=config.margin_tolerance,
        eq_tolerance=config.eq_tolerance,
        psd_tolerance=config.psd_tolerance,
        initial_moments=guess,
        use_completed_basis=config.use_completed_basis,
    )
    if low_result.feasible:
        return Figure5EnergyPoint(
            g=g,
            energy=low_energy,
            status=_figure5_status_label(low_result),
            rr_upper=rr_upper,
            moments=low_result.moments,
        )

    best_parameter = high_parameter
    best_result = high_result
    lower_infeasible = low_parameter
    for _ in range(config.bisection_steps):
        middle = 0.5 * (lower_infeasible + best_parameter)
        energy = figure5_energy_from_eta(g, middle) if use_eta_mode else middle
        result = figure5_feasibility(
            g=g,
            level=level,
            energy=energy,
            small_g_switch=config.small_g_switch,
            use_smallg_centered_basis=config.use_smallg_centered_basis,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=best_result.moments,
            use_completed_basis=config.use_completed_basis,
        )
        if result.feasible:
            best_parameter = middle
            best_result = result
        else:
            lower_infeasible = middle
    best_energy = figure5_energy_from_eta(g, best_parameter) if use_eta_mode else best_parameter
    return Figure5EnergyPoint(
        g=g,
        energy=best_energy,
        status=_figure5_status_label(best_result),
        rr_upper=rr_upper,
        moments=best_result.moments,
    )


def _find_x2_lower_bound(
    *,
    g: float,
    level: int,
    energy_point: Figure5EnergyPoint,
    config: Figure5Config,
) -> Figure5X2Point:
    if energy_point.energy is None:
        return Figure5X2Point(g=g, x2=None, status="energy:not_found", energy_argmin=None, moments=None)
    e_low = energy_point.energy
    e_high = max(e_low + config.x2_edge_slack, min(config.e_max, energy_point.rr_upper))
    if config.x2_lock_to_edge:
        edge_energy = min(e_high, e_low + config.x2_edge_slack)
        fallback_scan = np.linspace(e_low, e_high, max(2, config.x2_energy_scan_points))
        scan = np.unique(np.concatenate(([edge_energy], fallback_scan)))
    else:
        scan = np.unique(
            np.concatenate(([min(e_high, e_low + config.x2_edge_slack)], np.linspace(e_low, e_high, config.x2_energy_scan_points)))
        )
    best_result: Figure5SolveResult | None = None
    best_energy = None
    best_x2 = None
    guess = energy_point.moments
    for energy in scan:
        result = figure5_minimize_x2(
            g=g,
            level=level,
            energy=float(energy),
            small_g_switch=config.small_g_switch,
            use_smallg_centered_basis=config.use_smallg_centered_basis,
            use_smallg_parity_variables=config.use_smallg_parity_variables,
            use_smallg_nu_objective=config.use_smallg_nu_objective,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            margin_tolerance=config.margin_tolerance,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
            initial_moments=guess,
            use_completed_basis=config.use_completed_basis,
        )
        if result.feasible:
            guess = result.moments
            if best_x2 is None or (result.objective_value is not None and result.objective_value < best_x2):
                best_x2 = float(result.objective_value)
                best_energy = float(energy)
                best_result = result
    if best_result is None:
        return Figure5X2Point(g=g, x2=None, status="x2:not_found", energy_argmin=None, moments=None)
    return Figure5X2Point(
        g=g,
        x2=best_x2,
        status=_figure5_status_label(best_result),
        energy_argmin=best_energy,
        moments=best_result.moments,
    )


def scan_figure5_energy(config: Figure5Config) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, list[str]], dict[int, np.ndarray]]:
    g_values_desc = config.g_grid()
    energy_bounds: dict[int, np.ndarray] = {}
    statuses: dict[int, list[str]] = {}
    rr_upper: dict[int, np.ndarray] = {}
    for level in config.levels:
        level_energies = np.full(g_values_desc.size, np.nan, dtype=float)
        level_statuses: list[str] = []
        level_rr_upper = np.full(g_values_desc.size, np.nan, dtype=float)
        previous_moments: np.ndarray | None = None
        for index, g in enumerate(g_values_desc):
            energy_floor = None
            lower_level_index = config.levels.index(level) - 1
            if lower_level_index >= 0:
                lower_level = config.levels[lower_level_index]
                lower_energy_desc = energy_bounds[lower_level][::-1][index]
                if np.isfinite(lower_energy_desc):
                    energy_floor = max(config.e_min, float(lower_energy_desc) + config.energy_floor_slack)
            point = _find_energy_lower_bound(
                g=float(g),
                level=level,
                config=config,
                initial_moments=previous_moments,
                energy_floor=energy_floor,
            )
            level_statuses.append(point.status)
            level_rr_upper[index] = point.rr_upper
            if point.energy is not None:
                level_energies[index] = point.energy
                previous_moments = point.moments
        energy_bounds[level] = level_energies[::-1]
        statuses[level] = level_statuses[::-1]
        rr_upper[level] = level_rr_upper[::-1]
    return g_values_desc[::-1], energy_bounds, statuses, rr_upper


def scan_figure5_x2(
    *,
    config: Figure5Config,
    g_values: np.ndarray,
    energy_bounds: dict[int, np.ndarray],
    energy_statuses: dict[int, list[str]],
    rr_upper: dict[int, np.ndarray],
) -> tuple[dict[int, np.ndarray], dict[int, list[str]], dict[int, np.ndarray]]:
    del energy_statuses
    x2_bounds: dict[int, np.ndarray] = {}
    x2_statuses: dict[int, list[str]] = {}
    argmin_energies: dict[int, np.ndarray] = {}
    for level in config.levels:
        level_x2 = np.full(g_values.size, np.nan, dtype=float)
        level_statuses: list[str] = []
        level_argmin = np.full(g_values.size, np.nan, dtype=float)
        for index, g in enumerate(g_values):
            energy_point = Figure5EnergyPoint(
                g=float(g),
                energy=float(energy_bounds[level][index]) if np.isfinite(energy_bounds[level][index]) else None,
                status="",
                rr_upper=float(rr_upper[level][index]),
                moments=None,
            )
            point = _find_x2_lower_bound(g=float(g), level=level, energy_point=energy_point, config=config)
            level_statuses.append(point.status)
            if point.x2 is not None:
                level_x2[index] = point.x2
            if point.energy_argmin is not None:
                level_argmin[index] = point.energy_argmin
        x2_bounds[level] = level_x2
        x2_statuses[level] = level_statuses
        argmin_energies[level] = level_argmin
    return x2_bounds, x2_statuses, argmin_energies


def plot_figure5(
    g_values: np.ndarray,
    energy_bounds: dict[int, np.ndarray],
    x2_bounds: dict[int, np.ndarray],
    *,
    out_path: str | Path,
) -> None:
    plt = __import__("matplotlib")
    plt.use("Agg")
    import matplotlib.pyplot as pyplot

    colors = {
        7: "#e31a1c",
        8: "#6a3d9a",
        9: "#1f78b4",
        10: "#33a02c",
    }
    figure, axes = pyplot.subplots(1, 2, figsize=(12.0, 5.4))
    for level in sorted(energy_bounds):
        color = colors.get(level, None)
        mask = np.isfinite(energy_bounds[level])
        axes[0].plot(g_values[mask], energy_bounds[level][mask], color=color, linewidth=2.0, label=f"level {level}")
    axes[0].set_xlim(0.0, 1.0)
    axes[0].set_ylim(0.0, 0.3)
    axes[0].set_xlabel(r"$g$")
    axes[0].set_ylabel(r"$E$")
    axes[0].set_title("Figure 5 energy lower bounds")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    for level in sorted(x2_bounds):
        color = colors.get(level, None)
        mask = np.isfinite(x2_bounds[level])
        axes[1].plot(g_values[mask], x2_bounds[level][mask], color=color, linewidth=2.0, label=f"level {level}")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].set_ylim(0.0, 1.2)
    axes[1].set_xlabel(r"$g$")
    axes[1].set_ylabel(r"$\langle x^2\rangle$")
    axes[1].set_title("Figure 5 x^2 lower bounds")
    axes[1].grid(True, alpha=0.25)
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)


def run_figure5_scan(
    *,
    out_dir: str | Path,
    config: Figure5Config | None = None,
) -> dict[str, Any]:
    resolved_config = Figure5Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(resolved_config.to_json(), indent=2), encoding="utf-8")

    g_values, energy_bounds, energy_statuses, rr_upper = scan_figure5_energy(resolved_config)
    x2_bounds, x2_statuses, x2_argmins = scan_figure5_x2(
        config=resolved_config,
        g_values=g_values,
        energy_bounds=energy_bounds,
        energy_statuses=energy_statuses,
        rr_upper=rr_upper,
    )

    fieldnames = ["g"]
    for level in sorted(resolved_config.levels):
        fieldnames.extend(
            [
                f"level_{level}_energy",
                f"level_{level}_energy_status",
                f"level_{level}_rr_upper",
                f"level_{level}_x2",
                f"level_{level}_x2_status",
                f"level_{level}_x2_energy_argmin",
            ]
        )
    with (output_dir / "bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, g in enumerate(g_values):
            row: dict[str, Any] = {"g": float(g)}
            for level in sorted(resolved_config.levels):
                row[f"level_{level}_energy"] = float(energy_bounds[level][index]) if np.isfinite(energy_bounds[level][index]) else np.nan
                row[f"level_{level}_energy_status"] = energy_statuses[level][index]
                row[f"level_{level}_rr_upper"] = float(rr_upper[level][index]) if np.isfinite(rr_upper[level][index]) else np.nan
                row[f"level_{level}_x2"] = float(x2_bounds[level][index]) if np.isfinite(x2_bounds[level][index]) else np.nan
                row[f"level_{level}_x2_status"] = x2_statuses[level][index]
                row[f"level_{level}_x2_energy_argmin"] = float(x2_argmins[level][index]) if np.isfinite(x2_argmins[level][index]) else np.nan
            writer.writerow(row)

    plot_figure5(g_values, energy_bounds, x2_bounds, out_path=output_dir / "figure5_bounds.png")
    plot_figure5(g_values, energy_bounds, {level: rr_upper[level] for level in resolved_config.levels}, out_path=output_dir / "figure5_energy_and_rr_guides.png")

    summary_lines = [
        "# Figure 5 cubic SUSY QM lower bounds",
        "",
        "- Model: `W(x) = x^2 / 2 + g x^3 / 3`",
        "- Sector fixed to `epsilon = -1`",
        "- Variable regime: original `(x,p)` reducer, small/intermediate `g`",
        f"- Small-g switch: `g < {resolved_config.small_g_switch}`",
        f"- Small-g eta search: `{resolved_config.use_smallg_eta_scan}`",
        f"- Small-g centered basis (experimental): `{resolved_config.use_smallg_centered_basis}`",
        f"- Small-g parity moment variables (experimental): `{resolved_config.use_smallg_parity_variables}`",
        "- Basis mode:",
        f"  - `use_completed_basis = {resolved_config.use_completed_basis}`",
        "  - full theoretical basis remains `p^a x^b` with `2a + b <= L`",
        "  - active numerical run uses the smaller ambient basis with Hermitian completion",
        "- Left panel: fixed-energy feasibility + bisection for `E` lower bounds",
        "- Small-g left panel uses instanton-rescaled `eta` probes instead of linear energy probes",
        "- Right panel: `m_2` minimization is evaluated only in the narrow ground-state edge window",
        "",
        "Basis sizes:",
    ]
    summary_lines.extend(
        [
            f"- level {level}: full={figure5_basis_size(level)}, ambient={figure5_ambient_basis_size(level)}"
            for level in resolved_config.levels
        ]
    )
    summary_lines.append("")
    summary_lines.append("Status counts:")
    for level in resolved_config.levels:
        energy_success = int(np.isfinite(energy_bounds[level]).sum())
        x2_success = int(np.isfinite(x2_bounds[level]).sum())
        summary_lines.append(f"- level {level}: energy={energy_success}/{len(g_values)}, x2={x2_success}/{len(g_values)}")
    if resolved_config.g_min > 0.0:
        summary_lines.extend(
            [
                "",
                "Difficulty note:",
                f"- current run intentionally starts at `g = {resolved_config.g_min}`",
                "- below this range the ordinary solver becomes much less stable, matching the paper's small-g breakdown warning",
            ]
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "g_values": g_values,
        "energy_bounds": energy_bounds,
        "x2_bounds": x2_bounds,
        "energy_statuses": energy_statuses,
        "x2_statuses": x2_statuses,
        "rr_upper": rr_upper,
        "x2_argmins": x2_argmins,
    }
