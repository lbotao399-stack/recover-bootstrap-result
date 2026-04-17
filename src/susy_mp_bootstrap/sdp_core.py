from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .models_1d import PolynomialSuperpotential
from .moments_1d import build_recursion_constraints


@dataclass(frozen=True)
class LinearMomentConstraint:
    coefficients: dict[int, float]
    rhs: float = 0.0
    label: str = "constraint"

    def dense_row(self, moment_cutoff: int) -> np.ndarray:
        row = np.zeros(moment_cutoff + 1, dtype=float)
        for index, coefficient in self.coefficients.items():
            row[index] = float(coefficient)
        return row

    def to_json(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "rhs": self.rhs,
            "coefficients": {str(index): coefficient for index, coefficient in sorted(self.coefficients.items())},
        }


@dataclass(frozen=True)
class FixedEnergySDP:
    superpotential_label: str
    epsilon: int
    energy: float
    matrix_size: int
    moment_cutoff: int
    constraints: tuple[LinearMomentConstraint, ...]

    def moment_matrix_indices(self) -> list[list[int]]:
        return [[i + j for j in range(self.matrix_size)] for i in range(self.matrix_size)]

    def to_json(self) -> dict[str, Any]:
        return {
            "superpotential_label": self.superpotential_label,
            "epsilon": self.epsilon,
            "energy": self.energy,
            "matrix_size": self.matrix_size,
            "moment_cutoff": self.moment_cutoff,
            "moment_matrix_indices": self.moment_matrix_indices(),
            "constraints": [constraint.to_json() for constraint in self.constraints],
        }


@dataclass(frozen=True)
class FixedEnergySDPResult:
    status: str
    feasible: bool
    moments: np.ndarray | None
    solver: str


def _import_cvxpy():
    try:
        import cvxpy as cp  # type: ignore
    except Exception as exc:  # pragma: no cover - runtime dependent
        raise RuntimeError("cvxpy is required for the fixed-energy SDP solver") from exc
    return cp


def moment_matrix(values: np.ndarray, matrix_size: int) -> np.ndarray:
    return np.array([[values[i + j] for j in range(matrix_size)] for i in range(matrix_size)], dtype=float)


def build_fixed_energy_problem(
    superpotential: PolynomialSuperpotential,
    *,
    epsilon: int,
    energy: float,
    matrix_size: int,
    moment_cutoff: int | None = None,
    t_values: Iterable[int] | None = None,
) -> FixedEnergySDP:
    if matrix_size <= 0:
        raise ValueError("matrix_size must be positive")

    default_cutoff = 2 * (matrix_size - 1) + 2 * max(1, superpotential.degree)
    cutoff = default_cutoff if moment_cutoff is None else moment_cutoff
    constraints = [
        LinearMomentConstraint({0: 1.0}, rhs=1.0, label="normalization"),
    ]
    for index, coefficients in enumerate(
        build_recursion_constraints(
            superpotential,
            epsilon=epsilon,
            energy=energy,
            moment_cutoff=cutoff,
            t_values=t_values,
        )
    ):
        dense_coeffs = {moment: float(np.real(complex(value.evalf()))) for moment, value in coefficients.items()}
        constraints.append(LinearMomentConstraint(dense_coeffs, rhs=0.0, label=f"recursion_{index}"))
    return FixedEnergySDP(
        superpotential_label=superpotential.label,
        epsilon=epsilon,
        energy=float(energy),
        matrix_size=matrix_size,
        moment_cutoff=cutoff,
        constraints=tuple(constraints),
    )


def solve_fixed_energy_sdp(
    problem: FixedEnergySDP,
    *,
    solver: str = "SCS",
    solver_kwargs: dict[str, Any] | None = None,
) -> FixedEnergySDPResult:
    cp = _import_cvxpy()
    kwargs = {} if solver_kwargs is None else dict(solver_kwargs)
    kwargs.setdefault("warm_start", True)

    moments = cp.Variable(problem.moment_cutoff + 1)
    constraints = [moments[0] == 1.0, moment_matrix_cp(cp, moments, problem.matrix_size) >> 0]
    for constraint in problem.constraints[1:]:
        expr = 0
        for index, coefficient in constraint.coefficients.items():
            expr = expr + coefficient * moments[index]
        constraints.append(expr == constraint.rhs)

    optimization = cp.Problem(cp.Minimize(0), constraints)
    optimization.solve(solver=solver, **kwargs)

    status = str(optimization.status)
    feasible = status not in {"infeasible", "infeasible_inaccurate", "unbounded", "unbounded_inaccurate"}
    values = None
    if feasible and moments.value is not None:
        values = np.asarray(moments.value, dtype=float).reshape(-1)
    return FixedEnergySDPResult(status=status, feasible=feasible, moments=values, solver=solver)


def moment_matrix_cp(cp, values, matrix_size: int):
    rows = []
    for i in range(matrix_size):
        rows.append([values[i + j] for j in range(matrix_size)])
    matrix = cp.bmat(rows)
    return 0.5 * (matrix + matrix.T)


def export_sdpb_json(problem: FixedEnergySDP, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(problem.to_json(), indent=2), encoding="utf-8")

