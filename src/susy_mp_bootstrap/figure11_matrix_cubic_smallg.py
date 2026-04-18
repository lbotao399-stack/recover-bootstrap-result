from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any
import csv
import json
import math
import shutil

import numpy as np

from .figure10_matrix_cubic_largeg import (
    Figure10Reducer,
    build_figure10_problem,
    solve_figure10_point,
)


def figure11_critical_gc() -> float:
    return math.sqrt(1.0 / (6.0 * math.sqrt(3.0)))


def figure11_kink_g0() -> float:
    return math.sqrt(2.0) * figure11_critical_gc()


def figure11_default_g_grid() -> tuple[float, ...]:
    g0 = figure11_kink_g0()
    coarse = {
        0.20,
        0.24,
        0.28,
        0.32,
        0.36,
        0.40,
        0.42,
        0.46,
        0.50,
        0.55,
        0.60,
        0.70,
        0.80,
    }
    dense_offsets = {
        -0.020,
        -0.015,
        -0.010,
        -0.0075,
        -0.005,
        -0.0025,
        0.0,
        0.0025,
        0.005,
        0.0075,
        0.010,
        0.015,
        0.020,
    }
    dense = {g0 + offset for offset in dense_offsets}
    grid = sorted(value for value in coarse.union(dense) if value > 0.0)
    return tuple(grid)


@dataclass(frozen=True)
class Figure11Config:
    n: int = 100
    basis_level: Fraction = Fraction(4, 1)
    observable_level: Fraction = Fraction(8, 1)
    universe_source_level: Fraction = Fraction(8, 1)
    gauge_seed_level: Fraction = Fraction(5, 1)
    eom_seed_level: Fraction = Fraction(7, 1)
    reality_seed_level: Fraction = Fraction(5, 1)
    include_ground_block: bool = False
    solver: str = "SCS"
    solver_eps: float = 1e-5
    solver_max_iters: int = 50000
    eq_tolerance: float = 5e-4
    psd_tolerance: float = 5e-4
    g_values: tuple[float, ...] = figure11_default_g_grid()
    clip_negative_to_zero: bool = True

    def to_json(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "basis_level": float(self.basis_level),
            "observable_level": float(self.observable_level),
            "universe_source_level": float(self.universe_source_level),
            "gauge_seed_level": float(self.gauge_seed_level),
            "eom_seed_level": float(self.eom_seed_level),
            "reality_seed_level": float(self.reality_seed_level),
            "include_ground_block": self.include_ground_block,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "eq_tolerance": self.eq_tolerance,
            "psd_tolerance": self.psd_tolerance,
            "g_values": list(self.g_values),
            "clip_negative_to_zero": self.clip_negative_to_zero,
        }


@dataclass
class Figure11Point:
    g: float
    alpha: float
    role: str
    status: str
    feasible: bool
    energy_density_raw: float | None
    energy_raw: float | None
    energy_density_plot: float | None
    energy_plot: float | None
    eq_residual: float | None
    psd_residual: float | None
    clipped: bool


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _clip_energy(energy: float | None, *, clip_negative_to_zero: bool) -> tuple[float | None, bool]:
    if energy is None:
        return None, False
    if clip_negative_to_zero and energy < 0.0:
        return 0.0, True
    return energy, False


def plot_figure11(points: list[Figure11Point], *, out_path: str | Path) -> None:
    pyplot = _import_matplotlib()
    g_black = np.array([point.g for point in points if point.role == "black" and point.energy_plot is not None], dtype=float)
    e_black = np.array([point.energy_plot for point in points if point.role == "black" and point.energy_plot is not None], dtype=float)
    g_red = np.array([point.g for point in points if point.role == "red" and point.energy_plot is not None], dtype=float)
    e_red = np.array([point.energy_plot for point in points if point.role == "red" and point.energy_plot is not None], dtype=float)

    figure, axis = pyplot.subplots(figsize=(7.4, 4.6))
    if g_black.size:
        axis.scatter(g_black, e_black, color="black", s=24, label="numerical lower bound", zorder=3)
    if g_red.size:
        axis.scatter(g_red, e_red, color="#d62728", s=44, label=r"bound at $g_0$", zorder=4)
    axis.set_xlabel(r"$g$")
    axis.set_ylabel(r"$E_{\rm lower}$")
    axis.set_title("Figure 11")
    axis.set_xlim(min(point.g for point in points) - 0.015, max(point.g for point in points) + 0.02)
    finite_plot = [point.energy_plot for point in points if point.energy_plot is not None]
    axis.set_ylim(0.0, max(1.0, 1.08 * max(finite_plot, default=1.0)))
    axis.grid(True, alpha=0.22)
    if g_black.size or g_red.size:
        axis.legend(loc="upper left")
    figure.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out_path, dpi=190)
    pyplot.close(figure)


def run_figure11_scan(*, out_dir: str | Path, config: Figure11Config | None = None) -> Path:
    if config is None:
        config = Figure11Config()

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reducer = Figure10Reducer(
        n=config.n,
        basis_level=config.basis_level,
        observable_level=config.observable_level,
        universe_source_level=config.universe_source_level,
    )
    problem = build_figure10_problem(
        reducer,
        gauge_seed_level=config.gauge_seed_level,
        eom_seed_level=config.eom_seed_level,
        reality_seed_level=config.reality_seed_level,
        include_ground_block=config.include_ground_block,
    )

    g0 = figure11_kink_g0()
    points: list[Figure11Point] = []
    for g in config.g_values:
        result = solve_figure10_point(
            problem,
            reducer=reducer,
            g=g,
            solver=config.solver,
            solver_eps=config.solver_eps,
            solver_max_iters=config.solver_max_iters,
            eq_tolerance=config.eq_tolerance,
            psd_tolerance=config.psd_tolerance,
        )
        energy_density_raw = result.energy_density if result.feasible else None
        energy_raw = None if energy_density_raw is None else energy_density_raw * (config.n**2)
        energy_plot, clipped = _clip_energy(energy_raw, clip_negative_to_zero=config.clip_negative_to_zero)
        energy_density_plot = None if energy_plot is None else energy_plot / (config.n**2)
        points.append(
            Figure11Point(
                g=g,
                alpha=result.alpha,
                role="red" if math.isclose(g, g0, rel_tol=0.0, abs_tol=1e-12) else "black",
                status=result.status,
                feasible=result.feasible,
                energy_density_raw=energy_density_raw,
                energy_raw=energy_raw,
                energy_density_plot=energy_density_plot,
                energy_plot=energy_plot,
                eq_residual=result.eq_residual,
                psd_residual=result.psd_residual,
                clipped=clipped,
            )
        )

    plot_figure11(points, out_path=output_dir / "figure11_smallg.png")

    with (output_dir / "bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "g",
                "alpha",
                "role",
                "status",
                "feasible",
                "energy_density_raw",
                "energy_raw",
                "energy_density_plot",
                "energy_plot",
                "eq_residual",
                "psd_residual",
                "clipped",
            ],
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "g": point.g,
                    "alpha": point.alpha,
                    "role": point.role,
                    "status": point.status,
                    "feasible": int(point.feasible),
                    "energy_density_raw": point.energy_density_raw if point.energy_density_raw is not None else np.nan,
                    "energy_raw": point.energy_raw if point.energy_raw is not None else np.nan,
                    "energy_density_plot": point.energy_density_plot if point.energy_density_plot is not None else np.nan,
                    "energy_plot": point.energy_plot if point.energy_plot is not None else np.nan,
                    "eq_residual": point.eq_residual if point.eq_residual is not None else np.nan,
                    "psd_residual": point.psd_residual if point.psd_residual is not None else np.nan,
                    "clipped": int(point.clipped),
                }
            )

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_json(), handle, indent=2)

    negative_points = [point for point in points if point.energy_raw is not None and point.energy_raw < 0.0]
    summary_lines = [
        "# Figure 11 summary",
        "",
        "- Model: matrix cubic SUSY QM lower bound in the small-g / critical window.",
        "- Figure 11 reuses the Figure 10 level-8 matrix cubic algebra with `N = 100` and `alpha = g / sqrt(N)`.",
        f"- Critical point: `g_c = {figure11_critical_gc():.12f}`.",
        f"- Kink marker: `g_0 = sqrt(2) g_c = {g0:.12f}`.",
        f"- Ordinary PSD block dimensions: `{reducer.block_dimensions()}`.",
        f"- Canonical multi-trace universe size: `{len(reducer.canonical_universe)}`.",
        f"- Equality constraints: `{len(problem.equality_exprs)}`.",
        f"- Ground-state block enabled: `{config.include_ground_block}`.",
        f"- SDPB available in PATH: `{shutil.which('sdpb') is not None}`.",
        f"- Local fallback solver used for this run: `{config.solver}`.",
        f"- Negative raw-energy points: `{len(negative_points)}`.",
        f"- Plot clips negative raw energies to zero: `{config.clip_negative_to_zero}`.",
        "",
        "## Numerical note",
        "",
        "- Paper Figure 11 reports SDPB-certified lower bounds.",
        "- This local run reuses the same level-8 algebra but, without an `sdpb` binary/exporter, falls back to direct CVXPY/SCS minimization.",
        "- Raw negative values are retained in `bounds.csv`; the plotted curve is clipped at zero only for visualization.",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_dir / "figure11_smallg.png"
