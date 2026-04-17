from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import math

import numpy as np


@dataclass(frozen=True)
class Figure2ExConfig:
    levels: tuple[int, ...] = (5, 6, 7, 8)
    x_min: float = -0.5
    x_max: float = 0.5
    e_min: float = 0.0
    e_max: float = 5.0
    num_x: int = 241
    num_e: int = 251
    g: float = 1.0
    u_max: float = 20.0
    tolerance: float = 1e-9
    search_iterations: int = 28

    def to_json(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "num_x": self.num_x,
            "num_e": self.num_e,
            "g": self.g,
            "u_max": self.u_max,
            "tolerance": self.tolerance,
            "search_iterations": self.search_iterations,
        }


@dataclass(frozen=True)
class Figure2EuConfig:
    levels: tuple[int, ...] = (5, 6, 7, 8)
    u_min: float = 0.0
    u_max: float = 3.0
    e_min: float = 0.0
    e_max: float = 5.0
    num_u: int = 241
    num_e: int = 251
    g: float = 1.0
    epsilon: int = -1
    tolerance: float = 1e-9
    search_iterations: int = 28

    def to_json(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "u_min": self.u_min,
            "u_max": self.u_max,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "num_u": self.num_u,
            "num_e": self.num_e,
            "g": self.g,
            "epsilon": self.epsilon,
            "tolerance": self.tolerance,
            "search_iterations": self.search_iterations,
        }


def quartic_moments(
    level: int,
    *,
    energy: float,
    mean_x: float,
    u: float,
    epsilon: int,
    g: float = 1.0,
) -> dict[int, float]:
    if level < 2:
        raise ValueError("level must be at least 2")
    if epsilon not in (-1, 1):
        raise ValueError("epsilon must be +/- 1")
    if g <= 0:
        raise ValueError("g must be positive")

    sqrt2 = math.sqrt(2.0)
    g_sq = g * g
    moments: dict[int, float] = {
        0: 1.0,
        1: float(mean_x),
        2: float(u),
    }
    max_order = 2 * (level - 1)

    def get(index: int) -> float:
        if index < 0:
            return 0.0
        return moments.get(index, 0.0)

    for t in range(3, max_order + 1):
        moments[t] = (
            ((t - 3) * (t - 4) * (t - 5) / (2.0 * g_sq * (t - 1))) * get(t - 6)
            + (((4.0 * energy) - (1.0 / g_sq)) * (t - 3) / (g_sq * (t - 1))) * get(t - 4)
            - (sqrt2 * epsilon * (2 * t - 5) / (g * (t - 1))) * get(t - 3)
            - (2.0 * (t - 2) / (g_sq * (t - 1))) * get(t - 2)
        )
    return moments


def quartic_hankel_matrix(
    level: int,
    *,
    energy: float,
    mean_x: float,
    u: float,
    epsilon: int,
    g: float = 1.0,
) -> np.ndarray:
    moments = quartic_moments(level, energy=energy, mean_x=mean_x, u=u, epsilon=epsilon, g=g)
    return np.array(
        [[moments[i + j] for j in range(level)] for i in range(level)],
        dtype=float,
    )


def quartic_affine_hankel_pencil(
    level: int,
    *,
    energy: float,
    mean_x: float,
    epsilon: int,
    g: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_u0 = quartic_hankel_matrix(level, energy=energy, mean_x=mean_x, u=0.0, epsilon=epsilon, g=g)
    matrix_u1 = quartic_hankel_matrix(level, energy=energy, mean_x=mean_x, u=1.0, epsilon=epsilon, g=g)
    return matrix_u0, matrix_u1 - matrix_u0


def quartic_affine_mean_x_pencil(
    level: int,
    *,
    energy: float,
    u: float,
    epsilon: int,
    g: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    matrix_a0 = quartic_hankel_matrix(level, energy=energy, mean_x=0.0, u=u, epsilon=epsilon, g=g)
    matrix_a1 = quartic_hankel_matrix(level, energy=energy, mean_x=1.0, u=u, epsilon=epsilon, g=g)
    return matrix_a0, matrix_a1 - matrix_a0


def quartic_min_eigenvalue(
    matrix_u0: np.ndarray,
    matrix_u_slope: np.ndarray,
    *,
    u: float,
) -> float:
    matrix = matrix_u0 + u * matrix_u_slope
    return float(np.min(np.linalg.eigvalsh(matrix)))


def _maximize_concave_on_interval(
    evaluator,
    *,
    lower: float,
    upper: float,
    iterations: int,
) -> tuple[float, float]:
    if upper <= lower:
        value = evaluator(lower)
        return lower, value

    invphi = (math.sqrt(5.0) - 1.0) / 2.0
    left = float(lower)
    right = float(upper)
    x1 = right - invphi * (right - left)
    x2 = left + invphi * (right - left)
    f1 = evaluator(x1)
    f2 = evaluator(x2)

    for _ in range(iterations):
        if f1 < f2:
            left = x1
            x1 = x2
            f1 = f2
            x2 = left + invphi * (right - left)
            f2 = evaluator(x2)
        else:
            right = x2
            x2 = x1
            f2 = f1
            x1 = right - invphi * (right - left)
            f1 = evaluator(x1)

    boundary_points = [(lower, evaluator(lower)), (upper, evaluator(upper)), (x1, f1), (x2, f2)]
    best_u, best_value = max(boundary_points, key=lambda item: item[1])
    return float(best_u), float(best_value)


def quartic_projected_feasible(
    level: int,
    *,
    energy: float,
    mean_x: float,
    epsilon: int,
    g: float = 1.0,
    u_max: float = 20.0,
    tolerance: float = 1e-9,
    search_iterations: int = 28,
) -> tuple[bool, float, float]:
    lower = max(float(mean_x) ** 2, 0.0)
    upper = max(lower, float(u_max))
    matrix_u0, matrix_u_slope = quartic_affine_hankel_pencil(
        level,
        energy=energy,
        mean_x=mean_x,
        epsilon=epsilon,
        g=g,
    )

    def evaluator(u: float) -> float:
        return quartic_min_eigenvalue(matrix_u0, matrix_u_slope, u=u)

    best_u, best_value = _maximize_concave_on_interval(
        evaluator,
        lower=lower,
        upper=upper,
        iterations=search_iterations,
    )
    return best_value >= -tolerance, best_u, best_value


def quartic_projected_feasible_u(
    level: int,
    *,
    energy: float,
    u: float,
    epsilon: int,
    g: float = 1.0,
    tolerance: float = 1e-9,
    search_iterations: int = 28,
) -> tuple[bool, float, float]:
    if u < 0:
        return False, float("nan"), float("-inf")
    limit = math.sqrt(float(u))
    matrix_a0, matrix_a_slope = quartic_affine_mean_x_pencil(
        level,
        energy=energy,
        u=u,
        epsilon=epsilon,
        g=g,
    )

    def evaluator(mean_x: float) -> float:
        return quartic_min_eigenvalue(matrix_a0, matrix_a_slope, u=mean_x)

    best_a, best_value = _maximize_concave_on_interval(
        evaluator,
        lower=-limit,
        upper=limit,
        iterations=search_iterations,
    )
    return best_value >= -tolerance, best_a, best_value


def scan_figure2_ex(config: Figure2ExConfig) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    x_values = np.linspace(config.x_min, config.x_max, config.num_x)
    e_values = np.linspace(config.e_min, config.e_max, config.num_e)
    masks: dict[int, np.ndarray] = {}
    best_u_values: dict[int, np.ndarray] = {}

    candidate_mask = np.ones((config.num_e, config.num_x), dtype=bool)
    positive_indices = [index for index, x in enumerate(x_values) if x >= -1e-12]
    mirrored_index = {index: config.num_x - 1 - index for index in range(config.num_x)}

    for level in config.levels:
        mask = np.zeros((config.num_e, config.num_x), dtype=bool)
        u_star = np.full((config.num_e, config.num_x), np.nan, dtype=float)
        for e_index, energy in enumerate(e_values):
            for x_index in positive_indices:
                if not candidate_mask[e_index, x_index]:
                    continue
                mean_x = float(x_values[x_index])
                feasible, best_u, _ = quartic_projected_feasible(
                    level,
                    energy=float(energy),
                    mean_x=mean_x,
                    epsilon=-1,
                    g=config.g,
                    u_max=config.u_max,
                    tolerance=config.tolerance,
                    search_iterations=config.search_iterations,
                )
                mirror = mirrored_index[x_index]
                mask[e_index, x_index] = feasible
                u_star[e_index, x_index] = best_u if feasible else np.nan
                mask[e_index, mirror] = feasible
                u_star[e_index, mirror] = best_u if feasible else np.nan
        masks[level] = mask
        best_u_values[level] = u_star
        candidate_mask = mask
    return x_values, e_values, masks, best_u_values


def scan_figure2_eu(config: Figure2EuConfig) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    u_values = np.linspace(config.u_min, config.u_max, config.num_u)
    e_values = np.linspace(config.e_min, config.e_max, config.num_e)
    masks: dict[int, np.ndarray] = {}
    best_a_values: dict[int, np.ndarray] = {}

    candidate_mask = np.ones((config.num_e, config.num_u), dtype=bool)
    for level in config.levels:
        mask = np.zeros((config.num_e, config.num_u), dtype=bool)
        a_star = np.full((config.num_e, config.num_u), np.nan, dtype=float)
        for e_index, energy in enumerate(e_values):
            for u_index, u in enumerate(u_values):
                if not candidate_mask[e_index, u_index]:
                    continue
                feasible, best_a, _ = quartic_projected_feasible_u(
                    level,
                    energy=float(energy),
                    u=float(u),
                    epsilon=config.epsilon,
                    g=config.g,
                    tolerance=config.tolerance,
                    search_iterations=config.search_iterations,
                )
                mask[e_index, u_index] = feasible
                a_star[e_index, u_index] = best_a if feasible else np.nan
        masks[level] = mask
        best_a_values[level] = a_star
        candidate_mask = mask
    return u_values, e_values, masks, best_a_values


def write_feasible_points_csv(path: str | Path, x_values: np.ndarray, e_values: np.ndarray, masks: dict[int, np.ndarray]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "energy", "x"])
        writer.writeheader()
        for level in sorted(masks):
            e_indices, x_indices = np.nonzero(masks[level])
            for e_index, x_index in zip(e_indices, x_indices, strict=False):
                writer.writerow(
                    {
                        "level": level,
                        "energy": float(e_values[e_index]),
                        "x": float(x_values[x_index]),
                    }
                )


def write_feasible_u_points_csv(path: str | Path, u_values: np.ndarray, e_values: np.ndarray, masks: dict[int, np.ndarray]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "energy", "u"])
        writer.writeheader()
        for level in sorted(masks):
            e_indices, u_indices = np.nonzero(masks[level])
            for e_index, u_index in zip(e_indices, u_indices, strict=False):
                writer.writerow(
                    {
                        "level": level,
                        "energy": float(e_values[e_index]),
                        "u": float(u_values[u_index]),
                    }
                )


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_figure2_ex_regions(
    x_values: np.ndarray,
    e_values: np.ndarray,
    masks: dict[int, np.ndarray],
    *,
    out_path: str | Path,
) -> None:
    plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(8.8, 7.2))
    sorted_levels = sorted(masks)
    color_map = plt.get_cmap("viridis")
    color_positions = np.linspace(0.15, 0.95, len(sorted_levels))
    colors = {
        level: color_map(position)
        for level, position in zip(sorted_levels, color_positions, strict=False)
    }
    x_grid, e_grid = np.meshgrid(x_values, e_values)
    for level in sorted_levels:
        mask = masks[level].astype(float)
        color = colors[level]
        axis.contourf(
            x_grid,
            e_grid,
            mask,
            levels=[0.5, 1.5],
            colors=[color],
            alpha=0.14,
        )
        axis.contour(
            x_grid,
            e_grid,
            mask,
            levels=[0.5],
            colors=[color],
            linewidths=1.35,
        )
    for level in sorted_levels:
        axis.plot([], [], color=colors[level], linewidth=2.0, label=f"K={level}")
    axis.axvline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlim(x_values[0], x_values[-1])
    axis.set_ylim(e_values[0], e_values[-1])
    axis.set_xlabel(r"$\langle x \rangle$")
    axis.set_ylabel(r"$E$")
    axis.set_title(r"Figure 2 quartic correction: $E$ vs $\langle x \rangle$")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_figure2_eu_regions(
    u_values: np.ndarray,
    e_values: np.ndarray,
    masks: dict[int, np.ndarray],
    *,
    out_path: str | Path,
) -> None:
    plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(8.8, 7.2))
    sorted_levels = sorted(masks)
    color_map = plt.get_cmap("viridis")
    color_positions = np.linspace(0.15, 0.95, len(sorted_levels))
    colors = {
        level: color_map(position)
        for level, position in zip(sorted_levels, color_positions, strict=False)
    }
    u_grid, e_grid = np.meshgrid(u_values, e_values)
    for level in sorted_levels:
        mask = masks[level].astype(float)
        color = colors[level]
        axis.contourf(
            u_grid,
            e_grid,
            mask,
            levels=[0.5, 1.5],
            colors=[color],
            alpha=0.14,
        )
        axis.contour(
            u_grid,
            e_grid,
            mask,
            levels=[0.5],
            colors=[color],
            linewidths=1.35,
        )
    for level in sorted_levels:
        axis.plot([], [], color=colors[level], linewidth=2.0, label=f"K={level}")
    axis.set_xlim(u_values[0], u_values[-1])
    axis.set_ylim(e_values[0], e_values[-1])
    axis.set_xlabel(r"$\langle x^2 \rangle$")
    axis.set_ylabel(r"$E$")
    axis.set_title(r"Figure 2 quartic correction: $E$ vs $\langle x^2 \rangle$")
    axis.grid(True, alpha=0.2)
    axis.legend()
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_figure2_ex_scan(
    *,
    out_dir: str | Path,
    config: Figure2ExConfig | None = None,
) -> dict[str, Any]:
    resolved_config = Figure2ExConfig() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config.to_json(), indent=2),
        encoding="utf-8",
    )

    x_values, e_values, masks, best_u_values = scan_figure2_ex(resolved_config)
    write_feasible_points_csv(output_dir / "bounds.csv", x_values, e_values, masks)
    plot_figure2_ex_regions(
        x_values,
        e_values,
        masks,
        out_path=output_dir / "figure2_ex_k5_to_8.png",
    )

    summary_lines = [
        "# Figure 2 quartic correction: E vs <x>",
        "",
        "Model:",
        "- `W(x) = x/(sqrt(2) g) + g x^3/(3 sqrt(2))`",
        f"- `g = {resolved_config.g}`",
        "- right half (`x > 0`) uses `epsilon = -1`",
        "- left half (`x < 0`) is filled by the Z2 reflection",
        "",
        "Scan setup:",
        f"- levels: {', '.join(str(level) for level in resolved_config.levels)}",
        f"- x range: [{resolved_config.x_min}, {resolved_config.x_max}] with {resolved_config.num_x} points",
        f"- E range: [{resolved_config.e_min}, {resolved_config.e_max}] with {resolved_config.num_e} points",
        f"- u search interval: [x^2, {resolved_config.u_max}]",
        "",
        "Feasible point counts:",
    ]
    for level in sorted(masks):
        count = int(np.count_nonzero(masks[level]))
        boundary_hits = int(
            np.count_nonzero(
                masks[level]
                & np.isfinite(best_u_values[level])
                & (np.abs(best_u_values[level] - resolved_config.u_max) < 1e-3)
            )
        )
        summary_lines.append(f"- K={level}: {count} feasible grid points, {boundary_hits} upper-bound u hits")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "x_values": x_values,
        "e_values": e_values,
        "masks": masks,
        "best_u_values": best_u_values,
    }


def run_figure2_eu_scan(
    *,
    out_dir: str | Path,
    config: Figure2EuConfig | None = None,
) -> dict[str, Any]:
    resolved_config = Figure2EuConfig() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config.to_json(), indent=2),
        encoding="utf-8",
    )

    u_values, e_values, masks, best_a_values = scan_figure2_eu(resolved_config)
    write_feasible_u_points_csv(output_dir / "bounds.csv", u_values, e_values, masks)
    plot_figure2_eu_regions(
        u_values,
        e_values,
        masks,
        out_path=output_dir / "figure2_eu_k5_to_8.png",
    )

    summary_lines = [
        "# Figure 2 quartic correction: E vs <x^2>",
        "",
        "Model:",
        "- `W(x) = x/(sqrt(2) g) + g x^3/(3 sqrt(2))`",
        f"- `g = {resolved_config.g}`",
        f"- `epsilon = {resolved_config.epsilon}`",
        "- `a = <x>` is projected out",
        "",
        "Scan setup:",
        f"- levels: {', '.join(str(level) for level in resolved_config.levels)}",
        f"- x^2 range: [{resolved_config.u_min}, {resolved_config.u_max}] with {resolved_config.num_u} points",
        f"- E range: [{resolved_config.e_min}, {resolved_config.e_max}] with {resolved_config.num_e} points",
        "",
        "Feasible point counts:",
    ]
    for level in sorted(masks):
        count = int(np.count_nonzero(masks[level]))
        edge_hits = int(
            np.count_nonzero(
                masks[level]
                & np.isfinite(best_a_values[level])
                & (np.abs(np.abs(best_a_values[level]) - np.sqrt(np.maximum(u_values[np.newaxis, :], 0.0))) < 1e-3)
            )
        )
        summary_lines.append(f"- K={level}: {count} feasible grid points, {edge_hits} |a|=sqrt(u) hits")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "u_values": u_values,
        "e_values": e_values,
        "masks": masks,
        "best_a_values": best_a_values,
    }
