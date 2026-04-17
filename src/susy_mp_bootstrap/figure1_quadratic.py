from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json

import numpy as np


@dataclass(frozen=True)
class Figure1Config:
    levels: tuple[int, ...] = (4, 5, 6, 7)
    u_min: float = 0.0
    u_max: float = 10.0
    e_min: float = -1.0
    e_max: float = 10.0
    num_u: int = 501
    num_e: int = 551
    tolerance: float = 1e-9

    def to_json(self) -> dict[str, Any]:
        return {
            "levels": list(self.levels),
            "u_min": self.u_min,
            "u_max": self.u_max,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "num_u": self.num_u,
            "num_e": self.num_e,
            "tolerance": self.tolerance,
        }


@dataclass(frozen=True)
class Figure1LineConfig:
    min_level: int = 4
    max_level: int = 120
    e_min: float = -1.0
    e_max: float = 10.0
    num_e: int = 12001
    tolerance: float = 1e-9
    u_shift: float = 0.5

    def to_json(self) -> dict[str, Any]:
        return {
            "min_level": self.min_level,
            "max_level": self.max_level,
            "e_min": self.e_min,
            "e_max": self.e_max,
            "num_e": self.num_e,
            "tolerance": self.tolerance,
            "u_shift": self.u_shift,
        }


def quadratic_sho_even_moments(level: int, *, energy: float, u: float) -> dict[int, float]:
    if level < 2:
        raise ValueError("level must be at least 2")
    moments: dict[int, float] = {
        0: 1.0,
        1: 0.0,
        2: float(u),
        3: 0.0,
    }
    a = 2.0 * float(energy) + 1.0
    max_order = 2 * (level - 1)
    for t in range(4, max_order + 1):
        if t % 2 == 1:
            moments[t] = 0.0
            continue
        moments[t] = (
            ((t - 1) / t) * a * moments[t - 2]
            + ((t - 1) * (t - 2) * (t - 3) / (4.0 * t)) * moments[t - 4]
        )
    return moments


def quadratic_hankel_matrix(level: int, *, energy: float, u: float) -> np.ndarray:
    moments = quadratic_sho_even_moments(level, energy=energy, u=u)
    return np.array(
        [[moments[i + j] for j in range(level)] for i in range(level)],
        dtype=float,
    )


def _parity_block(matrix: np.ndarray, parity: int) -> np.ndarray:
    indices = [index for index in range(matrix.shape[0]) if index % 2 == parity]
    return matrix[np.ix_(indices, indices)]


def quadratic_feasible(level: int, *, energy: float, u: float, tolerance: float = 1e-9) -> bool:
    if u < 0:
        return False
    matrix = quadratic_hankel_matrix(level, energy=energy, u=u)
    for parity in (0, 1):
        block = _parity_block(matrix, parity)
        if block.size == 0:
            continue
        min_eigenvalue = float(np.min(np.linalg.eigvalsh(block)))
        if min_eigenvalue < -tolerance:
            return False
    return True


def quadratic_line_u(energy: float, *, shift: float = 0.5) -> float:
    return float(energy + shift)


def scan_figure1_region(config: Figure1Config) -> tuple[np.ndarray, np.ndarray, dict[int, np.ndarray]]:
    u_values = np.linspace(config.u_min, config.u_max, config.num_u)
    e_values = np.linspace(config.e_min, config.e_max, config.num_e)
    masks: dict[int, np.ndarray] = {}
    for level in config.levels:
        mask = np.zeros((config.num_e, config.num_u), dtype=bool)
        for e_index, energy in enumerate(e_values):
            for u_index, u in enumerate(u_values):
                mask[e_index, u_index] = quadratic_feasible(
                    level,
                    energy=float(energy),
                    u=float(u),
                    tolerance=config.tolerance,
                )
        masks[level] = mask
    return u_values, e_values, masks


def scan_figure1_line(config: Figure1LineConfig) -> tuple[np.ndarray, dict[int, np.ndarray]]:
    if config.min_level < 2:
        raise ValueError("min_level must be at least 2")
    if config.max_level < config.min_level:
        raise ValueError("max_level must be >= min_level")

    e_values = np.linspace(config.e_min, config.e_max, config.num_e)
    candidate_mask = np.ones(config.num_e, dtype=bool)
    masks: dict[int, np.ndarray] = {}

    for level in range(config.min_level, config.max_level + 1):
        mask = np.zeros(config.num_e, dtype=bool)
        candidate_indices = np.flatnonzero(candidate_mask)
        for index in candidate_indices:
            energy = float(e_values[index])
            u = quadratic_line_u(energy, shift=config.u_shift)
            mask[index] = quadratic_feasible(level, energy=energy, u=u, tolerance=config.tolerance)
        masks[level] = mask
        candidate_mask = mask
        if not np.any(candidate_mask):
            break
    return e_values, masks


def summarise_masks(
    u_values: np.ndarray,
    e_values: np.ndarray,
    masks: dict[int, np.ndarray],
) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []
    for level, mask in masks.items():
        for e_index, energy in enumerate(e_values):
            feasible_indices = np.flatnonzero(mask[e_index])
            if feasible_indices.size == 0:
                rows.append(
                    {
                        "level": level,
                        "energy": float(energy),
                        "u_min": np.nan,
                        "u_max": np.nan,
                        "n_feasible": 0,
                    }
                )
                continue
            rows.append(
                {
                    "level": level,
                    "energy": float(energy),
                    "u_min": float(u_values[feasible_indices[0]]),
                    "u_max": float(u_values[feasible_indices[-1]]),
                    "n_feasible": int(feasible_indices.size),
                }
            )
    return rows


def write_bounds_csv(path: str | Path, rows: list[dict[str, float | int]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "energy", "u_min", "u_max", "n_feasible"])
        writer.writeheader()
        writer.writerows(rows)


def line_mask_to_intervals(e_values: np.ndarray, mask: np.ndarray) -> list[tuple[float, float, int]]:
    intervals: list[tuple[float, float, int]] = []
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return intervals
    start = indices[0]
    prev = indices[0]
    count = 1
    for index in indices[1:]:
        if index == prev + 1:
            prev = index
            count += 1
            continue
        intervals.append((float(e_values[start]), float(e_values[prev]), count))
        start = index
        prev = index
        count = 1
    intervals.append((float(e_values[start]), float(e_values[prev]), count))
    return intervals


def _format_interval_preview(intervals: list[tuple[float, float, int]], *, preview_count: int = 3) -> str:
    if not intervals:
        return "none"

    def format_interval(interval: tuple[float, float, int]) -> str:
        e_min, e_max, n_points = interval
        return f"[{e_min:.6f}, {e_max:.6f}] ({n_points} pts)"

    if len(intervals) <= 2 * preview_count:
        return ", ".join(format_interval(interval) for interval in intervals)

    head = ", ".join(format_interval(interval) for interval in intervals[:preview_count])
    tail = ", ".join(format_interval(interval) for interval in intervals[-preview_count:])
    return f"{head}, ..., {tail}"


def write_line_bounds_csv(path: str | Path, e_values: np.ndarray, masks: dict[int, np.ndarray]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["level", "interval", "e_min", "e_max", "n_points"])
        writer.writeheader()
        for level in sorted(masks):
            intervals = line_mask_to_intervals(e_values, masks[level])
            if not intervals:
                writer.writerow({"level": level, "interval": -1, "e_min": np.nan, "e_max": np.nan, "n_points": 0})
                continue
            for interval_index, (e_min, e_max, n_points) in enumerate(intervals):
                writer.writerow(
                    {
                        "level": level,
                        "interval": interval_index,
                        "e_min": e_min,
                        "e_max": e_max,
                        "n_points": n_points,
                    }
                )


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_figure1_regions(
    u_values: np.ndarray,
    e_values: np.ndarray,
    masks: dict[int, np.ndarray],
    *,
    out_path: str | Path,
) -> None:
    plt = _import_matplotlib()
    figure, axis = plt.subplots(figsize=(8.5, 7.0))
    sorted_levels = sorted(masks)
    color_map = plt.get_cmap("viridis")
    color_positions = np.linspace(0.1, 0.95, len(sorted_levels))
    colors = {
        level: color_map(position)
        for level, position in zip(sorted_levels, color_positions, strict=False)
    }
    fill_regions = len(sorted_levels) <= 8
    u_grid, e_grid = np.meshgrid(u_values, e_values)
    for level in sorted_levels:
        mask = masks[level].astype(float)
        color = colors[level]
        if fill_regions:
            axis.contourf(
                u_grid,
                e_grid,
                mask,
                levels=[0.5, 1.5],
                colors=[color],
                alpha=0.16,
            )
        axis.contour(
            u_grid,
            e_grid,
            mask,
            levels=[0.5],
            colors=[color],
            linewidths=1.6 if not fill_regions else 1.3,
        )
    for level in sorted_levels:
        color = colors[level]
        axis.plot([], [], color=color, linewidth=2.0, label=f"K={level}")
    axis.set_xlim(u_values[0], u_values[-1])
    axis.set_ylim(e_values[0], e_values[-1])
    axis.set_xlabel(r"$\langle x^2 \rangle$")
    axis.set_ylabel(r"$E$")
    axis.set_title("Figure 1 quadratic toy-model bootstrap")
    axis.grid(True, alpha=0.25)
    if len(sorted_levels) > 12:
        legend_columns = 3
    elif len(sorted_levels) > 8:
        legend_columns = 2
    else:
        legend_columns = 1
    axis.legend(ncol=legend_columns, fontsize=9)
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_figure1_line_scan(
    e_values: np.ndarray,
    masks: dict[int, np.ndarray],
    *,
    out_path: str | Path,
) -> None:
    plt = _import_matplotlib()
    sorted_levels = sorted(masks)
    if not sorted_levels:
        raise ValueError("masks cannot be empty")
    image = np.vstack([masks[level] for level in sorted_levels]).astype(float)
    figure, axis = plt.subplots(figsize=(9.0, 7.2))
    axis.imshow(
        image,
        aspect="auto",
        origin="lower",
        interpolation="nearest",
        extent=[float(e_values[0]), float(e_values[-1]), sorted_levels[0] - 0.5, sorted_levels[-1] + 0.5],
        cmap="viridis",
        vmin=0.0,
        vmax=1.0,
    )
    axis.set_xlabel(r"$E$")
    axis.set_ylabel(r"$K$")
    axis.set_title(r"Quadratic bootstrap on $u = E + \frac{1}{2}$")
    axis.grid(False)
    axis.set_yticks(sorted_levels[:: max(1, len(sorted_levels) // 12)])
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def run_figure1_scan(
    *,
    out_dir: str | Path,
    config: Figure1Config | None = None,
) -> dict[str, Any]:
    resolved_config = Figure1Config() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config.to_json(), indent=2),
        encoding="utf-8",
    )
    u_values, e_values, masks = scan_figure1_region(resolved_config)
    rows = summarise_masks(u_values, e_values, masks)
    write_bounds_csv(output_dir / "bounds.csv", rows)
    plot_figure1_regions(
        u_values,
        e_values,
        masks,
        out_path=output_dir / "figure1_levels_4_5_6_7.png",
    )

    counts = {level: int(np.count_nonzero(mask)) for level, mask in masks.items()}
    summary_lines = [
        "# Figure 1 quadratic bootstrap",
        "",
        "Model: `W(x) = x^2 / 2`, `epsilon = -1`, parity-even projection.",
        "",
        "Scan setup:",
        f"- levels: {', '.join(str(level) for level in resolved_config.levels)}",
        f"- x^2 range: [{resolved_config.u_min}, {resolved_config.u_max}]",
        f"- E range: [{resolved_config.e_min}, {resolved_config.e_max}]",
        f"- grid: {resolved_config.num_u} x-points, {resolved_config.num_e} energy points",
        "",
        "Feasible grid counts:",
    ]
    for level in sorted(counts):
        summary_lines.append(f"- K={level}: {counts[level]} feasible points")
    summary_lines.extend(
        [
            "",
            "Construction:",
            "- moments are seeded by `m0 = 1`, `m1 = 0`, `m2 = u`, `m3 = 0`",
            "- all higher moments are generated by the quadratic recursion",
            "- the `K x K` Hankel matrix is required to be positive semidefinite",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "u_values": u_values,
        "e_values": e_values,
        "masks": masks,
        "rows": rows,
    }


def run_figure1_line_scan(
    *,
    out_dir: str | Path,
    config: Figure1LineConfig | None = None,
) -> dict[str, Any]:
    resolved_config = Figure1LineConfig() if config is None else config
    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(resolved_config.to_json(), indent=2),
        encoding="utf-8",
    )
    e_values, masks = scan_figure1_line(resolved_config)
    write_line_bounds_csv(output_dir / "bounds.csv", e_values, masks)
    plot_figure1_line_scan(
        e_values,
        masks,
        out_path=output_dir / "figure1_line_u_eq_E_plus_half.png",
    )

    surviving_levels = sorted(masks)
    summary_lines = [
        "# Figure 1 line scan",
        "",
        r"Constraint: `u = E + 1/2`.",
        "",
        "Scan setup:",
        f"- levels requested: {resolved_config.min_level}..{resolved_config.max_level}",
        f"- levels completed: {surviving_levels[0]}..{surviving_levels[-1]}",
        f"- energy range: [{resolved_config.e_min}, {resolved_config.e_max}]",
        f"- energy grid points: {resolved_config.num_e}",
        f"- higher levels were evaluated only on lower-level feasible grid points",
        "",
        "Feasible intervals:",
    ]
    for level in surviving_levels:
        intervals = line_mask_to_intervals(e_values, masks[level])
        if not intervals:
            summary_lines.append(f"- K={level}: none")
            continue
        total_points = sum(n_points for _, _, n_points in intervals)
        interval_text = _format_interval_preview(intervals)
        summary_lines.append(
            f"- K={level}: {len(intervals)} interval(s), {total_points} pts; {interval_text}"
        )
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return {
        "e_values": e_values,
        "masks": masks,
    }
