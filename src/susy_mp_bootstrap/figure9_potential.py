from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import csv
import json
import math

import numpy as np


FIGURE9_GC = (6.0 * math.sqrt(3.0)) ** (-0.5)
FIGURE9_DEFAULT_G_VALUES = (0.2, FIGURE9_GC, 0.5)


@dataclass(frozen=True)
class Figure9Config:
    g_values: tuple[float, float, float] = FIGURE9_DEFAULT_G_VALUES
    x_min: float = -4.6
    x_max: float = 1.55
    y_min: float = -1.0
    y_max: float = 2.0
    num_x: int = 1200

    def to_json(self) -> dict[str, Any]:
        return {
            "g_values": list(self.g_values),
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "num_x": self.num_x,
        }


def critical_coupling() -> float:
    return FIGURE9_GC


def figure9_source_formula_typo(x: float | np.ndarray, g: float) -> float | np.ndarray:
    x_array = np.asarray(x, dtype=float)
    values = 0.5 * (x_array**2 + 2.0 * g * x_array**2 + g * g * x_array**4 - 1.0 - 2.0 * g * x_array)
    if np.isscalar(x):
        return float(values)
    return values


def figure9_potential(x: float | np.ndarray, g: float) -> float | np.ndarray:
    x_array = np.asarray(x, dtype=float)
    values = 0.5 * (x_array**2 + 2.0 * g * x_array**3 + g * g * x_array**4 - 1.0 - 2.0 * g * x_array)
    if np.isscalar(x):
        return float(values)
    return values


def figure9_stationary_discriminant(g: float) -> float:
    return -(g**2) * (108.0 * g**4 - 1.0)


def figure9_stationary_points(g: float, *, tolerance: float = 1e-6) -> tuple[float, ...]:
    if abs(g) < tolerance:
        return (0.0,)
    roots = np.roots([2.0 * g * g, 3.0 * g, 1.0, -g])
    real_roots = sorted(float(root.real) for root in roots if abs(root.imag) < tolerance)
    if not real_roots:
        return tuple()
    deduplicated: list[float] = [real_roots[0]]
    for value in real_roots[1:]:
        if abs(value - deduplicated[-1]) > 10.0 * tolerance:
            deduplicated.append(value)
    return tuple(deduplicated)


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def _curve_label(g: float) -> str:
    if abs(g - FIGURE9_GC) < 1e-12:
        return r"$g = g_c$"
    return f"$g = {g:.1f}$"


def _csv_label(g: float) -> str:
    if abs(g - FIGURE9_GC) < 1e-12:
        return "g_gc"
    return f"g_{str(g).replace('.', '_')}"


def plot_figure9(
    x_values: np.ndarray,
    curves: dict[float, np.ndarray],
    *,
    out_path: str | Path,
    config: Figure9Config,
) -> None:
    pyplot = _import_matplotlib()
    figure, axis = pyplot.subplots(figsize=(7.2, 5.0))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for color, g in zip(colors, config.g_values, strict=False):
        axis.plot(x_values, curves[g], color=color, linewidth=2.3, label=_curve_label(g))
    axis.set_xlim(config.x_min, config.x_max)
    axis.set_ylim(config.y_min, config.y_max)
    axis.set_xlabel(r"$x$")
    axis.set_ylabel(r"$V(x)$")
    axis.set_title(r"$V(x)$")
    axis.grid(True, alpha=0.12)
    axis.legend(loc="upper right")
    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)


def run_figure9_plot(*, out_dir: str | Path, config: Figure9Config | None = None) -> Path:
    if config is None:
        config = Figure9Config()

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_values = np.linspace(config.x_min, config.x_max, config.num_x)
    curves = {g: np.asarray(figure9_potential(x_values, g), dtype=float) for g in config.g_values}

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_json(), handle, indent=2)

    fieldnames = ["x"] + [_csv_label(g) for g in config.g_values]
    with (output_dir / "curves.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, x_value in enumerate(x_values):
            row = {"x": float(x_value)}
            for g in config.g_values:
                row[_csv_label(g)] = float(curves[g][index])
            writer.writerow(row)

    summary_lines = [
        "# Figure 9 summary",
        "",
        "- Corrected bosonic potential used for plotting:",
        "  - `V(x) = 1/2 (x^2 + 2 g x^3 + g^2 x^4 - 1 - 2 g x)`",
        "- Reason:",
        "  - the printed `main.tex` formula has `2 g x^2`, but that cannot produce the stated critical behavior.",
        "  - the corrected cubic term follows from `W'(x) = x + g x^2`.",
        f"- Critical coupling: `g_c = {FIGURE9_GC:.15f}`",
        "",
        "## Stationary points",
    ]
    for g in config.g_values:
        points = ", ".join(f"{value:.8f}" for value in figure9_stationary_points(g))
        summary_lines.append(f"- `{_curve_label(g)}`: `{points}`")
    summary_lines.append("")
    summary_lines.append("The critical curve is identified by the vanishing derivative discriminant")
    summary_lines.append("`Delta = -g^2 (108 g^4 - 1)`.")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    plot_figure9(x_values, curves, out_path=output_dir / "figure9_potential.png", config=config)
    return output_dir / "figure9_potential.png"
