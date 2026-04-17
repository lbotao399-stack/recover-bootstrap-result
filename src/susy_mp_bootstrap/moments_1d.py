from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import sympy as sp

from .models_1d import PolynomialSuperpotential, x


MomentCoefficients = dict[int, sp.Expr]


def _poly_terms(poly_expr: sp.Expr, *, shift: int) -> MomentCoefficients:
    polynomial = sp.Poly(sp.expand(poly_expr), x)
    terms: MomentCoefficients = {}
    for (degree,), coefficient in polynomial.terms():
        index = shift + degree
        if index < 0:
            continue
        terms[index] = sp.simplify(terms.get(index, 0) + coefficient)
    return {index: coefficient for index, coefficient in terms.items() if coefficient != 0}


def _add_scaled(target: MomentCoefficients, source: MomentCoefficients, scale: sp.Expr) -> None:
    for index, coefficient in source.items():
        target[index] = sp.simplify(target.get(index, 0) + scale * coefficient)
        if target[index] == 0:
            del target[index]


def recursion_coefficients(
    superpotential: PolynomialSuperpotential,
    *,
    t: int,
    energy: sp.Expr | int | float,
    epsilon: int | sp.Expr,
) -> MomentCoefficients:
    if t < 0:
        raise ValueError("t must be non-negative")

    energy_expr = sp.sympify(energy)
    epsilon_expr = sp.sympify(epsilon)
    coefficients: MomentCoefficients = defaultdict(lambda: sp.Integer(0))

    if t - 1 >= 0:
        coefficients[t - 1] = sp.simplify(coefficients[t - 1] + 8 * t * energy_expr)
    if t - 3 >= 0:
        coefficients[t - 3] = sp.simplify(coefficients[t - 3] + t * (t - 1) * (t - 2))

    w1 = superpotential.derivative(1)
    w2 = superpotential.derivative(2)
    w3 = superpotential.derivative(3)

    _add_scaled(coefficients, _poly_terms(w1**2, shift=t - 1), -4 * t)
    _add_scaled(coefficients, _poly_terms(w2, shift=t - 1), -4 * t * epsilon_expr)
    _add_scaled(coefficients, _poly_terms(w2 * w1, shift=t), -4)
    _add_scaled(coefficients, _poly_terms(w3, shift=t), -2 * epsilon_expr)

    return {index: sp.simplify(coefficient) for index, coefficient in sorted(coefficients.items()) if coefficient != 0}


def harmonic_oscillator_recursion_coefficients(
    *,
    t: int,
    energy: sp.Expr | int | float,
    epsilon: int | sp.Expr,
    omega: sp.Expr | int | float,
) -> MomentCoefficients:
    energy_expr = sp.sympify(energy)
    epsilon_expr = sp.sympify(epsilon)
    omega_expr = sp.sympify(omega)
    coefficients: MomentCoefficients = {}
    if t - 4 >= 0:
        coefficients[t - 4] = sp.simplify((t - 3) * (t - 2) * (t - 1))
    if t - 2 >= 0:
        coefficients[t - 2] = sp.simplify(4 * (t - 1) * (2 * energy_expr - epsilon_expr * omega_expr))
    coefficients[t] = sp.simplify(-4 * t * omega_expr**2)
    return {index: coefficient for index, coefficient in sorted(coefficients.items()) if coefficient != 0}


def build_recursion_constraints(
    superpotential: PolynomialSuperpotential,
    *,
    epsilon: int,
    energy: sp.Expr | int | float,
    moment_cutoff: int,
    t_values: Iterable[int] | None = None,
) -> list[MomentCoefficients]:
    if moment_cutoff < 0:
        raise ValueError("moment_cutoff must be non-negative")
    if t_values is None:
        t_values = range(1, moment_cutoff + 1)
    constraints: list[MomentCoefficients] = []
    for t in t_values:
        coefficients = recursion_coefficients(superpotential, t=t, energy=energy, epsilon=epsilon)
        if not coefficients:
            continue
        if max(coefficients) > moment_cutoff:
            continue
        constraints.append(coefficients)
    return constraints

