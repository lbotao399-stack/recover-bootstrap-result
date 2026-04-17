from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import sympy as sp

x = sp.Symbol("x", real=True)
p = sp.Symbol("p", real=True)


def _normalize_coefficients(coefficients: Mapping[int, sp.Expr | int | float]) -> tuple[tuple[int, sp.Expr], ...]:
    normalized: list[tuple[int, sp.Expr]] = []
    for degree, coefficient in sorted(coefficients.items()):
        coeff_expr = sp.simplify(coefficient)
        if coeff_expr == 0:
            continue
        normalized.append((int(degree), coeff_expr))
    return tuple(normalized)


@dataclass(frozen=True)
class PolynomialSuperpotential:
    coefficients: tuple[tuple[int, sp.Expr], ...]
    label: str = "generic"

    @classmethod
    def from_coefficients(
        cls,
        coefficients: Mapping[int, sp.Expr | int | float],
        *,
        label: str = "generic",
    ) -> "PolynomialSuperpotential":
        return cls(coefficients=_normalize_coefficients(coefficients), label=label)

    @property
    def expr(self) -> sp.Expr:
        return sp.expand(sum(coeff * x**degree for degree, coeff in self.coefficients))

    @property
    def degree(self) -> int:
        if not self.coefficients:
            return 0
        return self.coefficients[-1][0]

    @property
    def leading_coefficient(self) -> sp.Expr:
        if not self.coefficients:
            return sp.Integer(0)
        return self.coefficients[-1][1]

    def derivative(self, order: int = 1) -> sp.Expr:
        if order < 0:
            raise ValueError("order must be non-negative")
        return sp.expand(sp.diff(self.expr, x, order))

    def sector_potential(self, epsilon: int | sp.Expr) -> sp.Expr:
        epsilon_expr = sp.sympify(epsilon)
        return sp.expand(
            sp.Rational(1, 2) * self.derivative(1) ** 2
            + sp.Rational(1, 2) * epsilon_expr * self.derivative(2)
        )

    def sector_hamiltonian(self, epsilon: int | sp.Expr) -> sp.Expr:
        return sp.expand(sp.Rational(1, 2) * p**2 + self.sector_potential(epsilon))

    def zero_mode_wavefunction(self, epsilon: int) -> sp.Expr:
        if epsilon not in (-1, 1):
            raise ValueError("epsilon must be -1 or 1")
        return sp.exp(-self.expr if epsilon == -1 else self.expr)

    def normalizable_zero_mode_sector(self) -> int | None:
        if self.degree <= 0:
            return None
        if self.degree % 2 == 1:
            return None
        leading = sp.simplify(self.leading_coefficient)
        if leading.is_positive:
            return -1
        if leading.is_negative:
            return 1
        return None


def harmonic_oscillator_superpotential(omega: sp.Expr | int | float) -> PolynomialSuperpotential:
    omega_expr = sp.sympify(omega)
    return PolynomialSuperpotential.from_coefficients(
        {2: sp.Rational(1, 2) * omega_expr},
        label="sho",
    )


def quartic_correction_superpotential(g: sp.Expr | int | float) -> PolynomialSuperpotential:
    g_expr = sp.sympify(g)
    sqrt2 = sp.sqrt(2)
    return PolynomialSuperpotential.from_coefficients(
        {
            1: 1 / (sqrt2 * g_expr),
            3: g_expr / (3 * sqrt2),
        },
        label="quartic_correction",
    )


def cubic_mp_superpotential(g: sp.Expr | int | float) -> PolynomialSuperpotential:
    g_expr = sp.sympify(g)
    return PolynomialSuperpotential.from_coefficients(
        {
            2: sp.Rational(1, 2),
            3: g_expr / 3,
        },
        label="cubic_mp_n1",
    )


def harmonic_oscillator_energy(
    omega: sp.Expr | int | float,
    *,
    n: int,
    epsilon: int,
) -> sp.Expr:
    omega_expr = sp.sympify(omega)
    return sp.simplify(sp.Abs(omega_expr) * (n + sp.Rational(1, 2)) + sp.Rational(1, 2) * epsilon * omega_expr)

