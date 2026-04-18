from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from fractions import Fraction
from pathlib import Path
from typing import Any
import csv
import json
import math

import numpy as np

from .matrix_words import PSI, PSIDAG, P, X


RawWord = tuple[str, ...]
MultiTraceMonomial = tuple[RawWord, ...]
ConstExpr = dict[int, complex]
PolyCoeff = tuple[complex, complex, complex]
PolyExpr = dict[int, PolyCoeff]

_ZERO_POLY: PolyCoeff = (0.0 + 0.0j, 0.0 + 0.0j, 0.0 + 0.0j)
_RAW_LETTERS = (X, P, PSI, PSIDAG)
_LETTER_LEVELS = {
    X: Fraction(1, 1),
    P: Fraction(2, 1),
    PSI: Fraction(3, 2),
    PSIDAG: Fraction(3, 2),
}
_LETTER_CHARGES = {
    X: 0,
    P: 0,
    PSI: -1,
    PSIDAG: 1,
}
_LETTER_DAGGER = {
    X: X,
    P: P,
    PSI: PSIDAG,
    PSIDAG: PSI,
}


def _import_cvxpy():
    import cvxpy as cp

    return cp


def _poly_const(value: complex) -> PolyCoeff:
    return (complex(value), 0.0 + 0.0j, 0.0 + 0.0j)


def _poly_linear(value: complex) -> PolyCoeff:
    return (0.0 + 0.0j, complex(value), 0.0 + 0.0j)


def _poly_quadratic(value: complex) -> PolyCoeff:
    return (0.0 + 0.0j, 0.0 + 0.0j, complex(value))


def _poly_add(left: PolyCoeff, right: PolyCoeff) -> PolyCoeff:
    return (left[0] + right[0], left[1] + right[1], left[2] + right[2])


def _poly_scale(poly: PolyCoeff, scalar: complex) -> PolyCoeff:
    return (scalar * poly[0], scalar * poly[1], scalar * poly[2])


def _poly_is_zero(poly: PolyCoeff, *, tolerance: float = 1e-13) -> bool:
    return abs(poly[0]) < tolerance and abs(poly[1]) < tolerance and abs(poly[2]) < tolerance


def _raw_word_level(word: RawWord) -> Fraction:
    return sum((_LETTER_LEVELS[letter] for letter in word), start=Fraction(0, 1))


def _raw_word_charge(word: RawWord) -> int:
    return sum(_LETTER_CHARGES[letter] for letter in word)


def _raw_word_p_count(word: RawWord) -> int:
    return sum(letter == P for letter in word)


def _raw_word_fermion_parity(word: RawWord) -> int:
    return sum(letter in (PSI, PSIDAG) for letter in word) % 2


def _raw_word_dagger(word: RawWord) -> RawWord:
    return tuple(_LETTER_DAGGER[letter] for letter in reversed(word))


def _monomial_level(monomial: MultiTraceMonomial) -> Fraction:
    return sum((_raw_word_level(word) for word in monomial), start=Fraction(0, 1))


def _monomial_charge(monomial: MultiTraceMonomial) -> int:
    return sum(_raw_word_charge(word) for word in monomial)


def _monomial_p_count(monomial: MultiTraceMonomial) -> int:
    return sum(_raw_word_p_count(word) for word in monomial)


def _monomial_dagger(monomial: MultiTraceMonomial) -> MultiTraceMonomial:
    return tuple(_raw_word_dagger(word) for word in reversed(monomial))


def _all_raw_words_up_to_level(max_level: Fraction) -> tuple[RawWord, ...]:
    words: list[RawWord] = [tuple()]
    frontier: list[tuple[RawWord, Fraction]] = [(tuple(), Fraction(0, 1))]
    while frontier:
        next_frontier: list[tuple[RawWord, Fraction]] = []
        for word, word_level in frontier:
            for letter in _RAW_LETTERS:
                new_level = word_level + _LETTER_LEVELS[letter]
                if new_level <= max_level:
                    new_word = word + (letter,)
                    words.append(new_word)
                    next_frontier.append((new_word, new_level))
        frontier = next_frontier
    return tuple(words)


@dataclass(frozen=True)
class Figure10Config:
    n: int = 100
    basis_level: Fraction = Fraction(4, 1)
    observable_level: Fraction = Fraction(8, 1)
    universe_source_level: Fraction = Fraction(8, 1)
    gauge_seed_level: Fraction = Fraction(5, 1)
    eom_seed_level: Fraction = Fraction(7, 1)
    reality_seed_level: Fraction = Fraction(5, 1)
    include_ground_block: bool = False
    shift_scale_min_g: float = 20.0
    g_values: tuple[float, ...] = (
        1.0,
        5.0,
        10.0,
        15.0,
        20.0,
        30.0,
        40.0,
        50.0,
        80.0,
        100.0,
        150.0,
        200.0,
        300.0,
        400.0,
        500.0,
        600.0,
    )
    fit_min_g: float = 20.0
    solver: str = "AUTO"
    solver_eps: float = 1e-5
    solver_max_iters: int = 50000
    eq_tolerance: float = 5e-4
    psd_tolerance: float = 5e-4

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
            "shift_scale_min_g": self.shift_scale_min_g,
            "g_values": list(self.g_values),
            "fit_min_g": self.fit_min_g,
            "solver": self.solver,
            "solver_eps": self.solver_eps,
            "solver_max_iters": self.solver_max_iters,
            "eq_tolerance": self.eq_tolerance,
            "psd_tolerance": self.psd_tolerance,
        }


@dataclass
class Figure10SolveResult:
    g: float
    alpha: float
    status: str
    objective_value: float | None
    energy_density: float | None
    eq_residual: float | None
    psd_residual: float | None
    feasible: bool


@dataclass
class _Figure10Problem:
    cp: Any
    problem: Any
    alpha_linear: Any
    alpha_quadratic: Any
    variables: Any
    ordinary_blocks: list[list[list[PolyExpr]]]
    ground_blocks: list[list[list[PolyExpr]]]
    equality_exprs: list[PolyExpr]
    objective_expr: PolyExpr
    energy_shift: float
    energy_scale: float


class Figure10Reducer:
    def __init__(
        self,
        *,
        n: int,
        basis_level: Fraction = Fraction(4, 1),
        observable_level: Fraction = Fraction(8, 1),
        universe_source_level: Fraction | None = None,
    ) -> None:
        if n <= 0:
            raise ValueError("n must be positive")
        self.n = int(n)
        self.basis_level = Fraction(basis_level)
        self.observable_level = Fraction(observable_level)
        if universe_source_level is None:
            universe_source_level = self.observable_level + Fraction(3, 1)
        self.universe_source_level = Fraction(universe_source_level)
        self._canonical_cache: dict[MultiTraceMonomial, dict[MultiTraceMonomial, complex]] = {}
        self._raw_expr_cache: dict[RawWord, ConstExpr] = {}

        self.all_raw_words = _all_raw_words_up_to_level(self.universe_source_level)
        self.raw_words_by_charge = {
            charge: tuple(word for word in self.all_raw_words if _raw_word_charge(word) == charge)
            for charge in (-2, -1, 0, 1, 2)
        }
        self.basis_blocks = self._build_basis_blocks()
        self.canonical_universe = self._build_canonical_universe()
        self.monomial_to_index = {monomial: index for index, monomial in enumerate(self.canonical_universe, start=1)}

    def _build_basis_blocks(self) -> dict[int, tuple[RawWord, ...]]:
        blocks: dict[int, list[RawWord]] = {-2: [], -1: [], 0: [], 1: [], 2: []}
        for word in self.all_raw_words:
            if _raw_word_level(word) <= self.basis_level:
                charge = _raw_word_charge(word)
                if charge in blocks:
                    blocks[charge].append(word)
        return {charge: tuple(blocks[charge]) for charge in sorted(blocks)}

    def block_dimensions(self) -> dict[int, int]:
        return {charge: len(words) for charge, words in self.basis_blocks.items()}

    def _normalize_monomial(self, monomial: MultiTraceMonomial) -> tuple[MultiTraceMonomial, complex]:
        coefficient = 1.0 + 0.0j
        filtered: list[RawWord] = []
        for factor in monomial:
            if len(factor) == 0:
                coefficient *= self.n
            else:
                filtered.append(factor)
        return tuple(filtered), coefficient

    def _rewrite_once(self, monomial: MultiTraceMonomial) -> list[tuple[MultiTraceMonomial, complex]] | None:
        for factor_index, factor in enumerate(monomial):
            for position in range(len(factor) - 1):
                left_letter = factor[position]
                right_letter = factor[position + 1]
                prefix = factor[:position]
                suffix = factor[position + 2 :]
                left_factors = monomial[:factor_index]
                right_factors = monomial[factor_index + 1 :]
                if left_letter == P and right_letter == X:
                    return [
                        (left_factors + (prefix + (X, P) + suffix,) + right_factors, 1.0 + 0.0j),
                        (left_factors + (prefix, suffix) + right_factors, -1.0j),
                    ]
                if left_letter == PSI and right_letter == PSIDAG:
                    return [
                        (left_factors + (prefix + (PSIDAG, PSI) + suffix,) + right_factors, -1.0 + 0.0j),
                        (left_factors + (prefix, suffix) + right_factors, 1.0 + 0.0j),
                    ]
        return None

    def canonicalize_monomial(self, monomial: MultiTraceMonomial) -> dict[MultiTraceMonomial, complex]:
        if monomial in self._canonical_cache:
            return self._canonical_cache[monomial]
        terms: dict[MultiTraceMonomial, complex] = {monomial: 1.0 + 0.0j}
        while True:
            changed = False
            updated: dict[MultiTraceMonomial, complex] = {}
            for current_monomial, coefficient in terms.items():
                rewritten = self._rewrite_once(current_monomial)
                if rewritten is None:
                    canonical_monomial, normalization = self._normalize_monomial(current_monomial)
                    new_value = updated.get(canonical_monomial, 0.0 + 0.0j) + coefficient * normalization
                    if abs(new_value) < 1e-13:
                        updated.pop(canonical_monomial, None)
                    else:
                        updated[canonical_monomial] = new_value
                    continue
                changed = True
                for rewritten_monomial, factor in rewritten:
                    new_value = updated.get(rewritten_monomial, 0.0 + 0.0j) + coefficient * factor
                    if abs(new_value) < 1e-13:
                        updated.pop(rewritten_monomial, None)
                    else:
                        updated[rewritten_monomial] = new_value
            terms = updated
            if not changed:
                break
        self._canonical_cache[monomial] = terms
        return terms

    def _build_canonical_universe(self) -> tuple[MultiTraceMonomial, ...]:
        universe: set[MultiTraceMonomial] = {tuple()}
        for word in self.raw_words_by_charge[0]:
            for monomial in self.canonicalize_monomial((word,)):
                if _monomial_charge(monomial) == 0 and _monomial_level(monomial) <= self.observable_level:
                    universe.add(monomial)
        return tuple(sorted(universe, key=lambda item: (float(_monomial_level(item)), len(item), item)))

    def _monomial_phase(self, monomial: MultiTraceMonomial) -> complex:
        return 1.0j if _monomial_p_count(monomial) % 2 else 1.0 + 0.0j

    def _const_expr_from_canonical_terms(self, terms: dict[MultiTraceMonomial, complex]) -> ConstExpr:
        expr: ConstExpr = {}
        for monomial, coefficient in terms.items():
            if _monomial_charge(monomial) != 0 or _monomial_level(monomial) > self.observable_level:
                continue
            if len(monomial) == 0:
                index = 0
                scaled = coefficient
            else:
                index = self.monomial_to_index[monomial]
                scaled = coefficient * self._monomial_phase(monomial)
            updated = expr.get(index, 0.0 + 0.0j) + scaled
            if abs(updated) < 1e-13:
                expr.pop(index, None)
            else:
                expr[index] = updated
        return expr

    @lru_cache(maxsize=None)
    def moment_expr_from_raw_word(self, word: RawWord) -> ConstExpr:
        if _raw_word_charge(word) != 0:
            return {}
        return self._const_expr_from_canonical_terms(self.canonicalize_monomial((word,)))

    def ordinary_entry_expr(self, left: RawWord, right: RawWord) -> ConstExpr:
        return self.moment_expr_from_raw_word(_raw_word_dagger(left) + right)

    def apply_hamiltonian(self, word: RawWord) -> dict[RawWord, PolyCoeff]:
        expr: dict[RawWord, PolyCoeff] = {}
        for index, letter in enumerate(word):
            prefix = word[:index]
            suffix = word[index + 1 :]
            if letter == X:
                contributions = {prefix + (P,) + suffix: _poly_const(-1.0j)}
            elif letter == P:
                contributions = {
                    prefix + (X,) + suffix: _poly_const(1.0j),
                    prefix + (X, X) + suffix: _poly_linear(3.0j),
                    prefix + (X, X, X) + suffix: _poly_quadratic(2.0j),
                    prefix + (PSIDAG, PSI) + suffix: _poly_linear(1.0j),
                    prefix + (PSI, PSIDAG) + suffix: _poly_linear(-1.0j),
                }
            elif letter == PSI:
                contributions = {
                    prefix + (PSI,) + suffix: _poly_const(-1.0 + 0.0j),
                    prefix + (X, PSI) + suffix: _poly_linear(-1.0 + 0.0j),
                    prefix + (PSI, X) + suffix: _poly_linear(-1.0 + 0.0j),
                }
            elif letter == PSIDAG:
                contributions = {
                    prefix + (PSIDAG,) + suffix: _poly_const(1.0 + 0.0j),
                    prefix + (X, PSIDAG) + suffix: _poly_linear(1.0 + 0.0j),
                    prefix + (PSIDAG, X) + suffix: _poly_linear(1.0 + 0.0j),
                }
            else:
                raise ValueError(f"unsupported letter {letter}")
            for mapped_word, coefficient in contributions.items():
                current = expr.get(mapped_word, _ZERO_POLY)
                updated = _poly_add(current, coefficient)
                if _poly_is_zero(updated):
                    expr.pop(mapped_word, None)
                else:
                    expr[mapped_word] = updated
        return expr

    def apply_shift_scaled_hamiltonian(self, word: RawWord, *, lambda_value: float) -> dict[RawWord, complex]:
        expr: dict[RawWord, complex] = {}
        for index, letter in enumerate(word):
            prefix = word[:index]
            suffix = word[index + 1 :]
            if letter == X:
                contributions = {prefix + (P,) + suffix: -1.0j}
            elif letter == P:
                contributions = {
                    prefix + (X,) + suffix: -0.5j * lambda_value,
                    prefix + (X, X, X) + suffix: 2.0j,
                    prefix + (PSIDAG, PSI) + suffix: 1.0j,
                    prefix + (PSI, PSIDAG) + suffix: -1.0j,
                }
            elif letter == PSI:
                contributions = {
                    prefix + (X, PSI) + suffix: -1.0 + 0.0j,
                    prefix + (PSI, X) + suffix: -1.0 + 0.0j,
                }
            elif letter == PSIDAG:
                contributions = {
                    prefix + (X, PSIDAG) + suffix: 1.0 + 0.0j,
                    prefix + (PSIDAG, X) + suffix: 1.0 + 0.0j,
                }
            else:
                raise ValueError(f"unsupported letter {letter}")
            for mapped_word, coefficient in contributions.items():
                updated = expr.get(mapped_word, 0.0 + 0.0j) + coefficient
                if abs(updated) < 1e-13:
                    expr.pop(mapped_word, None)
                else:
                    expr[mapped_word] = updated
        return expr

    def reality_constraints(self, *, seed_level: Fraction) -> list[PolyExpr]:
        constraints: list[PolyExpr] = []
        for word in self.raw_words_by_charge[0]:
            if _raw_word_level(word) > seed_level:
                continue
            expr: PolyExpr = {}
            _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(_raw_word_dagger(word)), 1.0)
            sign = -1.0 if _raw_word_p_count(word) % 2 == 0 else 1.0
            _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(word), sign)
            if expr:
                constraints.append(expr)
        return _deduplicate_polyexprs(constraints)

    def gauge_constraints(self, *, seed_level: Fraction, right: bool = False) -> list[PolyExpr]:
        constraints: list[PolyExpr] = []
        for word in self.raw_words_by_charge[0]:
            if _raw_word_level(word) > seed_level:
                continue
            expr: PolyExpr = {}
            if right:
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(word + (X, P)), 1.0j)
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(word + (P, X)), -1.0j)
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(word + (PSI, PSIDAG)), -1.0)
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(word + (PSIDAG, PSI)), -1.0)
            else:
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, P) + word), 1.0j)
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((P, X) + word), -1.0j)
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSI, PSIDAG) + word), -1.0)
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSIDAG, PSI) + word), -1.0)
            _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(word), 2.0 * self.n)
            if expr:
                constraints.append(expr)
        return _deduplicate_polyexprs(constraints)

    def eom_constraints(self, *, seed_level: Fraction) -> list[PolyExpr]:
        constraints: list[PolyExpr] = []
        for word in self.raw_words_by_charge[0]:
            if _raw_word_level(word) > seed_level:
                continue
            expr: PolyExpr = {}
            for mapped_word, coefficient in self.apply_hamiltonian(word).items():
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(mapped_word), coefficient)
            if expr:
                constraints.append(expr)
        return _deduplicate_polyexprs(constraints)

    def eom_constraints_shift_scaled(self, *, seed_level: Fraction, lambda_value: float) -> list[PolyExpr]:
        constraints: list[PolyExpr] = []
        for word in self.raw_words_by_charge[0]:
            if _raw_word_level(word) > seed_level:
                continue
            expr: PolyExpr = {}
            for mapped_word, coefficient in self.apply_shift_scaled_hamiltonian(word, lambda_value=lambda_value).items():
                _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(mapped_word), coefficient)
            if expr:
                constraints.append(expr)
        return _deduplicate_polyexprs(constraints)

    def objective_expr(self) -> PolyExpr:
        expr: PolyExpr = {}
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((P, P)), _poly_const(0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, X)), _poly_const(0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, X, X)), _poly_linear(1.0))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, X, X, X)), _poly_quadratic(0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSIDAG, PSI)), _poly_const(0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSI, PSIDAG)), _poly_const(-0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSIDAG, PSI, X)), _poly_linear(0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSI, PSIDAG, X)), _poly_linear(-0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, PSIDAG, PSI)), _poly_linear(0.5))
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, PSI, PSIDAG)), _poly_linear(-0.5))
        return expr

    def shift_scaled_objective_expr(self, *, lambda_value: float) -> PolyExpr:
        expr: PolyExpr = {}
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((P, P)), 0.5)
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, X, X, X)), 0.5)
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, X)), -0.25 * lambda_value)
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSIDAG, PSI, X)), 0.5)
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((PSI, PSIDAG, X)), -0.5)
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, PSIDAG, PSI)), 0.5)
        _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word((X, PSI, PSIDAG)), -0.5)
        return expr

    def ordinary_block_exprs(self) -> list[list[list[PolyExpr]]]:
        blocks: list[list[list[PolyExpr]]] = []
        for charge in (0, 1, -1, 2, -2):
            basis = self.basis_blocks[charge]
            block = []
            for left in basis:
                row = []
                for right in basis:
                    row.append(_polyexpr_from_constexpr(self.ordinary_entry_expr(left, right)))
                block.append(row)
            blocks.append(block)
        return blocks

    def ground_basis(self) -> tuple[RawWord, ...]:
        return (tuple(), (X,), (P,), (X, X), (PSIDAG, PSI))

    def ground_entry_expr(self, left: RawWord, right: RawWord) -> PolyExpr:
        expr: PolyExpr = {}
        for mapped_word, coefficient in self.apply_hamiltonian(right).items():
            _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(_raw_word_dagger(left) + mapped_word), coefficient)
        return expr

    def ground_entry_expr_shift_scaled(self, left: RawWord, right: RawWord, *, lambda_value: float) -> PolyExpr:
        expr: PolyExpr = {}
        for mapped_word, coefficient in self.apply_shift_scaled_hamiltonian(right, lambda_value=lambda_value).items():
            _polyexpr_add_constexpr(expr, self.moment_expr_from_raw_word(_raw_word_dagger(left) + mapped_word), coefficient)
        return expr

    def ground_block_exprs(self) -> list[list[list[PolyExpr]]]:
        basis = self.ground_basis()
        block: list[list[PolyExpr]] = []
        for left in basis:
            row: list[PolyExpr] = []
            for right in basis:
                row.append(self.ground_entry_expr(left, right))
            block.append(row)
        return [block]

    def ground_block_exprs_shift_scaled(self, *, lambda_value: float) -> list[list[list[PolyExpr]]]:
        basis = self.ground_basis()
        block: list[list[PolyExpr]] = []
        for left in basis:
            row: list[PolyExpr] = []
            for right in basis:
                row.append(self.ground_entry_expr_shift_scaled(left, right, lambda_value=lambda_value))
            block.append(row)
        return [block]


def _polyexpr_from_constexpr(expr: ConstExpr) -> PolyExpr:
    return {index: _poly_const(coefficient) for index, coefficient in expr.items()}


def _polyexpr_add(target: PolyExpr, index: int, coefficient: PolyCoeff) -> None:
    updated = _poly_add(target.get(index, _ZERO_POLY), coefficient)
    if _poly_is_zero(updated):
        target.pop(index, None)
    else:
        target[index] = updated


def _polyexpr_add_constexpr(target: PolyExpr, expr: ConstExpr, scale: complex | PolyCoeff) -> None:
    if isinstance(scale, tuple):
        scale_poly = scale
    else:
        scale_poly = _poly_const(scale)
    for index, coefficient in expr.items():
        _polyexpr_add(target, index, _poly_scale(scale_poly, coefficient))


def _deduplicate_polyexprs(expressions: list[PolyExpr]) -> list[PolyExpr]:
    unique: dict[tuple[tuple[int, PolyCoeff], ...], PolyExpr] = {}
    for expr in expressions:
        if not expr:
            continue
        key = tuple(sorted(expr.items(), key=lambda item: item[0]))
        unique[key] = expr
    return list(unique.values())


def _polyexpr_real(expr: PolyExpr, variables, alpha_linear, alpha_quadratic) -> Any:
    result = float(np.real(expr.get(0, _ZERO_POLY)[0])) + float(np.real(expr.get(0, _ZERO_POLY)[1])) * alpha_linear
    result = result + float(np.real(expr.get(0, _ZERO_POLY)[2])) * alpha_quadratic
    for index, coefficient in expr.items():
        if index == 0:
            continue
        term = (
            float(np.real(coefficient[0]))
            + float(np.real(coefficient[1])) * alpha_linear
            + float(np.real(coefficient[2])) * alpha_quadratic
        )
        result = result + term * variables[index - 1]
    return result


def _polyexpr_imag(expr: PolyExpr, variables, alpha_linear, alpha_quadratic) -> Any:
    result = float(np.imag(expr.get(0, _ZERO_POLY)[0])) + float(np.imag(expr.get(0, _ZERO_POLY)[1])) * alpha_linear
    result = result + float(np.imag(expr.get(0, _ZERO_POLY)[2])) * alpha_quadratic
    for index, coefficient in expr.items():
        if index == 0:
            continue
        term = (
            float(np.imag(coefficient[0]))
            + float(np.imag(coefficient[1])) * alpha_linear
            + float(np.imag(coefficient[2])) * alpha_quadratic
        )
        result = result + term * variables[index - 1]
    return result


def _build_real_psd_expression(cp, matrix_exprs: list[list[PolyExpr]], variables, alpha_linear, alpha_quadratic) -> Any:
    size = len(matrix_exprs)
    real_rows = [[0 for _ in range(size)] for _ in range(size)]
    imag_rows = [[0 for _ in range(size)] for _ in range(size)]
    for row in range(size):
        for column in range(size):
            real_rows[row][column] = _polyexpr_real(matrix_exprs[row][column], variables, alpha_linear, alpha_quadratic)
            imag_rows[row][column] = _polyexpr_imag(matrix_exprs[row][column], variables, alpha_linear, alpha_quadratic)
    block_rows = []
    for row in range(size):
        block_rows.append(real_rows[row] + [-entry for entry in imag_rows[row]])
    for row in range(size):
        block_rows.append(imag_rows[row] + real_rows[row])
    return cp.bmat(block_rows)


def _solver_order(cp, solver: str) -> tuple[str, ...]:
    if solver != "AUTO":
        return (solver,)
    installed = set(cp.installed_solvers())
    order = []
    for candidate in ("CLARABEL", "SCS", "CVXOPT"):
        if candidate in installed:
            order.append(candidate)
    return tuple(order)


def build_figure10_problem(
    reducer: Figure10Reducer,
    *,
    gauge_seed_level: Fraction,
    eom_seed_level: Fraction,
    reality_seed_level: Fraction,
    include_ground_block: bool = False,
) -> _Figure10Problem:
    cp = _import_cvxpy()
    variables = cp.Variable(len(reducer.canonical_universe))
    alpha_linear = cp.Parameter(nonneg=True, name="alpha_linear")
    alpha_quadratic = cp.Parameter(nonneg=True, name="alpha_quadratic")

    equality_exprs = []
    equality_exprs.extend(reducer.reality_constraints(seed_level=reality_seed_level))
    equality_exprs.extend(reducer.gauge_constraints(seed_level=gauge_seed_level, right=False))
    equality_exprs.extend(reducer.gauge_constraints(seed_level=gauge_seed_level, right=True))
    equality_exprs.extend(reducer.eom_constraints(seed_level=eom_seed_level))
    equality_exprs = _deduplicate_polyexprs(equality_exprs)

    ordinary_blocks = reducer.ordinary_block_exprs()
    constraints = []
    for expr in equality_exprs:
        real_expr = _polyexpr_real(expr, variables, alpha_linear, alpha_quadratic)
        imag_expr = _polyexpr_imag(expr, variables, alpha_linear, alpha_quadratic)
        if not isinstance(real_expr, float) or abs(real_expr) > 1e-13:
            constraints.append(real_expr == 0.0)
        if not isinstance(imag_expr, float) or abs(imag_expr) > 1e-13:
            constraints.append(imag_expr == 0.0)

    for block in ordinary_blocks:
        constraints.append(_build_real_psd_expression(cp, block, variables, alpha_linear, alpha_quadratic) >> 0)
    ground_blocks = reducer.ground_block_exprs() if include_ground_block else []
    for block in ground_blocks:
        constraints.append(_build_real_psd_expression(cp, block, variables, alpha_linear, alpha_quadratic) >> 0)

    objective_expr = reducer.objective_expr()
    objective = cp.Minimize(_polyexpr_real(objective_expr, variables, alpha_linear, alpha_quadratic))
    problem = cp.Problem(objective, constraints)
    return _Figure10Problem(
        cp=cp,
        problem=problem,
        alpha_linear=alpha_linear,
        alpha_quadratic=alpha_quadratic,
        variables=variables,
        ordinary_blocks=ordinary_blocks,
        ground_blocks=ground_blocks,
        equality_exprs=equality_exprs,
        objective_expr=objective_expr,
        energy_shift=0.0,
        energy_scale=1.0 / (reducer.n**2),
    )


def build_figure10_shift_scaled_problem(
    reducer: Figure10Reducer,
    *,
    g: float,
    gauge_seed_level: Fraction,
    eom_seed_level: Fraction,
    reality_seed_level: Fraction,
    include_ground_block: bool = False,
) -> _Figure10Problem:
    cp = _import_cvxpy()
    variables = cp.Variable(len(reducer.canonical_universe))
    alpha_linear = cp.Parameter(nonneg=True, name="alpha_linear")
    alpha_quadratic = cp.Parameter(nonneg=True, name="alpha_quadratic")
    alpha_linear.value = 0.0
    alpha_quadratic.value = 0.0

    alpha = float(g / math.sqrt(reducer.n))
    lambda_value = alpha ** (-4.0 / 3.0)

    equality_exprs = []
    equality_exprs.extend(reducer.reality_constraints(seed_level=reality_seed_level))
    equality_exprs.extend(reducer.gauge_constraints(seed_level=gauge_seed_level, right=False))
    equality_exprs.extend(reducer.gauge_constraints(seed_level=gauge_seed_level, right=True))
    equality_exprs.extend(reducer.eom_constraints_shift_scaled(seed_level=eom_seed_level, lambda_value=lambda_value))
    equality_exprs = _deduplicate_polyexprs(equality_exprs)

    ordinary_blocks = reducer.ordinary_block_exprs()
    constraints = []
    for expr in equality_exprs:
        real_expr = _polyexpr_real(expr, variables, alpha_linear, alpha_quadratic)
        imag_expr = _polyexpr_imag(expr, variables, alpha_linear, alpha_quadratic)
        if not isinstance(real_expr, float) or abs(real_expr) > 1e-13:
            constraints.append(real_expr == 0.0)
        if not isinstance(imag_expr, float) or abs(imag_expr) > 1e-13:
            constraints.append(imag_expr == 0.0)

    for block in ordinary_blocks:
        constraints.append(_build_real_psd_expression(cp, block, variables, alpha_linear, alpha_quadratic) >> 0)
    ground_blocks = reducer.ground_block_exprs_shift_scaled(lambda_value=lambda_value) if include_ground_block else []
    for block in ground_blocks:
        constraints.append(_build_real_psd_expression(cp, block, variables, alpha_linear, alpha_quadratic) >> 0)

    objective_expr = reducer.shift_scaled_objective_expr(lambda_value=lambda_value)
    objective = cp.Minimize(_polyexpr_real(objective_expr, variables, alpha_linear, alpha_quadratic))
    problem = cp.Problem(objective, constraints)
    return _Figure10Problem(
        cp=cp,
        problem=problem,
        alpha_linear=alpha_linear,
        alpha_quadratic=alpha_quadratic,
        variables=variables,
        ordinary_blocks=ordinary_blocks,
        ground_blocks=ground_blocks,
        equality_exprs=equality_exprs,
        objective_expr=objective_expr,
        energy_shift=1.0 / (32.0 * g * g),
        energy_scale=(alpha ** (2.0 / 3.0)) / (reducer.n**2),
    )


def _evaluate_polyexpr(expr: PolyExpr, values: np.ndarray, *, alpha: float) -> complex:
    result = expr.get(0, _ZERO_POLY)[0] + alpha * expr.get(0, _ZERO_POLY)[1] + alpha * alpha * expr.get(0, _ZERO_POLY)[2]
    for index, coefficient in expr.items():
        if index == 0:
            continue
        result = result + (coefficient[0] + alpha * coefficient[1] + alpha * alpha * coefficient[2]) * values[index - 1]
    return result


def _matrix_values(matrix_exprs: list[list[PolyExpr]], values: np.ndarray, *, alpha: float) -> np.ndarray:
    size = len(matrix_exprs)
    matrix = np.zeros((size, size), dtype=complex)
    for row in range(size):
        for column in range(size):
            matrix[row, column] = _evaluate_polyexpr(matrix_exprs[row][column], values, alpha=alpha)
    return 0.5 * (matrix + matrix.conj().T)


def _max_psd_residual(matrices: list[np.ndarray]) -> float:
    residuals: list[float] = []
    for matrix in matrices:
        norm = max(1.0, float(np.linalg.norm(matrix, ord=2)))
        minimum = float(np.linalg.eigvalsh(matrix).min())
        residuals.append(max(0.0, -minimum) / norm)
    return max(residuals, default=0.0)


def _max_eq_residual(expressions: list[PolyExpr], values: np.ndarray, *, alpha: float) -> float:
    residual = 0.0
    scale = max(1.0, float(np.linalg.norm(values, ord=np.inf)))
    for expr in expressions:
        value = _evaluate_polyexpr(expr, values, alpha=alpha)
        residual = max(residual, abs(value) / scale)
    return residual


def solve_figure10_point(
    problem_data: _Figure10Problem,
    *,
    reducer: Figure10Reducer,
    g: float,
    solver: str = "AUTO",
    solver_eps: float = 1e-7,
    solver_max_iters: int = 100000,
    eq_tolerance: float = 2e-6,
    psd_tolerance: float = 2e-6,
) -> Figure10SolveResult:
    alpha = float(g / math.sqrt(reducer.n))
    problem_data.alpha_linear.value = alpha
    problem_data.alpha_quadratic.value = alpha * alpha

    best = Figure10SolveResult(
        g=g,
        alpha=alpha,
        status="solver_not_run",
        objective_value=None,
        energy_density=None,
        eq_residual=None,
        psd_residual=None,
        feasible=False,
    )

    for solver_name in _solver_order(problem_data.cp, solver):
        solve_kwargs: dict[str, Any] = {"solver": solver_name, "warm_start": True, "verbose": False}
        if solver_name == "SCS":
            solve_kwargs["eps"] = solver_eps
            solve_kwargs["max_iters"] = solver_max_iters
        elif solver_name == "CLARABEL":
            solve_kwargs["max_iter"] = solver_max_iters
        try:
            problem_data.problem.solve(**solve_kwargs)
        except Exception:
            best = Figure10SolveResult(
                g=g,
                alpha=alpha,
                status=f"{solver_name.lower()}_failed",
                objective_value=None,
                energy_density=None,
                eq_residual=None,
                psd_residual=None,
                feasible=False,
            )
            continue

        status = str(problem_data.problem.status)
        if status not in {"optimal", "optimal_inaccurate"} or problem_data.variables.value is None:
            best = Figure10SolveResult(
                g=g,
                alpha=alpha,
                status=status,
                objective_value=None,
                energy_density=None,
                eq_residual=None,
                psd_residual=None,
                feasible=False,
            )
            continue

        values = np.asarray(problem_data.variables.value, dtype=float)
        eq_residual = _max_eq_residual(problem_data.equality_exprs, values, alpha=alpha)
        matrices = [_matrix_values(block, values, alpha=alpha) for block in problem_data.ordinary_blocks]
        matrices.extend(_matrix_values(block, values, alpha=alpha) for block in problem_data.ground_blocks)
        psd_residual = _max_psd_residual(matrices)
        objective_value = float(problem_data.problem.value) if problem_data.problem.value is not None else None
        feasible = (
            objective_value is not None
            and eq_residual <= eq_tolerance
            and psd_residual <= psd_tolerance
        )
        result = Figure10SolveResult(
            g=g,
            alpha=alpha,
            status=status,
            objective_value=objective_value,
            energy_density=None
            if objective_value is None
            else problem_data.energy_shift + problem_data.energy_scale * objective_value,
            eq_residual=eq_residual,
            psd_residual=psd_residual,
            feasible=feasible,
        )
        if feasible:
            return result
        best = result
    return best


def fit_figure10_power_law(g_values: np.ndarray, energy_density: np.ndarray, *, min_g: float) -> tuple[float, float, float]:
    mask = np.isfinite(energy_density) & (energy_density > 0.0) & (g_values >= min_g)
    if np.count_nonzero(mask) < 2:
        raise ValueError("not enough points for a power-law fit")
    x_values = np.log(g_values[mask])
    y_values = np.log(energy_density[mask])
    slope, intercept = np.polyfit(x_values, y_values, 1)
    return float(slope), float(intercept), float(np.exp(intercept))


def _import_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as pyplot

    return pyplot


def plot_figure10(
    g_values: np.ndarray,
    energy_density: np.ndarray,
    *,
    fit_min_g: float,
    out_path: str | Path,
) -> tuple[float, float, float]:
    pyplot = _import_matplotlib()
    slope, intercept, kappa = fit_figure10_power_law(g_values, energy_density, min_g=fit_min_g)
    curve_grid = np.geomspace(1.0, 600.0, 400)
    fit_curve = kappa * np.power(curve_grid, slope)
    paper_curve = 0.196 * np.power(curve_grid, 2.0 / 3.0)

    figure, axes = pyplot.subplots(1, 2, figsize=(11.2, 4.8))
    linear_axis, log_axis = axes

    linear_mask = np.isfinite(energy_density)
    linear_axis.plot(g_values[linear_mask], energy_density[linear_mask], color="#1f77b4", marker="o", linewidth=1.9)
    linear_axis.plot(curve_grid, fit_curve, color="#4c78a8", linewidth=2.2, linestyle="--")
    linear_axis.plot(curve_grid, paper_curve, color="#9c755f", linewidth=1.9, linestyle=":")
    linear_axis.set_xlabel(r"$g$")
    linear_axis.set_ylabel(r"$E_{\rm lb}/N^2$")
    linear_axis.set_title(r"Figure 10: large-$g$ lower bound")
    linear_axis.set_xlim(1.0, 600.0)
    linear_axis.grid(True, alpha=0.22)

    log_axis.loglog(g_values[linear_mask], energy_density[linear_mask], color="#1f77b4", marker="o", linewidth=1.9)
    log_axis.loglog(curve_grid, fit_curve, color="#4c78a8", linewidth=2.2, linestyle="--")
    log_axis.loglog(curve_grid, paper_curve, color="#9c755f", linewidth=1.9, linestyle=":")
    log_axis.set_xlabel(r"$g$")
    log_axis.set_ylabel(r"$E_{\rm lb}/N^2$")
    log_axis.set_title(r"log-log fit")
    log_axis.grid(True, alpha=0.22, which="both")

    fit_label = rf"fit: ${kappa:.3f}\, g^{{{slope:.3f}}}$"
    paper_label = r"paper guide: $0.196\, g^{2/3}$"
    linear_axis.legend(["bootstrap", fit_label, paper_label], loc="upper left")
    log_axis.legend(["bootstrap", fit_label, paper_label], loc="upper left")

    figure.tight_layout()
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=190)
    pyplot.close(figure)
    return slope, intercept, kappa


def run_figure10_scan(*, out_dir: str | Path, config: Figure10Config | None = None) -> Path:
    if config is None:
        config = Figure10Config()

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reducer = Figure10Reducer(
        n=config.n,
        basis_level=config.basis_level,
        observable_level=config.observable_level,
        universe_source_level=config.universe_source_level,
    )
    raw_problem = build_figure10_problem(
        reducer,
        gauge_seed_level=config.gauge_seed_level,
        eom_seed_level=config.eom_seed_level,
        reality_seed_level=config.reality_seed_level,
        include_ground_block=config.include_ground_block,
    )

    results: list[Figure10SolveResult] = []
    for g in config.g_values:
        if g >= config.shift_scale_min_g:
            problem_data = build_figure10_shift_scaled_problem(
                reducer,
                g=g,
                gauge_seed_level=config.gauge_seed_level,
                eom_seed_level=config.eom_seed_level,
                reality_seed_level=config.reality_seed_level,
                include_ground_block=config.include_ground_block,
            )
        else:
            problem_data = raw_problem
        results.append(
            solve_figure10_point(
                problem_data,
                reducer=reducer,
                g=g,
                solver=config.solver,
                solver_eps=config.solver_eps,
                solver_max_iters=config.solver_max_iters,
                eq_tolerance=config.eq_tolerance,
                psd_tolerance=config.psd_tolerance,
            )
        )

    g_values = np.asarray(config.g_values, dtype=float)
    energy_density = np.array(
        [result.energy_density if result.feasible and result.energy_density is not None else np.nan for result in results],
        dtype=float,
    )
    slope = np.nan
    intercept = np.nan
    kappa = np.nan
    if np.count_nonzero(np.isfinite(energy_density)) >= 2:
        slope, intercept, kappa = plot_figure10(
            g_values,
            energy_density,
            fit_min_g=config.fit_min_g,
            out_path=output_dir / "figure10_largeg.png",
        )

    with (output_dir / "bounds.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["g", "alpha", "status", "objective", "energy_density", "eq_residual", "psd_residual", "feasible"],
        )
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "g": result.g,
                    "alpha": result.alpha,
                    "status": result.status,
                    "objective": result.objective_value if result.objective_value is not None else np.nan,
                    "energy_density": result.energy_density if result.energy_density is not None else np.nan,
                    "eq_residual": result.eq_residual if result.eq_residual is not None else np.nan,
                    "psd_residual": result.psd_residual if result.psd_residual is not None else np.nan,
                    "feasible": int(result.feasible),
                }
            )

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config.to_json(), handle, indent=2)

    summary_lines = [
        "# Figure 10 summary",
        "",
        "- Model: cubic matrix SUSY QM lower bound at large g with N=100.",
        "- Coupling in the solver: `alpha = g / sqrt(N)`.",
        f"- Shift-scale activated for `g >= {config.shift_scale_min_g}`.",
        "- Ordinary PSD blocks: fermion-number `20, 8, 8, 4, 4`.",
        f"- Canonical multi-trace universe size: `{len(reducer.canonical_universe)}`.",
        f"- Equality constraints (raw builder): `{len(raw_problem.equality_exprs)}`.",
        f"- Ground-state thermal block enabled: `{config.include_ground_block}`.",
        f"- Fit region: `g >= {config.fit_min_g}`.",
        f"- Fit slope: `{slope:.6f}`." if np.isfinite(slope) else "- Fit slope: `nan`.",
        f"- Fit intercept: `{intercept:.6f}`." if np.isfinite(intercept) else "- Fit intercept: `nan`.",
        f"- Fit kappa: `{kappa:.6f}`." if np.isfinite(kappa) else "- Fit kappa: `nan`.",
        "",
        "## Checksums",
        f"- Block dimensions: `{reducer.block_dimensions()}`",
        f"- Charge-zero raw words with level <= 8: `{len(reducer.raw_words_by_charge[0])}`",
    ]
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return output_dir / "figure10_largeg.png"
