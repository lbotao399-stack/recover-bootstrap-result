# Phase 1 theory input

Status: confirmed for implementation.

## Core formulas

- SUSY algebra:
  `\{Q, Q^\dagger\} = 2H`
- Supercharges:
  `Q = (p + i W'(x)) psi`
  `Q^\dagger = (p - i W'(x)) psi^\dagger`
- Sector Hamiltonian:
  `H = 1/2 p^2 + 1/2 W'(x)^2 + 1/2 epsilon W''(x)`, `epsilon = +/- 1`
- Zero-mode equations:
  `psi_B' + W' psi_B = 0`
  `psi_F' - W' psi_F = 0`
- Zero modes:
  `psi_B = exp(-W)`
  `psi_F = exp(W)`

## 1D bootstrap recursion

For arbitrary polynomial `W`,

`8 t E <x^(t-1)> + t(t-1)(t-2)<x^(t-3)>`
`- 4t <x^(t-1) W'^2> - 4t epsilon <x^(t-1) W''>`
`- 4 <x^t W'' W'> - 2 epsilon <x^t W'''> = 0`

This is the M1 implementation target.

## SHO specialization

For `W = 1/2 omega x^2`,

`E = |omega|(n + 1/2) + 1/2 epsilon omega`

and the recursion becomes

`(t-3)(t-2)(t-1)<x^(t-4)>`
`+ 4(t-1)(2E - epsilon omega)<x^(t-2)>`
`- 4t omega^2 <x^t> = 0`

## Benchmarks to lock in tests

- unique zero-energy ground state
- paired positive-energy spectrum
- for `omega > 0`, the normalizable zero mode is the bosonic one

