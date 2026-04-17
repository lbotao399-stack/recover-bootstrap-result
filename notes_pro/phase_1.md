# Phase 1 theory input

Status: confirmed for implementation.

## Sector Hamiltonian

Fix one fermion sector:

`H = 1/2 p^2 + 1/2 W'(x)^2 + 1/2 epsilon W''(x)`, with `epsilon = +/- 1`.

Write moments as `m_n = <x^n>`.

Zero-mode equations:

- `psi_B' + W' psi_B = 0`
- `psi_F' - W' psi_F = 0`

so

- `psi_B = exp(-W)`
- `psi_F = exp(W)`

For polynomial `W`, odd leading degree gives no normalizable zero mode. Even leading degree gives one normalizable sector, fixed by the sign of the leading coefficient.

## General 1D recursion

Use

- `[x, p] = i`
- `[p, x^n] = -i n x^(n-1)`

which gives

`[p^2, x^n] = -2 i n x^(n-1) p - n(n-1) x^(n-2)`.

From `<[H, x^(t+1)]> = 0`,

`<x^t p> = i t/2 m_(t-1)`.

From `<[H, x^t p]> = 0`,

`4 t <x^(t-1) p^2> = -t(t-1)(t-2) m_(t-3) + 4 <x^t W'' W'> + 2 epsilon <x^t W'''>`.

From `<x^(t-1) H> = E m_(t-1)`,

`<x^(t-1) p^2> = 2 E m_(t-1) - <x^(t-1) W'^2> - epsilon <x^(t-1) W''>`.

Combining them gives the recursion

`8 t E m_(t-1) + t(t-1)(t-2) m_(t-3)`
`- 4 t <x^(t-1) W'^2> - 4 t epsilon <x^(t-1) W''>`
`- 4 <x^t W'' W'> - 2 epsilon <x^t W'''> = 0`.

This is the paper's 1D recursion.

## Polynomial reduction to moments

For

`W(x) = sum_(r=0)^d a_r x^r`,

we have

- `W' = sum_(r=1)^d r a_r x^(r-1)`
- `W'' = sum_(r=2)^d r(r-1) a_r x^(r-2)`
- `W''' = sum_(r=3)^d r(r-1)(r-2) a_r x^(r-3)`

and therefore

- `<x^(t-1) W'^2> = sum_(r,s=1)^d r s a_r a_s m_(t+r+s-3)`
- `<x^(t-1) W''> = sum_(r=2)^d r(r-1) a_r m_(t+r-3)`
- `<x^t W'' W'> = sum_(r=2)^d sum_(s=1)^d r(r-1) s a_r a_s m_(t+r+s-3)`
- `<x^t W'''> = sum_(r=3)^d r(r-1)(r-2) a_r m_(t+r-3)`

So for fixed `E`, the recursion is linear in the moments `m_n`. If `E` is also variable, the term `E m_(t-1)` makes the problem bilinear.

## Minimal moment SDP

For truncation `K`, define the Hankel moment matrix

`M^(K)_(ij) = m_(i+j)`, `i, j = 0, ..., K`.

Positivity of `<f^dagger f>` for any polynomial `f(x)` implies

`M^(K) >= 0`.

The minimal fixed-energy feasibility problem is

- `m_0 = 1`
- `M^(K) >= 0`
- `R_t(m; E) = 0`

with `R_t` given by the recursion above.

Scanning `E` then produces feasible windows that converge to the spectrum.

## SHO specialization

For `W = (omega/2) x^2`,

- `W' = omega x`
- `W'' = omega`
- `W''' = 0`

and the spectrum is

`E = |omega| (n + 1/2) + 1/2 epsilon omega`.

The recursion reduces to

`(t-3)(t-2)(t-1) m_(t-4) + 4 (t-1) (2E - epsilon omega) m_(t-2) - 4 t omega^2 m_t = 0`,

after the shift `t -> t - 1` needed to match the paper's displayed SHO convention.

## Benchmarks

- unique zero-energy ground state
- paired positive-energy spectrum
- for `omega > 0`, the normalizable zero mode is bosonic
