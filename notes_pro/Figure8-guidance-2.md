## Figure 8 guidance 2

- Primitive variables are raw single-trace word moments
  - `mu(w) = < tr w >`
  - with `w` built from `X, P, Psi, Psi^\dagger`

- Do not cyclically identify words.
  - `XP` and `PX` are distinct.

- Fermion-number grading:
  - nonzero net fermion number moments vanish
  - the paper basis splits into bosonic and fermionic blocks

- For the quadratic warmup, the practical route is:
  - use raw-word closure
  - use vacuum constraints instead of scalar Heisenberg recursion
  - build PSD blocks from concatenations `u^\dagger v`
  - build ground-state PSD blocks from `u^\dagger [H, v]`

- Minimal analytic sanity checks:
  - `m(P^2) = a^2 m(X^2)`
  - `m(XP) + m(PX) = 0`
  - with gauge:
    - `m(XP) - m(PX) = i N^2`
  - therefore
    - `m(XP) = i N^2 / 2`
    - `m(PX) = -i N^2 / 2`
    - `m(X^2) >= N^2 / (2 a)`

- For `a = 1` the exact lines to overlay are
  - `x2_exact(N) = N^2 / 2`
  - `x4_exact(N) = N (2 N^2 + 1) / 4`
