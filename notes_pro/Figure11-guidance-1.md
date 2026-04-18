# Figure 11 guidance 1

## Core statement

Figure 11 is the same matrix cubic level-8 bootstrap as Figure 10, but scanned in the small-\(g\) / critical window and, in the paper, solved with SDPB throughout.

## Shared data with Figure 10

- Model:
  \[
  W(X)=\frac12 \operatorname{Tr} X^2+\frac{\lambda}{3}\operatorname{Tr} X^3,
  \qquad
  \lambda=\frac{g}{\sqrt{N}},
  \qquad
  N=100.
  \]
- Hamiltonian:
  \[
  H=\frac12 \operatorname{tr}\!\Big(
  P^2+X^2+2\lambda X^3+\lambda^2 X^4+[\Psi^\dagger,\Psi]
  +\lambda\big([\Psi^\dagger,\Psi]X+X[\Psi^\dagger,\Psi]\big)
  \Big).
  \]
- Basis/level assignment:
  \[
  \ell(X)=1,\quad \ell(P)=2,\quad \ell(\Psi)=\ell(\Psi^\dagger)=\frac32,
  \]
  all matrix-valued words up to level \(4\), split by fermion number into blocks
  \[
  20,\ 8,\ 8,\ 4,\ 4.
  \]
- Constraints:
  fermion number symmetry, reality, gauge symmetry, and Heisenberg/EOM.

## Figure 11-specific data

- Bosonic critical coupling:
  \[
  g_c^2=\frac{1}{6\sqrt3},
  \qquad
  g_c\approx 0.3102016197.
  \]
- Kink point:
  \[
  g_0=\sqrt2\,g_c=\frac{1}{\sqrt{3\sqrt3}}
  \approx 0.4386913377.
  \]
- The paper plot shows:
  - black dots: lower bounds on a small-\(g\) grid;
  - red dot: the single point at \(g_0\).

## Numerical guidance

- Keep the same level-8 matrix cubic compiler/exporter as Figure 10.
- Scan densely near \(g_0\).
- Use SDPB for the final lower bounds; the paper caption states that all Figure 11 bounds were obtained with SDPB.
- Store both
  \[
  e(g)=E(g)/N^2
  \quad\text{and}\quad
  E(g)=N^2 e(g),
  \]
  but plot raw \(E_{\rm lower}(g)\) for Figure 11.
