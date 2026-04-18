# Figure 10 guidance 1

## Target

- Reproduce Figure 10 for the cubic matrix SUSY QM large-\(g\) lower bound.
- Plot \(E_{\rm lb}(g;N=100)/N^2\) versus \(g\), with \(g \in [1,600]\).
- Use the rescaled solver coupling
  \[
  \alpha = \frac{g}{\sqrt{N}}.
  \]

## Hamiltonian and objective

\[
H=\frac12 \,\tr\!\Big(
P^2+X^2+2\alpha X^3+\alpha^2 X^4+[\Psi^\dagger,\Psi]
+\alpha\big([\Psi^\dagger,\Psi]X+X[\Psi^\dagger,\Psi]\big)
\Big).
\]

Linear SDP objective:
\[
\begin{aligned}
E
={}&\frac12\,\langle \tr P^2\rangle
+\frac12\,\langle \tr X^2\rangle
+\alpha\,\langle \tr X^3\rangle
+\frac{\alpha^2}{2}\,\langle \tr X^4\rangle \\
&+\frac12\Big(\langle \tr \Psi^\dagger\Psi\rangle-\langle \tr \Psi\Psi^\dagger\rangle\Big) \\
&+\frac{\alpha}{2}\Big(
\langle \tr \Psi^\dagger\Psi X\rangle
-\langle \tr \Psi\Psi^\dagger X\rangle
+\langle \tr X\Psi^\dagger\Psi\rangle
-\langle \tr X\Psi\Psi^\dagger\rangle
\Big).
\end{aligned}
\]

## Basis and levels

- Raw untraced matrix words, not cyclically reduced.
- Levels:
  \[
  \ell(X)=1,\quad \ell(P)=2,\quad \ell(\Psi)=\ell(\Psi^\dagger)=\frac32.
  \]
- Level-8 bootstrap means basis words with level \(\le 4\).
- Fermion-number block sizes must be
  \[
  20,\ 8,\ 8,\ 4,\ 4.
  \]

## Main constraints

Only use:

1. fermion number symmetry,
2. reality,
3. gauge symmetry,
4. Heisenberg equations of motion,
5. ordinary PSD blocks.

Supercharge constraints are redundant at this level. The small ground-state thermal block is optional and mainly a stabilizer.

## Raw-word rules

### No cyclic canonicalization

Moments are
\[
m[w]=\langle \tr w\rangle
\]
for raw matrix-valued words \(w\). Do not identify \(AB\) with \(BA\).

### Gauge

\[
G=i[X,P]-(\Psi\Psi^\dagger+\Psi^\dagger\Psi)+2NI,
\]
\[
i\,m[XP\,O]-i\,m[PX\,O]-m[\Psi\Psi^\dagger O]-m[\Psi^\dagger\Psi O]+2N\,m[O]=0.
\]

### EOM letters

\[
[H,X]=-iP,
\]
\[
[H,P]=i\Big(X+3\alpha X^2+2\alpha^2 X^3+\alpha(\Psi^\dagger\Psi-\Psi\Psi^\dagger)\Big),
\]
\[
[H,\Psi]=-\Psi-\alpha(X\Psi+\Psi X),
\]
\[
[H,\Psi^\dagger]=\Psi^\dagger+\alpha(X\Psi^\dagger+\Psi^\dagger X).
\]

Apply Leibniz:
\[
[H,w]=\sum_r \cdots [H,A_r]\cdots
\]
and impose
\[
m([H,w])=0.
\]

## Expected large-\(g\) fit

Paper target:
\[
\frac{E_{\rm lb}}{N^2}\sim 0.196\, g^{2/3}.
\]

Useful checks:

- block dimensions \(20,8,8,4,4\),
- large-\(g\) tail slope near \(0.667\),
- \(N=100\) and \(N=1000\) nearly overlap after dividing by \(N^2\).
