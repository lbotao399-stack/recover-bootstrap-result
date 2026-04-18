# Figure 10 guidance 3: shift-scale for large g

## Motivation

For the matrix cubic large-\(g\) regime, the original \((X,P)\) formulation develops visible numerical drift even when the raw-word / multi-trace closure is already correct. The recommended cure is the same kind of shift-scale used in the 1d cubic problem.

This should be activated once \(g\) is large enough; in practice the current project uses it from about \(g \ge 20\).

## Rescaling

Let
\[
\alpha = \frac{g}{\sqrt N}.
\]

Then perform
\[
X=-\frac{1}{2\alpha} I + \alpha^{-1/3} Z,
\qquad
P=\alpha^{1/3} Q.
\]

## Shifted/scaled Hamiltonian

After the change of variables,
\[
H=\frac{N}{32\alpha^2}
+\frac{\alpha^{2/3}}{2}\,
\tr\!\left(
Q^2+Z^4+([\Psi^\dagger,\Psi]Z+Z[\Psi^\dagger,\Psi])
-\frac12 \alpha^{-4/3} Z^2
\right).
\]

Equivalently,
\[
H=\frac{N^2}{32 g^2}+\alpha^{2/3}\,\widehat H_\lambda,
\qquad
\lambda=\alpha^{-4/3}.
\]

The strong-coupling leading Hamiltonian is therefore the matrix version of the \(g^{2/3}\) large-coupling scaling already seen in the 1d cubic case.

## Scaled letter-level EOM

Using the shifted/scaled Hamiltonian, one can impose
\[
[\widehat H_\lambda, Z]=-iQ,
\]
\[
[\widehat H_\lambda, Q]
=
i\left(2 Z^3-\frac{\lambda}{2}Z+[\Psi^\dagger,\Psi]\right),
\]
\[
[\widehat H_\lambda,\Psi]=-(Z\Psi+\Psi Z),
\]
\[
[\widehat H_\lambda,\Psi^\dagger]=Z\Psi^\dagger+\Psi^\dagger Z.
\]

The gauge operator is unchanged in form:
\[
G=i[Z,Q]-(\Psi\Psi^\dagger+\Psi^\dagger\Psi)+2NI.
\]

## Scaled objective

The reduced objective is
\[
\widehat E=
\frac12\langle \tr Q^2\rangle
+\frac12\langle \tr Z^4\rangle
-\frac{\lambda}{4}\langle \tr Z^2\rangle
+\frac12\Big(
\langle \tr \Psi^\dagger\Psi Z\rangle
-\langle \tr \Psi\Psi^\dagger Z\rangle
+\langle \tr Z\Psi^\dagger\Psi\rangle
-\langle \tr Z\Psi\Psi^\dagger\rangle
\Big).
\]

Then the physical energy density is reconstructed as
\[
\frac{E}{N^2}
=
\frac{1}{32 g^2}
+\frac{\alpha^{2/3}}{N^2}\,\widehat E.
\]

## Practical use in the code

- Keep the same raw-word, non-cyclic, multi-trace cut-and-join closure.
- Keep the same fermion-number blocks and gauge constraints.
- For \(g < 20\), the original Hamiltonian is still acceptable.
- For \(g \ge 20\), solve the shifted/scaled SDP and convert back to the physical energy density only at the end.

This keeps the coefficients entering the EOM and objective at \(O(1)\) in the large-\(g\) regime and directly targets the tail behavior relevant to Figure 10.
