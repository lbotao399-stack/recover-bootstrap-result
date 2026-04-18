# Figure 10 guidance 2

## Core numerical advice

Figure 10 should inherit the raw-word / cut-and-join matrix machinery rather than the older cyclic traced-word scaffold.

\[
\text{Figure 10} \leftarrow \text{Figure 8 raw-word + cut-and-join engine}.
\]

The main algebraic checks are:

\[
44\ \text{basis words}
\;\to\;
143\ \text{canonical observables}
\;\to\;
27\ \text{real free variables after reality + gauge + EOM}.
\]

These numbers are paper-side checksums; mismatches should be logged explicitly.

## Multi-trace closure

Canonical observables must allow multi-trace monomials
\[
\mu[w_1|\cdots|w_k]
=
\Big\langle \prod_r \tr w_r\Big\rangle,
\]
not only single trace.

Local cut-and-join rules:

\[
\mu[U,P,X,V\mid \mathcal R]
=
\mu[U,X,P,V\mid \mathcal R]
-i\,\mu[U\mid V\mid \mathcal R],
\]
\[
\mu[U,\Psi,\Psi^\dagger,V\mid \mathcal R]
=
-\mu[U,\Psi^\dagger,\Psi,V\mid \mathcal R]
+\mu[U\mid V\mid \mathcal R].
\]

This closure is the matrix analogue of the split sector already needed for Figure 8.

## Reality and variable reduction

For a real ground-state convention,
\[
\mu[\mathcal M]^*=(-1)^{n_P(\mathcal M)}\mu[\mathcal M].
\]
Hence:

- even \(n_P\): real variable,
- odd \(n_P\): purely imaginary variable.

This should be used to reduce all SDP variables to real unknowns before realifying the Hermitian PSD blocks.

## Large-\(g\) strong-coupling rescaling

An optional stabilizing change of variables is
\[
X=-\frac{1}{2\alpha}I+\alpha^{-1/3}Z,
\qquad
P=\alpha^{1/3}Q,
\qquad
\alpha=\frac{g}{\sqrt N}.
\]
Then
\[
H=\frac{N}{32\alpha^2}
+\frac{\alpha^{2/3}}{2}\,
\tr\!\left(
Q^2+Z^4+([\Psi^\dagger,\Psi]Z+Z[\Psi^\dagger,\Psi])
-\frac12\alpha^{-4/3}Z^2
\right).
\]

This is not mandatory for a first Figure 10 run, but it is the natural cure when very-large-\(g\) points drift while the algebraic structure is already correct.

## Practical strategy

- Keep the first implementation on the original \((X,P,\Psi,\Psi^\dagger)\) variables.
- Saturate reality, gauge, and EOM only as far as the solver remains tractable.
- Treat the small \(5\times 5\) thermal block as a stabilizer, not as the main source of the lower bound.
- Log explicitly when large-\(g\) points fail because of compilation or solver instability rather than algebraic inconsistency.

## Reference tail

Use the paper guide curve
\[
\frac{E_{\rm lb}}{N^2}\approx 0.196\, g^{2/3}
\]
as the large-\(g\) comparison line on the plot.
