# Figure 11 guidance 2

## Core statement

Figure 11 is not a new algebra problem. It is the same level-8 matrix cubic SDP as Figure 10, with the scan moved to the critical window near \(g_0\), and solved as an SDPB lower-bound feasibility/bisection problem.

## Required structure

- Use the same raw non-cyclic matrix-word basis as Figure 10.
- Do not revert to cyclic single-trace canonicalization.
- Do not impose large-\(N\) factorization.
- Keep multi-trace observables in the variable set.
- Ordinary PSD only is enough for the first pass; the small thermal block is optional.

## Small-g / critical-window scan

Recommended three-part grid:

1. Left coarse points:
   \[
   g\in\{0.20,0.25,0.30,0.34,0.38,0.40,0.42,0.43\}.
   \]
2. Red point:
   \[
   g=g_0,\qquad
   g_0=\sqrt2\,g_c.
   \]
3. Right dense points:
   \[
   \delta=g-g_0,\qquad
   \delta\in\{10^{-4},3\cdot10^{-4},10^{-3},3\cdot10^{-3},10^{-2},3\cdot10^{-2},10^{-1}\},
   \]
   together with a few larger points such as \(0.50,0.55,0.60,0.70,0.80\).

## Expected behavior

- For \(g<g_0\), the lower bound should stay numerically near zero.
- At \(g=g_0\), the highlighted red point should also be near zero.
- For \(g>g_0\), the lower bound should lift off with a kink and feed naturally into the Figure 10 branch.

## Implementation note

The algebraic checksum should stay the Figure 10 one:

\[
44\ \text{basis words}
\;\to\;
(20,8,8,4,4)\ \text{blocks}
\;\to\;
27\ \text{real free variables after constraints}.
\]

The new work is the scan management and SDPB precision handling, not a new symbolic reduction.
