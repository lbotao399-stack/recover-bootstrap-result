# Figure 5 guidance 2

## Secondary strategy

This guidance keeps the same cubic model and same ordinary lower-bound problem, but emphasizes a more conservative small-\(g\) numerical strategy.

## Same model

\[
W(x)=\frac12x^2+g\frac13x^3,
\qquad
H=\frac12p^2+\frac12g^2x^4+gx^3+\frac12x^2-gx-\frac12
\]

in the \(\epsilon=-1\) sector.

## Same global recursion

The same original-variable recursion should be used:

\[
\begin{aligned}
P_{a,b}
={}&(2E+1)P_{a-2,b}
+2g\,P_{a-2,b+1}
-P_{a-2,b+2}
-2g\,P_{a-2,b+3}
-g^2P_{a-2,b+4}
\\
&+2ig(a-2)P_{a-3,b}
-2i(a-2)P_{a-3,b+1}
-6ig(a-2)P_{a-3,b+2}
-4ig^2(a-2)P_{a-3,b+3}
\\
&+(a-2)(a-3)P_{a-4,b}
+6g(a-2)(a-3)P_{a-4,b+1}
+6g^2(a-2)(a-3)P_{a-4,b+2}
\\
&+2ig(a-2)(a-3)(a-4)P_{a-5,b}
+4ig^2(a-2)(a-3)(a-4)P_{a-5,b+1}
\\
&-g^2(a-2)(a-3)(a-4)(a-5)P_{a-6,b}.
\end{aligned}
\]

## Practical small-g interpretation

The emphasis is:

- do not use the Figure 4 large-\(g\) shift-scaling for Figure 5
- work directly in the original variables
- accept that very small \(g\) will be numerically unstable
- treat \(g \lesssim 0.35\) as a likely breakdown region for the local solver stack

## Canonical basis

The same canonical reduced basis is recommended:

\[
\mathcal B_L^{\rm can}=\{p^a x^b:\ 2a+b\le L\}.
\]

## x^2 branch

The right panel should be computed in two stages:

1. find \(E_L^{\rm lb}(g)\)
2. keep \(E\) near the lower-energy branch, rather than minimizing \(m_2\) over all energies blindly

A practical version is

\[
E \le E_L^{\rm lb}(g)+\delta
\]

with small \(\delta\), or a restricted scan in

\[
[E_L^{\rm lb}(g),E^{\rm ub}(g)].
\]

## Continuation guidance

Numerically, start from larger \(g\) and continue downward. Warm-start both in \(g\) and across nearby energies. The hierarchy should satisfy

\[
E_{\rm low}^{(7)}\le E_{\rm low}^{(8)}\le E_{\rm low}^{(9)}\le E_{\rm low}^{(10)}.
\]

The \(x^2\) lower branch should show a similar tightening trend. Any clear inversion is a numerical artifact.
