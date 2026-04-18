# Figure 5 guidance 1

## Core route

For Figure 5, keep the same cubic model

\[
W(x)=\frac12 x^2+g\frac13 x^3,
\]

but move to small/intermediate \(g \in [0,1]\), raise the ordinary hierarchy to levels \(7,8,9,10\), and plot both

\[
E_0(g)
\qquad\text{and}\qquad
\langle x^2\rangle.
\]

The preferred route is to return to the original variables \((x,p)\). The large-\(g\) shifted-scaled reducer from Figure 4 is not the right primary tool here.

## Hamiltonian

Fix the \(\epsilon=-1\) sector:

\[
H=\frac12p^2+\frac12g^2x^4+gx^3+\frac12x^2-gx-\frac12.
\]

## Global recursion

Define

\[
P_{a,b}:=\langle p^a x^b\rangle,\qquad m_b:=\langle x^b\rangle.
\]

Base cases:

\[
P_{0,b}=m_b,
\qquad
P_{1,b}=-\frac{i b}{2}m_{b-1}.
\]

For \(a \ge 2\),

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
&+(a-2)(a-3)\Bigl(
P_{a-4,b}
+6gP_{a-4,b+1}
+6g^2P_{a-4,b+2}
\Bigr)
\\
&+2i(a-2)(a-3)(a-4)\Bigl(
gP_{a-5,b}
+2g^2P_{a-5,b+1}
\Bigr)
\\
&-g^2(a-2)(a-3)(a-4)(a-5)P_{a-6,b}.
\end{aligned}
\]

This is the main recursion for Figure 5.

## Pure moment relation

The specialized SUSY-QM pure-moment relation is

\[
(n-3)(n-4)(n-5)m_{n-6}
+4(n-3)(2E+1)m_{n-4}
+4g(2n-5)m_{n-3}
-4(n-2)m_{n-2}
-4g(2n-3)m_{n-1}
-4g^2(n-1)m_n
=0.
\]

This should be imposed as a linear equality, not used as a forward \(m_n/g^2\) elimination at small \(g\).

Low-order checks:

\[
2g^2m_3+3gm_2+m_1-g=0,
\]

\[
3g^2m_4+5gm_3+2m_2-3gm_1-(2E+1)=0.
\]

## Canonical reduced basis

For level \(L\), use

\[
\mathcal B_L^{\rm red}=\{p^a x^b:\ 2a+b\le L\}.
\]

Dimensions:

\[
L=7,8,9,10
\quad\Rightarrow\quad
20,\ 25,\ 30,\ 36.
\]

## Moment matrix

For

\[
O_{a,b}=p^a x^b,
\]

the matrix entry is

\[
M^{(L)}_{(a,b),(c,d)}
=\langle x^b p^{a+c} x^d\rangle
=\sum_{r=0}^{\min(b,a+c)}
i^r r!\binom{b}{r}\binom{a+c}{r}
P_{a+c-r,b+d-r}.
\]

All entries reduce to moments \(m_0,\dots,m_{2L}\).

## Energy lower bound

For fixed \((g,L)\), define

\[
E^{\rm lb}_L(g)=\inf\{E:\ M^{(L)}(E,g;m)\succeq0,\ m_0=1\ \text{is feasible}\}.
\]

Numerically, do outer bisection in \(E \in [0,0.3]\).

## x^2 lower bound

For fixed \(g,L\), after obtaining the energy lower bound, scan only over

\[
E\in [E_L^{\rm lb}(g),E^{\rm ub}(g)]
\]

and solve

\[
\min m_2
\quad\text{s.t.}\quad
M^{(L)}(E,g;m)\succeq0,\ m_0=1.
\]

This avoids blind energy scans for the right panel.
