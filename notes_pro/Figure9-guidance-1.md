# Figure 9 guidance

- Local `main.tex` prints
  - `V(x) = 1/2 (x^2 + 2 g x^2 + g^2 x^4 - 1 - 2 g x)`
  - but this is a typo.
- From
  - `Q = (P + i X + i g X^2) Psi`
  - one has `W'(x) = x + g x^2`.
- Therefore the bosonic-sector potential used in Figure 9 must be
  - `V(x) = 1/2 ((x + g x^2)^2 - (1 + 2 g x))`
  - `= 1/2 (x^2 + 2 g x^3 + g^2 x^4 - 1 - 2 g x)`.
- Its derivative is
  - `V'(x) = x + 3 g x^2 + 2 g^2 x^3 - g`.
- The stationary-point discriminant is
  - `Delta = -g^2 (108 g^4 - 1)`.
- Hence the critical coupling is
  - `g_c^2 = 1 / (6 sqrt(3))`.
- At `g = g_c`, two extrema merge into an inflection point, matching the paper caption.
