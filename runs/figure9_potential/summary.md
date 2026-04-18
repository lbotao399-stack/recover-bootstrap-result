# Figure 9 summary

- Corrected bosonic potential used for plotting:
  - `V(x) = 1/2 (x^2 + 2 g x^3 + g^2 x^4 - 1 - 2 g x)`
- Reason:
  - the printed `main.tex` formula has `2 g x^2`, but that cannot produce the stated critical behavior.
  - the corrected cubic term follows from `W'(x) = x + g x^2`.
- Critical coupling: `g_c = 0.310201619700700`

## Stationary points
- `$g = 0.2$`: `-4.76895855, -2.91111780, 0.18007635`
- `$g = g_c$`: `-2.54245976, 0.24935482`
- `$g = 0.5$`: `0.32471796`

The critical curve is identified by the vanishing derivative discriminant
`Delta = -g^2 (108 g^4 - 1)`.
