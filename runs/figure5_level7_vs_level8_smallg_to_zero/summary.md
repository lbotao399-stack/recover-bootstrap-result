# Figure 5 level-7 vs level-8 near zero

- Range: `g in [0, 0.25]`.
- Compared levels: `7` and `8`.
- Route: original `(x,p)` reducer with small-`g` instanton-rescaled `eta/rho` search.
- Exact point inserted at `g = 0`:
  - `E = 0`
  - `<x^2> = 1/2`
- At the sampled points, level-7 and level-8 energy lower bounds coincide.
- Both levels reach `g = 0.022` for energy and `<x^2>` in the current double-precision route.
- Caveat:
  - `level 8`, `g = 0.25` for `<x^2>` required a widened edge window and remains `optimal_inaccurate`.
  - Below about `g ~ 0.021`, the instanton scale underflows in the current double-precision route.
