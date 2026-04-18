# Figure 5 level-7 small-g to zero

- Level: `7` only.
- Range: `g in [0, 0.25]`.
- Small-g route: instanton-rescaled `eta/rho` search with original `(x,p)` reducer.
- Centered/parity preprocessing was tested but left disabled in this run because it degraded small-g feasibility in the current implementation.
- Exact point inserted at `g = 0`:
  - `E = 0`
  - `<x^2> = 1/2`
- Lowest nonzero point reached in double precision: `g = 0.022`.
- Below about `g ~ 0.021`, the instanton scale underflows in the current double-precision route.
