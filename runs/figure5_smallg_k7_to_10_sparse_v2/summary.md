# Figure 5 cubic SUSY QM lower bounds

- Model: `W(x) = x^2 / 2 + g x^3 / 3`
- Sector fixed to `epsilon = -1`
- Variable regime: original `(x,p)` reducer, small/intermediate `g`
- Basis mode:
  - `use_completed_basis = True`
  - full theoretical basis remains `p^a x^b` with `2a + b <= L`
  - active numerical run uses the smaller ambient basis with Hermitian completion
- Left panel: fixed-energy feasibility + bisection for `E` lower bounds
- Right panel: `m_2` minimization is evaluated only in the narrow ground-state edge window

Basis sizes:
- level 7: full=20, ambient=9
- level 8: full=25, ambient=9
- level 9: full=30, ambient=12
- level 10: full=36, ambient=12

Status counts:
- level 7: energy=7/7, x2=7/7
- level 8: energy=7/7, x2=6/7
- level 9: energy=7/7, x2=5/7
- level 10: energy=6/7, x2=6/7

Difficulty note:
- current run intentionally starts at `g = 0.4`
- below this range the ordinary solver becomes much less stable, matching the paper's small-g breakdown warning
