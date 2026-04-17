# Figure 4 seeded feasible curve (dense)

- Model: `W(x) = x^2/2 + g x^3/3`
- Sector: `epsilon = -1`
- Method: scaled finite-difference guide + bootstrap feasible seed at the guide energy
- Basis: exact canonical compression, ordinary `12x12`, ground-state `8x8`
- Dense visualization run with slightly relaxed seed certificate thresholds:
  - `psd_tolerance = 3e-6`
  - `margin_tolerance = 1e-3`
- This run improves branch tracking and image density, but it still does not refine the left/right feasible edges.

Status rows:
- g=0.0: lower=exact:g0, upper=exact:g0
- g=0.2: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal_inaccurate
- g=0.3: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=0.4: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=0.5: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=0.6: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=0.8: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=1.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal_inaccurate
- g=1.2: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=1.5: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal_inaccurate
- g=2.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=2.5: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=3.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal
- g=4.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=5.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal_inaccurate
- g=7.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=10.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=15.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=20.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=35.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal
- g=60.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=100.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal_inaccurate
