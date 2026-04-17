# Figure 4 seeded feasible curve

- Model: `W(x) = x^2/2 + g x^3/3`
- Sector: `epsilon = -1`
- Method: scaled finite-difference guide + bootstrap feasible seed at the guide energy
- Note: this run does not refine the left/right edge of the feasible window; it records the seed energy where ordinary and ground-state constraints are feasible.
- Difficulty:
  - fixed-energy feasible windows are very narrow
  - `cvxpy` compilation is the dominant cost
  - `CLARABEL` / `CVXOPT` leave a small negative margin near the feasible branch, while `SCS` is slower but more forgiving at small `g`

Status rows:
- g=0.0: lower=exact:g0, upper=exact:g0
- g=0.2: lower=SCS:optimal_inaccurate, upper=SCS:optimal_inaccurate
- g=0.5: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal
- g=1.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=2.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=3.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal_inaccurate
- g=5.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=8.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal
- g=12.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=20.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=35.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=60.0: lower=CLARABEL:optimal, upper=CLARABEL:optimal_inaccurate
- g=100.0: lower=CLARABEL:optimal_inaccurate, upper=CLARABEL:optimal
