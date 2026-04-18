# Figure 5 small-g hybrid eta sparse merge

- `g >= 0.4`: reused from `runs/figure5_smallg_k7_to_10_sparse_v2/`.
- `g = 0.35, 0.3, 0.25`: added by direct small-g eta probes.
- small-g probe mode: `rho/eta` search active; centered/parity preprocessing disabled because it degraded level-9 feasibility in the current implementation.
- confirmed extension:
  - level 7 reaches `g = 0.25` for both energy and `x^2`.
  - level 8 reaches `g = 0.25` for energy and `g = 0.30` for `x^2`.
  - level 9 reaches `g = 0.35` for both energy and `x^2`.
  - level 10 is still missing below `g = 0.5` in this local route.
