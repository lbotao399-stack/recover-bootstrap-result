# Figure 4 high-g fit-guided refinement

- Range: `g in [20,100]`
- Number of points: `16`
- Method:
  - use sparse branch points from `runs/figure4_seeded_feasible_curve/` as low-level anchors
  - fit `Ehat(lambda)` locally
  - search high-level feasibility only in a narrow predicted window
- Thresholds:
  - `psd_tolerance = 3e-06`
  - `margin_tolerance = 0.001`

Status rows:
- g=20.0 pred=2.052655 win=0.003000 low=2.052655 up=2.052655 ls=CLARABEL:optimal us=CLARABEL:optimal_inaccurate
- g=22.0 pred=2.189596 win=0.003000 low=2.189596 up=2.189596 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=24.0 pred=2.322322 win=0.003000 low=2.322322 up=2.322322 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=26.0 pred=2.451310 win=0.003000 low=2.451310 up=2.451310 ls=CLARABEL:optimal us=CLARABEL:optimal_inaccurate
- g=28.0 pred=2.576952 win=0.003000 low=2.576952 up=2.576952 ls=CLARABEL:optimal us=CLARABEL:optimal_inaccurate
- g=30.0 pred=2.699570 win=0.003000 low=2.699570 up=2.699570 ls=CLARABEL:optimal us=CLARABEL:optimal_inaccurate
- g=35.0 pred=2.994580 win=0.003000 low=2.994580 up=2.994580 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=40.0 pred=3.275596 win=0.003000 low=3.275596 up=3.275596 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=45.0 pred=3.544954 win=0.003000 low=3.544954 up=3.544954 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=50.0 pred=3.804381 win=0.003000 low=3.804381 up=3.804381 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=55.0 pred=4.055201 win=0.003000 low=4.055201 up=4.055201 ls=CLARABEL:optimal us=CLARABEL:optimal_inaccurate
- g=60.0 pred=4.298455 win=0.003000 low=4.298455 up=4.298455 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=70.0 pred=4.765476 win=0.003000 low=4.765476 up=4.765476 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=80.0 pred=5.210553 win=0.003000 low=5.210553 up=5.210553 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=90.0 pred=5.637313 win=0.003000 low=5.637313 up=5.637313 ls=CLARABEL:optimal_inaccurate us=CLARABEL:optimal_inaccurate
- g=100.0 pred=6.048448 win=0.003000 low=6.048448 up=6.048448 ls=CLARABEL:optimal us=CLARABEL:optimal_inaccurate
