# Figure 3 convex bootstrap

Model:
- `W(x) = x/(sqrt(2) g) + g x^3/(3 sqrt(2))`
- `epsilon = -1`

Scan setup:
- levels: 2, 3, 4, 5, 6, 7
- g range: [0.05, 5.0] with 61 points
- solver: AUTO
- pure-p constraint order: `2K + 0`
- higher levels inherit the previous level lower bound at the same `g` with a small numerical slack
- display plot masks out-of-range and obvious solver-artifact points; raw `bounds.csv` is unchanged

Status counts:
- K=2: optimal=57, optimal_inaccurate=4
- K=3: optimal=51, optimal_inaccurate=10
- K=4: optimal_inaccurate=61
- K=5: optimal_inaccurate=61
- K=6: optimal=1, optimal_inaccurate=60
- K=7: optimal_inaccurate=61
