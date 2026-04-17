# Figure 3 convex bootstrap

Model:
- reduced basis: `(1, q, ..., q^{K-1}, y, ..., y^{K-1})`
- shifted-scaled variables: `x = mu(g) + g^{-1/3} y`, `p = g^{1/3} q`
- stationary point: `g^2 mu^3 + mu + epsilon g / sqrt(2) = 0`
- `epsilon = -1`

Scan setup:
- levels: 2, 3, 4, 5, 6
- g range: [0.05, 5.0] with 61 points
- solver: AUTO
- pure-p constraint order: `2K + 2`
- continuation: warm-start by previous certified moments at the same level

Status counts:
- K=2: clarabel:optimal:cert=61
- K=3: clarabel:optimal:cert=55, clarabel:optimal_inaccurate:cert=5, scs:optimal_inaccurate:uncert=1
- K=4: clarabel:optimal:cert=55, clarabel:optimal_inaccurate:cert=6
- K=5: clarabel:optimal:cert=57, clarabel:optimal_inaccurate:cert=2, scs:optimal_inaccurate:uncert=2
- K=6: clarabel:optimal:cert=46, clarabel:optimal_inaccurate:cert=12, scs:optimal_inaccurate:uncert=3
