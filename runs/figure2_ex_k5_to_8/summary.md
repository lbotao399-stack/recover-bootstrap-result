# Figure 2 quartic correction: E vs <x>

Model:
- `W(x) = x/(sqrt(2) g) + g x^3/(3 sqrt(2))`
- `g = 1.0`
- right half (`x > 0`) uses `epsilon = -1`
- left half (`x < 0`) is filled by the Z2 reflection

Scan setup:
- levels: 5, 6, 7, 8
- x range: [-0.5, 0.5] with 241 points
- E range: [0.0, 5.0] with 251 points
- u search interval: [x^2, 20.0]

Feasible point counts:
- K=5: 4268 feasible grid points, 0 upper-bound u hits
- K=6: 934 feasible grid points, 0 upper-bound u hits
- K=7: 250 feasible grid points, 0 upper-bound u hits
- K=8: 58 feasible grid points, 0 upper-bound u hits
