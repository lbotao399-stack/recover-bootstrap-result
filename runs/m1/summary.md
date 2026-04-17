# M1 run summary

Model: SHO with omega=1, epsilon=-1.

Setup:
- matrix size: 3
- moment cutoff: 6
- recursion t values: 1, 3, 5
- solver: SCS

Observed:
- E=0 slice is feasible.
- The solver returns m2 ~= 0.500000, m4 ~= 0.750000.
- This matches the exact ground-state moments m2 = 1/2 and m4 = 3/4 at the implemented truncation.
