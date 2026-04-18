# Figure 5 small-g guidance 3

- Small-`g` failure is structural, not just solver tuning.
- Do not use forward recursion of the form
  - `m_n = lower moments / (4 g^2 (n-1))`
- Replace forward generation by implicit linear constraints:
  - pure-moment banded equalities, or
  - coupled `(m_n, k_n)` system with `k_n = <x^n p^2>`
- Use instanton-scaled energy variable for the lower branch:
  - `E = E_inst(g) exp(eta)`
  - `E_inst(g) = (2 pi)^(-1) exp[-1/(3 g^2)]`
- For small `g`, preconditioning by HO-centered / Hermite-type basis is theoretically preferred.
- For the `x^2` lower branch, a deformation route with `H_nu = H + nu x^2` is more stable than direct `min m_2`.
- Practical regime split:
  - `g >= 0.4`: current sparse CVXPY route is acceptable
  - `0.2 <= g < 0.4`: must remove any effective `1/g^2` forward elimination
  - `g < 0.2`: likely needs arbitrary-precision SDP
