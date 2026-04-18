# Figure 5 small-g guidance 4

- The main small-`g` pathology is:
  - explicit division by `g^2`
  - direct linear scan in `E`
  - ill-conditioned raw monomial basis
- Recommended small-`g` search variable:
  - `rho = 2 pi E exp[1/(3 g^2)]`
- Recommended `x^2` variable:
  - `nu = (<x^2> - 1/2) / g^2`
- Suggested exact basis recentering:
  - `mu(g) = -g/2 - 65 g^3 / 48`
  - `sigma^2(g) = 1/2 + 53 g^2 / 48`
  - work in centered/scaled variable `y = (x - mu)/sigma`
- Minimal hybrid strategy:
  - keep current formulation for `g >= 0.4`
  - switch to implicit equalities + instanton-scaled energy search for `g < 0.4`
- Continuation parameter for the nonperturbative branch:
  - `s = 1 / (3 g^2)`
- Even with these fixes, paper-level very small-`g` work is expected to require higher-precision SDP tools.
