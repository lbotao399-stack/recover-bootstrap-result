# Figure 6 support

## Goal

Recover Figure 6 for the cubic SUSY QM model

```tex
W(x)=\frac12x^2+\frac g3x^3
```

with

```tex
H=\frac12p^2+\frac12g^2x^4+gx^3+\frac12x^2-gx-\frac12
```

in the `epsilon=-1` sector, over

```tex
g\in[0,0.3],\qquad E\in[0,2].
```

The target plot is the level-10 lower and upper ground-state energy bounds.

## Mathematical structure

### Lower bound

Use the same ordinary bootstrap as Figure 5:

```tex
M^{(10)}(E,g;m)\succeq0,
\qquad
A(E,g)m=0,
\qquad
m_0=1.
```

### Upper bound

Add the ground-state PSD block:

```tex
G^{(10)}(E,g;m)\succeq0.
```

So Figure 6 is

```tex
\text{lower: } M\succeq0,
\qquad
\text{upper: } M\succeq0,\ G\succeq0.
```

## Ordinary reducer

Keep the Figure 5 original-variable canonical reducer:

```tex
P_{a,b}=\langle p^a x^b\rangle,\qquad
P_{0,b}=m_b,\qquad
P_{1,b}=-\frac{i b}{2}m_{b-1}.
```

The canonical ordinary basis at level 10 is

```tex
\{p^a x^b:\ 2a+b\le5\},
```

of size `12`, exactly compressing the paper's raw `20x20` basis.

## Ground reducer

Use the same compressed ground basis style as Figure 4:

```tex
\{x,x^2,x^3,x^4,p,px,px^2,p^2\},
```

of size `8`, exactly compressing the raw `11x11` paper basis.

The core new symbolic ingredient is the original-variable ground entry

```tex
G_{ij}=\langle O_i^\dagger[H,O_j]\rangle.
```

This should be built directly in `(x,p)` variables, then reduced by the existing `P_{a,b}` machinery.

## Stability rule

Do **not** use forward moment elimination from

```tex
m_n=\frac{\cdots}{4g^2(n-1)}.
```

All pure-moment relations must remain as linear equality constraints.

If any algebraic rewriting is used, it should be stable `O(1)`-denominator downward rewriting only.

## Small-g search variable

Use

```tex
E_{\rm inst}(g)=\frac1{2\pi}e^{-1/(3g^2)},
\qquad
E=E_{\rm inst}(g)e^\eta.
```

Therefore the boundary search variable is `eta`, not raw `E`.

## Numerical support

- continuation in `g`, from larger `g` to smaller `g`
- warm-start using previous feasible moments
- margin-certified feasibility, not bare solver status
- first version should stay on original basis; centered/parity preprocessing can remain optional

For the very-small-`g` tail, the likely bottleneck is precision rather than theory. The present local double-precision route is expected to become fragile there, consistent with the paper's use of `SDPB + Mathematica`.

## Direct mapping to current code

- reuse from `src/susy_mp_bootstrap/figure5_cubic_smallg.py`:
  - `Figure5CubicReducer.p_expr`
  - `matrix_entry_expr`
  - `pure_moment_relation`
  - `figure5_energy_from_eta`
- reuse from `src/susy_mp_bootstrap/figure4_cubic.py`:
  - dual-PSD feasibility structure
  - lower/upper bracket and refine logic
  - residual certification
- add:
  - original-variable `ground_entry_expr`
  - level-10 Figure 6 scan wrapper

## Minimum acceptance

The recovered Figure 6 output should satisfy

```tex
E_-^{(10)}(g)\le E_+^{(10)}(g),
```

and should stay compatible with the already recovered Figure 5 lower branch:

```tex
E_-^{(7)}\le E_-^{(8)}\le E_-^{(9)}\le E_-^{(10)}.
```
