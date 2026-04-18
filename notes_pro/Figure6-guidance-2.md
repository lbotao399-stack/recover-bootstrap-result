# Figure 6 guidance 2

## Core judgment

Figure 6 is the Figure 5 small-`g` ordinary bootstrap plus the Figure 4 ground-state block, both at level `10`, for

```tex
W(x)=\frac12x^2+\frac g3x^3,\qquad
H=\frac12p^2+\frac12g^2x^4+gx^3+\frac12x^2-gx-\frac12.
```

The main obstacle is **not** new physics. It is very-small-`g` conditioning at high level.

## What should be solved

### Lower bound

```tex
E_-^{(10)}(g)
=
\inf\Bigl\{
E:\exists m,\ A(E,g)m=0,\ M^{(10)}(E,g;m)\succeq0
\Bigr\}.
```

### Upper bound

```tex
E_+^{(10)}(g)
=
\sup\Bigl\{
E:\exists m,\ A(E,g)m=0,\ M^{(10)}(E,g;m)\succeq0,\ T^{(10)}(E,g;m)\succeq0
\Bigr\}.
```

Here `M` is the ordinary bootstrap matrix and `T` is the ground-state / zero-temperature thermal matrix.

## Ordinary basis

The raw level-10 string basis is the paper's `20x20` matrix.

The repo already knows how to compress the 1D raw string space exactly to canonical monomials

```tex
\mathcal B^{\rm can}_{10}=\{p^a x^b:\ 2a+b\le5\},
```

which has size `12`.

This is the correct ordinary block for Figure 6 in the current codebase.

## Ground basis

Likewise, the raw paper ground-state block is `11x11`, but the exact canonical compression gives

```tex
\mathcal B^{\rm gs}_{10}
=
\{x,x^2,x^3,x^4,p,px,px^2,p^2\},
```

of size `8`.

This matches the existing Figure 4 compressed ground basis logic.

## Small-g pathology and the right fix

The pure moment relation

```tex
(n-3)(n-4)(n-5)m_{n-6}
+4(n-3)(2E+1)m_{n-4}
+4g(2n-5)m_{n-3}
-4(n-2)m_{n-2}
-4g(2n-3)m_{n-1}
-4g^2(n-1)m_n=0
```

must **not** be turned into forward `m_n=(...)/(4g^2(n-1))`.

If anything is rewritten, the stable direction is downward, or better: no explicit elimination at all.

The preferred formulation is

```tex
m=(m_0,\dots,m_{20})
```

as decision variables, with all moment relations imposed as linear equalities.

## Better energy coordinate

Because the true small-`g` scale is instanton-small,

```tex
E_{\rm inst}(g)=\frac1{2\pi}e^{-1/(3g^2)},
```

the correct search variable is

```tex
\rho = 2\pi E\,e^{1/(3g^2)},\qquad
\eta=\log\rho,\qquad
E=E_{\rm inst}(g)e^\eta.
```

The repo already uses this strategy in Figure 5; Figure 6 should keep it.

## Moment centering

At small `g`,

```tex
m_{2k+1}=O(g),\qquad
m_{2k}=m^{\rm HO}_{2k}+O(g^2),
```

with

```tex
m^{\rm HO}_{2k}=\frac{(2k-1)!!}{2^k}.
```

The codebase already has low-order perturbative anchors and the centered Gaussian parameters

```tex
\mu(g)=-\frac g2-\frac{65}{48}g^3,\qquad
\sigma^2(g)=\frac12+\frac{53}{48}g^2.
```

These are useful, but the first Figure 6 implementation should not depend on them, because the current centered/parity path was already seen to hurt level-9 feasibility in the local route.

So the stable first version is:

```tex
\text{original basis}
+ \eta\text{-scan}
+ \text{implicit linear equalities}
+ \text{ground PSD}.
```

## Repo surgery

Minimal change set:

1. keep the Figure 5 original-variable reducer;
2. port the Figure 4 ground-state PSD driver;
3. add an original-variable ground commutator reducer;
4. solve lower/upper branches with margin-certified feasibility in `eta`;
5. only then consider centered-basis refinements.

## Acceptance checks

After the implementation is correct, one should see

```tex
E_-^{(7)}\le E_-^{(8)}\le E_-^{(9)}\le E_-^{(10)}\le E_+^{(10)}.
```

Also

```tex
\eta_\pm(g)=\log\frac{E_\pm(g)}{E_{\rm inst}(g)}
```

should stay `O(1)` and smooth, rather than jumping erratically.
