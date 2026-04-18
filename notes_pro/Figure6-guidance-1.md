# Figure 6 guidance 1

## Model

Use the same cubic SUSY QM / `N=1` MP model as Figure 5, in the `epsilon = -1` sector:

```tex
W(x)=\frac12 x^2+\frac g3 x^3,
\qquad
H=\frac12p^2+\frac12 g^2x^4+gx^3+\frac12x^2-gx-\frac12.
```

Figure 6 is not a new system. It is:

```tex
\text{Figure 5 ordinary level-10 SDP}
+ \text{ground-state / thermal PSD block}
+ \text{very-small-}g\text{ stabilization}.
```

## Main warning

Do **not** solve the pure moment recursion as

```tex
m_n=\frac{\cdots}{4g^2(n-1)}.
```

The exact relation

```tex
(n-3)(n-4)(n-5)m_{n-6}
+4(n-3)(2E+1)m_{n-4}
+4g(2n-5)m_{n-3}
-4(n-2)m_{n-2}
-4g(2n-3)m_{n-1}
-4g^2(n-1)m_n=0
```

must be imposed as **linear equalities**, not used for forward `m_n/g^2` elimination.

If one rewrites anything, only use stable `O(1)`-denominator forms such as solving for `m_{n-2}`, not for `m_n`.

In practice the best route is still:

```tex
m_0,\dots,m_{20}\ \text{all kept as variables}
\quad+\quad
\text{linear equalities}
\quad+\quad
\text{PSD blocks}.
```

## Ordinary block for the lower bound

Reuse the original-variable Figure 5 reducer:

```tex
P_{a,b}:=\langle p^a x^b\rangle,\qquad m_b:=\langle x^b\rangle,
```

with

```tex
P_{0,b}=m_b,\qquad
P_{1,b}=-\frac{i b}{2}m_{b-1},
```

and the existing global recursion for `P_{a,b}`.

The ordinary matrix entry remains

```tex
M_{(a,b),(c,d)}
=
\sum_{r=0}^{\min(b,a+c)}
i^r r!\binom{b}{r}\binom{a+c}{r}
P_{a+c-r,b+d-r}.
```

At level 10, the paper uses raw `20x20`. The current repo already compresses this exactly to canonical ambient size `12x12`.

## Ground-state block for the upper bound

The only genuinely new object for Figure 6 is

```tex
G_{ij}=\langle O_i^\dagger[H,O_j]\rangle \succeq 0.
```

For

```tex
O_{c,d}=p^c x^d,\qquad H=\frac12p^2+V(x),
```

with

```tex
V(x)=\frac12 g^2x^4+gx^3+\frac12x^2-gx-\frac12,
```

one may use

```tex
[H,p^c x^d]
=
\sum_{k=1}^{c}\binom{c}{k}i^k V^{(k)}(x)p^{c-k}x^d
-id\,p^{c+1}x^{d-1}
+\frac{d(d-1)}2\,p^c x^{d-2}.
```

and then commute `V^{(k)}` through the left `p^a`:

```tex
p^a f(x)
=
\sum_{\ell=0}^{a}\binom{a}{\ell}(-i)^\ell f^{(\ell)}(x)p^{a-\ell}.
```

Therefore

```tex
G_{(a,b),(c,d)}
=
\sum_{k=1}^{c}\sum_{\ell=0}^{a}
\binom{c}{k}\binom{a}{\ell}i^k(-i)^\ell
\langle x^b V^{(k+\ell)}(x)p^{a+c-k-\ell}x^d\rangle
-id\,\langle x^b p^{a+c+1}x^{d-1}\rangle
+\frac{d(d-1)}2\langle x^b p^{a+c}x^{d-2}\rangle.
```

Since

```tex
V'(x)=2g^2x^3+3gx^2+x-g,\quad
V''(x)=6g^2x^2+6gx+1,
```

```tex
V^{(3)}(x)=12g^2x+6g,\quad
V^{(4)}(x)=12g^2,\quad
V^{(m)}(x)=0\ (m\ge5),
```

every ground-state entry still reduces to linear expressions in `m_0,\dots,m_{20}`.

The paper raw size is `11x11`; the existing canonical compression should reduce it to `8x8`, exactly as in Figure 4.

## Small-g numerical variable

Do not scan linearly in `E`. Use the instanton-scaled variable

```tex
E_{\rm inst}(g)=\frac1{2\pi}e^{-1/(3g^2)},
\qquad
E=E_{\rm inst}(g)e^\eta.
```

Then scan or bracket in `eta`, not directly in `E`.

This is the same small-`g` variable already used in `figure5_cubic_smallg.py`.

## Recommended repo mapping

- Reuse `figure5_cubic_smallg.py` for:
  - original-variable `P_{a,b}` recursion
  - ordinary matrix entries
  - pure-moment linear equalities
  - `eta` parametrization
- Reuse `figure4_cubic.py` for:
  - ground-state PSD framework
  - lower/upper boundary bracket and refine logic
  - margin-based feasibility certificates
- Add a new original-variable ground commutator reducer.

## Practical conclusion

The right Figure 6 formulation is:

```tex
\text{ordinary PSD}
\quad+\quad
\text{ground PSD}
\quad+\quad
A(E,g)m=0
\quad+\quad
E=E_{\rm inst}(g)e^\eta,
```

with no `m_n/g^2` forward elimination.
