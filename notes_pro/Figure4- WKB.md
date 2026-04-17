# Figure 4 WKB

## Model and leading Hamiltonian

For Figure 4,

\[
W(x)=\frac12 x^2+g\frac13 x^3,\qquad g\ge 0.
\]

Fix the \(\epsilon=-1\) sector:

\[
H=\frac12p^2+\frac12g^2x^4+gx^3+\frac12x^2-gx-\frac12.
\]

Apply the exact shift

\[
x=y-\frac1{2g},
\]

then

\[
H=\frac12p^2+\frac12g^2y^4-gy-\frac14y^2+\frac1{32g^2}.
\]

Apply the scaling

\[
y=g^{-1/3}z,\qquad p=g^{1/3}q,\qquad [z,q]=i,
\]

and obtain

\[
H=\frac1{32g^2}+g^{2/3}\left[\frac12q^2+\frac12z^4-z-\frac14g^{-4/3}z^2\right].
\]

Therefore the strict large-\(g\) leading piece is

\[
h_0=\frac12q^2+\frac12z^4-z,
\]

or equivalently

\[
H_{\text{lead}}=g^{2/3}h_0=\frac12 g^{2/3}(q^2+z^4-2z).
\]

## WKB quantization

The leading potential is

\[
U(z)=\frac12 z^4-z.
\]

The classical momentum is

\[
q(z;\varepsilon)=\sqrt{2(\varepsilon-U(z))}=\sqrt{2\varepsilon-z^4+2z}.
\]

The turning points \(z_\pm(\varepsilon)\) are defined by

\[
U(z_\pm)=\varepsilon
\quad\Longleftrightarrow\quad
z_\pm^4-2z_\pm-2\varepsilon=0.
\]

Bohr-Sommerfeld quantization gives

\[
\oint q\,dz=2\pi\left(n+\frac12\right).
\]

For the ground state \(n=0\),

\[
\int_{z_-}^{z_+}\sqrt{2\varepsilon-z^4+2z}\,dz=\frac{\pi}{2}.
\]

This is the scalar equation defining the WKB coefficient.

## Numerical coefficient

Solving

\[
\int_{z_-}^{z_+}\sqrt{2\varepsilon-z^4+2z}\,dz=\frac{\pi}{2}
\]

gives

\[
\varepsilon_0^{\text{WKB}}=0.2610377147951411327.
\]

The corresponding turning points are

\[
z_-=-0.2587949027639901378,\qquad
z_+=1.3370889447325992462.
\]

So

\[
\kappa_{\text{WKB}}=0.2610377147951411327.
\]

## Figure 4 blue line

The WKB curve drawn in Figure 4 is

\[
E_0^{\text{WKB}}(g)=0.2610377147951411327\, g^{2/3}.
\]

Equivalently,

\[
\lim_{g\to\infty} g^{-2/3}E_0(g)=0.2610377147951411327.
\]

## Root-function form

Define

\[
F(\varepsilon):=
\int_{z_-(\varepsilon)}^{z_+(\varepsilon)}
\sqrt{2\varepsilon-z^4+2z}\,dz-\frac{\pi}{2},
\]

where \(z_\pm(\varepsilon)\) are the two real roots of

\[
z^4-2z-2\varepsilon=0.
\]

Then

\[
F(\varepsilon_0^{\text{WKB}})=0.
\]

For plotting, use directly

\[
E_{\text{blue}}(g)=0.2610377147951411327\, g^{2/3}.
\]
