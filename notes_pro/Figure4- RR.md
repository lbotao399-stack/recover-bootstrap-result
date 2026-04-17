# Figure 4 RR

## Leading Hamiltonian

For the cubic model

\[
W(x)=\frac12 x^2+g\frac13 x^3,
\]

in the \(\epsilon=-1\) sector,

\[
H_-=\frac12p^2+\frac12g^2x^4+gx^3+\frac12x^2-gx-\frac12.
\]

After the large-\(g\) scaling,

\[
H_-=\frac12 g^{2/3}\Bigl(q^2+z^4-2z+2g^{-2/3}z^3+g^{-4/3}z^2-g^{-2/3}\Bigr).
\]

The strict leading Hamiltonian is

\[
h_{\rm lead}=\frac12(q^2+z^4-2z).
\]

The paper's orange line is

\[
E_{\rm RR}^{(60)}(g)=\kappa_{60} g^{2/3},
\qquad
\kappa_{60}=\lambda_{\min}\bigl(h_{\rm lead}|_{V_{60}}\bigr).
\]

## Rayleigh-Ritz truncation

For a finite-dimensional trial space

\[
V_N=\mathrm{span}\{\phi_0,\dots,\phi_{N-1}\},
\]

the matrix

\[
H^{(N)}_{mn}=\langle \phi_m|H|\phi_n\rangle
\]

has smallest eigenvalue

\[
\lambda_0^{(N)}=\min_{0\neq c\in\mathbb C^N}\frac{c^\dagger H^{(N)}c}{c^\dagger c},
\]

with variational properties

\[
E_0(H)\le \lambda_0^{(N)},
\qquad
\lambda_0^{(N+1)}\le \lambda_0^{(N)}.
\]

## Harmonic-oscillator basis

Use a harmonic-oscillator basis

\[
|n;\Omega,\xi\rangle,\qquad n=0,\dots,N-1,
\]

with

\[
x=\xi+\frac{a+a^\dagger}{\sqrt{2\Omega}},
\qquad
p=i\sqrt{\frac{\Omega}{2}}(a^\dagger-a).
\]

Then

\[
H_{\rm lead}^{(N)}=\frac12(P^2+X^4-2X).
\]

The simplest paper-style choice is

\[
\xi=0,\qquad \Omega=1.
\]

## Leading-Hamiltonian coefficient

For the unshifted basis \((\xi=0,\Omega=1)\),

\[
\kappa_{60}\approx 0.28106780538565.
\]

Convergence:

\[
N=20:\ 0.2810556692,
\]
\[
N=30:\ 0.2810677503,
\]
\[
N=40:\ 0.2810678056,
\]
\[
N=50:\ 0.2810678054,
\]
\[
N=60:\ 0.2810678054.
\]

So the paper-style orange line is

\[
E_{\rm orange}(g)\approx 0.28106780538565\, g^{2/3}.
\]

## Better full finite-g RR benchmark

Using the exact shift-scaled Hamiltonian,

\[
H_-=\frac1{32g^2}+g^{2/3}\widehat H_\lambda,
\qquad
\lambda=g^{-4/3},
\]

\[
\widehat H_\lambda=\frac12 q^2+\frac12 z^4-z-\frac{\lambda}{4}z^2.
\]

The full finite-\(g\) Rayleigh-Ritz benchmark is

\[
E_{\rm RR,full}^{(N)}(g)=\frac1{32g^2}+g^{2/3}\lambda_{\min}\!\bigl(\widehat H_\lambda^{(N)}\bigr),
\]

with

\[
\widehat H_\lambda^{(N)}=\frac12P^2+\frac12X^4-X-\frac{\lambda}{4}X^2.
\]

This is a more useful finite-\(g\) upper benchmark than the paper's leading-only orange line.

## Classical basis improvement

For the full scaled potential

\[
U_\lambda(z)=\frac12 z^4-z-\frac{\lambda}{4}z^2,
\]

the classical stationary point satisfies

\[
2z^3-\frac{\lambda}{2}z-1=0.
\]

If \(x_0(\lambda)\) is the positive real root, use

\[
\xi=x_0(\lambda),
\qquad
\Omega=\sqrt{6x_0(\lambda)^2-\frac{\lambda}{2}}.
\]

This basis accelerates convergence for the full finite-\(g\) RR benchmark while preserving the variational upper-bound property.
