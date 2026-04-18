## Figure 8 guidance 1

- Model:
  - Quadratic supersymmetric matrix QM with
    - `W(X) = (a/2) Tr X^2`
    - `H = (1/2) tr(P^2 + a^2 X^2 + a [Psi^\dagger, Psi])`
  - Figure 8 uses `a = 1`.

- Exact benchmark lines:
  - `<tr X^2> = N^2 / 2`
  - `<tr X^4> = N (2 N^2 + 1) / 4`

- Operator set from the paper:
  - `{ I, X, P, Psi, Psi^\dagger, X^2, PX, P^2, Psi Psi^\dagger, Psi X, Psi P, Psi^\dagger X, Psi^\dagger P }`

- Key closure rules:
  - `[H, X] = -i P`
  - `[H, P] = i a^2 X`
  - `[H, Psi] = -a Psi`
  - `[H, Psi^\dagger] = a Psi^\dagger`
  - `[H, Pi] = 0`, with `Pi = Psi Psi^\dagger`

- Stronger vacuum constraints for the quadratic SUSY ground state:
  - `Q |0> = 0`
  - `Qbar |0> = 0`
  - In the bosonic vacuum these become
    - `m[w (P - i a X)] = 0`
    - `m[(P + i a X) w] = 0`
    for bosonic words `w`.

- Gauge constraint in the bosonic vacuum:
  - `m((X P - P X) w) = i N m(w)`

- Important implementation warning:
  - Do not treat traced words as cyclic.
  - In particular `m(XP) != m(PX)`.

- Figure 8 is a finite raw-word SDP, not an infinite recursion problem.
