## Figure 8 guidance 4

- 核心判断：
  - current `x^4` lower bound 约为 `N^3/4`，说明只吃到了最粗的 principal minor：
    - `det [[N, x2], [x2, x4]] >= 0`
    - 配合 `x2 = N^2 / 2` 得 `x4 >= N^3 / 4`
  - 这正是 single-trace closure 未包含 split sector 的信号。

- 概念纠正：
  - traced operator product 不能做 cyclic identification：
    - `<tr X P> != <tr P X>`
  - 也不能把 matrix commutator 当成简单的 `P X = X P - i N I` 矩阵恒等式；
    - 正确对象是 traced observable 的局部 cut-and-join。

- Multi-trace observable:
  - `mu[w1 | ... | wk] := < prod_r tr w_r >`
  - Figure 8 至少要允许 `k = 2`。

- 局部交换：
  - `mu[U, P, X, V] = mu[U, X, P, V] - i mu[U | V]`
  - `mu[U, Psi, Psidag, V] = - mu[U, Psidag, Psi, V] + mu[U | V]`
  - quartic bosonic reduction第一次显式出现 `mu[X|X] = <(tr X)^2>`。

- 最小 degree-2 split sector：
  - `d_xx := mu[X|X] = <(tr X)^2>`
  - `d_xp := mu[X|P] = <tr X tr P>`
  - `d_pp := mu[P|P] = <(tr P)^2>`
  - 它们就是 center-of-mass oscillator。

- quadratic ground state (`a = 1`)：
  - `<tr X> = <tr P> = 0`
  - `d_xx = d_pp = N / 2`
  - `d_xp = i N / 2`
  - `d_px = - i N / 2`
  - 不可强行设 `tr X = 0` 作为算子；这里只是 expectation 为零，center mode 不能删。

- quartic bosonic words 的关键 reduction：
  - `mu[XXXP] =  i (N x2 + d_xx/2)`
  - `mu[PXXX] = -i (N x2 + d_xx/2)`
  - `mu[XXPX] =  i d_xx/2`
  - `mu[XPXX] = -i d_xx/2`
  - `mu[XXPP] = x4 - N x2 - d_xx`
  - `mu[PPXX] = x4 - N x2 - d_xx`
  - `mu[XPXP] = d_xx/2`
  - `mu[PXPX] = d_xx/2`
  - `mu[XPPX] = x4`
  - `mu[PXXP] = x4`
  - by vacuum:
    - `mu[XPPP] = i x4`
    - `mu[PXPP] = i d_xx/2`
    - `mu[PPXP] = i (x4 - N x2 - d_xx)`
    - `mu[PPPX] = -i x4`
    - `mu[PPPP] = x4`

- tight certificate：
  - 如果 reduction 正确，则 bosonic minor `{I, X^2, P^2}` 应为
    - `[[N, x2, x2], [x2, x4, -N/4], [x2, -N/4, x4]] >= 0`
    - with `x2 = N^2/2`
  - 其行列式：
    - `det M = -(N/16) (N + 4 x4) (2 N^3 + N - 4 x4)`
  - hence
    - `x4 >= N (2 N^2 + 1) / 4`

- 对实现的直接要求：
  - 不再只表示 single-trace raw word；至少加入 degree-2 double-trace scalar sector
  - 最小新增：
    - `d_xx`, `d_xp`, `d_pp`
  - 原来的 matrix PSD blocks 继续保留
  - 必须加 unit tests 检查上面这些 quartic reductions 和 principal minor。
