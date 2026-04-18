## Figure 8 guidance 3

- 核心判断：
  - `x^2` 能精确而 `x^4` 偏弱，缺口不在更多 single-trace commutator，而在 quadratic matrix model 的 kinetic term 是二阶微分算子。
  - `x^4` 第一次遇到 split/join，即 single-trace 与 double-trace 耦合。

- 二阶 Heisenberg 结构：
  - 对只依赖 `X` 的 gauge-invariant 多项式 `F(X)`，
    - `[1/2 P_{ij} P_{ji}, F] = - i (dF/dX_{ij}) P_{ji} - 1/2 (d^2 F / dX_{ij} dX_{ji})`
  - stationary Heisenberg:
    - `0 = <[H,F]> = - i <(dF/dX_{ij}) P_{ji}> - 1/2 <Delta F>`
  - 缺失项就是 `-1/2 <Delta F>`；它在 trace algebra 上对应 split/join。

- quadratic vacuum:
  - `(P_{ij} - i a X_{ij}) |0> = 0`
  - `<0| (P_{ij} + i a X_{ij}) = 0`
  - 对齐次次数 `d` 的 `F`，可得
    - `a d <F> = 1/2 <Delta F>`

- single-trace 幂次的 split 公式：
  - `Delta tr X^n = n sum_{r=0}^{n-2} tr X^r tr X^{n-2-r}`
  - 特别地：
    - `Delta tr X^2 = 2 N^2`
    - `Delta tr X^4 = 8 N tr X^2 + 4 (tr X)^2`
    - `Delta (tr X)^2 = 2 N`

- 最小闭包：
  - `x2 := <tr X^2>`
  - `x4 := <tr X^4>`
  - `d11 := <(tr X)^2>`
  - relations:
    - `2 a x2 = N^2`
    - `2 a d11 = N`
    - `4 a x4 = 4 N x2 + 2 d11`
  - 所以
    - `x4 = N (2 N^2 + 1) / (4 a^2)`

- 对实现的最小修正：
  - 新增 double-trace 变量 `d11 = <(tr X)^2>`
  - 新增等式：
    - `<tr X> = 0`
    - `2 a x2 = N^2`
    - `2 a d11 = N`
    - `4 a x4 = 4 N x2 + 2 d11`
  - 可选加 scalar PSD block:
    - basis `{1, tr X}` 或 `{1, tr X, tr X^2}`

- 结论：
  - Figure 8 的 `x^4` tight line 需要 `Delta` 的 split/join，而不是继续在纯 single-trace raw-word closure 里加一阶对易约束。
