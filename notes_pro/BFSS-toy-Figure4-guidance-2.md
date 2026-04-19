# BFSS toy Figure 4 guidance 2

来源：用户于 2026-04-18 提供的追加诊断与解析指导。用途：修正 Appendix B.1 / Figure 4 当前错误实现；遇到 toy Figure 4 困难时优先回看本文件与 `BFSS-toy-Figure4-guidance-1.md`。

## 一句总诊断

\[
\boxed{
\text{当前图错误的主因不是分辨率，而是 }
\text{reduction 错}+\text{solver 未认证}+\text{plotting 伪连通}
}
\]

当前 coarse run 只扫了
\[
E=0.5,1.5,2.5,3.5,4.5,
\]
而且五个 slice 全部是 `optimal_inaccurate`。因此它不能当作 Figure 4 的近似版。

## 1. 先加硬拒绝：最低阶 PSD 必要条件

低阶恒等式：
\[
\langle x p_x\rangle=\frac{i}{2},\qquad
\langle p_x^2\rangle=\frac{2E}{3}.
\]

考虑 `{x,p_x}` 的主子块：
\[
\begin{pmatrix}
\langle x^2\rangle & \langle x p_x\rangle\\
\langle p_x x\rangle & \langle p_x^2\rangle
\end{pmatrix}\succeq 0
\]
得到
\[
\boxed{
\langle x^2\rangle \ge \frac{3}{8E}.
}
\]

特别地，
\[
E=0.5 \Rightarrow \langle x^2\rangle \ge 0.75.
\]

当前 coarse run 给出
\[
\langle x^2\rangle \in [0.689,0.699],
\]
因此该 slice 必错。

## 2. 当前 reduction 的结构性错误

理论目标应为
\[
A(E)u=0\Longrightarrow u=N(E)X,
\qquad
X=(\langle x^2\rangle,\langle x^4\rangle,\ldots,\langle x^{12}\rangle),
\]
即 parity/swap quotient 后只剩 6 个非平凡 free variables。

当前代码的实际做法是：

1. 组装所有 even-even moments 的大线性系统；
2. 尽量保留 pure-`x` columns；
3. 再用 QR 把 nullspace 补满。

在 level 12 时本地诊断为
\[
A(E)\in\mathbb R^{422\times 156},\qquad \mathrm{rank}\,A=146,
\]
所以
\[
\dim\ker A=10.
\]

当前 `free_keys` 除了
\[
(2,0,0,0),(4,0,0,0),(6,0,0,0),(8,0,0,0),(10,0,0,0),(12,0,0,0)
\]
之外，还混入额外 top-shell 方向。这样固定 `E` 的 SDP 实际是在一个被放松的 10 维空间上跑，slice 会被系统性放大。

## 3. 改法：用 deterministic shell recursion，不能再用 preferred+QR 补洞

应按解析指导中的两阶段做消元：

### Phase A：先用 `C=L-R` 消掉所有 `q=s+t>0` 的 moments

按 shell `\ell` 从低到高、按 `q` 从低到高、再按 `s` 递推，目标是把所有含动量矩写成 coordinate moments 的线性函数。

### Phase B：再用 `R` 消 mixed coordinate moments

最后只留
\[
u_{2k}:=\langle x^{2k}\rangle,\qquad k=1,\ldots,6.
\]

因此实现上应强制
\[
\texttt{free_keys}
=
\big((2,0,0,0),(4,0,0,0),\ldots,(12,0,0,0)\big).
\]

## 4. Solver 证书层

不能再把 `optimal_inaccurate` 直接当作可行。对每个候选点必须额外检查：

\[
r_{\mathrm{lin}}=\|A(E)u-b\|_\infty,
\qquad
\lambda_{\min}^{\mathrm{all}}=\min_b\lambda_{\min}(G_b),
\]
\[
\Delta_{\mathrm{HP}}
=
\langle x^2\rangle \frac{2E}{3}-\frac14.
\]

只接受满足
\[
\texttt{status}=\texttt{optimal},
\qquad
r_{\mathrm{lin}}\le 10^{-9},
\qquad
\lambda_{\min}^{\mathrm{all}}\ge -10^{-8},
\qquad
\Delta_{\mathrm{HP}}\ge -10^{-8}
\]
的 slice。

## 5. 作图规则

不能再对 sparse slices 直接 `fill_between`。正确顺序：

1. 先画通过证书的竖区间；
2. 再按 certified slices 的连通分量分开填充；
3. 边界附近再做自适应加密。

在当前只有 5 个点、且都未认证的情况下，只能画竖切片诊断图。

## 6. 应补的 regression

\[
\boxed{1}\quad \texttt{len(reduction.free_keys)}=6
\]
\[
\boxed{2}\quad
\texttt{reduction.free_keys}
=
((2,0,0,0),\ldots,(12,0,0,0))
\]
\[
\boxed{3}\quad
\forall E:\ \dim\ker A(E)=6
\]
\[
\boxed{4}\quad
\forall \text{returned point}:\ \lambda_{\min}^{\text{all}}\ge -10^{-8}
\]
\[
\boxed{5}\quad
\forall \text{returned point}:\ \langle x^2\rangle\ge \frac{3}{8E}-10^{-8}
\]
\[
\boxed{6}\quad
\texttt{feasible}\Rightarrow \texttt{status}=\texttt{optimal}
\text{ 或独立认证}
\]

## 7. 低阶解析回归：必须硬塞进代码

记
\[
u_{2k}:=\langle x^{2k}\rangle,\qquad
c_k:=\langle x^{2k}y^2\rangle.
\]

最低阶精确恒等式：
\[
\boxed{
\langle p_x^2\rangle=\langle p_y^2\rangle=\langle x^2y^2\rangle=\frac{2E}{3}.
}
\]
\[
\boxed{
\langle x p_x\rangle=\langle y p_y\rangle=\frac{i}{2}.
}
\]

### level-4 必要条件

\[
\boxed{
u_2=\langle x^2\rangle\ge \frac{3}{8E}.
}
\]

### level-6 更强必要条件

由 mixed `y`-block PSD 与 recursion 上界合并得
\[
\boxed{
216u_2^3-864E^2u_2^2-180Eu_2+256E^3+81\le 0.
}
\]

数值后果：

- `E=0.5`：整条 slice 必 infeasible；
- `E=1.5`：要求 `u_2` 约至少 `0.653`；
- `E=2.5,3.5,4.5`：低阶必要条件分别给出更高的允许下界。

## 8. 两个解析子块

### `x`-tower 子块

basis 取 `{p_x,x,x^3,x^5}`，Gram 应为
\[
\begin{pmatrix}
2E/3 & -i/2 & -3iu_2/2 & -5iu_4/2\\
i/2 & u_2 & u_4 & u_6\\
3iu_2/2 & u_4 & u_6 & u_8\\
5iu_4/2 & u_6 & u_8 & u_{10}
\end{pmatrix}\succeq 0.
\]

### `y`-mixed 子块

basis 取 `{p_y,y,x^2y,x^4y}`，Gram 应为
\[
\begin{pmatrix}
2E/3 & -i/2 & -iu_2/2 & -iu_4/2\\
i/2 & u_2 & 2E/3 & c_2\\
iu_2/2 & 2E/3 & c_2 & c_3\\
iu_4/2 & c_2 & c_3 & c_4
\end{pmatrix}\succeq 0.
\]

## 9. mixed moments 的 recursion 上界

从 `C_{2k+1,0,1,0}=0` 与 `R_{2k,0,0,0}=0` 可得
\[
\boxed{
c_{k+1}\le
\frac{2k+1}{2k+2}
\left(
2E u_{2k}+\frac{k(2k-1)}{2}u_{2k-2}
\right).
}
\]

特别地：
\[
c_2\le \frac32 E u_2+\frac38,
\qquad
c_3\le \frac56(2E u_4+3u_2),
\qquad
c_4\le \frac78(2E u_6+\tfrac{15}{2}u_4).
\]

## 10. reduced diagnostic region

在 full 80-word SDP 之前，可先做 reduced convex SDP，变量只取
\[
u_2,u_4,u_6,u_8,u_{10},u_{12},\quad c_2,c_3,c_4,
\]
约束为：

- pure moment Hankel PSD；
- `x`-tower PSD；
- `y`-tower PSD；
- `c_2,c_3,c_4` 的 recursion 上界。

该 reduced region 是 full toy bootstrap 的必要条件。若 full 输出落在 reduced region 之外，则 full 实现必错。
