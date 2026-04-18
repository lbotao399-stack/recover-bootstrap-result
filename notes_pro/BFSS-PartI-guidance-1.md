# BFSS Part I 综合指导 1

来源：2026-04-18 用户提供的两份综合指导，已按“依赖链优先”合并整理。本文档用于工作流冻结，不表示 BFSS figure 复现已经开始。

## 总原则

- 不按论文图号顺序做，按依赖链做。
- 不先做 `plot3d`；它把归一化、SO(9) block、自由变量消元、level-9 扫描四层误差源混在一起。
- 先复现“解析 + 低维 + 稳定”的量，最后碰“flat direction + 发散变量”的量。
- 当前应优先锁定的 quantitative 输出：
  - `Table freevariables`
  - `Table lowerbound`
  - `Fig. plot3d`
  - `Fig. plconvergence`
  - `Fig. plx4`
  - appendix 的 `Fig. x2y2`
  - appendix 的 `Fig. toyground`

主依赖链：

$$
\boxed{
\text{约定冻结}
\to
\text{toy model 冒烟测试}
\to
\text{BFSS 局部代数单测}
\to
\text{Table 1}
\to
\text{level-6 解析边界}
\to
\text{Table 2}
\to
\text{Fig.\ 2}
\to
\text{Fig.\ 3}
\to
\text{Fig.\ 1}
\to
\text{Appendix D 对照层}
}
$$

等价地，也可记成：

$$
\boxed{
\text{toy calibration}
\to
\text{BFSS conventions}
\to
\text{SO(9)/}\gamma\text{ algebra}
\to
\text{free-variable elimination}
\to
\text{low-level BFSS}
\to
\text{level 7\text{--}9 main scans}
\to
\text{3D figure}
\to
\text{MC comparison}
\to
\text{flat-direction figure}
}
$$

## 0. 先冻结全部约定

先锁定 paper 的物理输入与 large-$N$ 规范，不做数值，只做定义一致性。

Hamiltonian、supercharge 与代数：

$$
H=\frac12 \operatorname{Tr}\!\left(P_I^2-\frac12[X_I,X_J]^2-\psi_\alpha \gamma^I_{\alpha\beta}[X_I,\psi_\beta]\right),
$$

$$
Q_\alpha=\operatorname{Tr}\!\left(P_I\gamma^I_{\alpha\beta}\psi_\beta-\frac{i}{2}[X_I,X_J]\gamma^{IJ}_{\alpha\beta}\psi_\beta\right),
$$

$$
\{Q_\alpha,Q_\beta\}=2\delta_{\alpha\beta}H+2\gamma^I_{\alpha\beta}\operatorname{Tr}(X_I C),\qquad g=1.
$$

large-$N$ 归一化固定为

$$
\operatorname{tr}=\frac1N\operatorname{Tr},\qquad X,P,\psi\mapsto N^{-1/2}(X,P,\psi).
$$

只保留 single-trace SO(9) singlet 变量，动力学约束取

$$
\langle \{Q_\alpha,O_\alpha\}\rangle=0,
$$

正定性按 SO(9) block 写成

$$
a_{R,\bar R}\succeq 0.
$$

时间反演/厄米性层面用

$$
\mathcal P:=-iP
$$

把相关 correlator 统一成实数。

主 observable 固定为

$$
x_2:=\frac19\langle \operatorname{tr} X^I X^I\rangle,\qquad
K:=\frac1{18}\langle \operatorname{tr} P^I P^I\rangle,
$$

$$
OO:=\frac1{64}\langle \operatorname{tr} \psi_\alpha\psi_\alpha\psi_\beta\psi_\beta\rangle,\qquad
x_4:=\frac1{81}\langle \operatorname{tr} X^I X^I X^J X^J\rangle.
$$

hierarchy 固定为

$$
\ell(X)=1,\qquad \ell(P)=2,\qquad \ell(\psi)=\frac32.
$$

## 1. toy model 冒烟测试

先做 appendix toy model，不碰 BFSS 的 SO(9) 与 $\gamma$-代数负担。

### 1a. `Fig. x2y2`

先复现

$$
H_{\rm toy}=\frac12(p_x^2+p_y^2)+\frac12 x^2 y^2
$$

的 arbitrary eigenstate bootstrap，对应 archipelago 结构。检查：

- positivity / feasibility 扫描；
- island / peninsula / archipelago 的拓扑；
- 画图与边界提取。

这里用

$$
\langle OH\rangle=\langle HO\rangle=E\langle O\rangle,\qquad
\langle O^\dagger O\rangle\ge 0.
$$

### 1b. `Fig. toyground`

再做 toy ground-state bootstrap，对应 peninsula 到 island 的结构变化。使用

$$
\langle[H,O]\rangle=0,\qquad
\langle O^\dagger[H,O]\rangle\ge0,\qquad
\langle O^\dagger O\rangle\ge0.
$$

这一步应先把 SDP、扫描、作图流水线跑通。

## 2. BFSS 的基础约束 regression

四类约束先单独锁死：

$$
\langle \{Q_\alpha,\mathcal O_\alpha\}\rangle =0,
\qquad
\langle \operatorname{tr}\mathcal O, C\rangle =0,
$$

$$
\operatorname{tr}\mathcal O_1\cdots \mathcal O_n
=
\pm \operatorname{tr}\mathcal O_2\cdots \mathcal O_n\mathcal O_1
+\text{double trace},
$$

并在大 $N$ 下取

$$
\langle \operatorname{tr} A,\operatorname{tr} B\rangle \to \langle \operatorname{tr} A\rangle \langle \operatorname{tr} B\rangle.
$$

这一层的第一个硬单测是论文里的 4-fermion worked example。

## 3. SO(9) reduction 与 crossing/Fierz

最先锁定分解

$$
\mathbf{16}\otimes \mathbf{16}
=
\mathbf1+\mathbf9+\mathbf{36}+\mathbf{84}+\mathbf{126}.
$$

四费米相关函数写成

$$
\langle \operatorname{tr}(\psi_\alpha\psi_\beta\psi_\eta\psi_\epsilon)\rangle
=
a_1\,\delta_{\alpha\beta}\delta_{\eta\epsilon}
+a_9\,\gamma^I_{\alpha\beta}\gamma^I_{\eta\epsilon}
+\cdots
+a_{126}\,\gamma^{IJKL}_{\alpha\beta}\gamma^{IJKL}_{\eta\epsilon}.
$$

然后复现 crossing matrix $\mathbb F$，并验证

$$
\mathbb F^2=1.
$$

这一层的真实目标不是 4-fermi 本身，而是把显式大矩阵 positivity

$$
\mathcal M_{ij}\succeq 0
$$

压缩成 SO(9) block positivity

$$
a_{\bar R,R}\succeq 0.
$$

## 4. hierarchy 与自由变量消元

先按 cutoff 枚举变量，再解线性约束，把全部变量写成少数 free variables 的线性函数。

必须先锁住论文的计数表：

$$
(11,3),\ (38,4),\ (140,11),\ (569,18),\ (2528,59),\ (12077,149),
$$

分别对应 level cutoff

$$
4,5,6,7,8,9
$$

下的

$$
(\text{total variables},\ \text{free variables}).
$$

这一步就是 `Table freevariables` / `Table 1`。没有这张表，后面的任何图都不该做。

## 5. BFSS 的低层解析结果

### 5a. 4-fermion 与 6-fermion 边界

四费米先得到

$$
1\le OO\le 2.
$$

六费米（level 9）进一步改进上界到

$$
1\le OO\le 1.53125.
$$

### 5b. level-6 解析边界

对 level-6 的 $(x_2,K)$ 边界，定义

$$
V:=36K=-\langle \operatorname{tr} [X^I,X^J]^2\rangle \ge 0.
$$

论文给出

$$
V\le \frac43\sqrt{2x_2},
\qquad
3V^3-64Vx_2+16\le 0,
$$

交点给出

$$
x_2\ge \frac34\left(\frac{3}{50}\right)^{1/3}\approx 0.2936.
$$

这一步应先产出低层版 `plconvergence`。

## 6. `Table lowerbound` 与 `Fig. plconvergence`

先只做 bootstrap 内部结果，不叠 Monte Carlo 行。

lower bound 序列应先复现

$$
x_2^{\min}\ge 0.260,\ 0.294,\ 0.329,\ 0.340,\ 0.355,
$$

对应

$$
\text{level }5,6,7,8^+,9.
$$

其中

$$
8^+ = 8 + (\text{six-fermion constraints from level }9).
$$

也就是

$$
8^+ = (\text{all level 8 constraints}) + (\text{6-}\psi \text{ correlator constraints from level }9).
$$

这一步同时给出完整的 `Fig. plconvergence`。到这里，主体的二维主图已经基本锁住。

## 7. `Fig. plot3d`

等二维区域稳定后，再加第三轴 $OO$，做 level-9 的 allowed region：

$$
(x_2,K,OO).
$$

论文给出的 tip 近似是

$$
(x_2,K,OO)\approx (0.355,\ 0.495,\ 1.02).
$$

这一步本质上是“二维可行域 + 第三观测量”的联合扫描，在流程上必须晚于 `plconvergence`。

## 8. Monte Carlo extrapolation

这是独立支线，不应阻塞 bootstrap 内部主结果。

appendix 中的拟合 ansatz：

$$
\langle \operatorname{tr} [X^I,X^J]^2\rangle
=
b+a_T T^{9/5}+\frac{b_N}{N^2}+\frac{b_L}{L},
$$

$$
\langle \operatorname{tr} X^2\rangle
=
a+\frac{a_N}{N^2}+\frac{a_L}{L}.
$$

这一层只影响 `Table lowerbound` 的 Monte Carlo 对照行，不影响 BFSS 主图本身。

## 9. `Fig. plx4`

这张图最后做。因为它直接碰 flat-direction 相关的不稳定变量：

$$
x_4=\langle \operatorname{tr} X^2 X^2\rangle,
\qquad
\langle \operatorname{tr}(X^2)^m\rangle\quad (m\ge 2).
$$

论文的关键信息是：从高层开始，逼近 $x_2$ 的严格下界时，需要

$$
x_4\to +\infty,
$$

因此会出现 kink / steep wall。这里真正困难的是无界方向与 flat direction 混入观测量，而不只是 SDP 本身。

## 推荐复现顺序

$$
\boxed{
\text{x2y2}
\to
\text{toyground}
\to
\text{4}\psi/\text{Fierz}
\to
\text{Table freevariables}
\to
\text{plconvergence}_{4\text{--}6}
\to
\text{Table lowerbound}_{\rm bootstrap}
\to
\text{plconvergence}_{4\text{--}9}
\to
\text{plot3d}
\to
\text{Table lowerbound}_{\rm MC}
\to
\text{plx4}
}
$$

## 难度序

$$
\text{Fig.4}
<
\text{Fig.5}
<
\text{4-fermion 单测}
<
\text{Table 1}
<
\text{Eq.(43)\text{--}(45)}
<
\text{Table 2}
<
\text{Fig.2}
<
\text{Fig.3}
<
\text{Fig.1}.
$$

## BFSS 主结果最短路径

如果只看 BFSS 主结果，不含 toy warm-up，则最短路径是

$$
\boxed{
\text{Table 1}
\to
\text{Eq.(43)\text{--}(45)}
\to
\text{Table 2}
\to
\text{Fig.\ 2}
\to
\text{Fig.\ 3}
\to
\text{Fig.\ 1}
}
$$
