# BFSS toy Figure 4 综合指导 1

来源：2026-04-18 用户提供的两份 Appendix B / Figure 4 toy model 实现指导。本文档保留原始工作流与实现细节，供 `x2y2` archipelago 复现使用。

## 目标

先复现 Appendix B.1 / Figure 4，即 toy model 的 `x2y2` 图：

- Hamiltonian:
  $$
  H=\frac12(p_x^2+p_y^2)+\frac12 x^2 y^2,\qquad [x,p_x]=[y,p_y]=i.
  $$
- eigenstate 约束：
  $$
  \langle \mathcal{O} H \rangle = \langle H \mathcal{O} \rangle = E \langle \mathcal{O} \rangle,
  \qquad
  \langle \mathcal{O}^\dagger \mathcal{O} \rangle \ge 0.
  $$
- word:
  $$
  \mathcal{O}=x^m y^n p_x^s p_y^t.
  $$

论文 Appendix B.1 指出：

- toy model 可取自由变量为 $\langle x^m\rangle$；
- level 12 时最高 pure moment 是 $\langle x^{12}\rangle$；
- Figure 4 来自 fixed-$E$ 扫描后的 allowed slices 拼图。

## 1. level、truncation、word 集合

hierarchy:

$$
\ell(x)=\ell(y)=1,\qquad \ell(p_x)=\ell(p_y)=2.
$$

对有序 word

$$
W_{mnst}:=x^m y^n p_x^s p_y^t,
$$

其 level 为

$$
\ell(W_{mnst})=m+n+2s+2t.
$$

Figure 4 的自洽 truncation：

- unknown moments: $\ell \le 12$
- eigenstate words: $\ell(O)\le 8$
- positivity Gram basis: $\ell(W)\le 6$

于是：

- unknown moments 总数：`|W_12| = 588`
- eigenstate words 总数：`|W_8| = 175`
- positivity basis 总数：`|W_6| = 80`

其中

$$
\mathcal W_k:=\{(m,n,s,t)\in\mathbb Z_{\ge0}^4: m+n+2s+2t\le k\}.
$$

## 2. canonical rewrite 公式

所有实现都建立在一维 canonical ordering 上：

$$
p_x^s x^a
=
\sum_{k=0}^{\min(s,a)}
(-i)^k \binom{s}{k} a^{\underline k}\,
x^{a-k}p_x^{s-k},
$$

$$
p_y^t y^b
=
\sum_{\ell=0}^{\min(t,b)}
(-i)^\ell \binom{t}{\ell} b^{\underline \ell}\,
y^{b-\ell}p_y^{t-\ell},
$$

其中

$$
a^{\underline k}=\frac{a!}{(a-k)!},\qquad a^{\underline 0}=1.
$$

常用特例：

$$
p_x^2 x^m = x^m p_x^2 -2im\,x^{m-1}p_x - m(m-1)x^{m-2},
$$

$$
p_y^2 y^n = y^n p_y^2 -2in\,y^{n-1}p_y - n(n-1)y^{n-2},
$$

$$
p_x^s x^2 = x^2 p_x^s - 2is\,x p_x^{s-1}-s(s-1)p_x^{s-2},
$$

$$
p_y^t y^2 = y^2 p_y^t - 2it\,y p_y^{t-1}-t(t-1)p_y^{t-2}.
$$

## 3. 左、右 eigen-equation

记

$$
M_{mnst}:=\langle x^m y^n p_x^s p_y^t\rangle.
$$

### 左方程

$$
\begin{aligned}
0=L_{mnst}
=&\ \frac12 M_{m,n,s+2,t}
+\frac12 M_{m,n,s,t+2}
+\frac12 M_{m+2,n+2,s,t}
-E M_{mnst}\\
&- i m\,M_{m-1,n,s+1,t}
-\frac12 m(m-1)M_{m-2,n,s,t}\\
&- i n\,M_{m,n-1,s,t+1}
-\frac12 n(n-1)M_{m,n-2,s,t}.
\end{aligned}
$$

### 右方程

最稳实现写成

$$
\begin{aligned}
0=R_{mnst}
=&\ \frac12 M_{m,n,s+2,t}
+\frac12 M_{m,n,s,t+2}
-E M_{mnst}\\
&+\frac12
\sum_{a=0}^{\min(s,2)}
\sum_{b=0}^{\min(t,2)}
(-i)^{a+b}\binom{s}{a}\binom{t}{b}
\,2^{\underline a}2^{\underline b}\,
M_{m+2-a,n+2-b,s-a,t-b}.
\end{aligned}
$$

其中

$$
2^{\underline0}=1,\qquad 2^{\underline1}=2,\qquad 2^{\underline2}=2.
$$

### 实作建议

推荐实际使用

$$
C_{mnst}:=L_{mnst}-R_{mnst}=0,
\qquad
R_{mnst}=0.
$$

并按下面 cutoff：

- `C_{mnst}=0` for $\ell(m,n,s,t)\le 11$
- `R_{mnst}=0` for $\ell(m,n,s,t)\le 8$

理由：

- `C` 的最高 level 项抵消，适合先消 momentum；
- `R` 适合再消 mixed coordinate moments。

## 4. 对称性与实变量

离散对称：

$$
P_x:(x,p_x)\mapsto(-x,-p_x),\qquad
P_y:(y,p_y)\mapsto(-y,-p_y),\qquad
S:(x,p_x)\leftrightarrow(y,p_y).
$$

于是非零 moments 必须满足

$$
M_{mnst}=0
\quad\text{unless}\quad
m+s\equiv0\pmod2,\quad n+t\equiv0\pmod2.
$$

并且

$$
M_{mnst}=M_{nmts}.
$$

时间反演使得

$$
M_{mnst}^*=(-1)^{s+t}M_{mnst}.
$$

因此定义实变量

$$
\widehat M_{mnst}:=i^{\,s+t}M_{mnst}\in\mathbb R.
$$

推荐代码里直接用 $\widehat M$ 做线性消元与 SDP。

## 5. 为什么 free variables 只需取 $\langle x^m\rangle$

论文中“free variables are $\langle x^m\rangle$”的可实现版本：

固定 $E$ 后，把所有 level $\le 12$ 的 moments 组成向量 $u$，写成

$$
A_{12}(E)\,u=0.
$$

对列按如下顺序排序：

$$
u=
\Big(
\text{所有含动量 moments},
\text{所有 mixed coordinate moments},
X_0,X_1,\dots,X_{12}
\Big),
$$

其中

$$
X_r:=M_{r000}=\langle x^r\rangle.
$$

若分 parity/swap sector，则只剩

$$
X_0,X_2,\dots,X_{12}.
$$

行排序推荐：

1. `C=L-R`
2. `R`
3. symmetry
4. normalization `X_0=1`

直觉上是两层三角化：

1. 用 `C` 先降总动量次数 `q=s+t`
2. 用 `R` 再降 mixed-coordinate 的 `y` 次数

最后每个 moment 都写成

$$
M_{mnst}
=
\sum_r c_{mnst;r}(E)\,X_r.
$$

最稳实现不是手推 closed recursion，而是：

1. 固定数值 `E=E_*`
2. 组装矩阵 `A(E_*)`
3. 选定 `X_r` 作为 free columns
4. 线性代数求
   $$
   u_{\rm dep} = -A_{\rm dep}^{-1} A_X\,X
   $$

这就是代码版 recursion。

## 6. shell-by-shell 消元逻辑

对每个 shell $\ell$：

### Phase A：先消所有含 momentum 的 moments

设 `q=s+t`。

对 `q>0`：

- 若 `s>0`，用
  $$
  C_{m+1,n,s-1,t}=0
  $$
  解 `M_{m,n,s,t}`；
- 若 `s=0,t>0`，用
  $$
  C_{m,n+1,0,t-1}=0
  $$
  解 `M_{m,n,0,t}`。

### Phase B：再消 mixed coordinate moments

只剩

$$
\mu_{m,n}:=M_{m,n,0,0}.
$$

- 若 `m=0`，直接用 swap：
  $$
  \mu_{0,n}=\mu_{n,0}.
  $$
- 若 `m>0,n>0`，在 parity sector 中必有 `m,n` 都偶，于是用
  $$
  R_{m-2,n-2,0,0}=0
  $$
  解 $\mu_{m,n}$。

最终每个 shell 只留下新的 pure-`x` 变量

$$
u_\ell:=\langle x^\ell\rangle,
$$

而在 parity-reduced level 12 中只剩

$$
u_0=1,\quad u_2,u_4,u_6,u_8,u_{10},u_{12}.
$$

## 7. positivity / Gram matrix

取 basis

$$
\mathcal B_6=\{W_a\}_{a=1}^{80},\qquad \ell(W_a)\le 6.
$$

moment matrix:

$$
G_{ab}:=\langle W_a^\dagger W_b\rangle.
$$

则

$$
\langle O^\dagger O\rangle\ge0\ \forall O\in \mathrm{span}\,\mathcal B_6
\iff
G\succeq0.
$$

直接 canonical product 公式：

若

$$
W_a=x^{m_a}y^{n_a}p_x^{s_a}p_y^{t_a},\qquad
W_b=x^{m_b}y^{n_b}p_x^{s_b}p_y^{t_b},
$$

则

$$
\begin{aligned}
\langle W_a^\dagger W_b\rangle
=
\sum_{k=0}^{\min(s_a,m_a+m_b)}
\sum_{\ell=0}^{\min(t_a,n_a+n_b)}
&(-i)^{k+\ell}
\binom{s_a}{k}\binom{t_a}{\ell}\\
&\times (m_a+m_b)^{\underline k}(n_a+n_b)^{\underline \ell}\,
M_{m_a+m_b-k,\ n_a+n_b-\ell,\ s_a+s_b-k,\ t_a+t_b-\ell}.
\end{aligned}
$$

做完 substitution 后，

$$
G(E,X)=G^{(0)}(E)+\sum_r X_r\,G^{(r)}(E).
$$

若先做 parity/swap reduction，则只剩 even moments：

$$
G(E,X)=G^{(0)}(E)+\sum_{q=1}^{6} X_{2q}\,G^{(2q)}(E).
$$

论文指导里还指出：按 operator parity 分块后，`level <= 6` 的四个 parity blocks 大小分别是

$$
23,\quad 20,\quad 20,\quad 17.
$$

## 8. Figure 4 的扫描算法

不要把 $E$ 当作 SDP 变量。正确做法：

1. 外扫固定 `E=E_*`
2. 用已消元后的 affine Gram matrix 做两个 SDP：
   $$
   \mu_2^{\min}(E_*)=\min \langle x^2\rangle,
   \qquad
   \mu_2^{\max}(E_*)=\max \langle x^2\rangle
   $$
   subject to
   $$
   G(E_*,X)\succeq0,\qquad X_0=1.
   $$
3. 若 fixed-`E` slice feasible，则该能量对应 allowed interval
   $$
   [\mu_2^{\min}(E_*),\mu_2^{\max}(E_*)].
   $$

把所有有解 slice 叠起来，就是 Figure 4 的 archipelago。

## 9. 必做 unit tests / regression

### 归一化

$$
\langle 1\rangle = 1.
$$

### 一次动量

$$
\langle p_x\rangle=\langle p_y\rangle=0.
$$

### 基本 canonical 检查

$$
\langle x p_x\rangle=\langle y p_y\rangle=\frac{i}{2}.
$$

### parity checks

$$
\langle xy\rangle=\langle x p_y\rangle=\langle y p_x\rangle=\langle p_xp_y\rangle=0.
$$

### virial / level-4 regression

由方程可得

$$
\langle p_x^2\rangle=\langle p_y^2\rangle=\langle x^2y^2\rangle=\frac{2E}{3}.
$$

### 最低阶 PSD

至少应满足

$$
\langle x^2\rangle\ge0,\qquad \langle x^4\rangle\ge \langle x^2\rangle^2.
$$

## 10. 一句话接口

固定能量 $E$ 后，完整流程就是

$$
A_{12}(E)\,u=0
\quad\Longrightarrow\quad
u=N_{12}(E)\,X,
$$

再组装

$$
G_{ab}(E,X)=\langle W_a^\dagger W_b\rangle,
$$

要求

$$
G(E,X)\succeq0,
$$

最后做

$$
\min/\max\ \langle x^2\rangle.
$$

这就是 Appendix B.1 / Figure 4 的完整实现框架。
