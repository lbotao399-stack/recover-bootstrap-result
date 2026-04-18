# 阶段总结（截至 Figure 11）

## 一句话

这一阶段已经把本文的主线复现工作从 1D SUSY QM 推到了 matrix cubic model：Figure 1–4、Figure 8、Figure 9 已完成；Figure 5、Figure 6、Figure 10 已得到可用复现版本；Figure 11 已完成 small-`g` 脚手架与 diagnostic probe，但还不是 paper 中的 SDPB 最终版。

## 已完成部分

### 基础内核

- `M1`：1D analytic kernel 已完成
- 内容：
  - polynomial superpotential
  - zero-mode normalizability
  - 一般 1D recursion
  - SHO specialization
  - 最小 `⟨x^n⟩` PSD engine
- 代码：
  - `src/susy_mp_bootstrap/models_1d.py`
  - `src/susy_mp_bootstrap/moments_1d.py`
  - `src/susy_mp_bootstrap/sdp_core.py`

### Figure 1

- 已完成
- quadratic recursion + Hankel PSD
- 输出：
  - `runs/figure1_quadratic/`
  - `runs/figure1_quadratic_k4_to_20/`
  - `runs/figure1_line_u_eq_E_plus_half_k4_to_20_grid400/`

### Figure 2

- 已完成
- quartic correction 的 forward recursion 版本
- 输出：
  - `runs/figure2_ex_k5_to_8/`
  - `runs/figure2_eu_k5_to_8/`

### Figure 3

- 已完成
- quartic convex bootstrap
- 关键修正：
  - reduced basis
  - shifted/scaled variables
  - continuation
  - 显式 residual / PSD / nesting certification
- 输出：
  - `runs/figure3_convex_monotone_k2_to_7/`
  - `runs/figure3_certified_k5_k6/`
  - `runs/figure3_shifted_scaled_k2_to_6/`

### Figure 4

- 已完成
- cubic SUSY QM ground-state energy
- 内容：
  - lower / upper bootstrap
  - WKB curve
  - RR benchmark
  - high-`g` fit-guided refinement
- 代表输出：
  - `runs/figure4_combined_sparse_low_dense_high/figure4_bounds_with_rr_visible.png`
  - `runs/figure4_highg_20_to_100_fit_guided/figure4_bounds_fullframe_with_rr_visible.png`
  - `runs/figure4_near100_refined_edges/figure4_near100_zoom.png`
  - `runs/figure4_rr_benchmarks/figure4_rr_reference_curves.png`

### Figure 8

- 已完成
- quadratic matrix warmup
- 关键结果：
  - `x^2` lower bound 精确贴 exact line：`N^2/2`
  - `x^4` lower bound 精确贴 exact line：`N(2N^2+1)/4`
- 关键实现：
  - raw-word SDP
  - minimal double-trace split sector
  - cut-and-join closure
- 输出：
  - `runs/figure8_matrix_quadratic/`

### Figure 9

- 已完成
- cubic matrix bosonic-sector potential 直接作图
- 关键结果：
  - `g=0.2, g=g_c, g=0.5` 三条曲线已画出
  - `g_c^2 = 1 / (6 sqrt(3))`
- 输出：
  - `runs/figure9_potential/figure9_potential.png`

## 已推进但仍保留困难的部分

### Figure 5

- 已完成一条可用主线，但不是最终 fully-closed 版本
- 当前最稳结果：
  - `level 7` 已推进到 `g=0.022`
  - `g=0` exact endpoint 已接上
  - `level 7/8` near-zero comparison 已完成
- 输出：
  - `runs/figure5_level7_smallg_to_zero/`
  - `runs/figure5_level7_vs_level8_smallg_to_zero/`
  - `runs/figure5_smallg_hybrid_eta_sparse_merge/`
  - `runs/figure5_smallg_k7_to_10_sparse_v2/`
- 保留问题：
  - `level 10` small-`g` 可行性差
  - `H_nu` deformation、arbitrary-precision SDP、implicit/coupled `(m_n,k_n)` 还没接入

### Figure 6

- 已有 hierarchy 版本
- 当前结果：
  - `level 7 -> 10` 的 `eta = ln(E/E_inst)` hierarchy 已完成一版
  - `level 10` 已开始显出窗口收缩
- 输出：
  - `runs/figure6_level7/`
  - `runs/figure6_hierarchy_eta_levels7_to_10_sparse/`
- 保留问题：
  - very-small-`g` 端仍受窗口和双精度限制
  - `CVXPY` 编译开销很重

### Figure 10

- 已完成 large-`g` matrix cubic lower bound 的 hybrid 版本
- 方法：
  - raw non-cyclic matrix words
  - multi-trace cut-and-join closure
  - `g < 20` 原变量
  - `g >= 20` shifted/scaled 变量
- 当前 dense-tail 结果：
  - slope `0.6686`
  - `kappa = 0.2109`
- 代表输出：
  - `runs/figure10_largeg_level8_n100_shift_scaled_hybrid/`
  - `runs/figure10_largeg_level8_n100_shift_scaled_hybrid_dense_tail/`
- 保留问题：
  - 相比 paper guide `0.196 g^{2/3}`，当前 `kappa` 偏高
  - `143 -> 27` checksum 还没 fully saturated

### Figure 11

- 已完成 small-`g` 脚手架
- 已新增：
  - `src/susy_mp_bootstrap/figure11_matrix_cubic_smallg.py`
  - `notes_pro/Figure11-guidance-1.md`
  - `notes_pro/Figure11-guidance-2.md`
- 已有 diagnostic 输出：
  - `runs/figure11_smallg_level8_n100_probe9/figure11_smallg.png`
  - `runs/figure11_smallg_level8_n100_probe9/figure11_smallg_zoom.png`
  - `runs/figure11_smallg_level8_n100_probe9/bounds.csv`
- 保留问题：
  - paper 的 Figure 11 明确要求 SDPB
  - 本地没有 `sdpb` binary，仓库也还没有这套 matrix level-8 的 SDPB exporter
  - 当前 `CVXPY/SCS` fallback 在 `g_0` 附近会给出负的 raw lower bound，因此 probe 只能算 diagnostic，不是最终 Figure 11

## 本阶段未单独展开的图

- Figure 7 未在这轮仓库主线里单独做成一个独立模块或输出目录。

## 主代码入口

- `src/susy_mp_bootstrap/figure1_quadratic.py`
- `src/susy_mp_bootstrap/figure2_quartic.py`
- `src/susy_mp_bootstrap/figure3_convex.py`
- `src/susy_mp_bootstrap/figure4_cubic.py`
- `src/susy_mp_bootstrap/figure5_cubic_smallg.py`
- `src/susy_mp_bootstrap/figure8_matrix_quadratic.py`
- `src/susy_mp_bootstrap/figure9_potential.py`
- `src/susy_mp_bootstrap/figure10_matrix_cubic_largeg.py`
- `src/susy_mp_bootstrap/figure11_matrix_cubic_smallg.py`

## 理论笔记入口

- `notes_pro/`
- 与 Figure 4–11 相关的指导笔记都已按图号保存

## 总结

这一阶段的主要成果不是“把每一张图都完全 paper-tight 地终结”，而是把本文复现中的核心技术路线都打通了：

- 1D recursion / moment PSD
- convex bootstrap 的 reduced / shifted-scaled formulation
- cubic model 的 lower / upper / WKB / RR 对照
- small-`g` instanton-rescaled search
- matrix model 的 raw non-cyclic words
- multi-trace cut-and-join closure
- large-`g` matrix shift-scale

当前真正还卡住的，已经主要不是建模方向，而是两类数值终局问题：

- small-`g` 高 level / very-small-window 的高精度求解
- matrix cubic 在 Figure 10/11 所需的更完整 saturation 与 SDPB 工具链
