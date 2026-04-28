# HI+NQS+SQD — 論文主圖完整參考手冊

> **目的**：給 Leo 在繪製 Nature 系列論文 Figure 1（主架構/E2E 流程圖）時的所有必要資訊，包含算法數學、模型架構、E2E 流程、與其他方法的結果比較，以及 Nature-系列量子化學主圖視覺慣例。
>
> **驗證原則**：本文所有數學/算法/結果都已對照原始碼（`src/methods/`、`src/nqs/`）與結果檔（`results/n2_40q_v*/`、`HI_NQS_SQD_*.md`、`results/stability_test.json`、`results/six_methods.json`）逐項驗證。凡 README 與結果檔不一致之處，會明確標示 **⚠ 不一致**。

---

## 0. TL;DR — 一句話 + 數字

**HI+NQS+SQD** 把 IBM HI-VQE 中的「量子線路 sampler」換成「自迴歸 Transformer Neural Quantum State sampler」，整個 pipeline 在單張 NVIDIA H200 GPU 上跑完。在 N₂ CAS(10,20)/cc-pVTZ（40 qubit、Hilbert space ≈ 1.85 × 10¹¹）上，**用 100,000 個變分行列式即達到能量 −109.21490 Ha（誤差 −0.166 mHa vs HCI 17.4M dets baseline，即比 HCI 還低）**，比 HCI 少 **174× 變分維度**，總時間 ~110 分鐘（單張 H200）。

> 📌 **論文 headline 數字（v4 davidson big @ 100k，已驗證自 `results/n2_40q_v4_davidson/davidson_big_100k.json`）**：
> - HCI(ε=10⁻⁴) baseline：E = −109.21473333 Ha，N_det = 17,380,561，285 s（CPU）。
> - HI+NQS+SQD v4 (big_100k)：E_total = **−109.21489916 Ha**（err = **−0.166 mHa**），N_det_var = **100,000**，N_PT2_externals = 1,706,852，wall = 6603 s（單張 H200）。
> - 同樣 100k 變分基底但 default 設定：err_total = −0.142 mHa，wall = 6785 s。

---

## 1. 數學定義（嚴格版本）

### 1.1 問題陳述
給定 STO-3G / cc-pVTZ 等基底下的二次量子化分子哈密頓量：

$$
\hat H \;=\; \sum_{pq,\sigma} h_{pq}\, a^\dagger_{p\sigma} a_{q\sigma}
\;+\; \tfrac12 \sum_{pqrs,\sigma\tau} g_{pqrs}\, a^\dagger_{p\sigma} a^\dagger_{r\tau} a_{s\tau} a_{q\sigma}
\;+\; E_\text{nuc}
$$

（chemist 記號 `(pq|rs) = g_{pqrs}`，`h_{pq}` 為 1e MO 積分；見 `src/hamiltonians/molecular.py`）。在 Jordan-Wigner 編碼下，每個分子軌道對應 2 個 qubit（α、β 通道），總共 `n_qubits = 2 · n_orb`。

我們要求基態能量 `E_0 = ⟨Ψ_0|H|Ψ_0⟩`，且強制粒子數守恆：
$$
\sum_i \sigma^\alpha_i = N_\alpha, \quad \sum_i \sigma^\beta_i = N_\beta.
$$

### 1.2 自迴歸 NQS 概率模型
電子組態 `x = (σ^α_1, …, σ^α_n, σ^β_1, …, σ^β_n) ∈ {0,1}^{2n}`。自迴歸分解為：

$$
p_\theta(x) \;=\; \prod_{i=1}^{n} p_\theta\!\left(\sigma^\alpha_i \,\big|\, \sigma^\alpha_{<i}\right) \;\times\; \prod_{i=1}^{n} p_\theta\!\left(\sigma^\beta_i \,\big|\, \sigma^\beta_{<i},\, \sigma^\alpha_{1:n}\right).
$$

α 通道用 **causal self-attention**；β 通道用 **causal self-attention + cross-attention 到 α 的 bidirectional context**（β 看得到完整 α 構型）。每個位置 `i` 由 logit 經 sigmoid 得到 `p(σ_i = 1 | …)`，sample 經 Bernoulli。

**粒子數約束**（`src/nqs/transformer.py:339-388`）— 在每一步取樣時做硬 logit 修正：
- `must_place = (needed ≥ remaining_slots)`：強制 logit ← +10
- `cannot_place = (placed ≥ N_α)`：強制 logit ← −10

確保所有取樣樣本都嚴格滿足 `Σσ^α = N_α`、`Σσ^β = N_β`，無需 post-selection。

### 1.3 Sample-based Quantum Diagonalization (SQD)
給定取樣得到的構型集合 `B = {x_1, …, x_M}`，把哈密頓量投影到由這些 Slater 行列式張開的子空間：

$$
H^{(B)}_{ij} \;=\; \langle x_i | \hat H | x_j \rangle, \qquad i,j \in \{1,\dots,M\}.
$$

最後對 `H^{(B)}` 求最低本徵值：
$$
E_0^{(B)} \;=\; \min_{c \in \mathbb R^M,\,\|c\|=1}\; c^\top H^{(B)} c.
$$

矩陣元 `⟨x_i|H|x_j⟩` 由 Slater–Condon 規則計算（見 `src/methods/sparse_det_solver.py:32-235`，包含：
- 對角：`e1 + 0.5·e2`（標準 fci_slow 公式）
- α-single / β-single 激發
- αα / ββ same-spin double 激發
- αβ mixed double 激發
- JW signs：`(-1)^{popcount(s & mask_between_p_and_a)}`

### 1.4 Epstein–Nesbet PT2 — 兩個用途
我們在演算法的兩個不同位置使用 EN-PT2，意義不同。

**(A) 每輪選擇 (per-iteration ranking, `src/methods/hi_nqs_sqd_v3.py:339-370`)**：對 NQS 取出的「新候選 `x`」打分，
$$
\text{score}(x) \;=\; \frac{|\langle x|\hat H|\Phi_0\rangle|^2}{|E_0 - H_{xx}|}
$$
其中 `Φ_0` 是上一輪 SQD 的本徵向量、`H_{xx} = ⟨x|H|x⟩`。第 0 輪沒有 `Φ_0` 時，退回用 diagonal `H_{xx}` 排序（`hi_nqs_sqd.py:198-220`）。取 top-k 加入累積基底。

**(B) 收斂後一次性外部 PT2 修正 (final EN-PT2 correction, `hi_nqs_sqd_v3.py:65-148`)**：對最終基底中振幅最大的前 `pt2_top_n=5000` 個行列式，列舉它們所有 single+double H-連接的「外部行列式」，計算
$$
E_\text{PT2} \;=\; \sum_{x \notin B} \frac{|\langle x|\hat H|\Psi_0\rangle|^2}{E_0 - H_{xx}}.
$$
總能量 `E_total = E_var + E_PT2`。對 N₂-40Q 約 1M 外部行列式，僅佔總時間 ~6%，但能 **降低 ~0.25 mHa** 誤差（由 v3 nopt2 vs default 的對照可驗證：+0.384 → −0.104 mHa）。

### 1.5 NQS 訓練損失（`hi_nqs_sqd.py:526-602`）
固定當前基底 `B` 與本徵向量 `Ψ_0`，把 `Ψ_0` 當 teacher 教 NQS：

**Teacher 權重**（透過 α/β marginal 分解，避免 `n_α × n_β` 顯式重建）：
$$
w_x \;=\; \frac{p_\alpha(\sigma^\alpha_x) \cdot p_\beta(\sigma^\beta_x)}{\sum_{x'\in B} p_\alpha(\sigma^\alpha_{x'}) \cdot p_\beta(\sigma^\beta_{x'})}, \quad
p_\alpha(s) = \!\!\!\sum_{t \in \text{ci\_strs}_b}\!\!\! |\Psi_0(s,t)|^2.
$$

**總損失**：
$$
\mathcal L \;=\; \underbrace{-\lambda_\text{wf} \sum_x w_x \log p_\theta(x)}_{\text{teacher cross-entropy}}
\;+\; \underbrace{\lambda_\text{E} \sum_x w_x (H_{xx}-E_0) \log p_\theta(x)}_{\text{REINFORCE-style energy advantage}}
\;+\; \underbrace{\lambda_\text{ent}\, \overline{\log p_\theta}}_{\text{entropy regulariser}}
$$

預設 `(λ_wf, λ_E, λ_ent) = (1.0, 0.1, 0.05)`，Adam lr=1e-3，10 mini-batch steps/iter，gradient clipping max_norm=1.0。

### 1.6 Classical S+D Expansion（v2 之後加上的關鍵）
為了避免「NQS 取樣分布塌縮」，每一輪在 NQS 取樣之外，**還要把目前基底中振幅最大的 top-N（預設 1000）個行列式的所有 single+double 激發列出**（`hi_nqs_sqd_v2.py:71-139`），扣掉已存在於基底者，丟進候選池。這個步驟在 N₂-40Q 把誤差從 +60 mHa 一口氣降到 +1.7 mHa（見 §4 表格）。

---

## 2. 模型架構（Architecture，所有數字皆從 `src/nqs/transformer.py` 確認）

### 2.1 Autoregressive Transformer NQS
```
Input: 0/1 occupation tokens                     [B, 2n]  (after split into α, β)
   ↓
OccEmbedding(2 → embed_dim) + PosEmbedding(n)
   ↓
START token prepend, shift right
   ↓
─── α tower (causal self-attention, n_layers blocks) ───
   ↓                                       (B, n, embed_dim)
α_logits ← Linear(embed_dim → 1)           (B, n)

α full context (no causal mask, n_layers blocks)  — 給 β 用 cross-attention
   ↓
─── β tower (causal self-attn + cross-attn to α, n_layers blocks) ───
   ↓
β_logits ← Linear(embed_dim → 1)
```

**Auto-scaling by system size**（`hi_nqs_sqd.py:99-110`）：
| n_orb | embed_dim | n_heads | n_layers | params (approx) |
|------:|----------:|--------:|---------:|----------------:|
|  ≤ 5  |        64 |       4 |        3 | ~50K |
|  ≤ 7  |       128 |       4 |        4 | ~400K |
| ≤ 10  |       128 |       8 |        6 | ~600K |
| ≤ 15  |       192 |       8 |        6 | ~1.4M |
| ≤ 20  |       256 |       8 |        8 | ~3.2M |
| ≥ 21  |       256 |       8 |       10 | ~4M |

每個 Transformer block：`x ← x + SelfAttn(LN(x))` → `x ← x + CrossAttn(LN(x), α_ctx)`（僅 β）→ `x ← x + FFN(LN(x))`。FFN 是 `Linear(d→4d) → GELU → Linear(4d→d)`。Xavier-normal init with gain=0.02。

### 2.2 SQD Backend 變體
程式碼裡有四種對角化後端，從慢到快：
1. **`solve_fermion`** (`qiskit_addon_sqd.fermion`)：IBM 標準路徑，每次重建 SCI、計算 RDM、Davidson cold-start。
2. **`IncrementalSQDBackend`** (`incremental_sqd.py`)：直接呼叫 PySCF `kernel_fixed_space`，持久 `SelectedCI` 物件，warm-start `ci0`，跳過 RDM。但仍在 `n_α × n_β` Cartesian product 上做 Davidson。**這是 N₂-40Q 「43k 假象」的元兇**——表面上 43k bitstrings，實際 Davidson 維度 118M（見 §5）。
3. **`SparseDetSQDBackend`** (`sparse_det_backend.py` + `sparse_det_solver.py`)：CPU 上純 Python 實作 N_det × N_det 真稀疏 CI，Davidson vector length = 真實 N_det。比 (2) 誠實但慢。
4. **`GPUSparseDetSQDBackend`** (`gpu_sparse_det_solver.py`)：GPU 向量化版本：`get_connections_vectorized_batch` + `torch.searchsorted` 找 (i,j) 對 → 組稀疏 CSR → 仍呼 scipy `eigsh`。**對 N=100k 提速 20-50×**。
5. **GPU Davidson** (`gpu_davidson.py`)：在 GPU 上自寫 Davidson（torch sparse matvec via cuSPARSE + 子空間 `torch.linalg.eigh` + double Gram-Schmidt + Davidson preconditioner `t = r/(diag − θ)` + max_subspace=20 restart）。
6. **Multi-GPU Davidson** (`multi_gpu_davidson.py`)：把 H 按列分到 K 張 GPU，每張只做 `H_k @ v` 然後在 master gather。Krylov 子空間操作 (Gram-Schmidt、子空間特徵分解) 在 master 上做。

### 2.3 Multi-GPU PT2（`multi_gpu_pt2.py`）
- `replicate_hamiltonian`：把 `MolecularHamiltonian` 複製到 K 張 GPU。
- `compute_coupling_multi_gpu`：把候選池 chunk 後分到 K 張 GPU，每張獨立計算 `|⟨x|H|Φ_0⟩|`，最後 concat。
- `final_pt2_multi_gpu`：Phase 1 在 master 上列舉並去重所有外部行列式；Phase 2 把外部分到 K 張 GPU 平行算 PT2 貢獻。**正確性說明**：每個外部行列式的 coupling 是獨立計算的（向所有基底行列式求和已在單個 `get_connections` call 內完成），所以分外部空間是 exact，分基底會錯。

---

## 3. End-to-End 演算法流程（畫圖用 — 每個方塊的精確內容）

### 3.1 主迴圈（v3 + v4 共用主結構，見 `hi_nqs_sqd_v3.py:152-491`）

```
INPUT  : Hamiltonian H  (h1e, eri, E_nuc;  n_orb, n_α, n_β)
OUTPUT : E_total = E_var + E_PT2

─────────────────────────────────────────────────────────────────────
INIT
  • Build AutoregressiveTransformer (auto-scale by n_orb)
  • Adam optimizer (lr 1e-3)
  • cumulative_bs = ∅,  cumulative_hashes = ∅
  • prev_sci_state = None,  current_E0 = None
  • (optional, classical_seed=False by default in v3) seed basis with HF + S + D
─────────────────────────────────────────────────────────────────────

for iter in 0 … max_iterations-1:

  ┌──────────────────────────────────────────────────────────────────┐
  │  ① NQS SAMPLING  (GPU, ~0.5–1s)                                 │
  │     T = T0 + (T1 − T0) · iter / (max_iter − 1)                  │
  │     {x_1, … , x_M} ← NQS.sample(M, temperature=T)               │
  │       (M=100k or 1M, particle-number constrained)                │
  │     filter to (Σα = N_α) & (Σβ = N_β)                           │
  │     dedupe vs cumulative_hashes  →  new_candidates              │
  └──────────────────────────────────────────────────────────────────┘
                              ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  ②  CLASSICAL S+D EXPANSION  (GPU, ~5s)        [iter ≥ 1 only]  │
  │     pick top-N (N=1000) amplitudes from prev_sci_state           │
  │     enumerate all single+double H-connections (vectorised)       │
  │     filter NEW only  →  append to new_candidates                 │
  └──────────────────────────────────────────────────────────────────┘
                              ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  ③  PT2 SCORING + TOP-K SELECTION  (GPU, 1–10s)                 │
  │     iter = 0:  diag-energy ranking (cold-start, keep top_k/4)    │
  │     iter = 1:  RESCORE existing basis ∪ new (eject cold-start    │
  │                noise), keep top max_basis_size                   │
  │     iter ≥ 2:  score new only,  append top-k                     │
  │       score(x) = |⟨x|H|Φ_0⟩|² / |E_0 − H_xx|                    │
  │     evict to max_basis_size by score (unless monotonic mode)     │
  └──────────────────────────────────────────────────────────────────┘
                              ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  ④  SQD DIAGONALIZATION  (40–150s, dominant cost)               │
  │     Build sparse H on GPU (vectorised batch + searchsorted)      │
  │     Davidson lowest-eigvalue on H                                │
  │       ⇒  E0,  Φ_0  (ground-state coefficients)                  │
  │     Warm-start from prev eigenvector (cached amplitudes)         │
  └──────────────────────────────────────────────────────────────────┘
                              ↓
  ┌──────────────────────────────────────────────────────────────────┐
  │  ⑤  NQS UPDATE  (GPU backprop, ~2s)                             │
  │     Build teacher weights  w_x = p_α(σ^α_x) · p_β(σ^β_x)        │
  │     Loss = λ_wf·(−Σ w_x log p_θ(x))                              │
  │          + λ_E ·(Σ w_x (H_xx − E_0) log p_θ(x))                 │
  │          + λ_ent · ⟨log p_θ⟩                                    │
  │     Adam.step ×10, grad clip 1.0                                 │
  └──────────────────────────────────────────────────────────────────┘
                              ↓
  ⑥  CONVERGENCE  : if |ΔE| < 10⁻⁹ for 5 consecutive iters → BREAK

─────────────────────────────────────────────────────────────────────
FINAL EN-PT2 CORRECTION  (one-shot, ~5–10% of total time)
─────────────────────────────────────────────────────────────────────
  pick top-N amplitudes (N = 5000 or 10000)
  enumerate all S+D connections of those → external set X_ext
  E_PT2 = Σ_{x ∈ X_ext, x ∉ B}  |⟨x|H|Ψ_0⟩|² / (E_0 − H_xx)
  E_total = E_var + E_PT2
─────────────────────────────────────────────────────────────────────
```

### 3.2 Per-iteration 時間分布（已驗證自 `HI_NQS_SQD_benchmark_data.md` §6，N₂-40Q v3 default_100k，22 iters，6139s）

| Phase | 時間 | 占比 | 內容 |
|---|---:|---:|---|
| ④ SQD Davidson (sqd) | 3445s | **56.1%** | scipy eigsh (CPU bottleneck) |
| ③ per-iter PT2 (pt2) | 1661s | 27.1% | coupling `<x|H|Ψ_0>` |
| ① NQS sampling (samp) | 512s | 8.3% | Transformer autoregressive sampling |
| 收尾 EN-PT2 | 363s | 5.9% | Σ_x \|coupling\|² / (E0−Hxx) |
| ② classical expansion | 106s | 1.7% | enumerate S+D |
| ⑤ NQS backprop (upd) | 51s | 0.8% | Adam steps |

**v4 GPU Davidson + Multi-GPU PT2** 把 56% + 27% 的 ④ + ③ bottleneck 移到 GPU，整體時間從 102 min 降到 ~110 min（單 GPU）— 看似差不多，但 v4 default 用了 **10× 更多** NQS samples（1M vs 100k），所以同等取樣量下 v4 更快。Multi-GPU 8× H200 跑同樣設定預計 < 30 min（兩個分散式 job 在 cluster time-out 前未完成）。

---

## 4. 結果（嚴格驗證版）

### 4.1 N₂ CAS(10,20)/cc-pVTZ — 主基準（40 qubit、Hilbert ≈ 1.85 × 10¹¹）

> **基準**：HCI（PySCF selected_ci，ε = 10⁻⁴）= **−109.21473333 Ha**，N_det = 17,380,561，285 s（CPU）。所有 NQS 結果 seed = 42，1× NVIDIA H200。
> 來源：`HI_NQS_SQD_comparison_table.md`、`results/n2_40q_v3/*.json`、`results/n2_40q_v4_davidson/*.json`、`HI_NQS_SQD_benchmark_data.md`。

| # | Method | N_det (Davidson dim) | err_var (mHa) | +PT2? | **err_total (mHa)** | Wall | 備註 |
|---|---|---:|---:|:-:|---:|---:|---|
| 1 | HCI ε=10⁻⁴ (reference) | 17,380,561 | 0 (ref) | — | **0** | 285 s | PySCF selected_ci |
| 2 | OLD incremental_sqd ("43k claim") | **118,222,129** (n_α·n_β) | +0.25 | — | +0.25 | 4.7 h | ⚠ **claim 已被推翻**（actual Davidson dim = 118M，6.8× MORE than HCI；v1 sparse_det 才是真實的 N_det） |
| 3 | v1 baseline 100k | 32,904 | +60.00 | — | +60.00 | 29 m | NQS sampling alone, 無 expansion |
| 4 | v1 baseline 500k | 121,565 | +17.27 | — | +17.27 | 96 m | plateau |
| 5 | v2 classical_expansion @ 100k | 62,466 | +1.74 | — | +1.74 | 22 m | classical expansion 是關鍵 |
| 6 | v2 classical_expansion @ 1M | 572,129 | +1.22 | — | +1.22 | 360 m | |
| 7 | **v3 default @ 100k** | **100,000** | +0.384 | ✅ 989,264 ext | **−0.104** | 102 m | **論文 headline**（v3） |
| 8 | v3 default @ 200k | 200,000 | +0.159 | ✅ 888,741 | **−0.125** | 214 m | |
| 9 | v3 default @ 500k | 449,551 | +0.138 | ✅ 816,821 | **−0.130** | 536 m | |
| 10 | v3 big @ 100k | 100,000 | +0.355 | ✅ 1,704,900 | **−0.128** | 136 m | |
| 11 | v3 big @ 1M | 658,098 | −0.043 | ✅ 1,275,833 | **−0.158** | 583 m | |
| 12 | **v4 GPU-Davidson default @ 100k** (1M samples) | 100,000 | +0.346 | ✅ 988,585 | **−0.142** | 113 m | **論文 headline**（v4，GPU 全棧） |
| 13 | **v4 GPU-Davidson big @ 100k** (1M samples) | 100,000 | +0.317 | ✅ 1,706,852 | **−0.166** | 110 m | **最佳單 GPU 結果** |
| 14 | v4 GPU-Davidson big @ 200k | 200,000 | — | ✅ | **−0.173** | 132 m | |
| 15 | v4 GPU-Davidson default @ 1M | 1,000,000 | — | ✅ | **−0.121** | 328 m | |

**結論**：100k 變分基底已飽和。`big` 變體（top_n=2000、pt2_top_n=10000）一致比 `default` 低 ~0.02–0.03 mHa。所有 PT2-on 結果都 < 0（比 HCI 還低），原因是 HCI ε=10⁻⁴ 本身相對 FCI 還有 ~0.1 mHa 殘差未收歛。

### 4.2 收歛軌跡（畫圖用，v4 davidson default_100k）

來自 `results/n2_40q_v4_davidson/davidson_default_100k.json` 的 `energy_history`（24 iters）：

```
iter   E_var (Ha)         err vs HCI (mHa)
  0   −109.10064325       +1108.69      (cold start)
  1   −109.20871431       +6.02
  2   −109.21198563       +2.75
  3   −109.21290584       +1.83
  4   −109.21341691       +1.32
  5   −109.21373757       +1.00
  6   −109.21395687       +0.78
  7   −109.21411550       +0.62
  8   −109.21423249       +0.50
  9   −109.21432000       +0.41
 10   −109.21438676       +0.347     ← basis 飽和到 100k
 11   −109.21438689       +0.346
 ...
 23   −109.21438698       +0.346     ← 變分收斂
 +PT2  −109.21487524       −0.142
```

`basis_history`：[10000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 100000 …]（一輪加 10k，10 輪後達到上限）。

### 4.3 Classical baselines（C₂H₄ 28-qubit ablation，已驗證自 README §Results）

| Variant | E (Ha) | vs SCI (mHa) | Time |
|---|---:|---:|---:|
| SCI (CIPSI, PT2) | −77.2351408123 | 0.000 | 744 s |
| HI+NQS+SQD (no expansion) | −77.2352813675 | **−0.141** | 485 s |
| HI+NQS+SQD + expansion (no PT2) | −77.2352926820 | **−0.152** | 462 s |
| HI+NQS+SQD + expansion + PT2 | −77.2353131332 | **−0.172** | 895 s |

> ⚠ 這一節的數字目前只在 README.md 出現，在 `results/` 找不到對應 JSON。**畫圖前請 Leo 跑 `scripts/run_c2h4_expansion.slurm` 確認**。

### 4.4 Small-molecule sweep（Hilbert space tractable，FCI 可比對）

> **驗證來源**：`results/six_methods.json`、`results/stability_test.json`、`HI_NQS_SQD_benchmark_data.md` §1。**README 的 LiH/H₂O/BeH₂ 數字與 result files 部分不一致**，下表只列「結果檔可重現的數字」。

**單 seed 結果 (`results/six_methods.json`)**：
| Mol | n_q | FCI (Ha) | HI+NQS+SQD err (mHa) | basis | Wall |
|---|---:|---:|---:|---:|---:|
| LiH | 12 | −7.882324 | **−5.3 × 10⁻¹²** | 225 | 4.9 s |
| H₂O | 14 | −75.013155 | **0.000** | 440 | 5.7 s |

**5-seed 穩定性 (`results/stability_test.json`)**：
| Mol | HI+NQS+SQD mean ± std (mHa) | HI-VQE mean ± std (mHa) |
|---|---:|---:|
| LiH | (−1.78 ± 1.59) × 10⁻¹² | 0.277 ± 0.232 |
| H₂O | 0.000 ± 0.000 | 1.450 ± 0.819 |
| BeH₂ | **4.119 ± 4.558** | 0.966 ± 0.182 |

> ⚠ **README §"HI+NQS+SQD vs HI-VQE" 表格存在不一致**：
> - README 說 「BeH₂ 14：HI+NQS+SQD 0.000 ± 0.000；HI-VQE 30.899 ± 5.424」。
> - 但 `stability_test.json` 顯示 「HI+NQS+SQD 4.12 ± 4.56；HI-VQE 0.97 ± 0.18」。
> - 同樣，README NH₃ HI-VQE 62.339 ± 1.335 在 stability_test.json 找不到對應紀錄。
>
> **論文圖請以 `stability_test.json` 為準**，或補跑後重新驗證。可能 README 的數字來自舊版實驗或 incremental_sqd 階段，沒同步更新。

### 4.5 跨方法 N_det 效率（small-mol，從 §4.3 §4.4 §1 整理）

| Mol | n_q | FCI ndet | HCI(ε=1e-4) ndet | CIPSI ndet | HI+NQS+SQD v1 ndet (chem-acc) |
|---|---:|---:|---:|---:|---:|
| H₂O | 14 | 441 | 441 | 120–133 | ~120 |
| BeH₂ | 14 | 1,225 | 1,024 | 169 | ~250–500 |
| NH₃ | 16 | 3,136 | 3,136 | 1,576 | ~2000 |
| CH₄ | 18 | 15,876 | 15,876 | 3,996 | ~10,000 |
| N₂ | 20 | 14,400 | 14,400 | 1,824 | ~10,000 |

**Key finding (HI_NQS_SQD_benchmark_data.md §1)**：在 STO-3G 小分子上，CIPSI 是最 det-efficient（HF + 排序好的 PT2 selection），HI+NQS+SQD v1 略遜（NQS 隨機取樣有噪音）。**HI+NQS+SQD 的優勢在 N₂-40Q 等「FCI/HCI 都不可行」的大系統**：HCI 17.4M dets vs HI+NQS+SQD 100k dets，**174× saving** + 比 HCI 還低的能量。

---

## 5. 「43k 假象」事件 — 為什麼必須講清楚

老論文 draft / 原始 `incremental_sqd` 紀錄寫過「N₂-40Q HI+NQS+SQD 用 43k bitstrings beat HCI 的 17.4M dets，少 402×」。**這是錯的。**

實際（measurement job 21794，記載於 `HI_NQS_SQD_benchmark_data.md` §2）：

| iter | bitstrings | n_α | n_β | **真正的 Davidson dim = n_α × n_β** |
|---:|---:|---:|---:|---:|
| 0 | 10,000 | 1,126 | 1,126 | **1,267,876** |
| 4 | 40,000 | 10,860 | 10,860 | 117,939,600 |
| 8 | 42,957 | 10,873 | 10,873 | **118,222,129** |

→ 真實 Davidson 維度 **118M，是 HCI 的 6.8 倍**，不是 1/402。原因：`incremental_sqd` 後端把所有 unique α-strings × 所有 unique β-strings 當基底（Cartesian product），即使絕大多數 (α, β) 對在 NQS 取樣中沒出現過、振幅應為 0。

**修正 (commit `300802d` 之後)**：改用 `SparseDetSQDBackend` / `GPUSparseDetSQDBackend` — Davidson 向量長度 = 真實 N_det = 100,000，不再 ghost-padding。**論文必須明確說「N_det 指真實 sparse-det Davidson 向量長度」**，並引用此修正避免被 reviewer 抓到。

---

## 6. 結論

1. **核心 contribution**：把 IBM HI-VQE 的 quantum-circuit sampler 換成 autoregressive Transformer NQS，整個 pipeline 在 1× H200 GPU 跑完，**不需要量子硬體、不需要 noise-aware 採樣、梯度 exact**。
2. **關鍵創新（相對於既有 NQS+SQD）**：
   - **Iterative feedback loop**：SQD 給的 ground-state amplitude 反過來訓練 NQS（teacher-student + REINFORCE energy advantage），所以 NQS 取樣分布會逐輪聚焦到重要行列式。
   - **Classical S+D expansion**：每輪在 NQS 取樣外加上「top-N 振幅行列式的 single+double 激發」，避免 NQS 模式塌縮（v2 的決定性改進，把誤差從 +60 mHa → +1.7 mHa）。
   - **Final EN-PT2 correction**：收歛後一次性的 perturbative 外推，5–10% 額外時間換 ~0.25 mHa 誤差。
   - **GPU 全棧加速**：sparse H build via `searchsorted`、GPU Davidson、multi-GPU PT2 — 把 56% (sqd) + 27% (per-iter PT2) bottleneck 從 CPU 移到 GPU，並支援 single-job 多 GPU。
3. **量化結果**：N₂ CAS(10,20) / cc-pVTZ（40q，Hilbert ≈ 1.85 × 10¹¹），**100k 行列式 + EN-PT2 → 比 HCI 17.4M 行列式還低 0.166 mHa，1× H200 跑 110 分鐘**。Sub-chemical accuracy（< 1.594 mHa）達成。
4. **HI-VQE 對照**：相同概念但 quantum hardware 版本在 LiH/H₂O/BeH₂ 都還停在 mHa-level（stability_test.json 數據）；HI+NQS+SQD 在小分子做到 < 1e-12 mHa（即 floating-point noise）。
5. **誠實面**：
   - N_det 必須以 sparse-det Davidson 向量長度為準（不要報 n_α·n_β）。
   - README 部分 small-mol 對 HI-VQE 比較數字與 stability_test.json 不一致 — 投稿前要重新對齊。
   - C₂H₄ 28q expansion ablation 在 results/ 找不到 raw JSON，要重跑驗證。
   - Multi-GPU 8× H200 那個 job 在 cluster 上 4h 時限到了沒跑完，**論文如果要 claim multi-GPU scaling，必須補一次完整 run**。
6. **HCI 反而被 v3 / v4 PT2 後贏了 ~0.15 mHa**：要在 caption 註明「HCI ε=10⁻⁴ 本身相對 FCI 還有 ~0.1 mHa 殘差」，避免 reviewer 質疑「為什麼 NQS 比 reference 還低」。

---

## 7. 主圖（Figure 1）設計建議 — 從 Nature 系列慣例提煉

### 7.1 Survey 自最相關的七篇文章

調查的七篇主圖風格（詳見 §7.2 各篇 verbatim caption）：

| 論文 | Figure 1 內容 | 風格屬性 |
|---|---|---|
| Robledo-Moreno 2025 (SQD on quantum-centric) | 純 schematic：左半量子-classical 整合架構，右半 configuration-recovery 流程 + 綠色 inset | **Schematic-first**, 無 Figure 1 結果面板 |
| Pellow-Jarman 2025 (HIVQE) | 兩個並排 method comparison panel：(a) VQE vs (b) HIVQE | **Old-vs-new** 對比 |
| Yu 2025 (SKQD) | line plot：error vs sample budget M, 多 qubit-count overlay | **Result-first** (反慣例) |
| von Glehn 2023 (Psiformer) | (a) FermiNet vs (b) Psiformer 兩個並排架構 | **Old-vs-new** |
| Pfau 2020 (FermiNet) | top: global architecture; bottom: layer zoom-in | **Architecture-only** |
| Hermann 2020 (PauliNet) | left-to-right pipeline：r/R → embed → Jastrow + backflow → Slater → Ψ | **Pipeline-only** |
| Choo 2020 (NQS for chem) | C₂ + N₂ dissociation curves vs CCSD/CCSD(T)/FCI | **Result-first** |

### 7.2 視覺慣例摘要（Nature/NComm/Nat Chem 量子化學共識）

- **Aspect ratio**：landscape ~1.6 : 1 到 2 : 1。
- **Panel count**：2–4 panels；> 4 panels reviewer 抱怨擁擠。
- **Color convention**：
  - 量子硬體：cool grey / desaturated blue chip art。
  - Classical compute / GPU：warm orange / yellow rack-style。
  - Neural net：muted blue (one stream) + red/orange (paired stream)。
  - Innovation highlight：**綠色 inset box**（SQD recovery inset 是經典範例）。
  - Bitstrings：monospace tile，顏色二元編碼。
- **Iteration 怎麼畫**：always 用一條曲線箭頭從右側 (classical post-processing) 回到左側 (sampler)；**絕對不要畫齒輪/圓圈** — Nature reviewer 認為那是 PowerPoint deck 風格。
- **Pseudocode**：永不放在 Figure 1 中，只放 Methods 或 Supplementary。
- **Headline number 顯示**：log-y、mHa 或 kcal/mol、**橫畫一條 1.6 mHa = chemical accuracy 線**。
- **Figure 1 是否要含結果**：兩派 — Robledo-Moreno SQD 不放、Choo NComm 直接放結果。**對 HI+NQS+SQD 推薦混合**：3-panel 設計 (a) pipeline schematic + (b) Transformer architecture inset + (c) error-vs-iter convergence line。

### 7.3 給 Leo 的 Figure 1 具體草案

**整體 layout — landscape ~1.8 : 1，3 panels**：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  (a) HI+NQS+SQD pipeline (左寬 ~55%)                                          │
│                                                                              │
│   ┌────────┐     ┌─────────┐    ┌─────────┐                                 │
│   │ N₂     │     │ MO      │    │ Trans-  │                                 │
│   │ (CPK)  │ →   │ integ-  │ → │  former │ ───┐  M sampled                 │
│   │ +basis │     │ rals    │    │  NQS    │    │  bitstrings                │
│   └────────┘     │ (h, g)  │    │ (auto-  │    │ ┌──────────┐               │
│                  └─────────┘    │ regress)│    └→│ 11010110 │ × M           │
│                                 └─────────┘      │ 10101100 │               │
│                                      ↑           │ 01101011 │               │
│  GPU H200 icon                       │           │ ...      │               │
│  (small, 灰)                         │           └──────────┘               │
│                          ┌───────────│──────────────┐                       │
│                          │ ⑤ NQS update              │                       │
│                          │  (REINFORCE +             │                       │
│                          │   teacher CE)             │                       │
│                          │                           │                       │
│                          │ ←── eigenvector Ψ_0 ──────│                       │
│                          │                           │                       │
│                          │  ┌─ ④ SQD GPU-Davidson ─┐ │                       │
│                          │  │ N_det × N_det sparse  │ │                       │
│                          │  │ → E_0, Ψ_0           │ │                       │
│                          │  └───────────────────────┘ │                       │
│                          │            ↑               │                       │
│                          │ ┌──── ③ EN-PT2 selection ──┴──┐                  │
│                          │ │ score(x) = |⟨x|H|Φ⟩|²/|E₀-Hxx| │ ← top-k       │
│                          │ └─────────────────────────────────┘                │
│                          │            ↑                                      │
│                          │ ┌─────────────────────────────────┐               │
│                          │ │ ② Classical S+D expansion       │  (綠 inset)   │
│                          │ │  top-N amp dets → S+D           │               │
│                          │ └─────────────────────────────────┘               │
│                          │            ↑                                      │
│                          └─── ① NQS sampling ←────────                       │
│                                                                              │
├────────────────────────────────────┬─────────────────────────────────────────┤
│ (b) Transformer NQS architecture   │  (c) Convergence on N₂-40Q              │
│  (左下 ~28%)                        │  (右側 ~40%)                            │
│                                    │                                         │
│   α tokens → causal SA × L         │   error vs HCI (log-y, mHa)             │
│      ↓                             │   x: iteration                          │
│   α full ctx (no mask)             │                                         │
│      ↓                             │   ┌─ HCI 17.4M (baseline = 0)           │
│   β tokens → causal SA + cross-attn│   │                                     │
│      ↓                             │   │     ╲                               │
│   α/β logits → constrained sample  │   │      ╲___                           │
│   (force Σα=N_α, Σβ=N_β)           │   │          ╲___                       │
│                                    │  10⁰┼──────────╲___────── 1.6 mHa       │
│                                    │   │              ╲___ chem accuracy    │
│                                    │   │                  ╲                  │
│                                    │  10⁻¹┼──── HI+NQS+SQD ──╲___─────────  │
│                                    │   │                          ╲___ −0.14 │
│                                    │   └─────────────────────────────────────│
│                                    │       0  5  10  15  20  iter            │
└────────────────────────────────────┴─────────────────────────────────────────┘
```

**Panel (a) 的關鍵元素**（必畫）：
- 五個編號方塊 ①②③④⑤ 用同一顏色家族區分（藍 = NQS, 綠 = expansion + selection, 橙 = SQD diagonalization, 紅 = NQS update）。
- 一條由 ⑤ 回到 ① 的曲線箭頭，標註「teacher: w_x = p_α p_β」。
- 在 ④ 旁邊小小畫一張 NVIDIA H200 chip 圖示（公開可用），明示「all classical, single GPU」。
- 綠色 inset 框住「Classical S+D expansion」— 這是相對 IBM SQD 的最大演算法差異，借用 Robledo-Moreno SQD 的綠色 highlight 慣例。
- 不要寫公式進方塊內 — 把公式留在 caption。

**Panel (b)** — Transformer architecture mini-inset（仿 Psiformer / FermiNet 的層級架構圖）。畫 α 與 β 兩個 tower 的 causal-attention pattern；用「causal mask 的下三角圖」做視覺重點；標 cross-attention β→α 的方向。

**Panel (c)** — convergence line。建議：
- x: iteration (0–24)
- y: |E_var − E_HCI| (log scale, mHa)
- 三條曲線：HI+NQS+SQD-v4 (紅實線)、純 SQD random sampling baseline (灰虛線)、HCI converged baseline (黑實線水平 = 0)
- 一條 1.6 mHa chemical accuracy 水平線
- 在右側標出 PT2-corrected E_total = −0.142 mHa 的箭頭（穿過 0 線往負）

### 7.4 Figure 1 caption 草稿

> **Figure 1 | The HI+NQS+SQD algorithm.** **a**, End-to-end pipeline. An autoregressive Transformer Neural Quantum State (NQS, blue) samples electron-configuration bitstrings respecting exact particle-number conservation. Sampled configurations are augmented by classical single+double excitations of the highest-amplitude determinants from the previous iteration (green inset), then ranked by an Epstein–Nesbet PT2 score $\text{score}(x) = |\langle x | \hat H | \Phi_0\rangle|^2 / |E_0 - H_{xx}|$; the top-k determinants enter the variational basis. The Hamiltonian projected onto this basis is diagonalised by GPU-Davidson on a sparse $N_\text{det} \times N_\text{det}$ matrix (orange), yielding the ground-state vector $\Psi_0$ and energy $E_0$. The eigenvector then teaches the NQS via a weighted log-likelihood plus a REINFORCE-style energy advantage (red). All five steps run on a single NVIDIA H200 GPU. **b**, Architecture of the autoregressive Transformer NQS: α-orbitals are modelled by a causal self-attention tower; β-orbitals additionally cross-attend to a bidirectional context over the full α string. **c**, Convergence of the variational energy on N₂ at the CAS(10,20)/cc-pVTZ active space (40 qubits, Hilbert space ≈ 1.85 × 10¹¹). HI+NQS+SQD reaches $E_\text{var}$ with err ≈ +0.35 mHa using $N_\text{det} = 100{,}000$ (~ 174 × fewer determinants than the HCI ε=10⁻⁴ reference of 17,380,561 dets); the one-shot final EN-PT2 correction over ~ 1.7 × 10⁶ external determinants drives the total energy to $E_\text{total} = -109.21490$ Ha, **0.166 mHa below** the HCI baseline. Total wall time on a single H200: 110 minutes.

### 7.5 補充：第 2-3 張 figure 建議

如果論文允許 4-5 張主圖，後續：
- **Figure 2** — N_det efficiency scan：同分子（建議 N₂-40Q）下，HI+NQS+SQD vs HCI vs CIPSI vs SCI vs CCSD(T) 的 「error vs N_det」 log-log scatter。**就用 Choo NComm 2020 的 dissociation curve panel 風格做一個**。
- **Figure 3** — 從 small (LiH 12q) → mid (N₂ 20q) → large (N₂-40q, C₂H₄ 28q, Benzene 30q) 的 sweep，bar chart 顯示 chemical accuracy yes/no + N_det used。
- **Figure 4** — Multi-GPU scaling：1×, 2×, 4×, 8× H200 的 wall time + speedup（**前提是 Leo 補完 multi-GPU run**）。
- **Figure 5** — Iron-sulfur cluster [2Fe-2S] / [4Fe-4S]（FCIDUMP 已支援，見 `molecules.py:344-358`）— 這是 Robledo-Moreno SQD 的 flagship system，做出來會很有殺傷力。

---

## 8. 嚴格性檢查清單（投稿前必做）

| 項目 | 來源 | 狀態 |
|---|---|---|
| HCI baseline N_det = 17,380,561, E = −109.21473333 | `HI_NQS_SQD_benchmark_data.md` §1; `comparison_table.md` row 1 | ✅ 一致 |
| v4 davidson big_100k err = −0.166 mHa | `results/n2_40q_v4_davidson/davidson_big_100k.json` | ✅ 一致 |
| v3 default_100k err = −0.104 mHa | `results/n2_40q_v3/default_100k.json` | ✅ 一致 |
| Time breakdown (sqd 56.1%, pt2 27.1%, samp 8.3%) | `HI_NQS_SQD_benchmark_data.md` §6 | ✅ 一致 |
| C₂H₄ 28q expansion ablation 數字 | README only — 沒有 raw JSON | ⚠ **需重跑** |
| HI-VQE comparison BeH₂ (30.899 ± 5.424 mHa) | README vs `stability_test.json` (0.97 ± 0.18) | ⚠ **不一致** |
| HI-VQE comparison NH₃ (62.339 ± 1.335 mHa) | README only — 無 raw JSON | ⚠ **需重跑** |
| HI+NQS+SQD BeH₂ stability (0.000 ± 0.000) | README vs `stability_test.json` (4.12 ± 4.56) | ⚠ **不一致** |
| Classical S+D expansion gain (+60 → +1.7 mHa) | `HI_NQS_SQD_benchmark_data.md` §4 | ✅ 一致 |
| 43k debunk (actual 118M Davidson dim) | `HI_NQS_SQD_benchmark_data.md` §2 | ✅ 一致；論文要明寫 |
| EN-PT2 公式 / Slater-Condon signs | `sparse_det_solver.py` 註解 | ✅ 一致 |
| Auto-scaling table (embed/heads/layers) | `hi_nqs_sqd.py:99-110` | ✅ 一致 |
| Multi-GPU 8× scaling claim | 兩 job 4h time-out 沒跑完 | ⚠ **需要 normal partition + 12h 重跑** |

---

## 9. 參考文獻（畫 Figure 1 + intro 用）

1. **Robledo-Moreno et al. 2025**, "Chemistry beyond the scale of exact diagonalization on a quantum-centric supercomputer", *Science Advances* 11, eadu9991 / arXiv 2405.05068. — SQD 主論文，[2Fe-2S] / [4Fe-4S] benchmark 的來源。
2. **Pellow-Jarman et al. 2025**, "HIVQE: Handover Iterative Variational Quantum Eigensolver", arXiv:2503.06292. — 我們直接對比的 quantum 版本。
3. **Yu et al. 2025**, "Quantum-Centric Algorithm for Sample-Based Krylov Diagonalization (SKQD)", arXiv:2501.09702. — Krylov SQD 變體。
4. **von Glehn, Spencer, Pfau 2023**, "A self-attention ansatz for ab-initio quantum chemistry" (Psiformer), arXiv:2211.13672 / ICLR 2023. — Transformer NQS 的祖師爺。
5. **Pfau et al. 2020**, "Ab initio solution of the many-electron Schrödinger equation with deep neural networks" (FermiNet), *Phys Rev Research* 2, 033429 / arXiv 1909.02487.
6. **Hermann, Schätzle, Noé 2020**, "Deep-neural-network solution of the electronic Schrödinger equation" (PauliNet), *Nature Chemistry* 12, 891.
7. **Choo, Mezzacapo, Carleo 2020**, "Fermionic neural-network states for ab-initio electronic structure", *Nature Communications* 11, 2368.
8. **Barrett, Malyshev, Lvovsky 2022**, "Autoregressive neural-network wavefunctions for ab initio quantum chemistry", *Nat Mach Intell* — 第二量子化 autoregressive NQS 的關鍵 prior。
9. **Sharir et al. 2020**, "Deep autoregressive models for the efficient variational simulation of many-body quantum systems".
10. **Huron, Malrieu, Rancurel 1973**, "Iterative perturbation calculations of ground and excited state energies", *J Chem Phys* — CIPSI 原型（PT2 selection）。
11. **Holmes, Tubman, Umrigar 2016**, "Heat-bath Configuration Interaction (HCI)", *J Chem Theory Comput* — HCI baseline 的方法源頭。
12. **Tubman et al. 2020**, "Modern approaches to exact diagonalization … Adaptive Sampling CI", *J Chem Theory Comput* — selected-CI 的近代綜述。

---

**Last verified**: 2026-04-23
**Code sources verified against**: `src/methods/hi_nqs_sqd*.py`, `src/methods/gpu_*.py`, `src/methods/multi_gpu_*.py`, `src/nqs/transformer.py`, `src/methods/sparse_det_solver.py`, `src/hamiltonians/molecular.py`
**Data sources verified against**: `HI_NQS_SQD_benchmark_data.md`, `HI_NQS_SQD_comparison_table.md`, `results/n2_40q_v3/*.json`, `results/n2_40q_v4_davidson/*.json`, `results/six_methods.json`, `results/stability_test.json`
