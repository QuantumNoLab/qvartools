# HI+NQS+SQD N2-40Q Benchmark Data

System: N2-CAS(10,20), cc-pVTZ, HF canonical orbitals
Reference: HCI (PySCF selected_ci, ε=1e-4) = **-109.21473333 Ha**, ndets=17,380,561, t=285s
GPU: NVIDIA H200 80GB, 1 task = 1 GPU unless noted

## 1. Small-molecule N_det scan (STO-3G, FCI-tractable)

Goal: compare N_det required for chemical accuracy across HCI / CIPSI / HI+NQS+SQD.

### Per-molecule summary (vs FCI, mHa)

| Mol | Hilbert | FCI (Ha) | Method | N_det | err (mHa) |
|---|---:|---|---|---:|---:|
| **H2O** | 441 | -75.013155 | FCI | 441 | 0 (ref) |
| | | | HCI ε=1e-02 | 169 | +0.121 |
| | | | HCI ε=1e-03 | 361 | +0.003 |
| | | | HCI ε=1e-04 | 441 | +0.000 |
| | | | HCI ε=1e-05 | 441 | +0.000 |
| | | | HCI ε=1e-06 | 441 | +0.000 |
| | | | CIPSI budget=30 | 30 | +0.262 |
| | | | CIPSI budget=60 | 60 | +0.007 |
| | | | CIPSI budget=120 | 120 | +0.000 |
| | | | CIPSI budget=250 | 133 | +0.000 |
| | | | CIPSI budget=400 | 133 | +0.000 |
| | | | HI+NQS+SQD (v1, s=42) bud=30 | 30 | +0.331 |
| | | | HI+NQS+SQD (v1, s=42) bud=60 | 60 | +0.014 |
| | | | HI+NQS+SQD (v1, s=42) bud=120 | 120 | +0.000 |
| | | | HI+NQS+SQD (v1, s=42) bud=250 | 250 | +0.000 |
| | | | HI+NQS+SQD (v1, s=42) bud=400 | 400 | -0.000 |
| | | | HI+NQS+SQD (v1, s=2024) bud=30 | 30 | +0.331 |
| | | | HI+NQS+SQD (v1, s=2024) bud=60 | 60 | +0.033 |
| | | | HI+NQS+SQD (v1, s=2024) bud=120 | 120 | +0.000 |
| | | | HI+NQS+SQD (v1, s=2024) bud=250 | 250 | -0.000 |
| | | | HI+NQS+SQD (v1, s=2024) bud=400 | 400 | -0.000 |
| | | | HI+NQS+SQD (v1, s=777) bud=30 | 30 | +0.331 |
| | | | HI+NQS+SQD (v1, s=777) bud=60 | 60 | +0.013 |
| | | | HI+NQS+SQD (v1, s=777) bud=120 | 120 | +0.000 |
| | | | HI+NQS+SQD (v1, s=777) bud=250 | 250 | -0.000 |
| | | | HI+NQS+SQD (v1, s=777) bud=400 | 400 | +0.000 |
| **BeH2** | 1,225 | -15.595118 | FCI | 1,225 | 0 (ref) |
| | | | HCI ε=1e-02 | 100 | +0.697 |
| | | | HCI ε=1e-03 | 484 | +0.014 |
| | | | HCI ε=1e-04 | 1,024 | +0.000 |
| | | | HCI ε=1e-05 | 1,225 | +0.000 |
| | | | HCI ε=1e-06 | 1,225 | +0.000 |
| | | | CIPSI budget=50 | 50 | +0.024 |
| | | | CIPSI budget=100 | 100 | +0.001 |
| | | | CIPSI budget=250 | 169 | -0.000 |
| | | | CIPSI budget=500 | 169 | -0.000 |
| | | | CIPSI budget=1000 | 169 | -0.000 |
| | | | HI+NQS+SQD (v1, s=42) bud=50 | 50 | +0.081 |
| | | | HI+NQS+SQD (v1, s=42) bud=100 | 100 | +0.062 |
| | | | HI+NQS+SQD (v1, s=42) bud=250 | 250 | +0.002 |
| | | | HI+NQS+SQD (v1, s=42) bud=500 | 500 | +0.001 |
| | | | HI+NQS+SQD (v1, s=42) bud=1000 | 1,000 | +0.000 |
| | | | HI+NQS+SQD (v1, s=2024) bud=50 | 50 | +4.186 |
| | | | HI+NQS+SQD (v1, s=2024) bud=100 | 100 | +0.123 |
| | | | HI+NQS+SQD (v1, s=2024) bud=250 | 250 | +0.062 |
| | | | HI+NQS+SQD (v1, s=2024) bud=500 | 500 | +0.121 |
| | | | HI+NQS+SQD (v1, s=2024) bud=1000 | 1,000 | +0.000 |
| | | | HI+NQS+SQD (v1, s=777) bud=50 | 50 | +3.969 |
| | | | HI+NQS+SQD (v1, s=777) bud=100 | 100 | +0.006 |
| | | | HI+NQS+SQD (v1, s=777) bud=250 | 250 | +0.005 |
| | | | HI+NQS+SQD (v1, s=777) bud=500 | 500 | +0.004 |
| | | | HI+NQS+SQD (v1, s=777) bud=1000 | 1,000 | +0.000 |
| **NH3** | 3,136 | -55.517816 | FCI | 3,136 | 0 (ref) |
| | | | HCI ε=1e-02 | 576 | +0.484 |
| | | | HCI ε=1e-03 | 2,025 | +0.016 |
| | | | HCI ε=1e-04 | 3,136 | +0.000 |
| | | | HCI ε=1e-05 | 3,136 | +0.000 |
| | | | HCI ε=1e-06 | 3,136 | +0.000 |
| | | | CIPSI budget=100 | 100 | +0.816 |
| | | | CIPSI budget=250 | 250 | +0.247 |
| | | | CIPSI budget=500 | 500 | +0.132 |
| | | | CIPSI budget=1000 | 1,000 | +0.126 |
| | | | CIPSI budget=2000 | 1,576 | +0.126 |
| | | | HI+NQS+SQD (v1, s=42) bud=100 | 100 | +1.270 |
| | | | HI+NQS+SQD (v1, s=42) bud=250 | 250 | +0.958 |
| | | | HI+NQS+SQD (v1, s=42) bud=500 | 500 | +1.266 |
| | | | HI+NQS+SQD (v1, s=42) bud=1000 | 1,000 | +0.210 |
| | | | HI+NQS+SQD (v1, s=42) bud=2000 | 2,000 | +0.036 |
| | | | HI+NQS+SQD (v1, s=2024) bud=100 | 100 | +1.186 |
| | | | HI+NQS+SQD (v1, s=2024) bud=250 | 250 | +0.651 |
| | | | HI+NQS+SQD (v1, s=2024) bud=500 | 500 | +0.423 |
| | | | HI+NQS+SQD (v1, s=2024) bud=1000 | 1,000 | +0.065 |
| | | | HI+NQS+SQD (v1, s=2024) bud=2000 | 2,000 | +0.000 |
| | | | HI+NQS+SQD (v1, s=777) bud=100 | 100 | +12.727 |
| | | | HI+NQS+SQD (v1, s=777) bud=250 | 250 | +11.514 |
| | | | HI+NQS+SQD (v1, s=777) bud=500 | 500 | +11.290 |
| | | | HI+NQS+SQD (v1, s=777) bud=1000 | 1,000 | +0.132 |
| | | | HI+NQS+SQD (v1, s=777) bud=2000 | 2,000 | +0.104 |
| **CH4** | 15,876 | -39.806035 | FCI | 15,876 | 0 (ref) |
| | | | HCI ε=1e-02 | 1,089 | +1.219 |
| | | | HCI ε=1e-03 | 7,225 | +0.015 |
| | | | HCI ε=1e-04 | 14,400 | +0.000 |
| | | | HCI ε=1e-05 | 15,876 | +0.000 |
| | | | HCI ε=1e-06 | 15,876 | +0.000 |
| | | | CIPSI budget=200 | 200 | +0.731 |
| | | | CIPSI budget=500 | 500 | +0.118 |
| | | | CIPSI budget=1500 | 1,500 | +0.001 |
| | | | CIPSI budget=5000 | 3,996 | -0.000 |
| | | | CIPSI budget=10000 | 3,996 | -0.000 |
| | | | HI+NQS+SQD (v1, s=42) bud=200 | 200 | +33.517 |
| | | | HI+NQS+SQD (v1, s=42) bud=500 | 500 | +33.403 |
| | | | HI+NQS+SQD (v1, s=42) bud=1500 | 1,500 | +5.651 |
| | | | HI+NQS+SQD (v1, s=42) bud=5000 | 5,000 | +0.207 |
| | | | HI+NQS+SQD (v1, s=42) bud=10000 | 10,000 | +0.070 |
| | | | HI+NQS+SQD (v1, s=2024) bud=200 | 200 | +11.212 |
| | | | HI+NQS+SQD (v1, s=2024) bud=500 | 500 | +6.085 |
| | | | HI+NQS+SQD (v1, s=2024) bud=1500 | 1,500 | +0.894 |
| | | | HI+NQS+SQD (v1, s=2024) bud=5000 | 5,000 | +0.153 |
| | | | HI+NQS+SQD (v1, s=2024) bud=10000 | 10,000 | +0.002 |
| | | | HI+NQS+SQD (v1, s=777) bud=200 | 200 | +9.180 |
| | | | HI+NQS+SQD (v1, s=777) bud=500 | 500 | +10.524 |
| | | | HI+NQS+SQD (v1, s=777) bud=1500 | 1,500 | +1.359 |
| | | | HI+NQS+SQD (v1, s=777) bud=5000 | 5,000 | +0.185 |
| | | | HI+NQS+SQD (v1, s=777) bud=10000 | 10,000 | +0.027 |
| **N2** | 14,400 | -107.654122 | FCI | 14,400 | 0 (ref) |
| | | | HCI ε=1e-02 | 529 | +6.721 |
| | | | HCI ε=1e-03 | 7,396 | +0.005 |
| | | | HCI ε=1e-04 | 12,100 | +0.000 |
| | | | HCI ε=1e-05 | 13,689 | +0.000 |
| | | | HCI ε=1e-06 | 14,400 | +0.000 |
| | | | CIPSI budget=200 | 200 | +0.242 |
| | | | CIPSI budget=500 | 500 | +0.007 |
| | | | CIPSI budget=1500 | 1,500 | +0.000 |
| | | | CIPSI budget=5000 | 1,824 | -0.000 |
| | | | CIPSI budget=10000 | 1,824 | -0.000 |
| | | | HI+NQS+SQD (v1, s=42) bud=200 | 200 | +343.787 |
| | | | HI+NQS+SQD (v1, s=42) bud=500 | 500 | +345.788 |
| | | | HI+NQS+SQD (v1, s=42) bud=1500 | 1,500 | +16.312 |
| | | | HI+NQS+SQD (v1, s=42) bud=5000 | 5,000 | +0.492 |
| | | | HI+NQS+SQD (v1, s=42) bud=10000 | 10,000 | +0.036 |
| | | | HI+NQS+SQD (v1, s=2024) bud=200 | 200 | +316.849 |
| | | | HI+NQS+SQD (v1, s=2024) bud=500 | 500 | +325.212 |
| | | | HI+NQS+SQD (v1, s=2024) bud=1500 | 1,500 | +51.729 |
| | | | HI+NQS+SQD (v1, s=2024) bud=5000 | 5,000 | +0.198 |
| | | | HI+NQS+SQD (v1, s=2024) bud=10000 | 10,000 | +0.124 |

**Key finding**: On STO-3G small molecules, CIPSI is the most det-efficient,
then HCI, then HI+NQS+SQD v1. Example: N2 20Q, CIPSI at 200 dets gives +0.24 mHa;
v1 NQS-SQD needs 5000 dets for similar accuracy. This is v1 baseline on tiny systems.

## 2. The "43k" claim debunked (N2-40Q HF orbitals)

Old claim: HI+NQS+SQD reaches -109.2146 with "43k basis", 402x fewer than HCI's 17.4M dets.
Issue: "43k" = unique (α,β) pairs, but the `incremental_sqd` backend diagonalises in the
  full n_α × n_β product space. Measurement job 21794 reproduced the original run with
  the Davidson-dim printed per iter:

### Actual Davidson dim of the old "43k" run (per iter)

| iter | bitstrings | n_α | n_β | **Davidson dim (n_α × n_β)** |
|---:|---:|---:|---:|---:|
| 0 | 10,000 | 1,126 | 1,126 | **1,267,876** |
| 1 | 10,000 | 6,270 | 6,270 | **39,312,900** |
| 2 | 20,000 | 9,950 | 9,950 | **99,002,500** |
| 3 | 30,000 | 10,704 | 10,704 | **114,575,616** |
| 4 | 40,000 | 10,860 | 10,860 | **117,939,600** |
| 5 | 42,521 | 10,873 | 10,873 | **118,222,129** |
| 6 | 42,880 | 10,873 | 10,873 | **118,222,129** |
| 7 | 42,935 | 10,873 | 10,873 | **118,222,129** |
| 8 | 42,957 | 10,873 | 10,873 | **118,222,129** |
| **final** | **42,957** | | | |

Final energy: E = -109.2144853315  (err vs HCI: +0.248 mHa)

**Conclusion**: the old "43k" run really needed 118 million dets in its Davidson
  subspace — 6.8x MORE than HCI's 17.4M, not 402x fewer. The whole premise of that
  claim was wrong.

## 3. v1 with honest N_det (sparse_det backend)

Replaced `incremental_sqd` (product-space Davidson) with `sparse_det` (N_det Davidson).
Same NQS sampling, now honest measurement.

| Config | N_det | err vs HCI (mHa) | wall |
|---|---:|---:|---:|
| mb_100k | 100,000 | +52.80 | 16064s |
| mb_200k | 200,000 | +15.74 | 13650s |
| s_200k | 61,688 | +35.59 | 3103s |
| tk_50k | 315,178 | +13.40 | 15405s |
| var_cold | 227,589 | +46.15 | 7915s |
| var_slowlr | 532,890 | +11.13 | 17745s |
| var_steps10 | 155,116 | +16.52 | 10845s |

**v1 plateaus at +11 mHa even with 500k+ dets** — NQS sampling alone can't find
enough important dets. This motivated v2 and v3.

## 4. v2: classical_seed + classical_expansion ablation

Two improvements tested separately:
- **classical_seed**: iter 0 starts from HF + all singles + all doubles of HF
  (classical enumeration, ~1k dets), then NQS fills up to top_k.
- **classical_expansion**: each iter, after SQD, enumerate H-connections of top-N
  amplitude dets in current basis, add to candidate pool for next iter's PT2 scoring.

Variants: v1 (neither), v2_seed (A only), v2_expand (B only), v2_full (both).
Same seed 42 across all.

### Results @ same budget (n_samples/top_k/max_basis)

| Variant | 100k/10k/100k | 200k/20k/200k | 500k/50k/500k | 1m/100k/1m |
|---|---|---|---|---|
| v1 | +60.00 mHa / 32k dets / 29m | +60.00 mHa / 106k dets / 82m | +17.27 mHa / 121k dets / 96m | — |
| v2_seed | +19.04 mHa / 35k dets / 17m | +18.34 mHa / 78k dets / 40m | +15.26 mHa / 398k dets / 259m | +12.58 mHa / 487k dets / 371m |
| v2_expand | +1.74 mHa / 62k dets / 22m | +1.59 mHa / 76k dets / 25m | +1.31 mHa / 141k dets / 59m | +1.22 mHa / 572k dets / 360m |
| v2_full | +1.83 mHa / 66k dets / 25m | — | +1.71 mHa / 254k dets / 138m | +1.71 mHa / 190k dets / 99m |

**Key insight**: classical_expansion is the critical mechanism. Seed-only plateaus at
+19 mHa (NQS still mode-collapses). Expansion alone reaches +1.6 mHa. Full is same as
expand-only (seed doesn't help once expansion is on). **Use v2_expand or v3 in practice.**

## 5. v3: v2_expand + Epstein-Nesbet PT2 correction

After variational convergence, enumerate H-connections of top-N amp basis dets,
compute `E_PT2 = Σ |<x|H|Ψ_0>|² / (E_0 - H_xx)` over all external dets.

Variants:
- `default`: classical_expansion_top_n=1000, pt2_top_n=5000, PT2 on
- `nopt2`: same but PT2 off (ablate PT2)
- `small`: classical_expansion_top_n=200 (cheapest)
- `big`: classical_expansion_top_n=2000, pt2_top_n=10000

### All v3 results (14 runs)

| Variant | Budget | basis | N_PT2_externals | err_var | **err_total** | wall |
|---|---|---:|---:|---:|---:|---:|
| big | 100k | 100,000 | 1,704,900 | +0.355 | **-0.128** | 136m |
| big | 1m | 658,098 | 1,275,833 | -0.043 | **-0.158** | 583m |
| big | 200k | 200,000 | 1,603,489 | +0.051 | **-0.149** | 322m |
| big | 500k | 500,000 | 1,302,615 | -0.040 | **-0.158** | 664m |
| default | 100k | 100,000 | 989,264 | +0.384 | **-0.104** | 102m |
| default | 200k | 200,000 | 888,741 | +0.159 | **-0.125** | 214m |
| default | 500k | 449,551 | 816,821 | +0.138 | **-0.130** | 536m |
| nopt2 | 100k | 100,000 | 0 | +0.384 | **+0.384** | 89m |
| nopt2 | 200k | 200,000 | 0 | +0.159 | **+0.159** | 258m |
| nopt2 | 500k | 369,972 | 0 | +0.140 | **+0.140** | 365m |
| small | 100k | 100,000 | 989,085 | +1.015 | **-0.140** | 113m |
| small | 1m | 447,430 | 942,545 | +0.641 | **-0.136** | 688m |
| small | 200k | 124,627 | 990,133 | +0.984 | **-0.139** | 179m |
| small | 500k | 405,689 | 966,894 | +0.771 | **-0.136** | 546m |

**Headline**: v3 default @ 100k basis reaches **-0.10 mHa vs HCI** (174x fewer
dets than HCI's 17.4M). v3 big @ 100k reaches -0.13 mHa. All budgets saturate near
-0.15 mHa; 100k is sufficient. Note err_total < 0 means lower energy than HCI's
variational — consistent with HCI not being fully converged to FCI.

## 6. Per-phase time breakdown (v3 default_100k as example)

| Phase | Time | % of total | What it does |
|---|---:|---:|---|
| sqd | 3445s | 56.1% | sparse_det Davidson (scipy eigsh) |
| pt2 | 1661s | 27.1% | per-iter PT2 scoring on candidates |
| samp | 512s | 8.3% | NQS Transformer sampling |
| final_PT2 | 363s | 5.9% | one-shot Epstein-Nesbet PT2 correction |
| exp | 106s | 1.7% | classical singles+doubles expansion |
| upd | 51s | 0.8% | NQS backprop training |
| TOTAL | 6139s | 100% | (22 iterations) |

## 7. Version comparison on N2-40Q HF

All on the same system, same reference, same random seed (42).

| Method | Variational N_det | err_var (mHa) | +PT2 correction? | err_total (mHa) | wall |
|---|---:|---:|:-:|---:|---:|
| HCI (ε=1e-4) | 17,380,561 | 0 (baseline) | — | 0 | 285s |
| OLD v1 incremental_sqd ("43k claim") | **118,222,129** actual | +0.25 | — | +0.25 | 4.7h |
| v1 sparse_det, 100k budget | 32,904 | +60.00 | — | +60.00 | 29m |
| v1 sparse_det, 500k budget | 121,565 | +17.27 | — | +17.27 | 96m |
| v2_expand 100k | 62,466 | +1.74 | — | +1.74 | 22m |
| v2_expand 500k | 141,823 | +1.31 | — | +1.31 | 59m |
| **v3 default 100k** | **100,000** | +0.38 | ✅ 989,264 externals | **-0.10** | 102m |
| **v3 default 200k** | **200,000** | +0.16 | ✅ 888,741 externals | **-0.12** | 214m |
| **v3 default 500k** | **449,551** | +0.14 | ✅ 816,821 externals | **-0.13** | 536m |

## 8. Research-claim evolution

| Generation | Claim | Verdict |
|---|---|---|
| OLD (pre-bug-find) | "43k dets beats HCI 17.4M, 402x fewer" | **FALSE**: actual Davidson dim was 118M (6.8x MORE than HCI) |
| v1 honest (sparse_det) | Uses honest N_det but plateaus at +11 mHa with 500k dets | Negative result: NQS alone insufficient |
| v2_expand | Classical expansion reaches +1.3 mHa at 142k dets | Chemical accuracy with ~120x fewer dets than HCI |
| **v3 (+ PT2)** | **sub-chemical accuracy (-0.10 to -0.16 mHa) at 100k dets** | **174x fewer variational dets than HCI, tighter than HCI's own bound** |

Variational matrix diagonalised: 100k × 100k sparse.
Perturbative sum over ~1M external dets (cheap scalar operations, no matrix).
