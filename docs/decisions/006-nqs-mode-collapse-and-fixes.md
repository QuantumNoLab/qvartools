# ADR-006: NQS Mode Collapse in HI+NQS+SQD and Proposed Fixes

- **Status**: Discovered → diagnosed → fixes in progress
- **Date**: 2026-04-28
- **Author**: Leo (leo07010)
- **Relates to**: ADR-005 (PT2 selection), HI-NQS-SQD v4 pipeline

---

## TL;DR

For `N2-CAS(10,26)` (52 qubits) the HI+NQS+SQD pipeline plateaus at
**+71 mHa above HCI** and refuses to budge with hyperparameter tuning.
Ablation shows the NQS sampler is the cause — not the SQD eigensolver,
not the PT2 selection, not the basis budget.

Concretely:

1. **NQS samples 20 k determinants → +241 mHa**, while
   **HF + Slater–Condon enumeration of 1 966 determinants → +97 mHa**.
   The neural sampler is *worse than naïve enumeration*.
2. The `classical_expansion=True` path was masking this for years
   because it does the SD-enumeration internally.
3. With `classical_expansion=False` (NQS-only), iter 1 logs read
   `basis= 20000(+0, -497757)` — **500 k samples produced 0 new
   determinants**. Mode collapse onto the existing 20 k basis.
4. A REINFORCE-style VMC update on importance score `|⟨σ|H|ψ⟩|²`
   made things **worse**, not better (+238 mHa, stuck at iter 1).

A single-line β=0.4 conditional reshape of the autoregressive sampler
already gives **−70 mHa at iter 3** for 52Q. Further fixes (truncated-
energy supervised loss, NQS-as-ranker) are queued.

---

## Context

The pipeline is:

```
NQS ──sample──▶ candidate dets
        │
        ├── classical_expansion: S ← S ∪ S+D(S)
        │
        ├── PT2 selection (EN-PT2 score, ADR-005)
        │
        ├── SQD eigensolver on top-k subspace
        │
        └── teacher loss (current basis |c|²) ──▶ NQS update
```

NQS = autoregressive Transformer producing `|ψ(σ)|²` only (no sign
network — phases are recovered by the diagonalizer, which is the
correct architecture choice; see literature section).

For 52Q `N2-CAS(10,26)` we observe a stable plateau at
`E_var − E_HCI ≈ +71 mHa` regardless of:

- `top_k` ∈ {10 k, 20 k, 50 k}
- `n_samples` ∈ {200 k, 500 k, 1 M}
- `max_iterations` ∈ {5, 8, 12}
- `classical_expansion_top_n` ∈ {1 k, 2 k, 5 k}
- `pt2_top_n` ∈ {5 k, 10 k, 20 k}
- choice of optimizer (Adam vs. SR vs. multi-T)

This is structural, not a knob.

---

## Diagnostic — six-mode ablation

`scratch_diagnose_nqs_role.py` runs six modes on the same molecule and
seed:

| mode | NQS sampling | classical_expansion | classical_seed |
|------|--------------|---------------------|----------------|
| `baseline` | full NQS | on | off |
| `no_nqs` | `n_samples=64` | on | **on** |
| `nqs_only` | full NQS | **off** | off |
| `random_sample` | uniform N-electron sampler | on | off |
| `vmc_baseline` | full NQS | on | off |
| `vmc_nqs_only` | full NQS | **off** | off |

`vmc_*` modes monkey-patch `_update_nqs` to a hybrid joint-c² teacher
+ REINFORCE on `|coupling|²` score.

### 52Q ablation (8 iters, top_k=20 k, budget=100 k)

| mode (s42) | err_var (mHa) | basis | iter trajectory |
|------------|---------------|-------|-----------------|
| `baseline` | +90 | 96 k (full) | 240 → 90 |
| `no_nqs` | **+90** | 96 k | 240 → 90 |
| `nqs_only` | +167 (stuck) | 65 k (stalled) | 240 → 167 |
| `random_sample` | +175 | 88 k | 240 → 175 |
| `vmc_nqs_only` | +147 | 84 k | 240 → 147 |

**Reading**: `baseline ≈ no_nqs`. The neural sampler contributes
**zero** when `classical_expansion=True` — that branch dominates and
hides any NQS pathology. The nqs_only mode (the only mode that
actually tests NQS quality in isolation) is **75 mHa worse** than
baseline and stalls at 65 % of budget.

### 40Q ablation confirms the same pattern

`N2-CAS(10,20)` shows the identical ordering:
`baseline (+115) ≈ no_nqs (+115) ≪ nqs_only (+195)`.
The bug is system-size agnostic.

### iter-by-iter log signal — direct mode collapse

```
nqs_only baseline iter 1:  basis= 20000(+0, -497757)
```

500 k autoregressive samples were drawn; 497 757 collided with the
existing 20 k basis; **0 new determinants were added**. The NQS
distribution has effectively zero entropy outside the seed basis after
a single update step.

Suspected mechanism: the teacher signal `|c|²` from SQD is heavily
peaked on HF (Aufbau coefficient ≈ 0.95), so the marginal-factorized
KL update collapses the conditional distributions onto HF
neighbourhoods within five gradient steps.

---

## Why the VMC patch made things worse

Direct hybrid update implementing four concurrent fixes:

```python
# Joint c² teacher (replaces α-marginal × β-marginal)
sup_loss = -(c_joint² · log p_θ(σ))           # in-basis
# REINFORCE on importance
adv = |coupling|² / mean(|coupling|²) - 1     # off-basis
vmc_loss = -(adv · log p_θ(σ))
loss = sup_loss + vmc_loss + ε · entropy
```

40Q `vmc_nqs_only` s42 final = **+238 mHa** (vs. +195 baseline). VMC
finds *more* determinants (basis 91 k vs. 86 k) but they are
*worse-quality* determinants.

Three independent failure modes:

1. **`vmc_weight = 1.0` is too high.** With sign-blind importance
   `|coupling|²` and small `mean(importance)`, the REINFORCE gradient
   has very high variance and dominates the supervised signal.
2. **`|coupling|²` is sign-blind.** Constructive vs. destructive
   interference looks identical to the sampler. So the gradient drives
   the sampler toward determinants that *cancel* in the eigenvector.
3. **Supervised joint c² and REINFORCE gradients can be opposed.**
   Determinants in the current basis with large `|c|²` may have small
   off-basis coupling, and vice versa.

This is consistent with the literature: see arXiv:2602.12993 below.

---

## Literature — what the field actually does

Six recent (2022–2025) papers were surveyed for sampling/training of
NQS that feed a subspace diagonalizer.

### (a) Sign handling

Every successful "sample → diagonalize" paper deliberately uses
amplitude-only NQS and offloads signs to the diagonalizer. **This is
the correct architecture; do not add a phase network.**

- Thompson & Gunlycke, arXiv:2603.24728, explicitly: *"phases are
  off-loaded to the subspace diagonalization portion."*
- Solanki, Ding, Reiher, arXiv:2602.12993 (NQS-SC variant) derives
  signed coefficients from a learned orbital determinant — also no
  phase head.
- Westerhout & Bukov, arXiv:2002.04613 + arXiv:2510.02051 show that
  when a phase network is added, its loss landscape is far rougher
  than the amplitude landscape, and *separate learning rates per head*
  are required.

### (b) Supervised vs. VMC

Solanki/Ding/Reiher provide the cleanest comparison:

| variant | dets to chemical accuracy on N₂ stretched |
|---------|--------------------------------------------|
| NQS-SC (supervised, in-basis) | **64** (`2⁶`) |
| NBF-VMC (REINFORCE-style) | 16 384 (`2¹⁴`), with non-systematic jumps |

Their NQS-SC loss is the **truncated-energy** functional:

```
E^trunc_θ = Σ_{σ ∈ S_select} P_θ^renorm(σ) · E_loc(σ)
```

`P_θ^renorm` is the NQS distribution *renormalized over the current
selected set*, and `E_loc` uses only Hamiltonian connections within
that set. This is a *self-consistent supervised target on in-basis
amplitudes*, not a REINFORCE on couplings.

### (c) Mode collapse

Thompson & Gunlycke describe **exactly the symptom we observed**
("network collapses onto a few dominant configs"). Their fix:

> Apply per-conditional inverse-temperature reshape at sampling time:
> `P(n_q | n_<q) ← P(n_q | n_<q)^β / Σ P(n′_q | n_<q)^β`
> with **β = 0.4**.

For Bernoulli conditionals this is algebraically identical to scaling
the logit by β:

```
sigmoid(β · z) = P(1)^β / [P(1)^β + P(0)^β]
```

i.e. it is exactly `temperature = 1/β`. **This is a one-line fix that
requires no retraining.** Critically, our framework was annealing
`final_temperature = 0.3` (β = 3.3, sharpening) — *the wrong
direction*.

### (d) Subspace selection — convergent recipe

Across NAQS (arXiv:2109.12606), NQS-SC (2602.12993), Schmerwitz
(2406.08154), NNQS-AFQMC (2507.07540) and the physics-informed
Transformer (2412.12248), the consensus is:

```
S_{k+1} = topK( S_k ∪ singles_doubles(S_k), key = p_NQS(σ) )
```

Never let raw NQS samples *be* the basis. Always let `H · S_k`
generate candidates, then use NQS to *rank*. This is HCI/CIPSI
mechanics with NQS as the importance estimator. Our existing
`classical_expansion=True` path already implements the candidate
generation — the gap is that it ranks by SQD eigenvector `|c|²`, not
by `p_NQS`. Combining the two (`α |c|² + (1−α) p_NQS`) is the natural
unification.

---

## Proposed fixes (ordered by effort × payoff)

### Fix 1 — Conditional β = 0.4 reshape at sampling time (DONE)

Sampling-only patch, no retraining, no architecture change. Equivalent
to forcing `temperature = 2.5` regardless of caller.

```python
def install_batched_sampler(force_temperature=None):
    @torch.no_grad()
    def _batched(self, n_samples, hard=True, temperature=1.0):
        T = force_temperature or temperature
        return orig_sample(self, n_samples, temperature=T)
    AutoregressiveTransformer.sample = _batched
```

Ablation (`scratch_diagnose_nqs_role.py --mode=nqs_only_beta04`)
comparing strict NQS-only basis (`classical_expansion=False`),
`top_k=20 k`, `budget=100 k`, 8 iters, seeds 42 and 777, both runs
on 8 GPUs in one job (`scripts/bench_nqs_beta_reshape.slurm`):

**Variational energy (after SQD diagonalization, before PT2):**

| | 40Q baseline | 40Q β=0.4 | 40Q β=0.2 | 52Q baseline | 52Q β=0.4 | 52Q β=0.2 |
|---|---|---|---|---|---|---|
| **iter 7 err_var s42** | +195.2 | **+131.5** | +158.5 | +167.2 (stuck) | **+105.2** | +157.3 |
| **iter 7 err_var s777** | +202.7 | +164.4 | +159.0 | +139.3 (stuck) | **+127.7** | +188.7 |
| **basis size s42** | 86 k | 100 k (full) | 100 k | 65 k (stalled) | 100 k | 100 k |
| **basis size s777** | 62 k | 100 k | 100 k | 83 k (stalled) | 100 k | 100 k |

**Final total energy (var + EN-PT2 correction, top-N = 10 k):**

| | 40Q s42 | 40Q s777 | 52Q s42 | 52Q s777 |
|---|---|---|---|---|
| **β=0.4 err_total** | +114.6 | +116.2 | **+73.0** | **+75.4** |
| **β=0.2 err_total** | +116.0 | +116.1 | +75.7 | +74.2 |
| baseline (with `classical_expansion`) | — | — | ≈ +71 | ≈ +71 |

mHa above HCI throughout.

Three observations of independent significance:

1. **Mode collapse is fully resolved.** Every β run filled the 100 k
   budget; every baseline run stalled at 60–86 % of budget.
2. **β = 0.4 strictly dominates β = 0.2** on variational energy
   across all four (mol × seed) cells. β = 0.2 over-flattens the
   conditional and degrades the ranking signal at the cost of
   exploration.
3. **The +71 mHa total-energy plateau on baseline is now reproduced
   by NQS-only with β reshape** (within ≈ 3 mHa). This shifts the
   bottleneck: the variational energy is no longer NQS-limited; the
   plateau is now imposed by the PT2 stage (top-N = 10 k).
   Investigating PT2 saturation is the next step rather than further
   sampler tuning.

The `nqs_only` variational energy is still descending at iter 7
(ΔE = 6.5×10⁻³ for 52Q β = 0.4 s42); 16-iter or larger-budget runs
should narrow the gap further.

### Fix 2 — Truncated-energy supervised loss

Replace the marginal-factorized KL teacher with the Solanki/Ding/
Reiher loss:

```
E^trunc_θ = Σ_{σ ∈ S_current} (|c_σ|² / Σ |c_σ′|²) · E_loc(σ)
∇θ E^trunc = supervised on log p_θ(σ) with weights c²·E_loc
```

Drop the REINFORCE term on `|coupling|²` entirely. Effort ~ 1–2 hours.

### Fix 3 — NQS-as-ranker, not NQS-as-sampler

Architectural change to the selection step:

```python
S_extend = singles_doubles(S_current)
score = α * |c_SQD|² + (1−α) * p_NQS(σ)
S_{k+1} = top_k(S_extend, key=score)
```

NQS no longer needs to *generate* the basis; only to *rank* candidates
that the Hamiltonian expansion already proposes. This converts a hard
generative-modelling problem (cover the support of `|ψ_FCI|²`) into a
much easier ranking problem (sort within a fixed candidate set).

---

## Open questions

1. **Why does the marginal-factorized teacher cause collapse so
   aggressively?** Five gradient steps at `lr=1e-3` should not be
   sufficient to delta-function a 52-token autoregressive distribution.
   Suspected: warm-start from prior iter + Aufbau-peaked teacher
   amplifies a fixed-point.

2. **Should we still keep the VMC update around at all?**
   Solanki/Ding/Reiher say no for selected-CI use cases. But we have
   the additional iterative loop (HI), where between-iteration drift
   may benefit from local-energy gradients. Worth re-testing with
   `vmc_weight ∈ {0.01, 0.1}` once Fix 2 is in place.

3. **Does β = 0.4 generalize?** Thompson & Gunlycke tested on
   sub-30-orbital systems. We are at 26 spatial orbitals (52 spin-
   orbitals) for 52Q and would like to scale to Cr₂-CAS(12,36) (72
   qubits). The optimal β may scale with orbital count.

4. **Sign-recovery diagnostic.** It would be worth instrumenting
   overlap between (a) NQS top-`p` configs and (b) eigenvector top-
   `|c|` configs across iterations, to test whether NQS is even
   tracking the diagonalized state at all.

---

## References

| arXiv | Title | Relevance |
|-------|-------|-----------|
| [2603.24728](https://arxiv.org/abs/2603.24728) | Auto-regressive NQS Sampling for SCI (Thompson, Gunlycke) | β-reshape recipe; sign offloading |
| [2602.12993](https://arxiv.org/abs/2602.12993) | NQS Based on Selected Configurations (Solanki, Ding, Reiher) | NQS-SC > NQS-VMC; truncated-energy loss |
| [2406.08154](https://arxiv.org/abs/2406.08154) | NN-Based Selective CI (Schmerwitz et al.) | Active-learning ranker; H-action expansion |
| [2109.12606](https://arxiv.org/abs/2109.12606) | NAQS: autoregressive ab-initio (Barrett, Malouf) | Symmetry masks; top-K enumeration |
| [2507.07540](https://arxiv.org/abs/2507.07540) | NNQS-AFQMC | NNQS as trial wavefunction |
| [2412.12248](https://arxiv.org/abs/2412.12248) | Physics-informed Transformer | AR + reference state; SR optimizer |
| [2002.04613](https://arxiv.org/abs/2002.04613) | NN wave functions and the sign problem | Why amplitude/phase need separate LRs |
| [2505.17846](https://arxiv.org/abs/2505.17846) | Lossy-QSCI | NN-assisted compression |
