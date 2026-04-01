# ADR-005: PT2 Configuration Selection for HI-NQS v3

- **Status**: Proposed
- **Date**: 2026-04-02
- **Author**: George Chang, Jen-Yu Chang
- **Relates to**: Issue #25 (adaptive sampling RFC), PR #30 (original proposal)

---

## Context

The current `run_hi_nqs_sqd` adds ALL unique NQS samples to the
cumulative basis each iteration, relying on random sampling to find
important configurations. At 40+ qubits, NQS sampling covers < 0.01%
of the Hilbert space, and most samples are uninformative.

PR #30 (leo07010) proposed adding PT2-based perturbative selection to
filter NQS samples before adding them to the basis. The algorithmic
concept is sound but the implementation has critical issues: 3 API
crash bugs, hard-imported optional dependencies, deleted backward
compatibility (initial_basis, CAS FCI, logging), and a mean-field
approximation in the teacher signal that loses correlation information.

This ADR documents the design decisions for a correct reimplementation.

---

## Decisions

### D1: PT2 scoring formula — Epstein-Nesbet

**Options:** Epstein-Nesbet (EN), Møller-Plesset (MP), Heat-Bath CI (HCI)

**Choice: Epstein-Nesbet**

```
score(x) = |⟨x|H|Φ₀⟩|² / |E₀ - H_xx|
```

- EN uses the actual diagonal element `H_xx`, which naturally captures
  correlation effects in the denominator
- MP uses orbital energy sums, which requires a Fock operator (not
  always available in our framework)
- HCI uses `max_i |H_{xi} c_i|` without a denominator — simpler but
  gives no PT2 energy correction estimate
- EN is the standard in CIPSI/Quantum Package and our existing
  `SelectedCIExpander`

**Source:** Quantum Package docs, QMCPACK Selected CI docs, Holmes et
al. JCTC 2016.

### D2: NQS teacher signal — full |c_x|² joint distribution

**Options:** Full `|c_x|²`, α/β marginal product, uniform

**Choice: Full |c_x|²**

PR #30 used `alpha_marginal[a] × beta_marginal[b]` as teacher weights.
This is a mean-field approximation that loses alpha-beta correlation —
for strongly correlated molecules (Cr₂, bond-breaking), the joint
distribution `|c_{ab}|²` has off-diagonal structure that the product
approximation misses entirely.

The original `_train_nqs_teacher` used full `|c_x|²`, which is correct.
We preserve this approach.

**Source:** Lanczos-NQS paper (arXiv:2502.01264), Thompson & Gunlycke
(arXiv:2603.24728).

### D3: Basis eviction — coefficient-based (ASCI pattern)

**Options:** PT2 score from insertion time, |c_i|² after diag, random

**Choice: |c_i|² after each diagonalisation**

PR #30 stored PT2 scores from the iteration when each config was added
and used these for eviction. This is methodologically flawed: scores
from different iterations use different eigenvectors, making cross-
iteration comparison meaningless.

The Adaptive Sampling CI (ASCI) method by Tubman et al. uses the
correct approach: after each diagonalisation, keep the configs with
largest `|c_i|²` (CI coefficient magnitude). This naturally discards
configs that the eigenvector no longer considers important.

**Source:** Tubman et al. JCTC 2020, Quantum Package CIPSI truncation.

### D4: Diag backend — gpu_solve_fermion (preserve optional dep guard)

**Options:** `qiskit_addon_sqd.solve_fermion` (hard import), `gpu_solve_fermion` (guarded)

**Choice: gpu_solve_fermion**

PR #30 hard-imported `solve_fermion` from `qiskit_addon_sqd`, which
broke CI (the package is optional). We preserve the existing pattern:
`gpu_solve_fermion` with the try/except guard for `qiskit_addon_sqd`.

### D5: Backward compatibility — extend, don't replace

PR #30 deleted `initial_basis` (PR #20), CAS FCI support (PR #24),
logging, `__all__`, frozen dataclass, and NumPy-style docstrings.

**Choice:** Preserve ALL existing API. Add PT2 selection as new
parameters on the existing `HINQSSQDConfig`:

```python
@dataclass(frozen=True)
class HINQSSQDConfig:
    # ... existing fields preserved ...
    # New PT2 selection fields
    use_pt2_selection: bool = False       # opt-in, backward compatible
    pt2_top_k: int = 2000                 # configs kept per iteration
    max_basis_size: int = 10_000          # eviction threshold
    convergence_window: int = 3           # consecutive converged iters
    initial_temperature: float = 1.0      # annealing start
    final_temperature: float = 0.3        # annealing end
```

When `use_pt2_selection=False` (default), behavior is identical to
current code. This ensures all existing tests pass unchanged.

---

## Implementation Plan (TDD)

### P0: Config fields (backward compatible)

Add new fields to frozen `HINQSSQDConfig` with defaults that preserve
existing behavior (`use_pt2_selection=False`).  Also add 3-term loss
weights (`teacher_weight`, `energy_weight`, `entropy_weight`).

### P1: Standalone helpers in `methods/nqs/_pt2_helpers.py`

Three pure functions (no NQS dependency, independently testable):

1. `compute_pt2_scores(candidates, basis_coeffs, hamiltonian, e0)` —
   EN-PT2 scoring via `get_connections` (NOT `get_connections_vectorized_batch`).
   Uses existing `bitstring_format` utilities (NOT local reimplementation).
2. `evict_by_coefficient(basis, coeffs, max_size)` — keep highest |c_i|²
   (ASCI pattern).
3. `compute_temperature(iteration, max_iter, t_init, t_final)` — linear
   interpolation.

### P2: Enhance `_train_nqs_teacher` with 3-term loss

Add energy (REINFORCE with diagonal advantage) and entropy terms.
Use full `|c_x|²` teacher (NOT α/β marginal product — loses correlation).
Correctly call `nqs.log_prob(alpha, beta)` with 2 args (split at n_orb).
Keep original behavior when `energy_weight=0` and `entropy_weight=0`.

### P3: Integration into `run_hi_nqs_sqd`

Gate on `use_pt2_selection`:
- `True`: PT2 filter → eviction → temperature anneal → convergence window
- `False`: zero change to existing behavior

Preserve: `initial_basis`, CAS compat, logging, `__all__`, docstrings.

### P4: CIPSI sparse fallback (independent)

When `n_basis > 10K`, use `hamiltonian.build_sparse_hamiltonian(basis)` +
`scipy.sparse.linalg.eigsh`.  Uses OUR API (NOT PR #30's nonexistent
`get_sparse_matrix_elements`).

### Dependency graph

```
P0 (config) ──┬── P1 (helpers)  ──┐
              ├── P2 (3-term loss) ├── P3 (integration) ── P5 (review) ── P6 (PR)
              └── P4 (CIPSI sparse, independent) ──────────┘
```

P1, P2, P4 are independent after P0.  P3 depends on P1 + P2.

### Scope: SQD only, SKQD deferred

PT2 selection is added to `run_hi_nqs_sqd` only.  `run_hi_nqs_skqd`
already has Krylov expansion which serves a similar basis-enrichment
role.  Extending PT2 to SKQD is a future enhancement.

---

## Consequences

### Positive

- More accurate basis selection at scale (PT2 > random sampling)
- Basis eviction prevents unbounded memory growth
- Temperature annealing improves exploration→exploitation transition
- Fully backward compatible (opt-in via `use_pt2_selection`)

### Negative

- PT2 scoring adds O(n_candidates × n_connections) per iteration
- Eviction adds one eigenvector sort per iteration (negligible)
- More config parameters to tune

### Risks

- PT2 scoring is CPU-bound Python (get_connections loop) — may be slow at 40Q
- Coefficient-based eviction may discard configs that become important later
- Temperature annealing schedule may need per-system tuning

---

## References

- PR #30 (leo07010): original HI-NQS v3 proposal
- Quantum Package CIPSI: EN-PT2 standard
- Holmes et al. JCTC 2016: Heat-Bath CI comparison
- Tubman et al. JCTC 2020: ASCI coefficient-based selection
- arXiv:2503.06292: HI-VQE iteration strategy
- arXiv:2603.24728: Auto-regressive NQS for Selected CI
- arXiv:2502.01264: Lanczos-NQS (KL vs MSE for teacher)
