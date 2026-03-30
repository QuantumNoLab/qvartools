# ADR-004: CAS Active Space Support for 40–58 Qubit Molecules

- **Status**: Proposed
- **Date**: 2026-03-30
- **Author**: George Chang
- **Relates to**: Issue #21, ADR-003

---

## Context

qvartools currently supports molecules up to 28 qubits (C₂H₄ STO-3G, full
Hilbert space). To reach 40–58 qubit systems — the scale where quantum
variational methods become classically intractable and scientifically
interesting — we need **Complete Active Space (CAS)** reduction.

The sister repository **Flow-Guided-Krylov** has already demonstrated
successful runs at this scale:

| System | Qubits | Active Space | Hilbert Dim | Time | Verified |
|--------|--------|-------------|-------------|------|----------|
| N₂ cc-pVTZ | 40 | CAS(10,20) | 240,374,016 | 140s | Yes |
| N₂ cc-pVTZ | 52 | CAS(10,26) | 4,327,008,400 | 195s | Yes |
| Cr₂ cc-pVDZ | 40 | CAS(12,20) | ~1.5B | — | Defined |
| Cr₂ cc-pVDZ | 58 | CAS(12,29) | ~10¹⁰ | — | Defined |

However, a direct port is **not feasible**. Systematic code review
identifies **5 critical breaking points** and **7 silent degradation
issues** in qvartools' current architecture.

---

## Problem: 12 Breaking Points at 40 Qubits

### Critical (will crash or produce wrong results)

| ID | Component | File:Line | Failure Mode | Detail |
|----|-----------|-----------|-------------|--------|
| **C1** | `compute_molecular_integrals` | `integrals.py:113` | **No `cas`/`casci` parameter** | Cannot build CAS Hamiltonian at all. Function only does RHF → ao2mo. No CASSCF/CASCI path exists. |
| **C2** | `matrix_elements_fast()` | `hamiltonian.py:758` | **Hard 10K config cap** → `MemoryError` | `if n_configs > 10000: raise MemoryError(...)`. Any basis > 10K configs crashes. FGK routinely uses 15K basis. |
| **C3** | `_try_pyscf_fci()` | `fci.py:144–156` | **Rebuilds full molecule from geometry, ignores CAS** | For N₂ cc-pVTZ: rebuilds mol with ~60 orbitals → `fci.FCI(mf)` attempts full-space FCI (14e / ~60 orbs) → **hangs forever**. Does NOT use CAS integrals. |
| **C4** | FCI energy comparison | `fci.py:173` vs `hamiltonian.py:90` | **Apples-to-oranges energy** | CAS Hamiltonian uses `E_nuc = e_core` (frozen-core + nuclear). PySCF FCI returns energy with `mol.energy_nuc()` (nuclear only). `error_mha = (final_energy - exact_energy) * 1000` is **meaningless** — the two energies have different zero points. |
| **C5** | 26 pipeline scripts | `experiments/pipelines/**/*.py` | **All call `FCISolver().solve()`** | Every script expects `exact_energy` for comparison. C3 + C4 cascade: scripts either hang or report wrong error_mha for any CAS molecule. |

### Warning (silent degradation)

| ID | Component | File:Line | Issue | Quantified Impact at 40Q |
|----|-----------|-----------|-------|-------------------------|
| **W1** | `max_cartesian_size=1000` | `sqd.py:114,724` | Silently skips Cartesian product expansion if `|alpha| × |beta| > 1000` | With 100 unique alpha × 100 beta → product 10K → **silently skipped**, missing quantum correlations |
| **W2** | `max_doubles=5000` | `sqd.py:569–609` | Only generates ~2,700 of ~7,000 mixed α-β double excitations at n_orb=20 | `n_α × n_β × (n_orb−n_α) × (n_orb−n_β) = 5×5×15×15 = 5,625` → hits cap at ~2,700 → **44% of essential doubles missing** |
| **W3** | Krylov connection count | `krylov_expand.py` | ~7,875 connections per config at n_orb=20 (150 singles + 7,725 doubles) | 50 ref configs × 7,875 = **393,750 candidates per hop** → memory and time risk |
| **W4** | `very_large` tier params | `pipeline_config.py:295` | `max_krylov_dim=4` too conservative; FGK used 10–15 | FGK ablation: Krylov with 2K configs captures 99.8% of energy gain in 1.3s. `dim=4` may underperform. |
| **W5** | joblib integral cache | `integrals.py:243` | Cache key is `(geometry, basis, charge, spin)` — no `cas` parameter | Two different CAS sizes on same molecule would return cached result from the other. Also CASSCF is non-deterministic — FGK explicitly skips cache for CAS. |
| **W6** | `__init__` precomputation | `hamiltonian.py:148–156` | O(n_orb⁴) Python loop for `_h2e_sparse` dict at n_orb=20 | 160,000 dict insertions → ~0.2s (acceptable but noticeable vs n_orb=7 → 2,401 iters) |
| **W7** | `_ext` modules | `sbd_subprocess.py`, `cudaq_vqe.py` | sbd takes `nuclear_repulsion` param — must be `e_core` for CAS. VQE wrapper builds molecule internally, no CAS support. | sbd gives wrong energy if passed `mol.energy_nuc()` instead of `e_core`. VQE wrapper cannot run CAS molecules. |

### OK (verified safe at 40Q)

| Component | Why it's safe |
|-----------|---------------|
| `MolecularIntegrals` dataclass | Fields `n_orbitals`, `n_alpha`, `n_beta` are generic. `__post_init__` validates shapes, no size caps. |
| `MolecularHamiltonian.diagonal_element()` / `get_connections()` | Per-config cost, no dense matrix. O(n_orb²) single + O(n_orb⁴) double per config — same as FGK. |
| `AutoregressiveTransformer` NQS | Positional embeddings = `(1, n_orbitals, embed_dim)`. Token embedding = 3 tokens. Autoregressive sample fills n_orb positions sequentially. No hardcoded limits. |
| `ParticleConservingFlowSampler` | MLP input_dim = n_orbitals + context_dim. At n_orb=20: input=52. Works fine. |
| `DavidsonSolver` | Iterative eigensolver, triggers at >500 configs. Handles generalized eigenvalue via Cholesky. |
| `gpu_eigsh` (CuPy) | Sparse Lanczos, exactly what large systems need. |
| `_config_hash_batch` | `1 << 39` = 5.5×10¹¹, fits int64 (max 9.2×10¹⁸). Even at 58Q: `1 << 57` = 1.4×10¹⁷ < int64 max. |
| `pipeline.py` Hilbert computation | Uses `math.comb()` (arbitrary precision Python ints). |
| `configs_to_ibm_format` / `vectorized_dedup` | Operate on tensor rows, no length limits. |

---

## Decision

Implement CAS active-space support in **4 phases**, each independently
testable and mergeable (Small CLs principle). All phases follow TDD:
failing test first → minimal implementation → refactor.

### Phase 1: CAS Integral Infrastructure (P0 — blocks everything)

**Goal:** `compute_molecular_integrals(..., cas=(nelecas, ncas), casci=False)`

**Source:** Port CASSCF/CASCI logic from `Flow-Guided-Krylov/src/hamiltonians/molecular.py:2327–2477`.

**Key design choices:**

1. **New parameters on `compute_molecular_integrals`:**
   - `cas: tuple[int, int] | None = None` — `(nelecas, ncas)` or `None`
   - `casci: bool = False` — skip orbital optimisation when `True`
   - Auto-CASCI when config count > 50M (matches FGK `_FCI_CONFIG_LIMIT`)

2. **CAS integrals mapping to `MolecularIntegrals`:**
   - `h1e = h1e_cas` (active-space 1e integrals from `mc.h1e_for_cas()`)
   - `h2e = h2e_cas` (active-space 2e integrals via `ao2mo.full(mol, active_mo)`)
   - `nuclear_repulsion = e_core` (frozen-core energy, NOT `mol.energy_nuc()`)
   - `n_orbitals = ncas`
   - `n_alpha = nelecas // 2` (or tuple-aware)
   - `n_beta = nelecas // 2`

3. **Cr₂ special handling:**
   - `fix_spin_(ss=0)` on CASSCF to enforce singlet (prevents septet convergence)
   - `max_cycle=300` on RHF; ROHF fallback if RHF fails
   - Linear molecule FCI solver (`fci.direct_spin1.FCISolver`)

4. **Cache exclusion:** Do NOT cache CAS integrals via joblib (CASSCF is
   non-deterministic). Add `if cas is not None: return compute_molecular_integrals(...)` bypass in `cached_compute_molecular_integrals`.

**New molecule factories:**
- `create_n2_cas_hamiltonian(cas, basis, device)` — 24–52Q
- `create_cr2_hamiltonian(cas, basis, device)` — 24–58Q, with `fix_spin_`
- `create_benzene_hamiltonian(basis, device)` — 30Q CAS(6,15)

**Registry additions (12 new entries):**
- N₂-CAS(10,12/15/17/20/26): 24, 30, 34, 40, 52Q
- Cr₂ + Cr₂-CAS(12,18/20/26/28/29): 24, 36, 40, 52, 56, 58Q
- Benzene CAS(6,15): 30Q

**TDD plan:**
1. RED: `test_cas_parameter_signature` — verify `cas` and `casci` params exist
2. RED: `test_n2_cas_10_8_shapes` — h1e=(8,8), h2e=(8,8,8,8), n_orbitals=8
3. RED: `test_n2_cas_10_12_electron_counts` — n_alpha=5, n_beta=5
4. RED: `test_cas_nuclear_repulsion_is_ecore` — `e_core != mol.energy_nuc()`
5. RED: `test_casci_fallback_large_ncas` — CASCI auto-triggered when ncas>=15
6. RED: `test_cr2_singlet_not_septet` — energy < RHF energy (singlet is lower)
7. RED: `test_cache_bypassed_for_cas` — cached fn skips cache when cas != None
8. GREEN: Implement Phase 1
9. REFACTOR: Extract `_run_casscf()` helper, add `is_cas` field to mol_info

### Phase 2: Large-Basis Diag Path (P0 — blocks experiments)

**Goal:** Remove 10K config ceiling, add sparse H construction + iterative solver.

**Source:** `Flow-Guided-Krylov/src/utils/gpu_diag.py` sparse thresholds.

**Key design choices:**

1. **`matrix_elements_fast()` changes:**
   - Remove hard 10K limit (currently `MemoryError`)
   - New limit: 50K for dense (20 GB on DGX Spark 128 GB UMA)
   - Add `sparse: bool = False` kwarg: when `True`, return `scipy.sparse.coo_matrix`

2. **New `build_sparse_hamiltonian()` method on `MolecularHamiltonian`:**
   - For basis > 3K configs: build COO sparse matrix
   - Uses existing `get_connections()` per config → O(basis × connections_per_config)
   - At 15K × 7,875 connections: ~118M non-zero elements → ~2.8 GB COO

3. **Eigensolver dispatch update in `solve_generalized_eigenvalue`:**
   - <3K: dense `torch.linalg.eigh` (current)
   - 3K–8K: `scipy.sparse.linalg.eigsh` (Lanczos)
   - \>8K + CuPy available: `cupyx.scipy.sparse.linalg.eigsh` (GPU Lanczos)
   - Fallback: Davidson (current)

**TDD plan:**
1. RED: `test_sparse_h_eigenvalues_match_dense` — H₂ comparison
2. RED: `test_matrix_elements_sparse_coo_output` — verify COO format
3. RED: `test_15k_configs_no_crash` — was MemoryError, now succeeds
4. RED: `test_gpu_lanczos_dispatch` — verify CuPy path triggers at 8K
5. GREEN: Implement sparse path
6. REFACTOR: Unify dense/sparse dispatch

### Phase 3: CAS-Aware FCI & Pipeline Scripts (P0 — blocks experiments)

**Goal:** Pipelines gracefully handle CAS molecules where full FCI is impossible.

**Key design choices:**

1. **`FCISolver._try_pyscf_fci()` rewrite for CAS:**
   - Detect CAS via `mol_info.get("is_cas", False)`
   - For CAS: use CAS integrals directly with `pyscf.fci.direct_spin1.FCI`
     on the active-space h1e/h2e (NOT rebuild full molecule)
   - For full-space: keep current behaviour
   - If active-space FCI is still too large (>50M configs): return
     `energy=None, converged=False` (no crash)

2. **`_dense_fallback()` update:**
   - If `diag_dim > max_configs` AND PySCF unavailable: return
     `SolverResult(energy=None, ...)` instead of `RuntimeError`
   - Scripts detect `exact_energy is None` and print
     "FCI reference unavailable (Hilbert dim = X)"

3. **Pipeline scripts (26 files):**
   - Guard: `if exact_energy is None:` → use HF energy as baseline, print warning
   - `error_mha` only computed when `exact_energy` is available
   - `FlowGuidedKrylovPipeline(exact_energy=...)`: accept `None`

4. **Energy reference semantics for CAS:**
   - CAS Hamiltonian energy = electronic energy in active space + `e_core`
   - CAS FCI reference = `pyscf.fci.direct_spin1.FCI` on CAS integrals + `e_core`
   - Both use same zero point → `error_mha` is valid

**TDD plan:**
1. RED: `test_fci_solver_cas_uses_active_space_integrals` — NOT full molecule
2. RED: `test_fci_solver_returns_none_for_huge_cas` — no crash at 240M
3. RED: `test_pipeline_runs_without_fci_reference` — graceful None handling
4. RED: `test_cas_fci_energy_consistent_with_hamiltonian` — same e_core zero
5. GREEN: Implement
6. REFACTOR: Extract `_compute_cas_fci()` helper

### Phase 4: Parameter Auto-Scaling for CAS Systems (P1)

**Goal:** Tune pipeline parameters using FGK ablation data as ground truth.

**Key design choices (from FGK ablation results):**

1. **`very_large` tier adjustments:**
   - `max_krylov_dim`: 4 → 8 (FGK: Krylov with dim=10 captures 99.8% energy)
   - `max_accumulated_basis`: 16K → 20K (FGK used 15K basis at 40Q)
   - `samples_per_batch`: 2K → 3K (more samples needed to explore 240M space)

2. **SQD parameter scaling:**
   - `max_cartesian_size`: 1000 → adaptive: `min(50_000, C(n_orb, n_alpha) * C(n_orb, n_beta) // 10)`
   - `max_doubles`: 5000 → adaptive: `min(50_000, n_alpha * n_beta * (n_orb - n_alpha) * (n_orb - n_beta) * 2)`

3. **Krylov expansion guard:**
   - Add `max_connections_per_ref: int = 10_000` to prevent connection explosion
   - At n_orb=20: 7,875 connections/config → under limit, passes
   - At n_orb=26: ~38K connections/config → capped at 10K

4. **`is_cas` flag propagation:**
   - `mol_info["is_cas"] = True` for CAS molecules
   - `PipelineConfig.adapt_to_system_size()` checks `is_cas` for CAS-specific params
   - Pipeline scripts pass `is_cas` through to config adaptation

**TDD plan:**
1. RED: `test_adapt_to_system_size_cas_40q` — verify CAS-specific params
2. RED: `test_max_doubles_adaptive_formula` — correct scaling
3. RED: `test_krylov_max_connections_guard` — connections capped
4. GREEN: Implement
5. REFACTOR: Consolidate tier logic in `SystemScaler`

---

## Consequences

### Positive

- qvartools gains 40–58 qubit molecule support (N₂, Cr₂, Benzene families)
- Scientifically relevant: Cr₂ is a classic multi-reference benchmark
- All code paths tested before and after (TDD, no regression)
- Sparse diag path benefits existing molecules too (N₂ 20Q currently hits 10K limit)
- Energy reference is correct for both CAS and full-space molecules

### Negative

- PySCF becomes effectively required for CAS molecules (was fully optional before)
- 4 PRs of work — Phase 1–3 are P0 (each independently mergeable)
- Sparse path adds complexity to eigensolver dispatch
- FCI-free mode reduces confidence in energy accuracy for new molecules
- 26 pipeline scripts need modification (but change is mechanical: add None guard)

### Risks

| Risk | Mitigation |
|------|-----------|
| Cr₂ CASSCF convergence failure | ROHF fallback + `max_cycle=300` + `fix_spin_(ss=0)` (proven in FGK) |
| 52Q+ GPU memory for sparse H | COO format: 15K × 7.8K connections × 16 bytes = ~2 GB (fits 128 GB UMA) |
| CASCI gives worse integrals than CASSCF | Auto-CASCI only when FCI infeasible (>50M configs); smaller CAS uses CASSCF |
| Non-deterministic CASSCF breaks joblib cache | Skip cache for CAS integrals (explicit bypass) |
| `e_core` vs `E_nuc` confusion in downstream code | `is_cas` flag + CAS-specific FCI path ensures consistent zero point |
| Pipeline scripts break on CAS molecules | Phase 3 adds None guard to all 26 scripts before any CAS molecule is added to registry |

---

## Implementation Order & Dependencies

```
Phase 1 (CAS integrals) ─── prerequisite for everything
    │
    ├── Phase 2 (sparse diag) ─── required for basis > 10K
    │
    └── Phase 3 (FCI-free pipelines) ─── required for experiments
            │
            └── Phase 4 (param tuning) ─── accuracy optimisation
```

Phase 2 and Phase 3 are independent of each other after Phase 1 merges.
Phase 4 depends on Phase 3 (needs working pipeline scripts).

---

## References

- Flow-Guided-Krylov `src/molecules.py` — 24 molecule registry
- Flow-Guided-Krylov `src/hamiltonians/molecular.py:2327–2477` — CASSCF/CASCI
- Flow-Guided-Krylov `src/hamiltonians/molecular.py:2677–2862` — CAS factories
- Flow-Guided-Krylov `results/ablation_40q_52q.json` — 40Q/52Q results
- Flow-Guided-Krylov `docs/ABLATION-REPORT-40Q-52Q.md` — ablation analysis
- arXiv:2601.16637 — r-ccs-cms/sbd GPU-native SBD
- PySCF `mcscf` module — CASSCF/CASCI implementation
