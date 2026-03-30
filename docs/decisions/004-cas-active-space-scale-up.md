# ADR-004: CAS Active Space Support for 40–58 Qubit Molecules

- **Status**: Proposed
- **Date**: 2026-03-30
- **Author**: George Chang
- **Relates to**: Issue #21, ADR-003

---

## Context

qvartools currently supports molecules up to 28 qubits (C₂H��� STO-3G, full
Hilbert space). To reach 40–58 qubit systems — the scale where quantum
variational methods become classically intractable and scientifically
interesting — we need **Complete Active Space (CAS)** reduction.

The sister repository **Flow-Guided-Krylov** has already demonstrated
successful runs at this scale:

| System | Qubits | Active Space | Hilbert Dim | Time | Verified |
|--------|--------|-------------|-------------|------|----------|
| N₂ cc-pVTZ | 40 | CAS(10,20) | 240,374,016 | 140s | Yes |
| N₂ cc-pVTZ | 52 | CAS(10,26) | 4,327,008,400 | 195s | Yes |
| Cr₂ cc-pVDZ | 40 | CAS(12,20) | ~1.5B | ��� | Defined |
| Cr₂ cc-pVDZ | 58 | CAS(12,29) | ~10¹⁰ | — | Defined |

However, a direct port is **not feasible** — at least 3 critical breaking
points and 4 silent degradation issues exist in qvartools' current
architecture.

---

## Problem: 7 Breaking Points at 40 Qubits

### Critical (will crash)

| ID | Component | Location | Failure Mode |
|----|-----------|----------|-------------|
| C1 | `compute_molecular_integrals` | `integrals.py` | No `cas`/`casci` parameter — cannot build CAS Hamiltonian at all |
| C2 | `matrix_elements_fast()` | `hamiltonian.py:758` | Hard 10K config cap → `MemoryError` for basis > 10K |
| C3 | `FCISolver` / pipeline scripts | `fci.py:194` + 24 scripts | `max_configs=500K` → `RuntimeError` at 240M Hilbert dim; every experiment script calls FCI for reference |

### Warning (silent degradation)

| ID | Component | Location | Degradation |
|----|-----------|----------|-------------|
| W1 | `max_cartesian_size=1000` | `sqd.py:114` | Silently skips Cartesian expansion for large bases |
| W2 | `max_doubles=5000` | `sqd.py:569` | Only ~2,700/7,000 mixed double excitations generated |
| W3 | Krylov connection explosion | `krylov_expand.py` | n_orb=20 → ~28K connections/config, 140 万 candidates per hop |
| W4 | `very_large` tier params | `pipeline_config.py:295` | `max_krylov_dim=4` too conservative, accuracy loss |

### OK (no changes needed)

- `MolecularIntegrals` dataclass — field-agnostic
- `MolecularHamiltonian.diagonal_element()` / `get_connections()` — scales
- `AutoregressiveTransformer` NQS — no hardcoded limits
- `ParticleConservingFlowSampler` — MLP input scales linearly
- `DavidsonSolver` / `gpu_eigsh` — iterative, suited for large systems
- `pipeline.py` Hilbert space computation — uses `math.comb`, no overflow

---

## Decision

Implement CAS active-space support in **4 phases**, each independently
testable and mergeable (Small CLs principle). All phases follow TDD:
failing test first → minimal implementation → refactor.

### Phase 1: CAS Integral Infrastructure (P0 — blocks everything)

**Goal:** `compute_molecular_integrals(..., cas=(nelecas, ncas), casci=False)`

**Source:** Port CASSCF/CASCI logic from `Flow-Guided-Krylov/src/hamiltonians/molecular.py:2327–2477`.

**Key design choices:**
- `cas: tuple[int, int] | None = None` — `(nelecas, ncas)` or `None` for full space
- `casci: bool = False` — use CASCI (no orbital optimisation) when `True`
- Auto-CASCI fallback when config count > 50M (matches FGK threshold)
- CAS integrals returned in `MolecularIntegrals` with `n_orbitals=ncas`,
  `nuclear_repulsion=e_core` (includes frozen-core energy)

**TDD plan:**
1. RED: `test_n2_cas_10_8_integrals_shape` — verify h1e=(8,8), h2e=(8,8,8,8)
2. RED: `test_n2_cas_10_12_n_orbitals` — verify n_orbitals=12, n_alpha=5, n_beta=5
3. RED: `test_casci_fallback_large_cas` — verify CASCI used when ncas>=15
4. RED: `test_cr2_cas_12_12_fix_spin` — verify singlet, not septet
5. GREEN: Implement `cas`/`casci` in `compute_molecular_integrals`
6. REFACTOR: Extract `_run_casscf()` helper

**New molecule factories:**
- `create_n2_cas_hamiltonian(cas, basis, device)`
- `create_cr2_hamiltonian(cas, basis, device)` — with `fix_spin_(ss=0)`
- `create_benzene_hamiltonian(basis, device)` — CAS(6,15)

**Registry additions (12 new entries):**
- N₂-CAS(10,12/15/17/20/26): 24–52Q
- Cr₂ + Cr₂-CAS(12,18/20/26/28/29): 24–58Q
- Benzene CAS(6,15): 30Q

### Phase 2: Large-Basis Diagonalisation Path (P0 — blocks experiments)

**Goal:** Remove the 10K config ceiling in `matrix_elements_fast()` and
add a sparse Hamiltonian construction path.

**Source:** `Flow-Guided-Krylov/src/utils/gpu_diag.py` sparse thresholds.

**Key design choices:**
- **Sparse threshold:** 3K configs → switch from dense `torch.linalg.eigh`
  to `scipy.sparse.linalg.eigsh` (Lanczos)
- **GPU Lanczos threshold:** 8K configs → switch to CuPy `eigsh` if available
- `matrix_elements_fast()`: raise limit from 10K to 50K, add sparse COO
  output mode (`sparse=True` kwarg)
- `ProjectedHamiltonianBuilder`: add `build_sparse()` method returning
  `scipy.sparse.csr_matrix`

**TDD plan:**
1. RED: `test_sparse_h_matches_dense_h2` — build both, compare eigenvalues
2. RED: `test_matrix_elements_sparse_flag` — verify COO output
3. RED: `test_15k_configs_does_not_crash` — was MemoryError, now works
4. RED: `test_davidson_on_sparse_h` — Davidson with sparse input
5. GREEN: Implement sparse path
6. REFACTOR: Unify dense/sparse dispatch in `solve_generalized_eigenvalue`

### Phase 3: FCI-Free Pipeline Support (P0 — blocks experiment scripts)

**Goal:** Experiments gracefully handle systems without FCI reference.

**Key design choices:**
- `FCISolver.solve()`: if Hilbert dim > `max_configs`, return `SolverResult`
  with `energy=None` and `converged=False` instead of raising
- Pipeline scripts: detect `exact_energy is None`, print
  "FCI reference unavailable (Hilbert dim = X)", use HF energy as baseline
- `run_all_pipelines.py`: add `--skip-fci` flag, skip FCI for CAS molecules

**TDD plan:**
1. RED: `test_fci_solver_returns_none_for_huge_system` — no crash
2. RED: `test_pipeline_runs_without_fci_reference` — graceful fallback
3. GREEN: Modify FCISolver + pipeline scripts
4. REFACTOR: Extract `_compute_reference_energy()` helper

### Phase 4: Parameter Auto-Scaling for CAS Systems (P1)

**Goal:** Tune `very_large` tier and SQD parameters for 40Q+ accuracy.

**Key design choices (from FGK ablation data):**
- `max_krylov_dim`: 4 → 8 for CAS systems (FGK used 10–15 hops)
- `max_cartesian_size`: 1000 → 50,000 (or adaptive based on n_alpha × n_beta)
- `max_doubles`: 5000 → 20,000 (or remove cap, use n_orb-adaptive formula)
- `n_ref` in Krylov expansion: 50 → 20 for n_orb > 15 (reduce connection explosion)
- `max_new` in Krylov expansion: 500 → 2000 for CAS (FGK used 2000–5000)
- Add `is_cas` flag to `PipelineConfig` for CAS-specific tuning

**TDD plan:**
1. RED: `test_adapt_to_system_size_cas_40q` — verify CAS-specific params
2. RED: `test_krylov_max_connections_guard` — new param respected
3. GREEN: Implement adaptive scaling
4. REFACTOR: Move tier logic to `SystemScaler`

---

## Consequences

### Positive

- qvartools gains 40–58 qubit molecule support (N₂, Cr₂, Benzene families)
- Scientifically relevant: Cr₂ is a classic multi-reference benchmark
- All code paths tested before and after (TDD, no regression)
- Sparse diag path benefits existing molecules too (N₂ 20Q currently hits 10K limit)

### Negative

- PySCF becomes effectively required for CAS molecules (was fully optional before)
- 4 PRs of work — roughly 1 week of engineering
- Sparse path adds complexity to eigensolver dispatch
- FCI-free mode reduces confidence in energy accuracy for new molecules

### Risks

- Cr₂ CASSCF convergence is notoriously difficult (may need manual tuning per geometry)
- 52Q+ systems may require GPU memory management beyond current UMA sharing
- CASCI (no orbital optimisation) gives worse integrals than CASSCF for
  strongly correlated systems — but CASSCF's internal FCI is infeasible at 40Q+

---

## References

- Flow-Guided-Krylov `src/molecules.py` — 24 molecule registry with CAS support
- Flow-Guided-Krylov `results/ablation_40q_52q.json` — 40Q/52Q ablation data
- Flow-Guided-Krylov `src/hamiltonians/molecular.py:2677–2862` — CAS factory functions
- Flow-Guided-Krylov `docs/ABLATION-REPORT-40Q-52Q.md` — detailed analysis
- arXiv:2601.16637 — r-ccs-cms/sbd GPU-native SBD (ADR-003)
- PySCF mcscf module — CASSCF/CASCI implementation
