# ADR-002: Eliminate torch↔numpy roundtrips in pipeline hot paths

- **Status**: Deferred (data does not justify optimization at current system sizes)
- **Date**: 2026-03-27
- **Updated**: 2026-03-28
- **Author**: George Chang
- **Relates to**: Issue #12

---

## Context

The qvartools pipeline performs ~70 torch↔numpy conversions across modules.
The most severe are GPU→CPU→GPU roundtrips in the SQD and SKQD eigensolve
hot paths, where `torch.Tensor` (GPU) is converted to `np.ndarray` (CPU)
for scipy/numpy eigensolvers, then converted back to `torch.Tensor` (GPU).

DGX Spark (UMA) benchmarks show GPU→CPU direction at ~1 GB/s (page
migration), making these roundtrips disproportionately expensive. On
discrete GPUs (A100/H100, PCIe ~25-50 GB/s), the overhead is smaller but
still measurable at ~6% for n=2000.

## Diagnostic results (2026-03-28)

Profiling on DGX Spark GB10 with real pipeline runs invalidated
the original performance assumptions:

### SQD batch sizes are tiny for current molecules

| Molecule | Qubits | Hilbert dim | Basis | SQD batch n |
|----------|--------|-------------|-------|-------------|
| H2       | 4      | 4           | 2     | 1           |
| LiH      | 12     | 225         | 28    | 9           |

At n=9, the numpy roundtrip takes **0.112 ms** total.
`torch.linalg.eigh` on GPU takes **0.567 ms** — 5x slower.

### GPU eigensolve loses to CPU for n < 500

| n   | GPU path | CPU path | GPU/CPU |
|-----|----------|----------|---------|
| 10  | 0.89 ms  | 0.08 ms  | 10.9x slower |
| 50  | 1.93 ms  | 0.34 ms  | 5.6x slower  |
| 100 | 6.79 ms  | 0.93 ms  | 7.3x slower  |
| 200 | 17.6 ms  | 3.43 ms  | 5.1x slower  |
| 500 | 27.8 ms  | 72.8 ms  | **2.6x faster** |

GPU wins only at n >= 500. Current molecule registry max is
C2H4 (28 qubits), batch sizes unlikely to exceed ~200.

### Total conversion volume is negligible

Heisenberg-6 pipeline (3 Krylov steps):
- torch→numpy: 18 calls, 130 KB total
- numpy→torch: 0 calls

**Conclusion: the torch↔numpy roundtrip is not a bottleneck
at current system sizes.** The overhead becomes meaningful only
for large-scale systems (N2 CAS(10,10)+ where batch n > 500).

## Decision

~~Incrementally replace numpy-based eigensolve interfaces with
torch-native equivalents in the pipeline hot paths.~~

**Deferred.** The diagnostic data shows this optimization has
no impact at current system sizes and would add complexity
for zero benefit. Revisit when:

1. Molecule registry includes systems with Hilbert dimension > 50K
   (SQD batch n > 500 typical)
2. Benchmark scripts (`experiments/pipelines/`) report eigensolve
   as > 10% of total wall time
3. A100/H100 discrete GPU users report PCIe transfer bottlenecks

The diagnostic script `experiments/diagnose_roundtrips.py` is
available to re-measure when system sizes grow.

## Consequences

### Positive

- SQD hot path eliminates GPU→CPU→GPU roundtrip per batch (~15-43 ms saved
  on DGX Spark at n=1000-2000)
- SKQD eigensolve results stay as torch tensors, reducing downstream
  conversions
- Pattern established for future torch-native migration of other interfaces

### Negative

- `torch.linalg.eigh` on CUDA **synchronizes the device with the CPU**
  (documented PyTorch behavior). The pipeline still blocks at eigensolve
  — we eliminate the data copy, not the sync point.
- `torch.linalg.eigh` is **slower than CPU for n < ~800** due to kernel
  launch and sync overhead. A size threshold is required.
- `torch.linalg.eigh` has **known GPU bugs** (pytorch/pytorch#105359:
  memory corruption on FP32 low-rank; pytorch/pytorch#144384: precision
  errors vs numpy). FP64 and fallback paths mitigate these.
- `torch.linalg` does **not** support generalized eigenvalue problems
  (`Hv = ESv`). SKQD's Cholesky transform must be retained.

### Neutral

- `gpu_eigh` / `gpu_eigsh` / `solve_generalized_eigenvalue` interfaces
  are not changed. They continue to serve call sites outside the hot path.

## Principles

### TDD Rule
Every change follows Red → Green → Refactor:
1. Write a failing test that asserts the new behavior
2. Make it pass with the minimal code change
3. Clean up only the code you touched

### Boy Scout Rule
> "Leave the code cleaner than you found it." — Robert C. Martin

Each PR may include small, safe improvements to code it touches (e.g.,
removing a dead import, fixing a misleading comment). These must not
change behavior and must not expand the PR scope.

### Small CLs (Google Engineering Practices)
> "A CL makes a minimal change that addresses just one thing."

Each PR targets exactly one conversion path. No bundling. A reviewer
should be able to understand the entire change in under 10 minutes.

### Anti-patterns to avoid

| Anti-pattern | Meaning | How we avoid it |
|-------------|---------|-----------------|
| Over-generation | Producing more code than needed | Each PR changes <100 lines of src/. No new abstractions unless tested and used. |
| Premature abstraction | Creating helpers/interfaces "for the future" | No `TensorOrArray` union type. No `UnifiedEigensolver` base class. Change concrete call sites only. |
| Scope creep | "While I'm here, let me also fix..." | If a bug is found outside the PR scope, open an Issue. Don't fix it in this PR. |

---

## Change plan

### Dependency graph

```
PR-A (P0: SQD torch.linalg.eigh)
  ↓
PR-B (P2: SKQD torch.linalg.eigh)  ← can start after PR-A merges
  ↓
Issue-C (P1: GPU Slater-Condon)     ← separate RFC, not in this plan
Issue-D (P3: torch Lanczos)         ← separate RFC, not in this plan
```

PR-A and PR-B are independent in code (different files), but PR-A
goes first because SQD is the more common production path and the
pattern proven there is reused in PR-B.

---

### PR-A: Replace numpy eigensolve in SQD with torch.linalg.eigh

**Scope**: `src/qvartools/krylov/circuits/sqd.py` only

**What changes**:
- `_diagonalize_batch()` lines 756-783: remove `.cpu().numpy()` →
  `np.linalg.eigh` → `torch.from_numpy().to(device)` cycle
- Replace with `torch.linalg.eigh(H_work)` when `H_work.is_cuda` and
  `n > _TORCH_EIGH_MIN_SIZE` (threshold ~500, below which CPU numpy
  is faster due to GPU kernel launch overhead)
- Keep CPU numpy fallback for: (a) small matrices, (b) CPU device,
  (c) `torch.linalg.eigh` runtime errors

**What does NOT change**:
- `gpu_eigh` / `gpu_eigsh` interfaces (not touched)
- `solve_generalized_eigenvalue` (not touched)
- Any file outside `sqd.py`

**TDD plan**:
1. RED: Add test in `tests/test_krylov/` that creates a mock SQD
   batch on GPU, calls the eigensolve path, and asserts:
   (a) eigenvalues match CPU reference within 1e-8
   (b) eigenvector stays on GPU device (no intermediate CPU)
2. GREEN: Modify `_diagonalize_batch` to use `torch.linalg.eigh`
   with size threshold
3. REFACTOR: Remove now-unused `gpu_eigsh`/`gpu_eigh` imports from
   `sqd.py` if they become dead code (Boy Scout Rule)

**Validation**:
- Before: `run_all_pipelines.py h2 --device cuda` → 24/24 pass
- `pytest -m "not pyscf"` → all pass
- After: same two checks
- Performance: measure wall time of `dci_sqd.py h2 --device cuda`
  before and after

**Size estimate**: ~40 lines changed in `sqd.py`, ~40 lines new test

---

### PR-B: Replace numpy eigensolve in SKQD with torch.linalg.eigh

**Scope**: `src/qvartools/krylov/basis/skqd.py` only

**What changes**:
- `_solve_generalised_eigenproblem()` lines 208-277: for the GPU
  branch, keep the existing Cholesky transform (`L^{-1} H L^{-T}`,
  required because `torch.linalg.eigh` does not support generalized
  eigenvalue problems), but remove the numpy conversion at the end
  (lines 270-271: `.cpu().numpy()`)
- Return torch tensors directly to the caller instead of numpy arrays
- Update `run()` method to accept torch tensor eigenvalues/vectors
- Apply same size threshold as PR-A

**What does NOT change**:
- The Cholesky transform for generalized eigenvalue (must be retained)
- `scipy.sparse.linalg.expm_multiply` (P3, separate RFC)
- `ProjectedHamiltonianBuilder` output format (still scipy.sparse)
- `_compute_krylov_state` internal logic

**TDD plan**:
1. RED: Add test that SKQD eigensolve on GPU returns eigenvalues
   matching CPU reference within 1e-8, and output tensors are on
   the expected device
2. GREEN: Modify `_solve_generalised_eigenproblem` GPU branch to
   return torch tensors directly
3. REFACTOR: Simplify GPU/CPU branching (currently 3 code paths,
   can be reduced to 2)

**Validation**:
- Before: `run_all_pipelines.py h2 --device cuda` → 24/24
- `pytest -m "not pyscf"` → all pass
- After: same checks

**Size estimate**: ~60 lines changed in `skqd.py`, ~30 lines new test

---

### Issue-C: GPU-native Slater-Condon rules (P1, future RFC)

**NOT in this plan.** Requires rewriting Numba-compiled
`numba_get_connections()` as PyTorch tensor operations. This is a
significant algorithmic change (~500 lines) that affects the entire
Hamiltonian interface.

**Recommended approach** (for future RFC):
- Vectorize single/double excitation enumeration using `torch.combinations`
- Compute Jordan-Wigner signs via batch bitwise operations
- Keep Numba path as fallback for CPU-only environments
- Estimated impact: eliminates ~1.9 MB/batch CPU↔GPU transfer in
  Stage 1 training

---

### Issue-D: Torch-native Lanczos matrix exponential (P3, future RFC)

**NOT in this plan.** Requires implementing full Lanczos iteration with
reorthogonalization in PyTorch, replacing `scipy.sparse.linalg.expm_multiply`.

**Recommended approach** (for future RFC):
- Port mkp's `_expm_multiply_lanczos` from numpy to torch
- Support GPU tensors natively
- Keep scipy path as fallback
- Estimated impact: eliminates n^2 x 16 bytes (complex128) CPU transfer
  per Krylov step

---

## Validation protocol

Every PR in this plan must pass these checks before merge:

```bash
# 1. Unit tests (no regression)
pytest -m "not pyscf and not gpu" --tb=short -q

# 2. GPU unit tests (on DGX Spark or any CUDA machine)
pytest -m "gpu" --tb=short -q

# 3. PySCF integration tests
pytest -m "not gpu" --tb=short -q

# 4. Lint
ruff check src/ tests/ experiments/
ruff format --check src/ tests/ experiments/

# 5. End-to-end pipeline validation (CRITICAL)
cd experiments/pipelines
python run_all_pipelines.py h2 --device cuda
# Expected: 24/24 pipelines pass

# 6. Performance regression check
# Run the specific pipeline before and after, compare wall time
python pipelines/01_dci/dci_sqd.py h2 --device cuda   # PR-A
python pipelines/01_dci/dci_krylov_classical.py h2 --device cuda  # PR-B
```

---

## What this plan does NOT do

- Does NOT create new abstractions (`TensorOrArray`, `UnifiedBackend`)
- Does NOT change any public API signatures
- Does NOT touch `gpu_eigh`/`gpu_eigsh`/`solve_generalized_eigenvalue`
- Does NOT refactor the Hamiltonian interface
- Does NOT modify the pipeline orchestrator (`pipeline.py`)
- Does NOT introduce new dependencies

---

## Future directions (not in scope)

- **DLPack zero-copy**: For cases requiring CuPy (sparse eigsh),
  `torch.utils.dlpack.to_dlpack` provides zero-copy torch↔cupy
  conversion without going through numpy. Relevant when Issue-C
  (GPU Slater-Condon) is addressed.
- **cuSOLVER BF16x9**: cuSOLVER 13.2 supports BF16x9 emulated math
  mode for syevd, accelerating internal GEMMs on Blackwell GPUs.
  Not accessible via PyTorch API; would require CuPy or ctypes.

---

## References

- Issue #12: torch/numpy conversion elimination
- ADR-001: SKQD naming and NQS interoperability
- Google Small CLs: https://google.github.io/eng-practices/review/developer/small-cls.html
- Boy Scout Rule: https://deviq.com/principles/boy-scout-rule/
- DGX Spark UMA benchmark data: GPU→CPU ~1 GB/s, CPU→GPU ~60 GB/s
- torch.linalg.eigh GPU sync: https://docs.pytorch.org/docs/stable/generated/torch.linalg.eigh.html
- torch.linalg.eigh FP32 GPU bug: pytorch/pytorch#105359
- torch.linalg.eigh precision errors: pytorch/pytorch#144384
- DLPack zero-copy interop: https://docs.cupy.dev/en/stable/user_guide/interoperability.html
- Nygardian ADR format: https://www.cognitect.com/blog/2011/11/15/documenting-architecture-decisions
