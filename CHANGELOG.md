# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: `experiments/pipelines/` folders renamed from 2-digit to 3-digit prefix
  (`01_dci` → `001_dci`, ..., `09_vqe` → `009_vqe`). YAML configs in
  `experiments/pipelines/configs/` renamed to match. Leaves room for the 010-099
  method-as-pipeline catalog tier.
- **BREAKING**: `run_all_pipelines.py --only` argument now requires 3-digit group
  prefixes (e.g., `--only 001 002 004`). Passing 2-digit values like `--only 01 02`
  silently matched no groups; now emits a migration warning with the recommended
  3-digit form. Scripts, docs, and `docs/experiments_guide.md` updated.
- **BREAKING**: Rename `SampleBasedKrylovDiagonalization` to `ClassicalKrylovDiagonalization` (ADR-001)
- **BREAKING**: Rename `FlowGuidedSKQD` to `FlowGuidedKrylovDiag` (ADR-001)
- **BREAKING**: Default `subspace_mode` changed from `"skqd"` to `"classical_krylov"`
- `subspace_mode="skqd"` now routes to `QuantumCircuitSKQD` (real CUDA-Q SKQD)
- Old class names kept as deprecated aliases until v0.1.0
- `matrix_elements_fast()` dense config limit raised from 10K to 50K
- `FCISolver._dense_fallback()` returns `None` instead of raising `RuntimeError` for large Hilbert spaces

### Added
- **Pipeline catalog Tier 2**: 4 new pipeline folders (`010_hi_nqs_sqd`,
  `011_hi_nqs_skqd`, `012_nqs_sqd`, `013_nqs_skqd`) wrapping the
  `qvartools.methods.nqs.*` runners as first-class benchmark catalog entries.
  Total pipeline scripts: 26 → 33. Each method gets a folder; variants live
  as separate scripts inside the folder with a multi-section YAML config.
- `qvartools.methods.nqs.METHODS_REGISTRY`: public dict keyed by method id
  (`"nqs_sqd"`, `"nqs_skqd"`, `"hi_nqs_sqd"`, `"hi_nqs_skqd"`) mapping to
  runner function, config class, capability flags, and pipeline folder
  metadata. Used by 010-013 wrappers and available for benchmark harnesses.
- `src/qvartools/methods/nqs/_shared.py`: internal module with
  `build_autoregressive_nqs`, `extract_orbital_counts`,
  `validate_initial_basis` helpers extracted from the four NQS method
  modules to remove duplication.
- `experiments.config_loader.get_explicit_cli_args`: the previously
  private `_get_explicit_cli_args` is now the public entry point under
  this name (the leading-underscore alias is kept for backward
  compatibility with existing callers). Used by the 010-013 wrapper
  scripts to detect which CLI args were explicitly typed, so YAML
  section defaults actually apply when `--device`/molecule is omitted.
- `compute_molecular_integrals` now accepts `cas` and `casci` parameters for CAS active-space reduction
- 14 new CAS molecules in registry (26 total): N₂-CAS(10,12/15/17/20/26), Cr₂ + variants up to 72Q, Benzene CAS(6,15)
- IBM `solve_fermion` auto-enabled when `qiskit_addon_sqd` is installed (α×β Cartesian product, dramatically better accuracy)
- `_train_nqs_teacher` raises `ValueError` when `energy_weight > 0` without `hamiltonian`
- `compute_e_pt2`: EN-PT2 energy correction (`E_PT2 = Σ |⟨x|H|Ψ₀⟩|² / (E₀ - H_xx)`)
- `HINQSSQDConfig.compute_pt2_correction`: opt-in E_PT2 correction after final iteration
- `use_ibm_solver` changed to tri-state (`None`=auto, `True`=force, `False`=disable)
- `_compute_cas_integrals` helper with auto-CASCI fallback for large active spaces
- `MolecularHamiltonian.build_sparse_hamiltonian()` for O(nnz) sparse H construction
- Sparse eigenvalue dispatch in `gpu_solve_fermion` for basis > 8K configs
- CAS-aware `FCISolver` using active-space integrals directly (no full molecule rebuild)
- FCI-free pipeline support: 25 experiment scripts gracefully handle `exact_energy=None`
- PT2 configuration selection for HI+NQS+SQD (`use_pt2_selection=True`, ADR-005)
- `_pt2_helpers.py`: EN-PT2 scoring, ASCI coefficient eviction, temperature annealing
- 3-term NQS teacher loss (teacher KL + energy REINFORCE + entropy)
- CIPSI sparse fallback for basis > 10K via `build_sparse_hamiltonian`
- `TransformerAsNQS` adapter: enables `AutoregressiveTransformer` in NF training pipeline
- `NQSWithSampling` adapter: enables any `NeuralQuantumState` in HI training pipeline
- `qvartools._logging` module with `configure_logging()` and `get_logger()`
- `QVARTOOLS_LOG_LEVEL` environment variable for log level control
- CI: mypy type checking job, coverage threshold enforcement
- ADR-001 decision record at `docs/decisions/`
- `split_spin_strings()` and `cartesian_product_configs()` utilities in `_utils/formatting/bitstring_format.py`
- `SQDConfig.use_cartesian_product` flag for alpha×beta subspace expansion in SQD (PR #15)
- `qvartools._ext.sbd_subprocess` module: GPU-native sbd diagonalisation via subprocess wrapper (ADR-003 Phase 1, PR #17)
- `qvartools._ext.cudaq_vqe` module: CUDA-QX VQE and ADAPT-VQE pipeline wrapper with gate fusion and active space (PR #18)
- `initial_basis` keyword-only parameter on `run_hi_nqs_sqd()` and `run_hi_nqs_skqd()` for warm-starting cumulative basis from NF+DCI Stage 1-2 (Issue #10)
- ADR-002 decision record (deferred: torch/numpy roundtrip not a bottleneck)
- ADR-003 decision record (GPU-native SBD integration via r-ccs-cms/sbd)

### Removed
- S-CORE (`recover_configurations`) from HI-NQS-SQD IBM path — designed for quantum hardware noise, not needed for classical NQS samples (NH₃ 1.5 hr → 5 s)

### Fixed
- `nqs_sqd.py` and `nqs_skqd.py` were end-to-end broken: they accessed
  `mol_info["n_orbitals"]` directly, but `get_molecule()` does not populate
  that key. Routed through the new `extract_orbital_counts()` helper which
  falls back to `hamiltonian.integrals` (same fallback logic the HI methods
  already had). Both runners now smoke-tested on H₂.
- `TransformerNFSampler._build_nqs()` used wrong parameter name `hidden_dim` instead of `hidden_dims`
- `hi_nqs_sqd.py` passed tensors instead of numpy arrays to `vectorized_dedup`
- Groups 07/08 pipelines discarded NF+DCI basis when calling iterative NQS solvers (Issue #10)
- IBM `solve_fermion` returns electronic energy only; now correctly adds `nuclear_repulsion`
- CIPSI sparse path: `h_matrix.detach().cpu().numpy()` instead of `np.asarray` for CUDA tensors

## [0.0.0] - 2026-03-26

### Added
- Initial development release of qvartools
- Molecular and spin Hamiltonians with Jordan-Wigner mapping
- Neural quantum state architectures (Dense, Complex, RBM, Autoregressive Transformer)
- Normalizing flows with particle-number conservation
- Physics-guided mixed-objective training (teacher + variational energy + entropy)
- Sample-based Krylov quantum diagonalization (SKQD) with residual expansion
- CIPSI-style perturbative basis enrichment
- Unified solver interface (FCI, CCSD, SQD, SKQD, iterative variants)
- Molecule registry with 8 pre-configured benchmarks (H2 to C2H4)
- Automatic system-size scaling for network architectures and sampling budgets
- YAML-based experiment configuration system with CLI overrides
- Pipeline scripts integrated with shared config_loader (--config flag)
- Comprehensive API documentation with Sphinx (autodoc + napoleon)
- Full documentation site: getting started, user guide, API reference, tutorials, developer guide
- CODE_OF_CONDUCT.md (Contributor Covenant)
- Standalone examples (basic_h2, custom_pipeline, compare_solvers, spin_hamiltonian)
- GitHub CI workflows for testing and documentation
- Issue templates (bug report, feature request) and PR template
- Docker GPU support
- ReadTheDocs integration

[Unreleased]: https://github.com/QuantumNoLab/qvartools/compare/v0.0.0...HEAD
[0.0.0]: https://github.com/QuantumNoLab/qvartools/releases/tag/v0.0.0
