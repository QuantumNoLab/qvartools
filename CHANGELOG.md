# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- **BREAKING**: Rename `SampleBasedKrylovDiagonalization` to `ClassicalKrylovDiagonalization` (ADR-001)
- **BREAKING**: Rename `FlowGuidedSKQD` to `FlowGuidedKrylovDiag` (ADR-001)
- **BREAKING**: Default `subspace_mode` changed from `"skqd"` to `"classical_krylov"`
- `subspace_mode="skqd"` now routes to `QuantumCircuitSKQD` (real CUDA-Q SKQD)
- Old class names kept as deprecated aliases until v0.1.0
- `matrix_elements_fast()` dense config limit raised from 10K to 50K
- `FCISolver._dense_fallback()` returns `None` instead of raising `RuntimeError` for large Hilbert spaces

### Added
- `compute_molecular_integrals` now accepts `cas` and `casci` parameters for CAS active-space reduction
- 14 new CAS molecules in registry (26 total): N₂-CAS(10,12/15/17/20/26), Cr₂ + variants up to 72Q, Benzene CAS(6,15)
- IBM `solve_fermion` auto-enabled when `qiskit_addon_sqd` is installed (α×β Cartesian product, dramatically better accuracy)
- `_train_nqs_teacher` raises `ValueError` when `energy_weight > 0` without `hamiltonian`
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
