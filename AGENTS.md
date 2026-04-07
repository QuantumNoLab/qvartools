# agents.md — qvartools AI Agent Guide

> This document serves as a comprehensive reference for AI coding agents (e.g., Claude Code, Copilot, Cursor) working on the **qvartools** codebase. It provides the full project context, architecture, conventions, and actionable guidance needed to make accurate, well-scoped contributions.
>
> **Repository:** `https://github.com/QuantumNoLab/qvartools`
> **Author:** George Chang
> **License:** MIT
> **Version:** 0.0.0 (initial development release)
> **Python:** 3.10+
> **Last updated:** 2026-03-26

---

## 0. Claude Code — Quick Start

> **Claude Code reads this file automatically via `CLAUDE.md` → `@AGENTS.md`.**
> All agents, commands, and skills are in `.claude/`. Run `/agents` to list them.

### Essential Commands

```bash
pytest                                             # full suite
pytest -m "not pyscf and not gpu"                 # skip optional deps
pytest --cov=qvartools --cov-report=term-missing  # coverage
ruff check src/ tests/                            # lint
ruff format src/ tests/                           # format
```

### Non-Negotiable Conventions

- **Linter:** Ruff (E,F,W,I,UP — E501/E731/UP007/F821/F841 ignored in quantum modules)
- **Docstrings:** NumPy-style (Parameters, Returns, Raises, Examples)
- **Types:** Python 3.10+ syntax (`X | Y`, `list[int]`, `tuple[float, ...]`)
- **Configs:** Frozen dataclasses with `__post_init__` validation and sensible defaults
- **Results:** Always `SolverResult` / `SamplerResult` — never raw tuples
- **Optional deps:** Guard with `ImportError` + install hint — never hard-import pyscf/cudaq/qiskit
- **Commits:** Conventional Commits (`feat:`, `fix:`, `refactor:`, `docs:`)

### Critical Domain Rule

`krylov/basis/skqd.py` is **classical** linear algebra (misnamed). The real SKQD is
`krylov/circuits/circuit_skqd.py`. Never route `"skqd"` to the classical version.
See `HANDOVER.md` and `docs/decisions/001-skqd-naming-and-nqs-interoperability.md`.

### Agents — `.claude/agents/`

Auto-delegation activates when task matches description. Force with `@agent-name`.

| Agent | Invoke when... |
|-------|----------------|
| `python-pro` | Designing Protocol/ABC, NQS adapters, frozen dataclasses, mypy |
| `legacy-modernizer` | Multi-file rename, deprecation aliases, routing logic updates |
| `tdd-orchestrator` | Writing new test files, property-based particle-conservation tests |
| `code-reviewer` | Pre-PR: Ruff compliance, SOLID check, deprecated alias coverage |
| `architect-review` | Module boundary decisions, ABC/Protocol API design |
| `docs-architect` | Sphinx RST updates, NumPy docstrings, CHANGELOG |
| `tutorial-engineer` | examples/, quickstart docs |
| `conductor-validator` | Tracking HANDOVER.md phased task progress |
| `debugger` | Root cause analysis, regression test design |
| `test-automator` | Bulk test scaffolding |
| `ml-engineer` | PyTorch NQS / normalizing flow model design |
| `security-auditor` | Optional-dep guards, input boundary validation |

### Slash Commands — `.claude/commands/`

| Command | Purpose |
|---------|---------|
| `/tdd-cycle` | Red → Green → Refactor loop |
| `/tdd-red` | Write a failing test first |
| `/tdd-green` | Make it pass minimally |
| `/tdd-refactor` | Clean up after green |
| `/refactor-clean` | Systematic multi-file refactor |
| `/tech-debt` | Identify and plan tech debt |
| `/full-review` | Pre-merge comprehensive review |
| `/pr-enhance` | Improve PR description |
| `/doc-generate` | NumPy docstrings generation |
| `/code-explain` | Explain complex code sections |
| `/git-workflow` | Branch + Conventional Commit workflow |
| `/implement` | Conductor-driven step-by-step implementation |
| `/manage` | Conductor task management |
| `/test-generate` | Test scaffolding from source |
| `/python-scaffold` | New module scaffolding |

### Skills — `.claude/skills/`

Auto-loaded by agents. Invoke manually with `/<skill-name>`.

| Skill | When it activates |
|-------|------------------|
| `python-design-patterns` | Adapter, Protocol, ABC design |
| `python-type-safety` | mypy annotations, Protocol types, generics |
| `python-testing-patterns` | pytest fixtures, markers (`slow`, `gpu`, `pyscf`), parametrize |
| `python-packaging` | `__init__.py` exports, deprecated aliases, Hatchling |
| `python-code-style` | Ruff rule compliance, NumPy docstring format |
| `python-error-handling` | ImportError guards, RuntimeError conventions |
| `python-performance-optimization` | NumPy/PyTorch vectorisation, CuPy fallback |
| `python-anti-patterns` | Detecting common pitfalls before they land |
| `debugging-strategies` | Root cause, regression test design |
| `error-handling-patterns` | Optional-dep guard patterns |
| `git-advanced-workflows` | Branch management, rebase, Conventional Commits |
| `code-review-excellence` | Review standards, checklist |
| `context-driven-development` | Session handoff, spec-to-implementation |
| `track-management` | Phase task tracking |
| `workflow-patterns` | Multi-phase project workflows |

### In-Progress Work

Three phases tracked in `HANDOVER.md` (read it before starting any session):

| Phase | What | Key agent | Key command |
|-------|------|-----------|-------------|
| 1 | SKQD naming fix (15+ files) | `legacy-modernizer` | `/refactor-clean` |
| 2 | NQS interop adapters | `python-pro` | `/tdd-cycle` |
| 3 | Bug `transformer_nf_sampler.py:314` | `debugger` | `/tdd-red` |

Start a new session: `@conductor-validator` then `/implement`.

---

## 1. Project Overview

**qvartools** (Quantum Variational Toolkit) is a unified Python package for quantum variational methods applied to molecular ground-state energy estimation. It consolidates three core algorithmic families into reusable, well-scoped modules:

1. **NF-NQS** — Normalizing-Flow-guided Neural Quantum States
2. **SQD** — Sample-based Quantum Diagonalization
3. **SKQD** — Sample-based Krylov Quantum Diagonalization

The package targets quantum chemistry researchers who need fine-grained control over each algorithmic stage while still being able to run end-to-end pipelines with a single function call.

### Core Dependencies

| Package | Role | Required |
|---------|------|----------|
| `torch >= 2.0` | Neural networks, tensor operations | Yes |
| `numpy >= 1.24` | Array computation | Yes |
| `scipy >= 1.10` | Sparse linear algebra, eigensolvers | Yes |
| `pyscf >= 2.4` | Molecular integrals, RHF, CCSD, FCI | Optional |
| `numba >= 0.57` | JIT-compiled Slater-Condon kernels | Optional |
| `cupy-cuda12x` | GPU eigensolvers | Optional |
| `qiskit >= 1.0` | Quantum circuit integration | Optional |
| `qiskit-addon-sqd >= 0.12` | IBM SQD utilities | Optional |
| `ffsim >= 0.0.70` | LUCJ circuit sampler | Optional |
| `cudaq >= 0.14` | CUDA-Q quantum circuit simulation | Optional |
| `pyyaml >= 6.0` | YAML experiment configs | Optional |

### Build System

- **Build backend:** Hatchling (`hatchling.build`)
- **Package layout:** `src/qvartools/` (src-layout)
- **Linter/Formatter:** Ruff (config in `pyproject.toml`)
- **Type checker:** mypy
- **Test runner:** pytest with markers: `slow`, `gpu`, `pyscf`
- **Docs:** Sphinx + ReadTheDocs (`sphinx-rtd-theme`, NumPy-style docstrings)

---

## 2. Repository Structure

```
qvartools/
├── src/qvartools/                # Main package (src-layout)
│   ├── __init__.py               # Public API: PipelineConfig, FlowGuidedKrylovPipeline, configure_logging
│   ├── _logging.py               # Structured logging: configure_logging(), get_logger(), QVARTOOLS_LOG_LEVEL
│   ├── pipeline.py               # Top-level 3-stage orchestrator
│   ├── pipeline_config.py        # PipelineConfig dataclass (all hyperparameters)
│   │
│   ├── hamiltonians/             # Hamiltonian representations
│   │   ├── hamiltonian.py        # Hamiltonian ABC (diagonal_element, get_connections, to_dense, exact_ground_state)
│   │   ├── integrals.py          # MolecularIntegrals dataclass + compute_molecular_integrals (PySCF)
│   │   ├── pauli_string.py       # PauliString (apply, is_diagonal)
│   │   ├── molecular/            # Second-quantised molecular Hamiltonians
│   │   │   ├── hamiltonian.py    # MolecularHamiltonian (Jordan-Wigner, Slater-Condon, batch vectorised)
│   │   │   ├── jordan_wigner.py  # Numba JW sign kernels (numba_jw_sign_single, numba_jw_sign_double)
│   │   │   ├── slater_condon.py  # Numba excitation kernels (numba_get_connections)
│   │   │   └── pauli_mapping.py  # PauliSum, molecular_hamiltonian_to_pauli, heisenberg_hamiltonian_pauli
│   │   └── spin/                 # Spin-lattice Hamiltonians
│   │       ├── heisenberg.py     # HeisenbergHamiltonian (XYZ model, periodic/open BC)
│   │       └── tfim.py           # TransverseFieldIsing (tuneable range L, periodic/open BC)
│   │
│   ├── nqs/                      # Neural quantum state architectures
│   │   ├── neural_state.py       # NeuralQuantumState ABC (log_amplitude, phase, psi, probability)
│   │   ├── adapters.py           # TransformerAsNQS, NQSWithSampling (cross-pipeline adapters)
│   │   ├── architectures/
│   │   │   ├── dense.py          # DenseNQS, SignedDenseNQS, compile_nqs
│   │   │   ├── complex_nqs.py    # ComplexNQS (shared features, separate amplitude/phase heads)
│   │   │   └── rbm.py            # RBMQuantumState (Carleo-Troyer ansatz, complex weights)
│   │   └── transformer/
│   │       ├── attention.py      # CausalSelfAttention (KV cache), CrossAttention
│   │       └── autoregressive.py # AutoregressiveTransformer (alpha/beta channels, particle-conserving)
│   │
│   ├── flows/                    # Normalizing flows
│   │   ├── networks/
│   │   │   ├── coupling_network.py          # MultiModalPrior, CouplingNetwork (RealNVP building blocks)
│   │   │   ├── discrete_flow.py             # DiscreteFlowSampler (RealNVP, multi-modal prior)
│   │   │   └── particle_conserving_flow.py  # ParticleConservingFlowSampler (Gumbel top-k), verify_particle_conservation
│   │   └── training/
│   │       ├── loss_functions.py             # compute_teacher_loss, compute_physics_loss, compute_entropy_loss, compute_local_energy, ConnectionCache
│   │       ├── gumbel_topk.py               # GumbelTopK, SigmoidTopK (differentiable selection)
│   │       ├── physics_guided_training.py    # PhysicsGuidedConfig, PhysicsGuidedFlowTrainer (3-term loss)
│   │       └── flow_nqs_training.py          # FlowNQSConfig, PhysicsGuidedFlowTrainer (advanced: subspace diag, cosine LR, EMA)
│   │
│   ├── krylov/                   # Krylov subspace methods
│   │   ├── basis/
│   │   │   ├── skqd.py           # SKQDConfig, ClassicalKrylovDiagonalization (was SampleBasedKrylovDiagonalization)
│   │   │   ├── flow_guided.py    # FlowGuidedKrylovDiag (was FlowGuidedSKQD)
│   │   │   └── sampler.py        # KrylovBasisSampler (classical time-evolution sampling)
│   │   ├── expansion/
│   │   │   ├── residual_config.py        # ResidualExpansionConfig, _diagonalise_in_basis, _generate_candidate_configs
│   │   │   ├── residual_expander.py      # ResidualBasedExpander (iterative residual analysis)
│   │   │   ├── selected_ci_expander.py   # SelectedCIExpander (CIPSI-style PT2 selection)
│   │   │   ├── krylov_expand.py          # expand_basis_via_connections (H-connection growth)
│   │   │   └── residual_expansion.py     # Backward-compat shim (re-exports)
│   │   └── circuits/
│   │       ├── spectral.py       # compute_optimal_dt (spectral-range auto time-step)
│   │       ├── circuit_skqd.py   # QuantumSKQDConfig, QuantumCircuitSKQD (CUDA-Q Trotterized evolution)
│   │       └── sqd.py            # SQDConfig, SQDSolver (batch diag, S-CORE, spin symmetry, noise recovery)
│   │
│   ├── diag/                     # Subspace diagonalization
│   │   ├── eigen/
│   │   │   ├── eigenvalue.py             # solve_generalized_eigenvalue, compute_ground_state_energy, regularize_overlap_matrix
│   │   │   ├── davidson.py               # DavidsonSolver (iterative eigensolver)
│   │   │   ├── projected_hamiltonian.py  # ProjectedHamiltonianBuilder (sparse H in sampled basis)
│   │   │   └── eigensolver.py            # Backward-compat re-exports
│   │   └── selection/
│   │       ├── diversity_selection.py    # DiversityConfig, DiversitySelector (excitation-rank bucketing)
│   │       ├── bitstring.py              # Bitstring utilities (to_int, to_bitstring, merge, overlap)
│   │       └── excitation_rank.py        # compute_excitation_rank, compute_hamming_distance, bitpack_configs
│   │
│   ├── solvers/                  # High-level solver interfaces
│   │   ├── solver.py             # Solver ABC, SolverResult (frozen dataclass)
│   │   ├── reference/
│   │   │   ├── fci.py            # FCISolver (PySCF native, CAS-aware, or dense fallback)
│   │   │   └── ccsd.py           # CCSDSolver (PySCF CCSD)
│   │   ├── subspace/
│   │   │   ├── sqd.py            # SQDSolver (NF-sampling → direct diag)
│   │   │   ├── sqd_batched.py    # SQDBatchedSolver (self-consistent batching)
│   │   │   └── cipsi.py          # CIPSISolver (selected-CI)
│   │   ├── krylov/
│   │   │   ├── skqd.py           # SKQDSolver (NF → SKQD)
│   │   │   ├── skqd_expansion.py # SKQDSolverB, SKQDSolverC (H-connection expansion)
│   │   │   ├── nf_skqd.py        # NFSKQDSolver (NF online + Krylov)
│   │   │   └── dci_skqd.py       # DCISKQDSolverB, DCISKQDSolverC (deterministic CI → Krylov)
│   │   └── iterative/
│   │       ├── iterative_sqd.py  # IterativeNFSQDSolver (eigenvector feedback loop)
│   │       ├── iterative_skqd.py # IterativeNFSKQDSolver (eigenvector feedback loop)
│   │       └── _utils.py         # _create_flow, _bias_nqs (shared helpers)
│   │
│   ├── samplers/                 # Configuration samplers
│   │   ├── sampler.py            # Sampler ABC, SamplerResult (frozen dataclass)
│   │   ├── classical/
│   │   │   ├── nf_sampler.py               # NFSampler (flow + optional NQS)
│   │   │   └── transformer_nf_sampler.py   # TransformerNFSampler (autoregressive transformer)
│   │   └── quantum/
│   │       ├── trotter_sampler.py    # TrotterSampler (classical Trotter simulation)
│   │       ├── cudaq_sampler.py      # CUDAQCircuitSampler (UCCSD ansatz via CUDA-Q)
│   │       ├── cudaq_circuits.py     # uccsd_ansatz kernel, count_uccsd_params
│   │       └── lucj_sampler.py       # LUCJSampler (Qiskit + ffsim LUCJ circuit)
│   │
│   ├── molecules/                # Molecular system registry
│   │   └── registry.py           # MOLECULE_REGISTRY (26 molecules: 12 full-space + 14 CAS), get_molecule, list_molecules
│   │
│   ├── _ext/                     # Experimental GPU extensions
│   │   ├── __init__.py
│   │   ├── sbd_subprocess.py     # sbd_diagonalize, sbd_available (ADR-003 Phase 1)
│   │   └── cudaq_vqe.py          # run_cudaq_vqe (CUDA-QX VQE + ADAPT-VQE wrapper)
│   │
│   ├── methods/                  # End-to-end method pipelines
│   │   └── nqs/
│   │       ├── nqs_sqd.py        # NQSSQDConfig, run_nqs_sqd
│   │       ├── nqs_skqd.py       # NQSSKQDConfig, run_nqs_skqd
│   │       ├── hi_nqs_sqd.py     # HINQSSQDConfig, run_hi_nqs_sqd (initial_basis, PT2 selection)
│   │       ├── hi_nqs_skqd.py    # HINQSSKQDConfig, run_hi_nqs_skqd (initial_basis warm-start)
│   │       └── _pt2_helpers.py   # compute_pt2_scores, evict_by_coefficient, compute_temperature
│   │
│   └── _utils/                   # Internal utilities
│       ├── scaling/
│       │   ├── quality_presets.py   # QualityPreset, SystemTier, SystemMetrics, ScaledParameters
│       │   └── system_scaler.py     # SystemScaler (auto-adapt parameters to system size)
│       ├── formatting/
│       │   └── bitstring_format.py  # configs_to_ibm_format, ibm_format_to_configs, vectorized_dedup, hash_config, split_spin_strings, cartesian_product_configs
│       ├── hashing/
│       │   ├── config_hash.py       # ConfigHash, config_integer_hash (overflow-safe)
│       │   └── connection_cache.py  # ConnectionCache (LRU, stats tracking)
│       └── gpu/
│           ├── linear_algebra.py    # gpu_eigh, gpu_eigsh, gpu_solve_fermion (CuPy/PySCF/dense fallback)
│           ├── fci_solver.py        # GPUFCISolver, compute_gpu_fci (gpu4pyscf integration)
│           └── diagnostics.py       # gpu_solve_fermion (projected H diag), compute_occupancies
│
├── tests/                        # pytest test suite
│   ├── conftest.py               # Shared fixtures (h2_hamiltonian, beh2_hamiltonian, spin models)
│   ├── test_hamiltonians/
│   │   ├── test_base.py          # Hamiltonian ABC, PauliString, config↔index roundtrip
│   │   ├── test_molecular.py     # MolecularHamiltonian (JW, Slater-Condon, FCI) [pyscf]
│   │   └── test_spin.py          # Heisenberg, TFIM models
│   ├── test_nqs/
│   │   ├── test_dense.py         # DenseNQS, SignedDenseNQS
│   │   ├── test_complex.py       # ComplexNQS, RBMQuantumState
│   │   └── test_adapters.py     # TransformerAsNQS, NQSWithSampling adapter tests
│   ├── test_flows/
│   │   ├── test_discrete_flow.py        # DiscreteFlowSampler
│   │   └── test_particle_conserving.py  # ParticleConservingFlowSampler, verify_particle_conservation
│   ├── test_krylov/
│   │   ├── test_skqd.py          # SKQDConfig, ClassicalKrylovDiag, FlowGuidedKrylovDiag (backward compat)
│   │   └── test_naming.py        # ADR-001 rename tests, deprecated aliases, pipeline routing
│   │   ├── test_basis_sampler.py # KrylovBasisSampler
│   │   └── test_residual.py      # ResidualBasedExpander, SelectedCIExpander
│   ├── test_diag/
│   │   ├── test_eigensolver.py   # solve_generalized_eigenvalue, DavidsonSolver
│   │   └── test_diversity.py     # DiversitySelector, excitation_rank, hamming_distance
│   ├── test_solvers/
│   │   └── test_base.py          # Solver ABC, SolverResult
│   ├── test_methods/
│   │   └── test_initial_basis.py     # initial_basis warm-start contract + dedup tests
│   ├── test_utils/
│   │   ├── test_format_utils.py      # hash_config, vectorized_dedup
│   │   ├── test_connection_cache.py  # ConnectionCache
│   │   ├── test_logging.py          # configure_logging, get_logger
│   │   └── test_transformer_nf_sampler_bug.py  # _build_nqs fallback regression
│   └── test_integration/
│       ├── test_h2_pipeline.py       # Full H2 pipeline [pyscf]
│       ├── test_beh2_pipeline.py     # Full BeH2 pipeline [pyscf]
│       └── test_spin_pipeline.py     # Full spin model pipeline
│
├── examples/                     # Standalone runnable examples
│   ├── basic_h2.py               # Simplest usage: run_molecular_benchmark("H2")
│   ├── custom_pipeline.py        # Manual 3-stage pipeline on LiH
│   ├── compare_solvers.py        # FCI vs SQD vs SKQD comparison
│   ├── spin_hamiltonian.py       # Heisenberg and TFIM exact diagonalization
│   └── README.md                 # Example descriptions and usage
│
├── experiments/                  # Reproducible experiment scripts
│   ├── config_loader.py          # YAML loader with CLI overrides (--config, --device)
│   ├── profile_pipeline.py       # Wall-clock profiling of pipeline stages
│   └── pipelines/                # 33 end-to-end pipeline scripts (3-digit prefix catalog)
│       ├── run_all_pipelines.py  # Run all 33 pipelines and compare results
│       ├── configs/              # 13 YAML configs (9 ablation + 4 method-as-pipeline)
│       │
│       ├── 001_dci/              # Direct-CI (no NF): classical, quantum, SQD
│       ├── 002_nf_dci/           # NF + DCI merge: classical, quantum, SQD
│       ├── 003_nf_dci_pt2/       # NF + DCI + PT2 expansion: classical, quantum, SQD
│       ├── 004_nf_only/          # NF-only ablation: classical, quantum, SQD
│       ├── 005_hf_only/          # HF-only baseline: classical, quantum, SQD
│       ├── 006_iterative_nqs/    # Iterative NQS: classical, quantum, SQD
│       ├── 007_iterative_nqs_dci/    # NF+DCI merge → iterative NQS
│       ├── 008_iterative_nqs_dci_pt2/  # NF+DCI+PT2 → iterative NQS
│       ├── 009_vqe/              # CUDA-QX VQE: UCCSD, ADAPT-VQE
│       │
│       ├── 010_hi_nqs_sqd/       # HI+NQS+SQD: default, pt2, ibm_off
│       ├── 011_hi_nqs_skqd/      # HI+NQS+SKQD: default, ibm_on
│       ├── 012_nqs_sqd/          # NQS+SQD: default
│       └── 013_nqs_skqd/         # NQS+SKQD: default
│
├── docs/                         # Documentation
│   ├── architecture.md           # Design philosophy, module dependency graph
│   ├── api_reference.md          # Complete public API documentation
│   ├── examples.md               # 5 worked examples
│   ├── experiments_guide.md      # Experiment runner instructions
│   ├── Makefile                  # Sphinx build (make html)
│   ├── make.bat                  # Sphinx build (Windows)
│   └── source/                   # Sphinx RST source
│       ├── conf.py               # Sphinx config (autodoc, napoleon, intersphinx, mocked imports)
│       ├── index.rst             # Documentation homepage / TOC tree
│       ├── _static/.gitkeep      # Static assets placeholder
│       ├── getting_started/
│       │   ├── installation.rst  # Installation guide
│       │   └── quickstart.rst    # Quick start tutorial
│       ├── user_guide/
│       │   ├── overview.rst      # Package overview
│       │   ├── pipelines.rst     # Pipeline usage guide
│       │   └── yaml_configs.rst  # YAML configuration guide
│       ├── api/
│       │   ├── index.rst         # API reference landing page
│       │   ├── pipeline.rst      # Pipeline API
│       │   ├── hamiltonians.rst  # Hamiltonians API
│       │   ├── nqs.rst           # NQS API
│       │   ├── flows.rst         # Flows API
│       │   ├── krylov.rst        # Krylov API
│       │   ├── diag.rst          # Diag API
│       │   ├── solvers.rst       # Solvers API
│       │   ├── samplers.rst      # Samplers API
│       │   └── molecules.rst     # Molecules API
│       ├── tutorials/
│       │   ├── h2_pipeline.rst   # H2 tutorial
│       │   ├── beh2_pipeline.rst # BeH2 tutorial
│       │   └── custom_solver.rst # Custom solver tutorial
│       └── developer_guide/
│           ├── index.rst              # Developer guide index
│           ├── extending_solvers.rst  # Adding new solvers
│           └── extending_samplers.rst # Adding new samplers
│
├── .github/
│   ├── workflows/
│   │   ├── ci.yml                # Lint (ruff) + test (pytest, py3.10-3.12)
│   │   └── docs.yml              # Sphinx build on push to main
│   ├── PULL_REQUEST_TEMPLATE.md
│   └── ISSUE_TEMPLATE/
│       ├── bug_report.yml
│       └── feature_request.yml
│
├── pyproject.toml                # Build config, dependencies, ruff/pytest settings
├── README.md                     # Project overview, quickstart, architecture
├── INSTALL.md                    # Detailed installation guide (uv, pip, Docker)
├── CONTRIBUTING.md               # Development setup, code style, PR process
├── CHANGELOG.md                  # Keep-a-Changelog format
├── CITATION.cff                  # Academic citation metadata
├── CODE_OF_CONDUCT.md            # Contributor Covenant
├── LICENSE                       # MIT License (2024-2026 George Chang)
├── MANIFEST.in                   # Source distribution includes
├── Dockerfile.gpu                # CUDA 12.6 + Python 3.11 + cudaq + qiskit
├── .gitignore                    # Python, venv, testing, IDE, Sphinx, Jupyter, OS ignores
└── .readthedocs.yaml             # RTD build config (Python 3.11, Ubuntu 22.04)
```

---

## 3. Architecture & Data Flow

### Module Dependency Graph (No Upward Dependencies)

```
Level 0 (no deps):   hamiltonians    nqs    _utils
Level 1:             flows (depends on: hamiltonians, nqs)
Level 2:             krylov (depends on: hamiltonians, flows, nqs)
                     diag (depends on: hamiltonians)
                     samplers (depends on: hamiltonians, nqs, flows)
Level 3:             solvers (depends on: all above)
                     methods (depends on: solvers, samplers, krylov, nqs)
Level 4:             pipeline (depends on: all of the above)
                     molecules (depends on: hamiltonians)
```

### Pipeline Data Flow (3-Stage)

```
Stage 1: Train Flow + NQS          Stage 2: Basis Selection         Stage 3: Subspace Diag
┌─────────────────────────┐       ┌────────────────────────┐       ┌─────────────────────────┐
│ PhysicsGuidedFlowTrainer│       │ DiversitySelector      │       │ FlowGuidedKrylovDiag     │
│                         │       │                        │       │ QuantumCircuitSKQD       │
│ Loss = w_t * L_teacher  │  ──>  │ Excitation-rank        │  ──>  │   (skqd_quantum)         │
│      + w_p * L_physics  │       │   bucketing             │       │ SQDSolver (sqd)          │
│      + w_e * L_entropy  │       │ Hamming diversity      │       │                          │
│                         │       │ Essential config merge  │       │ Returns: final_energy    │
│ Output: accumulated_    │       │                        │       │                          │
│   basis (torch.Tensor)  │       │ Output: selected_basis │       │                          │
└─────────────────────────┘       └────────────────────────┘       └─────────────────────────┘

    OR (Direct-CI mode):
    HF + singles + doubles
    deterministically generated
```

### Key Abstractions (ABCs)

| ABC | Module | Must Implement | Used By |
|-----|--------|----------------|---------|
| `Hamiltonian` | `hamiltonians.hamiltonian` | `diagonal_element(config)`, `get_connections(config)` | Everything |
| `NeuralQuantumState` | `nqs.neural_state` | `log_amplitude(x)`, `phase(x)` | flows, solvers, samplers |
| `Solver` | `solvers.solver` | `solve(hamiltonian, mol_info) -> SolverResult` | experiments, methods |
| `Sampler` | `samplers.sampler` | `sample(n_samples) -> SamplerResult` | solvers, methods |

---

## 4. Molecule Registry

26 pre-configured molecular benchmarks (12 full-space + 14 CAS active-space) accessible via `get_molecule(name)`:

**Full-space molecules (4--28 qubits)**

| Name | Qubits | Basis Set | Geometry |
|------|--------|-----------|----------|
| H2 | 4 | sto-3g | 0.74 A bond |
| LiH | 12 | sto-3g | 1.6 A bond |
| BeH2 | 14 | sto-3g | linear, 1.33 A |
| H2O | 14 | sto-3g | 104.5 deg, 0.96 A |
| NH3 | 16 | sto-3g | tetrahedral |
| CH4 | 18 | sto-3g | tetrahedral |
| N2 | 20 | cc-pvdz | 1.0977 A |
| CO | 20 | sto-3g | 1.13 A |
| HCN | 22 | sto-3g | linear |
| C2H2 | 24 | sto-3g | linear |
| H2S | 26 | sto-3g | bent |
| C2H4 | 28 | sto-3g | planar |

**CAS active-space molecules (24--72 qubits)**

| Name | Qubits | Basis Set | Active Space |
|------|--------|-----------|--------------|
| N2-CAS(10,12) | 24 | cc-pvdz | 10e, 12o |
| Cr2 | 24 | sto-3g | 12e, 12o |
| N2-CAS(10,15) | 30 | cc-pvdz | 10e, 15o |
| Benzene | 30 | sto-3g | 6e, 15o |
| N2-CAS(10,17) | 34 | cc-pvdz | 10e, 17o |
| Cr2-CAS(12,18) | 36 | cc-pvdz | 12e, 18o |
| N2-CAS(10,20) | 40 | cc-pvtz | 10e, 20o |
| Cr2-CAS(12,20) | 40 | cc-pvdz | 12e, 20o |
| N2-CAS(10,26) | 52 | cc-pvtz | 10e, 26o |
| Cr2-CAS(12,26) | 52 | cc-pvdz | 12e, 26o |
| Cr2-CAS(12,28) | 56 | cc-pvdz | 12e, 28o |
| Cr2-CAS(12,29) | 58 | cc-pvdz | 12e, 29o |
| Cr2-CAS(12,32) | 64 | cc-pvdz | 12e, 32o |
| Cr2-CAS(12,36) | 72 | cc-pvdz | 12e, 36o |

---

## 5. Code Conventions

### Style & Formatting

- **Formatter/Linter:** Ruff (`ruff check` + `ruff format`)
- **Target:** Python 3.10 (`target-version = "py310"`)
- **Selected rules:** E, F, W, I, UP
- **Ignored rules:** E501 (line length), E731 (lambda), UP007 (Union syntax for cudaq stubs), F841/F821/F822 (cudaq runtime injections), E402 (config_loader path hack)
- **Per-file ignores:** `krylov/circuits/*.py` and `samplers/quantum/*.py` → F821 (cudaq gate names)
- **Type hints:** Modern Python 3.10+ syntax (`X | Y` unions, `list[int]` generics)
- **Docstrings:** NumPy-style (Parameters, Returns, Raises, Examples sections)

### Configuration Pattern

All configurable components use **frozen dataclasses** with sensible defaults:
- `PipelineConfig` — pipeline hyperparameters (not frozen, has `adapt_to_system_size()`)
- `SKQDConfig` — Krylov parameters (frozen, validated in `__post_init__`)
- `PhysicsGuidedConfig` — training parameters (frozen)
- `FlowNQSConfig` — advanced training parameters
- `DiversityConfig` — basis selection parameters (frozen, validated)
- `ResidualExpansionConfig` — expansion parameters (frozen, validated)
- `SQDConfig` — SQD parameters
- `QuantumSKQDConfig` — quantum circuit SKQD parameters
- `SolverResult` — immutable result container (frozen)
- `SamplerResult` — immutable result container (frozen)

### Return Types

- **Solvers** always return `SolverResult(energy, diag_dim, wall_time, method, converged, metadata)`
- **Samplers** always return `SamplerResult(configs, counts, metadata, log_probs, wall_time)`
- **Pipeline** returns `dict[str, Any]` with guaranteed key `"final_energy"`

### Error Handling

- `ImportError` with install instructions for optional deps (pyscf, cudaq, cupy, qiskit)
- `RuntimeError` for failed SCF convergence, missing basis, etc.
- `ValueError` for invalid configurations (validated in `__post_init__`)
- `MemoryError` for matrices exceeding safe limits (>50K configs in `matrix_elements_fast`); use `build_sparse_hamiltonian()` for larger bases

### Numba Strategy

- JW sign kernels and Slater-Condon rules use `@njit(cache=True)` when Numba is available
- A no-op `njit` shim is defined when Numba is absent, so the same code runs as pure Python
- Detection via `_HAS_NUMBA` flag in `jordan_wigner.py`

### GPU Strategy

- CuPy for GPU eigensolvers (optional, with CPU scipy fallback)
- PyTorch GPU tensors for vectorised Hamiltonian operations
- CUDA-Q for quantum circuit simulation
- `config.device` controls tensor placement throughout

---

## 6. Testing Guide

### Running Tests

```bash
# Full suite
pytest

# Skip optional-dependency tests
pytest -m "not pyscf and not gpu"

# Specific module
pytest tests/test_hamiltonians/

# With coverage
pytest --cov=qvartools --cov-report=term-missing
```

### Test Organization

| Directory | What It Tests | Markers |
|-----------|---------------|---------|
| `test_hamiltonians/` | Base ABC, molecular (JW, Slater-Condon), spin models | `pyscf` |
| `test_nqs/` | Dense, Complex, RBM architectures | — |
| `test_flows/` | Discrete flow, particle conservation | — |
| `test_krylov/` | SKQD, basis sampler, residual/selected-CI expansion | `pyscf` |
| `test_diag/` | Eigensolvers, diversity selection | — |
| `test_solvers/` | Solver ABC, SolverResult | — |
| `test_methods/` | initial_basis warm-start, shape validation | — |
| `test_ext/` | sbd subprocess, CUDA-QX VQE wrappers | `gpu`, `slow` |
| `test_utils/` | Format utils, connection cache | — |
| `test_integration/` | Full H2/BeH2/spin pipelines | `pyscf` |

### Shared Fixtures (conftest.py)

- `h2_hamiltonian` — H2 MolecularHamiltonian (4 qubits, requires PySCF)
- `beh2_hamiltonian` — BeH2 MolecularHamiltonian (14 qubits, requires PySCF)
- `heisenberg_4site` — 4-site periodic Heisenberg (Jx=Jy=Jz=1.0)
- `tfim_6site` — 6-site periodic TFIM (V=1.0, h=0.5)

### Chemical Accuracy Threshold

**1.6 mHa** (milliHartree) — the standard for chemical accuracy. Pipeline tests verify `error_mha < 1.6`.

---

## 7. Experiment System

### Config Loader Pattern

All 33 pipeline scripts use the shared `config_loader.py`:

```bash
python experiments/pipelines/002_nf_dci/nf_dci_krylov_classical.py h2 --device cuda
python experiments/pipelines/run_all_pipelines.py h2 --device cuda
python experiments/pipelines/002_nf_dci/nf_dci_krylov_classical.py lih \
    --config experiments/pipelines/configs/002_nf_dci.yaml --max-epochs 200

# New 010-013 method-as-pipeline catalog
python experiments/pipelines/010_hi_nqs_sqd/default.py h2 --device cuda
python experiments/pipelines/010_hi_nqs_sqd/pt2.py h2 --device cuda
```

**Precedence:** CLI args > YAML file > hardcoded defaults.

### 33 Pipeline Variants (9 ablation groups + 4 method-as-pipeline groups)

| Group | Basis Source | Classical Krylov | Quantum Krylov | SQD |
|-------|-------------|-----------------|----------------|-----|
| 01 DCI | HF+S+D | Yes | Yes | Yes |
| 02 NF+DCI | NF + CI merge | Yes | Yes | Yes |
| 03 NF+DCI+PT2 | NF + CI + PT2 | Yes | Yes | Yes |
| 04 NF-Only | NF only (ablation) | Yes | Yes | Yes |
| 05 HF-Only | HF reference (baseline) | Yes | Yes | Yes |
| 06 Iterative NQS | NQS loop | Yes | Yes | Yes |
| 07 NF+DCI → Iter NQS | NF+DCI merge → NQS loop | Yes | Yes | Yes |
| 08 NF+DCI+PT2 → Iter NQS | NF+DCI+PT2 → NQS loop | Yes | Yes | Yes |

---

## 8. Key Algorithms Reference

### Jordan-Wigner Mapping

Spin-orbitals ordered as `[alpha_0, ..., alpha_{n-1}, beta_0, ..., beta_{n-1}]`. The JW transformation maps fermionic creation/annihilation operators to Pauli strings with Z-chains for sign factors. Numba kernels (`numba_jw_sign_single`, `numba_jw_sign_double`) handle the parity counting.

### Slater-Condon Rules

Matrix elements between determinants differing by 1 or 2 spin-orbitals:
- **Single excitation:** `<Phi'|H|Phi> = h_pq + sum_r [(pq|rr) - delta_{sigma} (pr|rq)]`
- **Double excitation:** `<Phi'|H|Phi> = (pr|qs) - delta_{sigma_p sigma_r} (ps|qr)`

### Physics-Guided Training (3-Term Loss)

```
L = w_teacher * L_teacher + w_physics * L_physics + w_entropy * L_entropy

L_teacher = -sum_x p_nqs(x) * log p_flow(x)        # KL divergence
L_physics = sum_x p_nqs(x) * (E_loc(x) - baseline)  # Variational energy
L_entropy = sum_x p_flow(x) * log p_flow(x)          # Negative entropy
```

### SKQD (Sample-Based Krylov Quantum Diagonalization)

1. Construct Krylov basis: `{|psi_0>, H|psi_0>, H^2|psi_0>, ..., H^{K-1}|psi_0>}`
2. Sample configurations from each Krylov state
3. Build projected H and S matrices in the sampled basis
4. Solve generalized eigenvalue problem: `H_proj * c = E * S_proj * c`
5. Return lowest eigenvalue as ground-state energy estimate

### SQD (Sample-based Quantum Diagonalization)

1. Sample configurations from NF-NQS
2. Split into K random batches
3. For each batch: project H, diagonalize, extract occupancies
4. Self-consistent iteration: update occupancies, re-sample, repeat
5. Energy-variance extrapolation across batches

### Particle-Conserving Flow

Uses differentiable top-k (Gumbel-Softmax) to select exactly `n_alpha` and `n_beta` orbitals from learned logit scores, guaranteeing valid Slater determinants at every sample.

### Diversity Selection

Configurations are bucketed by excitation rank (0, 1, 2, 3, 4+) relative to HF reference. Within each bucket, greedy selection enforces minimum Hamming distance. Budget fractions: rank 0 = 5%, rank 1 = 15%, rank 2 = 40-50%, rank 3 = 25%, rank 4+ = 15%.

---

## 9. Common Agent Tasks

### Adding a New Molecule

1. Define geometry in `src/qvartools/molecules/registry.py` as `_NEWMOL_GEOMETRY`
2. Create factory function `_make_newmol(device)` following existing pattern
3. Add entry to **both** `MOLECULE_REGISTRY` (factory + n_qubits + description + basis) **and** `_MOLECULE_INFO_REGISTRY` (geometry + charge + spin + n_qubits + basis). If either is missing, the import-time consistency check at `registry.py:631` will raise `RuntimeError`.
4. Update README.md molecule table

### Adding a New Solver

1. Subclass `Solver` from `solvers/solver.py`
2. Implement `solve(hamiltonian, mol_info) -> SolverResult`
3. Export from appropriate `__init__.py`
4. Add test in `tests/test_solvers/`

### Adding a New NQS Architecture

1. Subclass `NeuralQuantumState` from `nqs/neural_state.py`
2. Implement `log_amplitude(x)` and `phase(x)`
3. Export from `nqs/architectures/__init__.py` and `nqs/__init__.py`
4. Add test in `tests/test_nqs/`

### Adding a New Hamiltonian

1. Subclass `Hamiltonian` from `hamiltonians/hamiltonian.py`
2. Implement `diagonal_element(config)` and `get_connections(config)`
3. Export from appropriate `__init__.py`
4. Add test in `tests/test_hamiltonians/`

### Adding a New Experiment Pipeline

1. Create script in `experiments/pipelines/<group>/` following the pattern:
   - Import `config_loader` via `sys.path.insert`
   - Auto-detect device, load molecule, compute FCI reference
   - Run pipeline stages (train → select → diag)
   - Report energy, error, and timing
2. Add entry to `experiments/pipelines/run_all_pipelines.py` PIPELINES list
3. Create or update YAML in `experiments/pipelines/configs/`

---

## 10. Gotchas & Important Details

### Spin-Orbital Convention

`MolecularHamiltonian` uses `num_sites = 2 * n_orbitals` with alpha orbitals at indices `[0, n_orb)` and beta orbitals at `[n_orb, 2*n_orb)`. The HF state fills `alpha[:n_alpha]` and `beta[:n_beta]`.

### Two PhysicsGuidedFlowTrainer Implementations

There are **two different classes** both named `PhysicsGuidedFlowTrainer`:
1. `flows/training/physics_guided_training.py` — simpler version with `PhysicsGuidedConfig`
2. `flows/training/flow_nqs_training.py` — advanced version with `FlowNQSConfig`, subspace diag, cosine LR

The pipeline imports from `flows/__init__.py` which re-exports from `physics_guided_training.py`. The `flow_nqs_training.py` version is used by `create_physics_guided_trainer()`.

### Optional Import Guards

Many modules guard optional imports with try/except and raise `ImportError` with install instructions:
- PySCF: `pip install pyscf`
- Numba: `pip install numba`
- CuPy: `pip install cupy-cuda12x`
- cudaq: `pip install cudaq`
- qiskit: `pip install qiskit qiskit-addon-sqd ffsim`

### Config Adaptation

`PipelineConfig.adapt_to_system_size(n_valid_configs)` automatically scales parameters based on Hilbert-space dimension:
- **small** (≤ 1K): only basis limits adjusted
- **medium** (≤ 5K): NQS dims increased to [384]*5
- **large** (≤ 20K): NQS dims [512]*5, samples 4K, epochs 600+
- **very_large** (> 20K): reduced Krylov dim (4), capped basis

### Hash-Based Optimisations

`MolecularHamiltonian` uses integer hashing (`_config_hash_batch`) with `torch.searchsorted` for O(n_conn * log(n_configs)) matrix element matching, replacing naive O(n_conn * n_configs) loops.

### Dense Matrix Guard

`matrix_elements_fast()` refuses to build matrices for >50,000 configurations (raises `MemoryError`). Use `build_sparse_hamiltonian()` or Davidson solver for larger systems.

### GPU Extensions (`_ext/`)

The `_ext/` subpackage is **experimental and optional**. `sbd_subprocess` requires the `sbd` binary compiled from r-ccs-cms/sbd + MPI runtime. `cudaq_vqe` requires CUDA-Q >= 0.14 and CUDA-QX Solvers >= 0.5. Both import-guard gracefully when deps are missing.

### Initial Basis Warm-Start

`run_hi_nqs_sqd()` and `run_hi_nqs_skqd()` accept `initial_basis: torch.Tensor | None = None` (keyword-only). The tensor must be 2D with shape `(n_configs, n_qubits)` — a `ValueError` is raised otherwise. Duplicates are automatically removed via `torch.unique`.

### SQD Cartesian Product Expansion

When `SQDConfig.use_cartesian_product=True` (default), SQD splits sampled configs into alpha/beta spin strings via `split_spin_strings()`, then enumerates all alpha×beta pairs via `cartesian_product_configs()`. This dramatically improves basis coverage for molecular Hamiltonians.

### IBM `solve_fermion` Energy Convention

IBM's `qiskit_addon_sqd.fermion.solve_fermion` returns **electronic energy only** (no nuclear repulsion). Always add `hamiltonian.integrals.nuclear_repulsion` to the result. Its `sci_state.amplitudes` is **2D** (n_alpha_strs × n_beta_strs), not 1D — use α/β marginals for NQS teacher weights.

### S-CORE is for Quantum Hardware Only

`recover_configurations` (S-CORE) in `qiskit_addon_sqd` is a noise-recovery technique for noisy quantum hardware samples. **Do not use it for classical NQS samples** — it adds massive overhead (NH₃: 1.5 hr → 5 s without it) with no accuracy benefit on clean samples.

---

## 11. CI/CD

### GitHub Actions

**CI Pipeline** (`.github/workflows/ci.yml`):
- **Lint job:** `ruff format --check` + `ruff check` on Python 3.11, pip cached
- **Typecheck job:** `mypy` on core modules (informational)
- **Smoke job:** Verifies 26+ molecules registered + all public modules importable
- **Test job:** `pytest` on Python 3.10, 3.11, 3.12 with `[dev,pyscf]` extras; coverage only on 3.11 (`--cov-fail-under=40`); excludes `gpu` marker
- **Docs job:** Sphinx build check on PRs (warns but doesn't block)
- **Global:** `concurrency: cancel-in-progress` cancels superseded runs; `fail-fast: false`

**Docs Pipeline** (`.github/workflows/docs.yml`):
- Sphinx build on push to main
- Uploads HTML artifact

### Pre-PR Checklist

```bash
ruff check src/ tests/
ruff format --check src/ tests/
pytest --cov=qvartools
```

### Commit Convention

[Conventional Commits](https://www.conventionalcommits.org/):
```
feat: add RBM wave-function ansatz
fix: correct sign in Jordan-Wigner mapping
docs: update solver API reference
```

---

## 12. Docker

### GPU Image (`Dockerfile.gpu`)

- Base: `nvidia/cuda:12.6.3-devel-ubuntu22.04`
- Python 3.11 via deadsnakes PPA
- PyTorch cu126, `[full,gpu,dev]` extras
- cudaq + qiskit quantum deps
- Default CMD: `pytest tests/ -v`

```bash
docker build -f Dockerfile.gpu -t qvartools-gpu .
docker run --gpus all --rm -it qvartools-gpu
```

---

## 13. Documentation

- **ReadTheDocs:** [qvartools.readthedocs.io/en/latest/](https://qvartools.readthedocs.io/en/latest/)
- **Sphinx config:** `docs/source/conf.py` (autodoc + napoleon + intersphinx)
- **Theme:** `sphinx-rtd-theme`
- **Mocked imports:** pyscf, numba, cupy, qiskit, ffsim, cudaq (for docs build without optional deps)
- **Sections:** Getting Started, User Guide, API Reference, Tutorials, Developer Guide
