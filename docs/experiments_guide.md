# Experiments Guide

This guide covers the 24 experiment pipeline scripts in `experiments/pipelines/`. Each pipeline combines a basis generation strategy with a diagonalization method. All pipelines can be configured via YAML files with CLI overrides.

---

## Pipeline Overview

All pipelines follow the same pattern:
1. Load a molecule from the registry
2. Compute the exact FCI energy for reference
3. Run the method-specific pipeline stages
4. Report energy, error vs exact, and timing

### Common Arguments

All scripts accept:
- Positional `molecule` argument (default: `h2`)
- `--config` flag pointing to a YAML configuration file
- `--device` flag (`cpu`, `cuda`, or `auto`)
- Pipeline-specific flags (see `--help`)

---

## Pipeline Groups

### Group 001: Direct-CI (no NF training)

Generates HF + singles + doubles deterministically, then diagonalizes.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `001_dci/dci_krylov_classical.py` | Classical Krylov | DCI -> SKQD time evolution |
| `001_dci/dci_krylov_quantum.py` | Quantum Krylov | DCI -> Trotterized circuit evolution |
| `001_dci/dci_sqd.py` | SQD | DCI -> noise + S-CORE batch diag |

### Group 002: NF-NQS + DCI Merge

Trains a normalizing flow, merges NF-sampled basis with Direct-CI essentials.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `002_nf_dci/nf_dci_krylov_classical.py` | Classical Krylov | NF+DCI merge -> SKQD |
| `002_nf_dci/nf_dci_krylov_quantum.py` | Quantum Krylov | NF+DCI merge -> Trotterized |
| `002_nf_dci/nf_dci_sqd.py` | SQD | NF+DCI merge -> noise + S-CORE |

### Group 003: NF + DCI + PT2 Expansion

Same as Group 02, plus CIPSI-style perturbative basis expansion via Hamiltonian connections.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `003_nf_dci_pt2/nf_dci_pt2_krylov_classical.py` | Classical Krylov | NF+DCI+PT2 -> SKQD |
| `003_nf_dci_pt2/nf_dci_pt2_krylov_quantum.py` | Quantum Krylov | NF+DCI+PT2 -> Trotterized |
| `003_nf_dci_pt2/nf_dci_pt2_sqd.py` | SQD | NF+DCI+PT2 -> noise + S-CORE |

### Group 004: NF-Only (Ablation)

NF training without DCI scaffolding. Tests pure NF generative power.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `004_nf_only/nf_krylov_classical.py` | Classical Krylov | NF-only -> SKQD |
| `004_nf_only/nf_krylov_quantum.py` | Quantum Krylov | NF-only -> Trotterized |
| `004_nf_only/nf_sqd.py` | SQD | NF-only -> noise + S-CORE |

### Group 005: HF-Only (Baseline)

Minimal baseline starting from a single Hartree-Fock reference state.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `005_hf_only/hf_krylov_classical.py` | Classical Krylov | HF -> Krylov discovers configs |
| `005_hf_only/hf_krylov_quantum.py` | Quantum Krylov | HF -> Trotterized circuit |
| `005_hf_only/hf_sqd.py` | SQD | HF -> noise + S-CORE |

### Group 006: Iterative NQS

Iterative autoregressive transformer NQS with eigenvector feedback.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `006_iterative_nqs/iter_nqs_krylov_classical.py` | Classical Krylov | NQS loop + H-connection expansion |
| `006_iterative_nqs/iter_nqs_krylov_quantum.py` | Quantum Krylov | NQS warmup + Trotterized |
| `006_iterative_nqs/iter_nqs_sqd.py` | SQD | NQS loop + batch diag |

### Group 007: NF + DCI Merge -> Iterative NQS

NF training and DCI merge (same as Group 02 stages 1-2), then iterative NQS refinement.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `007_iterative_nqs_dci/iter_nqs_dci_krylov_classical.py` | Classical Krylov | NF+DCI -> iterative NQS+Krylov |
| `007_iterative_nqs_dci/iter_nqs_dci_krylov_quantum.py` | Quantum Krylov | NF+DCI -> quantum Krylov |
| `007_iterative_nqs_dci/iter_nqs_dci_sqd.py` | SQD | NF+DCI -> iterative NQS+SQD |

### Group 008: NF + DCI + PT2 -> Iterative NQS

NF training, DCI merge, and PT2 expansion (same as Group 03 stages 1-2.5), then iterative NQS.

| Script | Diag Mode | Description |
|--------|-----------|-------------|
| `008_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_krylov_classical.py` | Classical Krylov | NF+DCI+PT2 -> iterative NQS+Krylov |
| `008_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_krylov_quantum.py` | Quantum Krylov | NF+DCI+PT2 -> quantum Krylov |
| `008_iterative_nqs_dci_pt2/iter_nqs_dci_pt2_sqd.py` | SQD | NF+DCI+PT2 -> iterative NQS+SQD |

---

## Tier 2: Method-as-Pipeline Catalog (010-099)

Each NQS method from `src/qvartools/methods/nqs/` is exposed as a runnable
benchmark folder. Variants of the same method live as separate scripts inside
the folder, with one multi-section YAML config per method (each section
corresponds to one variant script).

### Group 010: HI+NQS+SQD

Iterative HI loop with optional PT2 selection and IBM solver.

| Script | Variant | Description |
|--------|---------|-------------|
| `010_hi_nqs_sqd/default.py` | default | Baseline HI+NQS+SQD with auto IBM detect |
| `010_hi_nqs_sqd/pt2.py` | pt2 | `use_pt2_selection=True` + temperature annealing |
| `010_hi_nqs_sqd/ibm_off.py` | ibm_off | `use_ibm_solver=False` (force GPU fallback) |

Backed by `qvartools.methods.nqs.run_hi_nqs_sqd`.
Config: `experiments/pipelines/configs/010_hi_nqs_sqd.yaml` (3 sections).

### Group 011: HI+NQS+SKQD

Iterative HI loop with Krylov expansion.

| Script | Variant | Description |
|--------|---------|-------------|
| `011_hi_nqs_skqd/default.py` | default | Baseline HI+NQS+SKQD with Krylov expansion |
| `011_hi_nqs_skqd/ibm_on.py` | ibm_on | `use_ibm_solver=True` + S-CORE recovery |

Backed by `qvartools.methods.nqs.run_hi_nqs_skqd`.
Config: `experiments/pipelines/configs/011_hi_nqs_skqd.yaml` (2 sections).

### Group 012: NQS+SQD

Two-stage NQS+SQD (no iteration).

| Script | Variant | Description |
|--------|---------|-------------|
| `012_nqs_sqd/default.py` | default | Train NQS via VMC, sample, diagonalize |

Backed by `qvartools.methods.nqs.run_nqs_sqd`.
Config: `experiments/pipelines/configs/012_nqs_sqd.yaml`.

### Group 013: NQS+SKQD

Two-stage NQS+SKQD with Krylov expansion.

| Script | Variant | Description |
|--------|---------|-------------|
| `013_nqs_skqd/default.py` | default | Train NQS, sample, Krylov expand, diagonalize |

Backed by `qvartools.methods.nqs.run_nqs_skqd`.
Config: `experiments/pipelines/configs/013_nqs_skqd.yaml`.

### Numbering convention for future expansion

- **001-009**: ablation pipeline groups (current)
- **010-099**: method-as-pipeline catalog (4 used: 010-013, 86 free)
- **100-199**: cross-method sweeps (e.g., `100_h2_all_methods`)
- **200+**: hyperparameter sweeps (e.g., `200_hi_nqs_sqd_lr_sweep`)

The full method registry is exposed at runtime via
`qvartools.methods.nqs.METHODS_REGISTRY` for programmatic dispatch.

---

## Running All Pipelines

```bash
# Run all 33 pipelines on H2 and compare
python experiments/pipelines/run_all_pipelines.py h2 --device cuda

# Run only specific groups
python experiments/pipelines/run_all_pipelines.py h2 --only 001 002 004

# Skip quantum pipelines (no CUDA-Q needed)
python experiments/pipelines/run_all_pipelines.py h2 --skip-quantum

# Skip slow iterative pipelines
python experiments/pipelines/run_all_pipelines.py h2 --skip-iterative

# Save results to JSON
python experiments/pipelines/run_all_pipelines.py lih --output results.json
```

## Chemical Accuracy Threshold

All experiments compare results against **1.6 milliHartree (mHa)**, the conventional definition of chemical accuracy.

## Prerequisites

- `pyscf` must be installed for molecular integrals and FCI/CCSD
- GPU experiments require CUDA-enabled PyTorch
- Quantum Krylov pipelines require `cudaq`
- Large molecules (N2, CH4, C2H4) may take several minutes on CPU
