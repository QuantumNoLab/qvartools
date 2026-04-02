# qvartools

[![CI](https://github.com/QuantumNoLab/qvartools/actions/workflows/ci.yml/badge.svg)](https://github.com/QuantumNoLab/qvartools/actions/workflows/ci.yml)
[![Documentation](https://readthedocs.org/projects/qvartools/badge/?version=latest)](https://qvartools.readthedocs.io/en/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

**Quantum Variational Toolkit** -- a unified Python package for quantum variational methods applied to molecular ground-state energy estimation.

qvartools consolidates normalizing-flow-guided neural quantum states (NF-NQS), sample-based quantum diagonalization (SQD), and sample-based Krylov quantum diagonalization (SKQD) into reusable, well-scoped modules. It is designed for quantum chemistry researchers who want fine-grained control over each algorithmic stage while still being able to run end-to-end pipelines with a single function call.

## Key Features

- **Molecular and spin Hamiltonians** with efficient matrix-element computation, Jordan-Wigner mapping, and exact diagonalization
- **Neural quantum state architectures** including dense, signed, complex, RBM, and autoregressive transformer variants
- **Normalizing flows** with particle-number conservation and physics-guided mixed-objective training (teacher + variational energy + entropy)
- **Sample-based Krylov diagonalization (SKQD)** with residual expansion and CIPSI-style perturbative basis enrichment
- **Unified solver interface** covering FCI, CCSD, SQD, SKQD, and iterative NF variants -- all returning a common `SolverResult`
- **Automatic system-size scaling** that adapts network architectures and sampling budgets to the Hilbert-space dimension
- **YAML-based experiment configuration** with CLI overrides for reproducible experiments
- **Molecule registry** with 26 pre-configured benchmarks from H₂ (4 qubits) to Cr₂-CAS(12,36) (72 qubits)

## Installation

### pip

```bash
pip install -e .
```

### With optional dependencies

```bash
# PySCF for molecular integrals and CCSD
pip install -e ".[pyscf]"

# Full installation (PySCF + Numba JIT acceleration)
pip install -e ".[full]"

# Development tools (pytest, ruff, mypy)
pip install -e ".[dev]"

# Documentation build
pip install -e ".[docs]"
```

### uv

```bash
uv pip install -e ".[full,dev]"
```

### Docker (GPU)

```bash
docker build -f Dockerfile.gpu -t qvartools-gpu .
docker run --gpus all --rm -it qvartools-gpu python -c "import qvartools; print(qvartools.__version__)"
```

### Dependencies

Core: `torch`, `numpy`, `scipy`. Optional: `pyscf` (molecular integrals, CCSD), `numba` (JIT acceleration), `cupy` (GPU eigensolvers), `pyyaml` (YAML configs), `qiskit` (quantum circuits).

## Quick Start

```python
from qvartools import run_molecular_benchmark

# Run the full NF-SKQD pipeline on H2
results = run_molecular_benchmark("H2", verbose=True)

print(f"Ground-state energy: {results['final_energy']:.10f} Ha")
print(f"Basis size: {results['skqd_results']['basis_size']}")
```

For more control, use the pipeline directly:

```python
from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
from qvartools.molecules import get_molecule

hamiltonian, mol_info = get_molecule("LiH")

config = PipelineConfig(
    teacher_weight=0.5,
    physics_weight=0.4,
    entropy_weight=0.1,
)

pipeline = FlowGuidedKrylovPipeline(
    hamiltonian=hamiltonian,
    config=config,
    auto_adapt=True,
)

results = pipeline.run()
print(f"Energy: {results['final_energy']:.10f} Ha")
```

### Running experiment pipelines

```bash
# Run a single pipeline on H2
python experiments/pipelines/01_dci/dci_krylov_classical.py h2 --device cuda

# Run all 24 pipelines and compare
python experiments/pipelines/run_all_pipelines.py h2 --device cuda

# Skip quantum or iterative pipelines for faster validation
python experiments/pipelines/run_all_pipelines.py h2 --skip-quantum --skip-iterative

# Run with a YAML config override
python experiments/pipelines/02_nf_dci/nf_dci_krylov_classical.py lih \
    --config experiments/pipelines/configs/02_nf_dci.yaml --max-epochs 200
```

## Package Architecture

```
qvartools/
  hamiltonians/   Hamiltonian representations (molecular, Heisenberg, TFIM)
  nqs/            Neural quantum state architectures
  flows/          Normalizing flows and physics-guided training
  krylov/         Krylov subspace methods (SKQD, residual expansion)
  diag/           Eigensolvers, diversity selection, projected Hamiltonians
  solvers/        High-level solver interfaces (FCI, CCSD, SQD, SKQD, iterative)
  samplers/       Configuration samplers (NF, Trotter, CUDA-Q)
  molecules/      Molecular system registry and factory functions
  methods/        End-to-end method pipelines (HI-NQS-SQD, HI-NQS-SKQD)
  _ext/           GPU extensions (sbd subprocess, CUDA-QX VQE/ADAPT-VQE)
  _utils/         Internal utilities (caching, GPU helpers, format conversion)
  pipeline.py     Top-level orchestrator tying all stages together
```

Each subpackage is self-contained with a clean public API. Lower-level modules have no upward dependencies -- `hamiltonians` depends on nothing, `nqs` depends on nothing, `flows` depends on `hamiltonians` and `nqs`, and `pipeline` ties everything together at the top.

## Available Molecules

### Full-Space Systems

| Name | Qubits | Basis Set |
|------|--------|-----------|
| H2   | 4      | sto-3g    |
| LiH  | 12     | sto-3g    |
| BeH2 | 14     | sto-3g    |
| H2O  | 14     | sto-3g    |
| NH3  | 16     | sto-3g    |
| CH4  | 18     | sto-3g    |
| N2   | 20     | cc-pvdz   |
| CO   | 20     | sto-3g    |
| HCN  | 22     | sto-3g    |
| C2H2 | 24     | sto-3g    |
| H2S  | 26     | sto-3g    |
| C2H4 | 28     | sto-3g    |

### CAS Active-Space Systems

| Name | Qubits | Basis Set | Active Space |
|------|--------|-----------|-------------|
| N2-CAS(10,12)  | 24 | cc-pvdz | 10e, 12 orb |
| Cr2             | 24 | sto-3g  | CAS(12,12)  |
| N2-CAS(10,15)  | 30 | cc-pvdz | 10e, 15 orb |
| Benzene         | 30 | sto-3g  | CAS(6,15)   |
| N2-CAS(10,17)  | 34 | cc-pvdz | 10e, 17 orb |
| Cr2-CAS(12,18) | 36 | cc-pvdz | 12e, 18 orb |
| N2-CAS(10,20)  | 40 | cc-pvtz | 10e, 20 orb |
| Cr2-CAS(12,20) | 40 | cc-pvdz | 12e, 20 orb |
| N2-CAS(10,26)  | 52 | cc-pvtz | 10e, 26 orb |
| Cr2-CAS(12,26) | 52 | cc-pvdz | 12e, 26 orb |
| Cr2-CAS(12,28) | 56 | cc-pvdz | 12e, 28 orb |
| Cr2-CAS(12,29) | 58 | cc-pvdz | 12e, 29 orb |
| Cr2-CAS(12,32) | 64 | cc-pvdz | 12e, 32 orb |
| Cr2-CAS(12,36) | 72 | cc-pvdz | 12e, 36 orb |

## Documentation

Full documentation is available at [qvartools.readthedocs.io](https://qvartools.readthedocs.io/en/latest/).

- [Installation Guide](https://qvartools.readthedocs.io/en/latest/getting_started/installation.html)
- [Quickstart Tutorial](https://qvartools.readthedocs.io/en/latest/getting_started/quickstart.html)
- [API Reference](https://qvartools.readthedocs.io/en/latest/api/index.html)
- [Developer Guide](https://qvartools.readthedocs.io/en/latest/developer_guide/index.html)

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style guidelines, and the pull request process.

## Citation

If you use qvartools in your research, please cite:

```bibtex
@software{qvartools,
  author    = {George Chang},
  title     = {qvartools: Quantum Variational Toolkit},
  year      = {2026},
  url       = {https://github.com/QuantumNoLab/qvartools},
  version   = {0.0.0},
}
```

## License

qvartools is released under the [MIT License](LICENSE).
