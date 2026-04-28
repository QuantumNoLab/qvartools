# HI+NQS+SQD — Leo's working branch

This branch is a focused snapshot of the HI+NQS+SQD development tree:
**H**andover **I**terative **N**eural **Q**uantum **S**tate sampling +
**S**ample-based **Q**uantum **D**iagonalization for ab-initio quantum
chemistry. Only NQS-SQD source and its benchmarks are included; the
broader qvartools package (NF-NQS, SKQD, quantum-circuit pipelines)
has been stripped from this branch.

## Layout

```
src/                        — research source tree
  hamiltonians/molecular.py   — Slater–Condon, vectorized connections
  methods/
    hi_nqs_sqd{,_v2..v5}.py   — HI+NQS+SQD pipeline versions
    nqs_vmc_update.py         — joint-c² teacher + VMC REINFORCE
    nqs_sr_{update,optimizer}.py
    nqs_mcmc_sampler.py
    multi_temp_sampler.py
    multi_gpu_*.py            — multi-GPU sampling / Davidson / PT2
    gpu_{coupling,davidson,sparse_det_*,sparse_sqd}.py
    sparse_det_{backend,solver}.py
    incremental_sqd.py
  nqs/transformer.py          — autoregressive Transformer (|ψ|² only)
  molecules/                  — registry, FCIDUMP loaders, CAS factories
  solvers/
    sci.py                    — selected-CI expander
    fci.py                    — small-system reference

scripts/                    — SLURM job + worker pairs (≈150 files)
scratch_*.py                — driver scripts (one per experiment)
docs/decisions/             — ADRs (006 = current bottleneck analysis)

HI_NQS_SQD_benchmark_data.md     — accumulated benchmark numbers
HI_NQS_SQD_comparison_table.md   — method/molecule comparison table
docs/PAPER_FIGURE_GUIDE.md       — figure plan for the paper
```

## Current research status

See [`docs/decisions/006-nqs-mode-collapse-and-fixes.md`](docs/decisions/006-nqs-mode-collapse-and-fixes.md)
for the current bottleneck analysis. Headline:

- For `N2-CAS(10,26)` (52 qubits) the pipeline plateaus at +71 mHa
  above HCI when `classical_expansion=True`.
- Six-mode ablation isolates the NQS sampler as the cause: in
  `nqs_only` mode (no Slater–Condon expansion safety net) NQS samples
  collapse onto the existing basis — `500k samples → 0 new dets` at
  iter 1.
- A REINFORCE patch on `|coupling|²` made things worse (consistent
  with arXiv:2602.12993 NQS-VMC failure mode).
- A one-line conditional β=0.4 reshape (arXiv:2603.24728) at sampling
  time **broke the plateau by 76 mHa at iter 7** in early runs. The
  framework's `final_temperature=0.3` default was annealing the
  *wrong* direction (β = 3.3, sharpening into collapse).

Two further fixes (truncated-energy supervised loss; NQS-as-ranker
instead of NQS-as-sampler) are queued.

## Reproducing the β-reshape benchmark

```bash
# 8 GPUs, β ∈ {0.4, 0.2}, mol ∈ {40Q, 52Q}, seed ∈ {42, 777}
sbatch scripts/bench_nqs_beta_reshape.slurm
```

The driver is `scratch_diagnose_nqs_role.py`; modes
`nqs_only_beta04` / `nqs_only_beta02` install a sampler that forces
`temperature = 1/β`, equivalent to per-conditional β reshape on
Bernoulli autoregressive heads.

## Reference papers

| arXiv | Used for |
|-------|----------|
| [2603.24728](https://arxiv.org/abs/2603.24728) | β-reshape recipe; sign offloading to diagonalizer |
| [2602.12993](https://arxiv.org/abs/2602.12993) | NQS-SC (truncated energy) > NQS-VMC |
| [2406.08154](https://arxiv.org/abs/2406.08154) | Active-learning ranker on H-action candidates |
| [2109.12606](https://arxiv.org/abs/2109.12606) | NAQS — autoregressive ab-initio with symmetry masks |

Other ADRs in `docs/decisions/` cover earlier design decisions (PT2
selection, GPU-native SBD, CAS scale-up, SKQD naming).

## Branch hygiene

`feat/gpu-multi-davidson-v4` is the upstream HI-VQE working branch
this snapshot was lifted from. This branch is intentionally a
research snapshot, not a packaged release — it does not ship a
`pyproject.toml` or test suite. Use HI-VQE's environment
(`scripts/setup_cuda_env.sh`) to run.
