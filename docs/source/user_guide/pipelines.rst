Pipeline Methods
================

qvartools provides a standardized **benchmark catalog** of pipeline methods
under ``experiments/pipelines/``. The catalog uses 3-digit folder prefixes
so future methods can be added without renumbering. There are currently
**33 runnable pipeline scripts** organized in two tiers:

* **001-009** — Ablation groups (basis-generation strategies × diagonalization modes)
* **010-099** — Method-as-pipeline catalog (each NQS method as a first-class entry)

Tier 1 — Ablation groups (001-009)
----------------------------------

Eight ablation groups combine a basis generation strategy with one of three
diagonalization modes (Classical Krylov / Quantum Krylov / SQD), plus a
ninth group for VQE.

.. list-table::
   :header-rows: 1
   :widths: 28 12 12 48

   * - Group
     - NF Training
     - Diag Modes
     - Description
   * - 001 DCI
     - No
     - C / Q / SQD
     - Deterministic HF + singles + doubles
   * - 002 NF+DCI
     - Yes
     - C / Q / SQD
     - NF training + DCI merge
   * - 003 NF+DCI+PT2
     - Yes
     - C / Q / SQD
     - NF + DCI + perturbative expansion
   * - 004 NF-Only
     - Yes
     - C / Q / SQD
     - NF-only basis (ablation, no DCI)
   * - 005 HF-Only
     - No
     - C / Q / SQD
     - Single HF reference state (baseline)
   * - 006 Iterative NQS
     - Iterative
     - C / Q / SQD
     - Autoregressive NQS with eigenvector feedback
   * - 007 NF+DCI -> Iter NQS
     - Yes + Iterative
     - C / Q / SQD
     - NF+DCI merge then iterative NQS refinement
   * - 008 NF+DCI+PT2 -> Iter NQS
     - Yes + Iterative
     - C / Q / SQD
     - NF+DCI+PT2 then iterative NQS refinement
   * - 009 VQE
     - No
     - —
     - CUDA-QX VQE (UCCSD + ADAPT-VQE)

**Diag mode key:** C = Classical Krylov (SKQD), Q = Quantum Circuit Krylov
(Trotterized), SQD = batch diag with noise + S-CORE.

Tier 2 — Method-as-pipeline catalog (010-013)
---------------------------------------------

Each NQS method from ``src/qvartools/methods/nqs/`` is exposed as a runnable
benchmark folder. Variants of the same method live as separate scripts inside
the folder, with one multi-section YAML config per method.

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Pipeline folder
     - Variants
     - Method (``qvartools.methods.nqs``)
   * - ``010_hi_nqs_sqd``
     - ``default``, ``pt2``, ``ibm_off``
     - ``run_hi_nqs_sqd`` — iterative HI loop, optional PT2 selection / IBM solver
   * - ``011_hi_nqs_skqd``
     - ``default``, ``ibm_on``
     - ``run_hi_nqs_skqd`` — iterative HI loop with Krylov expansion
   * - ``012_nqs_sqd``
     - ``default``
     - ``run_nqs_sqd`` — two-stage NQS+SQD (no iteration)
   * - ``013_nqs_skqd``
     - ``default``
     - ``run_nqs_skqd`` — two-stage NQS+SKQD with Krylov expansion

Programmatic dispatch via ``METHODS_REGISTRY``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The four methods are also exposed at runtime as
:data:`qvartools.methods.nqs.METHODS_REGISTRY`, a dict keyed by a stable
method id (``"nqs_sqd"``, ``"nqs_skqd"``, ``"hi_nqs_sqd"``, ``"hi_nqs_skqd"``).
Each value is a sub-dict containing:

* ``run_fn`` — the runner function (e.g. ``run_hi_nqs_sqd``)
* ``config_cls`` — the frozen ``dataclass`` config type
* ``iterative`` — whether the method is an iterative HI loop
* ``has_krylov_expansion`` — whether the basis is grown via Hamiltonian connections
* ``has_ibm_solver`` — whether the method can call IBM ``solve_fermion``
* ``has_pt2_selection`` — whether PT2-based config selection is supported
* ``supports_initial_basis`` — whether a warm-start ``initial_basis`` tensor is accepted
* ``description`` — a short human-readable summary
* ``pipeline_folder`` — the 3-digit experiment folder that wraps this method

The 010-013 wrapper scripts use this registry so they can dispatch by id
without hard-importing each method module, and downstream tooling (benchmark
harnesses, configuration validators, hyperparameter sweep runners) can do
the same.

.. code-block:: python

   from qvartools.methods.nqs import METHODS_REGISTRY
   from qvartools.molecules import get_molecule

   info = METHODS_REGISTRY["hi_nqs_sqd"]
   config = info["config_cls"](n_iterations=5, device="cuda")

   hamiltonian, mol_info = get_molecule("H2")
   result = info["run_fn"](hamiltonian, mol_info, config=config)
   print(f"Energy: {result.energy:.10f} Ha")

Numbering convention
--------------------

The 3-digit prefix scheme leaves room for catalog growth:

* **001-009** — Ablation pipeline groups (current)
* **010-099** — Method-as-pipeline catalog (4 used: 010-013, 86 free)
* **100-199** — Cross-method sweeps (e.g., ``100_h2_all_methods``)
* **200+** — Hyperparameter sweeps

The FlowGuidedKrylovPipeline
-----------------------------

The main pipeline class orchestrates up to four stages:

**Stage 1: Train** — Joint physics-guided training of the normalizing flow and
NQS using a mixed objective (teacher KL-divergence + variational energy +
entropy regularization).

**Stage 2: Select** — Extract accumulated basis configurations from the trained
flow and apply diversity-aware selection to ensure representation across
excitation ranks.

**Stage 2.5: Expand** (Groups 003, 008 only) — Enlarge the basis via CIPSI-style
perturbative selection using Hamiltonian connections.

**Stage 3: Diagonalize** — Run Classical Krylov (SKQD), Quantum Circuit Krylov,
or SQD (batch diag) to compute the ground-state energy.

.. code-block:: python

   from qvartools import PipelineConfig, FlowGuidedKrylovPipeline
   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("BeH2")

   config = PipelineConfig(
       skip_nf_training=False,
       subspace_mode="classical_krylov",   # "classical_krylov", "skqd", or "sqd"
       teacher_weight=0.5,
       physics_weight=0.4,
       entropy_weight=0.1,
   )

   pipeline = FlowGuidedKrylovPipeline(
       hamiltonian=hamiltonian,
       config=config,
       auto_adapt=True,  # auto-scale parameters to system size
   )

   results = pipeline.run()

Iterative Pipelines
--------------------

Groups 006-008 (and method-as-pipeline entries 010-011) use an iterative loop
where the ground-state eigenvector from each diagonalization is fed back as a
training target for the next NQS iteration. The cleanest way to call an
iterative method programmatically is via ``METHODS_REGISTRY``:

.. code-block:: python

   from qvartools.methods.nqs import METHODS_REGISTRY
   from qvartools.molecules import get_molecule

   hamiltonian, mol_info = get_molecule("H2")
   # mol_info from get_molecule() omits orbital counts; every runner extracts
   # them from hamiltonian.integrals automatically via extract_orbital_counts.

   info = METHODS_REGISTRY["hi_nqs_skqd"]
   config = info["config_cls"](
       n_iterations=10,
       n_samples_per_iter=5000,
       device="cuda",
   )

   result = info["run_fn"](hamiltonian, mol_info, config=config)
   print(f"Energy: {result.energy:.10f} Ha")
   print(f"Converged: {result.converged}")

Direct imports still work if you prefer explicit symbols
(``from qvartools.methods.nqs import HINQSSKQDConfig, run_hi_nqs_skqd``), but
the registry-based pattern is what the 010-013 pipeline wrappers use and is
more robust to future additions.

Running Experiment Scripts
--------------------------

All 33 pipelines live in ``experiments/pipelines/``:

.. code-block:: bash

   # Run a single ablation pipeline
   python experiments/pipelines/001_dci/dci_krylov_classical.py h2 --device cuda

   # Run a method-as-pipeline benchmark
   python experiments/pipelines/010_hi_nqs_sqd/default.py h2 --device cuda
   python experiments/pipelines/010_hi_nqs_sqd/pt2.py h2 --device cuda
   python experiments/pipelines/013_nqs_skqd/default.py h2 --device cpu

   # Run all 33 pipelines and compare
   python experiments/pipelines/run_all_pipelines.py h2 --device cuda

   # Filter by group prefix (3-digit)
   python experiments/pipelines/run_all_pipelines.py h2 --only 001 005 010

   # Run with a YAML config
   python experiments/pipelines/002_nf_dci/nf_dci_krylov_classical.py lih \
       --config experiments/pipelines/configs/002_nf_dci.yaml --max-epochs 200

See :doc:`yaml_configs` for details on the configuration system.
