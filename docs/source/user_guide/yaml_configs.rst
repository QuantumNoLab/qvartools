YAML Configuration System
=========================

qvartools experiment scripts support a three-tier configuration system:

1. **Built-in defaults** — sensible values for all parameters
2. **YAML config files** — reproducible experiment configurations
3. **CLI overrides** — quick parameter adjustments

CLI arguments take precedence over YAML values, which take precedence over
built-in defaults.

Using Config Files
------------------

Each pipeline group has a matching YAML config in
``experiments/pipelines/configs/``:

.. code-block:: bash

   python experiments/pipelines/002_nf_dci/nf_dci_krylov_classical.py \
       --config experiments/pipelines/configs/002_nf_dci.yaml

CLI Overrides
-------------

Any parameter can be overridden on the command line:

.. code-block:: bash

   # Use YAML config but override the molecule and max epochs
   python experiments/pipelines/002_nf_dci/nf_dci_krylov_classical.py lih \
       --config experiments/pipelines/configs/002_nf_dci.yaml \
       --max-epochs 200 \
       --teacher-weight 0.6

Available Config Files
----------------------

Tier 1 — Ablation groups (one flat YAML per group):

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - Pipeline Group
   * - ``001_dci.yaml``
     - Direct-CI (HF+S+D) — no NF training
   * - ``002_nf_dci.yaml``
     - NF-trained + Direct-CI merged basis
   * - ``003_nf_dci_pt2.yaml``
     - NF + DCI + PT2 perturbative expansion
   * - ``004_nf_only.yaml``
     - NF-only basis (ablation, no DCI merge)
   * - ``005_hf_only.yaml``
     - HF-only reference state (baseline)
   * - ``006_iterative_nqs.yaml``
     - Iterative NQS sampling + diag
   * - ``007_iterative_nqs_dci.yaml``
     - NF+DCI merge then iterative NQS
   * - ``008_iterative_nqs_dci_pt2.yaml``
     - NF+DCI+PT2 then iterative NQS
   * - ``009_vqe.yaml``
     - CUDA-QX VQE (UCCSD + ADAPT-VQE)

Tier 2 — Method-as-pipeline catalog (one multi-section YAML per method):

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - File
     - Method (sections inside)
   * - ``010_hi_nqs_sqd.yaml``
     - HI+NQS+SQD — sections: ``default``, ``pt2``, ``ibm_off``
   * - ``011_hi_nqs_skqd.yaml``
     - HI+NQS+SKQD — sections: ``default``, ``ibm_on``
   * - ``012_nqs_sqd.yaml``
     - NQS+SQD — section: ``default``
   * - ``013_nqs_skqd.yaml``
     - NQS+SKQD — section: ``default``

Config File Structure
---------------------

A typical YAML config file looks like this:

.. code-block:: yaml

   # ---- Molecule -----------------------------------------------
   molecule: h2                  # Molecule identifier

   # ---- Training loss weights ----------------------------------
   teacher_weight: 0.5           # Teacher KL-divergence weight
   physics_weight: 0.4           # Physics-informed energy weight
   entropy_weight: 0.1           # Entropy regularisation weight

   # ---- Training parameters ------------------------------------
   max_epochs: 400               # Maximum training epochs
   min_epochs: 100               # Minimum before early stopping
   samples_per_batch: 2000       # Samples per training batch

   # ---- SKQD parameters ----------------------------------------
   max_krylov_dim: 15            # Maximum Krylov dimension
   shots_per_krylov: 100000      # Shots per Krylov vector

   # ---- Hardware -----------------------------------------------
   device: auto                  # auto, cpu, or cuda

All keys are flat (no nested sections). Keys use underscores and match the
``PipelineConfig`` field names where applicable.

Parameter Reference
-------------------

Common Parameters
^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``molecule``
     - ``h2``
     - Molecule identifier (h2, lih, beh2, h2o, nh3, ch4, n2, c2h4)
   * - ``device``
     - ``auto``
     - PyTorch device: ``auto`` (detect GPU), ``cpu``, ``cuda``
   * - ``verbose``
     - ``true``
     - Print detailed progress

Training Parameters
^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``teacher_weight``
     - ``0.5``
     - Weight for teacher KL-divergence loss
   * - ``physics_weight``
     - ``0.4``
     - Weight for variational energy loss
   * - ``entropy_weight``
     - ``0.1``
     - Weight for entropy regularization
   * - ``max_epochs``
     - auto-scaled
     - Maximum training epochs
   * - ``min_epochs``
     - auto-scaled
     - Minimum epochs before early stopping
   * - ``samples_per_batch``
     - auto-scaled
     - Samples drawn per training batch

SKQD Parameters
^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``max_krylov_dim``
     - auto-scaled
     - Maximum Krylov subspace dimension
   * - ``shots_per_krylov``
     - auto-scaled
     - Shot budget per Krylov vector

SQD Parameters
^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Default
     - Description
   * - ``sqd_num_batches``
     - auto-scaled
     - Number of SQD sample batches
   * - ``sqd_self_consistent_iters``
     - ``5``
     - Self-consistent iteration count
   * - ``sqd_noise_rate``
     - auto-scaled
     - Bitflip noise rate for shot simulation

Auto-Scaling
------------

When parameters are not specified in the config file or CLI, qvartools
automatically scales them based on the Hilbert-space size. This auto-scaling
uses the number of valid configurations (determined by the molecule's orbital
and electron counts) to choose appropriate values for training epochs, samples,
network sizes, and SKQD/SQD parameters.

Explicit config values always override auto-scaled defaults.
