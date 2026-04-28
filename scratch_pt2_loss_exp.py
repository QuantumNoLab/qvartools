"""A/B experiment: PT2-augmented teacher vs standard teacher on N2 20Q.

Hypothesis: the current HI-NQS-SQD loss trains NQS to reproduce |c_x|² for x∈V,
which structurally cannot tell it about important external dets (x∉V with small
|c_x| but large coupling |t_x| = |Σ_i c_i H_xi|). Extending the teacher to
include the top-K external dets (weighted by PT2 score |t_x|²/|E - H_xx|) gives
NQS a direct training signal for the exploration space SCI actually needs.

Experiment:
  - Same molecule (N2 20Q, FCI = -107.654122 Ha, Hilbert = 14,400)
  - Same seed, same NQS arch, same budget
  - max_basis_size = 500 (3.5% of Hilbert) forces plateau
  - Variant A: standard _update_nqs (teacher = V amplitudes only)
  - Variant B: PT2-augmented (teacher = V ∪ top-K external PT2 dets)

Measure:
  - Final variational E vs FCI
  - Convergence speed (iter to hit a given threshold)
  - Composition of V (overlap with HCI-style top dets)
"""
from __future__ import annotations

import copy
import time
import numpy as np
import torch

from src.molecules import get_molecule
from src.methods.hi_nqs_sqd import (
    HINQSSQDConfig, run_hi_nqs_sqd,
)
import src.methods.hi_nqs_sqd as hi_mod

from scratch_pt2_methods import build_pt2_candidates, compute_fci_reference


# ==========================================================================
# PT2-augmented _update_nqs (drop-in replacement)
# ==========================================================================
def make_pt2_augmented_update(h1e, eri, ecore, top_K_ext=500, lambda_ext=0.3,
                                verbose=False):
    """Factory: returns an _update_nqs variant that mixes V teacher with
    top-K external PT2 dets.

    The returned function has the same signature as hi_mod._update_nqs so it
    can be monkey-patched in without changing any other code.
    """
    # Reuse the original helpers from hi_mod
    _ibm_format_to_configs = hi_mod._ibm_format_to_configs

    def _update_nqs_pt2_aug(
        nqs, optimizer, cumulative_bs, e0,
        sci_state, hamiltonian, cfg, device,
        n_orb, n_qubits,
    ):
        # ---- Step 1: standard V teacher (copied from original _update_nqs) ----
        configs = _ibm_format_to_configs(cumulative_bs, n_orb, n_qubits)
        n_V = len(configs)

        amps = np.abs(sci_state.amplitudes) ** 2
        alpha_marginal = amps.sum(axis=1)
        beta_marginal = amps.sum(axis=0)
        alpha_marginal /= max(alpha_marginal.sum(), 1e-30)
        beta_marginal /= max(beta_marginal.sum(), 1e-30)

        alpha_map = {int(s): float(v)
                     for s, v in zip(sci_state.ci_strs_a, alpha_marginal)}
        beta_map = {int(s): float(v)
                    for s, v in zip(sci_state.ci_strs_b, beta_marginal)}

        powers_msb = (1 << np.arange(n_orb - 1, -1, -1)).astype(np.int64)
        bs_int = np.asarray(cumulative_bs).astype(np.int64)
        a_ints = (bs_int[:, :n_orb] * powers_msb).sum(axis=1)
        b_ints = (bs_int[:, n_orb:] * powers_msb).sum(axis=1)

        V_alpha_vals = np.array([alpha_map.get(int(a), 0.0) for a in a_ints])
        V_beta_vals = np.array([beta_map.get(int(b), 0.0) for b in b_ints])
        V_weights = V_alpha_vals * V_beta_vals
        total_V = V_weights.sum()
        if total_V > 0:
            V_weights /= total_V

        # ---- Step 2: external PT2 teacher ----
        # Get full-c_V from sci_state by looking up each (a_int, b_int)
        ci_a_lookup = {int(s): i for i, s in enumerate(sci_state.ci_strs_a)}
        ci_b_lookup = {int(s): i for i, s in enumerate(sci_state.ci_strs_b)}
        c_V = np.zeros(n_V, dtype=np.float64)
        for k in range(n_V):
            ia = ci_a_lookup.get(int(a_ints[k]), -1)
            ib = ci_b_lookup.get(int(b_ints[k]), -1)
            if ia >= 0 and ib >= 0:
                c_V[k] = sci_state.amplitudes[ia, ib]

        ext_alpha, ext_beta, t, H_diag = build_pt2_candidates(
            a_ints.astype(np.int64), b_ints.astype(np.int64),
            c_V, h1e, eri, n_orb,
        )

        ext_configs_t = None
        ext_weights = None
        K_actual = 0

        if len(t) > 0:
            denom = (e0 - ecore) - H_diag
            denom_safe = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            pt2_score = t ** 2 / np.abs(denom_safe)

            K_actual = min(top_K_ext, len(pt2_score))
            if K_actual > 0:
                top_idx = np.argsort(pt2_score)[::-1][:K_actual]
                ext_a_sel = ext_alpha[top_idx]
                ext_b_sel = ext_beta[top_idx]
                ext_scores_sel = pt2_score[top_idx]

                # Convert (alpha_int, beta_int) → NQS config tensor
                # NQS config convention: config[k] = bit k of alpha_int (LSB-first)
                # verified from _ibm_format_to_configs path in hi_nqs_sqd.py
                ext_configs_np = np.zeros((K_actual, 2 * n_orb), dtype=np.int64)
                bit_shifts = np.arange(n_orb)
                ext_configs_np[:, :n_orb] = (
                    (ext_a_sel[:, None] >> bit_shifts) & 1
                ).astype(np.int64)
                ext_configs_np[:, n_orb:] = (
                    (ext_b_sel[:, None] >> bit_shifts) & 1
                ).astype(np.int64)
                ext_configs_t = torch.from_numpy(ext_configs_np).long()

                ext_weights = ext_scores_sel / ext_scores_sel.sum()

        # ---- Step 3: combine teachers ----
        if ext_configs_t is not None and len(ext_weights) > 0:
            combined_configs = torch.cat([configs, ext_configs_t], dim=0)
            V_part = (1 - lambda_ext) * V_weights
            E_part = lambda_ext * ext_weights
            combined_w = np.concatenate([V_part, E_part])
            combined_w /= max(combined_w.sum(), 1e-30)
            if verbose:
                print(f"      [pt2-aug] V={n_V}, ext_K={K_actual}, "
                      f"λ={lambda_ext}, top-1 PT2 score={ext_weights[0]:.2e}")
        else:
            combined_configs = configs
            combined_w = V_weights

        teacher_t = torch.from_numpy(combined_w).float().to(device)
        n_total = len(combined_configs)

        with torch.no_grad():
            diag_e = hamiltonian.diagonal_elements_batch(combined_configs)
            if isinstance(diag_e, torch.Tensor):
                diag_e_t = diag_e.to(device=device, dtype=torch.float32)
            else:
                diag_e_t = torch.tensor(
                    np.asarray(diag_e, dtype=np.float64),
                    dtype=torch.float32, device=device,
                )
            advantage = diag_e_t - e0

        # ---- Step 4: training loop (exactly like original _update_nqs) ----
        max_batch = min(5000, n_total)
        for step in range(cfg.nqs_steps):
            optimizer.zero_grad()
            if n_total > max_batch:
                idx = torch.randperm(n_total)[:max_batch]
                batch_configs = combined_configs[idx].float().to(device)
                batch_teacher = teacher_t[idx]
                batch_teacher = batch_teacher / max(batch_teacher.sum(), 1e-30)
                batch_advantage = advantage[idx]
            else:
                batch_configs = combined_configs.float().to(device)
                batch_teacher = teacher_t
                batch_advantage = advantage

            log_probs = nqs.log_prob(batch_configs)

            loss_teacher = -(batch_teacher * log_probs).sum()
            loss_energy = (batch_teacher * batch_advantage * log_probs).sum()
            loss_entropy = log_probs.mean()

            loss = (cfg.teacher_weight * loss_teacher
                    + cfg.energy_weight * loss_energy
                    + cfg.entropy_weight * loss_entropy)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(nqs.parameters(), max_norm=1.0)
            optimizer.step()

    return _update_nqs_pt2_aug


# ==========================================================================
# Experiment runner
# ==========================================================================
def run_variant(mol_name, cfg, seed, variant: str,
                h1e, eri, ecore, top_K_ext, lambda_ext):
    """Run HI-NQS-SQD with optional PT2-augmented teacher."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    H, info = get_molecule(mol_name)

    original_update = hi_mod._update_nqs

    try:
        if variant == "pt2_aug":
            hi_mod._update_nqs = make_pt2_augmented_update(
                h1e=h1e, eri=eri, ecore=ecore,
                top_K_ext=top_K_ext, lambda_ext=lambda_ext,
                verbose=False,
            )

        t0 = time.time()
        result = run_hi_nqs_sqd(H, info, config=cfg)
        wall = time.time() - t0
    finally:
        hi_mod._update_nqs = original_update

    return result, wall


def main():
    MOLECULE = "N2"  # 20Q, Hilbert=14400, FCI solvable
    SEED = 2024
    TOP_K_EXT = 500      # how many external dets to mix into teacher per iter
    LAMBDA_EXT = 0.3     # teacher weight given to external vs V part

    # Get integrals for FCI + PT2-aug setup
    H, info = get_molecule(MOLECULE)
    h1e = np.asarray(H.integrals.h1e, dtype=np.float64)
    eri = np.asarray(H.integrals.h2e, dtype=np.float64)
    ecore = float(H.integrals.nuclear_repulsion)
    norb = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta

    # FCI reference
    print(f"{'='*78}")
    print(f"  {MOLECULE}: {norb} orbitals, ({n_alpha},{n_beta}) electrons")
    t0 = time.time()
    E_FCI = compute_fci_reference(h1e, eri, norb, n_alpha, n_beta, ecore)
    print(f"  FCI = {E_FCI:.10f} Ha  (t={time.time()-t0:.1f}s)")
    print(f"{'='*78}")

    # Tight budget to force plateau
    base_cfg = HINQSSQDConfig(
        max_iterations=25,
        convergence_threshold=1e-6,
        convergence_window=3,
        n_samples=2000,
        top_k=200,
        max_basis_size=500,   # tight cap — forces NQS to really cover the right 500
        nqs_steps=10,
        nqs_lr=1e-3,
        teacher_weight=1.0,
        energy_weight=0.1,
        entropy_weight=0.05,
        initial_temperature=1.0,
        final_temperature=0.3,
        warm_start=True,
        use_incremental_sqd=False,
        use_sparse_det_solver=True,
        monotonic_basis=False,
    )

    # --- Variant A: standard teacher ---
    print(f"\n[A] Standard teacher (V-only, no PT2 augmentation)")
    result_A, t_A = run_variant(
        MOLECULE, base_cfg, SEED, "standard",
        h1e, eri, ecore, TOP_K_EXT, LAMBDA_EXT,
    )

    # --- Variant B: PT2-augmented teacher ---
    print(f"\n[B] PT2-augmented teacher (V ∪ top-{TOP_K_EXT} ext, λ={LAMBDA_EXT})")
    result_B, t_B = run_variant(
        MOLECULE, base_cfg, SEED, "pt2_aug",
        h1e, eri, ecore, TOP_K_EXT, LAMBDA_EXT,
    )

    # --- Compare ---
    print(f"\n{'='*78}")
    print(f"  RESULTS")
    print(f"{'='*78}")
    print(f"  FCI                     = {E_FCI:.10f} Ha")
    print(f"  [A] standard     E      = {result_A.energy:.10f} Ha "
          f"({(result_A.energy-E_FCI)*1000:+.3f} mHa vs FCI)")
    print(f"      basis final         = {result_A.diag_dim}")
    print(f"      converged           = {result_A.converged}  "
          f"t={t_A:.1f}s")
    print(f"  [B] PT2-augmented E    = {result_B.energy:.10f} Ha "
          f"({(result_B.energy-E_FCI)*1000:+.3f} mHa vs FCI)")
    print(f"      basis final         = {result_B.diag_dim}")
    print(f"      converged           = {result_B.converged}  "
          f"t={t_B:.1f}s")

    delta = (result_A.energy - result_B.energy) * 1000
    print(f"\n  B improves over A by   = {delta:+.3f} mHa  "
          f"({'BETTER' if delta > 0 else 'WORSE'})")

    # Iter-by-iter comparison
    hA = result_A.metadata.get("energy_history", [])
    hB = result_B.metadata.get("energy_history", [])
    max_iter = max(len(hA), len(hB))
    print(f"\n  iter-by-iter (err vs FCI in mHa):")
    print(f"    {'iter':>4}  {'E_A err':>10}  {'E_B err':>10}  {'B-A':>9}")
    for i in range(max_iter):
        eA_str = f"{(hA[i]-E_FCI)*1000:+.3f}" if i < len(hA) else "   —   "
        eB_str = f"{(hB[i]-E_FCI)*1000:+.3f}" if i < len(hB) else "   —   "
        diff = "—"
        if i < len(hA) and i < len(hB):
            diff = f"{(hB[i]-hA[i])*1000:+.3f}"
        print(f"    {i:>4}  {eA_str:>10}  {eB_str:>10}  {diff:>9}")


if __name__ == "__main__":
    main()
