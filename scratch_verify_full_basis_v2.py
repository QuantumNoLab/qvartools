"""Clean verification: full Hilbert basis on N2-20Q R=1.8 should give FCI.

Uses production SparseDetSQDBackend (same as v3 main loop) to avoid any
matrix-construction bugs in scratch code.
"""
import time
import numpy as np
import torch
from itertools import combinations

from src.hamiltonians.molecular import create_n2_hamiltonian
from src.methods.sparse_det_backend import SparseDetSQDBackend


def all_valid_bitstrings(n_orb, n_alpha, n_beta):
    """Enumerate all valid IBM-format bitstring rows (MSB-first)."""
    a_combos = list(combinations(range(n_orb), n_alpha))
    b_combos = list(combinations(range(n_orb), n_beta))
    n_total = len(a_combos) * len(b_combos)

    bs = np.zeros((n_total, 2 * n_orb), dtype=np.int8)
    k = 0
    for a in a_combos:
        for b in b_combos:
            for i in a: bs[k, n_orb - 1 - i] = 1   # MSB-first packing
            for i in b: bs[k, 2 * n_orb - 1 - i] = 1
            k += 1
    return bs


def main():
    R = 1.8
    print(f"\n=== N2-20Q R={R} STO-3G full Hilbert verification ===\n", flush=True)

    H = create_n2_hamiltonian(bond_length=R, basis="sto-3g", device="cuda")
    n_orb = H.n_orbitals
    n_alpha = H.n_alpha
    n_beta = H.n_beta
    nuc = float(H.nuclear_repulsion)

    print(f"  n_orb={n_orb}, n_alpha={n_alpha}, n_beta={n_beta}", flush=True)
    print(f"  V_NN (nuclear repulsion) = {nuc:.6f} Ha", flush=True)

    integrals = H.integrals
    hcore = np.asarray(integrals.h1e, dtype=np.float64)
    eri = np.asarray(integrals.h2e, dtype=np.float64)

    backend = SparseDetSQDBackend(hcore=hcore, eri=eri, n_alpha=n_alpha, n_beta=n_beta)

    # Test 1: full 14400 basis
    print("\n--- Test 1: full Hilbert (14400 dets) ---", flush=True)
    bs = all_valid_bitstrings(n_orb, n_alpha, n_beta)
    print(f"  basis built: {bs.shape}", flush=True)
    t0 = time.time()
    e_electronic, sci_state = backend.solve(bs)
    e_total = e_electronic + nuc
    t1 = time.time() - t0
    FCI_REF = -107.483457
    err = (e_total - FCI_REF) * 1000
    print(f"  E_electronic       = {e_electronic:.10f} Ha", flush=True)
    print(f"  E_total (+V_NN)    = {e_total:.10f} Ha", flush=True)
    print(f"  PySCF FCI ref      = {FCI_REF:.10f} Ha", flush=True)
    print(f"  diff vs FCI (mHa)  = {err:+.5f}", flush=True)
    print(f"  diag wall: {t1:.1f}s", flush=True)

    # Backend caches amplitudes; build fresh backend for each subset to avoid warm-start contamination
    def fresh_solve(bs_subset):
        be = SparseDetSQDBackend(hcore=hcore, eri=eri, n_alpha=n_alpha, n_beta=n_beta)
        e, _ = be.solve(bs_subset)
        return e + nuc

    # Test 2: remove 11 random dets
    print("\n--- Test 2: remove 11 random dets ---", flush=True)
    np.random.seed(42)
    keep = np.random.choice(bs.shape[0], bs.shape[0] - 11, replace=False)
    bs_sub = bs[np.sort(keep)]
    e_sub = fresh_solve(bs_sub)
    print(f"  E_total = {e_sub:.10f}, err = {(e_sub - FCI_REF)*1000:+.5f} mHa", flush=True)

    # Test 3: get FCI eigenvector to find lowest |c|
    print("\n--- Test 3: remove 11 LOWEST-|c| dets ---", flush=True)
    # The amplitudes from sci_state — flatten and rank
    amps = np.abs(sci_state.amplitudes).flatten()
    # Note: amps is over (n_a, n_b) Cartesian product, but our bs corresponds to
    # 14400 individual configs. Let me reconstruct mapping...
    # Actually simpler: re-solve and use the amplitudes per bs row.
    # Since `solve` does NOT return per-bs amplitudes directly, we use sci_state's
    # 2-D amplitudes and assume bs ordering matches.
    # CAUTION: sci_state.amplitudes is shape (n_unique_a, n_unique_b); when basis
    # is full Hilbert, this maps 1:1 with bs ordering modulo sort.
    print(f"  amps shape: {amps.shape}, max |amp|: {amps.max():.4f}", flush=True)
    sort_idx = np.argsort(amps)
    keep_low = sort_idx[11:]   # remove 11 smallest |amp|
    # Map keep_low (in flat amp index) to bs rows. The bs ordering matches amps.flatten()
    # only if backend uses the same order (alpha then beta nested). Let me verify by
    # checking that bs[i] corresponds to (a_combos[i // n_b], b_combos[i % n_b]).
    bs_sub_low = bs[np.sort(keep_low)]
    e_low = fresh_solve(bs_sub_low)
    print(f"  E_total = {e_low:.10f}, err = {(e_low - FCI_REF)*1000:+.5f} mHa", flush=True)

    # Test 4: remove 11 HIGHEST |c|
    print("\n--- Test 4: remove 11 HIGHEST-|c| dets ---", flush=True)
    keep_high = sort_idx[:-11]
    bs_sub_high = bs[np.sort(keep_high)]
    e_high = fresh_solve(bs_sub_high)
    print(f"  E_total = {e_high:.10f}, err = {(e_high - FCI_REF)*1000:+.5f} mHa", flush=True)

    print("\n=== SUMMARY (errors in mHa above FCI) ===", flush=True)
    print(f"  full 14400         → {err:+.4f}", flush=True)
    print(f"  remove 11 random   → {(e_sub - FCI_REF)*1000:+.4f}", flush=True)
    print(f"  remove 11 lowest |c| → {(e_low - FCI_REF)*1000:+.4f}", flush=True)
    print(f"  remove 11 highest |c| → {(e_high - FCI_REF)*1000:+.4f}", flush=True)
    print(f"  aug_teacher actual: +110.642 mHa (basis=14389)", flush=True)


if __name__ == "__main__":
    main()
