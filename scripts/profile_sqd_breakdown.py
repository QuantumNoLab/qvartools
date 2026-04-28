#!/usr/bin/env python3
"""
Profile SQD breakdown to find the real bottleneck.

For a N2-CAS(10,20) 40Q system with basis size ~40K, measure time spent in:
  1. bitstring_matrix_to_ci_strs
  2. build link_index
  3. build hdiag
  4. absorb_h1e + ao2mo.restore
  5. Davidson iterations (pure eig)
  6. RDM computation (for comparison)

This tells us exactly where to optimize.
"""
import sys, time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyscf import gto, scf, mcscf, ao2mo
from pyscf.fci import selected_ci, direct_spin1, addons
from pyscf.fci.selected_ci import SCIvector, _all_linkstr_index
from qiskit_addon_sqd.fermion import bitstring_matrix_to_ci_strs
from src.hamiltonians.molecular import create_n2_cas_hamiltonian

print("Building N2-CAS(10,20) 40Q...", flush=True)
H = create_n2_cas_hamiltonian(bond_length=1.10, basis="cc-pvtz", cas=(10, 20), device="cpu")
hcore = H.integrals.h1e
eri = H.integrals.h2e
n_orb = H.n_orbitals
n_alpha = H.n_alpha
n_beta = H.n_beta
n_qubits = 2 * n_orb

# Load the actual cumulative_bs from a previous run (simulate Iter 4 basis ~40K)
# For profiling, we'll generate a synthetic 40K basis
print("Generating synthetic 40K basis...", flush=True)
from itertools import combinations
import random
random.seed(42)

all_alpha_strs = list(combinations(range(n_orb), n_alpha))
all_beta_strs = list(combinations(range(n_orb), n_beta))
n_combinations = min(40000, len(all_alpha_strs) * len(all_beta_strs))

# Sample 40K unique (alpha, beta) pairs
basis_pairs = []
seen = set()
while len(basis_pairs) < min(40000, n_combinations):
    a = random.choice(all_alpha_strs)
    b = random.choice(all_beta_strs)
    if (a, b) not in seen:
        seen.add((a, b))
        basis_pairs.append((a, b))

# Build bitstring matrix
def make_bs(a_pat, b_pat):
    bs = np.zeros(n_qubits, dtype=bool)
    for i in a_pat:
        bs[n_orb - 1 - i] = True
    for i in b_pat:
        bs[n_qubits - 1 - i] = True
    return bs

bs_matrix = np.stack([make_bs(a, b) for a, b in basis_pairs])
print(f"basis shape: {bs_matrix.shape}", flush=True)

# =====================================================================
# Profile each step
# =====================================================================

print("\n--- Profiling each step ---", flush=True)
nelec = (n_alpha, n_beta)

# Step 1: bitstring → ci_strs
t0 = time.time()
ci_strs = bitstring_matrix_to_ci_strs(bs_matrix, open_shell=False)
new_a = np.asarray(ci_strs[0])
new_b = np.asarray(ci_strs[1])
t1 = time.time() - t0
print(f"  Step 1 (bitstring → ci_strs):     {t1*1000:>8.1f} ms  → na={len(new_a)}, nb={len(new_b)}", flush=True)

# Step 2: build myci
t0 = time.time()
myci = selected_ci.SelectedCI()
myci = addons.fix_spin_(myci, ss=0, shift=0.1)
t2 = time.time() - t0
print(f"  Step 2 (build myci + fix_spin):   {t2*1000:>8.1f} ms", flush=True)

# Step 3: link_index (this is suspected bottleneck)
t0 = time.time()
link_index = _all_linkstr_index((new_a, new_b), n_orb, nelec)
t3 = time.time() - t0
print(f"  Step 3 (build link_index):        {t3*1000:>8.1f} ms", flush=True)

# Step 4: hdiag
t0 = time.time()
hdiag = myci.make_hdiag(hcore, eri, (new_a, new_b), n_orb, nelec, compress=True)
t4 = time.time() - t0
print(f"  Step 4 (make_hdiag):              {t4*1000:>8.1f} ms", flush=True)

# Step 5: absorb_h1e + ao2mo.restore
t0 = time.time()
h2e_abs = direct_spin1.absorb_h1e(hcore, eri, n_orb, nelec, .5)
h2e_abs = ao2mo.restore(1, h2e_abs, n_orb)
t5 = time.time() - t0
print(f"  Step 5 (absorb_h1e + ao2mo):      {t5*1000:>8.1f} ms", flush=True)

# Step 6: single contract_2e (one H·v)
from pyscf.fci.selected_ci import _as_SCIvector
na, nb = len(new_a), len(new_b)
v_init = np.random.randn(na * nb)
v_init /= np.linalg.norm(v_init)
v_sci = _as_SCIvector(v_init.reshape(na, nb), (new_a, new_b))

t0 = time.time()
hv = myci.contract_2e(h2e_abs, v_sci, n_orb, nelec, link_index)
t6 = time.time() - t0
print(f"  Step 6 (single contract_2e H·v):  {t6*1000:>8.1f} ms", flush=True)

# Step 7: full kernel_fixed_space (cold start)
t0 = time.time()
e, sci_vec = selected_ci.kernel_fixed_space(
    myci, hcore, eri, n_orb, nelec, (new_a, new_b),
)
t7 = time.time() - t0
print(f"  Step 7 (full kernel_fixed_space): {t7*1000:>8.1f} ms  → E={e:.6f}", flush=True)

# Step 8: full kernel_fixed_space (warm-start with previous sci_vec)
t0 = time.time()
e_warm, sci_vec_warm = selected_ci.kernel_fixed_space(
    myci, hcore, eri, n_orb, nelec, (new_a, new_b), ci0=sci_vec,
)
t8 = time.time() - t0
print(f"  Step 8 (kernel_fixed_space warm): {t8*1000:>8.1f} ms  → E={e_warm:.6f}", flush=True)

print(f"\n  Total (Step 7, cold call): {t7:.1f}s", flush=True)
print(f"  Speedup from warm-start:   {t7/t8:.1f}x", flush=True)
print(f"\n  Breakdown:", flush=True)
print(f"    Setup (Steps 1-5):      {(t1+t2+t3+t4+t5)*1000:.0f} ms ({(t1+t2+t3+t4+t5)/t7*100:.1f}% of cold)", flush=True)
print(f"    Pure Davidson (~):      {(t7-t1-t2-t3-t4-t5)*1000:.0f} ms ({((t7-t1-t2-t3-t4-t5))/t7*100:.1f}% of cold)", flush=True)
