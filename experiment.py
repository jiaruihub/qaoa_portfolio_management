#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from itertools import product

from qiskit import QuantumCircuit
from qiskit_aer import Aer

# Adjust these imports to match your actual filenames / module names
from portfolio_costs import portfolio_cost, build_ising_coeffs
from qaoa_portfolio import build_qaoa_circuit, qaoa_energy_statevector


# ============================================================
# 1. Generate a synthetic portfolio optimization instance
# ============================================================

def generate_random_portfolio_instance(n, seed=123):
    """
    Generate a random (mu, sigma) pair for testing.

    - mu: random expected returns in [0, 0.2]
    - sigma: random positive semidefinite covariance matrix
    """
    rng = np.random.default_rng(seed)
    mu = 0.2 * rng.random(n)

    # Make a random PSD covariance: sigma = A A^T, then scale
    A = rng.normal(size=(n, n))
    sigma = A @ A.T
    # Normalize diagonal to something like [0.05, 0.2]
    diag = np.diag(sigma)
    scale = 0.1 / np.mean(diag)
    sigma *= scale

    return mu, sigma


# ============================================================
# 2. Classical brute-force optimum (for small n)
# ============================================================

def classical_optimum(mu, sigma, q, B):
    """
    Brute-force search over all bitstrings of length n and
    find the feasible portfolio (sum z_i == B) with minimum F(z).
    """
    n = len(mu)
    best_val = None
    best_z = None

    for bits in product([0, 1], repeat=n):
        z = np.array(bits, dtype=int)
        if z.sum() != B:
            continue
        val = portfolio_cost(z, mu, sigma, q)
        if best_val is None or val < best_val:
            best_val = val
            best_z = z.copy()

    return best_z, best_val


# ============================================================
# 3. Simple random-search optimizer for QAOA angles
# ============================================================

def random_qaoa_optimize(W, w, p, num_trials=50, c=0.0, seed=42):
    """
    Very simple random-search over QAOA parameters:

        gammas, betas ~ Uniform[0, 2π]^p

    Returns the best energy and corresponding angles.
    """
    rng = np.random.default_rng(seed)
    best_energy = None
    best_gammas = None
    best_betas = None

    for _ in range(num_trials):
        gammas = 2 * np.pi * rng.random(p)
        betas = 2 * np.pi * rng.random(p)
        energy = qaoa_energy_statevector(W, w, gammas, betas, c)
        if best_energy is None or energy < best_energy:
            best_energy = energy
            best_gammas = gammas
            best_betas = betas

    return best_energy, best_gammas, best_betas


# ============================================================
# 4. Sample output bitstrings from QAOA circuit
# ============================================================

def sample_qaoa_bitstrings(W, w, gammas, betas, shots=2000):
    """
    Build the QAOA circuit with given angles, run on qasm_simulator,
    and return counts over bitstrings.
    """
    qc = build_qaoa_circuit(W, w, gammas, betas)
    qc.measure_all()

    backend = Aer.get_backend("qasm_simulator")
    job = backend.run(qc, shots=shots)
    result = job.result()
    counts = result.get_counts(qc)

    return counts


def bitstring_to_array(bitstring, n):
    """
    Convert a string like '1010' (Qiskit's big-endian) to a numpy array
    [z_0, ..., z_{n-1}] with z_0 being qubit 0, etc.

    Qiskit uses little-endian order for qubits, but prints bitstrings with
    qubit 0 on the right. So '1010' means:

        qubit 3: 1
        qubit 2: 0
        qubit 1: 1
        qubit 0: 0

    We'll reverse the string to align indices.
    """
    bits_reversed = bitstring[::-1]
    return np.array([int(b) for b in bits_reversed], dtype=int)



# In[2]:


# ============================================================
# 5. Full example: n=6 portfolio, p-layer QAOA
# ============================================================

if __name__ == "__main__":
    # Problem size and hyperparameters
    n = 6
    q = 1.0 / 3.0       # risk preference
    B = 3               # choose exactly 3 assets
    A = 5.0             # penalty factor (you can play with this)
    lam = 1.0           # global scaling factor for Hamiltonian
    p = 2               # QAOA depth

    # 1) Generate random instance
    mu, sigma = generate_random_portfolio_instance(n, seed=123)

    print("mu =", mu)
    print("sigma =\n", sigma)

    # 2) Classical optimum for comparison
    z_opt, F_opt = classical_optimum(mu, sigma, q, B)
    print("\nClassical optimum:")
    print("  z*     =", z_opt)
    print("  F(z*)  =", F_opt)

    # 3) Build Ising Hamiltonian coefficients
    W, w, c = build_ising_coeffs(mu, sigma, q, B, A, lam)
    print("\nIsing coefficients:")
    print("W (ZZ couplings) =\n", W)
    print("w (local fields) =", w)

    # 4) Random-search QAOA optimization
    print("\nRunning random-search QAOA optimization...")
    best_energy, best_gammas, best_betas = random_qaoa_optimize(
        W, w, p, num_trials=80, c=c, seed=999
    )

    print("\nBest QAOA energy (statevector expectation):", best_energy)
    print("Best gammas:", best_gammas)
    print("Best betas:", best_betas)

    # 5) Sample bitstrings from the best QAOA circuit
    counts = sample_qaoa_bitstrings(W, w, best_gammas, best_betas, shots=4000)

    # Sort bitstrings by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)

    print("\nTop QAOA bitstrings (by frequency):")
    for bitstring, cnt in sorted_counts[:10]:
        z = bitstring_to_array(bitstring, n)
        F_val = portfolio_cost(z, mu, sigma, q)
        print(f"  {bitstring} (count={cnt})  z={z},  F(z)={F_val:.6f},  sum(z)={z.sum()}")

    # 6) Compare best sampled feasible portfolio to classical optimum
    best_sample_F = None
    best_sample_z = None
    total_shots = sum(counts.values())

    for bitstring, cnt in counts.items():
        z = bitstring_to_array(bitstring, n)
        if z.sum() != B:
            continue  # only compare feasible
        F_val = portfolio_cost(z, mu, sigma, q)
        if best_sample_F is None or F_val < best_sample_F:
            best_sample_F = F_val
            best_sample_z = z

    if best_sample_z is not None:
        print("\nBest FEASIBLE portfolio sampled by QAOA:")
        print("  z_QAOA =", best_sample_z)
        print("  F(z_QAOA) =", best_sample_F)
        print("  Gap to classical optimum:", best_sample_F - F_opt)
    else:
        print("\nNo feasible (sum z_i == B) portfolio was sampled. "
              "You may want to adjust A, p, or num_trials.")


# In[ ]:




