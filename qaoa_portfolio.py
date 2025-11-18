import numpy as np

from qiskit import QuantumCircuit
from qiskit_aer import Aer

# If your previous code is in another file, e.g. `portfolio_costs.py`, adjust this import:
# from portfolio_costs import build_ising_coeffs, portfolio_cost, penalized_cost
from portfolio_costs import build_ising_coeffs


# ============================================================
# 1. QAOA building blocks: cost layer and mixer layer
# ============================================================

def add_cost_layer(qc: QuantumCircuit, gamma: float, W: np.ndarray, w: np.ndarray):
    """
    Add a single QAOA cost layer U_C(gamma) = exp(-i gamma H)
    for Ising Hamiltonian

        H = sum_{i<j} W_ij Z_i Z_j - sum_i w_i Z_i  + const.
    """
    n = len(w)

    # Two-qubit ZZ terms
    for i in range(n):
        for j in range(i + 1, n):
            J = W[i, j]
            if abs(J) > 1e-12:  # skip zeros
                qc.cx(i, j)
                qc.rz(2.0 * gamma * J, j)
                qc.cx(i, j)

    # Single-qubit Z terms
    for i in range(n):
        coeff = -w[i]  # because H has -sum_i w_i Z_i
        if abs(coeff) > 1e-12:
            qc.rz(2.0 * gamma * coeff, i)


def add_mixer_layer(qc: QuantumCircuit, beta: float):
    """
    Standard X-mixer layer:

        M = sum_i X_i
        U_M(beta) = exp(-i beta M)

    Since RX(theta) = exp(-i theta X / 2), we have:
        exp(-i beta X) = RX(2 * beta).
    """
    n = qc.num_qubits
    for i in range(n):
        qc.rx(2.0 * beta, i)


# ============================================================
# 2. Build a full p-layer QAOA circuit
# ============================================================

def build_qaoa_circuit(W: np.ndarray,
                       w: np.ndarray,
                       gammas: np.ndarray,
                       betas: np.ndarray) -> QuantumCircuit:
    """
    Build a depth-p QAOA circuit for the Ising Hamiltonian defined by W, w.
    """
    gammas = np.asarray(gammas, dtype=float)
    betas = np.asarray(betas, dtype=float)
    assert gammas.shape == betas.shape, "gammas and betas must have same shape"
    p = len(gammas)
    n = len(w)

    qc = QuantumCircuit(n)

    # Initial state: |+>^{⊗ n}
    qc.h(range(n))

    # QAOA layers
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        add_cost_layer(qc, gamma, W, w)
        add_mixer_layer(qc, beta)

    # Measurements added later if needed
    return qc


# ============================================================
# 3. (Optional) expectation value ⟨H⟩ with statevector
# ============================================================

def compute_energy_from_samples(bitstrings: np.ndarray,
                                probs: np.ndarray,
                                W: np.ndarray,
                                w: np.ndarray,
                                c: float = 0.0) -> float:
    """
    Compute the expectation value ⟨H⟩ from a list of bitstrings and probabilities,
    where

        H = sum_{i<j} W_ij Z_i Z_j - sum_i w_i Z_i + c I
    """
    n = len(w)
    energy = 0.0

    for z_bits, p in zip(bitstrings, probs):
        # Convert 0/1 bits to Z eigenvalues: +1 (for 0) or -1 (for 1)
        z_eig = np.where(z_bits == 0, 1.0, -1.0)

        # Compute classical energy H(z_eig)
        val = 0.0
        # ZZ terms
        for i in range(n):
            for j in range(i + 1, n):
                val += W[i, j] * z_eig[i] * z_eig[j]
        # Z terms
        for i in range(n):
            val += -w[i] * z_eig[i]

        val += c  # constant shift
        energy += p * val

    return float(energy)


def qaoa_energy_statevector(W: np.ndarray,
                            w: np.ndarray,
                            gammas: np.ndarray,
                            betas: np.ndarray,
                            c: float = 0.0) -> float:
    """
    Build the QAOA circuit, run it with a statevector simulator,
    and compute ⟨H⟩ analytically (for debugging / small n).
    """
    qc = build_qaoa_circuit(W, w, gammas, betas)

    backend = Aer.get_backend("statevector_simulator")
    job = backend.run(qc)
    result = job.result()
    statevector = result.get_statevector(qc)

    # Bitstrings and probabilities
    n = len(w)
    num_states = 2**n
    probs = np.abs(statevector)**2
    bitstrings = np.array(
        [[(i >> j) & 1 for j in range(n)] for i in range(num_states)],
        dtype=int
    )

    energy = compute_energy_from_samples(bitstrings, probs, W, w, c)
    return energy
