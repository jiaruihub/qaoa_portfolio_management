import numpy as np
from itertools import product


# -----------------------------
# 1. Classical cost functions
# -----------------------------

def portfolio_cost(z, mu, sigma, q):
    """
    Classical Markowitz-style portfolio cost (Eq. (1) in the paper):

        F(z) = q * sum_{i,j} z_i z_j sigma_{ij}
               - (1 - q) * sum_i z_i mu_i

    Parameters
    ----------
    z : array_like, shape (n,)
        Binary vector of 0/1 portfolio inclusion variables.
    mu : array_like, shape (n,)
        Expected returns μ_i.
    sigma : array_like, shape (n, n)
        Covariance matrix σ_{ij}.
    q : float
        Risk-preference parameter in [0, 1].

    Returns
    -------
    float
        Value of F(z).
    """
    z = np.asarray(z, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    risk_term = q * z @ sigma @ z
    return_term = (1.0 - q) * mu @ z

    return risk_term - return_term


def penalized_cost(z, mu, sigma, q, B, A):
    """
    Penalized cost F^(A)(z) = F(z) + A (sum_i z_i - B)^2 (Eq. (9)).

    Parameters
    ----------
    z : array_like, shape (n,)
        Binary 0/1 vector.
    mu, sigma, q : as in portfolio_cost.
    B : int
        Budget (number of assets to include).
    A : float
        Penalty factor.

    Returns
    -------
    float
        Penalized cost F^(A)(z).
    """
    z = np.asarray(z, dtype=float)
    base = portfolio_cost(z, mu, sigma, q)
    penalty = A * (z.sum() - B)**2
    return base + penalty


# ----------------------------------------------------------
# 2. Brute-force helpers for small n (for analysis / A, λ)
# ----------------------------------------------------------

def all_bitstrings(n):
    """Generate all {0,1}^n as numpy arrays."""
    for bits in product([0, 1], repeat=n):
        yield np.array(bits, dtype=int)


def brute_force_stats(mu, sigma, q, B, A):
    """
    For small n, compute:
      - F_min, F_max over feasible z (sum z_i = B)
      - F_nf_min over infeasible z
      - mean F over feasible z

    This matches Sect. 2.2–2.3 definitions in the paper.
    """
    n = len(mu)
    F_feasible = []
    F_infeasible = []

    for z in all_bitstrings(n):
        F_val = portfolio_cost(z, mu, sigma, q)
        if z.sum() == B:
            F_feasible.append(F_val)
        else:
            # use penalized F^(A) for unfeasible
            F_infeasible.append(penalized_cost(z, mu, sigma, q, B, A))

    F_feasible = np.array(F_feasible)
    F_infeasible = np.array(F_infeasible)

    F_min = F_feasible.min()
    F_max = F_feasible.max()
    F_mean = F_feasible.mean()
    F_nf_min = F_infeasible.min()

    return F_min, F_max, F_mean, F_nf_min


# -----------------------------------------------------------------
# 3. Penalty factor update ΔA (Eq. (13)) — optional, small-n only
# -----------------------------------------------------------------

def penalty_update_delta_A(mu, sigma, q, B, A, z_star, F_min, F_mean):
    """
    Compute ΔA from Eq. (13):

        ΔA = [ 0.5 (F_min + F_mean) - F_nf_min(z_star) ] / (sum_i z*_i - B)^2

    Here we assume z_star is the unfeasible state achieving F_nf_min at current A.

    This is only practical for small n where brute force is feasible.
    """
    z_star = np.asarray(z_star, dtype=int)
    numerator = 0.5 * (F_min + F_mean) - penalized_cost(z_star, mu, sigma, q, B, A)
    denom = (z_star.sum() - B)**2
    if denom == 0:
        raise ValueError("z_star must be infeasible: sum(z_star) != B.")
    return numerator / denom


# ----------------------------------------------------------------
# 4. Build Ising-form Hamiltonian coefficients (Eq. (15))
# ----------------------------------------------------------------

def build_ising_coeffs(mu, sigma, q, B, A, lam=1.0):
    """
    Build the Ising Hamiltonian coefficients corresponding to the quantum
    operator F̂ (Eq. (14)–(15)):

        F̂ = λ F^(A)((I + Z_1)/2, ..., (I + Z_n)/2)
           = sum_{i<j} W_ij Z_i Z_j - sum_i w_i Z_i + c I

    with:

        W_ij = λ/2 * (q σ_ij + A),  for i < j
        w_i  = λ/2 * [ (1 - q) μ_i + A (2B - n) - q * sum_j σ_ij ]

    The constant c is irrelevant for QAOA dynamics and is set to 0 here.

    Parameters
    ----------
    mu : array_like, shape (n,)
        Expected returns μ_i.
    sigma : array_like, shape (n, n)
        Covariance matrix σ_ij.
    q : float
        Risk-preference parameter.
    B : int
        Budget.
    A : float
        Penalty factor.
    lam : float, optional
        Global scaling factor λ (Sec. 3.2.1). Defaults to 1.0.

    Returns
    -------
    W : ndarray, shape (n, n)
        Symmetric matrix of ZZ couplings (only i<j used).
    w : ndarray, shape (n,)
        Local Z-field coefficients (appears as -sum_i w_i Z_i).
    c : float
        Constant energy shift (set to 0.0 here).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = len(mu)

    # Two-body ZZ couplings
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            W[i, j] = 0.5 * lam * (q * sigma[i, j] + A)
            W[j, i] = W[i, j]  # symmetric

    # Local Z-field coefficients
    w = np.zeros(n, dtype=float)
    for i in range(n):
        row_sum_sigma = sigma[i, :].sum()
        w[i] = 0.5 * lam * ((1.0 - q) * mu[i] + A * (2 * B - n) - q * row_sum_sigma)

    # Constant term c is not needed for QAOA (global phase only)
    c = 0.0

    return W, w, c
