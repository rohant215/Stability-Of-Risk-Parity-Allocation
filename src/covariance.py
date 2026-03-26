"""
covariance.py
=============
Utilities for generating and estimating covariance matrices.

Covers:
  - Random SPD matrix generation with controlled condition number
  - Sample covariance estimation
  - Ledoit-Wolf shrinkage
  - Factor model covariance structures (for realistic simulations)
"""

import numpy as np
from numpy.linalg import eigvalsh, norm


# ---------------------------------------------------------------------------
# 1.  Random SPD matrices with controlled kappa
# ---------------------------------------------------------------------------

def make_covariance(n, kappa=10.0, seed=None):
    """
    Generate a random n x n SPD covariance matrix with condition number ~ kappa.

    Method: random orthogonal Q, eigenvalues spaced log-uniformly in [1, kappa].

    Parameters
    ----------
    n     : int, number of assets
    kappa : float, target condition number lambda_max / lambda_min
    seed  : int or None

    Returns
    -------
    Sigma : (n, n) SPD matrix
    """
    rng = np.random.default_rng(seed)
    # Random orthogonal matrix via QR of random Gaussian
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    # Log-uniform eigenvalues in [1, kappa]
    eigs = np.exp(np.linspace(0, np.log(kappa), n))
    Sigma = Q @ np.diag(eigs) @ Q.T
    # Symmetrise to remove floating-point asymmetry
    Sigma = (Sigma + Sigma.T) / 2
    return Sigma


def make_factor_covariance(n, k=3, idio_var=0.1, seed=None):
    """
    Generate a factor-model covariance: Sigma = B B^T + D
    where B is n x k factor loadings and D is diagonal idiosyncratic variance.

    This produces a near-singular matrix when k << n, realistic for equities.

    Parameters
    ----------
    n        : number of assets
    k        : number of factors
    idio_var : idiosyncratic variance level (scalar)
    seed     : random seed
    """
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((n, k)) / np.sqrt(k)
    D = np.diag(rng.uniform(idio_var * 0.5, idio_var * 1.5, n))
    Sigma = B @ B.T + D
    Sigma = (Sigma + Sigma.T) / 2
    return Sigma


def make_toeplitz_covariance(n, rho=0.5):
    """
    Generate Toeplitz covariance: Sigma_ij = rho^|i-j|.
    Condition number grows with n and rho.
    """
    idx = np.arange(n)
    Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
    return Sigma


def make_block_covariance(n, n_blocks=3, within_rho=0.8, between_rho=0.1):
    """
    Block-diagonal-ish covariance with n_blocks asset groups.
    High within-block correlation, low between-block correlation.
    """
    block_size = n // n_blocks
    Sigma = np.full((n, n), between_rho)
    for b in range(n_blocks):
        start = b * block_size
        end = start + block_size if b < n_blocks - 1 else n
        Sigma[start:end, start:end] = within_rho
    np.fill_diagonal(Sigma, 1.0)
    return Sigma


# ---------------------------------------------------------------------------
# 2.  Sample covariance estimation
# ---------------------------------------------------------------------------

def sample_covariance(returns):
    """
    Compute unbiased sample covariance from a (T x n) returns matrix.
    S = 1/(T-1) * sum_t (r_t - r_bar)(r_t - r_bar)^T
    """
    T, n = returns.shape
    r_bar = returns.mean(axis=0)
    demeaned = returns - r_bar
    S = (demeaned.T @ demeaned) / (T - 1)
    return S


def simulate_returns(Sigma, T, seed=None):
    """
    Draw T observations from N(0, Sigma).

    Parameters
    ----------
    Sigma : (n, n) true covariance
    T     : number of time periods
    seed  : random seed

    Returns
    -------
    returns : (T, n) matrix of simulated returns
    """
    rng = np.random.default_rng(seed)
    n = Sigma.shape[0]
    L = np.linalg.cholesky(Sigma)
    Z = rng.standard_normal((T, n))
    return Z @ L.T


def estimation_error(Sigma, T, seed=None):
    """
    Simulate estimation error E = S_hat - Sigma for a given sample size T.
    Returns E and its Frobenius norm.
    """
    returns = simulate_returns(Sigma, T, seed=seed)
    S_hat = sample_covariance(returns)
    E = S_hat - Sigma
    return E, norm(E, 'fro')


# ---------------------------------------------------------------------------
# 3.  Shrinkage estimators
# ---------------------------------------------------------------------------

def ledoit_wolf_analytical(returns):
    """
    Analytical Ledoit-Wolf shrinkage towards scaled identity.
    Oracle-free: shrinkage intensity alpha is estimated from data.

    Reference: Ledoit & Wolf (2004), Journal of Multivariate Analysis.

    Returns
    -------
    Sigma_lw : (n, n) shrinkage estimator
    alpha    : float, estimated shrinkage intensity
    """
    T, n = returns.shape
    S = sample_covariance(returns)
    mu = np.trace(S) / n

    # Estimate alpha analytically (simplified Oracle approximating shrinkage)
    delta_sq = norm(S - mu * np.eye(n), 'fro') ** 2
    # Estimate noise in off-diagonal elements
    R = returns - returns.mean(axis=0)
    phi = 0.0
    for t in range(T):
        r = R[t]
        outer = np.outer(r, r)
        phi += norm(outer - S, 'fro') ** 2
    phi /= T ** 2

    alpha = min(phi / delta_sq, 1.0)
    Sigma_lw = (1 - alpha) * S + alpha * mu * np.eye(n)
    return Sigma_lw, alpha


# ---------------------------------------------------------------------------
# 4.  Rolling covariance (for empirical section)
# ---------------------------------------------------------------------------

def rolling_covariance(returns, window):
    """
    Compute rolling sample covariance matrices from a (T x n) returns matrix.

    Parameters
    ----------
    returns : (T, n) array of asset returns
    window  : int, rolling window length

    Returns
    -------
    Sigmas : list of (n, n) covariance matrices, length T - window + 1
    dates  : list of indices corresponding to the last observation in each window
    """
    T, n = returns.shape
    Sigmas = []
    indices = []
    for t in range(window - 1, T):
        window_returns = returns[t - window + 1: t + 1]
        Sigmas.append(sample_covariance(window_returns))
        indices.append(t)
    return Sigmas, indices
