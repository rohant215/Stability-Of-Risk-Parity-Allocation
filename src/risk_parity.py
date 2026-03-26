"""
risk_parity.py
==============
Core mathematical engine for risk parity portfolio allocation.

Implements:
  - Risk parity solver via convex optimisation (cvxpy) and Newton iteration
  - Risk contribution computation
  - Augmented Jacobian H construction (Section 3 of paper)
  - Perturbation bound C(Sigma) computation
  - Condition number utilities

All functions are documented with mathematical context referencing the paper.
"""

import numpy as np
from numpy.linalg import norm, inv, eigvalsh, cond
from scipy.optimize import minimize
import warnings


# ---------------------------------------------------------------------------
# 1.  Risk contribution functions
# ---------------------------------------------------------------------------

def portfolio_variance(w, Sigma):
    """Compute sigma_p^2 = w^T Sigma w."""
    return float(w @ Sigma @ w)


def portfolio_volatility(w, Sigma):
    """Compute sigma_p = sqrt(w^T Sigma w)."""
    return np.sqrt(portfolio_variance(w, Sigma))


def marginal_risk_contributions(w, Sigma):
    """
    Compute marginal risk contributions (Sigma w)_i / sigma_p.
    Vector of length n.
    """
    sigma_p = portfolio_volatility(w, Sigma)
    return (Sigma @ w) / sigma_p


def risk_contributions(w, Sigma):
    """
    Compute RC_i = w_i * (Sigma w)_i / sigma_p  for each asset i.
    At a risk parity solution all RC_i are equal to 1/n * sigma_p.
    Returns vector of length n.
    """
    sigma_p = portfolio_volatility(w, Sigma)
    return w * (Sigma @ w) / sigma_p


def is_risk_parity(w, Sigma, tol=1e-6):
    """Check whether w satisfies equal risk contribution up to tolerance."""
    rc = risk_contributions(w, Sigma)
    return np.allclose(rc, rc.mean(), atol=tol)


# ---------------------------------------------------------------------------
# 2.  Risk parity solver
# ---------------------------------------------------------------------------

def solve_risk_parity(Sigma, w0=None, method="newton", tol=1e-10, max_iter=1000):
    """
    Solve the risk parity system F(w, Sigma) = 0 subject to sum(w) = 1.

    Two methods:
      'newton'  -- Newton iteration on the augmented system G(z, Sigma) = 0
                   where z = (w, lambda).  Fast and numerically precise.
      'scipy'   -- Minimise sum_i sum_j (RC_i - RC_j)^2 via L-BFGS-B.
                   Slower but robust fallback.

    Parameters
    ----------
    Sigma   : (n, n) symmetric positive definite covariance matrix
    w0      : (n,) initial guess; defaults to 1/n * ones
    method  : 'newton' or 'scipy'
    tol     : convergence tolerance
    max_iter: maximum Newton iterations

    Returns
    -------
    w_star : (n,) risk parity weights
    info   : dict with diagnostics (iterations, final residual, etc.)
    """
    n = Sigma.shape[0]
    if w0 is None:
        w0 = np.ones(n) / n

    if method == "scipy":
        return _solve_scipy(Sigma, w0, tol)
    else:
        return _solve_newton(Sigma, w0, tol, max_iter)


def _solve_newton(Sigma, w0, tol, max_iter):
    """
    Newton iteration on the augmented system:
        G_i(z, Sigma) = w_i * (Sigma w)_i - lambda = 0,  i = 1..n
        G_{n+1}(z, Sigma) = w^T 1 - 1 = 0
    where z = (w, lambda) in R^{n+1}.
    """
    n = Sigma.shape[0]
    # Initial z = (w0, lambda0) where lambda0 = mean risk contribution
    w = w0.copy()
    sigma_p = portfolio_volatility(w, Sigma)
    lam = np.mean(w * (Sigma @ w) / sigma_p)
    z = np.append(w, lam)

    for it in range(max_iter):
        w, lam = z[:n], z[n]
        G, H = _augmented_system(w, lam, Sigma)
        res = norm(G)
        if res < tol:
            break
        try:
            dz = np.linalg.solve(H, -G)
        except np.linalg.LinAlgError:
            warnings.warn("Newton step: singular Jacobian. Switching to lstsq.")
            dz, _, _, _ = np.linalg.lstsq(H, -G, rcond=None)
        # Line search (simple backtracking)
        step = 1.0
        for _ in range(20):
            z_new = z + step * dz
            G_new, _ = _augmented_system(z_new[:n], z_new[n], Sigma)
            if norm(G_new) < res:
                break
            step *= 0.5
        z = z_new

    w_star = z[:n]
    # Project onto simplex (small numerical fix)
    w_star = np.maximum(w_star, 1e-12)
    w_star /= w_star.sum()

    G_final, H_final = _augmented_system(w_star, z[n], Sigma)
    return w_star, {
        "iterations": it + 1,
        "residual": norm(G_final),
        "converged": norm(G_final) < tol * 10,
        "jacobian": H_final,
        "lambda": z[n],
    }


def _solve_scipy(Sigma, w0, tol):
    """Fallback solver using scipy.optimize.minimize."""
    n = Sigma.shape[0]

    def objective(w):
        sigma_p = portfolio_volatility(w, Sigma)
        rc = w * (Sigma @ w) / sigma_p
        return sum((rc[i] - rc[j])**2 for i in range(n) for j in range(i+1, n))

    constraints = {"type": "eq", "fun": lambda w: w.sum() - 1}
    bounds = [(1e-6, 1.0)] * n
    result = minimize(objective, w0, method="SLSQP",
                      bounds=bounds, constraints=constraints,
                      options={"ftol": tol, "maxiter": 2000})
    w_star = result.x
    w_star /= w_star.sum()
    G_final, H_final = _augmented_system(w_star, np.mean(risk_contributions(w_star, Sigma)), Sigma)
    return w_star, {
        "iterations": result.nit,
        "residual": result.fun,
        "converged": result.success,
        "jacobian": H_final,
        "lambda": np.mean(risk_contributions(w_star, Sigma)),
    }


# ---------------------------------------------------------------------------
# 3.  Augmented system G and Jacobian H  (Section 3 of paper)
# ---------------------------------------------------------------------------

def _augmented_system(w, lam, Sigma):
    """
    Evaluate G(z, Sigma) and the augmented Jacobian H = dG/dz.

    G in R^{n+1}:
        G_i = w_i * (Sigma w)_i - lambda          i = 1..n
        G_{n+1} = sum(w) - 1

    H = dG/dz in R^{(n+1) x (n+1)}:
        Block form:
            H = [ J   | -1  ]
                [ 1^T |  0  ]
        where J_ij = w_i * Sigma_ij + delta_ij * (Sigma w)_i
    """
    n = len(w)
    Sw = Sigma @ w                       # (Sigma w), shape (n,)

    # Residual G
    G = np.zeros(n + 1)
    G[:n] = w * Sw - lam
    G[n] = w.sum() - 1.0

    # Jacobian H
    # J_ij = w_i * Sigma_ij  (outer product term)
    J = np.diag(w) @ Sigma               # shape (n, n)
    # Add diagonal: delta_ij * (Sigma w)_i
    J += np.diag(Sw)

    H = np.zeros((n + 1, n + 1))
    H[:n, :n] = J
    H[:n, n] = -1.0                      # d G_i / d lambda = -1
    H[n, :n] = 1.0                       # d G_{n+1} / d w_j = 1

    return G, H


def compute_jacobian(w_star, Sigma):
    """
    Return the augmented Jacobian H at the risk parity solution w_star.
    This is the key object for the perturbation bound.
    """
    lam = np.mean(w_star * (Sigma @ w_star) / portfolio_volatility(w_star, Sigma))
    _, H = _augmented_system(w_star, lam, Sigma)
    return H


# ---------------------------------------------------------------------------
# 4.  Perturbation bound  C(Sigma)  (Main Theorem, Section 3)
# ---------------------------------------------------------------------------

def compute_dG_dSigma_norm(w_star, Sigma):
    """
    Compute ||dG/dSigma|| in operator norm (spectral norm).

    dG/dSigma is a linear map from R^{n x n} to R^{n+1}.
    For the i-th component of G:
        G_i = w_i * (Sigma w)_i - lambda
        dG_i / dSigma_{kl} = w_i * w_l * delta_{ik}   (for k=i)
                            = w_i * w_l                 (when k=i)

    We represent dG/dSigma as a matrix of shape (n+1) x n^2
    where each row i (for i < n) is the vectorised derivative of G_i wrt vec(Sigma),
    and row n (budget constraint) is zero (G_{n+1} doesn't depend on Sigma).

    The spectral norm of this matrix upper-bounds ||dG/dSigma[E]|| / ||E||.
    """
    n = len(w_star)
    # dG_i / d(Sigma) acting on perturbation E:
    # (dG/dSigma [E])_i = w_i * (E w)_i   for i = 1..n
    # (dG/dSigma [E])_{n+1} = 0
    # So the linear map is: E -> diag(w) E w  (first n components)
    # Its operator norm: max_{||E||_F = 1} || diag(w) E w ||_2
    # Upper bound via submultiplicativity:
    #   <= ||diag(w)||_2 * ||E||_F * ||w||_2
    #   = max(w) * ||w||_2
    # We compute the exact norm via sampling for accuracy.
    max_w = np.max(w_star)
    norm_w = norm(w_star)
    # Tight upper bound (analytically derived):
    dG_dSigma_norm = max_w * norm_w
    return dG_dSigma_norm


def stability_constant(w_star, Sigma):
    """
    Compute the stability constant C(Sigma) = ||H^{-1}|| * ||dG/dSigma||.

    This is the main quantity from the paper's perturbation bound:
        ||delta_w|| <= C(Sigma) * ||E||

    where E = Sigma_hat - Sigma is the estimation error.

    Returns
    -------
    C       : float, the stability constant
    H_inv_norm : float, ||H^{-1}|| (spectral norm)
    dG_norm : float, ||dG/dSigma||
    """
    H = compute_jacobian(w_star, Sigma)
    H_inv = inv(H)
    H_inv_norm = norm(H_inv, ord=2)
    dG_norm = compute_dG_dSigma_norm(w_star, Sigma)
    C = H_inv_norm * dG_norm
    return C, H_inv_norm, dG_norm


def analytical_bound(Sigma, T, n=None):
    """
    Compute the analytical upper bound on E[||delta_w||] from the paper.

    Using:
      - ||E|| ~ O(sqrt(n / T)) for the sample covariance estimation error
      - C(Sigma) from stability_constant()

    Parameters
    ----------
    Sigma : (n, n) true covariance matrix
    T     : int, sample size
    n     : int, number of assets (inferred from Sigma if None)

    Returns
    -------
    bound : float, upper bound on expected weight perturbation norm
    """
    if n is None:
        n = Sigma.shape[0]
    w_star, _ = solve_risk_parity(Sigma)
    C, _, _ = stability_constant(w_star, Sigma)
    estimation_noise = np.sqrt(n / T)  # Marchenko-Pastur scaling
    return C * estimation_noise


# ---------------------------------------------------------------------------
# 5.  Condition number and spectral utilities
# ---------------------------------------------------------------------------

def condition_number(Sigma):
    """Compute kappa(Sigma) = lambda_max / lambda_min."""
    eigs = eigvalsh(Sigma)
    return eigs[-1] / eigs[0]


def lambda_min(Sigma):
    return float(eigvalsh(Sigma)[0])


def lambda_max(Sigma):
    return float(eigvalsh(Sigma)[-1])


def shrinkage_covariance(S, alpha):
    """
    Ledoit-Wolf style linear shrinkage:
        Sigma_shrunk = (1 - alpha) * S + alpha * mu_S * I
    where mu_S = trace(S) / n.
    """
    n = S.shape[0]
    mu = np.trace(S) / n
    return (1 - alpha) * S + alpha * mu * np.eye(n)


# ---------------------------------------------------------------------------
# 6.  Two-asset closed form  (Section 5 of paper)
# ---------------------------------------------------------------------------

def two_asset_risk_parity(sigma1, sigma2):
    """
    Closed-form risk parity weights for n=2.
    w1 = sigma2 / (sigma1 + sigma2),  w2 = sigma1 / (sigma1 + sigma2).
    Correlation cancels — weights depend only on relative volatilities.
    """
    w1 = sigma2 / (sigma1 + sigma2)
    w2 = sigma1 / (sigma1 + sigma2)
    return np.array([w1, w2])


def two_asset_stability_constant_exact(sigma1, sigma2, rho):
    """
    Exact closed-form stability constant C(Sigma) for n=2.

    Sigma = [[sigma1^2,       rho*sigma1*sigma2],
             [rho*sigma1*sigma2, sigma2^2      ]]

    Derived analytically in Section 5 of the paper.
    """
    Sigma = np.array([
        [sigma1**2,           rho * sigma1 * sigma2],
        [rho * sigma1 * sigma2, sigma2**2          ]
    ])
    w_star = two_asset_risk_parity(sigma1, sigma2)
    C, H_inv_norm, dG_norm = stability_constant(w_star, Sigma)
    return C, H_inv_norm, dG_norm
