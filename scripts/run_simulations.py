"""
run_simulations.py
==================
Standalone script that runs all simulation experiments from Sections 3 and 4
of the paper and saves results to ../results/.

Run from project root:
    python scripts/run_simulations.py

Outputs:
    results/sim_kappa_sweep.csv     -- C(Sigma) vs kappa sweep
    results/sim_bound_verification.csv  -- empirical vs theoretical bound
    results/sim_instability_grid.npy    -- (kappa x T) heatmap data
    results/sim_factor_model.csv     -- factor model results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
from numpy.linalg import norm

from risk_parity import solve_risk_parity, stability_constant, condition_number
from covariance import make_covariance, make_factor_covariance, estimation_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_kappa_sweep(n=6, T=300, N_MC=500, n_kappa=30):
    """Sweep kappa(Sigma) and record C(Sigma) and empirical weight error."""
    print(f'\n[1/4] Kappa sweep (n={n}, T={T}, MC={N_MC})')
    kappa_values = np.logspace(0.3, 3.0, n_kappa)
    rows = []
    for ki, kappa in enumerate(kappa_values):
        Sigma = make_covariance(n, kappa=kappa, seed=ki)
        w_star, _ = solve_risk_parity(Sigma)
        C, H_inv_norm, dG_norm = stability_constant(w_star, Sigma)

        errors = []
        for s in range(N_MC):
            E, E_norm = estimation_error(Sigma, T, seed=s)
            Sigma_hat = Sigma + E
            if np.linalg.eigvalsh(Sigma_hat)[0] > 0:
                w_hat, _ = solve_risk_parity(Sigma_hat)
                errors.append(norm(w_hat - w_star))

        rows.append({
            'kappa': kappa, 'C': C, 'H_inv_norm': H_inv_norm,
            'dG_norm': dG_norm,
            'emp_mean': np.mean(errors), 'emp_std': np.std(errors),
            'emp_max': np.max(errors),
        })
        if (ki + 1) % 5 == 0:
            print(f'  kappa={kappa:.1f}, C={C:.3f}, emp_mean={np.mean(errors):.6f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'sim_kappa_sweep.csv'), index=False)
    print(f'  Saved sim_kappa_sweep.csv')
    return df


def run_bound_verification(n=5, kappa=50.0, N_MC=800):
    """Verify O(T^{-1/2}) convergence rate."""
    print(f'\n[2/4] Bound verification (n={n}, kappa={kappa})')
    T_values = np.array([50, 100, 200, 500, 1000, 2000, 5000])
    Sigma = make_covariance(n, kappa=kappa, seed=1)
    w_star, _ = solve_risk_parity(Sigma)
    C, _, _ = stability_constant(w_star, Sigma)

    rows = []
    for T in T_values:
        errs = []
        for s in range(N_MC):
            E, _ = estimation_error(Sigma, T, seed=s)
            Sigma_hat = Sigma + E
            if np.linalg.eigvalsh(Sigma_hat)[0] > 0:
                w_hat, _ = solve_risk_parity(Sigma_hat)
                errs.append(norm(w_hat - w_star))
        rows.append({
            'T': T,
            'emp_mean': np.mean(errs),
            'emp_std': np.std(errs),
            'theoretical_bound': C * np.sqrt(n / T),
        })
        print(f'  T={T:5d}: emp={np.mean(errs):.6f}, bound={C*np.sqrt(n/T):.6f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'sim_bound_verification.csv'), index=False)
    print('  Saved sim_bound_verification.csv')
    return df


def run_instability_grid(n=8, N_MC=300):
    """(kappa x T) heatmap of mean weight perturbation norm."""
    print(f'\n[3/4] Instability grid (n={n}, MC={N_MC})')
    kappa_grid = np.logspace(0.5, 2.8, 14)
    T_grid = [50, 100, 200, 400, 800, 1600, 3200]
    dispersion = np.zeros((len(kappa_grid), len(T_grid)))

    for i, kappa in enumerate(kappa_grid):
        Sigma = make_covariance(n, kappa=kappa, seed=i)
        w_star, _ = solve_risk_parity(Sigma)
        for j, T in enumerate(T_grid):
            errs = []
            for s in range(N_MC):
                E, _ = estimation_error(Sigma, T, seed=s)
                Sigma_hat = Sigma + E
                if np.linalg.eigvalsh(Sigma_hat)[0] > 0:
                    w_hat, _ = solve_risk_parity(Sigma_hat)
                    errs.append(norm(w_hat - w_star))
            dispersion[i, j] = np.mean(errs) if errs else np.nan
        print(f'  kappa={kappa:.1f} done')

    np.save(os.path.join(RESULTS_DIR, 'sim_instability_grid.npy'), dispersion)
    np.save(os.path.join(RESULTS_DIR, 'sim_kappa_grid.npy'), kappa_grid)
    np.save(os.path.join(RESULTS_DIR, 'sim_T_grid.npy'), np.array(T_grid))
    print('  Saved sim_instability_grid.npy')
    return dispersion


def run_factor_model(n=10, T=250, N_MC=500):
    """Factor model: vary number of factors k, record instability."""
    print(f'\n[4/4] Factor model (n={n}, T={T}, MC={N_MC})')
    k_values = [1, 2, 3, 5, 8]
    rows = []
    for k in k_values:
        Sigma = make_factor_covariance(n, k=k, idio_var=0.05, seed=42)
        w_star, _ = solve_risk_parity(Sigma)
        C, _, _ = stability_constant(w_star, Sigma)
        kappa = condition_number(Sigma)

        errs = []
        for s in range(N_MC):
            E, _ = estimation_error(Sigma, T, seed=s)
            Sigma_hat = Sigma + E
            if np.linalg.eigvalsh(Sigma_hat)[0] > 0:
                w_hat, _ = solve_risk_parity(Sigma_hat)
                errs.append(norm(w_hat - w_star))

        rows.append({'k': k, 'kappa': kappa, 'C': C,
                     'emp_mean': np.mean(errs), 'emp_std': np.std(errs)})
        print(f'  k={k}: kappa={kappa:.1f}, C={C:.4f}, emp_mean={np.mean(errs):.6f}')

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, 'sim_factor_model.csv'), index=False)
    print('  Saved sim_factor_model.csv')
    return df


if __name__ == '__main__':
    print('Running all simulation experiments...')
    print('(This takes ~10-15 minutes depending on hardware)')
    print('=' * 60)

    run_kappa_sweep()
    run_bound_verification()
    run_instability_grid()
    run_factor_model()

    print('\n' + '=' * 60)
    print('All simulations complete. Results in ./results/')
