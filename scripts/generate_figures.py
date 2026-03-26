"""
generate_figures.py
===================
Generates all publication-quality figures for the paper
from pre-computed simulation results.

Run AFTER run_simulations.py:
    python scripts/generate_figures.py

Requires: results/ directory populated by run_simulations.py

Outputs all figures to figures/ directory as 300dpi PNGs.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numpy.linalg import norm

from risk_parity import (
    solve_risk_parity, stability_constant,
    two_asset_risk_parity, two_asset_stability_constant_exact, condition_number
)
from covariance import make_covariance, estimation_error

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


STYLE = {
    'figure.figsize': (8, 4.5),
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'font.size': 11,
    'axes.titlesize': 12,
    'font.family': 'serif',
}


def save(name):
    path = os.path.join(FIGURES_DIR, name + '.png')
    plt.savefig(path, dpi=300, bbox_inches='tight')
    print(f'  Saved {name}.png')
    plt.close()


def fig_bound_vs_kappa():
    """Figure: C(Sigma) and empirical error vs kappa."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'sim_kappa_sweep.csv'))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(df['kappa'], df['C'], 'b-o', ms=4, label=r'$C(\Sigma)$ (theoretical)')
        ax.plot(df['kappa'], df['emp_mean'], 'r--s', ms=4, label=r'Empirical $\mathbb{E}[\|\hat{w}-w^*\|]$')
        ax.fill_between(df['kappa'],
                        df['emp_mean'] - df['emp_std'],
                        df['emp_mean'] + df['emp_std'],
                        alpha=0.15, color='red')
        ax.set_xscale('log')
        ax.set_xlabel(r'Condition number $\kappa(\Sigma)$')
        ax.set_ylabel(r'Weight perturbation / stability constant')
        ax.set_title(r'Stability constant $C(\Sigma)$ grows with $\kappa(\Sigma)$')
        ax.legend()
        plt.tight_layout()
    save('fig3_bound_vs_kappa')


def fig_convergence_rate():
    """Figure: convergence rate O(T^{-1/2})."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'sim_bound_verification.csv'))
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.loglog(df['T'], df['emp_mean'], 'bo-', ms=5, label='Empirical mean')
        ax.loglog(df['T'], df['theoretical_bound'], 'r--', lw=2, label='Theoretical bound')
        ref = df['emp_mean'].iloc[0] * np.sqrt(df['T'].iloc[0] / df['T'])
        ax.loglog(df['T'], ref, ':', color='gray', label=r'$O(T^{-1/2})$ reference')
        ax.set_xlabel(r'Sample size $T$')
        ax.set_ylabel(r'$\|\hat{w} - w^*\|_2$')
        ax.set_title(r'Weight perturbation converges at $O(T^{-1/2})$')
        ax.legend()
        plt.tight_layout()
    save('fig4_convergence_rate')


def fig_instability_heatmap():
    """Figure: (kappa x T) heatmap."""
    dispersion = np.load(os.path.join(RESULTS_DIR, 'sim_instability_grid.npy'))
    kappa_grid = np.load(os.path.join(RESULTS_DIR, 'sim_kappa_grid.npy'))
    T_grid = np.load(os.path.join(RESULTS_DIR, 'sim_T_grid.npy'))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(dispersion, aspect='auto', origin='lower', cmap='YlOrRd')
        plt.colorbar(im, ax=ax, label=r'$\mathbb{E}[\|\hat{w} - w^*\|_2]$')
        ax.set_xticks(range(len(T_grid)))
        ax.set_xticklabels(T_grid.astype(int))
        ax.set_yticks(range(0, len(kappa_grid), 2))
        ax.set_yticklabels([f'{k:.0f}' for k in kappa_grid[::2]])
        ax.set_xlabel('Sample size $T$')
        ax.set_ylabel(r'Condition number $\kappa(\Sigma)$')
        ax.set_title('Weight instability: high $\\kappa$, low $T$ is the danger zone')
        plt.tight_layout()
    save('fig5_instability_heatmap')


def fig_two_asset_rho_independence():
    """Figure: C(Sigma) independent of rho for n=2."""
    sigma1, sigma2 = 0.12, 0.20
    rho_values = np.linspace(-0.85, 0.85, 60)
    C_vals = [two_asset_stability_constant_exact(sigma1, sigma2, rho)[0] for rho in rho_values]

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(rho_values, C_vals, 'b-', lw=2)
        ax.set_xlabel(r'Correlation $\rho$')
        ax.set_ylabel(r'$C(\Sigma)$')
        ax.set_title(r'$n=2$: $C(\Sigma)$ is independent of $\rho$')
        ax.set_ylim(0, max(C_vals) * 1.3)
        plt.tight_layout()
    save('fig8_C_rho_independence')


def fig_factor_model():
    """Figure: factor model instability."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'sim_factor_model.csv'))
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        k_labels = [str(k) for k in df['k']]
        ax1.bar(k_labels, df['kappa'], color='steelblue')
        ax1.set_xlabel('Number of factors $k$')
        ax1.set_ylabel(r'$\kappa(\Sigma)$')
        ax1.set_title('Condition number by factor structure')
        ax2.bar(k_labels, df['emp_mean'], color='tomato',
                yerr=df['emp_std'], capsize=4)
        ax2.set_xlabel('Number of factors $k$')
        ax2.set_ylabel(r'Empirical $\mathbb{E}[\|\hat{w}-w^*\|]$')
        ax2.set_title('Weight instability by factor structure')
        plt.suptitle('Factor model: fewer factors → higher instability', fontsize=12)
        plt.tight_layout()
    save('fig7_factor_instability')


if __name__ == '__main__':
    print('Generating all paper figures...')

    missing = []
    for fname in ['sim_kappa_sweep.csv', 'sim_bound_verification.csv',
                  'sim_instability_grid.npy', 'sim_factor_model.csv']:
        if not os.path.exists(os.path.join(RESULTS_DIR, fname)):
            missing.append(fname)

    if missing:
        print(f'\nMissing results files: {missing}')
        print('Run scripts/run_simulations.py first.')
        sys.exit(1)

    fig_bound_vs_kappa()
    fig_convergence_rate()
    fig_instability_heatmap()
    fig_two_asset_rho_independence()
    fig_factor_model()

    print(f'\nAll figures saved to {FIGURES_DIR}')
