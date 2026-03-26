"""
plotting.py
===========
Shared plotting utilities for all notebooks and scripts.
All figures saved to ../figures/ by default.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

STYLE = {
    "figure.figsize": (9, 5),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
}


def savefig(name, dpi=150):
    path = os.path.join(FIGURES_DIR, name if name.endswith(".png") else name + ".png")
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


def plot_bound_vs_kappa(kappas, C_values, empirical_errors=None, title="Stability constant vs condition number"):
    """Plot C(Sigma) as a function of kappa(Sigma)."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.plot(kappas, C_values, "b-o", markersize=4, label=r"$C(\Sigma)$ (theoretical)")
        if empirical_errors is not None:
            ax.plot(kappas, empirical_errors, "r--s", markersize=4, label=r"Empirical $\|\delta w\|$")
        ax.set_xlabel(r"Condition number $\kappa(\Sigma)$")
        ax.set_ylabel(r"Stability constant / weight perturbation")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
    return fig


def plot_weight_dispersion(kappas, dispersion_matrix, n_label="n"):
    """
    Plot weight dispersion (std of delta_w across simulations) as a heatmap
    over kappa and sample size T.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(dispersion_matrix, aspect="auto", origin="lower", cmap="YlOrRd")
        plt.colorbar(im, ax=ax, label=r"$\|\hat{w} - w^*\|_2$")
        ax.set_title(f"Weight perturbation norm ({n_label} assets)")
        ax.set_xlabel(r"Sample size $T$")
        ax.set_ylabel(r"Condition number $\kappa(\Sigma)$")
        plt.tight_layout()
    return fig


def plot_risk_contributions(w, Sigma, labels=None, title="Risk contributions"):
    """Bar chart of risk contributions at a given weight vector."""
    from risk_parity import risk_contributions
    with plt.rc_context(STYLE):
        rc = risk_contributions(w, Sigma)
        n = len(w)
        if labels is None:
            labels = [f"Asset {i+1}" for i in range(n)]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(labels, w, color="steelblue")
        axes[0].set_title("Portfolio weights")
        axes[0].set_ylabel("Weight")
        axes[1].bar(labels, rc, color="tomato")
        axes[1].axhline(rc.mean(), color="black", linestyle="--", label=f"Equal = {rc.mean():.4f}")
        axes[1].set_title(title)
        axes[1].set_ylabel("Risk contribution")
        axes[1].legend()
        plt.tight_layout()
    return fig


def plot_two_asset_sensitivity(sigma1_range, C_values, rho_values, title="Two-asset stability constant"):
    """
    Plot exact stability constant for two-asset case across sigma1 values
    for multiple rho values.
    """
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        for i, rho in enumerate(rho_values):
            ax.plot(sigma1_range, C_values[i], label=rf"$\rho = {rho:.1f}$")
        ax.set_xlabel(r"$\sigma_1$ (asset 1 volatility)")
        ax.set_ylabel(r"$C(\Sigma)$")
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
    return fig


def plot_rolling_kappa_and_turnover(dates, kappas, turnovers, title="Rolling kappa and portfolio turnover"):
    """Two-panel plot: rolling condition number and rolling portfolio turnover."""
    with plt.rc_context(STYLE):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
        ax1.plot(dates, kappas, color="navy", linewidth=1)
        ax1.set_ylabel(r"$\kappa(\Sigma)$")
        ax1.set_title(title)
        ax2.plot(dates, turnovers, color="firebrick", linewidth=1)
        ax2.set_ylabel("Portfolio turnover")
        ax2.set_xlabel("Date")
        plt.tight_layout()
    return fig


def plot_bound_verification(T_values, empirical_mean, theoretical_bound, n, kappa):
    """Compare empirical ||delta_w|| to theoretical bound as T varies."""
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots()
        ax.loglog(T_values, empirical_mean, "bo-", markersize=5, label="Empirical mean")
        ax.loglog(T_values, theoretical_bound, "r--", linewidth=2, label="Theoretical bound")
        ref = empirical_mean[0] * np.sqrt(T_values[0] / T_values)
        ax.loglog(T_values, ref, "gray", linestyle=":", linewidth=1, label=r"$O(T^{-1/2})$ reference")
        ax.set_xlabel(r"Sample size $T$")
        ax.set_ylabel(r"$\|\hat{w} - w^*\|_2$")
        ax.set_title(f"Bound verification: n={n}, κ={kappa:.0f}")
        ax.legend()
        plt.tight_layout()
    return fig
