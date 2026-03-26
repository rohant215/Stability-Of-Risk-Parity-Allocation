# Risk Parity Stability — Code Repository

**On the Stability of Risk Parity Allocations under Covariance Perturbations**
Rohan Thosar, 2026

---

## Project Structure

```
risk_parity/
├── src/
│   ├── risk_parity.py     Core solver, Jacobian, perturbation bound C(Sigma)
│   ├── covariance.py      SPD matrix generation, sample estimation, shrinkage
│   └── plotting.py        Shared figure utilities
│
├── notebooks/
│   ├── 01_mathematical_setup.ipynb      Section 2: augmented system, IFT setup
│   ├── 02_main_theorem.ipynb            Section 3: perturbation bound derivation
│   ├── 03_instability_regimes.ipynb     Section 4: kappa/T heatmaps, T* threshold
│   ├── 04_two_asset_verification.ipynb  Section 5: closed-form n=2 case
│   └── 05_empirical_etf.ipynb           Section 7: rolling ETF analysis
│
├── scripts/
│   ├── run_simulations.py   Run all Monte Carlo experiments -> results/
│   └── generate_figures.py  Generate publication figures -> figures/
│
├── results/               (created by run_simulations.py)
├── figures/               (created by scripts)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A: Notebooks (recommended for exploration)

```bash
cd risk_parity
jupyter notebook
```

Open notebooks in order: 01 → 02 → 03 → 04 → 05.
Each notebook is self-contained and runs independently.

### Option B: Scripts (for full paper reproduction)

```bash
# Step 1: Run all Monte Carlo simulations (~10-15 min)
python scripts/run_simulations.py

# Step 2: Generate all paper figures
python scripts/generate_figures.py
```

Figures are saved to `figures/` as 300dpi PNGs ready for the paper.

---

## Key Functions (src/risk_parity.py)

| Function | Description |
|---|---|
| `solve_risk_parity(Sigma)` | Newton solver for w* |
| `compute_jacobian(w_star, Sigma)` | Augmented Jacobian H |
| `stability_constant(w_star, Sigma)` | C(Sigma) = \|\|H^{-1}\|\| * \|\|dG/dSigma\|\| |
| `two_asset_risk_parity(s1, s2)` | Closed-form n=2 weights |
| `analytical_bound(Sigma, T)` | Upper bound on E[\|\|delta_w\|\|] |

---

## Mathematical Summary

The main result is:

> **Theorem 1.** At the risk parity solution w*(Sigma), for perturbation E = Sigma_hat - Sigma:
>
>     ||w_hat - w*|| <= C(Sigma) * ||E|| + O(||E||^2)
>
> where C(Sigma) = ||H^{-1}|| * ||dG/dSigma||, and H is the (n+1)x(n+1) augmented Jacobian.

The instability regime is characterised by large kappa(Sigma) (ill-conditioned covariance)
and large n/T ratio (high dimension relative to sample size).
