# Perturbation Bounds for Risk Parity Allocations under Covariance Estimation Error

**Author:** Rohan Thosar
**Date:** February 2026
**Type:** Master's Research Project — Applied Mathematics / Quantitative Finance

---

## Overview

Risk parity is a portfolio allocation rule in which each asset contributes equally to total portfolio risk. Unlike mean–variance optimisation, it requires no expected return estimates and depends solely on the covariance matrix of asset returns. In practice, that covariance matrix must be *estimated* from finite samples, and this estimation error propagates nonlinearly into allocation decisions.

This project provides a rigorous mathematical characterisation of that propagation. Using tools from nonlinear analysis and matrix conditioning theory, it derives an explicit perturbation bound on the weight vector and identifies the structural regimes in which small estimation errors are amplified into large, costly allocation changes.

**The central result is:**

> For a risk parity solution w\* = f(Σ) and a perturbed covariance matrix Σ̂ = Σ + E,
>
> ‖ŵ − w\*‖ ≤ C(Σ) · ‖E‖ + O(‖E‖²)
>
> where **C(Σ) = ‖H⁻¹‖ · ‖∂G/∂Σ‖** is an explicit stability constant, and H is the (n+1)×(n+1) augmented Jacobian of the risk parity system.

This bound is verified analytically in the two-asset case, numerically across simulated covariance structures, and empirically on 17 years of ETF data (2007–2024).

---

## Repository Structure

```
risk_parity/
├── src/
│   ├── risk_parity.py          
│   ├── covariance.py           
│   └── plotting.py             
│
├── notebooks/
│   ├── 01_mathematical_setup.ipynb
│   ├── 02_main_theorem.ipynb             
│   ├── 03_instability_regimes.ipynb
│   ├── 04_two_asset_verification.ipynb
│   └── 05_empirical_etf.ipynb            
│
├── scripts/
│   ├── run_simulations.py       
│   └── generate_figures.py      
│
├── results/                     
├── figures/                     
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

To run the empirical notebook, `yfinance` is also required:

```bash
pip install yfinance
```

To reproduce all results from scratch:

```bash
# Step 1: run all simulations (~10-15 minutes)
python scripts/run_simulations.py

# Step 2: generate all paper figures
python scripts/generate_figures.py
```

---

## Mathematical Framework

### Risk Parity as an Implicit System

Given weights w ∈ ℝⁿ and SPD covariance matrix Σ, portfolio volatility is σ_p = √(wᵀΣw). The risk contribution of asset i is:

```
RC_i(w, Σ) = w_i · (Σw)_i / σ_p
```

Risk parity requires RC₁ = RC₂ = ··· = RC_n together with Σᵢwᵢ = 1. Introducing a Lagrange multiplier λ yields the augmented system G(z, Σ) = 0 where z = (w, λ) ∈ ℝⁿ⁺¹:

```
G_i(z, Σ) = w_i(Σw)_i − λ = 0,   i = 1,…,n
G_{n+1}(z, Σ) = 1ᵀw − 1 = 0
```

At a solution, λ equals the common risk contribution — it has a direct financial interpretation as the equal risk budget per asset.

### The Augmented Jacobian

The (n+1)×(n+1) Jacobian H = ∂G/∂z takes the block form:

```
H = [ J   | −1 ]
    [ 1ᵀ  |  0 ]
```

where J_ij = w_i · Σ_ij + δ_ij · (Σw)_i. Provided H is nonsingular at the solution, the Implicit Function Theorem guarantees that w\* depends smoothly on Σ, and the first-order perturbation satisfies:

```
δz ≈ −H⁻¹ · (∂G/∂Σ)[E]
```

### The Stability Constant C(Σ)

Taking norms yields the main bound ‖δw‖ ≤ C(Σ) · ‖E‖, where:

```
C(Σ) = ‖H⁻¹‖ · ‖∂G/∂Σ‖
```

- **‖H⁻¹‖** is the spectral norm of the inverse Jacobian — large when H is ill-conditioned
- **‖∂G/∂Σ‖** is bounded above by max(w\*) · ‖w\*‖₂

Since estimation error scales as ‖E‖ ~ O(√(n/T)) by the Marchenko-Pastur law, the bound implies:

```
E[‖ŵ − w*‖] ≤ C(Σ) · √(n/T)
```

The minimum sample size to achieve stability tolerance ε is therefore:

```
T*(ε) = ⌈n · C(Σ)² / ε²⌉
```

---

## Results

### Figure 1 — Solver Verification: Equal Risk Contributions

![Figure 1](figures/fig1_risk_contributions.png)

The Newton solver converges to machine precision in under 20 iterations for all tested covariance matrices. The right panel confirms that all four assets achieve exactly equal risk contributions at w\*, each contributing 0.2622 to total portfolio risk. The unequal portfolio weights in the left panel (ranging from 0.16 to 0.41) demonstrate that risk parity is not equal weighting — assets with higher volatility receive lower allocation so that their *risk* contribution matches that of lower-volatility assets.

---

### Figure 2 — Two-Asset Correlation Independence

![Figure 2](figures/fig2_two_asset_correlation_independence.png)

For n = 2, the closed-form solution is w₁\* = σ₂/(σ₁+σ₂), w₂\* = σ₁/(σ₁+σ₂). Correlation ρ cancels entirely — the two-asset RP portfolio is pure inverse-volatility weighting. This figure verifies that result numerically: w₁\* = 0.6250 exactly across all ρ ∈ [−0.85, 0.85], with the numerical solver indistinguishable from the closed form at every point.

---

### Figure 3 — Main Theorem: Stability Constant vs Condition Number

![Figure 3](figures/fig3_bound_vs_kappa.png)

This figure plots the theoretical stability constant C(Σ) (blue, solid) alongside empirically measured ‖δw‖ (red, dashed) for 30 covariance matrices with κ(Σ) ranging from 2 to 1000 (n = 6, T = 300, 500 Monte Carlo draws each). The theoretical bound consistently lies above the empirical mean, confirming validity with zero violations. The two prominent spikes in C(Σ) — at κ ≈ 150 and κ ≈ 800 — correspond to random draws where the Jacobian H was particularly ill-conditioned. Crucially, the empirical errors do not spike at the same points, indicating the bound is conservative rather than sharp on average.

---

### Figure 4 — Convergence Rate: O(T⁻¹/²) Verification

![Figure 4](figures/fig4_convergence_rate.png)

On a log-log scale (n = 5, κ = 50, 800 Monte Carlo draws), the empirical mean ‖ŵ − w\*‖ runs parallel to the O(T⁻¹/²) reference across T ∈ {50, 100, 200, 500, 1000, 2000, 5000}. The theoretical bound remains a tight upper envelope throughout.

| Sample size T | Empirical mean | Theoretical bound | Ratio |
|---|---|---|---|
| 50 | 0.04647 | 0.05503 | 0.84 |
| 100 | 0.02716 | 0.03891 | 0.70 |
| 200 | 0.01826 | 0.02751 | 0.66 |
| 500 | 0.01168 | 0.01740 | 0.67 |
| 1000 | 0.00816 | 0.01230 | 0.66 |
| 2000 | 0.00580 | 0.00870 | 0.67 |
| 5000 | 0.00375 | 0.00550 | 0.68 |

The ratio of empirical mean to theoretical bound stabilises at approximately 0.67, meaning the bound is tight to within one third — a strong result for a general-purpose analytical bound derived purely from matrix norms.

---

### Figure 5 — Instability Heatmap: The Danger Zone

![Figure 5](figures/fig5_instability_heatmap.png)

This is the central diagnostic figure of the project. It shows E[‖ŵ − w\*‖] as a heatmap over a 14×7 grid of (κ(Σ), T) values for n = 8 assets, with 300 Monte Carlo draws per cell. κ(Σ) ranges from 3 to 631; T ranges from 50 to 3200.

Key observations:

- The **danger zone** (dark red, top-left) corresponds to high κ combined with small T. At κ ≈ 186, T = 50, expected weight error reaches 0.26 — a portfolio that is 26% wrong on average relative to the true risk parity allocation
- Increasing T from 50 to 3200 at κ = 186 reduces the error to approximately 0.05, but does not resolve it — more data alone cannot rescue a badly conditioned covariance
- At κ ≈ 3 (well-conditioned), even T = 50 produces errors below 0.03, confirming that **conditioning dominates sample size** as the primary stability driver
- The boundary between stable (error < 0.05) and unstable (error > 0.15) regimes runs diagonally from approximately (κ = 16, T = 50) to (κ = 420, T = 400)
- Grid min: 0.0058, Grid max: 0.2791

---

### Figure 6 — Minimum Sample Size T\* for Stable Estimation

![Figure 6](figures/fig6_min_sample_size.png)

This figure derives the minimum sample size T\*(ε) required to guarantee E[‖δw‖] < ε = 0.02 for n ∈ {4, 8, 16}. For n = 4 (blue), T\* grows broadly with κ, reaching ~20,000 at κ = 1000 — implying that a portfolio with a highly ill-conditioned covariance structure would need decades of daily data to achieve 2% weight stability. The violent spikes (n = 16, κ ≈ 170, T\* reaching 10⁷) reflect catastrophic near-degenerate covariance draws where the Jacobian becomes nearly singular. The non-monotone behaviour across all three curves confirms that C(Σ) — and therefore T\* — is sensitive to eigenstructure beyond κ alone.

---

### Figure 7 — Factor Model Instability

![Figure 7](figures/fig7_factor_instability.png)

This figure tests the framework on factor-structured covariance matrices (n = 10, T = 250 — a realistic equity portfolio setting). As the number of factors k decreases from 8 to 1, κ(Σ) rises from 61 to 318.

| Factors k | κ(Σ) | C(Σ) | Empirical ‖δw‖ | Empirical std |
|---|---|---|---|---|
| 1 | 317.8 | 41.0 | 0.158 | 0.200 |
| 2 | 174.2 | 7.4 | 0.280 | 0.160 |
| 3 | 119.8 | 4.9 | 0.062 | 0.073 |
| 5 | 83.2 | 13.1 | 0.306 | 0.085 |
| 8 | 61.4 | 6.6 | 0.206 | 0.156 |

The relationship between k and empirical instability is strikingly non-monotone: k = 3 produces the lowest instability (0.062) despite being the third most ill-conditioned, while k = 5 produces the highest instability (0.306) at a moderate κ. This demonstrates that κ alone is an incomplete predictor — the eigenstructure's interaction with the risk parity solution (captured in C(Σ)) matters materially.

---

### Figure 8 — C(Σ) Independence from ρ for n = 2

![Figure 8](figures/fig8_C_rho_independence.png)

For the two-asset case, this figure shows C(Σ) as ρ varies from −0.85 to 0.85. The y-axis offset (+1.8981e1) indicates a base value near 18.98, with variation across the full ρ range on the order of 10⁻⁴ — numerically negligible. This confirms analytically that **correlation structure does not drive instability in the two-asset case**. The slight upward curve toward positive ρ is a second-order numerical artefact. For practical purposes, C(Σ) is constant in ρ for fixed σ₁, σ₂.

---

### Figure 9 — C(Σ) Surface: Volatility Asymmetry

![Figure 9](figures/fig9_two_asset_C_surface.png)

This figure shows how C(Σ) behaves as a function of σ₁ for σ₂ = 0.20 and ρ ∈ {0.1, 0.2, 0.3, 0.4}. The blow-up as σ₁ → 0 is sharp and severe: at σ₁ = 0.05, C(Σ) exceeds 100 for ρ = 0.1, meaning a 1% estimation error in Σ translates to a 100%+ error in allocation weights. As σ₁ → σ₂ (volatility parity), C(Σ) collapses toward its minimum — confirming that symmetric volatility profiles are the most stable for risk parity. Higher correlation (red) reduces C(Σ) at all σ₁ values, but the effect is secondary to the volatility asymmetry.

---

### Figure 10 — Two-Asset Instability Characterisation

![Figure 10](figures/fig10_two_asset_blowup.png)

The left panel confirms that C(Σ) → ∞ as σ₁ → 0, with the blowup concentrated below σ₁ ≈ 0.10. The right panel shows C(Σ) plotted against κ(Σ): the near-vertical line reflects the rapid κ growth as σ₁ → 0, with C growing from ~7 to ~50 across κ ∈ [10, 750]. Together, the two panels establish the instability condition for n = 2: **instability is driven by volatility asymmetry, which manifests as ill-conditioning**.

---

### Figure 11 — Rolling κ(Σ) and Portfolio Turnover (2007–2024)

![Figure 11](figures/fig11_rolling_kappa_turnover.png)

This is the central empirical figure. The top panel shows rolling κ(Σ_t) over 4,336 trading days. The bottom panel shows daily portfolio turnover. Key features:

- **GFC 2008–09:** κ spikes to ~1,950 during the peak of the financial crisis, with a corresponding cluster of high-turnover days (turnover reaching 0.8+). The κ deterioration begins ahead of the recognised crisis window, consistent with the theory predicting instability before it is realised
- **COVID 2020:** κ reaches its all-time high of ~2,776 in early 2020, the sharpest and highest spike in the 17-year sample. Portfolio turnover spikes simultaneously to 0.82 — the single largest reallocation event in the dataset
- **2012–2014 and 2015–2016:** Two secondary κ elevations (reaching ~1,300 and ~1,500 respectively) each accompanied by elevated but more moderate turnover clusters
- **2022:** Post-COVID interest rate shock produces a third distinct κ elevation (~1,300), visible as the portfolio restructures away from long-duration fixed income
- Empirical κ range: 201 to 2,776. Mean: 718. All values are well above the κ = 420 threshold identified in Figure 5 as the boundary of the high-instability regime at T = 126

---

### Figure 12 — Rolling Risk Parity Weights (10 ETFs, 2007–2025)

![Figure 12](figures/fig12_rolling_weights.png)

This figure shows the time-varying allocation across all 10 ETFs. Mean weights across the full sample:

| ETF | Description | Mean weight |
|---|---|---|
| IEF | Medium Treasuries | 0.216 |
| LQD | Corporate Bonds | 0.196 |
| TLT | Long Treasuries | 0.134 |
| GLD | Gold | 0.083 |
| SPY | US Equities | 0.059 |
| EFA | Intl Equities | 0.049 |
| VNQ | REITs | 0.053 |
| HYG | High Yield | 0.071 |
| EEM | EM Equities | 0.047 |
| USO | Oil | 0.048 |

Fixed income ETFs (IEF, LQD, TLT) dominate the portfolio by weight throughout, reflecting their structurally lower volatility. The sharp vertical spikes — most visibly in IEF and LQD around 2009, 2012, 2016, and 2019–2020 — correspond to the high-κ, high-turnover episodes identified in Figure 11. The post-2022 structural shift (rising rates reducing IEF/TLT weights, equity weights rising) is clearly visible in the right portion of the chart.

---

### Figure 13 — Condition Number Spikes During Market Stress

![Figure 13](figures/fig13_kappa_crises.png)

Crisis event annotation confirms the theoretical mechanism. The GFC window (pink) coincides with the first major κ spike (~1,950). The COVID window (orange) coincides with the 2020 κ peak (~2,776). Both episodes demonstrate that market stress — characterised by rapid correlation shifts and volatility clustering — directly translates into covariance ill-conditioning, which in turn amplifies the sensitivity of risk parity allocations to estimation error. The 2016 elevation (~1,500, unmarked) corresponds to the global equity volatility episode driven by oil price collapse and China slowdown concerns.

---

### Figure 14 — Lagged κ Predicts Turnover

![Figure 14](figures/fig14_kappa_predicts_turnover.png)

This scatter plot tests whether κ(Σ_{t−1}) predicts next-day portfolio turnover. The log-linear fit (red) has slope 0.027. The structure in the scatter is clear: virtually all high-turnover observations (turnover > 0.2) are concentrated at κ > 500, while the dense cluster of near-zero turnover observations spans the full κ range. The relationship is noisy — reflecting the fat-tailed, spike-driven nature of the turnover distribution — but directionally consistent with the theoretical prediction that elevated conditioning predicts elevated allocation sensitivity to daily estimation noise in the rolling covariance window.

---

## Summary of Findings

| Finding | Evidence |
|---|---|
| The perturbation bound is valid with zero violations | Figure 3, Figure 4 — bound holds across all 30 κ values and all 7 T values |
| Convergence is O(T⁻¹/²), ratio to bound ≈ 0.67 | Figure 4, bound verification table |
| Conditioning dominates sample size as stability driver | Figure 5 — low-κ portfolios stable even at T=50; high-κ unstable even at T=3200 |
| Correlation cancels in n=2; only volatility asymmetry drives instability | Figures 2, 8, 9, 10 |
| Factor structure produces non-monotone instability via eigenstructure effects | Figure 7 — k=5 more unstable than k=1 despite lower κ |
| Real-world κ values (201–2,776) place portfolios in the high-instability regime | Figure 5 comparison, empirical summary statistics |
| Crisis periods produce κ spikes and simultaneous turnover spikes | Figures 11, 13 — GFC peak κ~1,950; COVID peak κ~2,776; max turnover 0.82 |
| Lagged κ positively predicts subsequent turnover | Figure 14, log-linear slope 0.027 |

---

## Key Functions Reference

| Function | File | Description |
|---|---|---|
| `solve_risk_parity(Sigma)` | `src/risk_parity.py` | Newton solver for w\*, returns weights and diagnostics |
| `compute_jacobian(w_star, Sigma)` | `src/risk_parity.py` | Augmented Jacobian H at solution |
| `stability_constant(w_star, Sigma)` | `src/risk_parity.py` | Returns C(Σ), ‖H⁻¹‖, ‖∂G/∂Σ‖ |
| `two_asset_risk_parity(s1, s2)` | `src/risk_parity.py` | Closed-form n=2 weights |
| `analytical_bound(Sigma, T)` | `src/risk_parity.py` | Upper bound on E[‖δw‖] given Σ and sample size T |
| `make_covariance(n, kappa)` | `src/covariance.py` | Random SPD matrix with target κ |
| `make_factor_covariance(n, k)` | `src/covariance.py` | Factor model Σ = BBᵀ + D |
| `rolling_covariance(returns, window)` | `src/covariance.py` | Rolling sample covariance matrices |
| `ledoit_wolf_analytical(returns)` | `src/covariance.py` | Ledoit-Wolf shrinkage estimator |

---

## References

- Maillard, S., Roncalli, T., & Teiletche, J. (2010). The properties of equally weighted risk contribution portfolios. *Journal of Portfolio Management.*
- Roncalli, T. (2013). *Introduction to Risk Parity and Budgeting.* Chapman & Hall/CRC.
- Tasche, D. (2008). Capital allocation to business units and sub-portfolios: the Euler principle.
- Michaud, R. (1989). The Markowitz optimisation enigma: Is optimised optimal? *Financial Analysts Journal.*
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis.*
- Laloux, L., Cizeau, P., Bouchaud, J.-P., & Potters, M. (1999). Noise dressing of financial correlation matrices. *Physical Review Letters.*
- Bun, J., Bouchaud, J.-P., & Potters, M. (2017). Cleaning large correlation matrices. *Physics Reports.*
- Higham, N. J. (2002). *Accuracy and Stability of Numerical Algorithms.* SIAM.
