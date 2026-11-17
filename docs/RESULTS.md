# Detailed Results

All numbers below are computed from `reports/evaluation_results.csv` (242 genuine and 968 attack test windows) and `reports/classical_baselines_results.csv`. Stage 2 values are oracle upper bounds (see the README).

## Passive baseline (BehaveFormerMAE)

| Cohort | MSE (mean ± SD) | Error |
|---|---|---|
| Genuine | 0.0048 ± 0.0050 | FRR 7.44% (18/242) |
| T0 scripted bot | 0.0003 ± 0.0001 | FAR 0.00% (0/242) |
| T1 Langevin replay | 0.0048 ± 0.0050 | FAR 92.56% (224/242) |
| T2 OU minimum-jerk | 0.0009 ± 0.0005 | FAR 100.00% (242/242) |
| T3 fGn matched | 0.0112 ± 0.0067 | FAR 67.77% (164/242) |
| All attacks | | FAR 65.08% (630/968) |

Passive rule: MSE ≤ 0.0137 (95th percentile of genuine validation MSE) and speed SD ≥ 10 px/s. T0 is rejected by the speed-variance gate, not by its reconstruction error. Passive ROC AUC 0.3622 (95% bootstrap CI 0.3303 to 0.3932, B = 10,000).

ANOVA over genuine, T0, and T2 MSE: F(2, 723) = 173.25, p = 3.38e-62. Tukey HSD: T0 vs genuine -0.0045 (p < 0.001), T2 vs genuine -0.0039 (p < 0.001), T2 vs T0 +0.0006 (p = 0.056). Fisher's exact test, T0 vs each of T1 to T3: p < 1e-68.

## M-DFA features (Stage 1, all 242 windows per cohort)

| Cohort | H1 (mean ± SD) | H2 | ΔH = H1 - H2 | log10 F(4) |
|---|---|---|---|---|
| Genuine | 0.8639 ± 0.1774 | 0.5734 | 0.2905 | 2.4980 |
| T1 replay | 0.8634 ± 0.1771 | 0.5733 | 0.2901 | 2.4980 |
| T2 OU | 1.0485 ± 0.1443 | 0.9149 | 0.1336 | 1.7153 |
| T3 fGn | 0.7659 ± 0.1778 | 0.6463 | 0.1196 | 2.9828 |

T0 has constant speed, so its exponents are undefined. H1 ROC AUC (B = 10,000): 0.5982 over all tiers (T0 scored as 0; 95% CI 0.5649 to 0.6316), 0.4642 over T1 to T3 (95% CI 0.4237 to 0.5042); per-tier H1 ROC AUC: T1 0.5015, T2 0.2241, T3 0.6672.

## Stage 1 cascade and operating regimes

| Regime | H1* | ΔH min | log F min | FAR T1 | FAR T2 | FAR T3 | FAR | FRR |
|---|---|---|---|---|---|---|---|---|
| Frictionless | 0.6146 | n/a | n/a | 81.82 | 100.00 | 50.41 | 58.06 | 18.60 |
| Balanced | 0.6443 | 0.08 | 1.85 | 71.90 | 24.79 | 33.47 | 32.54 | 28.10 |
| High-security | 0.9058 | 0.12 | 2.00 | 41.74 | 9.92 | 11.57 | 15.81 | 57.85 |

All regimes include the passive gate; T0 FAR is 0% in every regime. H1 thresholds are calibrated on validation data; the ΔH and log F thresholds are fixed constants.

Balanced cascade vs passive detector on identical windows: 315 attack windows newly rejected, none newly accepted (McNemar χ² = 313.00, p = 4.84e-70); 50 genuine windows newly rejected (χ² = 48.02, p = 4.2e-12). Balanced error 36.26% → 30.32%, difference 5.94 points (95% bootstrap CI 2.94 to 8.83, B = 10,000).

## Two-stage escalation (oracle)

Windows with H1 in [H1* - 0.05, H1*) are escalated: 14 genuine (5.79%), 14 T1, 13 T3, 0 T0, 0 T2. With the oracle, all 14 genuine windows are recovered (FRR 28.10% → 22.31%) and all 27 attack windows are rejected. Escalated windows were not accepted at Stage 1, so the FAR is unchanged at 32.54%.

## Two-sided passive threshold ablation

Calibrating a lower reconstruction bound $\tau_{\text{low}}$ on the 167 genuine validation windows alongside $\tau_{\text{high}} = P_{95} = 0.0137$ tests whether rejecting windows that are "too easy" to reconstruct ($\text{MSE} < \tau_{\text{low}}$) mitigates smooth synthetic generators (T0, T2):

| Rule (validation-calibrated) | $\tau_{\text{low}}$ | Genuine FRR | FAR T0 | FAR T1 | FAR T2 | FAR T3 | Overall FAR | Balanced Error |
|---|---|---|---|---|---|---|---|---|
| One-sided ($\text{MSE} \le 0.0137, \sigma_v \ge 10$) | — | 7.44% | 0.00% | 92.56% | 100.00% | 67.77% | 65.08% | 36.26% |
| Two-sided $[P_{2.5}, P_{95}]$ (MSE only) | 0.000545 | 9.09% | 0.41% | 90.91% | 67.36% | 67.77% | 56.61% | 32.85% |
| Two-sided $[P_5, P_{95}]$ (MSE only) | 0.000568 | 9.09% | 0.00% | 90.91% | 65.29% | 67.77% | 55.99% | 32.54% |
| Two-sided $[P_{10}, P_{95}]$ (MSE only) | 0.000691 | 12.81% | 0.00% | 87.19% | 59.92% | 67.77% | 53.72% | 33.26% |
| Two-sided $[P_5, P_{95}]$ + Stage 1 M-DFA | 0.000568 | 28.93% | 0.00% | 71.07% | 24.79% | 33.47% | 32.33% | 30.63% |

- **Elimination of T0 without the variance gate:** At $\tau_{\text{low}} = P_5 = 0.000568$, all 242 T0 windows ($\max \text{MSE} = 0.000548$) fall below $\tau_{\text{low}}$ and are rejected ($0.00\%$ FAR) even without the $\sigma_v \ge 10\text{ px/s}$ gate.
- **Partial mitigation of T2:** The symmetric $[P_5, P_{95}]$ bound rejects 84 of 242 T2 windows (cutting passive T2 FAR from $100.00\%$ to $65.29\%$, McNemar $\chi^2 = 86.01, p = 1.79 \times 10^{-20}$ across all attacks), but $65.29\%$ of T2 windows still overlap the lower tail of genuine MSE ($[0.000568, 0.0029]$). All 84 T2 windows rejected by $\tau_{\text{low}} = P_5$ are already rejected by Stage 1 M-DFA (leaving cascaded T2 FAR unchanged at $24.79\%$).
- **Two-sided continuous ROC AUC ($B = 10{,}000$):** Scoring deviation from the genuine validation median ($\text{MSE}_{\text{med, val}} = 0.002883$) fixes the passive ROC inversion, raising AUC from $0.3622$ to **$0.6128$** ($95\%$ CI $0.5678$ to $0.6557$) for $-\lvert \text{MSE} - \text{MSE}_{\text{med, val}}\rvert$ and **$0.7186$** ($95\%$ CI $0.6868$ to $0.7497$) for $-\lvert \log_{10}(\text{MSE}) - \text{median}_{\text{val}}(\log_{10}\text{MSE})\rvert$.

## Unimodal checks

- **Balabit (user12, 233 sessions):** clean H1 0.6812 vs 0.6093 with Gaussian positional noise (σ = 0.05); Welch's t = 3.5640, p = 4.12e-4.
- **CMU (51 subjects, 20,400 repetitions):** the fatigue model with d = 0.1 produced a stable mean flight-time dilation of +0.0080 s.
