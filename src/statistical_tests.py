"""Recompute all inferential statistics reported in docs/RESULTS.md and the manuscript.

Consumes `reports/evaluation_results.csv` (242 genuine and 968 attack test windows)
and computes:
  1. One-way ANOVA and Tukey HSD over Genuine, T0, and T2 passive MSE.
  2. Fisher's exact tests comparing T0 passive acceptance against T1, T2, and T3.
  3. Non-parametric bootstrap 95% CIs (B = 10,000, seed = 42) for:
       - Passive MSE ROC AUC (all tiers)
       - Stage 1 M-DFA H1 ROC AUC (all tiers, T0 scored as 0)
       - Stage 1 M-DFA H1 ROC AUC (T1-T3 only)
       - Balanced error reduction (Passive gate -> Stage 1 Balanced cascade)
  4. McNemar's paired chi-square tests (with continuity correction) comparing
     Passive gate vs. Stage 1 Balanced cascade on attack and genuine windows.
"""
import os
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CSV_PATH = os.path.join(ROOT, "reports", "evaluation_results.csv")


def bootstrap_auc_ci(y_true: np.ndarray, y_score: np.ndarray, n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    n_pos, n_neg = len(pos), len(neg)
    aucs = np.empty(n_boot)
    labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])
    for i in range(n_boot):
        s_pos = pos[rng.integers(0, n_pos, n_pos)]
        s_neg = neg[rng.integers(0, n_neg, n_neg)]
        aucs[i] = roc_auc_score(labels, np.concatenate([s_pos, s_neg]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y_true, y_score)), float(lo), float(hi)


def main():
    df = pd.read_csv(CSV_PATH)
    df_test = df[df["split"] == "test"].copy()

    # Stage-1 oracle-free H1 exponent (stored directly in 'h1'; verified against 'h2 + delta_h')
    h1_recomputed = df_test["h2"] + df_test["delta_h"]
    assert np.allclose(df_test["h1"].dropna(), h1_recomputed.dropna(), atol=1e-12), "Stage-1 H1 mismatch"
    df_test["h1_stage1"] = df_test["h1"].fillna(0.0)

    gen = df_test[df_test["tier"] == "genuine"]
    t0 = df_test[df_test["tier"] == "scripted_bot"]
    t1 = df_test[df_test["tier"] == "replay_langevin"]
    t2 = df_test[df_test["tier"] == "ou_generator"]
    t3 = df_test[df_test["tier"] == "fgn_matched"]
    att = df_test[df_test["is_genuine"] == 0]

    print("=" * 72)
    print("1. PASSIVE MSE: ONE-WAY ANOVA & TUKEY HSD (Genuine, T0, T2)")
    print("=" * 72)
    f_stat, p_val = stats.f_oneway(gen["mae_error"], t0["mae_error"], t2["mae_error"])
    df_between = 2
    df_within = len(gen) + len(t0) + len(t2) - 3
    print(f"ANOVA F({df_between}, {df_within}) = {f_stat:.2f}, p = {p_val:.2e}")

    tukey = stats.tukey_hsd(gen["mae_error"], t0["mae_error"], t2["mae_error"])
    names = ["Genuine", "T0", "T2"]
    for i in range(3):
        for j in range(i + 1, 3):
            diff = np.mean([gen["mae_error"], t0["mae_error"], t2["mae_error"]][j]) - np.mean(
                [gen["mae_error"], t0["mae_error"], t2["mae_error"]][i]
            )
            p_ij = tukey.pvalue[i, j]
            print(f"  Tukey HSD {names[j]} vs {names[i]}: diff = {diff:+.4f}, p = {p_ij:.4g}")

    print("\n" + "=" * 72)
    print("2. FISHER'S EXACT TEST (T0 Passive Acceptance vs T1, T2, T3)")
    print("=" * 72)
    t0_pass = int(t0["passive_passed"].sum())
    t0_fail = len(t0) - t0_pass
    for name, sub in [("T1", t1), ("T2", t2), ("T3", t3)]:
        sub_pass = int(sub["passive_passed"].sum())
        sub_fail = len(sub) - sub_pass
        table = [[t0_pass, t0_fail], [sub_pass, sub_fail]]
        _, p_fish = stats.fisher_exact(table)
        print(f"  T0 ({t0_pass}/{len(t0)}) vs {name} ({sub_pass}/{len(sub)}): Fisher exact p = {p_fish:.2e}")

    print("\n" + "=" * 72)
    print("3. ROC AUC & 95% BOOTSTRAP CONFIDENCE INTERVALS (B = 10,000)")
    print("=" * 72)
    y_all = df_test["is_genuine"].to_numpy()
    auc_p, lo_p, hi_p = bootstrap_auc_ci(y_all, -df_test["mae_error"].to_numpy(), seed=42)
    print(f"  Passive MSE ROC AUC (all tiers):   {auc_p:.4f} (95% CI {lo_p:.4f} to {hi_p:.4f})")

    auc_a, lo_a, hi_a = bootstrap_auc_ci(y_all, df_test["h1_stage1"].to_numpy(), seed=42)
    print(f"  Stage 1 H1 ROC AUC (all tiers):    {auc_a:.4f} (95% CI {lo_a:.4f} to {hi_a:.4f})")

    df_t13 = df_test[df_test["tier"] != "scripted_bot"]
    auc_13, lo_13, hi_13 = bootstrap_auc_ci(
        df_t13["is_genuine"].to_numpy(), df_t13["h1_stage1"].to_numpy(), seed=42
    )
    print(f"  Stage 1 H1 ROC AUC (T1-T3 only):   {auc_13:.4f} (95% CI {lo_13:.4f} to {hi_13:.4f})")

    for t_name, t_df in [("T1", t1), ("T2", t2), ("T3", t3)]:
        sub_df = pd.concat([gen, t_df], ignore_index=True)
        auc_t, lo_t, hi_t = bootstrap_auc_ci(
            sub_df["is_genuine"].to_numpy(), sub_df["h1_stage1"].to_numpy(), seed=42
        )
        print(f"  Stage 1 H1 ROC AUC ({t_name} only):      {auc_t:.4f} (95% CI {lo_t:.4f} to {hi_t:.4f})")

    print("\n" + "=" * 72)
    print("4. MCNEMAR'S PAIRED TESTS & BALANCED ERROR (Passive vs Stage 1 Cascade)")
    print("=" * 72)
    # Attacks: passive_passed == 1 & authenticated_50ms == 0 -> newly rejected
    att_p = att["passive_passed"].to_numpy()
    att_c = att["authenticated_50ms"].to_numpy()
    b_att = int(np.sum((att_p == 1) & (att_c == 0)))  # newly rejected
    c_att = int(np.sum((att_p == 0) & (att_c == 1)))  # newly accepted
    chi2_att = (abs(b_att - c_att) - 1) ** 2 / (b_att + c_att)
    p_att = stats.chi2.sf(chi2_att, df=1)
    print(f"  Attacks: {b_att} newly rejected, {c_att} newly accepted -> McNemar chi2 = {chi2_att:.2f}, p = {p_att:.2e}")

    # Genuine: passive_passed == 1 & authenticated_50ms == 0 -> newly rejected
    gen_p = gen["passive_passed"].to_numpy()
    gen_c = gen["authenticated_50ms"].to_numpy()
    b_gen = int(np.sum((gen_p == 1) & (gen_c == 0)))
    c_gen = int(np.sum((gen_p == 0) & (gen_c == 1)))
    chi2_gen = (abs(b_gen - c_gen) - 1) ** 2 / (b_gen + c_gen)
    p_gen = stats.chi2.sf(chi2_gen, df=1)
    print(f"  Genuine: {b_gen} newly rejected, {c_gen} newly accepted -> McNemar chi2 = {chi2_gen:.2f}, p = {p_gen:.2e}")

    # Balanced error = 0.5 * (FRR + FAR)
    frr_p = 100.0 * (1.0 - np.mean(gen_p))
    far_p = 100.0 * np.mean(att_p)
    be_p = 0.5 * (frr_p + far_p)

    frr_c = 100.0 * (1.0 - np.mean(gen_c))
    far_c = 100.0 * np.mean(att_c)
    be_c = 0.5 * (frr_c + far_c)
    diff_be = be_p - be_c

    rng = np.random.default_rng(42)
    n_gen, n_att = len(gen_p), len(att_p)
    diffs = np.empty(10000)
    for i in range(10000):
        idx_g = rng.integers(0, n_gen, n_gen)
        idx_a = rng.integers(0, n_att, n_att)
        d_frr = 100.0 * ((1.0 - np.mean(gen_p[idx_g])) - (1.0 - np.mean(gen_c[idx_g])))
        d_far = 100.0 * (np.mean(att_p[idx_a]) - np.mean(att_c[idx_a]))
        diffs[i] = 0.5 * (d_frr + d_far)
    lo_d, hi_d = np.percentile(diffs, [2.5, 97.5])
    print(
        f"  Balanced Error: {be_p:.2f}% -> {be_c:.2f}%, reduction = {diff_be:.2f} pp (95% CI {lo_d:.2f} to {hi_d:.2f})"
    )

    print("\n" + "=" * 72)
    print("5. TWO-SIDED PASSIVE MSE ABLATION (Validation-Calibrated [tau_low, tau_high])")
    print("=" * 72)
    # Exact genuine validation MSE percentiles (N = 167 validation windows)
    tau_high = 0.0137
    val_med = 0.002883
    val_log_med = -2.540155

    for label_q, tau_low in [("P2.5", 0.000545), ("P5.0 (Symmetric 90%)", 0.000568), ("P10.0", 0.000691)]:
        cond_2s_pure = (df_test["mae_error"] >= tau_low) & (df_test["mae_error"] <= tau_high)
        cond_2s_var = cond_2s_pure & (df_test["std_v"] >= 10.0)
        cond_2s_casc = cond_2s_var & (df_test["active_passed"] == 1)
        print(f"  [{label_q}: tau_low = {tau_low:.6f}, tau_high = {tau_high:.4f}]")
        for m_name, c_mask in [
            ("Two-sided MSE only (no std_v gate)", cond_2s_pure),
            ("Two-sided MSE + std_v >= 10 px/s", cond_2s_var),
            ("Two-sided Passive + Stage 1 M-DFA", cond_2s_casc),
        ]:
            frr_m = 100.0 * (1.0 - c_mask[df_test["tier"] == "genuine"].mean())
            t0_m = 100.0 * c_mask[df_test["tier"] == "scripted_bot"].mean()
            t1_m = 100.0 * c_mask[df_test["tier"] == "replay_langevin"].mean()
            t2_m = 100.0 * c_mask[df_test["tier"] == "ou_generator"].mean()
            t3_m = 100.0 * c_mask[df_test["tier"] == "fgn_matched"].mean()
            far_m = 100.0 * c_mask[df_test["is_genuine"] == 0].mean()
            be_m = 0.5 * (frr_m + far_m)
            print(
                f"    {m_name:35s}: FRR={frr_m:5.2f}% | "
                f"FAR: T0={t0_m:4.2f}%, T1={t1_m:5.2f}%, T2={t2_m:5.2f}%, T3={t3_m:5.2f}% | "
                f"All={far_m:5.2f}% | BalErr={be_m:5.2f}%"
            )

    # McNemar test: One-sided passive gate vs Symmetric Two-Sided [P5, P95] passive gate on attacks
    att_2s_p5 = ((att["mae_error"] >= 0.000568) & (att["mae_error"] <= tau_high) & (att["std_v"] >= 10.0)).to_numpy().astype(int)
    b_2s = int(np.sum((att_p == 1) & (att_2s_p5 == 0)))
    c_2s = int(np.sum((att_p == 0) & (att_2s_p5 == 1)))
    chi2_2s = (abs(b_2s - c_2s) - 1) ** 2 / (b_2s + c_2s)
    p_2s = stats.chi2.sf(chi2_2s, df=1)
    print(f"  McNemar (One-sided vs Two-sided [P5, P95] on attacks): {b_2s} newly rejected -> chi2 = {chi2_2s:.2f}, p = {p_2s:.2e}")

    # Continuous two-sided deviation ROC AUCs with B=10,000 bootstrap CIs
    s_abs = -np.abs(df_test["mae_error"].to_numpy() - val_med)
    auc_abs, lo_abs, hi_abs = bootstrap_auc_ci(y_all, s_abs, seed=42)
    print(f"  Two-sided Linear Deviation ROC AUC (-|MSE - med_val|):         {auc_abs:.4f} (95% CI {lo_abs:.4f} to {hi_abs:.4f})")

    s_log = -np.abs(np.log10(df_test["mae_error"].to_numpy() + 1e-9) - val_log_med)
    auc_log, lo_log, hi_log = bootstrap_auc_ci(y_all, s_log, seed=42)
    print(f"  Two-sided Log10 Deviation ROC AUC (-|log10(MSE) - med_val|):   {auc_log:.4f} (95% CI {lo_log:.4f} to {hi_log:.4f})")


if __name__ == "__main__":
    main()
