"""Generate formal figures for IEEE T-IFS Journal Manuscript.

Figures produced in reports/figures/:
  1. fig1_dual_gap_architecture.png: Dual-Gap Threat Model Schematic
  2. fig2_adversary_trajectories.png: Trajectories, Speeds & Spectra across T0-T3 vs Genuine
  3. fig3_mdfa_scaling_curves.png: M-DFA Fluctuation Scaling Curves
  4. fig5_roc_curves.png: ROC Curves (Passive Reconstruction vs Active M-DFA)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import roc_curve, auc

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from src.pipeline import build_windows, kinematics
from src.attackers import generate, VictimProfile

CSV_PATH = os.path.join(ROOT, "reports", "evaluation_results.csv")
JSON_PATH = os.path.join(ROOT, "reports", "results_summary.json")
NPZ_PATH = os.path.join(ROOT, "reports", "dfa_curves.npz")
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Publication styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#444444'
plt.rcParams['axes.linewidth'] = 0.8


def generate_fig1():
    """Figure 1: Architectural Schematic of the Dual-Gap Continuous Authentication Paradigm."""
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # Title box
    ax.text(50, 96, "The Dual-Gap Continuous Authentication Paradigm", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#1a202c")
    ax.text(50, 91.5, "Resolving the Asymmetric Arms Race Between Generative Deepfakes and Passive Anomaly Classifiers",
            ha="center", va="center", fontsize=10, style="italic", color="#4a5568")

    # Left Box: Offensive Deepfake Generative Models
    box_red = patches.FancyBboxPatch((4, 48), 42, 38, boxstyle="round,pad=1.5", 
                                     edgecolor="#e53e3e", facecolor="#fff5f5", linewidth=2)
    ax.add_patch(box_red)
    ax.text(25, 82, "OFFENSIVE GENERATIVE MODEL", ha="center", va="center", fontsize=11, fontweight="bold", color="#c53030")
    ax.text(25, 78, "[ Hybrid GAN-VAE / Diffusion / Davies-Harte fGn ]", ha="center", va="center", fontsize=8.5, color="#742a2a")
    
    red_items = [
        "• Bypasses passive autoencoders via macro mimicry",
        "• Synthesizes smooth, low-jerk coordinates",
        "• THE OFFENSIVE GAP: Lacks embodied closed-loop",
        "  visuomotor feedback & neuromuscular reflexes",
        "• Suffers complete phase collapse under active probe"
    ]
    y_pos = 72
    for item in red_items:
        ax.text(6, y_pos, item, fontsize=8.2, color="#2d3748", family="monospace" if "GAP" in item else "sans-serif")
        y_pos -= 5.5

    # Right Box: Passive Continuous Classifiers
    box_blue = patches.FancyBboxPatch((54, 48), 42, 38, boxstyle="round,pad=1.5", 
                                      edgecolor="#3182ce", facecolor="#ebf8ff", linewidth=2)
    ax.add_patch(box_blue)
    ax.text(75, 82, "PASSIVE DEFENSIVE CLASSIFIER", ha="center", va="center", fontsize=11, fontweight="bold", color="#2b6cb0")
    ax.text(75, 78, "[ Spatio-Temporal Masked Autoencoders / LSTMs ]", ha="center", va="center", fontsize=8.5, color="#2c5282")
    
    blue_items = [
        "• Observes streaming telemetry retrospectively",
        "• Evaluates proxy reconstruction error MSE",
        "• THE DEFENSIVE GAP: Blind to physical origin;",
        "  assigns low MSE to over-smoothed deepfakes",
        "• Catastrophic evasion: 100% FAR on Tier 2 bots"
    ]
    y_pos = 72
    for item in blue_items:
        ax.text(56, y_pos, item, fontsize=8.2, color="#2d3748", family="monospace" if "GAP" in item else "sans-serif")
        y_pos -= 5.5

    # Arrow from Offensive to Passive (Bypass)
    ax.annotate("", xy=(54, 67), xytext=(46, 67),
                arrowprops=dict(arrowstyle="->", color="#e53e3e", lw=3.0))
    ax.text(50, 70, "Macro Evasion\n(100% FAR)", ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#c53030")

    # Bottom Box: The Agentic Active Defense Engine
    box_green = patches.FancyBboxPatch((15, 6), 70, 34, boxstyle="round,pad=1.5", 
                                       edgecolor="#2f855a", facecolor="#f0fff4", linewidth=2.5)
    ax.add_patch(box_green)
    ax.text(50, 36, "AGENTIC ACTIVE DEFENSE SYSTEM (M-DFA + LangGraph)", ha="center", va="center", fontsize=12, fontweight="bold", color="#22543d")
    ax.text(50, 32, "NIST SP 800-207 Zero Trust Policy Engine & Dynamic Enforcement", ha="center", va="center", fontsize=9, color="#276749")

    green_cols = [
        ("Path A: Stage 1 M-DFA Cascade", [
            "• τ1 = 50ms latency probe",
            "• Multi-scale DFA (H1, ΔH, ℓ_F)",
            "• 94.21% genuine session share",
            "• Cuts overall FAR to 32.54%",
            "• Evaluated on unperturbed data"
        ]),
        ("Path B: Step-Up Challenge (Oracle Bound)", [
            "• τ2 = 150ms step-up probe",
            "• Triggered on |H1 - H1*| <= 0.05",
            "• Theoretical analytical bound",
            "• Oracle challenge bound: 0.00% FAR",
            "• Genuine FRR bound: 22.31%"
        ])
    ]
    
    # Path A text
    ax.text(32, 27, green_cols[0][0], fontsize=9.5, fontweight="bold", color="#234e52")
    y_p = 23
    for it in green_cols[0][1]:
        ax.text(20, y_p, it, fontsize=8.2, color="#2d3748")
        y_p -= 4

    # Path B text
    ax.text(68, 27, green_cols[1][0], fontsize=9.5, fontweight="bold", color="#234e52")
    y_p = 23
    for it in green_cols[1][1]:
        ax.text(56, y_p, it, fontsize=8.2, color="#2d3748")
        y_p -= 4

    # Arrows from top boxes to bottom active defense
    ax.annotate("", xy=(30, 40), xytext=(25, 48),
                arrowprops=dict(arrowstyle="->", color="#2f855a", lw=2, linestyle="--"))
    ax.annotate("", xy=(70, 40), xytext=(75, 48),
                arrowprops=dict(arrowstyle="->", color="#2f855a", lw=2, linestyle="--"))

    out_path = os.path.join(FIGURES_DIR, "fig1_dual_gap_architecture.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_fig2():
    """Figure 2: Real BB-MAS Trajectories, Speeds & Cohort Kinematics across T0-T3 vs Genuine."""
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.8), dpi=300)

    # Load representative held-out BB-MAS test window (Subject 57, Window 19; H1 = 0.8834, near cohort median)
    # where Genuine and T1 are accepted, and T0, T2, and T3 are rejected.
    res = build_windows()
    windows = res["windows"]
    train_wins = [w for w in windows if w.split == "train"]
    test_wins = [w for w in windows if w.split == "test"]
    w = next((win for win in test_wins if win.subject == 57 and win.index == 19), test_wins[0])
    prof = VictimProfile([tw for tw in train_wins if tw.subject == w.subject])

    att_t0 = generate("scripted_bot", w, prof)
    att_t1 = generate("replay_langevin", w, prof)
    att_t2 = generate("ou_generator", w, prof)
    att_t3 = generate("fgn_matched", w, prof)

    # 1. Spatial Trajectories (Top Left - Framed to Desktop Viewport)
    ax1 = axes[0, 0]
    ax1.plot(w.x, w.y, color="#1b7837", lw=2.4, label=f"Genuine Human (Subj. {w.subject}, Win. {w.index})")
    ax1.plot(att_t0["x"], att_t0["y"], color="#3182ce", lw=2.0, linestyle="--", label="T0: Scripted Linear Bot")
    ax1.plot(att_t1["x"], att_t1["y"], color="#e66101", lw=1.4, linestyle="-", alpha=0.85, label="T1: Replay + Langevin OU")
    ax1.plot(att_t2["x"], att_t2["y"], color="#7570b3", lw=1.8, linestyle="-.", label="T2: OU Minimum-Jerk")
    ax1.plot(att_t3["x"], att_t3["y"], color="#d7191c", lw=1.5, linestyle=":", alpha=0.85, label="T3: fGn Matched (Unbounded Walk)")
    all_desk_x = np.concatenate([w.x, att_t0["x"], att_t1["x"], att_t2["x"]])
    all_desk_y = np.concatenate([w.y, att_t0["y"], att_t1["y"], att_t2["y"]])
    pad_x = max(np.ptp(all_desk_x) * 0.14, 100.0)
    pad_y = max(np.ptp(all_desk_y) * 0.18, 100.0)
    ax1.set_xlim(all_desk_x.min() - pad_x, all_desk_x.max() + pad_x)
    ax1.set_ylim(all_desk_y.min() - pad_y, all_desk_y.max() + pad_y)
    ax1.text(
        0.97, 0.04,
        "Note: T3 fGn is an unanchored\nrandom walk (exits viewport)",
        transform=ax1.transAxes, ha="right", va="bottom", fontsize=7.5, style="italic",
        color="#9b2c2c", bbox=dict(boxstyle="round,pad=0.25", facecolor="#fff5f5", edgecolor="#feb2b2", alpha=0.9)
    )
    ax1.set_title(f"(a) Real BB-MAS & Synthesized 2D Trajectories (Subj. {w.subject}, Win. {w.index})", fontsize=10.8, fontweight="bold")
    ax1.set_xlabel("X Coordinate (pixels)", fontsize=10)
    ax1.set_ylabel("Y Coordinate (pixels)", fontsize=10)
    ax1.legend(loc="upper left", fontsize=7.6, frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # 2. Instantaneous Speed Profiles (Top Right)
    v_gen, _ = kinematics(w.t, w.x, w.y)
    v_t0, _ = kinematics(att_t0["t"], att_t0["x"], att_t0["y"])
    v_t1, _ = kinematics(att_t1["t"], att_t1["x"], att_t1["y"])
    v_t2, _ = kinematics(att_t2["t"], att_t2["x"], att_t2["y"])
    v_t3, _ = kinematics(att_t3["t"], att_t3["x"], att_t3["y"])
    sample_idx = np.arange(len(v_gen))

    ax2 = axes[0, 1]
    ax2.plot(sample_idx, v_gen, color="#1b7837", lw=1.5, label="Genuine Human")
    ax2.plot(sample_idx, v_t0, color="#3182ce", lw=2.0, linestyle="--", label="T0 Scripted (Constant Speed)")
    ax2.plot(sample_idx, v_t1, color="#e66101", lw=1.1, alpha=0.80, label="T1 Replay + Langevin")
    ax2.plot(sample_idx, v_t2, color="#7570b3", lw=1.8, linestyle="-.", label="T2 OU Minimum-Jerk")
    ax2.plot(sample_idx, v_t3, color="#d7191c", lw=1.0, linestyle=":", alpha=0.70, label="T3 fGn Matched")
    ax2.set_title(f"(b) Instantaneous Speed Profiles $v_k$ (Subj. {w.subject}, Win. {w.index})", fontsize=10.8, fontweight="bold")
    ax2.set_xlabel("Event Step $k$ ($N = 256$ steps)", fontsize=10)
    ax2.set_ylabel("Speed (px/s, symlog scale)", fontsize=10)
    ax2.set_yscale("symlog", linthresh=50.0)
    ax2.legend(loc="upper right", fontsize=7.6, frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.5)

    # 3. Speed Variance Bar Comparison (Bottom Left - Test Cohort Means, N=242 per tier, symlog scale)
    df_eval = pd.read_csv(CSV_PATH)
    test_eval = df_eval[df_eval["split"] == "test"]
    ax3 = axes[1, 0]
    labels = ["Genuine", "T0 Scripted", "T1 Langevin", "T2 Min-Jerk", "T3 fGn"]
    tier_keys = ["genuine", "scripted_bot", "replay_langevin", "ou_generator", "fgn_matched"]
    stds = [float(test_eval[test_eval["tier"] == tk]["std_v"].mean()) for tk in tier_keys]
    colors = ["#1b7837", "#3182ce", "#e66101", "#7570b3", "#d7191c"]
    bars = ax3.bar(labels, stds, color=colors, edgecolor="black", width=0.55)
    ax3.set_yscale("symlog", linthresh=10.0)
    ax3.axhline(y=10.0, color="red", linestyle="--", lw=1.9, label=r"$\sigma_{\min} = 10.0$ px/s Gate")
    ax3.set_title(r"(c) Cohort Mean Speed SD $\sigma_v$ ($N = 242$ Windows/Tier)", fontsize=10.8, fontweight="bold")
    ax3.set_ylabel("Mean Speed Std Dev (px/s, symlog)", fontsize=10)
    ax3.set_ylim(0, max(stds) * 2.8)
    ax3.legend(loc="upper left", fontsize=8.2, frameon=True)
    ax3.grid(axis="y", linestyle=":", alpha=0.5)
    for bar in bars:
        h = bar.get_height()
        y_txt = h * 1.18 if h > 0 else 0.8
        ax3.text(bar.get_x() + bar.get_width()/2, y_txt, f"{h:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    # 4. Hurst Exponent Distribution Boxplot (Bottom Right)
    ax4 = axes[1, 1]
    data_h1 = [
        (test_eval[test_eval["tier"] == "genuine"]["h2"] + test_eval[test_eval["tier"] == "genuine"]["delta_h"]).dropna().values,
        np.array([]),  # T0 scripted bots have constant speed (undefined H1)
        (test_eval[test_eval["tier"] == "replay_langevin"]["h2"] + test_eval[test_eval["tier"] == "replay_langevin"]["delta_h"]).dropna().values,
        (test_eval[test_eval["tier"] == "ou_generator"]["h2"] + test_eval[test_eval["tier"] == "ou_generator"]["delta_h"]).dropna().values,
        (test_eval[test_eval["tier"] == "fgn_matched"]["h2"] + test_eval[test_eval["tier"] == "fgn_matched"]["delta_h"]).dropna().values,
    ]
    bplot = ax4.boxplot(data_h1, tick_labels=labels, patch_artist=True, medianprops=dict(color="black", lw=1.5))
    for patch, col in zip(bplot['boxes'], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.7)
    ax4.axhspan(0.5943, 0.6443, color="#cbd5e0", alpha=0.65, label=r"Escalation Band $[0.5943, 0.6443)$")
    ax4.axhline(y=0.6443, color="black", linestyle="--", lw=1.8, label=r"Balanced $H_1^* = 0.6443$")
    ax4.text(
        2, 0.72, "undefined\n($\\sigma_v = 0$)",
        ha="center", va="center", fontsize=8.0, style="italic", color="#2b6cb0",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ebf8ff", edgecolor="#90cdf4", alpha=0.9)
    )
    ax4.set_title("(d) Short-Range Hurst Exponent $H_1$ Scaling", fontsize=10.8, fontweight="bold")
    ax4.set_ylabel("Hurst Exponent $H_1$", fontsize=10)
    ax4.legend(loc="lower left", fontsize=8.0, frameon=True)
    ax4.grid(axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_adversary_trajectories.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_fig3():
    """Figure 3: Multi-Scale DFA Scaling Curves."""
    if not os.path.exists(NPZ_PATH):
        print(f"NPZ file not found at {NPZ_PATH}, skipping fig3.")
        return

    data = np.load(NPZ_PATH)
    scales = data["scales"]
    log_s = np.log10(scales)

    palette = {
        "genuine": ("#1b7837", "o-", "Genuine Human ($H_1 = 0.86$)"),
        "replay_langevin": ("#e66101", "s--", "T1: Replay + Langevin SDE ($H_1 = 0.86$)"),
        "ou_generator": ("#7570b3", "^-.", "T2: OU Minimum-Jerk ($H_1 = 1.05$)"),
        "fgn_matched": ("#d7191c", "d:", "T3: fGn Noise-Matched ($H_1 = 0.77$)"),
    }

    fig, ax = plt.subplots(figsize=(9, 6), dpi=300)
    for tier, (col, fmt, lbl) in palette.items():
        mean_key = f"{tier}_mean"
        if mean_key in data:
            curve = np.log10(data[mean_key])
            ax.plot(log_s, curve, fmt, label=lbl, color=col, linewidth=2.2, markersize=4.5)

    # Highlight short-range and long-range windows
    ax.axvspan(np.log10(4), np.log10(16), color="#fee08b", alpha=0.35, label=r"Visuomotor Scaling Window ($s \in [4, 16]$)")
    ax.axvspan(np.log10(17), np.log10(64), color="#e0f3f8", alpha=0.35, label=r"Long-Range Drift Window ($s \in [17, 64]$)")

    ax.set_title("Multi-Scale Detrended Fluctuation Analysis (M-DFA) Scaling Curves", fontsize=13, fontweight="bold")
    ax.set_xlabel(r"Scale $\log_{10} s$ (Segment time lag in discrete event units)", fontsize=11.5)
    ax.set_ylabel(r"Detrended Fluctuation $\log_{10} F(s)$", fontsize=11.5)
    ax.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9, loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(FIGURES_DIR, "fig3_mdfa_scaling_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def generate_fig5():
    """Figure 5: ROC Curves (Passive Reconstruction vs Active M-DFA)."""
    df = pd.read_csv(CSV_PATH)
    test_gen = df[(df["split"] == "test") & (df["is_genuine"] == 1)]
    test_att = df[(df["split"] == "test") & (df["is_genuine"] == 0)]

    # Genuine = 1 (positive class), Attacker = 0 (negative class)
    # FPR = Impostor Acceptance (FAR), TPR = Genuine Acceptance (1 - FRR)
    y_true_test = test_gen["is_genuine"].tolist() + test_att["is_genuine"].tolist()

    # Passive MSE: lower error -> more genuine -> score = -mae_error
    y_scores_p = (-test_gen["mae_error"]).tolist() + (-test_att["mae_error"]).tolist()
    fpr_p, tpr_p, _ = roc_curve(y_true_test, y_scores_p)
    auc_p = float(auc(fpr_p, tpr_p))

    # Active DFA H1: raw Stage 1 physical exponent (h2 + delta_h, fillna(0) for T0)
    h1_gen_test = (test_gen["h2"] + test_gen["delta_h"]).fillna(0.0).tolist()
    h1_att_test = (test_att["h2"] + test_att["delta_h"]).fillna(0.0).tolist()
    y_scores_a = h1_gen_test + h1_att_test
    fpr_a, tpr_a, _ = roc_curve(y_true_test, y_scores_a)
    auc_a = float(auc(fpr_a, tpr_a))

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.plot(fpr_p, tpr_p, color="#d73027", linewidth=2.5, label=f"Passive Reconstruction MSE (AUC = {auc_p:.4f})")
    ax.plot(fpr_a, tpr_a, color="#1a9850", linewidth=2.5, label=f"Active Stage 1 $H_1$ (AUC = {auc_a:.4f}; T1–T3 = 0.464)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance Level (AUC = 0.5000)")

    ax.set_xlabel("False Positive Rate (Impostor Acceptance / FAR)", fontsize=11.5)
    ax.set_ylabel("True Positive Rate (Genuine Verification / 1 - FRR)", fontsize=11.5)
    ax.set_title("Receiver Operating Characteristic (ROC) Comparison\non Frozen Test Evaluation Split (N = 1,210 Windows)", fontsize=12.5, fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9.5)
    ax.grid(True, linestyle="--", alpha=0.4)

    out_path = os.path.join(FIGURES_DIR, "fig5_roc_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated {out_path}")


def main():
    print("Generating formal manuscript figures...")
    generate_fig1()
    generate_fig2()
    generate_fig3()
    generate_fig5()
    print("All figures generated successfully in reports/figures/!")


if __name__ == "__main__":
    main()
