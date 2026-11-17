"""Train BehaveFormerMAE and record epoch-by-epoch loss history for Figure N20.

Consumes genuine train and validation windows from `src/pipeline.py`,
records train and validation MSE reconstruction loss for 15 epochs,
saves `reports/mae_training_history.csv`, and renders
`reports/figures/fig_mae_training_loss.png`.
"""
import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.pipeline import build_windows, Normalizer, featurize
from src.classifier_baseline import BehaveFormerMAE

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
REPORTS_DIR = os.path.join(ROOT, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("Building windows and chronological splits...")
    res = build_windows()
    windows = res["windows"]
    train_windows = [w for w in windows if w.split == "train"]
    val_windows = [w for w in windows if w.split == "val"]
    print(f"Windows: train={len(train_windows)}, val={len(val_windows)}")

    normalizer = Normalizer().fit(train_windows)

    train_feats = np.stack([featurize(w.t, w.x, w.y, normalizer) for w in train_windows], axis=0)
    val_feats = np.stack([featurize(w.t, w.x, w.y, normalizer) for w in val_windows], axis=0)

    train_tensor = torch.from_numpy(train_feats).to(DEVICE)
    val_tensor = torch.from_numpy(val_feats).to(DEVICE)

    train_loader = DataLoader(TensorDataset(train_tensor), batch_size=16, shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(val_tensor), batch_size=16, shuffle=False)

    model = BehaveFormerMAE(hidden_dim=64, num_layers=4, nhead=4, mask_ratio=0.15).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 15
    history = []

    print(f"Training BehaveFormerMAE for {epochs} epochs on {DEVICE}...")
    for ep in range(1, epochs + 1):
        # Training
        model.train()
        train_loss = 0.0
        for (bx,) in train_loader:
            b_dict = {
                "mouse_spatial": bx[:, :, :2],
                "mouse_kinematics": bx[:, :, 2:],
                "mouse_mask": torch.zeros((bx.shape[0], bx.shape[1]), dtype=torch.bool, device=DEVICE),
            }
            optimizer.zero_grad()
            recon, _, _, _ = model(b_dict, apply_mask=True)
            loss = nn.functional.mse_loss(recon, bx)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation (masked training objective vs unmasked inference reconstruction)
        model.eval()
        val_loss = 0.0
        val_unmasked_loss = 0.0
        with torch.no_grad():
            for (vx,) in val_loader:
                v_dict = {
                    "mouse_spatial": vx[:, :, :2],
                    "mouse_kinematics": vx[:, :, 2:],
                    "mouse_mask": torch.zeros((vx.shape[0], vx.shape[1]), dtype=torch.bool, device=DEVICE),
                }
                v_recon_masked, _, _, _ = model(v_dict, apply_mask=True)
                val_loss += nn.functional.mse_loss(v_recon_masked, vx).item()
                v_recon_unmasked, _, _, _ = model(v_dict, apply_mask=False)
                val_unmasked_loss += nn.functional.mse_loss(v_recon_unmasked, vx).item()
        val_loss /= len(val_loader)
        val_unmasked_loss /= len(val_loader)

        history.append({
            "epoch": ep,
            "train_mse_loss": round(train_loss, 5),
            "val_mse_loss": round(val_loss, 5),
            "val_unmasked_mse": round(val_unmasked_loss, 5),
        })
        print(
            f"  Epoch {ep:2d}/{epochs:2d} -> Train Masked MSE: {train_loss:.5f} | "
            f"Val Masked MSE: {val_loss:.5f} | Val Unmasked (Inference) MSE: {val_unmasked_loss:.5f}"
        )

    # Save CSV
    df_hist = pd.DataFrame(history)
    csv_out = os.path.join(REPORTS_DIR, "mae_training_history.csv")
    df_hist.to_csv(csv_out, index=False)
    print(f"Saved training history to {csv_out}")

    # Plot figure
    plt.figure(figsize=(8.2, 5.4), dpi=300)
    plt.plot(df_hist["epoch"], df_hist["train_mse_loss"], "o-", color="#2b6cb0", lw=2.2, markersize=5,
             label="Training Masked Loss (15% Token Masking, Dropout=0.1)")
    plt.plot(df_hist["epoch"], df_hist["val_mse_loss"], "s--", color="#c53030", lw=2.0, markersize=5,
             label="Validation Masked Loss (15% Token Masking, Eval Mode)")
    plt.plot(df_hist["epoch"], df_hist["val_unmasked_mse"], "^-.", color="#2f855a", lw=2.0, markersize=5,
             label="Validation Unmasked Reconstruction MSE (Inference Mode)")
    plt.title("BehaveFormerMAE Self-Supervised Reconstruction Convergence", fontsize=12.5, fontweight="bold")
    plt.xlabel("Training Epoch (Batch Size = 16, AdamW, $\\text{lr}=10^{-4}$)", fontsize=11)
    plt.ylabel("Mean Squared Error (MSE)", fontsize=11)
    plt.xticks(df_hist["epoch"])
    plt.grid(True, linestyle="--", alpha=0.45)
    plt.legend(frameon=True, facecolor="white", edgecolor="#ccc", fontsize=9.5)
    plt.tight_layout()

    fig_out = os.path.join(FIGURES_DIR, "fig_mae_training_loss.png")
    plt.savefig(fig_out, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Generated convergence figure {fig_out}")


if __name__ == "__main__":
    main()

