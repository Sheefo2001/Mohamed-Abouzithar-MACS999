"""Evaluate classical anomaly detectors on the 242/968 benchmark test protocol.

Consumes the exact 92-subject split from `src/pipeline.py` (517 train, 167 val,
242 test) and evaluates One-Class SVM, Local Outlier Factor (LOF), Isolation
Forest, and Gaussian Mixture Model (GMM) across all 4 attacker tiers
(T0 scripted_bot, T1 replay_langevin, T2 ou_generator, T3 fgn_matched).

Saves results to reports/classical_baselines_results.csv and prints Table X.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.pipeline import build_windows
from src.attackers import generate, VictimProfile, TIERS

def extract_features(w):
    """Extract standard 12-dimensional kinematic summary statistics."""
    dt = np.diff(w.t)
    dt = np.where(dt <= 0, 1e-3, dt)
    v = np.hypot(np.diff(w.x), np.diff(w.y)) / dt
    a = np.diff(v, prepend=v[0]) / dt
    feats = [
        np.mean(w.x), np.std(w.x), np.ptp(w.x),
        np.mean(w.y), np.std(w.y), np.ptp(w.y),
        np.mean(v), np.std(v), np.max(v),
        np.mean(a), np.std(a), np.max(a)
    ]
    return np.nan_to_num(feats)

def main():
    print("Building windows from src.pipeline...")
    res = build_windows()
    windows = res['windows']
    train_windows = [w for w in windows if w.split == 'train']
    val_windows = [w for w in windows if w.split == 'val']
    test_windows = [w for w in windows if w.split == 'test']

    print(f"Windows: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")

    X_train = np.array([extract_features(w) for w in train_windows])
    X_val = np.array([extract_features(w) for w in val_windows])
    X_test_gen = np.array([extract_features(w) for w in test_windows])

    # Standardize using training statistics only (leak-free)
    mu = np.mean(X_train, axis=0)
    sig = np.std(X_train, axis=0) + 1e-6

    X_train_z = (X_train - mu) / sig
    X_val_z = (X_val - mu) / sig
    X_test_gen_z = (X_test_gen - mu) / sig

    # Profiles for generating attacker tiers
    subject_train = {}
    for w in train_windows:
        subject_train.setdefault(w.subject, []).append(w)
    profiles = {sid: VictimProfile(wins) for sid, wins in subject_train.items()}

    # Generate test attack features per tier
    tier_test_feats = {}
    for tier in TIERS:
        t_list = []
        for w in test_windows:
            att_raw = generate(tier, w, profiles[w.subject])
            class AttW: pass
            aw = AttW()
            aw.t, aw.x, aw.y = att_raw['t'], att_raw['x'], att_raw['y']
            t_list.append(extract_features(aw))
        tier_test_feats[tier] = (np.array(t_list) - mu) / sig

    # Models with hyperparameter calibration
    models = {
        'One-class SVM (RBF)': OneClassSVM(nu=0.08, kernel='rbf', gamma='scale'),
        'Local outlier factor': LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.08),
        'Isolation Forest (n = 100)': IsolationForest(contamination=0.08, random_state=42),
    }

    # GMM trained on clean train, threshold calibrated on clean val
    gmm = GaussianMixture(n_components=3, random_state=42)
    gmm.fit(X_train_z)
    val_scores = gmm.score_samples(X_val_z)
    gmm_thresh = np.percentile(val_scores, 8)

    results = []

    for name, model in models.items():
        model.fit(X_train_z)
        pred_gen = model.predict(X_test_gen_z)
        frr = np.mean(pred_gen == -1) * 100
        
        tier_fars = {}
        all_att_preds = []
        for tier in TIERS:
            pred_att = model.predict(tier_test_feats[tier])
            tier_far = np.mean(pred_att == 1) * 100
            tier_fars[tier] = tier_far
            all_att_preds.extend(pred_att == 1)
        
        overall_far = np.mean(all_att_preds) * 100
        bal_err = (frr + overall_far) / 2
        results.append({
            'Method': name,
            'FRR': frr,
            'T0_FAR': tier_fars['scripted_bot'],
            'T1_FAR': tier_fars['replay_langevin'],
            'T2_FAR': tier_fars['ou_generator'],
            'T3_FAR': tier_fars['fgn_matched'],
            'FAR': overall_far,
            'Bal_Error': bal_err
        })

    # GMM evaluation
    gmm_pred_gen = (gmm.score_samples(X_test_gen_z) < gmm_thresh)
    gmm_frr = np.mean(gmm_pred_gen) * 100
    gmm_tier_fars = {}
    gmm_all_att = []
    for tier in TIERS:
        p_att = (gmm.score_samples(tier_test_feats[tier]) >= gmm_thresh)
        gmm_tier_fars[tier] = np.mean(p_att) * 100
        gmm_all_att.extend(p_att)
    gmm_far = np.mean(gmm_all_att) * 100
    results.append({
        'Method': 'Gaussian mixture model',
        'FRR': gmm_frr,
        'T0_FAR': gmm_tier_fars['scripted_bot'],
        'T1_FAR': gmm_tier_fars['replay_langevin'],
        'T2_FAR': gmm_tier_fars['ou_generator'],
        'T3_FAR': gmm_tier_fars['fgn_matched'],
        'FAR': gmm_far,
        'Bal_Error': (gmm_frr + gmm_far) / 2
    })

    df = pd.DataFrame(results)
    out_path = os.path.join(ROOT, "reports", "classical_baselines_results.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved results to {out_path}")
    print("\n" + "=" * 95)
    print("TABLE 5.10 & SECTION 5.7: CLASSICAL BASELINE EVALUATION (242 GENUINE, 968 ATTACK WINDOWS)")
    print("=" * 95)
    print(f"{'Method':<28} | {'FRR':>7} | {'T0 FAR':>7} | {'T1 FAR':>7} | {'T2 FAR':>7} | {'T3 FAR':>7} | {'Macro FAR':>9} | {'Bal Error':>9}")
    print("-" * 95)
    for _, r in df.iterrows():
        print(f"{r['Method']:<28} | {r['FRR']:>6.2f}% | {r['T0_FAR']:>6.2f}% | {r['T1_FAR']:>6.2f}% | {r['T2_FAR']:>6.2f}% | {r['T3_FAR']:>6.2f}% | {r['FAR']:>8.2f}% | {r['Bal_Error']:>8.2f}%")
    print("-" * 95)
    print(f"{'BehaveFormerMAE (passive)':<28} |   7.44% |   0.00% |  92.56% | 100.00% |  67.77% |   65.08% |    36.26%")
    print(f"{'Stage 1 cascade (Path A)':<28} |  28.10% |   0.00% |  71.90% |  24.79% |  33.47% |   32.54% |    30.32%")
    print("=" * 95)

if __name__ == "__main__":
    main()
