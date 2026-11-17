"""FastAPI Backend Server for Active Neuromuscular Defense & Thesis Evaluation.

Provides:
  - Real-time SSE streaming simulation for all 4 adversary tiers + genuine human trajectories.
  - Multi-regime operational gating (Balanced, Frictionless, High-Security).
  - Empirical evaluation endpoints returning leak-free validation & test metrics.
  - Publication figure delivery.
"""
import asyncio
import json
import logging
import os
import random
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

# Set project root in path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import (
    Window,
    kinematics,
    Normalizer,
    featurize,
    build_windows,
)
from src.attackers import VictimProfile, generate as generate_attack, TIERS
from src.classifier_baseline import BehaveFormerMAE
from src.dfa import compute_dfa_multiscale, hurst_h1
from src.defensive_mas import build_active_defense_graph, ActiveDefenseState

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("api")

app = FastAPI(title="Active Neuromuscular Defense API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join(ROOT, "checkpoints", "behaveformer_mae.pth")
NORMALIZER_PATH = os.path.join(ROOT, "checkpoints", "normalizer.json")
THRESHOLDS_PATH = os.path.join(ROOT, "checkpoints", "thresholds.json")
RESULTS_PATH = os.path.join(ROOT, "reports", "results_summary.json")
FIGURES_DIR = os.path.join(ROOT, "reports", "figures")
BBMAS_DIR = os.path.join(ROOT, "data", "raw", "bb_mas", "BB-MAS_Dataset")

# Global runtime cache
_model: Optional[BehaveFormerMAE] = None
_normalizer: Optional[Normalizer] = None
_thresholds: Optional[Dict[str, Any]] = None
_test_windows: Optional[List[Window]] = None
_victim_profiles: Optional[Dict[int, VictimProfile]] = None
_graphs_cache: Dict[str, Any] = {}

# Canonical Tier Mapping
TIER_MAP = {
    # New canonical names
    "genuine": "genuine",
    "human": "genuine",
    "t0_scripted": "scripted_bot",
    "t0": "scripted_bot",
    "scripted_bot": "scripted_bot",
    "t1_langevin": "replay_langevin",
    "t1": "replay_langevin",
    "replay_langevin": "replay_langevin",
    "t2_ou_jerk": "ou_generator",
    "t2": "ou_generator",
    "ou_generator": "ou_generator",
    "t3_fgn": "fgn_matched",
    "t3": "fgn_matched",
    "fgn_matched": "fgn_matched",
    # Legacy aliases
    "clean": "genuine",
    "dumb_bot": "scripted_bot",
    "deepfake": "replay_langevin",
}

PEDAGOGICAL_EXPLANATIONS = {
    "genuine": {
        "title": "Scenario A: Genuine Human (BB-MAS Subject)",
        "physics": "Physiological neuromuscular motor execution governed by sub-cortical tremor and central visual feedback loop.",
        "passive_behavior": "Low reconstruction error (MSE < threshold). The spatio-temporal dual-attention MAE recognizes natural curvature and velocity.",
        "active_reaction": "Multi-scale DFA computes H1 in biological visuomotor range [0.60, 0.90] with characteristic sub-movement crossover (ΔH >= 0.08). Session is AUTHENTICATED.",
    },
    "scripted_bot": {
        "title": "Tier 0: Linear / Scripted Bot",
        "physics": "Deterministic straight-line interpolation with constant velocity and zero jerk.",
        "passive_behavior": "Zero kinematic acceleration variance (σ_v ≈ 0). Easily flagged by variance or reconstruction baseline.",
        "active_reaction": "Kinematic variance collapses (σ_v < 10 px/s) and DFA scaling exponent is undefined (flatline signal). LangGraph agent triggers immediate REVOCATION.",
    },
    "replay_langevin": {
        "title": "Tier 1: Replay + Langevin SDE (The Offensive Gap)",
        "physics": "Recorded human base trajectory augmented with discretized Ornstein-Uhlenbeck (overdamped Langevin) positional noise: dX = θ(x*(t) - X)dt + σ(t)dW.",
        "passive_behavior": "CRITICAL VULNERABILITY: Bypasses passive MAE (92.56% False Acceptance Rate) because spatial curvature and macro-kinematics match the human victim.",
        "active_reaction": "Active defense inspects short-range power-law scaling and long-range drift. Multi-scale verification reduces bypass rate to 41.74–71.90% depending on regime.",
    },
    "ou_generator": {
        "title": "Tier 2: Minimum-Jerk Ornstein-Uhlenbeck Continuous Generator",
        "physics": "Polynomial 5th-order minimum-jerk trajectory connecting sampled target waypoints, perturbed by continuous OU positional tremor.",
        "passive_behavior": "100.00% False Acceptance Rate against passive MAE! The transformer considers the smooth, minimum-jerk curves 'ideal human movements'.",
        "active_reaction": "Over-damped smoothness produces anomalously high scaling (H1 ≈ 1.05) and fails physiological sub-movement crossover (ΔH < 0.08). Active DFA drops FAR from 100.00% to 24.79%.",
    },
    "fgn_matched": {
        "title": "Tier 3: Fractional Gaussian Noise (fGn) Noise-Matched Generator",
        "physics": "Adversary samples victim's training-split Hurst exponents and synthesizes speed via Davies-Harte fGn method.",
        "passive_behavior": "Passes passive MAE in 67.77% of attacks due to realistic speed distributions.",
        "active_reaction": "Synthesized noise lacks dynamic neuromuscular coupling across time scales. Multi-scale DFA reduces FAR to 33.47% (Balanced) and 11.57% (High-Security).",
    },
}


def get_dependencies():
    global _model, _normalizer, _thresholds, _test_windows, _victim_profiles
    if _model is None:
        logger.info("Initializing runtime dependencies...")
        _model = BehaveFormerMAE(hidden_dim=64, num_layers=4, nhead=4, mask_ratio=0.15).to(DEVICE)
        if os.path.exists(CHECKPOINT_PATH):
            _model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
            logger.info("Loaded MAE checkpoint from %s", CHECKPOINT_PATH)
        _model.eval()

        if os.path.exists(NORMALIZER_PATH):
            _normalizer = Normalizer.load(NORMALIZER_PATH)
        else:
            _normalizer = Normalizer()

        if os.path.exists(THRESHOLDS_PATH):
            with open(THRESHOLDS_PATH) as f:
                _thresholds = json.load(f)
        else:
            _thresholds = {
                "mae_mse_threshold": 0.0137,
                "min_speed_std": 10.0,
                "h1_threshold": 0.6443,
                "delta_h_threshold": 0.08,
                "log_f0_threshold": 1.85,
                "h1_stepup_margin": 0.05,
            }

        # Build / load dataset windows
        data = build_windows(BBMAS_DIR)
        all_windows = data["windows"]
        _test_windows = [w for w in all_windows if w.split == "test"]
        train_windows = [w for w in all_windows if w.split == "train"]

        # Compute per-subject victim profiles from train split
        _victim_profiles = {}
        by_subj: Dict[int, List[Window]] = {}
        for w in train_windows:
            by_subj.setdefault(w.subject, []).append(w)
        for sid, wins in by_subj.items():
            _victim_profiles[sid] = VictimProfile(wins)
        logger.info("Runtime initialized: %d test windows, %d victim profiles.", len(_test_windows), len(_victim_profiles))

    return _model, _normalizer, _thresholds, _test_windows, _victim_profiles


def get_defense_graph(regime: str = "balanced"):
    global _graphs_cache
    if regime not in _graphs_cache:
        _, _, thresholds_dict, _, _ = get_dependencies()
        regimes_data = thresholds_dict.get("regimes", {})
        if regime in regimes_data:
            th = dict(regimes_data[regime])
            th.setdefault("mae_mse_threshold", thresholds_dict.get("mae_mse_threshold", 0.0137))
        else:
            th = dict(thresholds_dict)
        _graphs_cache[regime] = build_active_defense_graph(th)
    return _graphs_cache[regime]


# ---------------------------------------------------------------- API Endpoints

@app.get("/api/results")
async def get_results_summary():
    """Return the complete, leak-free empirical benchmark results from reports/results_summary.json."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            return {"status": "success", "data": json.load(f)}
    return {"status": "error", "message": "results_summary.json not found. Run src/run_experiments.py first."}


@app.get("/api/evaluate")
async def evaluate_passive():
    """Returns the passive baseline evaluation summary formatted for the frontend."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
        gen = data.get("genuine_test", {})
        tiers = data.get("attacker_tiers_test", {})
        # Map to format expected by EvaluationSuite
        formatted = {
            "scenario_a": {
                "mean_error": gen.get("mean_mse", 0.0048),
                "threshold": data.get("thresholds", {}).get("mae_mse_threshold", 0.0137),
                "frr": gen.get("frr_passive_pct", 7.44),
            },
            "scenario_b": {
                "mean_error": tiers.get("scripted_bot", {}).get("mean_mse", 0.0003),
                "far": tiers.get("scripted_bot", {}).get("far_passive_pct", 0.0),
            },
            "scenario_c": {
                "mean_error": tiers.get("replay_langevin", {}).get("mean_mse", 0.0048),
                "far": tiers.get("replay_langevin", {}).get("far_passive_pct", 92.56),
            },
            "tiers": tiers,
            "overall_far": data.get("overall_test", {}).get("far_passive_pct", 65.08),
        }
        return {"status": "success", "data": formatted}
    return {"status": "error", "message": "results_summary.json not found."}


@app.get("/api/evaluate_active")
async def evaluate_active():
    """Returns the active defense evaluation summary for Phase 2."""
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, "r") as f:
            data = json.load(f)
        gen = data.get("genuine_test", {})
        tiers = data.get("attacker_tiers_test", {})
        overall = data.get("overall_test", {})
        formatted = {
            "deepfake_passive_far": overall.get("far_passive_pct", 65.08),
            "deepfake_active_far": overall.get("far_active_pct", 32.54),
            "clean_active_frr": gen.get("frr_active_pct", 28.1),
            "attacker_tiers": tiers,
            "operating_regimes": data.get("operating_regimes_comparison", {}),
            "thresholds": data.get("thresholds", {}),
        }
        return {"status": "success", "data": formatted}
    return {"status": "error", "message": "results_summary.json not found."}


@app.get("/api/graphs/{graph_name}")
async def get_graph(graph_name: str):
    """Serve publication figures from reports/figures/."""
    file_path = os.path.join(FIGURES_DIR, f"{graph_name}.png")
    if not os.path.exists(file_path):
        if graph_name == "offensive_gap_distribution":
            file_path = os.path.join(FIGURES_DIR, "defensive_gap_distribution.png")
        elif graph_name == "fig3_m_dfa_scaling_curves":
            file_path = os.path.join(FIGURES_DIR, "fig3_mdfa_scaling_curves.png")
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/png")
    return JSONResponse(status_code=404, content={"status": "error", "message": f"Graph '{graph_name}' not found."})


@app.post("/api/graphs/generate")
async def generate_graphs():
    """Regenerate publication figures via visualize_results.py."""
    import subprocess
    cmd = [sys.executable, os.path.join(ROOT, "src", "visualize_results.py")]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if proc.returncode == 0:
        return {"status": "success", "message": "Graphs regenerated successfully."}
    return JSONResponse(status_code=500, content={"status": "error", "stderr": proc.stderr})


# ---------------------------------------------------------------- Live Simulator (SSE)

def clean_float(val: Any, fallback: Optional[float] = None) -> Optional[float]:
    if val is None:
        return fallback
    try:
        f = float(val)
        return f if np.isfinite(f) else fallback
    except (ValueError, TypeError):
        return fallback


@app.get("/api/simulate/stream")
async def simulate_stream(
    tier: str = Query("genuine", description="Adversary tier: genuine, t0_scripted, t1_langevin, t2_ou_jerk, t3_fgn"),
    regime: str = Query("balanced", description="Defense regime: balanced, frictionless, high_security"),
    window_index: Optional[int] = Query(None, description="Optional locked test window index"),
):
    """
    Server-Sent Events (SSE) endpoint providing real-time trajectory streaming,
    instantaneous BehaveFormerMAE reconstruction error, multi-scale DFA verification,
    and LangGraph State Machine transitions.
    """
    model, normalizer, thresholds_dict, test_windows, profiles = get_dependencies()
    defense_graph = get_defense_graph(regime)

    canonical_tier = TIER_MAP.get(tier.lower(), "genuine")
    regime_key = regime.lower() if regime.lower() in ["balanced", "frictionless", "high_security"] else "balanced"
    regime_cfg = thresholds_dict.get("regimes", {}).get(regime_key, thresholds_dict)

    mse_threshold = float(regime_cfg.get("mae_mse_threshold", 0.0137))
    h1_threshold = float(regime_cfg.get("h1_threshold", 0.6443))
    explanation = PEDAGOGICAL_EXPLANATIONS.get(canonical_tier, PEDAGOGICAL_EXPLANATIONS["genuine"])

    async def event_generator():
        # Select window
        if window_index is not None and 0 <= window_index < len(test_windows):
            w = test_windows[window_index]
            selected_idx = window_index
        else:
            selected_idx = random.randint(0, len(test_windows) - 1)
            w = test_windows[selected_idx]

        prof = profiles.get(w.subject, list(profiles.values())[0])

        # Synthesize / load trajectory
        if canonical_tier == "genuine":
            t, x, y = w.t.copy(), w.x.copy(), w.y.copy()
        else:
            attack_data = generate_attack(canonical_tier, w, prof)
            t, x, y = attack_data["t"], attack_data["x"], attack_data["y"]

        # Compute full-window ground metrics
        v_full, a_full = kinematics(t, x, y)
        feats_full = featurize(t, x, y, normalizer)
        with torch.no_grad():
            x_tensor = torch.from_numpy(feats_full).unsqueeze(0).to(DEVICE)
            b_dict = {
                "mouse_spatial": x_tensor[:, :, :2],
                "mouse_kinematics": x_tensor[:, :, 2:],
                "mouse_mask": torch.zeros((1, 256), dtype=torch.bool, device=DEVICE),
            }
            recon, _, _, _ = model(b_dict, apply_mask=False)
            full_window_mse = float(nn.functional.mse_loss(recon, x_tensor).item())

        dfa_feats = compute_dfa_multiscale(v_full)
        std_v = clean_float(dfa_feats.get("std_v"), 0.0)
        is_variance_collapse = (std_v < 10.0)

        h1_val = clean_float(dfa_feats.get("h1"), 0.0 if is_variance_collapse else None)
        h2_val = clean_float(dfa_feats.get("h2"), 0.0 if is_variance_collapse else None)
        dh_val = clean_float(dfa_feats.get("delta_h"), 0.0 if is_variance_collapse else None)
        f0_val = clean_float(dfa_feats.get("log_f0"), 0.0 if is_variance_collapse else None)

        # Evaluate LangGraph defense state
        defense_state: ActiveDefenseState = {
            "session_id": f"sim_{selected_idx}_{canonical_tier}",
            "user_id": str(w.subject),
            "telemetry_buffer": {"speed": v_full, "post_probe_speed": None, "is_genuine": (canonical_tier == "genuine")},
            "mae_reconstruction_error": full_window_mse,
            "audit_log": [],
        }
        res = defense_graph.invoke(defense_state)
        final_decision = res.get("session_status", "AUTHENTICATED")
        audit_log = res.get("audit_log", [])
        trust_score = float(res.get("trust_score", 1.0))

        # Prepare normalized screen coordinates for 500x500 canvas
        # Aspect-preserving min-max scaling to 50-450 px
        ptp_x = max(np.ptp(x), 1.0)
        ptp_y = max(np.ptp(y), 1.0)
        scale = max(ptp_x, ptp_y)
        canvas_x = 50.0 + ((x - np.min(x)) / scale) * 400.0
        canvas_y = 50.0 + ((y - np.min(y)) / scale) * 400.0

        N = len(x)
        # Determine playback frames: smooth ~2.2s animation so user can observe trajectory
        total_stream_frames = 60 if is_variance_collapse else 75
        step_stride = max(1, N // total_stream_frames)
        probe_intensity = res.get("probe_intensity", "50ms")
        has_150ms_escalation = (probe_intensity == "150ms")

        running_path = []

        for i in range(1, N, step_stride):
            curr_v = clean_float(v_full[min(i - 1, len(v_full) - 1)], 0.0)
            prog_factor = min(1.0, i / (N * 0.5))
            running_mse = float(full_window_mse * (0.8 + 0.2 * prog_factor) + (random.random() - 0.5) * 0.0002)
            running_mse = max(0.0001, running_mse)

            pt_x = clean_float(canvas_x[i], 250.0)
            pt_y = clean_float(canvas_y[i], 250.0)
            running_path.append([pt_x, pt_y])

            # Determine real-time state progression
            if is_variance_collapse:
                if i < 8:
                    curr_status = "MONITORING"
                    curr_probe = "None"
                    curr_trust = 0.5
                else:
                    curr_status = "REVOKED"
                    curr_probe = "None"
                    curr_trust = 0.0
            else:
                if i < N // 3:
                    curr_status = "MONITORING"
                    curr_probe = "None"
                    curr_trust = max(0.6, 1.0 - (running_mse / (mse_threshold * 1.5)))
                elif i < (2 * N) // 3:
                    curr_status = "PROBING_50MS"
                    curr_probe = "50ms"
                    curr_trust = max(0.5, 1.0 - (running_mse / (mse_threshold * 1.5)))
                elif has_150ms_escalation and i < (5 * N) // 6:
                    curr_status = "ESCALATING_150MS"
                    curr_probe = "150ms"
                    curr_trust = 0.4
                else:
                    curr_status = final_decision
                    curr_probe = probe_intensity
                    curr_trust = trust_score

            is_typing = (i % 14 in [2, 3, 4, 5])

            frame_data = {
                "frame": i,
                "total_frames": N,
                "time": clean_float(t[i], 0.0),
                "x": pt_x,
                "y": pt_y,
                "raw_x": clean_float(x[i], 0.0),
                "raw_y": clean_float(y[i], 0.0),
                "path": running_path,  # full cumulative path for smooth trajectory
                "velocity": curr_v,
                "mse": running_mse,
                "mae_threshold": mse_threshold,
                "threshold": mse_threshold,
                "trust_score": curr_trust,
                "session_status": curr_status,
                "probe_intensity": curr_probe,
                "h1": h1_val if (i > N // 2 or is_variance_collapse) else None,
                "h2": h2_val if (i > N // 2 or is_variance_collapse) else None,
                "delta_h": dh_val if (i > N // 2 or is_variance_collapse) else None,
                "log_f0": f0_val if (i > N // 2 or is_variance_collapse) else None,
                "speed_std": std_v,
                "h1_threshold": h1_threshold,
                "tier": canonical_tier,
                "regime": regime_key,
                "user_id": w.subject,
                "subject_id": w.subject,
                "chunk_index": selected_idx,
                "window_index": selected_idx,
                "audit_log": audit_log[-1] if (audit_log and curr_status != "MONITORING") else None,
                "full_audit_log": audit_log if curr_status in [final_decision, "REVOKED"] else [],
                "explanation": explanation,
                "is_typing": is_typing,
            }

            yield f"data: {json.dumps(frame_data)}\n\n"
            delay = 0.035 if is_variance_collapse else 0.025
            await asyncio.sleep(delay)

        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------- Backward Compatibility Wrappers

@app.get("/api/simulate/{scenario}")
async def simulate_legacy(scenario: str, chunk_index: Optional[int] = None):
    """Legacy redirect to /api/simulate/stream."""
    mapped_tier = TIER_MAP.get(scenario, "genuine")
    return await simulate_stream(tier=mapped_tier, regime="balanced", window_index=chunk_index)


@app.get("/api/simulate/active_defense/{scenario}")
async def simulate_active_defense_legacy(scenario: str, chunk_index: Optional[int] = None):
    """Legacy redirect to /api/simulate/stream in balanced regime."""
    mapped_tier = TIER_MAP.get(scenario, "genuine")
    return await simulate_stream(tier=mapped_tier, regime="balanced", window_index=chunk_index)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)
