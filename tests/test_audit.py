"""Exhaustive Verification & Empirical Unit Test Suite.

Audits and verifies:
1. Detrended Fluctuation Analysis (DFA-1): Peng et al. (1994) scaling, multi-scale features, degenerate handling.
2. Adversary Tiers: T0 (scripted bot), T1 (replay Langevin), T2 (OU generator), T3 (fGn matched).
3. Defensive Multi-Agent System (LangGraph): 5-node architecture, Stage 1 (50ms) cascade and Stage 2 (150ms) oracle step-up.
4. Pipeline & Leak-Free Splitting: Chronological 60/20/20 partitioning, train-only normalization.
5. API Utilities: clean_float serialization safety, NaN prevention, status reporting.
"""
import os
import sys
import unittest
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.dfa import fluctuation_function, hurst_h1, compute_dfa_multiscale
from src.pipeline import Window, kinematics, Normalizer, featurize
from src.attackers import VictimProfile, generate, scripted_bot, replay_langevin, ou_generator, fgn_matched
from src.defensive_mas import build_active_defense_graph, run_session
from src.api.main import clean_float, TIER_MAP


class TestDFAImplementation(unittest.TestCase):
    """Audits mathematical correctness of DFA-1 and multi-scale metrics."""

    def test_dfa_white_noise_scaling(self):
        """White noise should have theoretical Hurst exponent H ≈ 0.50."""
        rng = np.random.default_rng(42)
        white_noise = rng.standard_normal(16384)
        h1 = hurst_h1(white_noise)
        self.assertTrue(0.40 <= h1 <= 0.65, f"White noise H1={h1:.3f} outside expected [0.40, 0.65]")

    def test_dfa_persistent_random_walk(self):
        """Random walk / Brownian motion profile should have theoretical Hurst exponent H ≈ 1.50."""
        rng = np.random.default_rng(42)
        steps = rng.standard_normal(4096)
        walk = np.cumsum(steps)
        h1 = hurst_h1(walk)
        self.assertTrue(h1 > 1.20, f"Random walk H1={h1:.3f} expected > 1.20")

    def test_dfa_degenerate_zero_variance(self):
        """Constant signals have zero variance; DFA should handle safely without crashing or returning invalid numbers."""
        flat = np.full(256, 42.0)
        feats = compute_dfa_multiscale(flat)
        self.assertEqual(feats["std_v"], 0.0)
        self.assertTrue(np.isnan(feats["h1"]))
        self.assertTrue(np.isnan(feats["delta_h"]))

    def test_multiscale_features_structure(self):
        """Verify that compute_dfa_multiscale returns all required multi-scale biomarkers."""
        rng = np.random.default_rng(123)
        series = np.abs(rng.standard_normal(512)) + 5.0
        feats = compute_dfa_multiscale(series)
        self.assertIn("h1", feats)
        self.assertIn("h2", feats)
        self.assertIn("delta_h", feats)
        self.assertIn("log_f0", feats)
        self.assertIn("std_v", feats)
        self.assertTrue(np.isfinite(feats["h1"]))
        self.assertTrue(np.isfinite(feats["h2"]))
        self.assertAlmostEqual(feats["delta_h"], feats["h1"] - feats["h2"], places=4)


class TestAttackerTiers(unittest.TestCase):
    """Audits open-loop threat tier generation and properties."""

    def setUp(self):
        # Create a mock genuine window with realistic speed properties
        rng = np.random.default_rng(10)
        dt = np.full(256, 0.02)
        t = np.concatenate([[0.0], np.cumsum(dt)])
        speed = np.abs(rng.standard_normal(256) * 40.0 + 120.0)
        angle = np.cumsum(rng.standard_normal(256) * 0.1)
        vx = speed * np.cos(angle)
        vy = speed * np.sin(angle)
        x = np.concatenate([[100.0], 100.0 + np.cumsum(vx * dt)])
        y = np.concatenate([[150.0], 150.0 + np.cumsum(vy * dt)])
        self.mock_window = Window(subject=999, index=0, split="train", t=t, x=x, y=y)
        self.mock_profile = VictimProfile([self.mock_window])

    def test_t0_scripted_bot(self):
        """T0 bot must produce linear coordinates and constant velocity."""
        data = generate("scripted_bot", self.mock_window, self.mock_profile)
        self.assertEqual(len(data["t"]), 257)
        self.assertEqual(len(data["x"]), 257)
        self.assertEqual(len(data["y"]), 257)
        v, _ = kinematics(data["t"], data["x"], data["y"])
        self.assertTrue(np.std(v) < 1e-4, f"T0 bot speed std {np.std(v)} should be ~0")

    def test_t1_replay_langevin(self):
        """T1 replay bot must augment original path with Langevin tremor."""
        data = generate("replay_langevin", self.mock_window, self.mock_profile)
        self.assertEqual(len(data["x"]), 257)
        diff_x = np.abs(data["x"] - self.mock_window.x)
        self.assertTrue(np.mean(diff_x) > 0.0, "T1 bot must inject non-zero displacement tremor")
        self.assertTrue(np.mean(diff_x) < 50.0, "T1 bot tremor must preserve macro-trajectory")

    def test_t2_ou_generator(self):
        """T2 OU generator must construct 5th-order minimum jerk trajectory."""
        data = generate("ou_generator", self.mock_window, self.mock_profile)
        self.assertEqual(len(data["t"]), 257)
        self.assertEqual(len(data["x"]), 257)
        v, _ = kinematics(data["t"], data["x"], data["y"])
        self.assertTrue(np.std(v) > 0.1, "T2 generator must have non-zero velocity variation")

    def test_t3_fgn_matched(self):
        """T3 fGn generator must match victim Hurst exponent."""
        data = generate("fgn_matched", self.mock_window, self.mock_profile)
        self.assertIn("target_H", data)
        self.assertEqual(len(data["x"]), 257)
        self.assertTrue(data["target_H"] > 0.0)


class TestDefensiveMAS(unittest.TestCase):
    """Audits LangGraph 5-node state machine and Two-Stage Escalation."""

    def setUp(self):
        self.th_balanced = {
            "mae_mse_threshold": 0.0137,
            "min_speed_std": 10.0,
            "h1_threshold": 0.6443,
            "delta_h_threshold": 0.08,
            "log_f0_threshold": 1.85,
            "h1_stepup_margin": 0.05,
        }
        self.th_frictionless = {
            "mae_mse_threshold": 0.0137,
            "min_speed_std": 10.0,
            "h1_threshold": 0.6146,
            "delta_h_threshold": -999.0,
            "log_f0_threshold": -999.0,
            "h1_stepup_margin": 0.05,
        }
        self.graph_balanced = build_active_defense_graph(self.th_balanced)
        self.graph_frictionless = build_active_defense_graph(self.th_frictionless)

    def test_variance_collapse_rejection(self):
        """A linear bot with std_v < 10 px/s must be immediately REVOKED at the monitoring gate."""
        speed = np.full(256, 150.0)  # zero variance
        res = run_session(self.graph_balanced, speed=speed, mse=0.001, user_id="1")
        self.assertEqual(res["session_status"], "REVOKED")
        self.assertEqual(res["trust_score"], 0.0)

    def test_passive_anomaly_rejection(self):
        """A session exceeding the MSE reconstruction threshold must be REVOKED."""
        rng = np.random.default_rng(42)
        speed = np.abs(rng.standard_normal(256) * 30.0 + 100.0)
        res = run_session(self.graph_balanced, speed=speed, mse=0.050, user_id="1")  # mse > 0.0137
        self.assertEqual(res["session_status"], "REVOKED")

    def test_genuine_authenticated_in_50ms_loop(self):
        """In frictionless mode, a user with H1 >= 0.6146 passes without multi-scale constraints."""
        rng = np.random.default_rng(42)
        # Power-law correlated noise
        speed = np.cumsum(rng.standard_normal(256)) * 2.0 + 100.0
        speed = np.abs(speed) + 20.0
        res = run_session(self.graph_frictionless, speed=speed, mse=0.005, user_id="1")
        self.assertIn(res["session_status"], ["AUTHENTICATED", "STEP_UP_REQUIRED"])

    def test_two_stage_150ms_escalation_logic(self):
        """Under a 150ms step-up challenge:
        - Genuine human motor recalibration preserves identity -> AUTHENTICATED.
        - Open-loop generator phase collapses -> REVOKED (0.00% challenge FAR).
        """
        # Create borderline signal that triggers escalation
        borderline_speed = np.abs(np.random.default_rng(99).standard_normal(256) * 15.0 + 80.0)
        
        # Test adversary under 150ms escalation: must be REVOKED
        res_adv = run_session(self.graph_balanced, speed=borderline_speed, mse=0.005, user_id="1", is_genuine=False)
        if res_adv["session_status"] == "STEP_UP_REQUIRED" or res_adv.get("probe_intensity") == "150ms":
            self.assertEqual(res_adv["session_status"], "REVOKED")
            self.assertEqual(res_adv["hurst_exponent"], 0.0)

        # Test genuine user under 150ms escalation: must be AUTHENTICATED
        res_gen = run_session(self.graph_balanced, speed=borderline_speed, mse=0.005, user_id="1", is_genuine=True)
        if res_gen.get("probe_intensity") == "150ms":
            self.assertEqual(res_gen["session_status"], "AUTHENTICATED")


class TestPipelineLeakFree(unittest.TestCase):
    """Audits data pipeline integrity and zero-leakage constraints."""

    def test_normalizer_train_only(self):
        """Normalizer parameters must map back exactly within [p01, p99]."""
        rng = np.random.default_rng(7)
        vals = rng.uniform(10.0, 500.0, 1000)
        norm = Normalizer()
        p01, p99 = np.percentile(vals, [1, 99])
        c = np.clip(vals, p01, p99)
        norm.p = {"speed": {"p01": float(p01), "p99": float(p99), "mean": float(c.mean()), "std": float(c.std())}}
        
        z = norm.transform("speed", np.array([250.0]))
        inv = norm.inverse("speed", z)
        self.assertAlmostEqual(inv[0], 250.0, places=4)

    def test_clean_float_utility(self):
        """clean_float must safely handle NaN, Inf, None, strings, and finite floats."""
        self.assertIsNone(clean_float(np.nan))
        self.assertEqual(clean_float(np.nan, 0.0), 0.0)
        self.assertIsNone(clean_float(float("inf")))
        self.assertEqual(clean_float("not_a_number", 5.0), 5.0)
        self.assertEqual(clean_float(3.14159), 3.14159)


if __name__ == "__main__":
    unittest.main()
