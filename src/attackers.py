"""Open-loop attacker tiers for the BB-MAS mouse experiments.

Every attacker returns raw events (t, x, y) with 257 samples, which then pass
through exactly the same feature code (`pipeline.featurize`) and DFA code as
genuine windows. Detectors never receive the class label.

Each attacker only uses information an adversary could plausibly hold:
  T0 scripted_bot     : nothing about the victim except the window extent.
  T1 replay_langevin  : a stolen copy of the victim's recorded trajectory
                        (the test window itself) plus Langevin tremor.
                        This is the strongest tier and the one the old code
                        called the "biological deepfake".
  T2 ou_generator     : victim's *training-split* positions and inter-event
                        intervals only; minimum-jerk movements between
                        sampled targets plus Ornstein-Uhlenbeck tremor.
  T3 fgn_matched      : victim's *training-split* speed statistics and DFA
                        exponents; speed synthesised from fractional Gaussian
                        noise whose Hurst parameter is drawn from the
                        victim's genuine training H1 values (noise-tuned
                        attacker aimed at static-H1 rules).

All tiers are open-loop: none of them can observe or react to a probe.
"""
from typing import Dict, List

import numpy as np

from src.dfa import hurst_h1
from src.pipeline import WINDOW_EVENTS, Window, kinematics

TIERS = ["scripted_bot", "replay_langevin", "ou_generator", "fgn_matched"]


def _rng(subject: int, index: int, tier: str, split: str) -> np.random.Generator:
    key = f"{subject}|{index}|{tier}|{split}".encode()
    return np.random.default_rng(int.from_bytes(key, "little") % (2 ** 63))


# ---------------------------------------------------------------- helpers
def ou_tremor(t: np.ndarray, x_star: np.ndarray, rng, theta=200.0, sigma0=7.9, kappa=1.0):
    """Exact-discretisation OU process around an intended path x_star.

    dX = theta (x*(t) - X) dt + sigma(t) dW,  sigma(t) = sigma0 (1 + kappa t / T).
    Units: theta [1/s], sigma0 [px / sqrt(s)]. Values follow the conference
    paper (0.2 /ms, 0.25 px/sqrt(ms)); kappa doubles the amplitude across the
    window to model within-window fatigue drift.
    """
    T = max(t[-1], 1e-9)
    dev = np.zeros_like(x_star)
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        decay = np.exp(-theta * dt)
        sig = sigma0 * (1.0 + kappa * t[k] / T)
        sd = sig * np.sqrt((1.0 - decay ** 2) / (2.0 * theta))
        dev[k] = dev[k - 1] * decay + sd * rng.standard_normal()
    return x_star + dev


def fgn_davies_harte(n: int, H: float, rng) -> np.ndarray:
    """Exact fractional Gaussian noise of length n (unit variance)."""
    H = float(np.clip(H, 0.05, 0.95))
    k = np.arange(n + 1)
    gamma = 0.5 * (np.abs(k - 1) ** (2 * H) - 2 * np.abs(k) ** (2 * H) + np.abs(k + 1) ** (2 * H))
    row = np.concatenate([gamma, gamma[-2:0:-1]])
    lam = np.fft.fft(row).real
    lam[lam < 0] = 0.0
    m = len(row)
    w = rng.standard_normal(m) + 1j * rng.standard_normal(m)
    z = np.fft.fft(np.sqrt(lam / m) * w)[:n].real
    return (z - z.mean()) / (z.std() or 1.0)


class VictimProfile:
    """Per-subject statistics computed from that subject's TRAIN windows only."""

    def __init__(self, windows: List[Window]):
        self.dts = np.concatenate([np.diff(w.t) for w in windows])
        self.pos = np.concatenate([np.stack([w.x, w.y], 1) for w in windows])
        logv = []
        self.h1 = []
        for w in windows:
            v, _ = kinematics(w.t, w.x, w.y)
            logv.append(np.log(v + 1.0))
            h = hurst_h1(v)
            if np.isfinite(h):
                self.h1.append(h)
        logv = np.concatenate(logv)
        self.logv_mean, self.logv_std = float(logv.mean()), float(logv.std())
        self.h1 = np.array(self.h1) if self.h1 else np.array([0.7])


# ---------------------------------------------------------------- tiers
def scripted_bot(w: Window, prof: VictimProfile, rng) -> Dict[str, np.ndarray]:
    t = np.linspace(0.0, max(float(w.t[-1]), 1.0), WINDOW_EVENTS)
    dx = float(w.x[-1] - w.x[0])
    dy = float(w.y[-1] - w.y[0])
    if np.hypot(dx, dy) < 20.0:
        span_x = float(max(np.ptp(w.x), 180.0))
        span_y = float(max(np.ptp(w.y), 120.0))
        x = np.linspace(float(w.x[0]), float(w.x[0] + span_x), WINDOW_EVENTS)
        y = np.linspace(float(w.y[0]), float(w.y[0] + span_y), WINDOW_EVENTS)
    else:
        x = np.linspace(float(w.x[0]), float(w.x[-1]), WINDOW_EVENTS)
        y = np.linspace(float(w.y[0]), float(w.y[-1]), WINDOW_EVENTS)
    return {"t": t, "x": x, "y": y}


def replay_langevin(w: Window, prof: VictimProfile, rng) -> Dict[str, np.ndarray]:
    x = ou_tremor(w.t, w.x.astype(float), rng)
    y = ou_tremor(w.t, w.y.astype(float), rng)
    return {"t": w.t.copy(), "x": x, "y": y}


def ou_generator(w: Window, prof: VictimProfile, rng) -> Dict[str, np.ndarray]:
    dt = rng.choice(prof.dts, size=WINDOW_EVENTS - 1, replace=True)
    t = np.concatenate([[0.0], np.cumsum(dt)])
    xs, ys = np.empty(WINDOW_EVENTS), np.empty(WINDOW_EVENTS)
    k = 0
    cur = prof.pos[rng.integers(len(prof.pos))]
    while k < WINDOW_EVENTS:
        length = int(rng.integers(15, 60))
        tgt = prof.pos[rng.integers(len(prof.pos))]
        tau = np.linspace(0, 1, length, endpoint=False)
        s = 10 * tau ** 3 - 15 * tau ** 4 + 6 * tau ** 5  # minimum-jerk profile
        seg = cur[None, :] + (tgt - cur)[None, :] * s[:, None]
        m = min(length, WINDOW_EVENTS - k)
        xs[k:k + m], ys[k:k + m] = seg[:m, 0], seg[:m, 1]
        k += m
        cur = tgt
    return {"t": t, "x": ou_tremor(t, xs, rng), "y": ou_tremor(t, ys, rng)}


def fgn_matched(w: Window, prof: VictimProfile, rng) -> Dict[str, np.ndarray]:
    n = WINDOW_EVENTS - 1
    H = float(rng.choice(prof.h1))
    z = fgn_davies_harte(n, H, rng)
    v = np.exp(prof.logv_mean + prof.logv_std * z) - 1.0
    v = np.clip(v, 0.0, None)
    dt = rng.choice(prof.dts, size=n, replace=True)
    heading = np.cumsum(rng.normal(0, 0.15, n)) + rng.uniform(0, 2 * np.pi)
    step = v * dt
    x = np.concatenate([[w.x[0]], w.x[0] + np.cumsum(step * np.cos(heading))])
    y = np.concatenate([[w.y[0]], w.y[0] + np.cumsum(step * np.sin(heading))])
    t = np.concatenate([[0.0], np.cumsum(dt)])
    return {"t": t, "x": x, "y": y, "target_H": H}


GENERATORS = {
    "scripted_bot": scripted_bot,
    "replay_langevin": replay_langevin,
    "ou_generator": ou_generator,
    "fgn_matched": fgn_matched,
}


def generate(tier: str, w: Window, prof: VictimProfile) -> Dict[str, np.ndarray]:
    rng = _rng(w.subject, w.index, tier, w.split)
    return GENERATORS[tier](w, prof, rng)
