"""Detrended Fluctuation Analysis (DFA-1) for cursor-speed series.

Single source of truth for the scaling exponent used everywhere in the
repository (experiments, the defense graph, figures). The estimator follows
Peng et al. (1994): integrate the mean-removed series, split the profile into
non-overlapping windows of size s, remove a least-squares linear trend in each
window, and regress log F(s) on log s.

The short-range exponent H1 is fitted over s in [4, 16] samples, as specified
in the conference paper. When the series has (numerically) zero variance, DFA
is undefined and NaN is returned instead of a placeholder value.
"""
from typing import Dict, Optional

import numpy as np

SCALES_H1 = np.arange(4, 17)  # s = 4..16 inclusive


def fluctuation_function(series: np.ndarray, scales: np.ndarray = SCALES_H1) -> np.ndarray:
    """Return F(s) for every s in `scales` (NaN where undefined)."""
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    out = np.full(len(scales), np.nan)
    if len(x) < 2 * int(np.max(scales)) or np.std(x) < 1e-12:
        return out

    profile = np.cumsum(x - x.mean())
    for i, s in enumerate(scales):
        s = int(s)
        n_seg = len(profile) // s
        if n_seg < 2:
            continue
        segs = profile[: n_seg * s].reshape(n_seg, s)
        t = np.arange(s, dtype=np.float64)
        # Vectorised least-squares linear detrend of every segment.
        t_c = t - t.mean()
        slope = (segs - segs.mean(axis=1, keepdims=True)) @ t_c / (t_c @ t_c)
        trend = segs.mean(axis=1, keepdims=True) + slope[:, None] * t_c[None, :]
        rms2 = np.mean((segs - trend) ** 2)
        out[i] = np.sqrt(rms2) if rms2 > 0 else np.nan
    return out


def hurst_h1(series: np.ndarray, scales: np.ndarray = SCALES_H1) -> float:
    """Short-range DFA exponent H1 (slope of log F vs log s); NaN if undefined."""
    F = fluctuation_function(series, scales)
    ok = np.isfinite(F) & (F > 0)
    if ok.sum() < 3:
        return float("nan")
    slope, _ = np.polyfit(np.log(scales[ok]), np.log(F[ok]), 1)
    return float(slope)


def compute_dfa_hurst(velocity_series: np.ndarray, scale_min: int = 4,
                      scale_max: Optional[int] = None) -> Dict[str, float]:
    """Backward-compatible wrapper returning {'h1', 'h2'}.

    h1 uses s in [4, 16]; h2 uses s in [17, N//4] when the series is long enough.
    """
    x = np.asarray(velocity_series, dtype=np.float64)
    h1 = hurst_h1(x)
    if scale_max is None:
        scale_max = len(x) // 4
    h2 = float("nan")
    if scale_max > 17:
        h2 = hurst_h1(x, np.arange(17, scale_max + 1))
    return {"h1": h1, "h2": h2}


def compute_dfa_multiscale(series: np.ndarray, scales: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute multi-scale DFA features: H1 (4..16), H2 (17..64), delta_H (crossover gap), log_F0, and std_v."""
    if scales is None:
        scales = np.arange(4, 65)
    x = np.asarray(series, dtype=np.float64)
    x = x[np.isfinite(x)]
    std_v = float(np.std(x)) if len(x) > 0 else 0.0
    if len(x) < 32 or std_v < 1e-6:
        return {
            "h1": float("nan"),
            "h2": float("nan"),
            "delta_h": float("nan"),
            "log_f0": float("nan"),
            "std_v": std_v,
        }
    fluc = fluctuation_function(x, scales)
    s1_mask = (scales >= 4) & (scales <= 16)
    s2_mask = (scales >= 17) & (scales <= 64)

    ok1 = np.isfinite(fluc[s1_mask]) & (fluc[s1_mask] > 0)
    ok2 = np.isfinite(fluc[s2_mask]) & (fluc[s2_mask] > 0)

    h1 = float(np.polyfit(np.log(scales[s1_mask][ok1]), np.log(fluc[s1_mask][ok1]), 1)[0]) if ok1.sum() >= 3 else float("nan")
    h2 = float(np.polyfit(np.log(scales[s2_mask][ok2]), np.log(fluc[s2_mask][ok2]), 1)[0]) if ok2.sum() >= 3 else float("nan")
    delta_h = float(h1 - h2) if (np.isfinite(h1) and np.isfinite(h2)) else float("nan")
    log_f0 = float(np.log10(fluc[0])) if (np.isfinite(fluc[0]) and fluc[0] > 0) else float("nan")

    return {
        "h1": h1,
        "h2": h2,
        "delta_h": delta_h,
        "log_f0": log_f0,
        "std_v": std_v,
    }


if __name__ == "__main__":
    # Self-check against series with known scaling.
    rng = np.random.default_rng(0)
    white = rng.standard_normal(4096)
    walk_increments_of_walk = np.cumsum(rng.standard_normal(4096))
    print(f"white noise  H1 = {hurst_h1(white):.3f}  (theory 0.5)")
    print(f"random walk  H1 = {hurst_h1(walk_increments_of_walk):.3f}  (theory 1.5)")
    print(f"constant     H1 = {hurst_h1(np.ones(256))}  (undefined -> nan)")
