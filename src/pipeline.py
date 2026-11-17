"""BB-MAS desktop mouse pipeline: cleaning, windowing, splitting, normalisation.

Everything downstream (MAE training, attacker generation, DFA, baselines)
consumes the window records produced here, so every class of input goes
through exactly the same feature code.

Pipeline (per subject, `<id>_Mouse_Move.csv`):
  1. Parse absolute timestamps (``YYYY-MM-DD HH:MM:SS.fff``) to seconds.
  2. Sort by time, drop rows with missing coordinates, drop events whose
     timestamp does not advance (dt <= 0) so speed is always finite.
  3. Split the stream into segments at idle gaps > 30 s; windows never
     straddle a gap.
  4. Cut each segment into non-overlapping windows of 257 events, i.e.
     256 speed samples v_k = ||p_k - p_{k-1}|| / (t_k - t_{k-1})  [px/s].
  5. Chronological per-subject split of windows: first 60 % train, next
     20 % validation, last 20 % test. Subjects with < 3 windows are excluded.

Model features per window (256 x 4): per-window aspect-preserving min-max
positions, and speed / acceleration winsorised at the train-split 1st/99th
percentiles then z-scored with train-split statistics (fit on genuine train
windows only; no information from validation or test).
"""
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BBMAS_DIR = os.path.join(ROOT, "data", "raw", "bb_mas", "BB-MAS_Dataset")
NORMALIZER_PATH = os.path.join(ROOT, "checkpoints", "normalizer.json")

WINDOW_SPEEDS = 256
WINDOW_EVENTS = WINDOW_SPEEDS + 1
MAX_IDLE_S = 30.0
SPLIT_FRACTIONS = (0.6, 0.2, 0.2)
MIN_WINDOWS_PER_SUBJECT = 3


@dataclass
class Window:
    subject: int
    index: int                 # chronological index within the subject
    split: str                 # "train" | "val" | "test"
    t: np.ndarray              # (257,) seconds
    x: np.ndarray              # (257,) px
    y: np.ndarray              # (257,) px
    meta: Dict = field(default_factory=dict)


def kinematics(t: np.ndarray, x: np.ndarray, y: np.ndarray):
    """Speed (px/s) and acceleration (px/s^2), each of length len(t) - 1."""
    dt = np.diff(t)
    v = np.hypot(np.diff(x), np.diff(y)) / dt
    a = np.diff(v, prepend=v[0]) / dt
    return v, a


def load_subject_events(subject_dir: str, subject: int) -> Optional[pd.DataFrame]:
    path = os.path.join(subject_dir, f"{subject}_Mouse_Move.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, usecols=["pX", "pY", "time"])
    df["t"] = pd.to_datetime(df["time"], format="mixed", errors="coerce")
    df = df.dropna(subset=["t", "pX", "pY"]).sort_values("t", kind="stable")
    t = (df["t"] - df["t"].iloc[0]).dt.total_seconds().to_numpy()
    out = pd.DataFrame({"t": t, "x": df["pX"].to_numpy(float), "y": df["pY"].to_numpy(float)})
    keep = np.concatenate([[True], np.diff(out["t"].to_numpy()) > 0])
    return out[keep].reset_index(drop=True)


def segment_windows(ev: pd.DataFrame) -> List[Dict[str, np.ndarray]]:
    t, x, y = ev["t"].to_numpy(), ev["x"].to_numpy(), ev["y"].to_numpy()
    gaps = np.where(np.diff(t) > MAX_IDLE_S)[0] + 1
    bounds = np.concatenate([[0], gaps, [len(t)]])
    wins = []
    for s0, s1 in zip(bounds[:-1], bounds[1:]):
        n = (s1 - s0) // WINDOW_EVENTS
        for w in range(n):
            a = s0 + w * WINDOW_EVENTS
            b = a + WINDOW_EVENTS
            wins.append({"t": t[a:b] - t[a], "x": x[a:b], "y": y[a:b]})
    return wins


def build_windows(bbmas_dir: str = BBMAS_DIR) -> Dict:
    """Return {'windows': [Window], 'report': {...}} for the whole dataset."""
    subjects = sorted(int(d) for d in os.listdir(bbmas_dir) if d.isdigit())
    windows: List[Window] = []
    report = {"subjects_found": len(subjects), "missing_mouse_file": [],
              "excluded_lt_min_windows": {}, "events_raw": 0, "events_kept": 0}
    for sid in subjects:
        sdir = os.path.join(bbmas_dir, str(sid))
        path = os.path.join(sdir, f"{sid}_Mouse_Move.csv")
        if not os.path.exists(path):
            report["missing_mouse_file"].append(sid)
            continue
        report["events_raw"] += sum(1 for _ in open(path, encoding="utf-8")) - 1
        ev = load_subject_events(sdir, sid)
        report["events_kept"] += len(ev)
        wins = segment_windows(ev)
        if len(wins) < MIN_WINDOWS_PER_SUBJECT:
            report["excluded_lt_min_windows"][sid] = len(wins)
            continue
        n = len(wins)
        n_tr = int(np.floor(SPLIT_FRACTIONS[0] * n))
        n_va = int(np.floor(SPLIT_FRACTIONS[1] * n))
        n_va = max(n_va, 1)
        n_te = n - n_tr - n_va
        if n_te < 1:
            n_tr -= 1
            n_te = 1
        splits = ["train"] * n_tr + ["val"] * n_va + ["test"] * n_te
        for i, (w, sp) in enumerate(zip(wins, splits)):
            windows.append(Window(sid, i, sp, w["t"], w["x"], w["y"]))
    report["subjects_used"] = len({w.subject for w in windows})
    for sp in ("train", "val", "test"):
        report[f"windows_{sp}"] = sum(w.split == sp for w in windows)
    return {"windows": windows, "report": report}


class Normalizer:
    """Train-fitted winsorise + z-score for speed and acceleration."""

    def __init__(self, params: Optional[Dict] = None):
        self.p = params

    def fit(self, windows: List[Window]) -> "Normalizer":
        v_all, a_all = [], []
        for w in windows:
            v, a = kinematics(w.t, w.x, w.y)
            v_all.append(v)
            a_all.append(a)
        v_all, a_all = np.concatenate(v_all), np.concatenate(a_all)
        p = {}
        for name, arr in (("speed", v_all), ("accel", a_all)):
            lo, hi = np.percentile(arr, [1, 99])
            c = np.clip(arr, lo, hi)
            p[name] = {"p01": float(lo), "p99": float(hi),
                       "mean": float(c.mean()), "std": float(c.std() or 1.0)}
        self.p = p
        return self

    def transform(self, name: str, arr: np.ndarray) -> np.ndarray:
        q = self.p[name]
        return (np.clip(arr, q["p01"], q["p99"]) - q["mean"]) / q["std"]

    def inverse(self, name: str, z: np.ndarray) -> np.ndarray:
        """Map z-scores back to physical units (exact inside [p01, p99])."""
        q = self.p[name]
        return z * q["std"] + q["mean"]

    def save(self, path: str = NORMALIZER_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.p, f, indent=2)

    @classmethod
    def load(cls, path: str = NORMALIZER_PATH) -> "Normalizer":
        with open(path) as f:
            return cls(json.load(f))


def featurize(t: np.ndarray, x: np.ndarray, y: np.ndarray, norm: Normalizer) -> np.ndarray:
    """(257,) raw events -> (256, 4) float32 model input [x, y, speed_z, accel_z]."""
    v, a = kinematics(t, x, y)
    xs, ys = x[1:], y[1:]
    span = max(np.ptp(xs), np.ptp(ys), 1.0)
    px = (xs - xs.min()) / span
    py = (ys - ys.min()) / span
    feats = np.stack([px, py, norm.transform("speed", v), norm.transform("accel", a)], axis=1)
    return feats.astype(np.float32)


if __name__ == "__main__":
    data = build_windows()
    print(json.dumps({k: v for k, v in data["report"].items()}, indent=2, default=str))
