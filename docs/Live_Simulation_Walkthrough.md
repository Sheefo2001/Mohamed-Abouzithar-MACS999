# Live Simulation Walkthrough: Active Neuromuscular Defense Architecture

This document provides a comprehensive, step-by-step walkthrough of the live web application simulation. It details the underlying mathematical mechanics, the visual outputs, and the academic novelty across each threat tier and operational defense regime.

---

## 1. Environment & Setup

- **Frontend Interface**: React + Vite SPA running on `localhost:5173`.
- **Backend API**: FastAPI running on `localhost:8000` (`src/api/main.py`), utilizing PyTorch (CUDA acceleration if available) and LangGraph.
- **Starting the System**:
  ```powershell
  # Launch backend and frontend simultaneously:
  .\start_dashboard.ps1
  ```
  Or independently:
  ```powershell
  uvicorn src.api.main:app --reload --port 8000
  npm run dev --prefix webapp
  ```

---

## 2. Interactive Simulator Elements & Controls

### A. Operational Regime Selector
Deployers can dynamically toggle the LangGraph agent between three validation-calibrated operating points:
1. **Balanced Mode (Recommended for Enterprise)**:
   - $H_1^* = 0.6443, \Delta H \ge 0.08, \log F_0 \ge 1.85, \sigma_v \ge 10.0\text{ px/s}$.
   - Optimal trade-off: **28.1% Genuine FRR** vs. **32.54% Net Adversary FAR** (50.0% relative threat reduction).
2. **Frictionless Mode (Low-Friction Portals)**:
   - $H_1^* = 0.6146, \sigma_v \ge 10.0\text{ px/s}$.
   - Minimizes human false alarms: **18.6% Genuine FRR**, while eliminating naive scripted bots (0.0% FAR).
3. **High-Security Mode (EER Operating Point)**:
   - $H_1^* = 0.9058, \Delta H \ge 0.12, \log F_0 \ge 2.00, \sigma_v \ge 10.0\text{ px/s}$.
   - Zero-tolerance for financial / critical infrastructure: **15.81% Net Adversary FAR** (T0: 0.00%, T1: 41.74%, T2: 9.92%, T3: 11.57%).

### B. Target Lock (Replay Proofing)
- **OFF**: Each simulation run samples a random human test window from the 92 BB-MAS subjects.
- **LOCKED**: Locks the current subject ID and window index. This allows the evaluator to test how different attack tiers attempt to forge or impersonate the *exact same human user session*.

---

## 3. The 4 Adversary Threat Tiers (Under the Hood)

### Scenario A: Genuine Human (BB-MAS Benchmark)
- **Physiological Source**: Actual recorded mouse trajectory from the SU-AIS BB-MAS dataset (held-out 20% chronological test split).
- **Passive Model (`BehaveFormerMAE`)**: Reconstructs trajectory with low Mean Squared Error ($\text{MSE} \approx 0.0048 &lt; 0.0137$).
- **Active DFA MAS**: Computes $H_1 \approx 0.86$, verifying biological sub-conscious persistence ($H_1 \ge 0.6443$) and natural multi-scale crossover ($\Delta H \ge 0.08$). Status: `AUTHENTICATED`.

---

### Tier 0: Scripted Bot (Linear Interpolation)
- **Mathematical Model**: Straight-line spatial interpolation connecting waypoints with constant velocity and zero jerk.
- **Passive Reaction**: Fails kinematic variance gate ($\sigma_v = 0.0\text{ px/s} &lt; 10.0\text{ px/s}$).
- **Active DFA MAS**: Scaling exponent is undefined (flatline signal). Immediate session revocation with zero delay. Status: `REVOKED` (**0.00% FAR**).

---

### Tier 1: Replay + Langevin SDE (The Offensive Gap)
- **Mathematical Model**: Stolen human base trajectory $x^*(t)$ perturbed by a first-order Ornstein-Uhlenbeck (overdamped Langevin) process on position:
  $$dX_t = \theta (x^*(t) - X_t)\,dt + \sigma(t)\,dW_t, \quad \sigma(t) = \sigma_0 \left(1 + \kappa \frac{t}{T}\right)$$
  with exact discretization ($\theta = 200\text{ s}^{-1}, \sigma_0 = 7.9\text{ px}/\sqrt{\text{s}}, \kappa = 1.0$).
- **Passive Vulnerability (The Offensive Gap)**: **92.56% False Acceptance Rate**. The passive autoencoder is completely fooled because the spatial curvature matches human patterns.
- **Active DFA MAS**: In Balanced Mode, FAR is **71.90%** (matching genuine acceptance); in High-Security Mode, FAR is reduced to **41.74%**.

---

### Tier 2: Minimum-Jerk Ornstein-Uhlenbeck Continuous Generator
- **Mathematical Model**: 5th-order minimum-jerk polynomial curves connecting sampled target points, perturbed by continuous Ornstein-Uhlenbeck positional tremor (`ou_tremor`).
- **Passive Vulnerability**: **100.00% False Acceptance Rate**. Passive autoencoders consider smooth minimum-jerk trajectories "ideal human movements".
- **Active DFA MAS**: Over-damped smoothness produces an abnormally elevated short-range scaling ($H_1 \approx 1.05$) and fails the biological crossover and fluctuation amplitude gates. FAR drops from **100.00% down to 24.79%** (and **9.92%** in High-Security). Status: `REVOKED`.

---

### Tier 3: Fractional Gaussian Noise (fGn) Noise-Matched Generator
- **Mathematical Model**: Adversary extracts victim's training-split Hurst exponents and synthesizes speed profiles via the Davies-Harte fGn method.
- **Passive Vulnerability**: **67.77% False Acceptance Rate** against passive monitoring.
- **Active DFA MAS**: Synthesized fractal noise lacks dynamic physiological coupling across time scales. Active gating cuts evasion to **33.47%** (Balanced) and **11.57%** (High-Security). Status: `REVOKED`.

---

## 4. Real-Time Telemetry & MAS Console

During session playback, the simulation updates four coordinated views:
1. **Kinematic Canvas (500x500)**: Renders normalized cursor movement with SVG polyline trajectory trail, color-coded theme cues, 50ms strobe glitch indicating active latency challenge injection, and 150ms supra-threshold step-up escalation alerts.
2. **DFA Telemetry Gauges**: Real-time numerical display of:
   - $H_1$ (Short-range visuomotor scale $s \in [4, 16]$)
   - $H_2$ (Long-range scale $s \in [17, 64]$)
   - Crossover Gap $\Delta H = H_1 - H_2 \ge 0.08$
   - Fluctuation Magnitude $\ell_F = \log_{10} F(4) \ge 1.85$
   - Speed Standard Deviation $\sigma_v$ (px/s, min $10.0\text{ px/s}$)
3. **Dual Real-Time Graphs**:
   - Top: Instantaneous reconstruction MSE vs. Anomaly Threshold.
   - Bottom: Instantaneous velocity profile.
4. **LangGraph Agent Audit Console**: Streams structured audit records detailing the mathematical justification for each decision (e.g. `[Passive check passed]`, `[Stage 1 Probe Requested (50ms)]`, `[Stage-2 Oracle Step-Up Resolution]`, `[Decision: REVOKED -> failed multi-scale neuromotor check]`).

---

## 5. Summary for Thesis Defense Presentation

When presenting the live simulation to advisors or examination committees:
1. **Demonstrate the Passive Collapse**: Run **Tier 1** in standard mode to show how the MSE remains below the zero-trust threshold, visually proving **The Offensive Gap**.
2. **Demonstrate the Active Solution**: Show how the LangGraph MAS transitions through `MONITORING` $\rightarrow$ `PROBING` $\rightarrow$ `DFA ANALYSIS` $\rightarrow$ `REVOCATION`, catching the deepfake using physics rather than black-box retraining.
3. **Show Usability Control**: Switch between **Balanced**, **Frictionless**, and **High-Security** modes to illustrate the empirical operational frontier.
