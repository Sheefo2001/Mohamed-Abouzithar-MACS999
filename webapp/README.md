# Continuous Behavioral Authentication Simulator — Web Front End

React + TypeScript + Vite single-page application for visualizing the passive `BehaveFormerMAE` baseline, the four open-loop synthetic attacker tiers (T0–T3), and the deterministic LangGraph defense state machine (`src/defensive_mas.py`).

## Prerequisites

- Node.js 18+ and npm
- The FastAPI back end running on `http://localhost:8000`:

```bash
uvicorn src.api.main:app --reload --port 8000
```

## Development & Build

```bash
npm install
npm run dev      # starts Vite dev server on http://localhost:5173
npm run build    # type-checks and builds production bundle into dist/
```

## Notes on Reported Metrics

- **Stage 1 (Path A):** Evaluates the passive gate ($\text{MSE} \le 0.0137, \sigma_v \ge 10\text{ px/s}$) and the multi-scale DFA cascade ($H_1, \Delta H = H_1 - H_2, \ell_F = \log_{10} F(4)$) on recorded BB-MAS test windows.
- **Stage 2 (Path B):** Evaluated as an analytical oracle upper bound (Thesis Sections 3.6.5 and 3.8.3), where borderline genuine windows ($H_1 \in [H_1^* - 0.05, H_1^*)$) are recovered ($\text{FRR}: 28.10\% \to 22.31\%$) and borderline attack windows are rejected ($\text{FAR}$ unchanged at $32.54\%$).
