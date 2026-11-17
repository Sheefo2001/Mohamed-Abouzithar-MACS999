# Unimodal Sanity-Check Results (Balabit & CMU)

This document records the supplementary unimodal checks described in Thesis Section 4.1.3 and Section 5.9 (Balabit Mouse Dynamics and CMU Keystroke Dynamics).

## 1. Balabit Dataset (Static $H_1$ Sensitivity to Positional Noise, `user12`)
- **Sessions Evaluated:** `233` (`user12`)
- **Clean Human Mean $H_1$:** `0.6812`
- **Noisy Trajectory Mean $H_1$ ($\sigma = 0.05$):** `0.6093`
- **Welch's T-Statistic:** `3.5640`
- **P-Value:** `4.1183e-04`

**Interpretation:** Adding uncalibrated Gaussian positional jitter ($\sigma = 0.05$) to `[0, 1]`-normalised Balabit mouse coordinates significantly lowers the static short-range DFA scaling exponent $H_1$ ($p = 4.12 \times 10^{-4}$) by injecting uncorrelated high-frequency velocity increments. This check evaluates static $H_1$ sensitivity to positional noise on a single user without a latency probe; the main multi-tier benchmark is conducted on SU-AIS BB-MAS.

## 2. CMU Dataset (ACT-R Cognitive Fatigue Numerical Check)
- **Subjects / Repetitions:** `51 subjects`, `20,400 password repetitions`
- **Average Flight-Time Dilation ($d = 0.1$):** `+0.0080 s`

**Interpretation:** Applying the ACT-R fatigue equation ($d = 0.1$, hold damping $0.20$) to the first ten hold and flight intervals across all 20,400 CMU repetitions produces a stable mean flight-time dilation of `+0.0080 s` without numerical instability. This verifies the numerical behavior of the fatigue transformation on recorded keystroke timings rather than evaluating keystroke spoofing end-to-end.
