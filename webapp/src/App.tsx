import { useState, useEffect, useRef } from 'react';
import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import axios from 'axios';
import { 
  Activity, BarChart2, ShieldAlert, Shield, BrainCircuit, FileText, 
  Lock, Unlock, Zap, Eye, CheckCircle, AlertTriangle 
} from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';

const API_BASE = 'http://localhost:8000/api';

// ---------------------------------------------------------------- Navigation Sidebar
function Sidebar() {
  const location = useLocation();
  return (
    <div className="sidebar">
      <h1 className="text-gradient">ACTIVE CBA</h1>
      <Link to="/" className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}>
        <Activity size={22} /> Dashboard Hub
      </Link>
      <Link to="/evaluate" className={`nav-link ${location.pathname === '/evaluate' ? 'active' : ''}`}>
        <BarChart2 size={22} /> Evaluation Suite
      </Link>
      <Link to="/phase2" className={`nav-link ${location.pathname === '/phase2' ? 'active' : ''}`}>
        <Shield size={22} /> Agentic Defense Hub
      </Link>
      <Link to="/phase2-evaluate" className={`nav-link ${location.pathname === '/phase2-evaluate' ? 'active' : ''}`}>
        <BrainCircuit size={22} /> Phase 2 Evaluation
      </Link>
      <Link to="/simulator" className={`nav-link ${location.pathname === '/simulator' ? 'active' : ''}`}>
        <ShieldAlert size={22} /> Live Simulator
      </Link>
      <Link 
        to="/professor-update" 
        className={`nav-link ${location.pathname === '/professor-update' ? 'active' : ''}`} 
        style={{ borderTop: '1px solid rgba(255,255,255,0.1)', marginTop: '10px', paddingTop: '10px' }}
      >
        <FileText size={22} /> Benchmark Summary
      </Link>
      
      <div style={{ marginTop: 'auto', color: 'var(--text-muted)', fontSize: '0.85rem', lineHeight: '1.6' }}>
        <p style={{ fontWeight: 700, color: 'var(--text-main)' }}>Thesis Pipeline v2.0</p>
        <p>Model: BehaveFormerMAE</p>
        <p>Defense: Two-Stage Active MAS</p>
        <p>Dataset: SU-AIS BB-MAS (92 Subjects)</p>
        <p>Validation: Leak-Free Chronological</p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Dashboard Hub
function DashboardHub() {
  return (
    <div>
      <div className="glass-panel">
        <h2 className="title-glow" style={{ fontSize: '2.5rem', marginBottom: '20px' }}>
          Hypotheses H₁₁ & H₁₂ <span className="text-gradient">Benchmark Overview</span>
        </h2>
        <p style={{ lineHeight: '1.8', fontSize: '1.15rem', color: 'var(--text-muted)', maxWidth: '900px' }}>
          Welcome to the <strong>Continuous Behavioral Biometric Security Command Center</strong>. 
          This benchmark evaluates the vulnerability of passive continuous authentication systems to 
          synthetic trajectory attacks (<strong>The Defensive Gap</strong>, H₁₁), and evaluates a deterministic 
          multi-agent system combining <strong>Multi-Scale Detrended Fluctuation Analysis (M-DFA)</strong> 
          with a two-stage 50 ms / 150 ms probing protocol (H₁₂).
        </p>

        <div className="metric-grid" style={{ marginTop: '36px' }}>
          <div className="glass-panel" style={{ borderColor: 'var(--danger)', background: 'rgba(255, 42, 42, 0.04)' }}>
            <h3 style={{ color: 'var(--danger)', fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <AlertTriangle size={20} /> The Defensive Gap (H₁₁)
            </h3>
            <div className="metric-value danger">92.56%</div>
            <p style={{ fontWeight: 600 }}>T1 Replay + Langevin FAR</p>
            <hr style={{ borderColor: 'var(--glass-border)', margin: '14px 0' }}/>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              Passive BehaveFormerMAE + variance gating achieves 0.00% FAR on linear scripted bots (T0), but accepts 92.56% of T1 replay and 100.00% of smooth T2 minimum-jerk trajectories.
            </p>
          </div>

          <div className="glass-panel" style={{ borderColor: 'var(--success)', background: 'rgba(0, 255, 136, 0.04)' }}>
            <h3 style={{ color: 'var(--success)', fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle size={20} /> Stage 1 M-DFA Cascade (H₁₂)
            </h3>
            <div className="metric-value success">-50.0%</div>
            <p style={{ fontWeight: 600 }}>Relative FAR Reduction (Stage 1)</p>
            <hr style={{ borderColor: 'var(--glass-border)', margin: '14px 0' }}/>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              Multi-scale DFA gating cuts overall adversary acceptance from 65.08% down to 32.54% (Balanced, FRR 28.10%) and 15.81% (High-Security) via crossover (ΔH) and fluctuation magnitude (ℓ_F).
            </p>
          </div>

          <div className="glass-panel" style={{ borderColor: 'var(--accent-purple)', background: 'rgba(176, 38, 255, 0.04)' }}>
            <h3 style={{ color: 'var(--accent-purple)', fontSize: '1.25rem', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Zap size={20} /> Stage 2 Step-Up (Oracle Bound)
            </h3>
            <div className="metric-value" style={{ color: 'var(--accent-purple)' }}>22.31%</div>
            <p style={{ fontWeight: 600 }}>Oracle Post-Escalation FRR</p>
            <hr style={{ borderColor: 'var(--glass-border)', margin: '14px 0' }}/>
            <p style={{ color: 'var(--text-muted)', fontSize: '0.95rem' }}>
              94.21% of genuine sessions are decided at Stage 1. Under the analytical oracle bound (Section 3.8.3), escalating the 5.79% borderline genuine windows lowers FRR from 28.10% to 22.31% with FAR unchanged at 32.54%.
            </p>
          </div>
        </div>

        <div style={{ marginTop: '36px', display: 'flex', gap: '16px', flexWrap: 'wrap' }}>
          <Link to="/simulator" className="btn-premium" style={{ textDecoration: 'none' }}>
            Launch Live Interactive Simulator
          </Link>
          <Link to="/phase2" className="btn-premium" style={{ textDecoration: 'none', background: 'transparent', borderColor: 'var(--glass-border)' }}>
            Explore 5-Node Agentic Architecture
          </Link>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Evaluation Suite
function EvaluationSuite() {
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [graphsReady, setGraphsReady] = useState(true);

  useEffect(() => {
    axios.get(`${API_BASE}/evaluate`)
      .then(res => {
        if (res.data?.status === 'success') {
          setMetrics(res.data.data);
        }
      })
      .catch(console.error);
  }, []);

  const runEvaluation = async () => {
    setLoading(true);
    try {
      const res = await axios.get(`${API_BASE}/evaluate`);
      setMetrics(res.data.data);
      await axios.post(`${API_BASE}/graphs/generate`);
      setGraphsReady(true);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <div className="glass-panel">
        <h2 className="title-glow text-gradient" style={{ fontSize: '2.2rem' }}>Passive Baseline Evaluation (H₁₁)</h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '8px', fontSize: '1.1rem' }}>
          Measured on 242 held-out test windows from 92 BB-MAS subjects. Demonstrates the structural vulnerability of observation-only models.
        </p>

        <button className="btn-premium" style={{ marginTop: '24px' }} onClick={runEvaluation} disabled={loading}>
          {loading ? 'Refreshing Benchmark Reports...' : 'Re-verify Passive Evaluation'}
        </button>

        {metrics && (
          <div className="metric-grid">
            <div className="glass-panel" style={{ background: 'rgba(0, 255, 136, 0.05)', borderColor: 'var(--success)' }}>
              <h3 style={{ color: 'var(--success)', fontSize: '1.2rem' }}>Genuine Human Baseline</h3>
              <div className="metric-value success">{(metrics.scenario_a.mean_error ?? 0.0048).toFixed(4)}</div>
              <p style={{ fontWeight: '600' }}>Mean Squared Error (MSE)</p>
              <hr style={{ borderColor: 'var(--glass-border)', margin: '16px 0' }}/>
              <div className="stat-row">
                <span>Anomaly Threshold:</span>
                <span style={{ fontWeight: 'bold' }}>{(metrics.scenario_a.threshold ?? 0.0137).toFixed(4)}</span>
              </div>
              <div className="stat-row">
                <span>Passive False Rejection Rate:</span>
                <span>{(metrics.scenario_a.frr ?? 7.44).toFixed(2)}%</span>
              </div>
            </div>

            <div className="glass-panel" style={{ background: 'rgba(0, 240, 255, 0.05)', borderColor: 'var(--accent-cyan)' }}>
              <h3 style={{ color: 'var(--accent-cyan)', fontSize: '1.2rem' }}>T0: Scripted Bot (Linear)</h3>
              <div className="metric-value" style={{ color: 'var(--accent-cyan)' }}>
                {(metrics.scenario_b.mean_error ?? 0.0003).toFixed(4)}
              </div>
              <p style={{ fontWeight: '600' }}>Mean Squared Error (MSE)</p>
              <hr style={{ borderColor: 'var(--glass-border)', margin: '16px 0' }}/>
              <div className="stat-row">
                <span>False Acceptance Rate:</span>
                <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>0.00% (100% REJECTED)</span>
              </div>
              <div className="stat-row">
                <span>Vulnerability Status:</span>
                <span style={{ color: 'var(--success)', fontWeight: 'bold' }}>CAUGHT BY KINEMATIC VARIANCE</span>
              </div>
            </div>

            <div className="glass-panel" style={{ background: 'rgba(255, 42, 42, 0.05)', borderColor: 'var(--danger)' }}>
              <h3 style={{ color: 'var(--danger)', fontSize: '1.2rem' }}>T1: Replay + Langevin SDE</h3>
              <div className="metric-value danger">{(metrics.scenario_c.mean_error ?? 0.0048).toFixed(4)}</div>
              <p style={{ fontWeight: '600' }}>Mean Squared Error (MSE)</p>
              <hr style={{ borderColor: 'var(--glass-border)', margin: '16px 0' }}/>
              <div className="stat-row">
                <span>False Acceptance Rate:</span>
                <span style={{ fontWeight: 'bold', color: 'var(--danger)' }}>{(metrics.scenario_c.far ?? 92.56).toFixed(2)}%</span>
              </div>
              <div className="stat-row">
                <span>Vulnerability Status:</span>
                <span style={{ color: 'var(--danger)', fontWeight: 'bold' }}>BYPASSED (The Offensive Gap)</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {graphsReady && (
        <div className="glass-panel" style={{ marginTop: '32px' }}>
          <h2 className="title-glow" style={{ marginBottom: '24px' }}>Empirical Density Distributions (300 DPI)</h2>
          <div className="image-gallery">
            <img src={`${API_BASE}/graphs/defensive_gap_distribution?t=${Date.now()}`} alt="The Defensive Gap: Passive Classifier Structural Vulnerability" />
            <img src={`${API_BASE}/graphs/dumb_bot_validation?t=${Date.now()}`} alt="Scripted bots reconstruct below the threshold" />
          </div>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------- Agentic Defense Hub (Complete 5 Nodes)
function AgenticDefenseHub() {
  return (
    <div>
      <div className="glass-panel">
        <h2 className="title-glow" style={{ fontSize: '2.5rem', marginBottom: '20px' }}>
          <span style={{ color: 'var(--accent-purple)' }}>Phase 2: Complete 5-Node Agentic Active Defense (MAS)</span>
        </h2>
        <p style={{ lineHeight: '1.8', fontSize: '1.15rem', color: 'var(--text-muted)' }}>
          To overcome the structural blindness of passive classifiers, we deploy an autonomous 
          <strong> LangGraph Multi-Agent System (MAS)</strong>. When an interaction stream passes passive verification 
          or exhibits ambiguous variance, the agent initiates an active neuromuscular probe to interrogate the closed-loop motor 
          control reflex.
        </p>

        {/* 5-Node Architecture Cards */}
        <div className="metric-grid" style={{ marginTop: '32px' }}>
          <div className="glass-panel" style={{ borderColor: 'var(--accent-cyan)' }}>
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '12px', fontSize: '1.2rem' }}>
              1. Continuous Monitoring Node
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6', fontSize: '0.95rem' }}>
              Continuously evaluates trajectory featurization and kinematic variance. If velocity standard deviation collapses 
              (σᵥ &lt; 10.0 px/s), it terminates mechanical scripted bots immediately without wasting computational resources.
            </p>
          </div>

          <div className="glass-panel" style={{ borderColor: 'var(--accent-purple)' }}>
            <h3 style={{ color: 'var(--accent-purple)', marginBottom: '12px', fontSize: '1.2rem' }}>
              2. Stage 1 Probe Node (50 ms)
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6', fontSize: '0.95rem' }}>
              Requests a 50 ms cursor-latency probe per Section 3.6.3. Because BB-MAS is a pre-recorded dataset without live probe perturbations, Stage 1 evaluates the M-DFA features of the recorded window itself.
            </p>
          </div>

          <div className="glass-panel" style={{ borderColor: 'var(--success)' }}>
            <h3 style={{ color: 'var(--success)', marginBottom: '12px', fontSize: '1.2rem' }}>
              3. Multi-Scale DFA Analysis Node
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6', fontSize: '0.95rem' }}>
              Computes the Short-Range Hurst Exponent (H₁, s ∈ [4, 16]) and Long-Range Exponent (H₂, s ∈ [17, 64]). 
              Checks short-range scaling (H₁ ≥ 0.6443), multi-scale crossover (ΔH = H₁ − H₂ ≥ 0.08), and fluctuation magnitude (ℓ_F = log₁₀ F(4) ≥ 1.85).
            </p>
          </div>

          <div className="glass-panel" style={{ borderColor: 'var(--warning)', background: 'rgba(255, 170, 0, 0.04)' }}>
            <h3 style={{ color: 'var(--warning)', marginBottom: '12px', fontSize: '1.2rem' }}>
              4. Escalating Probe Node (150 ms Oracle)
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6', fontSize: '0.95rem' }}>
              Conditional step-up node per Sections 3.6.5 and 3.8.3. Triggers only when H₁ falls in the borderline band [H₁* − 0.05, H₁*). Evaluated offline as an explicit analytical oracle upper bound.
            </p>
          </div>

          <div className="glass-panel" style={{ borderColor: 'var(--accent-cyan)' }}>
            <h3 style={{ color: 'var(--accent-cyan)', marginBottom: '12px', fontSize: '1.2rem' }}>
              5. Zero-Trust Decision Arbiter
            </h3>
            <p style={{ color: 'var(--text-muted)', lineHeight: '1.6', fontSize: '0.95rem' }}>
              Synthesizes multi-scale exponents, trust scores, and escalation states into an autonomous 
              Policy Enforcement Point (PEP) decision: AUTHENTICATED, STEP_UP_REQUIRED, or REVOKED with structured audit logging.
            </p>
          </div>
        </div>

        {/* Featured Deep-Dive: Path A vs Path B */}
        <div className="glass-panel" style={{ marginTop: '40px', borderLeft: '4px solid var(--accent-purple)' }}>
          <h3 style={{ color: 'white', fontSize: '1.4rem', marginBottom: '14px' }}>
            Stage 1 Cascade (Path A) vs. Stage 2 Step-Up (Path B Oracle Bound)
          </h3>
          <p style={{ color: 'var(--text-muted)', lineHeight: '1.8', fontSize: '1.05rem' }}>
            Because BB-MAS was recorded without live cursor-latency perturbations, Stage 1 measures static M-DFA features on recorded windows, while Stage 2 is evaluated strictly as an analytical oracle upper bound (Section 3.8.3).
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '20px', marginTop: '20px' }}>
            <div style={{ background: 'rgba(0,0,0,0.3)', padding: '20px', borderRadius: '14px', border: '1px solid var(--glass-border)' }}>
              <h4 style={{ color: 'var(--accent-cyan)', marginBottom: '8px' }}>Path A: Stage 1 Cascade (Measured)</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                Label-blind evaluation on recorded BB-MAS test windows. Cuts overall adversary FAR from 65.08% to 32.54% (Balanced Mode, FRR 28.10%), a 5.94 percentage-point reduction in balanced error.
              </p>
            </div>

            <div style={{ background: 'rgba(176, 38, 255, 0.05)', padding: '20px', borderRadius: '14px', border: '1px solid var(--accent-purple)' }}>
              <h4 style={{ color: 'var(--accent-purple)', marginBottom: '8px' }}>Path B: Two-Stage Step-Up (Analytical Oracle Bound)</h4>
              <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem', lineHeight: '1.6' }}>
                Escalates borderline windows with H₁ ∈ [0.5943, 0.6443) (5.79% of genuine sessions). Under an ideal 150 ms oracle (Section 3.8.3), the 14 borderline genuine windows are recovered (FRR 28.10% → 22.31%) and the 27 borderline attack windows are rejected, leaving overall FAR unchanged at <strong>32.54%</strong>.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Phase 2 Evaluation Suite
function Phase2EvaluationSuite() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    axios.get(`${API_BASE}/results`)
      .then(res => {
        if (res.data?.status === 'success') {
          setData(res.data.data);
        }
      })
      .catch(console.error);
  }, []);

  const tiers = data?.attacker_tiers_test || {
    scripted_bot: { far_passive_pct: 0.0, far_active_pct: 0.0, far_cascaded_pct: 0.0, far_post_escalation_150ms_pct: 0.0, mean_h1: 0.0 },
    replay_langevin: { far_passive_pct: 92.56, far_active_pct: 71.9, far_cascaded_pct: 71.9, far_post_escalation_150ms_pct: 71.9, mean_h1: 0.8634 },
    ou_generator: { far_passive_pct: 100.0, far_active_pct: 24.79, far_cascaded_pct: 24.79, far_post_escalation_150ms_pct: 24.79, mean_h1: 1.0485 },
    fgn_matched: { far_passive_pct: 67.77, far_active_pct: 33.47, far_cascaded_pct: 33.47, far_post_escalation_150ms_pct: 33.47, mean_h1: 0.7659 },
  };

  const regimes = data?.operating_regimes_comparison || {
    balanced: { genuine_frr_pct: 28.1, overall_far_pct: 32.54, threshold_h1: 0.6443 },
    frictionless: { genuine_frr_pct: 18.6, overall_far_pct: 58.06, threshold_h1: 0.6146 },
    high_security: { genuine_frr_pct: 57.85, overall_far_pct: 15.81, threshold_h1: 0.9058 },
  };

  const escalation = data?.two_stage_escalation_150ms || {
    unobtrusive_50ms_share_pct: 94.21,
    genuine_escalation_trigger_pct: 5.79,
    genuine_frr_post_escalation_pct: 22.31,
    overall_far_post_escalation_pct: 32.54
  };

  return (
    <div>
      <div className="glass-panel">
        <h2 className="title-glow text-gradient" style={{ fontSize: '2.2rem' }}>
          Empirical Multi-Scale Defense Isolation (H₁₂)
        </h2>
        <p style={{ color: 'var(--text-muted)', marginTop: '8px', fontSize: '1.1rem' }}>
          Evaluating passive monitoring vs. active M-DFA verification across all 4 adversary tiers and 3 operational regimes.
        </p>

        {/* Operating Regimes Table with Formatted Math */}
        <div style={{ marginTop: '32px' }}>
          <h3 style={{ color: 'var(--text-main)', fontSize: '1.3rem', marginBottom: '16px' }}>
            Operational Decision Regimes (Validation-Calibrated)
          </h3>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.1)', color: 'var(--accent-cyan)' }}>
                  <th style={{ padding: '12px' }}>Regime</th>
                  <th style={{ padding: '12px' }}>Threshold H₁*</th>
                  <th style={{ padding: '12px' }}>Multi-Scale Constraints</th>
                  <th style={{ padding: '12px' }}>Genuine FRR</th>
                  <th style={{ padding: '12px' }}>Adversary Net FAR</th>
                  <th style={{ padding: '12px' }}>Target SLA Profile</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', background: 'rgba(0, 255, 136, 0.05)' }}>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--success)' }}>Balanced (Recommended)</td>
                  <td style={{ padding: '12px', fontFamily: 'monospace' }}>0.6443</td>
                  <td style={{ padding: '12px', fontSize: '0.9rem' }}>ΔH ≥ 0.08, ℓ_F ≥ 1.85, σᵥ ≥ 10.0 px/s</td>
                  <td style={{ padding: '12px', fontWeight: 'bold' }}>{Number(regimes.balanced.genuine_frr_pct).toFixed(2)}%</td>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--success)' }}>{Number(regimes.balanced.overall_far_pct).toFixed(2)}%</td>
                  <td style={{ padding: '12px', color: 'var(--text-muted)' }}>Optimal Enterprise Security-Usability</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--accent-cyan)' }}>Frictionless</td>
                  <td style={{ padding: '12px', fontFamily: 'monospace' }}>0.6146</td>
                  <td style={{ padding: '12px', fontSize: '0.9rem' }}>σᵥ ≥ 10.0 px/s</td>
                  <td style={{ padding: '12px', fontWeight: 'bold' }}>{Number(regimes.frictionless.genuine_frr_pct).toFixed(2)}%</td>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--warning)' }}>{Number(regimes.frictionless.overall_far_pct).toFixed(2)}%</td>
                  <td style={{ padding: '12px', color: 'var(--text-muted)' }}>Low-Friction Corporate Portals</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--danger)' }}>High-Security (EER)</td>
                  <td style={{ padding: '12px', fontFamily: 'monospace' }}>0.9058</td>
                  <td style={{ padding: '12px', fontSize: '0.9rem' }}>ΔH ≥ 0.12, ℓ_F ≥ 2.00, σᵥ ≥ 10.0 px/s</td>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--warning)' }}>{Number(regimes.high_security.genuine_frr_pct).toFixed(2)}%</td>
                  <td style={{ padding: '12px', fontWeight: 'bold', color: 'var(--success)' }}>{Number(regimes.high_security.overall_far_pct).toFixed(2)}%</td>
                  <td style={{ padding: '12px', color: 'var(--text-muted)' }}>Zero-Tolerance Financial Infrastructure</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        {/* 4-Tier Breakdown: Path A (50ms) vs Path B (150ms Escalation) */}
        <div style={{ marginTop: '36px' }}>
          <h3 style={{ color: 'var(--text-main)', fontSize: '1.3rem', marginBottom: '16px' }}>
            Path A (50 ms Only) vs. Path B (Two-Stage 150 ms Escalation)
          </h3>
          <div className="metric-grid">
            <div className="glass-panel" style={{ borderColor: 'var(--success)' }}>
              <h3 style={{ color: 'var(--success)' }}>T0 Scripted Bot</h3>
              <div className="metric-value success">0.00%</div>
              <p style={{ fontWeight: 600 }}>Path A & Path B FAR</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Mechanical signals lack physiological micro-tremor variance (σᵥ = 0.0 px/s).
              </p>
            </div>

            <div className="glass-panel" style={{ borderColor: 'var(--warning)' }}>
              <h3 style={{ color: 'var(--warning)' }}>T1 Replay + Langevin</h3>
              <div className="metric-value" style={{ color: 'var(--warning)' }}>{tiers.replay_langevin.far_cascaded_pct.toFixed(1)}%</div>
              <p style={{ fontWeight: 600 }}>Path A (50 ms Only) FAR</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Reduced from 92.56% passive acceptance. Drops to 41.74% in High-Security Mode.
              </p>
            </div>

            <div className="glass-panel" style={{ borderColor: 'var(--accent-purple)' }}>
              <h3 style={{ color: 'var(--accent-purple)' }}>T2 OU Minimum-Jerk</h3>
              <div className="metric-value" style={{ color: 'var(--accent-purple)' }}>{tiers.ou_generator.far_cascaded_pct.toFixed(1)}%</div>
              <p style={{ fontWeight: 600 }}>Path A (50 ms Only) FAR</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Dropped from 100.0% passive bypass to 24.79% via scale crossover failure (ΔH &lt; 0.08).
              </p>
            </div>

            <div className="glass-panel" style={{ borderColor: 'var(--accent-cyan)' }}>
              <h3 style={{ color: 'var(--accent-cyan)' }}>T3 fGn Matched</h3>
              <div className="metric-value" style={{ color: 'var(--accent-cyan)' }}>{tiers.fgn_matched.far_cascaded_pct.toFixed(1)}%</div>
              <p style={{ fontWeight: 600 }}>Path A (50 ms Only) FAR</p>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '8px' }}>
                Dropped from 67.77% passive bypass to 33.47% (and 11.57% in High-Security Mode).
              </p>
            </div>
          </div>
        </div>

        {/* Escalation Statistics Card */}
        <div className="glass-panel" style={{ marginTop: '32px', background: 'rgba(176, 38, 255, 0.03)', borderColor: 'var(--accent-purple)' }}>
          <h4 style={{ color: 'var(--accent-purple)', fontSize: '1.2rem', marginBottom: '10px' }}>
            Two-Stage Escalation Performance (Path B Analytical Oracle Bound)
          </h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: '16px', fontSize: '0.95rem' }}>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Stage 1 Unescalated Genuine Share:</span>
              <p style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--success)' }}>{escalation.unobtrusive_50ms_share_pct}%</p>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Genuine Escalation Rate:</span>
              <p style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--accent-cyan)' }}>{escalation.genuine_escalation_trigger_pct}%</p>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Oracle Post-Escalation Genuine FRR:</span>
              <p style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--success)' }}>{escalation.genuine_frr_post_escalation_pct}%</p>
            </div>
            <div>
              <span style={{ color: 'var(--text-muted)' }}>Post-Escalation Adversary FAR:</span>
              <p style={{ fontWeight: 700, fontSize: '1.2rem', color: 'var(--accent-purple)' }}>{escalation.overall_far_post_escalation_pct}%</p>
            </div>
          </div>
        </div>
      </div>

      <div className="glass-panel" style={{ marginTop: '32px' }}>
        <h2 className="title-glow" style={{ marginBottom: '24px' }}>Publication Artifacts (300 DPI)</h2>
        <div className="image-gallery">
          <img src={`${API_BASE}/graphs/threat_mitigation_far_bar?t=${Date.now()}`} alt="Empirical Threat Mitigation across Tiers" />
          <img src={`${API_BASE}/graphs/fig3_mdfa_scaling_curves?t=${Date.now()}`} alt="M-DFA Log-Log Scaling Curves" />
          <img src={`${API_BASE}/graphs/fig4_passive_vs_active_roc?t=${Date.now()}`} alt="ROC Comparison" />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Interactive Live Simulator (Fast, Trail & 150ms Support)
function LiveSimulator() {
  const initialWindowParam = new URLSearchParams(window.location.search).get('window')
    ?? new URLSearchParams(window.location.search).get('window_index');
  const initialWindow = initialWindowParam !== null && initialWindowParam !== '' && !Number.isNaN(Number(initialWindowParam))
    ? Number(initialWindowParam)
    : null;

  const [tier, setTier] = useState<string>('genuine');
  const [regime, setRegime] = useState<string>('balanced');
  const [isRunning, setIsRunning] = useState(false);
  const [dataPoints, setDataPoints] = useState<any[]>([]);
  const [pathHistory, setPathHistory] = useState<[number, number][]>([]);
  const [currentXY, setCurrentXY] = useState({ x: 250, y: 250 });
  const [status, setStatus] = useState<string | null>(null);
  const [probeIntensity, setProbeIntensity] = useState<string>('None');
  const [currentUserId, setCurrentUserId] = useState<number | null>(null);
  const [currentWindowIndex, setCurrentWindowIndex] = useState<number | null>(initialWindow);
  const [isTargetLocked, setIsTargetLocked] = useState(initialWindow !== null);
  const [isTyping, setIsTyping] = useState(false);
  const [isTheaterMode, setIsTheaterMode] = useState(false);

  // Real-time Telemetry & DFA Features
  const [currentMse, setCurrentMse] = useState<number>(0.0);
  const [mseThreshold, setMseThreshold] = useState<number>(0.0137);
  const [trustScore, setTrustScore] = useState<number>(1.0);
  const [h1Val, setH1Val] = useState<number | null>(null);
  const [h2Val, setH2Val] = useState<number | null>(null);
  const [deltaHVal, setDeltaHVal] = useState<number | null>(null);
  const [logF0Val, setLogF0Val] = useState<number | null>(null);
  const [speedStdVal, setSpeedStdVal] = useState<number | null>(null);
  const [h1Threshold, setH1Threshold] = useState<number>(0.6443);
  const [auditLog, setAuditLog] = useState<any[]>([]);
  const [explanation, setExplanation] = useState<any>(null);

  const eventSource = useRef<EventSource | null>(null);

  const startSimulation = (selectedTier: string, selectedRegime: string) => {
    if (eventSource.current) {
      eventSource.current.close();
      eventSource.current = null;
    }

    setTier(selectedTier);
    setRegime(selectedRegime);
    setDataPoints([]);
    setPathHistory([]);
    setCurrentXY({ x: 250, y: 250 });
    setStatus(null);
    setProbeIntensity('None');
    setTrustScore(1.0);
    setH1Val(null);
    setH2Val(null);
    setDeltaHVal(null);
    setLogF0Val(null);
    setSpeedStdVal(null);
    setAuditLog([]);
    setIsRunning(true);

    if (!isTargetLocked) {
      setCurrentUserId(null);
      setCurrentWindowIndex(null);
    }

    const queryParams = new URLSearchParams({
      tier: selectedTier,
      regime: selectedRegime,
    });
    if (isTargetLocked && currentWindowIndex !== null) {
      queryParams.set('window_index', currentWindowIndex.toString());
    }

    const streamUrl = `${API_BASE}/simulate/stream?${queryParams.toString()}`;
    const es = new EventSource(streamUrl);
    eventSource.current = es;

    let timeIndex = 0;

    es.onmessage = (event) => {
      const raw = event.data;
      if (raw === '[DONE]') {
        setIsRunning(false);
        es.close();
        return;
      }

      let parsed: any;
      try {
        const sanitized = raw.replace(/:\s*NaN/g, ': null');
        parsed = JSON.parse(sanitized);
      } catch (err) {
        console.error('Failed to parse SSE event data:', raw, err);
        return;
      }
      timeIndex++;

      if (timeIndex === 1) {
        setCurrentUserId(parsed.user_id);
        setCurrentWindowIndex(parsed.window_index);
        setMseThreshold(parsed.mae_threshold);
        setH1Threshold(parsed.h1_threshold);
        setExplanation(parsed.explanation);
      }

      setCurrentMse(parsed.mse);
      setTrustScore(parsed.trust_score);
      setStatus(parsed.session_status);
      setProbeIntensity(parsed.probe_intensity || 'None');
      setIsTyping(parsed.is_typing);

      if (parsed.h1 !== null) setH1Val(parsed.h1);
      if (parsed.h2 !== null) setH2Val(parsed.h2);
      if (parsed.delta_h !== null) setDeltaHVal(parsed.delta_h);
      if (parsed.log_f0 !== null) setLogF0Val(parsed.log_f0);
      if (parsed.speed_std !== null) setSpeedStdVal(parsed.speed_std);

      // Trajectory Coordinates & Trail
      setCurrentXY({ x: parsed.x, y: parsed.y });
      if (parsed.path && Array.isArray(parsed.path)) {
        setPathHistory(parsed.path);
      } else {
        setPathHistory(prev => [...prev, [parsed.x, parsed.y]]);
      }

      // Chart Datapoints
      setDataPoints((prev) => {
        const next = [
          ...prev,
          {
            time: timeIndex,
            mse: parsed.mse,
            velocity: parsed.velocity,
          },
        ];
        if (next.length > 50) next.shift();
        return next;
      });

      // Audit Logs
      if (parsed.audit_log) {
        setAuditLog((prev) => {
          if (prev.length === 0 || prev[prev.length - 1].event !== parsed.audit_log.event) {
            return [...prev, parsed.audit_log];
          }
          return prev;
        });
      }
    };

    es.onerror = () => {
      setIsRunning(false);
      es.close();
    };
  };

  useEffect(() => {
    return () => {
      if (eventSource.current) eventSource.current.close();
      document.body.classList.remove('theater-mode');
    };
  }, []);

  const toggleTheaterMode = () => {
    const next = !isTheaterMode;
    setIsTheaterMode(next);
    if (next) {
      document.body.classList.add('theater-mode');
    } else {
      document.body.classList.remove('theater-mode');
    }
  };

  const getThemeColor = () => {
    if (status === 'AUTHENTICATED') return 'var(--success)';
    if (status === 'REVOKED') return 'var(--danger)';
    if (status === 'ESCALATING_150MS' || probeIntensity === '150ms') return 'var(--accent-purple)';
    if (status === 'PROBING_50MS' || probeIntensity === '50ms') return 'var(--accent-cyan)';
    if (tier === 'genuine') return 'var(--success)';
    if (tier === 't0_scripted') return 'var(--accent-cyan)';
    if (tier === 't1_langevin') return 'var(--danger)';
    if (tier === 't2_ou_jerk') return 'var(--accent-purple)';
    return 'var(--warning)';
  };

  return (
    <div className="glass-panel" style={{ marginTop: '20px' }}>
      {/* Simulator Header */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '20px' }}>
        <div>
          <h2 className="title-glow" style={{ marginBottom: '12px' }}>
            Interactive Neuromuscular Defense Simulator
          </h2>
          <p style={{ color: 'var(--text-muted)', maxWidth: '650px', fontSize: '1.05rem', lineHeight: '1.6' }}>
            Execute real-time biometric session authentication. Select an adversary threat tier and operational regime 
            to observe how the LangGraph agent executes two-stage 50 ms / 150 ms probing.
          </p>
        </div>

        {/* Action Controls */}
        <div style={{ display: 'flex', gap: '12px', alignItems: 'center', flexWrap: 'wrap' }}>
          {/* Target Lock Toggle */}
          <div style={{ 
            display: 'flex', alignItems: 'center', gap: '10px', 
            background: 'rgba(0,0,0,0.4)', padding: '10px 16px', borderRadius: '12px',
            border: isTargetLocked ? '1px solid var(--accent-cyan)' : '1px solid var(--glass-border)'
          }}>
            {isTargetLocked ? <Lock size={18} color="var(--accent-cyan)" /> : <Unlock size={18} color="var(--text-muted)" />}
            <div style={{ display: 'flex', flexDirection: 'column' }}>
              <span style={{ fontSize: '0.85rem', fontWeight: 700, color: isTargetLocked ? 'var(--accent-cyan)' : 'var(--text-main)' }}>
                Target Lock
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                {isTargetLocked && currentUserId ? `Subject #${currentUserId}` : isTargetLocked && currentWindowIndex !== null ? `Window #${currentWindowIndex}` : 'Random Subject'}
              </span>
            </div>
            <button 
              className="sim-btn"
              style={{
                background: isTargetLocked ? 'var(--accent-cyan)' : 'transparent',
                color: isTargetLocked ? 'black' : 'var(--text-muted)',
                border: '1px solid rgba(255,255,255,0.2)',
                padding: '6px 12px', borderRadius: '8px', cursor: 'pointer', fontWeight: 700, fontSize: '0.8rem'
              }}
              onClick={() => setIsTargetLocked(!isTargetLocked)}
            >
              {isTargetLocked ? 'LOCKED' : 'OFF'}
            </button>
          </div>

          <button className="btn-premium" style={{ padding: '12px 20px', fontSize: '0.85rem' }} onClick={toggleTheaterMode}>
            {isTheaterMode ? 'Exit Theater Mode' : '🎬 Theater Mode'}
          </button>
        </div>
      </div>

      {/* Operational Regime Switcher */}
      <div style={{ marginTop: '24px' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Select Operational Defense Regime
        </span>
        <div className="regime-selector" style={{ marginTop: '8px' }}>
          <button 
            className={`regime-btn ${regime === 'balanced' ? 'active balanced' : ''}`}
            onClick={() => { setRegime('balanced'); if (isRunning) startSimulation(tier, 'balanced'); }}
          >
            <Shield size={16} /> Balanced Mode (H₁* = 0.6443)
          </button>
          <button 
            className={`regime-btn ${regime === 'frictionless' ? 'active frictionless' : ''}`}
            onClick={() => { setRegime('frictionless'); if (isRunning) startSimulation(tier, 'frictionless'); }}
          >
            <Zap size={16} /> Frictionless Mode (H₁* = 0.6146)
          </button>
          <button 
            className={`regime-btn ${regime === 'high_security' ? 'active high-sec' : ''}`}
            onClick={() => { setRegime('high_security'); if (isRunning) startSimulation(tier, 'high_security'); }}
          >
            <Lock size={16} /> High-Security Mode (H₁* = 0.9058)
          </button>
        </div>
      </div>

      {/* Adversary Tier Buttons */}
      <div style={{ marginTop: '20px' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.5px' }}>
          Select Adversary Threat Tier
        </span>
        <div className="scenario-selector" style={{ marginTop: '8px', flexWrap: 'wrap' }}>
          <button 
            className={`scenario-btn ${tier === 'genuine' ? 'active clean' : ''}`}
            onClick={() => startSimulation('genuine', regime)}
          >
            Scenario A: Genuine Human
          </button>
          <button 
            className={`scenario-btn ${tier === 't0_scripted' ? 'active dumb' : ''}`}
            onClick={() => startSimulation('t0_scripted', regime)}
          >
            T0: Scripted Bot (Linear)
          </button>
          <button 
            className={`scenario-btn ${tier === 't1_langevin' ? 'active deepfake' : ''}`}
            onClick={() => startSimulation('t1_langevin', regime)}
          >
            T1: Replay + Langevin SDE
          </button>
          <button 
            className={`scenario-btn`}
            style={{ 
              borderColor: 'var(--accent-purple)', 
              color: tier === 't2_ou_jerk' ? 'white' : 'var(--accent-purple)', 
              background: tier === 't2_ou_jerk' ? 'rgba(176, 38, 255, 0.2)' : 'transparent' 
            }}
            onClick={() => startSimulation('t2_ou_jerk', regime)}
          >
            T2: OU Minimum-Jerk
          </button>
          <button 
            className={`scenario-btn`}
            style={{ 
              borderColor: 'var(--accent-cyan)', 
              color: tier === 't3_fgn' ? 'black' : 'var(--accent-cyan)', 
              background: tier === 't3_fgn' ? 'var(--accent-cyan)' : 'transparent' 
            }}
            onClick={() => startSimulation('t3_fgn', regime)}
          >
            T3: fGn Noise-Matched
          </button>
        </div>
      </div>

      {/* Pedagogical Explanation Card */}
      {explanation && (
        <div className="pedagogical-card" style={{ borderLeft: `4px solid ${getThemeColor()}` }}>
          <h4 style={{ color: 'white', marginBottom: '8px', fontSize: '1.15rem' }}>{explanation.title}</h4>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: '14px', marginTop: '12px', fontSize: '0.9rem' }}>
            <div>
              <span style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>Physics / Threat Model:</span>
              <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>{explanation.physics}</p>
            </div>
            <div>
              <span style={{ color: 'var(--warning)', fontWeight: 700 }}>Passive Model Behavior:</span>
              <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>{explanation.passive_behavior}</p>
            </div>
            <div>
              <span style={{ color: 'var(--success)', fontWeight: 700 }}>Active MAS Defense Action:</span>
              <p style={{ color: 'var(--text-muted)', marginTop: '4px' }}>{explanation.active_reaction}</p>
            </div>
          </div>
        </div>
      )}

      {/* Main Simulator Viewport */}
      <div className="simulator-container">
        {/* Trajectory Canvas with SVG Path Trail */}
        <div className={`sim-canvas ${status === 'PROBING_50MS' ? 'glitch-active' : ''}`}>
          {/* SVG Trajectory Trail */}
          <svg className="sim-svg-trail" width="500" height="500">
            {pathHistory.length > 1 && (
              <polyline
                fill="none"
                stroke={getThemeColor()}
                strokeWidth={tier === 't0_scripted' ? '3' : '2.5'}
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeDasharray={tier === 't0_scripted' ? 'none' : tier === 't2_ou_jerk' ? 'none' : '4, 4'}
                points={pathHistory.map(p => `${p[0]},${p[1]}`).join(' ')}
                opacity="0.9"
              />
            )}
            {currentXY.x !== 0 && (
              <circle
                cx={currentXY.x}
                cy={currentXY.y}
                r="6"
                fill={getThemeColor()}
                stroke="#ffffff"
                strokeWidth="2"
                style={{ filter: `drop-shadow(0 0 10px ${getThemeColor()})` }}
              />
            )}
          </svg>

          {/* Status Overlay Badges */}
          {status === 'AUTHENTICATED' && <div className="secured-alert">AUTHENTICATED (VERIFIED)</div>}
          {status === 'REVOKED' && <div className="bypassed-alert" style={{ background: 'rgba(255, 42, 42, 0.3)' }}>SESSION REVOKED (ANOMALY)</div>}
          {status === 'PROBING_50MS' && <div className="bypassed-alert" style={{ background: 'rgba(0, 240, 255, 0.3)', color: 'black' }}>STAGE 1: 50ms PROBE REQUESTED...</div>}
          {status === 'ESCALATING_150MS' && (
            <div className="bypassed-alert" style={{ background: 'rgba(176, 38, 255, 0.4)', color: 'white', border: '1px solid var(--accent-purple)' }}>
              STAGE 2 ORACLE: 150ms STEP-UP CHALLENGE (Section 3.6.5)
            </div>
          )}
          {status === 'MONITORING' && <div className="secured-alert" style={{ background: 'rgba(255, 255, 255, 0.1)', color: 'white' }}>PASSIVE MONITORING...</div>}
          {status === 'STEP_UP_REQUIRED' && <div className="bypassed-alert" style={{ background: 'rgba(255, 170, 0, 0.3)', color: 'white' }}>STEP-UP CHALLENGE REQUIRED</div>}

          {/* Subject Badge */}
          {currentUserId !== null && (
            <div style={{ position: 'absolute', bottom: '18px', left: '18px', color: 'rgba(255,255,255,0.7)', zIndex: 5 }}>
              <div style={{ fontWeight: 800, fontSize: '0.95rem' }}>Subject: BB-MAS #{currentUserId}</div>
              <div style={{ fontFamily: 'monospace', fontSize: '0.85rem' }}>Window #{currentWindowIndex}</div>
            </div>
          )}

          {/* Keystroke Indicator */}
          {isRunning && (
            <div style={{ 
              position: 'absolute', bottom: '18px', right: '18px', zIndex: 5,
              display: 'flex', alignItems: 'center', gap: '8px',
              padding: '6px 10px', borderRadius: '8px',
              background: isTyping ? 'rgba(0, 255, 136, 0.2)' : 'rgba(255, 255, 255, 0.05)',
              border: isTyping ? '1px solid var(--success)' : '1px solid rgba(255, 255, 255, 0.1)',
            }}>
              <div style={{ 
                width: '8px', height: '8px', borderRadius: '50%', 
                background: isTyping ? 'var(--success)' : 'rgba(255,255,255,0.2)' 
              }} />
              <span style={{ fontSize: '0.75rem', fontWeight: 700, color: isTyping ? 'var(--success)' : 'rgba(255,255,255,0.5)' }}>
                KEYSTROKES
              </span>
            </div>
          )}
        </div>

        {/* Telemetry Panels & Real-Time Charts */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '20px' }}>
          {/* Multi-Scale DFA Telemetry Dashboard */}
          <div className="glass-panel" style={{ padding: '18px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <h3 style={{ fontSize: '1.1rem', color: 'var(--text-main)', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Eye size={18} color="var(--accent-cyan)" /> Real-Time Neuromotor Telemetry
              </h3>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Gate: <span style={{ fontWeight: 'bold', color: getThemeColor() }}>{status || 'IDLE'}</span>
                {probeIntensity !== 'None' && (
                  <span style={{ marginLeft: '10px', padding: '2px 8px', borderRadius: '4px', background: probeIntensity === '150ms' ? 'rgba(176,38,255,0.2)' : 'rgba(0,240,255,0.2)', color: probeIntensity === '150ms' ? 'var(--accent-purple)' : 'var(--accent-cyan)', fontWeight: 700 }}>
                    Probe: {probeIntensity}
                  </span>
                )}
              </div>
            </div>

            <div className="dfa-badge-grid">
              <div className="dfa-badge">
                <span className="dfa-badge-title">Short-Range H₁</span>
                <span className="dfa-badge-value" style={{ color: (h1Val ?? 0) >= h1Threshold ? 'var(--success)' : 'var(--danger)' }}>
                  {h1Val !== null ? h1Val.toFixed(3) : '---'}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Thresh: {h1Threshold.toFixed(3)}</span>
              </div>

              <div className="dfa-badge">
                <span className="dfa-badge-title">Long-Range H₂</span>
                <span className="dfa-badge-value" style={{ color: 'var(--accent-cyan)' }}>
                  {h2Val !== null ? h2Val.toFixed(3) : '---'}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>s ∈ [17, 64]</span>
              </div>

              <div className="dfa-badge">
                <span className="dfa-badge-title">Crossover ΔH</span>
                <span className="dfa-badge-value" style={{ color: (deltaHVal ?? 0) >= 0.08 ? 'var(--success)' : 'var(--warning)' }}>
                  {deltaHVal !== null ? deltaHVal.toFixed(3) : '---'}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>H₁ − H₂</span>
              </div>

              <div className="dfa-badge">
                <span className="dfa-badge-title">Fluctuation ℓ_F</span>
                <span className="dfa-badge-value" style={{ color: 'var(--text-main)' }}>
                  {logF0Val !== null ? logF0Val.toFixed(2) : '---'}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>log₁₀ F(4)</span>
              </div>

              <div className="dfa-badge">
                <span className="dfa-badge-title">Speed σᵥ</span>
                <span className="dfa-badge-value" style={{ color: (speedStdVal ?? 0) >= 10.0 ? 'var(--success)' : 'var(--danger)' }}>
                  {speedStdVal !== null ? `${speedStdVal.toFixed(1)}` : '---'}
                </span>
                <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>px/s</span>
              </div>
            </div>
          </div>

          {/* LangGraph Agent Console */}
          <div className="glass-panel" style={{ padding: '16px', border: '1px solid var(--accent-purple)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '10px' }}>
              <span style={{ color: 'var(--accent-purple)', fontWeight: 700, fontSize: '0.95rem' }}>
                LangGraph State Machine (MAS)
              </span>
              <span style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                Trust Score: <strong style={{ color: trustScore < 0.5 ? 'var(--danger)' : 'var(--success)' }}>{trustScore.toFixed(2)}</strong>
              </span>
            </div>

            <div className="audit-terminal">
              {auditLog.length === 0 ? (
                <span style={{ color: 'var(--text-muted)' }}>Agent listening on continuous telemetry buffer...</span>
              ) : (
                auditLog.map((log, idx) => (
                  <div key={idx} style={{ marginBottom: '4px' }}>
                    <span style={{ color: 'var(--accent-cyan)' }}>[{log.event}]</span>{' '}
                    {log.mse !== undefined && `MSE=${log.mse.toFixed(4)} `}
                    {log.hurst_exponent !== undefined && `H1=${log.hurst_exponent.toFixed(3)} `}
                    {log.delta_h !== undefined && `ΔH=${log.delta_h.toFixed(3)} `}
                    {log.reason && <span style={{ color: 'var(--warning)' }}>&rarr; {log.reason}</span>}
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Real-time Reconstruction Loss Chart */}
          <div className="sim-chart" style={{ flex: 1, minHeight: '220px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '12px' }}>
              <span style={{ color: getThemeColor(), fontWeight: 700, fontSize: '0.95rem' }}>
                Live Reconstruction Error (MSE) vs Anomaly Threshold
              </span>
              <span style={{ fontFamily: 'monospace', fontSize: '0.85rem', color: currentMse > mseThreshold ? 'var(--danger)' : 'var(--success)' }}>
                MSE: {currentMse.toFixed(4)} / Threshold: {mseThreshold.toFixed(4)}
              </span>
            </div>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={dataPoints}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                <XAxis dataKey="time" stroke="rgba(255,255,255,0.2)" />
                <YAxis stroke="rgba(255,255,255,0.2)" domain={[0, Math.max(0.02, mseThreshold * 1.8)]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: 'rgba(5,5,10,0.9)', border: `1px solid ${getThemeColor()}` }} 
                  formatter={(val: any) => [Number(val).toFixed(4), "MSE"]} 
                />
                <ReferenceLine y={mseThreshold} stroke="var(--danger)" strokeDasharray="3 3" label={{ position: 'top', value: 'Threshold', fill: 'var(--danger)' }} />
                <Line type="monotone" dataKey="mse" stroke={getThemeColor()} strokeWidth={2.5} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Professor Presentation
function ProfessorPresentation() {
  const [data, setData] = useState<any>(null);

  useEffect(() => {
    axios.get(`${API_BASE}/results`)
      .then(res => {
        if (res.data?.status === 'success') {
          setData(res.data.data);
        }
      })
      .catch(console.error);
  }, []);

  const tiers = data?.attacker_tiers_test || {
    scripted_bot: { far_passive_pct: 0.0, far_active_pct: 0.0, far_cascaded_pct: 0.0, far_post_escalation_150ms_pct: 0.0, mean_h1: 0.0 },
    replay_langevin: { far_passive_pct: 92.56, far_active_pct: 71.9, far_cascaded_pct: 71.9, far_post_escalation_150ms_pct: 71.9, mean_h1: 0.8634 },
    ou_generator: { far_passive_pct: 100.0, far_active_pct: 24.79, far_cascaded_pct: 24.79, far_post_escalation_150ms_pct: 24.79, mean_h1: 1.0485 },
    fgn_matched: { far_passive_pct: 67.77, far_active_pct: 33.47, far_cascaded_pct: 33.47, far_post_escalation_150ms_pct: 33.47, mean_h1: 0.7659 },
  };
  const overall = data?.overall_test || { far_passive_pct: 65.08, far_cascaded_pct: 32.54, far_post_escalation_150ms_pct: 32.54, auc_passive: 0.3622, auc_active: 0.5982 };
  const h1Thresh = data?.thresholds?.h1_threshold ?? 0.6443;
  const genFrr = data?.genuine_test?.frr_cascaded_pct ?? 28.10;
  const escalation = data?.two_stage_escalation_150ms || {
    unobtrusive_50ms_share_pct: 94.21,
    genuine_escalation_trigger_pct: 5.79,
    genuine_frr_post_escalation_pct: 22.31,
    overall_far_post_escalation_pct: 32.54
  };

  return (
    <div className="glass-panel" style={{ marginTop: '20px' }}>
      <h2 className="title-glow text-gradient" style={{ fontSize: '2.5rem', marginBottom: '16px' }}>
        Thesis Results: Verified Empirical Benchmark
      </h2>
      <p style={{ color: 'var(--text-muted)', fontSize: '1.15rem', marginBottom: '32px' }}>
        Empirical evaluation on 92 validated BB-MAS subjects across 1,210 held-out test windows (242 genuine + 968 attack across 4 tiers) using a strictly chronological 60/20/20 split (517 train / 167 val / 242 test genuine windows).
      </p>

      <div className="metric-grid" style={{ marginBottom: '40px' }}>
        {/* Tier Performance Section */}
        <div className="glass-panel" style={{ borderColor: 'var(--accent-cyan)' }}>
          <h3 style={{ color: 'var(--accent-cyan)', fontSize: '1.3rem', marginBottom: '16px' }}>
            1. Adversary Threat Tiers
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '14px', fontSize: '0.95rem' }}>
            Measured False Acceptance Rates across 4 open-loop threat tiers (242 test windows per tier = 968 attack windows).
          </p>
          <hr style={{ borderColor: 'var(--glass-border)', margin: '14px 0' }}/>
          <div className="stat-row">
            <span>T0: Scripted Bot (Linear):</span>
            <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>
              {tiers.scripted_bot.far_cascaded_pct.toFixed(2)}% FAR (100% Rejected)
            </span>
          </div>
          <div className="stat-row">
            <span>T1: Replay + Langevin:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--warning)' }}>
              {tiers.replay_langevin.far_cascaded_pct.toFixed(2)}% FAR (vs {tiers.replay_langevin.far_passive_pct.toFixed(2)}% Passive)
            </span>
          </div>
          <div className="stat-row">
            <span>T2: OU Minimum-Jerk:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>
              {tiers.ou_generator.far_cascaded_pct.toFixed(2)}% FAR (vs {tiers.ou_generator.far_passive_pct.toFixed(2)}% Passive)
            </span>
          </div>
          <div className="stat-row">
            <span>T3: fGn Noise-Matched:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>
              {tiers.fgn_matched.far_cascaded_pct.toFixed(2)}% FAR (vs {tiers.fgn_matched.far_passive_pct.toFixed(2)}% Passive)
            </span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '14px' }}>
            *T0 is eliminated by velocity variance checking (σᵥ = 0). T2 minimum-jerk generators are caught by multi-scale crossover (ΔH &lt; 0.08) and fluctuation magnitude (ℓ_F &lt; 1.85).
          </p>
        </div>

        {/* Defense Comparison */}
        <div className="glass-panel" style={{ borderColor: 'var(--success)' }}>
          <h3 style={{ color: 'var(--success)', fontSize: '1.3rem', marginBottom: '16px' }}>
            2. Threat Mitigation Delta
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '14px', fontSize: '0.95rem' }}>
            Aggregate performance across all 1,210 test windows (242 genuine + 968 attack).
          </p>
          <hr style={{ borderColor: 'var(--glass-border)', margin: '14px 0' }}/>
          <div className="stat-row">
            <span>Overall Passive Baseline FAR:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--danger)' }}>{Number(overall.far_passive_pct).toFixed(2)}% (630/968)</span>
          </div>
          <div className="stat-row">
            <span>Path A (Stage 1 Cascade) FAR:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>{Number(overall.far_cascaded_pct).toFixed(2)}% (315/968)</span>
          </div>
          <div className="stat-row">
            <span>Path B (Stage 2 Oracle) FAR:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--accent-purple)' }}>{Number(overall.far_post_escalation_150ms_pct).toFixed(2)}% (315/968)</span>
          </div>
          <div className="stat-row">
            <span>Path A Genuine FRR (vs 7.44% Passive):</span>
            <span style={{ fontWeight: 'bold', color: 'var(--warning)' }}>{Number(genFrr).toFixed(2)}% (68/242)</span>
          </div>
          <div className="stat-row">
            <span>Stage 1 H₁ ROC AUC (All / T1–T3):</span>
            <span style={{ fontWeight: 'bold', color: 'var(--accent-cyan)' }}>{overall.auc_active?.toFixed(4) ?? '0.5982'} / 0.4642</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '14px' }}>
            *Stage 1 cascade halves attack acceptance (65.08% → 32.54%) and reduces balanced error from 36.26% to 30.32% (-5.94 pp).
          </p>
        </div>

        {/* Tradeoffs & Calibration */}
        <div className="glass-panel" style={{ borderColor: 'var(--accent-purple)' }}>
          <h3 style={{ color: 'var(--accent-purple)', fontSize: '1.3rem', marginBottom: '16px' }}>
            3. Calibration & Stage 2 Oracle Bound
          </h3>
          <p style={{ color: 'var(--text-muted)', marginBottom: '14px', fontSize: '0.95rem' }}>
            Two-stage adaptive state machine per Sections 3.6.5 and 3.8.3.
          </p>
          <hr style={{ borderColor: 'var(--glass-border)', margin: '14px 0' }}/>
          <div className="stat-row">
            <span>Balanced H₁* Threshold:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--accent-purple)' }}>{h1Thresh}</span>
          </div>
          <div className="stat-row">
            <span>Stage 1 Unescalated Genuine Share:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>{escalation.unobtrusive_50ms_share_pct}% (228/242)</span>
          </div>
          <div className="stat-row">
            <span>Genuine Escalation Rate:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--warning)' }}>{escalation.genuine_escalation_trigger_pct}% (14/242)</span>
          </div>
          <div className="stat-row">
            <span>Oracle Post-Escalation Genuine FRR:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--success)' }}>{escalation.genuine_frr_post_escalation_pct}% (54/242)</span>
          </div>
          <div className="stat-row">
            <span>Oracle Post-Escalation Overall FAR:</span>
            <span style={{ fontWeight: 'bold', color: 'var(--accent-purple)' }}>{escalation.overall_far_post_escalation_pct}% (0/27 escalated pass)</span>
          </div>
          <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '14px' }}>
            *Stage 2 is evaluated as an analytical oracle upper bound (Section 3.8.3): escalating the 14 borderline genuine windows lowers FRR from 28.10% to 22.31%, while rejecting the 27 borderline attack windows leaves overall FAR unchanged at 32.54%.
          </p>
        </div>
      </div>

      <div style={{ textAlign: 'center', marginTop: '20px' }}>
        <Link to="/simulator" className="btn-premium" style={{ textDecoration: 'none', display: 'inline-block' }}>
          Launch Interactive Live Simulator
        </Link>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------- Main App Shell
export default function App() {
  return (
    <BrowserRouter>
      <div className="app-container">
        <Sidebar />
        <div className="main-content">
          <Routes>
            <Route path="/" element={<DashboardHub />} />
            <Route path="/evaluate" element={<EvaluationSuite />} />
            <Route path="/phase2" element={<AgenticDefenseHub />} />
            <Route path="/phase2-evaluate" element={<Phase2EvaluationSuite />} />
            <Route path="/simulator" element={<LiveSimulator />} />
            <Route path="/professor-update" element={<ProfessorPresentation />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  );
}
