"""Noisy leaky integrate-and-fire neuron: white-noise (diffusion) input and
Ornstein-Uhlenbeck (coloured) input.

Model (Brunel & Hakim 1999, Neural Comput. 11, 1621-1671; Brunel 2000,
J. Comput. Neurosci. 8, 183-208), V in mV measured from rest, t in ms:

    tau dV/dt = -V + mu + sigma sqrt(tau) xi(t)

with xi Gaussian white noise, <xi(t) xi(t')> = delta(t - t'). When V reaches
theta a spike is emitted, V is reset to V_r and held there for tau_ref. mu is
the mean input and sigma the noise amplitude, both in mV. Without the
threshold V is an Ornstein-Uhlenbeck process with mean mu and variance
sigma^2 / 2. This is the diffusion approximation of a neuron receiving many
small Poisson synaptic inputs (Brunel 2000, sec. 3).

Coloured input (Brunel & Sergi 1998; Fourcaud & Brunel 2002, Neural Comput.
14, 2057-2110): the noise is first filtered by a synapse,

    tau   dV/dt   = -V + mu + I_s
    tau_s dI_s/dt = -I_s + sigma sqrt(tau) xi(t),

so I_s is an Ornstein-Uhlenbeck current with variance sigma^2 tau / (2 tau_s)
and correlation time tau_s; tau_s -> 0 gives back the white-noise model.

Parameters (Brunel 2000): tau = 20 ms, theta = 20 mV, V_r = 10 mV,
tau_ref = 2 ms. Every neuron starts with a spike at t = 0 (V = V_r,
refractory until t = tau_ref), so every inter-spike interval
(ISI), including the first, is an independent draw of the renewal process.
Each neuron records its first K ISIs; K is fixed per condition before the
run, so the sample is not length-biased.

Integrators:
- White noise: the exact Gaussian transition of the free OU process over
  each step, V' = mu + (V - mu) e^(-h/tau) + sigma sqrt((1 - e^(-2h/tau))/2) N,
  plus a Brownian-bridge test for threshold crossings inside a step: if both
  end points are below theta a spike is still emitted with probability
  exp(-2 (theta - V)(theta - V') tau / (sigma^2 h)). dt = 0.05 ms. The
  refractory period ends inside a step and the remainder of that step is
  integrated exactly, so ISIs are not quantised to dt. Spike times are the
  linear interpolation of a direct crossing or the midpoint of the active
  part of the step for a bridge crossing. The same scheme without the bridge
  test ("naive", Euler-type threshold detection) is run for comparison.
- Coloured noise: I_s by its exact OU transition, V by the trapezoidal rule
  (linear, so solved explicitly), dt = 0.01 ms; I_s is taken as linear
  within a step. Spike times by linear interpolation.

Validation (validate()):

1. Closed form, free membrane. With OU input (tau_s/tau = 0.05) and no
   threshold the stationary variance of V must equal
   sigma^2 / (2 (1 + tau_s/tau)) and that of I_s sigma^2 tau / (2 tau_s),
   within 2%. (For white noise the V update is the exact transition, so the
   corresponding check would be tautological and is not run.)
2. Closed form, rate. The firing rate 1 / <ISI> for 20 (mu, sigma)
   conditions must match the Siegert formula (Brunel & Hakim 1999; Brunel 2000)
       1/nu = tau_ref + tau sqrt(pi) int_{(V_r-mu)/sigma}^{(theta-mu)/sigma}
              e^(u^2) (1 + erf u) du
   within 2% and within 4 standard errors.
3. Closed form, CV. The ISI coefficient of variation must match
       CV^2 = 2 pi (nu tau)^2 int_{y_r}^{y_theta} e^(x^2) dx
              int_{-inf}^{x} e^(y^2) (1 + erf y)^2 dy
   (Brunel 2000) within 3% and 4 standard errors. Standard errors are
   jackknife estimates over neurons.
4. Timestep. At mu = 15, sigma = 5 mV (noise-driven firing) the bridge
   scheme must stay within 2% of Siegert at dt = 0.02 ... 0.5 ms; the naive
   scheme is reported for comparison: its rate is biased low, and is
   compared (not asserted) with Siegert at a threshold raised by
   0.5826 sigma sqrt(dt/tau), the discrete-monitoring correction of Broadie,
   Glasserman & Kou (1997), 0.5826 = |zeta(1/2)| / sqrt(2 pi).
5. Published correction, coloured noise. For tau_s/tau = k small, Fourcaud &
   Brunel (2002; first derived by Brunel & Sergi 1998) give the
   rate as the Siegert formula with theta and V_r both shifted by
   sigma (alpha/2) sqrt(k), alpha = sqrt(2) |zeta(1/2)| = 2.0652. The
   simulated rate at k = 0.01 and 0.05 must match it within 3%, and must
   differ from the uncorrected white-noise Siegert rate by more than the
   corrected one does.

RNG: NumPy PCG64 seeded with SEED (printed), one spawned stream per
experiment. scipy is used only for validation (quadrature, erfcx, zeta).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.special import erfcx, zeta

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
TAU = 20.0           # ms, membrane time constant
THETA = 20.0         # mV, threshold (from rest)
V_R = 10.0           # mV, reset
TAU_REF = 2.0        # ms, absolute refractory period
SEED = 1729

DT_WHITE = 0.05      # ms
DT_COL = 0.01        # ms
N_NEURON = 500       # neurons per condition, white-noise rate/CV runs
T_MAX = 12000.0      # ms, hard cap per run (every neuron must finish before)
ISI_BUDGET = 5000.0  # ms of expected ISI time per neuron used to size K

# (sigma, mu) conditions for the rate and CV checks, mV
CONDITIONS = ([(2.0, mu) for mu in (18.0, 19.0, 20.0, 22.0, 25.0, 30.0)]
              + [(5.0, mu) for mu in (12.0, 15.0, 18.0, 20.0, 25.0, 30.0, 35.0)]
              + [(10.0, mu) for mu in (5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0)])

DT_STUDY = [0.5, 0.2, 0.05, 0.02]   # ms
STUDY_COND = (5.0, 15.0)            # (sigma, mu)
N_STUDY = 1000

K_COL = [0.01, 0.025, 0.05, 0.1]    # tau_s / tau
COL_CONDS = [(5.0, 15.0), (5.0, 20.0)]
N_COL = 400
COL_BUDGET = 2000.0

RATE_TOL, CV_TOL, Z_MAX = 0.02, 0.03, 4.0
DT_TOL = 0.02
FB_TOL = 0.03
FB_ASSERT_K = [0.01, 0.05]
VAR_TOL = 0.02

ZETA_HALF = -1.4603545088095868
ALPHA = np.sqrt(2.0) * abs(ZETA_HALF)
BGK = abs(ZETA_HALF) / np.sqrt(2.0 * np.pi)   # 0.5826, discrete-monitoring shift


# ------------------------------------------------------------ model functions
def run_white(mu, sigma, K, rng, dt=DT_WHITE, bridge=True, theta=THETA,
              t_max=T_MAX, rec=0, t_rec=0.0):
    """Integrate independent white-noise LIF neurons (one per entry of mu,
    sigma, K) until each has K ISIs. Returns ISI sums and the recording."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    K = np.asarray(K)
    n = mu.size
    V = np.full(n, V_R)
    ref = np.full(n, TAU_REF)       # refractory time left at step start
    t_last = np.zeros(n)
    cnt = np.zeros(n, int)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    D = sigma ** 2 / TAU            # diffusion coefficient of V, mV^2/ms
    n_steps = int(round(t_max / dt))
    n_rec = int(round(t_rec / dt)) if rec else 0
    trace = np.zeros((n_rec, rec))
    spk = [[] for _ in range(rec)]
    for i in range(n_steps):
        t = i * dt
        h = np.clip(dt - ref, 0.0, dt)
        ref = np.maximum(ref - dt, 0.0)
        a = np.exp(-h / TAU)
        Vn = mu + (V - mu) * a + sigma * np.sqrt(0.5 * (1.0 - a * a)) * rng.standard_normal(n)
        active = h > 0.0
        direct = active & (Vn >= theta)
        t0 = t + dt - h
        ts = t0 + h * (theta - V) / np.where(direct, Vn - V, 1.0)
        spike = direct
        if bridge:
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.exp(-2.0 * (theta - V) * (theta - Vn) / (D * h))
            bcross = active & ~direct & (rng.random(n) < p)
            ts = np.where(bcross, t0 + 0.5 * h, ts)
            spike = direct | bcross
        V = np.where(active, Vn, V)
        if spike.any():
            idx = np.flatnonzero(spike)
            isi = ts[idx] - t_last[idx]
            take = cnt[idx] < K[idx]
            j = idx[take]
            s1[j] += isi[take]
            s2[j] += isi[take] ** 2
            cnt[j] += 1
            t_last[idx] = ts[idx]
            V[idx] = V_R
            ref[idx] = TAU_REF - (t + dt - ts[idx])
            for k in idx[idx < rec]:
                if i < n_rec:
                    spk[k].append(ts[k])
        if i < n_rec:
            trace[i] = V[:rec]
        elif (cnt >= K).all():
            break
    return dict(cnt=cnt, s1=s1, s2=s2, t_end=(i + 1) * dt, trace=trace, spikes=spk)


def run_coloured(mu, sigma, k, K, rng, dt=DT_COL, theta=THETA, t_max=T_MAX,
                 rec=0, t_rec=0.0, free_samples=0):
    """Integrate LIF neurons driven by OU current with tau_s = k tau.
    With theta = inf and free_samples > 0, returns samples of (V, I_s)."""
    mu = np.asarray(mu, float)
    sigma = np.asarray(sigma, float)
    tau_s = np.asarray(k, float) * TAU
    K = np.asarray(K)
    n = mu.size
    var_I = sigma ** 2 * TAU / (2.0 * tau_s)
    e_I = np.exp(-dt / tau_s)
    sd_I = np.sqrt(var_I * (1.0 - e_I ** 2))
    I = np.sqrt(var_I) * rng.standard_normal(n)
    if np.isinf(theta):
        V = mu + np.sqrt(sigma ** 2 / (2.0 * (1.0 + tau_s / TAU))) * rng.standard_normal(n)
        ref = np.zeros(n)
    else:
        V = np.full(n, V_R)
        ref = np.full(n, TAU_REF)
    t_last = np.zeros(n)
    cnt = np.zeros(n, int)
    s1 = np.zeros(n)
    s2 = np.zeros(n)
    n_steps = int(round(t_max / dt))
    n_rec = int(round(t_rec / dt)) if rec else 0
    trace = np.zeros((n_rec, rec))
    spk = [[] for _ in range(rec)]
    samples = []
    for i in range(n_steps):
        t = i * dt
        h = np.clip(dt - ref, 0.0, dt)
        ref = np.maximum(ref - dt, 0.0)
        In = I * e_I + sd_I * rng.standard_normal(n)
        I_start = I + (In - I) * (dt - h) / dt
        g = h / (2.0 * TAU)
        Vn = (V * (1.0 - g) + (h / TAU) * (mu + 0.5 * (I_start + In))) / (1.0 + g)
        active = h > 0.0
        spike = active & (Vn >= theta)
        V_old = V
        V = np.where(active, Vn, V)
        I = In
        if free_samples and i % free_samples == 0:
            samples.append((V.copy(), I.copy()))
        if spike.any():
            idx = np.flatnonzero(spike)
            # linear interpolation over the active part of the step
            ts = (t + dt - h[idx]) + h[idx] * (theta - V_old[idx]) / (Vn[idx] - V_old[idx])
            isi = ts - t_last[idx]
            take = cnt[idx] < K[idx]
            j = idx[take]
            s1[j] += isi[take]
            s2[j] += isi[take] ** 2
            cnt[j] += 1
            t_last[idx] = ts
            V[idx] = V_R
            ref[idx] = TAU_REF - (t + dt - ts)
            for kk in idx[idx < rec]:
                if i < n_rec:
                    spk[kk].append(ts[np.searchsorted(idx, kk)])
        if i < n_rec:
            trace[i] = V[:rec]
        elif not free_samples and (cnt >= K).all():
            break
    return dict(cnt=cnt, s1=s1, s2=s2, t_end=(i + 1) * dt, trace=trace,
                spikes=spk, samples=samples)


# ------------------------------------------------------------ closed forms
def siegert_isi(mu, sigma, theta=THETA, vr=V_R):
    """Mean ISI in ms (Siegert formula, Brunel 2000)."""
    I = quad(lambda u: erfcx(-u), (vr - mu) / sigma, (theta - mu) / sigma, limit=200)[0]
    return TAU_REF + TAU * np.sqrt(np.pi) * I


def siegert_rate(mu, sigma, theta=THETA, vr=V_R):
    """Firing rate in Hz."""
    return 1000.0 / siegert_isi(mu, sigma, theta, vr)


def brunel_cv(mu, sigma):
    """ISI CV (Brunel 2000); e^(y^2)(1 + erf y) = erfcx(-y)."""
    inner = lambda x: quad(lambda y: erfcx(-y) ** 2 * np.exp(-y * y), -np.inf, x, limit=200)[0]
    yr, yt = (V_R - mu) / sigma, (THETA - mu) / sigma
    var = 2.0 * np.pi * TAU ** 2 * quad(lambda x: np.exp(x * x) * inner(x), yr, yt, limit=200)[0]
    return np.sqrt(var) / siegert_isi(mu, sigma)


def fourcaud_brunel_rate(mu, sigma, k):
    """Siegert rate with threshold and reset shifted by sigma alpha/2 sqrt(k)
    (Fourcaud & Brunel 2002)."""
    s = sigma * 0.5 * ALPHA * np.sqrt(k)
    return siegert_rate(mu, sigma, THETA + s, V_R + s)


def naive_rate(mu, sigma, dt):
    """Rate expected when the threshold is checked only on the grid: Siegert
    with theta raised by 0.5826 x (SD of one step's noise increment),
    sigma sqrt(dt/tau) (Broadie, Glasserman & Kou 1997). Reported, not asserted."""
    return siegert_rate(mu, sigma, THETA + BGK * sigma * np.sqrt(dt / TAU), V_R)


def isi_stats(r, K):
    """Rate (Hz) and CV pooled over neurons, with jackknife (over neurons)
    standard errors. Every neuron has exactly K ISIs."""
    s1, s2 = r["s1"], r["s2"]
    n = s1.size
    N = K * n

    def est(S1, S2, NN):
        m = S1 / NN
        v = S2 / NN - m ** 2
        v = v * NN / (NN - 1)
        return 1000.0 / m, np.sqrt(v) / m

    rate, cv = est(s1.sum(), s2.sum(), N)
    jr, jc = est(s1.sum() - s1, s2.sum() - s2, N - K)
    se_r = np.sqrt((n - 1) / n * ((jr - jr.mean()) ** 2).sum())
    se_c = np.sqrt((n - 1) / n * ((jc - jc.mean()) ** 2).sum())
    return rate, cv, se_r, se_c


def size_K(isi_ms, budget):
    return int(np.clip(np.floor(budget / isi_ms), 3, 400))


# ------------------------------------------------------------ simulate
def simulate():
    ss = np.random.SeedSequence(SEED).spawn(6)
    R = {}
    t0 = _time.time()

    # 1. free membrane with coloured input (no threshold)
    rng = np.random.default_rng(ss[0])
    kf, sf, nf = 0.05, 5.0, 2000
    fr = run_coloured(np.zeros(nf), np.full(nf, sf), np.full(nf, kf), np.zeros(nf, int),
                      rng, theta=np.inf, t_max=500.0, free_samples=100)
    Vs = np.array([s[0] for s in fr["samples"]])
    Is = np.array([s[1] for s in fr["samples"]])
    R["free"] = dict(k=kf, sigma=sf, V=Vs, I=Is)
    print(f"  free membrane: {_time.time() - t0:.1f} s")

    # 2. white noise, rate and CV over the (sigma, mu) grid
    t1 = _time.time()
    rng = np.random.default_rng(ss[1])
    sig = np.array([c[0] for c in CONDITIONS])
    mus = np.array([c[1] for c in CONDITIONS])
    pred_isi = np.array([siegert_isi(m, s) for s, m in CONDITIONS])
    Ks = np.array([size_K(x, ISI_BUDGET) for x in pred_isi])
    rw = run_white(np.repeat(mus, N_NEURON), np.repeat(sig, N_NEURON),
                   np.repeat(Ks, N_NEURON), rng)
    rows = []
    for c, (s, m) in enumerate(CONDITIONS):
        sl = slice(c * N_NEURON, (c + 1) * N_NEURON)
        sub = {key: rw[key][sl] for key in ("s1", "s2", "cnt")}
        assert (sub["cnt"] == Ks[c]).all(), f"condition {s, m}: not every neuron reached K ISIs"
        rate, cv, se_r, se_c = isi_stats(sub, Ks[c])
        rows.append(dict(sigma=s, mu=m, K=Ks[c], rate=rate, cv=cv, se_r=se_r, se_c=se_c,
                         rate_th=1000.0 / pred_isi[c], cv_th=brunel_cv(m, s)))
    R["grid"] = rows
    R["grid_t_end"] = rw["t_end"]
    print(f"  white-noise grid: {_time.time() - t1:.1f} s (ran to {rw['t_end']:.0f} ms)")

    # 3. timestep study, bridge vs naive threshold detection
    t1 = _time.time()
    rng = np.random.default_rng(ss[2])
    s, m = STUDY_COND
    Kst = size_K(siegert_isi(m, s), 2000.0)
    study = []
    for dt in DT_STUDY:
        for bridge in (True, False):
            r = run_white(np.full(N_STUDY, m), np.full(N_STUDY, s), np.full(N_STUDY, Kst),
                          rng, dt=dt, bridge=bridge)
            assert (r["cnt"] == Kst).all()
            rate, cv, se_r, se_c = isi_stats(r, Kst)
            study.append(dict(dt=dt, bridge=bridge, rate=rate, se_r=se_r, cv=cv))
    R["study"] = study
    R["study_th"] = siegert_rate(m, s)
    print(f"  timestep study: {_time.time() - t1:.1f} s")

    # 4. coloured (OU) input
    t1 = _time.time()
    rng = np.random.default_rng(ss[3])
    combos = [(s, m, k) for (s, m) in COL_CONDS for k in K_COL]
    Kc = np.array([size_K(1000.0 / fourcaud_brunel_rate(m, s, k), COL_BUDGET) for s, m, k in combos])
    rc = run_coloured(np.repeat([c[1] for c in combos], N_COL),
                      np.repeat([c[0] for c in combos], N_COL),
                      np.repeat([c[2] for c in combos], N_COL),
                      np.repeat(Kc, N_COL), rng)
    col = []
    for c, (s, m, k) in enumerate(combos):
        sl = slice(c * N_COL, (c + 1) * N_COL)
        sub = {key: rc[key][sl] for key in ("s1", "s2", "cnt")}
        assert (sub["cnt"] == Kc[c]).all(), f"coloured {s, m, k}: not every neuron reached K ISIs"
        rate, cv, se_r, se_c = isi_stats(sub, Kc[c])
        col.append(dict(sigma=s, mu=m, k=k, K=Kc[c], rate=rate, se_r=se_r, cv=cv,
                        rate_fb=fourcaud_brunel_rate(m, s, k), rate_white=siegert_rate(m, s)))
    R["col"] = col
    print(f"  coloured input: {_time.time() - t1:.1f} s (ran to {rc['t_end']:.0f} ms)")

    # 5. example traces
    rng = np.random.default_rng(ss[4])
    s, m = STUDY_COND
    R["tr_white"] = run_white([m], [s], [10 ** 6], rng, t_max=500.0, rec=1, t_rec=500.0)
    R["tr_col"] = run_coloured([m], [s], [0.1], [10 ** 6], rng, t_max=500.0, rec=1, t_rec=500.0)
    R["tr_cond"] = (s, m)
    print(f"  total simulate(): {_time.time() - t0:.1f} s")
    return R


# ------------------------------------------------------------ validate
def validate(R):
    print("\n1. Free membrane, coloured input (no threshold)")
    f = R["free"]
    var_V, var_I = f["V"].var(), f["I"].var()
    th_V = f["sigma"] ** 2 / (2.0 * (1.0 + f["k"]))
    th_I = f["sigma"] ** 2 * TAU / (2.0 * f["k"] * TAU)
    eV, eI = var_V / th_V - 1.0, var_I / th_I - 1.0
    print(f"   var V = {var_V:.4f} vs sigma^2/(2(1+k)) = {th_V:.4f} ({eV * 100:+.2f}%)")
    print(f"   var I_s = {var_I:.3f} vs sigma^2 tau/(2 tau_s) = {th_I:.3f} ({eI * 100:+.2f}%)")
    R["free_err"] = (var_V, th_V, eV, var_I, th_I, eI)

    print("\n2-3. White noise: rate vs Siegert, CV vs Brunel (2000)")
    print("   sigma    mu    K    rate  Siegert   err%     z |    CV  theory   err%     z")
    er, ec, zr, zc = [], [], [], []
    for r in R["grid"]:
        e1 = r["rate"] / r["rate_th"] - 1.0
        e2 = r["cv"] / r["cv_th"] - 1.0
        z1 = (r["rate"] - r["rate_th"]) / r["se_r"]
        z2 = (r["cv"] - r["cv_th"]) / r["se_c"]
        er.append(e1); ec.append(e2); zr.append(z1); zc.append(z2)
        print(f"   {r['sigma']:5.1f} {r['mu']:5.1f} {r['K']:4d} {r['rate']:7.3f} {r['rate_th']:8.3f}"
              f" {e1 * 100:+6.2f} {z1:+5.2f} | {r['cv']:6.4f} {r['cv_th']:7.4f} {e2 * 100:+6.2f} {z2:+5.2f}")
    er, ec, zr, zc = map(np.array, (er, ec, zr, zc))
    print(f"   rate: max |err| {np.abs(er).max() * 100:.2f}%, mean |err| {np.abs(er).mean() * 100:.2f}%,"
          f" max |z| {np.abs(zr).max():.2f}, chi2/dof {np.mean(zr ** 2):.2f}")
    print(f"   CV:   max |err| {np.abs(ec).max() * 100:.2f}%, mean |err| {np.abs(ec).mean() * 100:.2f}%,"
          f" max |z| {np.abs(zc).max():.2f}, chi2/dof {np.mean(zc ** 2):.2f}")
    R["grid_err"] = (er, ec, zr, zc)

    print(f"\n4. Timestep study at sigma = {STUDY_COND[0]}, mu = {STUDY_COND[1]} mV;"
          f" Siegert {R['study_th']:.3f} Hz")
    for r in R["study"]:
        e = r["rate"] / R["study_th"] - 1.0
        r["err"] = e
        extra = ""
        if not r["bridge"]:
            r["pred"] = naive_rate(STUDY_COND[1], STUDY_COND[0], r["dt"])
            extra = f"  shifted-threshold prediction {r['pred']:.3f} Hz ({r['rate'] / r['pred'] * 100 - 100:+.2f}%)"
        print(f"   dt = {r['dt']:5.2f} ms  {'bridge' if r['bridge'] else 'naive ':6s}"
              f"  rate {r['rate']:7.3f} +- {r['se_r']:.3f} Hz  ({e * 100:+6.2f}%){extra}")
    nv = [r for r in R["study"] if not r["bridge"]]
    sl = np.polyfit(np.log([r["dt"] for r in nv]), np.log([abs(r["err"]) for r in nv]), 1)[0]
    R["naive_slope"] = sl
    print(f"   naive scheme: log-log slope of |error| vs dt = {sl:.3f} (sqrt(dt) law: 0.5)")

    print(f"\n5. Coloured (OU) input vs Fourcaud & Brunel (2002), alpha = {ALPHA:.4f}")
    print("   sigma    mu      k    rate   F&B   err%    z | white Siegert  err%")
    for r in R["col"]:
        r["err_fb"] = r["rate"] / r["rate_fb"] - 1.0
        r["err_w"] = r["rate"] / r["rate_white"] - 1.0
        r["z_fb"] = (r["rate"] - r["rate_fb"]) / r["se_r"]
        print(f"   {r['sigma']:5.1f} {r['mu']:5.1f} {r['k']:6.3f} {r['rate']:7.3f} {r['rate_fb']:6.3f}"
              f" {r['err_fb'] * 100:+6.2f} {r['z_fb']:+5.2f} | {r['rate_white']:7.3f}"
              f" {r['err_w'] * 100:+7.2f}")

    # ---- assertions
    z_sc = zeta(0.5)
    if np.isfinite(z_sc):
        assert abs(z_sc - ZETA_HALF) < 1e-10, "zeta(1/2) constant wrong"
    assert abs(eV) < VAR_TOL and abs(eI) < VAR_TOL, "free-membrane variance off the closed form"
    assert np.abs(er).max() < RATE_TOL, f"rate off Siegert by {np.abs(er).max() * 100:.2f}%"
    assert np.abs(zr).max() < Z_MAX, f"rate off Siegert by {np.abs(zr).max():.2f} SE"
    assert np.abs(ec).max() < CV_TOL, f"CV off Brunel formula by {np.abs(ec).max() * 100:.2f}%"
    assert np.abs(zc).max() < Z_MAX, f"CV off Brunel formula by {np.abs(zc).max():.2f} SE"
    for r in R["study"]:
        if r["bridge"]:
            assert abs(r["err"]) < DT_TOL, f"bridge scheme off Siegert by {r['err'] * 100:.2f}% at dt = {r['dt']}"
    for r in R["col"]:
        if any(np.isclose(r["k"], FB_ASSERT_K)):
            assert abs(r["err_fb"]) < FB_TOL, (
                f"coloured rate off Fourcaud-Brunel by {r['err_fb'] * 100:.2f}% at k = {r['k']}")
            assert abs(r["err_fb"]) < abs(r["err_w"]), "correction does not improve on white Siegert"
    print("\nAll assertions passed.")


# ------------------------------------------------------------ plot
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    colors = {2.0: "tab:blue", 5.0: "tab:orange", 10.0: "tab:green"}

    # (a) traces
    a = ax[0, 0]
    s, m = R["tr_cond"]
    for tr, off, lab, col in ((R["tr_white"], 0.0, "white noise", "k"),
                              (R["tr_col"], -25.0, "OU input, $\\tau_s$ = 2 ms (offset −25 mV)", "tab:purple")):
        t = np.arange(tr["trace"].shape[0]) * (DT_WHITE if tr is R["tr_white"] else DT_COL)
        a.plot(t, tr["trace"][:, 0] + off, color=col, lw=0.6, label=lab)
        for ts in tr["spikes"][0]:
            a.plot([ts, ts], [THETA + off, THETA + off + 8], color=col, lw=1)
        a.axhline(THETA + off, color=col, ls=":", lw=0.8)
    a.set_xlabel("t (ms)")
    a.set_ylabel("V (mV from rest)")
    a.set_title(f"Traces, $\\mu$ = {m:g} mV, $\\sigma$ = {s:g} mV")
    a.legend(fontsize=8, loc="lower right")

    # (b) rate vs mu
    a = ax[0, 1]
    for sg in sorted(colors):
        rows = [r for r in R["grid"] if r["sigma"] == sg]
        mm = np.linspace(min(r["mu"] for r in rows) - 1, max(r["mu"] for r in rows) + 1, 80)
        a.plot(mm, [siegert_rate(x, sg) for x in mm], color=colors[sg], lw=1.5,
               label=f"Siegert, $\\sigma$ = {sg:g}")
        a.errorbar([r["mu"] for r in rows], [r["rate"] for r in rows],
                   yerr=[2 * r["se_r"] for r in rows], fmt="o", ms=4, color=colors[sg], mfc="white")
    a.set_xlabel("$\\mu$ (mV)")
    a.set_ylabel("rate (Hz)")
    er = R["grid_err"][0]
    a.set_title(f"Rate: simulation (o) vs Siegert, max error {np.abs(er).max() * 100:.2f}%")
    a.legend(fontsize=8)

    # (c) CV vs mu/sigma
    a = ax[0, 2]
    for sg in sorted(colors):
        rows = [r for r in R["grid"] if r["sigma"] == sg]
        mm = np.linspace(min(r["mu"] for r in rows), max(r["mu"] for r in rows), 40)
        a.plot(mm / sg, [brunel_cv(x, sg) for x in mm], color=colors[sg], lw=1.5,
               label=f"Brunel (2000), $\\sigma$ = {sg:g}")
        a.errorbar([r["mu"] / sg for r in rows], [r["cv"] for r in rows],
                   yerr=[2 * r["se_c"] for r in rows], fmt="o", ms=4, color=colors[sg], mfc="white")
    a.set_xscale("log")
    a.set_xlabel("$\\mu/\\sigma$")
    a.set_ylabel("ISI CV")
    ec = R["grid_err"][1]
    a.set_title(f"CV: simulation (o) vs closed form, max error {np.abs(ec).max() * 100:.2f}%")
    a.legend(fontsize=8)

    # (d) timestep study
    a = ax[1, 0]
    sg, mu = STUDY_COND
    dd = np.geomspace(min(DT_STUDY) / 1.5, max(DT_STUDY) * 1.5, 60)
    a.axhline(R["study_th"], color="k", lw=1.5, label="Siegert (dt → 0)")
    a.plot(dd, [naive_rate(mu, sg, x) for x in dd], "r--", lw=1,
           label="Siegert, θ + 0.5826 σ√(dt/τ)")
    for bridge, lab, col in ((True, "exact OU + Brownian bridge", "tab:blue"),
                             (False, "exact OU, threshold on grid only", "tab:red")):
        rows = [r for r in R["study"] if r["bridge"] == bridge]
        a.errorbar([r["dt"] for r in rows], [r["rate"] for r in rows],
                   yerr=[2 * r["se_r"] for r in rows], fmt="o", color=col, label=lab, ms=4)
    a.set_xscale("log")
    a.set_xticks(DT_STUDY)
    a.set_xticklabels([f"{x:g}" for x in DT_STUDY])
    a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a.set_xlabel("dt (ms)")
    a.set_ylabel("rate (Hz)")
    a.set_title(f"Timestep, $\\mu$ = {mu:g}, $\\sigma$ = {sg:g} mV (±2 SE)")
    a.legend(fontsize=8)

    # (e) coloured noise
    a = ax[1, 1]
    kk = np.linspace(0, max(K_COL), 60)
    for (sg, mu), col in zip(COL_CONDS, ("tab:orange", "tab:brown")):
        rows = [r for r in R["col"] if r["sigma"] == sg and r["mu"] == mu]
        a.plot(np.sqrt(kk), [fourcaud_brunel_rate(mu, sg, x) for x in kk], color=col, lw=1.5,
               label=f"Fourcaud & Brunel, $\\mu$ = {mu:g}")
        a.errorbar([np.sqrt(r["k"]) for r in rows], [r["rate"] for r in rows],
                   yerr=[2 * r["se_r"] for r in rows], fmt="o", color=col, mfc="white", ms=4)
        wr = [r for r in R["grid"] if r["sigma"] == sg and r["mu"] == mu]
        if wr:
            a.errorbar([0], [wr[0]["rate"]], yerr=[2 * wr[0]["se_r"]], fmt="s", color=col, ms=5)
        a.axhline(siegert_rate(mu, sg), color=col, ls=":", lw=0.8)
    a.set_xlabel("$\\sqrt{\\tau_s/\\tau}$")
    a.set_ylabel("rate (Hz)")
    a.set_title("OU input, $\\sigma$ = 5 mV: sim (o), white noise (■), Siegert (dotted)")
    a.legend(fontsize=8)

    # (f) free membrane distribution
    a = ax[1, 2]
    f = R["free"]
    var_V, th_V, eV, var_I, th_I, eI = R["free_err"]
    a.hist(f["V"].ravel(), bins=120, density=True, color="tab:purple", alpha=0.5,
           label=f"OU input, k = {f['k']:g}: var {var_V:.3f}")
    x = np.linspace(-12, 12, 300)
    a.plot(x, np.exp(-x ** 2 / (2 * th_V)) / np.sqrt(2 * np.pi * th_V), "k", lw=1.5,
           label=f"$\\sigma^2/(2(1+k))$ = {th_V:.3f}")
    wv = f["sigma"] ** 2 / 2
    a.plot(x, np.exp(-x ** 2 / (2 * wv)) / np.sqrt(2 * np.pi * wv), "k--", lw=1,
           label=f"white noise $\\sigma^2/2$ = {wv:.3f}")
    a.set_xlabel("V − $\\mu$ (mV), no threshold")
    a.set_ylabel("density")
    a.set_title(f"Free membrane: variance error {eV * 100:+.2f}% (V), {eI * 100:+.2f}% (I$_s$)")
    a.legend(fontsize=8)

    fig.suptitle("Noisy leaky integrate-and-fire: diffusion approximation (Siegert) and OU input"
                 " (Fourcaud–Brunel)", fontsize=13)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "noisy_lif.png"
    fig.savefig(out, dpi=150)
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    print(f"RNG seed: {SEED} (PCG64, one spawned stream per experiment)")
    t0 = _time.time()
    R = simulate()
    validate(R)
    plot(R)
    print(f"Runtime {_time.time() - t0:.1f} s")
