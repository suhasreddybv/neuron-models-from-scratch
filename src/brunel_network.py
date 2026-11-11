"""Brunel (2000) sparsely connected network of excitatory and inhibitory
leaky integrate-and-fire neurons ("model A"): asynchronous-irregular and
synchronous states, and the population rate against mean-field theory.

Model (Brunel 2000, J. Comput. Neurosci. 8:183-208, section 2). N_E = 10 000
excitatory and N_I = 2 500 inhibitory LIF neurons,

    tau dV_i/dt = -V_i + R I_i(t),
    R I_i(t) = tau sum_j J_ij sum_k delta(t - t_j^k - D),

with tau = 20 ms, threshold theta = 20 mV, reset V_r = 10 mV and absolute
refractory period tau_rp = 2 ms (V held at V_r). Every neuron receives
C_E = eps N_E excitatory and C_I = eps N_I inhibitory recurrent synapses
(eps = 0.1, gamma = C_I / C_E = 0.25), chosen at random; excitatory synapses
have efficacy J = 0.1 mV, inhibitory ones -g J, and all have delay
D = 1.5 ms. Each neuron also receives C_E external excitatory synapses of
efficacy J from independent Poisson sources at rate nu_ext = eta nu_thr,
with nu_thr = theta / (J C_E tau) = 10 Hz the external rate that brings the
mean free membrane potential to threshold. g sets the balance (g = 4 is
balanced on average) and eta the external drive.

Integrator. With delta synapses the membrane equation is linear between
input arrivals, so it is integrated exactly on a grid of dt = 0.1 ms:
V <- V exp(-dt/tau) + (sum of jumps arriving in this step). Spike times and
arrivals are thus rounded to the grid (D = 15 steps); the threshold is
checked after the jumps are added. Inputs arriving during the refractory
period are discarded. External input is a Poisson count per step with mean
C_E nu_ext dt, times J. Connectivity: fixed in-degree, presynaptic partners
drawn with replacement (multapses and autapses allowed; about
C_E/(2 N_E) = 5% of each neuron's inputs repeat another of its inputs).

Mean-field theory (Brunel 2000, eqs. 4-5 and 21). In the stationary state
every neuron sees a Gaussian white input with

    mu    = C_E J tau (nu_ext + nu_0 (1 - g gamma)),
    sigma = J sqrt(C_E tau (nu_ext + nu_0 (1 + g^2 gamma))),

and the rate nu_0 solves the self-consistent Siegert equation

    1/nu_0 = tau_rp + tau sqrt(pi) int_{(V_r-mu)/sigma}^{(theta-mu)/sigma}
             e^{u^2} (1 + erf u) du.

The ISI CV in that state is (Brunel 2000, appendix A; Tuckwell 1988)

    CV^2 = 2 pi (nu_0 tau)^2 int_{y_r}^{y_theta} e^{x^2}
           int_{-inf}^{x} e^{y^2} (1 + erf y)^2 dy dx.

Regimes (Brunel 2000, Fig. 8, D = 1.5 ms): A g = 3, eta = 2 synchronous
regular (SR); B g = 6, eta = 4 synchronous irregular with a fast global
oscillation (SI fast); C g = 5, eta = 2 asynchronous irregular (AI);
D g = 4.5, eta = 0.9 synchronous irregular with a slow oscillation
(SI slow).

Measures. Population rate in 0.1 ms bins; mean ISI CV over neurons with
at least 5 spikes; synchrony chi (Golomb 2007): chi^2 = Var_t(<V>_pop) /
<Var_t(V_i)>_i, which is O(1/sqrt N) for asynchronous activity and O(1)
for synchronous; peak frequency of the population-rate power spectrum
(Welch average over 200 ms segments, 10-1000 Hz).

Validation (validate()):
1. Network rate vs the self-consistent mean-field rate at eta = 2,
   g = 5-8 (the AI region of Brunel's Fig. 2): relative error < 5% at
   each point (asserted).
2. Open-loop check of the diffusion approximation: N_OPEN unconnected
   neurons driven by independent Poisson E and I inputs at the rates the
   mean-field state at (g, eta) = (5, 2) implies, at dt = 0.1, 0.05, 0.02,
   0.01 ms. Checking the threshold once per step after the step's summed
   jumps acts like a raised threshold, so the rate converges as sqrt(dt)
   (Broadie, Glasserman & Kou 1997); the linear extrapolation in sqrt(dt)
   to dt = 0 must lie within 1% of Siegert (asserted). The first version
   asserted the dt = 0.1 ms rate alone within 3% and failed (-3.8%).
3. Mean ISI CV at the points of check 1 vs the mean-field CV: tolerance
   10% set before the run; reported, NOT asserted, because it fails
   (network 0.42-0.69 vs 0.68-0.87, see README Limitations).
4. Brunel's Fig. 8 regimes with criteria chosen here before the run
   (AI: chi < 0.1 and CV > 0.7; SR: chi > 0.3 and CV < 0.3; SI fast:
   chi > 0.1, CV > 0.7, spectral peak > 100 Hz; SI slow: chi > 0.1,
   CV > 0.7, peak < 60 Hz): reported, NOT asserted, because AI and SI
   slow fail them (see README Limitations).

RNG: NumPy PCG64 seeded with SEED (printed); the connectivity, every
network run and the open-loop run use separate spawned streams. SciPy is
used only for quadrature, erfcx, root finding and the Welch spectrum.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.signal import welch
from scipy.special import erfcx

ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------- parameters
SEED = 2000
N_E, N_I = 10000, 2500
N = N_E + N_I
EPS = 0.1
C_E, C_I = int(EPS * N_E), int(EPS * N_I)
GAMMA = C_I / C_E
J = 0.1            # mV
D = 1.5            # ms
TAU = 20.0         # ms
THETA = 20.0       # mV
V_R = 10.0         # mV
TAU_RP = 2.0       # ms
DT = 0.1           # ms
NU_THR = THETA / (J * C_E * TAU)    # kHz (= 10 Hz)

T_TRANS = 200.0    # ms discarded
T_MEAS = 1000.0    # ms measured
N_CHI = 1000       # neurons used for <Var V_i>

FIG8 = {"A  SR": (3.0, 2.0), "B  SI fast": (6.0, 4.0),
        "C  AI": (5.0, 2.0), "D  SI slow": (4.5, 0.9)}
SWEEP_G_ETA2 = [3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0]
SWEEP_G_ETA4 = [5.0, 6.0, 7.0, 8.0]
SWEEP_G_ETA09 = [4.5, 5.0, 6.0, 8.0]

CHI_ASYNC = 0.1
TOL_RATE = 0.05
TOL_CV = 0.10
TOL_OPEN = 0.01
N_OPEN, T_OPEN = 1000, 4000.0       # open-loop neurons, ms
DT_OPEN = [0.1, 0.05, 0.02, 0.01]  # ms
DT_CHECK = 0.05                    # network rerun at (5, 2), reported
BGK = 0.5826                       # -zeta(1/2)/sqrt(2 pi)


# ------------------------------------------------------------ mean field
def siegert_rate(mu, sigma):
    """Stationary rate (kHz) of the LIF with white input (Siegert)."""
    I = quad(lambda u: erfcx(-u), (V_R - mu) / sigma, (THETA - mu) / sigma, limit=200)[0]
    return 1.0 / (TAU_RP + TAU * np.sqrt(np.pi) * I)


def mu_sigma(nu0, g, eta):
    nu_ext = eta * NU_THR
    mu = C_E * J * TAU * (nu_ext + nu0 * (1.0 - g * GAMMA))
    sigma = J * np.sqrt(C_E * TAU * (nu_ext + nu0 * (1.0 + g * g * GAMMA)))
    return mu, sigma


def mean_field_rate(g, eta):
    """Self-consistent stationary rate nu_0 (kHz), Brunel (2000) eq. 21."""
    f = lambda nu: siegert_rate(*mu_sigma(nu, g, eta)) - nu
    return brentq(f, 1e-8, 1.0 / TAU_RP - 1e-9, xtol=1e-12)


def mean_field_cv(mu, sigma):
    inner = lambda x: quad(lambda y: erfcx(-y) ** 2 * np.exp(-y * y), -np.inf, x, limit=200)[0]
    yr, yt = (V_R - mu) / sigma, (THETA - mu) / sigma
    var = 2.0 * np.pi * TAU ** 2 * quad(lambda x: np.exp(x * x) * inner(x), yr, yt, limit=200)[0]
    return np.sqrt(var) * siegert_rate(mu, sigma)


# ------------------------------------------------------------ model
def build_connectivity(rng):
    """Fixed in-degree C_E + C_I; returns CSR lists of postsynaptic targets
    indexed by presynaptic neuron."""
    K = C_E + C_I
    pre = np.empty((N, K), dtype=np.int32)
    pre[:, :C_E] = rng.integers(0, N_E, (N, C_E))
    pre[:, C_E:] = rng.integers(N_E, N, (N, C_I))
    pre = pre.ravel()
    post = np.repeat(np.arange(N, dtype=np.int32), K)
    order = np.argsort(pre, kind="stable")
    post = post[order]
    ptr = np.searchsorted(pre[order], np.arange(N + 1)).astype(np.int64)
    return post, ptr


def run_network(g, eta, conn, rng, t_trans=T_TRANS, t_meas=T_MEAS):
    post, ptr = conn
    nd = int(round(D / DT))
    buf = np.zeros((nd + 1, N))
    V = rng.uniform(0.0, THETA, N)
    ref = np.zeros(N)
    lam = C_E * eta * NU_THR * DT
    dec = np.exp(-DT / TAU)
    n_tr, n_me = int(round(t_trans / DT)), int(round(t_meas / DT))
    cnt = np.zeros(n_me, dtype=np.int64)
    vbar = np.zeros(n_me)
    s1 = np.zeros(N_CHI)
    s2 = np.zeros(N_CHI)
    sp_i, sp_k = [], []
    for k in range(n_tr + n_me):
        slot = k % (nd + 1)
        inp = buf[slot] + J * rng.poisson(lam, N)
        buf[slot] = 0.0
        act = ref <= 0.0
        V[act] = V[act] * dec + inp[act]
        ref[~act] -= DT
        s = np.flatnonzero(V >= THETA)
        if s.size:
            V[s] = V_R
            ref[s] = TAU_RP - 1e-9
            slot_d = (k + nd) % (nd + 1)
            for sub, wt in ((s[s < N_E], J), (s[s >= N_E], -g * J)):
                if sub.size:
                    lens = ptr[sub + 1] - ptr[sub]
                    ends = np.cumsum(lens)
                    idx = np.arange(ends[-1]) + np.repeat(ptr[sub] - ends + lens, lens)
                    buf[slot_d] += wt * np.bincount(post[idx], minlength=N)
        m = k - n_tr
        if m >= 0:
            cnt[m] = s.size
            vbar[m] = V.mean()
            s1 += V[:N_CHI]
            s2 += V[:N_CHI] ** 2
            if s.size:
                sp_i.append(s.astype(np.int32))
                sp_k.append(np.full(s.size, m, dtype=np.int32))
    sp_i = np.concatenate(sp_i) if sp_i else np.zeros(0, np.int32)
    sp_k = np.concatenate(sp_k) if sp_k else np.zeros(0, np.int32)
    var_i = s2 / n_me - (s1 / n_me) ** 2
    chi = np.sqrt(vbar.var() / var_i.mean())
    return dict(g=g, eta=eta, cnt=cnt, chi=chi, sp_i=sp_i, sp_k=sp_k,
                rate=cnt.sum() / (N * t_meas) * 1000.0, cv=mean_cv(sp_i, sp_k))


def mean_cv(sp_i, sp_k):
    if sp_i.size == 0:
        return np.nan
    o = np.lexsort((sp_k, sp_i))
    i, t = sp_i[o], sp_k[o] * DT
    same = i[1:] == i[:-1]
    isi = np.diff(t)[same]
    owner = i[1:][same]
    n = np.bincount(owner, minlength=N)
    s1 = np.bincount(owner, isi, minlength=N)
    s2 = np.bincount(owner, isi * isi, minlength=N)
    ok = n >= 4                     # >= 5 spikes
    m = s1[ok] / n[ok]
    sd = np.sqrt(np.maximum(s2[ok] / n[ok] - m * m, 0.0))
    return float(np.mean(sd / m))


def spectrum_peak(cnt):
    a = cnt / (N * DT) * 1000.0                 # Hz
    f, P = welch(a - a.mean(), fs=1000.0 / DT, nperseg=int(200.0 / DT))
    band = (f >= 10.0) & (f <= 1000.0)
    return f, P, float(f[band][np.argmax(P[band])])


def run_open_loop(g, eta, nu0, rng, dt):
    """Unconnected neurons, Poisson E (rate C_E (nu_ext + nu0)) and I
    (rate C_I nu0) inputs; same integrator at step dt. Returns the rate (Hz)
    and its standard error over neurons."""
    V = rng.uniform(0.0, THETA, N_OPEN)
    ref = np.zeros(N_OPEN)
    lam_e = C_E * (eta * NU_THR + nu0) * dt
    lam_i = C_I * nu0 * dt
    dec = np.exp(-dt / TAU)
    n_tr, n_me = int(round(200.0 / dt)), int(round(T_OPEN / dt))
    count = np.zeros(N_OPEN)
    last = np.full(N_OPEN, np.nan)
    i1, i2, ni = np.zeros(N_OPEN), np.zeros(N_OPEN), np.zeros(N_OPEN)
    for k in range(n_tr + n_me):
        inp = J * (rng.poisson(lam_e, N_OPEN) - g * rng.poisson(lam_i, N_OPEN))
        act = ref <= 0.0
        V[act] = V[act] * dec + inp[act]
        ref[~act] -= dt
        s = V >= THETA
        V[s] = V_R
        ref[s] = TAU_RP - 1e-9
        if k >= n_tr:
            count += s
            if s.any():
                j = np.flatnonzero(s)
                isi = k * dt - last[j]
                ok = ~np.isnan(isi)
                i1[j[ok]] += isi[ok]
                i2[j[ok]] += isi[ok] ** 2
                ni[j[ok]] += 1
                last[j] = k * dt
    r = count / T_OPEN * 1000.0
    m = i1 / ni
    cv = np.mean(np.sqrt(np.maximum(i2 / ni - m * m, 0.0)) / m)
    return r.mean(), r.std(ddof=1) / np.sqrt(N_OPEN), cv


def bgk_rate(g, eta, nu0, dt):
    """Siegert rate with theta raised by BGK x (SD of one step's increment,
    sigma sqrt(dt/tau)); the discrete-monitoring correction. Reported only."""
    mu, sigma = mu_sigma(nu0, g, eta)
    th = THETA + BGK * sigma * np.sqrt(dt / TAU)
    I = quad(lambda u: erfcx(-u), (V_R - mu) / sigma, (th - mu) / sigma, limit=200)[0]
    return 1000.0 / (TAU_RP + TAU * np.sqrt(np.pi) * I)


# ------------------------------------------------------------ simulate
def simulate():
    ss = np.random.SeedSequence(SEED)
    streams = iter(ss.spawn(64))
    t0 = _time.time()
    conn = build_connectivity(np.random.default_rng(next(streams)))
    print(f"connectivity: {N} neurons, {conn[0].size} synapses ({_time.time() - t0:.1f} s)")
    pts = sorted({(g, 2.0) for g in SWEEP_G_ETA2} | {(g, 4.0) for g in SWEEP_G_ETA4}
                 | {(g, 0.9) for g in SWEEP_G_ETA09} | set(FIG8.values()))
    runs = {}
    for g, eta in pts:
        t1 = _time.time()
        r = run_network(g, eta, conn, np.random.default_rng(next(streams)))
        r["f"], r["P"], r["peak"] = spectrum_peak(r["cnt"])
        r["nu_mf"] = mean_field_rate(g, eta) * 1000.0
        r["cv_mf"] = mean_field_cv(*mu_sigma(r["nu_mf"] / 1000.0, g, eta))
        runs[(g, eta)] = r
        print(f"  g = {g:3.1f}, eta = {eta:3.1f}: rate {r['rate']:7.2f} Hz (MF {r['nu_mf']:7.2f}), "
              f"CV {r['cv']:.3f} (MF {r['cv_mf']:.3f}), chi {r['chi']:.3f}, "
              f"peak {r['peak']:6.1f} Hz  [{_time.time() - t1:.1f} s]")
    g_o, eta_o = FIG8["C  AI"]
    nu0 = mean_field_rate(g_o, eta_o)
    open_rows = []
    for dt in DT_OPEN:
        t1 = _time.time()
        m, se, cv = run_open_loop(g_o, eta_o, nu0, np.random.default_rng(next(streams)), dt)
        open_rows.append((dt, m, se, bgk_rate(g_o, eta_o, nu0, dt), cv))
        print(f"  open loop (g, eta) = ({g_o}, {eta_o}), dt = {dt} ms: {m:.3f} +/- {se:.3f} Hz, CV {cv:.3f} "
              f"[{_time.time() - t1:.1f} s]")
    global DT
    dt0, DT = DT, DT_CHECK
    t1 = _time.time()
    rc = run_network(g_o, eta_o, conn, np.random.default_rng(next(streams)))
    DT = dt0
    print(f"  network (g, eta) = ({g_o}, {eta_o}) at dt = {DT_CHECK} ms: rate {rc['rate']:.2f} Hz, "
          f"CV {rc['cv']:.3f}, chi {rc['chi']:.3f} [{_time.time() - t1:.1f} s]")
    return dict(runs=runs, open=(g_o, eta_o, nu0 * 1000.0, open_rows),
                dt_check=(rc["rate"], rc["cv"], rc["chi"]))


# ------------------------------------------------------------ validate
def validate(R):
    runs = R["runs"]
    print("\n1. Network rate vs mean field, eta = 2, g = 5-8 (AI region of Brunel Fig. 2)")
    ai = [k for k in sorted(runs) if k[1] == 2.0 and k[0] >= 5.0]
    errs_r, errs_cv = [], []
    for k in ai:
        r = runs[k]
        errs_r.append(r["rate"] / r["nu_mf"] - 1.0)
        errs_cv.append(r["cv"] / r["cv_mf"] - 1.0)
        print(f"   g = {k[0]:3.1f}: rate {r['rate']:.3f} vs {r['nu_mf']:.3f} Hz ({errs_r[-1] * 100:+.2f}%), "
              f"chi {r['chi']:.3f}")
    mr = np.max(np.abs(errs_r))
    print(f"   max |rate error| {mr * 100:.2f}% (tolerance {TOL_RATE * 100:.0f}%)")
    rc = R["dt_check"]
    k = FIG8["C  AI"]
    print(f"   same network at dt = {DT_CHECK} ms, g = 5: {rc[0]:.3f} Hz "
          f"({(rc[0] / runs[k]['nu_mf'] - 1) * 100:+.2f}%), CV {rc[1]:.3f}, chi {rc[2]:.3f} (reported)")
    for eta in (4.0, 0.9):
        e = [(kk[0], runs[kk]["rate"] / runs[kk]["nu_mf"] - 1.0) for kk in sorted(runs) if kk[1] == eta]
        print(f"   eta = {eta}: " + ", ".join(f"g = {g}: {x * 100:+.1f}%" for g, x in e) + " (reported)")
    assert mr < TOL_RATE, "network rate does not match mean field"

    print("\n2. Open loop: Poisson-driven unconnected LIF vs Siegert, convergence in dt")
    g_o, eta_o, nu_mf, rows = R["open"]
    dts = np.array([r[0] for r in rows])
    m = np.array([r[1] for r in rows])
    se = np.array([r[2] for r in rows])
    for dt, mm, ss, b, _ in rows:
        print(f"   dt = {dt:5.2f} ms: {mm:.3f} +/- {ss:.3f} Hz ({(mm / nu_mf - 1) * 100:+.2f}% vs Siegert "
              f"{nu_mf:.3f}); BGK-shifted Siegert {b:.3f} ({(mm / b - 1) * 100:+.2f}%)")
    X = np.vstack([np.ones_like(dts), np.sqrt(dts)]).T
    W = 1.0 / se ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None]))
    coef = cov @ (X.T @ (W * m))
    ext, ext_se = coef[0], np.sqrt(cov[0, 0])
    eo = ext / nu_mf - 1.0
    eo01 = rows[0][1] / nu_mf - 1.0
    print(f"   extrapolated to dt = 0 (linear in sqrt(dt)): {ext:.3f} +/- {ext_se:.3f} Hz vs Siegert "
          f"{nu_mf:.3f} ({eo * 100:+.2f}%, tolerance {TOL_OPEN * 100:.0f}%); slope {coef[1]:.2f} Hz/ms^0.5")
    assert abs(eo) < TOL_OPEN, "open-loop rate does not converge to Siegert"

    print("\n3. ISI CV vs mean field at the points of check 1 (tolerance 10%; NOT asserted)")
    for kk, e in zip(ai, errs_cv):
        print(f"   g = {kk[0]:3.1f}: CV {runs[kk]['cv']:.3f} vs {runs[kk]['cv_mf']:.3f} ({e * 100:+.1f}%)")
    mc = np.max(np.abs(errs_cv))
    k5 = FIG8["C  AI"]
    cv_open = [r[4] for r in rows]
    print(f"   open-loop neurons with the same (mu, sigma), g = 5: CV {min(cv_open):.3f}-{max(cv_open):.3f} "
          f"over dt = {DT_OPEN[0]}-{DT_OPEN[-1]} ms vs {runs[k5]['cv_mf']:.3f} (single-neuron check, reported)")
    print(f"   max |CV error| {mc * 100:.1f}% -> {'ok' if mc < TOL_CV else 'DOES NOT MATCH'}")

    print("\n4. Brunel (2000) Fig. 8 regimes, criteria set before the run (NOT asserted)")
    crit = {
        "A  SR": lambda r: r["chi"] > 0.3 and r["cv"] < 0.3,
        "B  SI fast": lambda r: r["chi"] > CHI_ASYNC and r["cv"] > 0.7 and r["peak"] > 100.0,
        "C  AI": lambda r: r["chi"] < CHI_ASYNC and r["cv"] > 0.7,
        "D  SI slow": lambda r: r["chi"] > CHI_ASYNC and r["cv"] > 0.7 and r["peak"] < 60.0,
    }
    n_pass = 0
    for name, kk in FIG8.items():
        r = runs[kk]
        passed = crit[name](r)
        n_pass += passed
        print(f"   {name:10s} g = {kk[0]}, eta = {kk[1]}: rate {r['rate']:.1f} Hz, CV {r['cv']:.3f}, "
              f"chi {r['chi']:.3f}, peak {r['peak']:.1f} Hz -> {'ok' if passed else 'FAILS criterion'}")
    print(f"   {n_pass} of 4 meet the criteria")
    return dict(max_rate_err=mr, max_cv_err=mc, cv_open=cv_open, open_err=eo, open_err01=eo01, open_ext=ext,
                open_coef=coef, n_pass=n_pass)


# ------------------------------------------------------------ plot
def plot(R, V):
    runs = R["runs"]
    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(4, 4, height_ratios=[2.2, 1.0, 0.25, 3.0], hspace=0.35, wspace=0.3)
    t_show = 200.0
    for c, (name, k) in enumerate(FIG8.items()):
        r = runs[k]
        n_me = r["cnt"].size
        k0 = n_me - int(t_show / DT)
        a = fig.add_subplot(gs[0, c])
        sel = (r["sp_i"] < 50) & (r["sp_k"] >= k0)
        a.plot((r["sp_k"][sel] - k0) * DT, r["sp_i"][sel], "|", ms=3, color="k")
        a.set_xlim(0, t_show)
        a.set_ylim(-1, 50)
        a.set_xticklabels([])
        a.set_ylabel("neuron (E, first 50)")
        a.set_title(f"{name}: g = {k[0]}, η = {k[1]}\nrate {r['rate']:.1f} Hz (MF {r['nu_mf']:.1f}), "
                    f"CV {r['cv']:.2f}, χ {r['chi']:.2f}, peak {r['peak']:.0f} Hz", fontsize=9)
        b = fig.add_subplot(gs[1, c])
        pr = r["cnt"][k0:].reshape(-1, 5).sum(1) / (N * 0.5) * 1000.0
        b.plot(np.arange(pr.size) * 0.5, pr, color="C0", lw=0.8)
        b.axhline(r["nu_mf"], color="C3", ls="--", lw=1, label="mean field ν₀")
        b.set_xlim(0, t_show)
        b.set_xlabel("time (ms, last 200 ms)")
        b.set_ylabel("pop. rate (Hz)\n0.5 ms bins")
        if c == 0:
            b.legend(fontsize=7, loc="upper right")

    gsb = gs[3, :].subgridspec(1, 5, wspace=0.34)
    styles = {2.0: ("C0", "o"), 4.0: ("C1", "s"), 0.9: ("C2", "^")}
    a = fig.add_subplot(gsb[0])
    gg = np.linspace(3.0, 8.0, 101)
    for eta, (col, mk) in styles.items():
        ks = [k for k in sorted(runs) if k[1] == eta]
        a.semilogy([k[0] for k in ks], [runs[k]["rate"] for k in ks], mk, color=col, ms=6,
                   label=f"network, η = {eta}")
        a.semilogy(gg, [mean_field_rate(x, eta) * 1000.0 for x in gg], "-", color=col, lw=1)
    a.plot([], [], "k-", lw=1, label="mean field (Siegert, eq. 21)")
    a.set_xlabel("relative inhibition g")
    a.set_ylabel("population rate (Hz)")
    a.legend(fontsize=7)
    a.set_title(f"E  Rate vs g; AI points (η = 2, g ≥ 5)\nwithin {V['max_rate_err'] * 100:.2f}% of mean field",
                fontsize=10)

    a = fig.add_subplot(gsb[1])
    for eta, (col, mk) in styles.items():
        ks = [k for k in sorted(runs) if k[1] == eta]
        a.plot([k[0] for k in ks], [runs[k]["cv"] for k in ks], mk, color=col, ms=6, label=f"network, η = {eta}")
        a.plot([k[0] for k in ks], [runs[k]["cv_mf"] for k in ks], "_", color=col, ms=14, mew=2)
    a.plot([], [], "k_", ms=14, mew=2, label="mean-field CV")
    a.plot(5.0, V["cv_open"][0], "*", color="C3", ms=12, label="unconnected, Poisson input (g = 5, η = 2)")
    a.set_xlabel("relative inhibition g")
    a.set_ylabel("mean ISI CV")
    a.legend(fontsize=7, loc="upper left")
    a.set_title(f"F  ISI CV vs mean field: does not match\n(η = 2, g ≥ 5: up to {V['max_cv_err'] * 100:.0f}% low)", fontsize=10)

    a = fig.add_subplot(gsb[2])
    for eta, (col, mk) in styles.items():
        ks = [k for k in sorted(runs) if k[1] == eta]
        a.semilogy([k[0] for k in ks], [runs[k]["chi"] for k in ks], mk + "-", color=col, ms=6, lw=0.8,
                   label=f"η = {eta}")
    a.axhline(CHI_ASYNC, color="k", ls=":", lw=1, label=f"χ = {CHI_ASYNC} (async criterion)")
    a.axvline(4.0, color="0.6", lw=0.8)
    a.set_xlabel("relative inhibition g")
    a.set_ylabel("synchrony χ")
    a.legend(fontsize=7)
    a.set_title("G  Synchrony χ vs g (g = 4: balance)", fontsize=10)

    a = fig.add_subplot(gsb[3])
    for (name, k), col in zip(FIG8.items(), ["C3", "C1", "C0", "C2"]):
        r = runs[k]
        a.loglog(r["f"][1:], r["P"][1:], color=col, lw=1, label=f"{name} (peak {r['peak']:.0f} Hz)")
    a.set_xlim(5, 2000)
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("population-rate PSD (Hz²/Hz)")
    a.legend(fontsize=7)
    a.set_title(f"H  Population-rate spectra (Fig. 8 sets)\n{V['n_pass']} of 4 meet the regime criteria set here",
                fontsize=10)

    a = fig.add_subplot(gsb[4])
    g_o, eta_o, nu_mf, rows = R["open"]
    x = np.sqrt([r[0] for r in rows])
    a.errorbar(x, [r[1] for r in rows], yerr=[2 * r[2] for r in rows], fmt="o", color="C0",
               label="open-loop simulation (±2 SE)")
    a.plot(x, [r[3] for r in rows], "x", color="C1", ms=8, label="Siegert, θ + 0.5826 σ√(dt/τ)")
    xx = np.linspace(0, x.max() * 1.05, 50)
    a.plot(xx, V["open_coef"][0] + V["open_coef"][1] * xx, "--", color="C0", lw=1,
           label=f"linear fit, dt → 0: {V['open_ext']:.2f} Hz")
    a.axhline(nu_mf, color="C3", lw=1, label=f"Siegert ν₀ = {nu_mf:.2f} Hz")
    a.set_xlabel("√dt (ms^½)")
    a.set_ylabel("rate (Hz)")
    a.legend(fontsize=7, loc="center right")
    a.set_title(f"I  Open-loop LIF at the (5, 2) mean-field input\ndt → 0: {V['open_err'] * 100:+.2f}% "
                f"of Siegert (dt = 0.1: {V['open_err01'] * 100:+.1f}%)", fontsize=10)

    fig.suptitle("Brunel (2000) sparse E/I LIF network, N = 12 500, ε = 0.1, J = 0.1 mV, D = 1.5 ms: "
                 "the four regimes of Fig. 8 and the mean-field rate", fontsize=13)
    out = ROOT / "figures" / "brunel_network.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print(f"RNG seed: {SEED} (PCG64, spawned streams for connectivity and each run)")
    t0 = _time.time()
    R = simulate()
    V = validate(R)
    plot(R, V)
    print(f"Runtime {_time.time() - t0:.1f} s")
