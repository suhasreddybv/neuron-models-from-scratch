"""Triplet spike-timing-dependent plasticity (Pfister & Gerstner 2006) and its
frequency dependence: the Sjöström, Turrigiano & Nelson (2001) pairing
protocol, the drift under Poisson input, and the BCM-like threshold on the
postsynaptic rate.

Rule (all-to-all interactions, Pfister & Gerstner 2006, eqs. 1-4), t in ms.
Each synapse has two presynaptic traces r1, r2 and sees two postsynaptic
traces o1, o2:

    dr1/dt = -r1/tau_+,  dr2/dt = -r2/tau_x,   r1, r2 -> +1 at a pre spike
    do1/dt = -o1/tau_-,  do2/dt = -o2/tau_y,   o1, o2 -> +1 at a post spike
    at a pre spike:   w -> w - o1(t) [A2- + A3- r2(t - eps)]
    at a post spike:  w -> w + r1(t) [A2+ + A3+ o2(t - eps)]

eps -> 0+ means the weight update reads r2 and o2 before the spike's own
increment (and r1, o1 are not incremented by the spike that reads them). The
pair terms (A2+-) give the usual exponential STDP window; the triplet terms
make potentiation grow with the postsynaptic rate (post-pre-post triplets)
and depression with the presynaptic rate (pre-post-pre). The weight is
additive and unbounded, as in the paper's fits.

Integrator. The traces are linear, so between spikes they are advanced by
their exact exponential decay (event-driven; no discretisation error and no
dt). All experiments are spike-train driven: regular pairing trains and
independent Poisson trains.

Parameters. Pfister & Gerstner (2006) Table 3 (visual cortex, fitted to
Sjöström et al. 2001), tau_+ = 16.8 ms, tau_- = 33.7 ms fixed:
  all-to-all full:    A2+ = 5e-10, A3+ = 6.2e-3, A2- = 7e-3, A3- = 2.3e-4,
                      tau_x = 101 ms, tau_y = 125 ms   (paper's error E = 0.33)
  all-to-all minimal: A2+ = 0, A3+ = 6.5e-3, A2- = 7.1e-3, A3- = 0,
                      tau_y = 114 ms                   (paper's error E = 0.34)
  pair-only control:  the full set with A3+ = A3- = 0.
Pairing protocol as simulated by Pfister & Gerstner (Methods): n = 60 pairs
at regular intervals 1/rho, post shifted by dt = t_post - t_pre = +-10 ms,
rho = 0.1, 10, 20, 40, 50 Hz. Data (Pfister & Gerstner 2006 Table 1, from
Sjöström 2001): weight change, mean +- SEM,
  dt = +10 ms:  0.04 +- 0.05, 0.14 +- 0.10, 0.29 +- 0.14, 0.53 +- 0.11, 0.56 +- 0.26
  dt = -10 ms: -0.29 +- 0.08, -0.41 +- 0.11, -0.34 +- 0.10, 0.56 +- 0.32, 0.75 +- 0.19
(signs were lost in our copy of the table and are restored from the text and
figure: LTD for post-pre below 20 Hz, LTP for both orders at 40-50 Hz; the
sign of the 0.04 at 0.1 Hz is not recoverable and does not matter, see check 3).

Closed forms used for validation (independent of the trace code):

  Exact all-to-all sum for any trains (direct sums over spike pairs):
    dw = sum_{post j} R1_j [A2+ + A3+ O2_j] - sum_{pre i} O1_i [A2- + A3- R2_i],
    R1_j = sum_{pre i: t_i < t_j} e^{-(t_j - t_i)/tau_+},
    O2_j = sum_{post k: t_k < t_j} e^{-(t_j - t_k)/tau_y}, and likewise O1_i, R2_i.
  Periodic steady state, period T, post lagging pre by d = dt mod T in (0, T):
    with g(tau, a) = e^{-a/tau} / (1 - e^{-T/tau}),
    dw per period = g(tau_+, d) [A2+ + A3+ g(tau_y, T)]
                    - g(tau_-, T - d) [A2- + A3- g(tau_x, T)].
  Independent Poisson pre (rate rho_x) and post (rho_y) trains, traces
  starting at 0 (derived here; the stationary limit is Pfister & Gerstner
  eq. 6): E[r1(t)] = rho_x tau_+ (1 - e^{-t/tau_+}), and r1 and o2 at a post
  spike are independent, so
    E[dw(T)] = rho_x rho_y [A2+ tau_+ I1(tau_+) + A3+ rho_y tau_+ tau_y I2(tau_+, tau_y)
                            - A2- tau_- I1(tau_-) - A3- rho_x tau_- tau_x I2(tau_-, tau_x)],
    I1(a) = T - a(1 - e^{-T/a}),
    I2(a, b) = T - a(1 - e^{-T/a}) - b(1 - e^{-T/b}) + c(1 - e^{-T/c}),  c = ab/(a + b),
  -> T rho_x rho_y [A2+ tau_+ - A2- tau_- - A3- tau_- tau_x rho_x + A3+ tau_+ tau_y rho_y].
  The drift is zero at the BCM-like threshold on the postsynaptic rate
    theta(rho_x) = (A2- tau_- + A3- tau_- tau_x rho_x - A2+ tau_+) / (A3+ tau_+ tau_y)
  (finite-T version: the same with I1, I2 / T factors).
  Normalised error of the fit (Pfister & Gerstner eq. 5):
    E = (1/p) sum_i ((dw_i^exp - dw_i^model) / SEM_i)^2,  p = 10.

Validation (validate()):
1. Pairing protocol, 60 pairs, rho = 0.1-60 Hz, dt = +-10 ms, both parameter
   sets and the control: trace implementation vs exact direct sum, and on
   5 independent 20 Hz Poisson train pairs over 20 s; max |error| < 1e-10.
2. 400-pair trains at the five protocol frequencies: increment of the last
   pair vs the periodic steady-state closed form; max |error| < 1e-12.
3. Sjöström frequency dependence (the roadmap's qualitative criterion,
   criteria chosen here): for the full set, pre-post |dw| < 0.01 at 0.1 Hz
   and increasing with rho; post-pre LTD at 0.1, 10, 20 Hz and LTP at 50 Hz,
   sign change (brentq on the direct sum) between 20 and 50 Hz; every model
   point within 3 SEM of the data. The pair-only control must fail (LTD for
   post-pre at every frequency). Quantitative: E from the 10 data points vs
   the published E (0.33 full, 0.34 minimal); asserted to lie inside the
   interval of E values allowed by the rounding of Table 1 (means and SEMs
   given to +-0.005) widened by the rounding of the published E (+-0.005).
4. Poisson drift: 1000 independent synapses per condition, T = 100 s,
   rho_y = 2, 5, 10, 15, 20, 30, 40 Hz; full set at rho_x = 5, 10, 20 Hz and
   minimal set at rho_x = 10 Hz (28 conditions). Sample mean of dw(T) vs
   E[dw(T)]; |z| < 4 in every condition (standard error from the sample).
5. BCM threshold: weighted least-squares line through mean dw/(rho_x rho_y T)
   vs rho_y in each of the 4 groups, zero crossing vs the finite-T closed
   form theta; relative error < 3%.

RNG: NumPy PCG64 seeded with SEED (printed), one spawned stream per
experiment. SciPy is used only for brentq in check 3.
"""

from pathlib import Path
import math
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ---------------------------------------------------------------- parameters
SEED = 2006

TAU_P, TAU_M = 16.8, 33.7                                   # ms
FULL = dict(A2p=5e-10, A3p=6.2e-3, A2m=7e-3, A3m=2.3e-4, tx=101.0, ty=125.0)
MINIMAL = dict(A2p=0.0, A3p=6.5e-3, A2m=7.1e-3, A3m=0.0, tx=101.0, ty=114.0)
PAIR_ONLY = dict(FULL, A3p=0.0, A3m=0.0)
E_PUBLISHED = dict(full=0.33, minimal=0.34)

N_PAIRS = 60
LAGS = (10.0, -10.0)                                        # t_post - t_pre (ms)
FREQS = np.array([0.1, 10.0, 20.0, 40.0, 50.0])             # Hz
FREQ_GRID = np.concatenate([[0.1, 1.0, 2.0], np.arange(5.0, 60.1, 2.5)])
DATA_MEAN = {10.0: np.array([0.04, 0.14, 0.29, 0.53, 0.56]),
             -10.0: np.array([-0.29, -0.41, -0.34, 0.56, 0.75])}
DATA_SEM = {10.0: np.array([0.05, 0.10, 0.14, 0.11, 0.26]),
            -10.0: np.array([0.08, 0.11, 0.10, 0.32, 0.19])}
N_LONG = 400

# Poisson experiments (rates in Hz; converted to /ms internally)
POIS_T = 100_000.0                                          # ms
POIS_M = 1000
RHO_Y = np.array([2.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0])
GROUPS = [("full", FULL, 5.0), ("full", FULL, 10.0), ("full", FULL, 20.0),
          ("minimal", MINIMAL, 10.0)]

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- model
def triplet_trains(pre, post, P, history=False):
    """Event-driven all-to-all triplet STDP for one synapse, w0 = 0.
    Returns the final weight change (and the weight after every pre spike)."""
    ev = sorted([(t, 0) for t in pre] + [(t, 1) for t in post])
    r1 = r2 = o1 = o2 = 0.0
    w, tl = 0.0, ev[0][0]
    hist = []
    for t, is_post in ev:
        d = t - tl
        tl = t
        r1 *= math.exp(-d / TAU_P)
        r2 *= math.exp(-d / P["tx"])
        o1 *= math.exp(-d / TAU_M)
        o2 *= math.exp(-d / P["ty"])
        if is_post:
            w += r1 * (P["A2p"] + P["A3p"] * o2)      # o2 read before increment
            o1 += 1.0
            o2 += 1.0
        else:
            w -= o1 * (P["A2m"] + P["A3m"] * r2)      # r2 read before increment
            r1 += 1.0
            r2 += 1.0
            hist.append(w)
    return (w, np.array(hist)) if history else w


def pairing_trains(freq, lag, n=N_PAIRS):
    T = 1000.0 / freq
    pre = 100.0 + T * np.arange(n)
    return pre, pre + lag


def poisson_open_loop(rng, rho_x, rho_y, P, T):
    """Independent synapses (one per element of the rate/parameter arrays),
    each with independent Poisson pre (rho_x) and post (rho_y) trains in /ms,
    generated as one Poisson stream of rate rho_x + rho_y with each event
    labelled pre with probability rho_x/(rho_x + rho_y). Vectorised over
    synapses; events after T are ignored."""
    N = rho_x.size
    lam, p_pre = rho_x + rho_y, rho_x / (rho_x + rho_y)
    r1, r2, o1, o2, w, t = (np.zeros(N) for _ in range(6))
    dp, dx = 1.0 / TAU_P, 1.0 / P["tx"]
    dm, dy = 1.0 / TAU_M, 1.0 / P["ty"]
    block = 500
    while t.min() <= T:
        dts = rng.exponential(1.0, (block, N)) / lam
        ispre = rng.random((block, N)) < p_pre
        for j in range(block):
            d = dts[j]
            t += d
            r1 *= np.exp(-d * dp)
            r2 *= np.exp(-d * dx)
            o1 *= np.exp(-d * dm)
            o2 *= np.exp(-d * dy)
            pre = ispre[j]
            dw = np.where(pre, -o1 * (P["A2m"] + P["A3m"] * r2),
                          r1 * (P["A2p"] + P["A3p"] * o2))
            dw[t > T] = 0.0
            w += dw
            r1 += pre
            r2 += pre
            o1 += ~pre
            o2 += ~pre
    return w


# ---------------------------------------------------------------- closed forms
def direct_sum(pre, post, P):
    """Exact all-to-all triplet sum by direct summation over spike pairs."""
    pre, post = np.asarray(pre, float), np.asarray(post, float)
    def lagged(a, b, tau):                    # sum over b < a of e^{-(a-b)/tau}
        d = a[:, None] - b[None, :]
        return np.where(d > 0, np.exp(-np.where(d > 0, d, 0) / tau), 0).sum(1)
    R1 = lagged(post, pre, TAU_P)
    O2 = lagged(post, post, P["ty"])
    O1 = lagged(pre, post, TAU_M)
    R2 = lagged(pre, pre, P["tx"])
    return (R1 * (P["A2p"] + P["A3p"] * O2)).sum() - (O1 * (P["A2m"] + P["A3m"] * R2)).sum()


def periodic_per_pair(freq, lag, P):
    T = 1000.0 / freq
    d = lag % T
    g = lambda tau, a: math.exp(-a / tau) / (1 - math.exp(-T / tau))
    return (g(TAU_P, d) * (P["A2p"] + P["A3p"] * g(P["ty"], T))
            - g(TAU_M, T - d) * (P["A2m"] + P["A3m"] * g(P["tx"], T)))


def I1(a, T):
    return T - a * (1 - math.exp(-T / a))


def I2(a, b, T):
    c = a * b / (a + b)
    return T - a * (1 - math.exp(-T / a)) - b * (1 - math.exp(-T / b)) + c * (1 - math.exp(-T / c))


def drift_coeffs(P, rho_x, T):
    """E[dw(T)] = rho_x rho_y (c0 + c1 rho_y), rates in /ms."""
    c0 = (P["A2p"] * TAU_P * I1(TAU_P, T) - P["A2m"] * TAU_M * I1(TAU_M, T)
          - P["A3m"] * rho_x * TAU_M * P["tx"] * I2(TAU_M, P["tx"], T))
    c1 = P["A3p"] * TAU_P * P["ty"] * I2(TAU_P, P["ty"], T)
    return c0, c1


def theta_stationary(P, rho_x):
    return ((P["A2m"] * TAU_M + P["A3m"] * TAU_M * P["tx"] * rho_x - P["A2p"] * TAU_P)
            / (P["A3p"] * TAU_P * P["ty"]))


def fit_error(model):
    """Pfister & Gerstner eq. 5 with Table 1 data; plus its range under the
    rounding of the table (+-0.005 on every mean and SEM)."""
    terms, lo, hi = [], [], []
    for lag in LAGS:
        for m, s, y in zip(DATA_MEAN[lag], DATA_SEM[lag], model[lag]):
            # the 0.1 Hz pre-post sign is unknown: take both signs for the range
            means = [m, -m] if (lag > 0 and m == 0.04) else [m]
            dmin, dmax = np.inf, 0.0
            for mm in means:
                a, b = mm - 0.005 - y, mm + 0.005 - y
                dmin = min(dmin, 0.0 if a * b <= 0 else min(abs(a), abs(b)))
                dmax = max(dmax, abs(a), abs(b))
            terms.append(((m - y) / s) ** 2)
            lo.append((dmin / (s + 0.005)) ** 2)
            hi.append((dmax / (s - 0.005)) ** 2)
    p = len(terms)
    return sum(terms) / p, sum(lo) / p, sum(hi) / p


# ---------------------------------------------------------------- simulate
def simulate():
    ss = np.random.SeedSequence(SEED)
    rng_trains, rng_pois = [np.random.default_rng(s) for s in ss.spawn(2)]
    R = {}
    sets = dict(full=FULL, minimal=MINIMAL, pair=PAIR_ONLY)
    tt = _time.time()
    # 1. pairing protocol on a frequency grid, trace code and direct sum
    for name, P in sets.items():
        for lag in LAGS:
            tr, ds = [], []
            for f in FREQ_GRID:
                pre, post = pairing_trains(f, lag)
                tr.append(triplet_trains(pre, post, P))
                ds.append(direct_sum(pre, post, P))
            R[("grid", name, lag)] = np.array(tr)
            R[("grid_ds", name, lag)] = np.array(ds)
            R[("proto", name, lag)] = np.array(
                [triplet_trains(*pairing_trains(f, lag), P) for f in FREQS])
            R[("proto_ds", name, lag)] = np.array(
                [direct_sum(*pairing_trains(f, lag), P) for f in FREQS])
    # cumulative weight along 60-pair trains (for the figure)
    for f in (10.0, 50.0):
        for lag in LAGS:
            R[("cum", f, lag)] = triplet_trains(*pairing_trains(f, lag), FULL, history=True)[1]
    # Poisson trains vs direct sum
    errs, sums = [], []
    for _ in range(5):
        Tn = 20_000.0
        p_ = np.sort(rng_trains.uniform(0, Tn, rng_trains.poisson(0.02 * Tn)))
        q_ = np.sort(rng_trains.uniform(0, Tn, rng_trains.poisson(0.02 * Tn)))
        wt = triplet_trains(p_, q_, FULL)
        wd = direct_sum(p_, q_, FULL)
        sums.append(wd)
        errs.append(abs(wt - wd))
    R["train_err"], R["train_sum"] = np.array(errs), np.array(sums)
    # 2. long trains: last-pair increment vs periodic steady state
    long_err = []
    for name in ("full", "minimal"):
        P = sets[name]
        for lag in LAGS:
            for f in FREQS:
                _, h = triplet_trains(*pairing_trains(f, lag, N_LONG), P, history=True)
                long_err.append((name, lag, f, h[-1] - h[-2], periodic_per_pair(f, lag, P)))
    R["long"] = long_err
    print(f"  pairing protocols + trains: {_time.time() - tt:.1f} s")

    # 4. Poisson drift, all 28 conditions in one vectorised run per parameter set
    tt = _time.time()
    pois = {}
    for name, P in (("full", FULL), ("minimal", MINIMAL)):
        rx = [g[2] for g in GROUPS if g[0] == name]
        rho_x = np.repeat(np.repeat(np.array(rx), RHO_Y.size), POIS_M) / 1000.0
        rho_y = np.tile(np.repeat(RHO_Y, POIS_M), len(rx)) / 1000.0
        w = poisson_open_loop(rng_pois, rho_x, rho_y, P, POIS_T)
        w = w.reshape(len(rx), RHO_Y.size, POIS_M)
        for i, r in enumerate(rx):
            pois[(name, r)] = w[i]
    R["pois"] = pois
    print(f"  Poisson drift (28 conditions x {POIS_M} synapses, {POIS_T/1000:.0f} s): "
          f"{_time.time() - tt:.1f} s")
    return R


# ---------------------------------------------------------------- validate
def validate(R):
    V = {}
    print("\n1. Trace implementation vs exact direct sum")
    e_grid = max(np.abs(R[("grid", n, l)] - R[("grid_ds", n, l)]).max()
                 for n in ("full", "minimal", "pair") for l in LAGS)
    e_tr = R["train_err"].max()
    V.update(e_grid=e_grid, e_trains=e_tr)
    print(f"   60-pair protocol, {FREQ_GRID.size} frequencies x 2 lags x 3 sets: max |err| {e_grid:.2e}")
    print(f"   5 Poisson train pairs (20 Hz, 20 s), dw = {np.array2string(R['train_sum'], precision=3)}: "
          f"max |err| {e_tr:.2e} (tolerance 1e-10)")
    assert e_grid < 1e-10 and e_tr < 1e-10

    print(f"\n2. {N_LONG}-pair trains: last-pair increment vs periodic steady-state closed form")
    e_long = max(abs(a - b) for *_, a, b in R["long"])
    V["e_long"] = e_long
    for name, lag, f, a, b in R["long"]:
        if name == "full":
            print(f"   full, dt = {lag:+.0f} ms, {f:5.1f} Hz: {a:+.6e} vs {b:+.6e}")
    print(f"   max |err| over both sets {e_long:.2e} (tolerance 1e-12)")
    assert e_long < 1e-12

    print("\n3. Sjöström et al. (2001) frequency dependence, 60 pairs, dt = +-10 ms")
    for name in ("full", "minimal", "pair"):
        print(f"   {name:8s} dt = +10: {np.array2string(R[('proto', name, 10.0)], precision=3)}; "
              f"dt = -10: {np.array2string(R[('proto', name, -10.0)], precision=3)}")
    print(f"   data     dt = +10: {DATA_MEAN[10.0]}; dt = -10: {DATA_MEAN[-10.0]}")
    pp, pm = R[("proto", "full", 10.0)], R[("proto", "full", -10.0)]
    cross = brentq(lambda f: direct_sum(*pairing_trains(f, -10.0), FULL), 20.0, 50.0, xtol=1e-10)
    cross_ss = brentq(lambda f: periodic_per_pair(f, -10.0, FULL), 20.0, 50.0, xtol=1e-10)
    z = np.concatenate([(R[("proto", "full", l)] - DATA_MEAN[l]) / DATA_SEM[l] for l in LAGS])
    V.update(cross=cross, cross_ss=cross_ss, zmax_data=np.abs(z).max())
    print(f"   full: post-pre sign change at {cross:.3f} Hz (60 pairs; periodic steady state "
          f"{cross_ss:.3f} Hz); max |model - data|/SEM {np.abs(z).max():.2f} (assert < 3)")
    ok = (abs(pp[0]) < 0.01 and np.all(np.diff(pp) > 0) and np.all(pm[:3] < 0) and pm[-1] > 0
          and 20 < cross < 50 and np.abs(z).max() < 3)
    ctrl = R[("proto", "pair", -10.0)]
    print(f"   pair-only control (A3 = 0): post-pre max dw {ctrl.max():+.3f} "
          f"(LTD at every frequency: {bool(np.all(ctrl < 0))}); pre-post max {R[('proto','pair',10.0)].max():.1e}")
    assert ok and np.all(ctrl < 0)
    for name, P in (("full", FULL), ("minimal", MINIMAL)):
        model = {l: R[("proto", name, l)] for l in LAGS}
        E, lo, hi = fit_error(model)
        Ep = E_PUBLISHED[name]
        V[("E", name)] = (E, lo, hi)
        print(f"   {name:8s} normalised error E = {E:.4f} vs published {Ep:.2f}; range allowed by "
              f"Table 1 rounding [{lo:.4f}, {hi:.4f}] (assert {Ep:.2f} +- 0.005 overlaps it)")
        assert lo - 0.005 <= Ep <= hi + 0.005

    print(f"\n4. Poisson drift, T = {POIS_T/1000:.0f} s, {POIS_M} synapses per condition")
    zs, rels = [], []
    for name, P, rx in GROUPS:
        w = R["pois"][(name, rx)]
        c0, c1 = drift_coeffs(P, rx / 1000, POIS_T)
        ry = RHO_Y / 1000
        th = rx / 1000 * ry * (c0 + c1 * ry)
        m = w.mean(1)
        se = w.std(1, ddof=1) / math.sqrt(POIS_M)
        z = (m - th) / se
        V[("drift", name, rx)] = (m, se, th)
        zs.append(z)
        rels.append(np.abs(m / th - 1))
        print(f"   {name:7s} rho_x = {rx:4.1f} Hz: mean dw {np.array2string(m, precision=3)}")
        print(f"   {'':7s} {'':14s} closed form {np.array2string(th, precision=3)}; "
              f"z {np.array2string(z, precision=2)}")
    zs = np.concatenate(zs)
    V["z_drift"] = zs
    print(f"   max |z| over 28 conditions {np.abs(zs).max():.2f} (tolerance 4); "
          f"chi2/dof {np.mean(zs**2):.2f}")
    assert np.abs(zs).max() < 4

    print("\n5. BCM-like threshold on the postsynaptic rate")
    for name, P, rx in GROUPS:
        m, se, _ = V[("drift", name, rx)]
        ry = RHO_Y / 1000
        q = m / (rx / 1000 * ry)
        sq = se / (rx / 1000 * ry)
        Wt = 1 / sq**2
        X = np.vstack([np.ones_like(ry), ry]).T
        cov = np.linalg.inv(X.T @ (X * Wt[:, None]))
        a, b = cov @ (X.T @ (Wt * q))
        thf = -a / b
        g = np.array([-1 / b, a / b**2])
        sth = math.sqrt(g @ cov @ g)
        c0, c1 = drift_coeffs(P, rx / 1000, POIS_T)
        thT = -c0 / c1
        thS = theta_stationary(P, rx / 1000)
        rel = thf / thT - 1
        V[("theta", name, rx)] = (thf * 1000, sth * 1000, thT * 1000, thS * 1000, rel)
        print(f"   {name:7s} rho_x = {rx:4.1f} Hz: fitted {1000*thf:.3f} +- {1000*sth:.3f} Hz vs "
              f"closed form {1000*thT:.3f} Hz (stationary {1000*thS:.3f}); {100*rel:+.2f}%, "
              f"z = {(thf-thT)/sth:+.2f} (tolerance 3%)")
        assert abs(rel) < 0.03
    print("\nAll checks passed.")
    return V


# ---------------------------------------------------------------- plot
def plot(R, V):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    cols = {10.0: "C3", -10.0: "C0"}

    a = ax[0, 0]
    for lag in LAGS:
        c = cols[lag]
        a.plot(FREQ_GRID, R[("grid_ds", "full", lag)], "-", color=c, lw=1.4,
               label=f"full model, Δt = {lag:+.0f} ms (direct sum)")
        a.plot(FREQ_GRID, R[("grid", "full", lag)], "o", ms=3, color=c, mfc="none")
        a.plot(FREQ_GRID, R[("grid", "minimal", lag)], "--", color=c, lw=1, label=f"minimal model, Δt = {lag:+.0f}")
        a.plot(FREQ_GRID, R[("grid", "pair", lag)], ":", color=c, lw=1.2, label=f"pair terms only, Δt = {lag:+.0f}")
        mfc = c
        a.errorbar(FREQS, DATA_MEAN[lag], DATA_SEM[lag], fmt="s", color=c, mfc=mfc, ms=6,
                   capsize=3, label=f"Sjöström 2001 (P&G Table 1), Δt = {lag:+.0f}")
    a.axhline(0, color="0.7", lw=0.6)
    a.axvline(V["cross"], color="0.5", ls=":", lw=0.8)
    a.set_xlabel("pairing frequency ρ (Hz)")
    a.set_ylabel("weight change after 60 pairs")
    a.set_title(f"A  Frequency dependence; E = {V[('E','full')][0]:.3f} (paper 0.33), "
                f"post-pre crossover {V['cross']:.1f} Hz")
    a.legend(fontsize=6.5, loc="upper left", ncol=1)

    a = ax[0, 1]
    for name, mk in (("full", "o"), ("minimal", "s"), ("pair", "^")):
        for lag in LAGS:
            err = np.abs(R[("grid", name, lag)] - R[("grid_ds", name, lag)]) + 1e-20
            a.semilogy(FREQ_GRID, err, mk, ms=3, mfc="none", color=cols[lag], alpha=0.7)
    le = [abs(x - y) + 1e-20 for *_, x, y in R["long"]]
    lf = [f for _, _, f, _, _ in R["long"]]
    a.semilogy(lf, le, "kx", ms=5, label=f"{N_LONG}-pair last increment vs periodic closed form")
    a.semilogy([], [], "o", color="0.4", mfc="none", label="60 pairs: traces vs direct sum (3 sets, ±10 ms)")
    a.axhline(1e-10, color="k", ls=":", lw=1, label="tolerance 1e-10")
    a.set_ylim(1e-20, 1e-6)
    a.set_xlabel("pairing frequency ρ (Hz)")
    a.set_ylabel("|error| in Δw")
    a.set_title(f"B  Implementation errors; Poisson trains vs direct sum {V['e_trains']:.1e}")
    a.legend(fontsize=7.5, loc="upper right")

    a = ax[0, 2]
    for (f, ls) in ((10.0, "-"), (50.0, "--")):
        for lag in LAGS:
            h = R[("cum", f, lag)]
            k = np.arange(1, h.size + 1)
            a.plot(k, h, ls, color=cols[lag], lw=1.3, label=f"{f:.0f} Hz, Δt = {lag:+.0f} ms")
            s = periodic_per_pair(f, lag, FULL)
            a.plot(k, h[-1] + s * (k - k[-1]), ":", color="k", lw=0.7)
    a.plot([], [], ":", color="k", lw=0.7, label="closed-form steady slope per pair")
    a.axhline(0, color="0.7", lw=0.6)
    a.set_xlabel("pair number")
    a.set_ylabel("cumulative Δw (full model)")
    a.set_title("C  Build-up along the train: triplet term needs ~τ_y of history")
    a.legend(fontsize=7.5)

    a = ax[1, 0]
    ryf = np.linspace(0.5, 42, 200)
    for i, (name, P, rx) in enumerate(GROUPS):
        m, se, th = V[("drift", name, rx)]
        c0, c1 = drift_coeffs(P, rx / 1000, POIS_T)
        norm = rx * POIS_T / 1000                            # number of pre spikes
        cf = rx / 1000 * (ryf / 1000) * (c0 + c1 * ryf / 1000)
        lbl = f"{name}, ρ_x = {rx:.0f} Hz"
        a.plot(ryf, 1e3 * cf / norm, "-", color=f"C{i}", lw=1, label=lbl + " (closed form)")
        a.errorbar(RHO_Y, 1e3 * m / norm, 2e3 * se / norm, fmt="o", ms=4, color=f"C{i}", mfc="none", capsize=2)
    a.axhline(0, color="0.7", lw=0.6)
    a.set_xlabel("postsynaptic rate ρ_y (Hz)")
    a.set_ylabel("E[Δw] / (ρ_x T)   (per presynaptic spike, ×10⁻³)")
    a.set_title(f"D  Poisson drift, T = {POIS_T/1000:.0f} s, {POIS_M} synapses (±2 SE)")
    a.legend(fontsize=7.5)

    a = ax[1, 1]
    z = V["z_drift"]
    a.plot(np.arange(z.size), z, "o", ms=5)
    for k in range(1, len(GROUPS)):
        a.axvline(k * RHO_Y.size - 0.5, color="0.8", lw=0.6)
    a.axhline(0, color="0.7", lw=0.6)
    for s in (-4, 4):
        a.axhline(s, color="k", ls=":", lw=1)
    a.set_xticks([RHO_Y.size * (k + 0.5) - 0.5 for k in range(len(GROUPS))])
    a.set_xticklabels([f"{n}\nρ_x = {r:.0f} Hz" for n, _, r in GROUPS], fontsize=8)
    a.set_ylabel("z = (simulated − closed form) / SE")
    a.set_ylim(-4.5, 4.5)
    a.set_title(f"E  Drift vs closed form: max |z| {np.abs(z).max():.2f}, χ²/dof {np.mean(z**2):.2f}")

    a = ax[1, 2]
    rxf = np.linspace(0, 25, 100)
    a.plot(rxf, [1000 * theta_stationary(FULL, r / 1000) for r in rxf], "k-", lw=1,
           label="full: θ(ρ_x) closed form (stationary)")
    a.plot(rxf, [1000 * theta_stationary(MINIMAL, r / 1000) for r in rxf], "k--", lw=1,
           label="minimal: θ closed form")
    for i, (name, P, rx) in enumerate(GROUPS):
        thf, sth, thT, thS, rel = V[("theta", name, rx)]
        a.errorbar([rx], [thf], [2 * sth], fmt="o", color=f"C{i}", capsize=3,
                   label=f"{name}, ρ_x = {rx:.0f} Hz: {thf:.2f} vs {thT:.2f} Hz ({100*rel:+.2f}%)")
    a.set_xlabel("presynaptic rate ρ_x (Hz)")
    a.set_ylabel("threshold θ on ρ_y (Hz)")
    a.set_title("F  BCM-like threshold: zero of the drift (±2 SE)")
    a.legend(fontsize=7.5, loc="lower right")

    for a_ in ax.flat:
        a_.title.set_fontsize(10)
    fig.suptitle("Triplet STDP (Pfister & Gerstner 2006): Sjöström frequency dependence, "
                 "Poisson drift and the BCM-like threshold", fontsize=13)
    fig.tight_layout()
    out = ROOT / "figures" / "triplet_stdp.png"
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print(f"RNG seed: {SEED} (PCG64, one spawned stream per experiment)")
    t0 = _time.time()
    R = simulate()
    V = validate(R)
    plot(R, V)
    print(f"Runtime {_time.time() - t0:.1f} s")
