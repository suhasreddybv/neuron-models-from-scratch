"""Drift-diffusion model (DDM) of two-alternative forced-choice decisions.

Model (the pure DDM in the notation of Bogacz, Brown, Moehlis, Holmes &
Cohen 2006; Ratcliff 1978):

    dx = A dt + c dW,     x(0) = x0,     stop when x first reaches +z or -z,

x the accumulated evidence for alternative 1 over alternative 2, A the drift
(mean evidence per second), c the noise amplitude and dW a Wiener increment.
Reaching +z is a correct choice (A > 0), -z an error. The reaction time is
RT = DT + T0, DT the first-passage (decision) time and T0 a fixed
non-decision time. Parameters: c = 0.1 (the scale Ratcliff uses), z = 0.06
(boundary separation a = 2z = 0.12), T0 = 0.3 s; drifts A = 0 ... 0.4 s^-1
stand for stimulus strengths. Units: x dimensionless, t in s.

Integrator: Euler-Maruyama at dt = 0.5 ms. For constant drift the Gaussian
increment A dt + c sqrt(dt) N(0, 1) is exact, so the only discretisation
error is in the boundary test. A Brownian-bridge test catches crossings
inside a step: if both x_n and x_{n+1} are inside, the path crossed +z in
between with probability exp(-2 (z - x_n)(z - x_{n+1}) / (c^2 dt)), and -z
likewise, independent of A. A crossing on the grid is timed by linear
interpolation, a bridge crossing at the step midpoint. The same scheme
without the bridge ("naive": test x on the grid only, decision time = grid
time) is run in the timestep study.

Closed forms used for validation:
(1) Choice probability (gambler's ruin for Brownian motion with drift):
    P+(x0) = (1 - exp(-2A(x0 + z)/c^2)) / (1 - exp(-4Az/c^2)),
    = (x0 + z)/(2z) at A = 0; for x0 = 0 the error rate is
    ER = 1 / (1 + exp(2Az/c^2)) (Bogacz et al. 2006).
(2) Mean decision time from Wald's identity E[x_T] - x0 = A E[T]:
    E[T] = (z (2 P+ - 1) - x0) / A,  = (z^2 - x0^2)/c^2 at A = 0;
    for x0 = 0 this is DT = (z/A) tanh(Az/c^2) (Bogacz et al. 2006).
(3) Decision-time variance for x0 = 0 (Wagenmakers, van der Maas & Grasman
    2007, EZ-diffusion, with a = 2z, y = -A a / c^2):
    Var T = (a c^2 / (2 A^3)) (2y e^y - e^{2y} + 1) / (e^y + 1)^2,
    = (2/3) z^4/c^4 at A = 0.
(4) First-passage-time densities at each boundary, large-time series
    (Feller 1968; Navarro & Fuss 2009), in units where c = 1 (a' = 2z/c,
    v = A/c, w = (x0 + z)/(2z)):
    g-(t) = (pi/a'^2) exp(-v a' w) sum_k k sin(k pi w) exp(-lambda_k t),
    lambda_k = v^2/2 + k^2 pi^2 / (2 a'^2); g+ by v -> -v, w -> 1 - w.
    Term by term this gives the CDF, P_b - sum_k coef_k exp(-lambda_k t) /
    lambda_k, and the moments E[T^n; b] = n! sum_k coef_k / lambda_k^(n+1),
    used for Kolmogorov-Smirnov tests, conditional means, and as an
    independent check of (1)-(3).
(5) Timestep bias of the naive scheme: testing a Gaussian random walk on
    the grid only acts, to leading order, like moving each boundary out by
    beta c sqrt(dt), beta = |zeta(1/2)| / sqrt(2 pi) = 0.5826 (Siegmund 1979;
    Broadie, Glasserman & Kou 1997). Prediction: (1)-(2) with z replaced by
    z + beta c sqrt(dt).

Validation (ROADMAP note: simulated accuracy and mean RT against the closed
forms for the pure DDM): accuracy and mean RT at 8 drifts (40 000 trials
each) and at 5 biased starting points (A = 0.1), against (1)-(2) as z-scores;
the decision-time variance against (3); the correct and error decision-time
distributions against (4) by KS tests and conditional means; and a timestep
study, dt = 0.25 ... 16 ms with 100 000 trials, for the bridge scheme (no
bias expected) and the naive scheme against (5).

RNG: NumPy PCG64 seeded with SEED (printed).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate          # validation only

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Parameters
SEED = 1978
C = 0.1                 # noise amplitude
Z = 0.06                # boundaries at +-z
T0 = 0.3                # non-decision time (s)
DT = 5e-4               # timestep (s)
T_MAX = 10.0            # trial cut-off (s); P(DT > 10 s) < 1e-14 here
DRIFTS = np.array([0.0, 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4])
N_TRIALS = 40_000
A_BIAS = 0.1
X0S = np.array([-0.04, -0.02, 0.0, 0.02, 0.04])
A_DT = 0.2
DTS = np.array([0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]) * 1e-3
N_DT = 100_000
ZETA_HALF = -1.4603545088095868
BETA = abs(ZETA_HALF) / np.sqrt(2.0 * np.pi)     # 0.5826
K_SERIES = 400          # terms for densities / CDFs (t >= 1 ms)
K_MOMENT = 20_000       # terms for moments

# Tolerances
Z_TOL = 4.0             # |z| for every Monte Carlo comparison
KS_P_MIN = 1e-4         # smallest acceptable KS p-value (about 30 tests)
SERIES_TOL = 1e-8       # series vs closed forms, relative
NAIVE_DT_MAX = 4e-3     # naive scheme must match (5) for dt <= this
NAIVE_DT_SEP = 1e-3     # ... and be excluded from unshifted (1)-(2) for dt >= this


# ---------------------------------------------------------------------------
# Closed forms
def p_upper(A, x0=0.0, z=Z, c=C):
    """Probability of reaching +z first, closed form (1)."""
    A, x0 = np.broadcast_arrays(np.asarray(A, float), np.asarray(x0, float))
    k = 2.0 * A / c**2
    ks = np.where(A == 0.0, 1.0, k)
    with np.errstate(over="ignore", invalid="ignore"):
        p = np.expm1(-ks * (x0 + z)) / np.expm1(-2.0 * ks * z)
    return np.where(A == 0.0, (x0 + z) / (2.0 * z), p)


def mean_dt(A, x0=0.0, z=Z, c=C):
    """Mean decision time, closed form (2) (Wald's identity)."""
    A, x0 = np.broadcast_arrays(np.asarray(A, float), np.asarray(x0, float))
    p = p_upper(A, x0, z, c)
    As = np.where(A == 0.0, 1.0, A)
    return np.where(A == 0.0, (z**2 - x0**2) / c**2, (z * (2.0 * p - 1.0) - x0) / As)


def var_dt(A, z=Z, c=C):
    """Decision-time variance for x0 = 0, closed form (3) (EZ diffusion)."""
    A = np.asarray(A, float)
    a = 2.0 * z
    As = np.where(A == 0.0, 1.0, A)
    y = -As * a / c**2
    v = a * c**2 / (2.0 * As**3) * (2.0 * y * np.exp(y) - np.exp(2.0 * y) + 1.0) / (np.exp(y) + 1.0)**2
    return np.where(A == 0.0, 2.0 / 3.0 * z**4 / c**4, v)


def series(A, x0, boundary, K, z=Z, c=C):
    """Coefficients and rates of the large-time series (4) for one boundary."""
    a = 2.0 * z / c
    v = A / c
    w = (x0 + z) / (2.0 * z)
    if boundary > 0:
        v, w = -v, 1.0 - w
    k = np.arange(1, K + 1, dtype=float)
    lam = 0.5 * v**2 + k**2 * np.pi**2 / (2.0 * a**2)
    coef = np.pi / a**2 * np.exp(-v * a * w) * k * np.sin(k * np.pi * w)
    return coef, lam


def fpt_density(t, A, x0, boundary):
    """Defective first-passage density at one boundary (integrates to P_b)."""
    coef, lam = series(A, x0, boundary, K_SERIES)
    t = np.asarray(t, float)
    g = np.exp(-np.outer(np.maximum(t, 1e-3), lam)) @ coef
    return np.where(t < 1e-3, 0.0, g)


def fpt_moments(A, x0, boundary):
    """(P_b by quadrature of the density, E[T; b], E[T^2; b]) from the series."""
    coef, lam = series(A, x0, boundary, K_MOMENT)
    m1 = np.sum(coef / lam**2)
    m2 = 2.0 * np.sum(coef / lam**3)
    p_quad = integrate.quad(lambda t: fpt_density(np.array([t]), A, x0, boundary)[0],
                            1e-3, T_MAX, limit=400, points=[0.05, 0.3, 1.0])[0]
    return p_quad, m1, m2


def conditional_cdf(A, x0, boundary):
    """Callable conditional CDF F_b(t) / P_b of the decision time, from (4)."""
    pb = p_upper(A, x0) if boundary > 0 else 1.0 - p_upper(A, x0)
    coef, lam = series(A, x0, boundary, K_SERIES)
    tg = np.linspace(0.0, T_MAX, 20_001)
    tail = np.exp(-np.outer(np.maximum(tg, 1e-3), lam)) @ (coef / lam)
    F = np.where(tg < 1e-3, 0.0, (pb - tail) / pb)
    F = np.clip(F, 0.0, 1.0)
    return lambda t: np.interp(t, tg, F)


# ---------------------------------------------------------------------------
# Model
def simulate(A, x0, rng, dt=DT, bridge=True, t_max=T_MAX, z=Z, c=C):
    """Simulate independent DDM trials (one per entry of A, x0).

    Returns choice (+1 upper, -1 lower, 0 not decided by t_max) and the
    decision time (s, NaN if undecided)."""
    A, x0 = np.broadcast_arrays(np.asarray(A, float), np.asarray(x0, float))
    n = A.size
    choice = np.zeros(n, np.int8)
    t_dec = np.full(n, np.nan)
    idx = np.arange(n)
    x = x0.ravel().copy()
    a_ = A.ravel().copy()
    sq = c * np.sqrt(dt)
    inv = 2.0 / (c * c * dt)
    step = 0
    while idx.size and step * dt < t_max:
        m = idx.size
        xn = x + a_ * dt + sq * rng.standard_normal(m)
        up = xn >= z
        lo = xn <= -z
        if bridge:
            inside = ~(up | lo)
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                pu = np.exp(-inv * (z - x) * (z - xn))
                pl = np.exp(-inv * (z + x) * (z + xn))
                fu = (z - x) / (xn - x)
                fl = (-z - x) / (xn - x)
            u = rng.random(m)
            bu = inside & (u < pu)
            bl = inside & ~bu & (u < pu + pl)
            frac = np.where(up, fu, np.where(lo, fl, 0.5))
            up = up | bu
            lo = lo | bl
        else:
            frac = np.ones(m)
        done = up | lo
        if done.any():
            d = idx[done]
            choice[d] = np.where(up[done], 1, -1)
            t_dec[d] = (step + frac[done]) * dt
            keep = ~done
            idx, x, a_ = idx[keep], xn[keep], a_[keep]
        else:
            x = xn
        step += 1
    return choice, t_dec


def sample_paths(A, n, rng, dt=DT, t_max=1.5, z=Z, c=C):
    """A few full trajectories for display (grid test only), frozen at the bound."""
    steps = int(round(t_max / dt))
    inc = A * dt + c * np.sqrt(dt) * rng.standard_normal((n, steps))
    x = np.concatenate([np.zeros((n, 1)), np.cumsum(inc, axis=1)], axis=1)
    out = np.full_like(x, np.nan)
    hit = np.zeros(n, int)
    for i in range(n):
        j = np.flatnonzero(np.abs(x[i]) >= z)
        end = j[0] if j.size else steps
        out[i, :end + 1] = x[i, :end + 1]
        out[i, end] = np.clip(out[i, end], -z, z)
        hit[i] = int(np.sign(x[i, end])) if j.size else 0
    return np.arange(steps + 1) * dt, out, hit


# ---------------------------------------------------------------------------
def summarise(choice, t, A, x0):
    """Monte Carlo statistics of one condition and their closed-form targets."""
    n = choice.size
    p_th = float(p_upper(A, x0))
    p_sim = np.mean(choice == 1)
    se_p = np.sqrt(p_th * (1.0 - p_th) / n)
    m_th = float(mean_dt(A, x0))
    m_sim = t.mean()
    sd = t.std(ddof=1)
    r = dict(A=A, x0=x0, n=n, p_th=p_th, p_sim=p_sim, z_p=(p_sim - p_th) / se_p,
             se_p=se_p, m_th=m_th, m_sim=m_sim, se_m=sd / np.sqrt(n),
             z_m=(m_sim - m_th) / (sd / np.sqrt(n)), sd_sim=sd)
    if x0 == 0.0:
        v_th = float(var_dt(A))
        dev = t - m_sim
        m4 = np.mean(dev**4)
        se_v = np.sqrt((m4 - sd**4) / n)
        r.update(v_th=v_th, v_sim=sd**2, z_v=(sd**2 - v_th) / se_v, se_v=se_v)
    cond = {}
    for b in (1, -1):
        sel = choice == b
        nb = int(sel.sum())
        pq, m1, m2 = fpt_moments(A, x0, b)
        pb = p_th if b > 0 else 1.0 - p_th
        e = dict(n=nb, p_quad=pq, pb=pb, mean_th=m1 / pb, m1=m1, m2=m2)
        if nb >= 100:
            tb = t[sel]
            e.update(mean_sim=tb.mean(), se=tb.std(ddof=1) / np.sqrt(nb))
            e["z"] = (e["mean_sim"] - e["mean_th"]) / e["se"]
            e["ks"] = stats.kstest(tb, conditional_cdf(A, x0, b))
        cond[b] = e
    r["cond"] = cond
    return r


def validate():
    rng = np.random.default_rng(SEED)
    R = {}

    # 1. accuracy, mean DT, variance and distributions vs drift, x0 = 0
    t1 = _time.time()
    A_all = np.repeat(DRIFTS, N_TRIALS)
    ch, td = simulate(A_all, 0.0, rng)
    assert (ch != 0).all(), "undecided trials at t_max"
    R["drift"] = [summarise(ch[A_all == A], td[A_all == A], A, 0.0) for A in DRIFTS]
    R["hist"] = dict(A=0.2, choice=ch[A_all == 0.2], t=td[A_all == 0.2])
    print(f"1. Drift sweep, x0 = 0, {N_TRIALS} trials per drift, dt = {DT * 1e3:.2f} ms "
          f"(bridge), {_time.time() - t1:.1f} s")
    print("   A      P(+z) sim   closed    z     | mean RT sim  closed (s)   z    "
          "| SD(DT) sim  closed    z   | KS p (+, -)")
    for r in R["drift"]:
        ks = [r["cond"][b]["ks"].pvalue if "ks" in r["cond"][b] else np.nan for b in (1, -1)]
        print(f"   {r['A']:.3f}  {r['p_sim']:.4f}  {r['p_th']:.4f}  {r['z_p']:+5.2f} | "
              f"{r['m_sim'] + T0:.4f}  {r['m_th'] + T0:.4f}  {r['z_m']:+5.2f} | "
              f"{r['sd_sim']:.4f}  {np.sqrt(r['v_th']):.4f}  {r['z_v']:+5.2f} | "
              f"{ks[0]:.3f}, {ks[1]:.3f}")

    # 2. biased starting point
    t1 = _time.time()
    x_all = np.repeat(X0S, N_TRIALS)
    ch, td = simulate(A_BIAS, x_all, rng)
    assert (ch != 0).all(), "undecided trials at t_max"
    R["bias"] = [summarise(ch[x_all == x], td[x_all == x], A_BIAS, x) for x in X0S]
    print(f"2. Biased start, A = {A_BIAS}, {N_TRIALS} trials per x0, {_time.time() - t1:.1f} s")
    print("   x0      P(+z) sim  closed    z    | mean DT sim  closed   z    "
          "| E[DT|+] sim  series   z    | E[DT|-] sim  series   z    | KS p (+, -)")
    for r in R["bias"]:
        cp, cm = r["cond"][1], r["cond"][-1]
        print(f"   {r['x0']:+.3f}  {r['p_sim']:.4f}  {r['p_th']:.4f}  {r['z_p']:+5.2f} | "
              f"{r['m_sim']:.4f}  {r['m_th']:.4f}  {r['z_m']:+5.2f} | "
              f"{cp['mean_sim']:.4f}  {cp['mean_th']:.4f}  {cp['z']:+5.2f} | "
              f"{cm['mean_sim']:.4f}  {cm['mean_th']:.4f}  {cm['z']:+5.2f} | "
              f"{cp['ks'].pvalue:.3f}, {cm['ks'].pvalue:.3f}")

    # 3. series vs closed forms (independent derivations)
    errs = []
    for r in R["drift"] + R["bias"]:
        cp, cm = r["cond"][1], r["cond"][-1]
        errs.append(abs(cp["p_quad"] - cp["pb"]) / cp["pb"])
        errs.append(abs(cm["p_quad"] - cm["pb"]) / cm["pb"])
        errs.append(abs(cp["m1"] + cm["m1"] - r["m_th"]) / r["m_th"])
        if "v_th" in r:
            v_ser = cp["m2"] + cm["m2"] - (cp["m1"] + cm["m1"])**2
            errs.append(abs(v_ser - r["v_th"]) / r["v_th"])
    R["series_err"] = max(errs)
    print(f"3. Series (4) vs closed forms (1)-(3), 13 conditions: max relative "
          f"difference {R['series_err']:.1e} (tolerance {SERIES_TOL:.0e})")

    # 4. timestep study
    t1 = _time.time()
    study = []
    for dt in DTS:
        for bridge in (True, False):
            ch, td = \
                simulate(np.full(N_DT, A_DT), 0.0, rng, dt=dt, bridge=bridge)
            assert (ch != 0).all()
            er = np.mean(ch == -1)
            m = td.mean()
            se_m = td.std(ddof=1) / np.sqrt(N_DT)
            zs = Z + BETA * C * np.sqrt(dt)
            er_ex, m_ex = 1.0 - float(p_upper(A_DT)), float(mean_dt(A_DT))
            er_sh, m_sh = 1.0 - float(p_upper(A_DT, 0.0, zs)), float(mean_dt(A_DT, 0.0, zs))
            se_er = np.sqrt(er_ex * (1 - er_ex) / N_DT)
            study.append(dict(dt=dt, bridge=bridge, er=er, m=m, se_m=se_m, se_er=se_er,
                              er_ex=er_ex, m_ex=m_ex, er_sh=er_sh, m_sh=m_sh,
                              z_m_ex=(m - m_ex) / se_m, z_er_ex=(er - er_ex) / se_er,
                              z_m_sh=(m - m_sh) / se_m, z_er_sh=(er - er_sh) / se_er))
    R["study"] = study
    print(f"4. Timestep study, A = {A_DT}, {N_DT} trials per point, {_time.time() - t1:.1f} s "
          f"(exact ER {study[0]['er_ex']:.5f}, DT {study[0]['m_ex'] * 1e3:.2f} ms)")
    print("   dt (ms) scheme  ER       z(exact) z(shift) | DT (ms)   z(exact) z(shift) | "
          "shifted-bound prediction ER, DT")
    for s in study:
        print(f"   {s['dt'] * 1e3:6.2f}  {'bridge' if s['bridge'] else 'naive ':6s}  "
              f"{s['er']:.5f}  {s['z_er_ex']:+6.2f}  {s['z_er_sh']:+6.2f}  | "
              f"{s['m'] * 1e3:7.2f}  {s['z_m_ex']:+7.2f}  {s['z_m_sh']:+6.2f}  | "
              f"{s['er_sh']:.5f}, {s['m_sh'] * 1e3:.2f}")
    nv = [s for s in study if not s["bridge"]]
    dts = np.array([s["dt"] for s in nv])
    exc = np.array([s["m"] - s["m_ex"] for s in nv])
    R["naive_slope"] = np.polyfit(np.log(dts), np.log(exc), 1)[0]
    print(f"   naive DT excess vs dt: log-log slope {R['naive_slope']:.3f} (leading order 0.5)")

    # ---------------- assertions ----------------
    zs_all, ks_all = [], []
    for r in R["drift"] + R["bias"]:
        zs_all += [r["z_p"], r["z_m"]] + ([r["z_v"]] if "z_v" in r else [])
        for b in (1, -1):
            e = r["cond"][b]
            if "ks" in e:
                zs_all.append(e["z"])
                ks_all.append(e["ks"].pvalue)
    R["max_z"] = max(abs(np.array(zs_all)))
    R["n_z"] = len(zs_all)
    R["min_ks"] = min(ks_all)
    R["n_ks"] = len(ks_all)
    br = [s for s in study if s["bridge"]]
    R["max_z_bridge"] = max(max(abs(s["z_m_ex"]), abs(s["z_er_ex"])) for s in br)
    R["max_z_naive_sh"] = max(max(abs(s["z_m_sh"]), abs(s["z_er_sh"]))
                              for s in nv if s["dt"] <= NAIVE_DT_MAX + 1e-12)
    R["min_z_naive_ex"] = min(abs(s["z_m_ex"]) for s in nv if s["dt"] >= NAIVE_DT_SEP - 1e-12)
    print(f"Summary: max |z| {R['max_z']:.2f} over {R['n_z']} Monte Carlo comparisons "
          f"(tolerance {Z_TOL}); min KS p {R['min_ks']:.4f} over {R['n_ks']} tests "
          f"(tolerance {KS_P_MIN:.0e})")
    print(f"         bridge scheme max |z| vs exact over dt = 0.25-16 ms: {R['max_z_bridge']:.2f}; "
          f"naive scheme max |z| vs shifted bound (dt <= 4 ms): {R['max_z_naive_sh']:.2f}, "
          f"min |z| of DT vs unshifted (dt >= 1 ms): {R['min_z_naive_ex']:.1f}")
    assert R["series_err"] < SERIES_TOL, "series disagrees with closed forms"
    assert R["max_z"] < Z_TOL, f"Monte Carlo off closed form, |z| = {R['max_z']:.2f}"
    assert R["min_ks"] > KS_P_MIN, f"KS test rejects series distribution, p = {R['min_ks']:.2e}"
    assert R["max_z_bridge"] < Z_TOL, "bridge scheme biased in dt"
    assert R["max_z_naive_sh"] < Z_TOL, "naive scheme does not follow shifted-boundary prediction"
    assert R["min_z_naive_ex"] > Z_TOL, "naive bias not resolved"
    print("All DDM checks passed.")

    R["paths"] = sample_paths(0.2, 10, rng)
    return R


# ---------------------------------------------------------------------------
def plot(R):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    ccol, ecol = "tab:blue", "tab:red"

    # A: sample paths
    ax = axes[0, 0]
    t, X, hit = R["paths"]
    for i in range(X.shape[0]):
        ax.plot(t, X[i], color=ccol if hit[i] > 0 else (ecol if hit[i] < 0 else "0.5"), lw=0.8)
    ax.axhline(Z, color="k", lw=1)
    ax.axhline(-Z, color="k", lw=1)
    ax.axhline(0, color="0.7", lw=0.5, ls=":")
    ax.set_xlabel("time from evidence onset (s)")
    ax.set_ylabel("accumulated evidence x")
    ax.set_title(f"A. Sample paths, A = 0.2, c = {C}, z = ±{Z}")

    # B: RT distributions vs series
    ax = axes[0, 1]
    H = R["hist"]
    edges = np.arange(0, 2.0 + 1e-9, 0.02)
    n = H["t"].size
    for b, col, sgn, lab in ((1, ccol, 1, "correct"), (-1, ecol, -1, "error")):
        tb = H["t"][H["choice"] == b]
        cnt, _ = np.histogram(tb, edges)
        dens = cnt / (n * np.diff(edges))
        ax.bar(edges[:-1] + T0, sgn * dens, width=np.diff(edges), align="edge",
               color=col, alpha=0.35, label=f"simulated {lab} ({tb.size} trials)")
    tg = np.linspace(0, 2.0, 801)
    ax.plot(tg + T0, fpt_density(tg, H["A"], 0.0, 1), color=ccol, lw=1.5, label="series g₊(t)")
    ax.plot(tg + T0, -fpt_density(tg, H["A"], 0.0, -1), color=ecol, lw=1.5, label="series −g₋(t)")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlim(0.2, 1.8)
    ax.set_xlabel("reaction time RT = DT + T0 (s)")
    ax.set_ylabel("density (errors mirrored)")
    ax.set_title(f"B. RT densities at A = 0.2; KS p ≥ {R['min_ks']:.3f} in all {R['n_ks']} tests")
    ax.legend(fontsize=8)

    # C: psychometric
    Ad = np.linspace(0, 0.42, 200)
    D = R["drift"]
    A = np.array([r["A"] for r in D])
    ax = axes[0, 2]
    ax.plot(Ad, p_upper(Ad), color="k", lw=1.2, label="closed form 1/(1 + e^(−2Az/c²))")
    ax.errorbar(A, [r["p_sim"] for r in D], yerr=[2 * r["se_p"] for r in D], fmt="o",
                color=ccol, ms=5, capsize=2, label="simulated ± 2 SE")
    ax.set_xlabel("drift A (s⁻¹)")
    ax.set_ylabel("P(correct)")
    ax.set_title("C. Accuracy vs drift")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # D: chronometric
    ax = axes[0, 3]
    ax.plot(Ad, mean_dt(Ad) + T0, color="k", lw=1.2, label="closed form T0 + (z/A) tanh(Az/c²)")
    for b, col, lab, off in ((1, ccol, "correct", -0.003), (-1, ecol, "error", 0.003)):
        mm = [(r["A"] + off, r["cond"][b]["mean_sim"] + T0, 2 * r["cond"][b]["se"])
              for r in D if "mean_sim" in r["cond"][b]]
        mm = np.array(mm)
        ax.errorbar(mm[:, 0], mm[:, 1], yerr=mm[:, 2], fmt="o", ms=4, color=col, capsize=2,
                    label=f"simulated mean RT, {lab} ± 2 SE")
    ax.plot(A, [r["m_sim"] + T0 for r in D], "k+", ms=9, label="simulated mean RT, all")
    ax.set_xlabel("drift A (s⁻¹)")
    ax.set_ylabel("mean RT (s)")
    ax.set_title("D. Mean RT vs drift (correct = error at x0 = 0)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # E: SD of DT
    ax = axes[1, 0]
    ax.plot(Ad, np.sqrt(var_dt(Ad)), color="k", lw=1.2, label="EZ closed form √Var T")
    sdv = np.array([np.sqrt(r["v_sim"]) for r in D])
    sde = np.array([r["se_v"] / (2 * np.sqrt(r["v_sim"])) for r in D])
    ax.errorbar(A, sdv, yerr=2 * sde, fmt="o", color="tab:purple", ms=5, capsize=2,
                label="simulated ± 2 SE")
    ax.set_xlabel("drift A (s⁻¹)")
    ax.set_ylabel("SD of decision time (s)")
    ax.set_title("E. Decision-time spread vs drift")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # F: biased start, choice probability
    B = R["bias"]
    xd = np.linspace(-Z, Z, 200)
    x0 = np.array([r["x0"] for r in B])
    ax = axes[1, 1]
    ax.plot(xd, p_upper(A_BIAS, xd), color="k", lw=1.2, label="closed form (1)")
    ax.errorbar(x0, [r["p_sim"] for r in B], yerr=[2 * r["se_p"] for r in B], fmt="o",
                color=ccol, ms=5, capsize=2, label="simulated ± 2 SE")
    ax.set_xlabel("starting point x0")
    ax.set_ylabel("P(+z)")
    ax.set_title(f"F. Biased start, A = {A_BIAS}: choice probability")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # G: biased start, conditional mean DT
    ax = axes[1, 2]
    xs = np.linspace(-0.05, 0.05, 41)
    for b, col, lab in ((1, ccol, "+z"), (-1, ecol, "−z")):
        th = []
        for xv in xs:
            coef, lam = series(A_BIAS, xv, b, K_MOMENT)
            pb = float(p_upper(A_BIAS, xv)) if b > 0 else 1 - float(p_upper(A_BIAS, xv))
            th.append(np.sum(coef / lam**2) / pb)
        ax.plot(xs, th, color=col, lw=1.2, label=f"series E[DT | {lab}]")
        ax.errorbar(x0, [r["cond"][b]["mean_sim"] for r in B],
                    yerr=[2 * r["cond"][b]["se"] for r in B], fmt="o", ms=5, color=col,
                    capsize=2, label=f"simulated, {lab} ± 2 SE")
    ax.plot(xd, mean_dt(A_BIAS, xd), "k--", lw=1, label="Wald E[DT] (all trials)")
    ax.set_xlabel("starting point x0")
    ax.set_ylabel("mean decision time (s)")
    ax.set_title("G. Biased start: conditional decision times")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # H: timestep study
    ax = axes[1, 3]
    S = R["study"]
    for bridge, col, lab in ((True, "tab:green", "Brownian bridge"), (False, "tab:orange", "grid test only")):
        ss = [s for s in S if s["bridge"] == bridge]
        ax.errorbar([s["dt"] * 1e3 for s in ss], [(s["m"] - s["m_ex"]) * 1e3 for s in ss],
                    yerr=[2e3 * s["se_m"] for s in ss], fmt="o", color=col, ms=5, capsize=2,
                    label=f"simulated, {lab} ± 2 SE")
    dd = np.geomspace(0.2e-3, 20e-3, 100)
    ax.plot(dd * 1e3, (mean_dt(A_DT, 0.0, Z + BETA * C * np.sqrt(dd)) - mean_dt(A_DT)) * 1e3,
            color="k", lw=1, label="boundary z + 0.5826 c√dt")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel("dt (ms)")
    ax.set_ylabel("mean DT − closed form (ms)")
    ax.set_title(f"H. Timestep bias at A = {A_DT} (naive slope {R['naive_slope']:.2f})")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "drift_diffusion.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print(f"RNG seed: {SEED} (PCG64)")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
