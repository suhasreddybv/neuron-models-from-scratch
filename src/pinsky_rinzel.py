"""Pinsky-Rinzel two-compartment CA3 pyramidal cell.

Equations (Pinsky & Rinzel 1994, J. Comput. Neurosci. 1, 39-60, eqs. 1-2
and the appendix; kinetics reduced from Traub et al. 1991). Voltages are in
mV shifted so that the leak reversal is 0 (rest is near -4.6 mV), t in ms,
currents in uA/cm^2 of total membrane, Ca in arbitrary units:

  soma      C_m dVs/dt = -I_L(Vs) - I_Na - I_KDR + (g_c/p)(Vd - Vs) + I_s/p
  dendrite  C_m dVd/dt = -I_L(Vd) - I_Ca - I_KAHP - I_KC
                         + (g_c/(1-p))(Vs - Vd) + I_d/(1-p)

  I_L    = g_L (V - V_L)
  I_Na   = g_Na m_inf(Vs)^2 h (Vs - V_Na),  m_inf = a_m / (a_m + b_m)
  I_KDR  = g_KDR n (Vs - V_K)
  I_Ca   = g_Ca s^2 (Vd - V_Ca)
  I_KC   = g_KC c chi(Ca) (Vd - V_K),       chi = min(Ca/250, 1)
  I_KAHP = g_KAHP q (Vd - V_K)
  dCa/dt = -0.13 I_Ca - 0.075 Ca
  dx/dt  = a_x (1 - x) - b_x x               for x = h, n, s, c, q

  a_m = 0.32 (13.1 - V)/(exp((13.1 - V)/4) - 1)   b_m = 0.28 (V - 40.1)/(exp((V - 40.1)/5) - 1)
  a_h = 0.128 exp((17 - V)/18)                    b_h = 4 / (1 + exp((40 - V)/5))
  a_n = 0.016 (35.1 - V)/(exp((35.1 - V)/5) - 1)  b_n = 0.25 exp(0.5 - 0.025 V)
  a_s = 1.6 / (1 + exp(-0.072 (V - 65)))          b_s = 0.02 (V - 51.1)/(exp((V - 51.1)/5) - 1)
  a_c = exp((V - 10)/11 - (V - 6.5)/27) / 18.975  (V <= 50),  2 exp((6.5 - V)/27)  (V > 50)
  b_c = 2 exp((6.5 - V)/27) - a_c                 (V <= 50),  0                    (V > 50)
  a_q = min(0.00002 Ca, 0.01)                     b_q = 0.001

C_m = 3 uF/cm^2, p = 0.5, g_L = 0.1, g_Na = 30, g_KDR = 15, g_Ca = 10,
g_KAHP = 0.8, g_KC = 15 mS/cm^2, V_Na = 120, V_Ca = 140, V_K = -15,
V_L = 0 mV; standard coupling g_c = 2.1 mS/cm^2. Stimulus: a constant
somatic current I_s from t = 0, I_d = 0, starting from the paper's initial
state (Vs, Vd, h, n, s, c, q, Ca) = (-4.6, -4.5, 0.999, 0.001, 0.009,
0.007, 0.01, 0.2).

Integrator: classical RK4 written out, fixed dt = 0.02 ms, vectorised over
(I_s, g_c) pairs. The fastest linear mode is the soma-dendrite voltage
difference, rate g_c (1/p + 1/(1-p)) / C_m, so RK4 is stable up to about
g_c = 50 at this dt; nothing larger is run. Spikes are upward crossings
of Vs = 20 mV (linear interpolation).

Validation (validate()):

1. Closed form, rest state. With every gate at steady state, Ca_inf =
   -0.13 I_Ca(Vd)/0.075 is explicit in Vd, so rest solves two algebraic
   equations in (Vs, Vd). At I_s = -0.5 and -1 a run started on the root
   must stay there within 1e-6 mV for 4000 ms, every eigenvalue of the
   finite-difference Jacobian there must have negative real part, and a
   run started 1 mV off (Vs and Vd) must return within 1e-3 mV (the slow
   q/Ca mode, time constant ~1 s, is not finished decaying by then). At
   I_s = 0 the cell already bursts slowly, so there is no stable rest.
2. Limiting cases, g_c -> infinity and g_c -> 0. For large g_c the two
   compartments become one isopotential cell, C_m dV/dt = -I_L - p(I_Na +
   I_KDR) - (1-p)(I_Ca + I_KAHP + I_KC) + I_s, written as its own
   equation (rhs(..., one=True)). For g_c -> 0 the soma is an isolated
   Na/K_DR compartment driven by I_s/p (the g_c = 0 run). Asserted: the interspike interval of the
   full model approaches each limit with error proportional to 1/g_c
   (resp. g_c), log-log slope within 0.15 of -1 over g_c = 12.5-50 (resp.
   +1 over g_c = 0.000625-0.0025), and the
   Richardson extrapolation from the two most extreme g_c values matches
   the reduced model within 1%. The weak-coupling limit is only reached
   for 2 g_c << g_L = 0.1: the passive dendrite has so little leak that
   even g_c = 0.04 holds it well above its own rest, so the error is not
   linear in g_c there (shown, not used in the slope fit).
3. Reference solver. A bursting run (I_s = 0.75, g_c = 2.1) and a
   single-spiking run (I_s = 1.5, g_c = 10.5) are repeated with scipy's
   DOP853 (rtol = atol = 1e-9). Asserted: equal spike counts and spikes per
   burst, mean period within 1e-3 relative.
4. Paper, qualitative. Pinsky & Rinzel (1994) report that the standard
   g_c = 2.1 gives somatic bursts driven by a dendritic Ca spike, and that
   strong coupling makes the cell behave like one compartment, which does
   not burst. Asserted: bursting (several spikes per event, a dendritic Ca
   spike) at g_c = 2.1, I_s = 0.75, and single spikes at g_c >= 10.5 for
   every I_s tested. The g_c boundaries found by the sweep are reported,
   not asserted against a published number.

No random numbers are used. scipy is used only for validation (root
finding, curve fit, reference solver).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
C_m, p = 3.0, 0.5                                 # uF/cm^2, soma area fraction
g_L, g_Na, g_KDR = 0.1, 30.0, 15.0                # mS/cm^2
g_Ca, g_KAHP, g_KC = 10.0, 0.8, 15.0              # mS/cm^2
V_Na, V_Ca, V_K, V_L = 120.0, 140.0, -15.0, 0.0   # mV (shifted)
G_C_STD = 2.1                                     # mS/cm^2, standard coupling
Y0 = np.array([-4.6, -4.5, 0.999, 0.001, 0.009, 0.007, 0.01, 0.2])

DT = 0.02                                         # ms, RK4 step
T_END = 4000.0                                    # ms
T_FROM = 2000.0                                   # ms, analysis window start
V_SPIKE = 20.0                                    # mV, somatic spike detection
I_MAP = np.array([0.75, 1.5, 2.5])                # uA/cm^2, regime sweep
GC_MAP = np.unique(np.round(np.concatenate((np.geomspace(0.25, 25.0, 31),
                                            [G_C_STD, 10.5])), 4))
GC_BIG = np.array([6.25, 12.5, 25.0, 50.0])
GC_SMALL = np.array([0.04, 0.02, 0.01, 0.005, 0.0025, 0.00125, 0.000625])
I_REST = np.array([-0.5, -1.0])
REF_RUNS = [(0.75, G_C_STD), (1.5, 10.5)]
REF_TOL = 1e-9
REF_REL_TOL = 1e-3
REST_TOL = 1e-6
REST_TOL_KICK = 1e-3
SLOPE_TOL = 0.15
SLOPE_FROM = 12.5           # mS/cm^2, slope fit for g_c -> inf uses g_c >= this
SLOPE_TO = 0.0025           # mS/cm^2, slope fit for g_c -> 0 uses g_c <= this
EXTRAP_TOL = 0.01
BURST_RATIO = 3.0           # max ISI / min ISI above this: bursting


# ------------------------------------------------------------------- model
def vtrap(x):
    """x / (1 - exp(-x)), continuous at x = 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-7
    return np.where(small, 1.0 + 0.5 * x, x / np.where(small, 1.0, -np.expm1(-x)))


def soma_rates(V):
    a_m = 1.28 * vtrap((V - 13.1) / 4.0)          # = 0.32 (13.1-V)/(e^((13.1-V)/4)-1)
    b_m = 1.4 * vtrap((40.1 - V) / 5.0)
    a_h = 0.128 * np.exp((17.0 - V) / 18.0)
    b_h = 4.0 / (1.0 + np.exp((40.0 - V) / 5.0))
    a_n = 0.08 * vtrap((V - 35.1) / 5.0)
    b_n = 0.25 * np.exp(0.5 - 0.025 * V)
    return a_m, b_m, a_h, b_h, a_n, b_n


def dend_rates(V, Ca):
    a_s = 1.6 / (1.0 + np.exp(-0.072 * (V - 65.0)))
    b_s = 0.1 * vtrap((51.1 - V) / 5.0)
    e = np.exp((6.5 - V) / 27.0)
    low = V <= 50.0
    a_c = np.where(low, np.exp((V - 10.0) / 11.0 - (V - 6.5) / 27.0) / 18.975, 2.0 * e)
    b_c = np.where(low, 2.0 * e - a_c, 0.0)
    a_q = np.minimum(2e-5 * Ca, 0.01)
    return a_s, b_s, a_c, b_c, a_q, 0.001


def soma_currents(Vs, h, n):
    a_m, b_m = soma_rates(Vs)[:2]
    m = a_m / (a_m + b_m)
    return g_Na * m * m * h * (Vs - V_Na), g_KDR * n * (Vs - V_K)


def dend_currents(Vd, s, c, q, Ca):
    I_Ca = g_Ca * s * s * (Vd - V_Ca)
    I_KC = g_KC * c * np.minimum(Ca / 250.0, 1.0) * (Vd - V_K)
    return I_Ca, I_KC, g_KAHP * q * (Vd - V_K)


def gate_rhs(Vs, Vd, h, n, s, c, q, Ca):
    _, _, a_h, b_h, a_n, b_n = soma_rates(Vs)
    a_s, b_s, a_c, b_c, a_q, b_q = dend_rates(Vd, Ca)
    return (a_h * (1 - h) - b_h * h, a_n * (1 - n) - b_n * n, a_s * (1 - s) - b_s * s,
            a_c * (1 - c) - b_c * c, a_q * (1 - q) - b_q * q)


def rhs(y, I_s, g_c, one=None, I_d=0.0):
    """Two-compartment model; y = (Vs, Vd, h, n, s, c, q, Ca). Columns where
    `one` is True are the g_c -> infinity reduction instead: both voltages
    follow the area-weighted one-compartment equation (start them equal)."""
    Vs, Vd, h, n, s, c, q, Ca = y
    I_Na, I_KDR = soma_currents(Vs, h, n)
    I_Ca, I_KC, I_AHP = dend_currents(Vd, s, c, q, Ca)
    dVs = (-g_L * (Vs - V_L) - I_Na - I_KDR + g_c / p * (Vd - Vs) + I_s / p) / C_m
    dVd = (-g_L * (Vd - V_L) - I_Ca - I_KC - I_AHP + g_c / (1 - p) * (Vs - Vd)
           + I_d / (1 - p)) / C_m
    if one is not None:
        dV1 = (-g_L * (Vs - V_L) - p * (I_Na + I_KDR) - (1 - p) * (I_Ca + I_KC + I_AHP)
               + I_s + I_d) / C_m
        dVs, dVd = np.where(one, dV1, dVs), np.where(one, dV1, dVd)
    return np.array((dVs, dVd) + gate_rhs(Vs, Vd, h, n, s, c, q, Ca)
                    + (-0.13 * I_Ca - 0.075 * Ca,))


def simulate(f, y0, args, t_end=T_END, dt=DT, keep=None, keep_cols=None):
    """RK4 for a batch: y0 has shape (n_state, N), args are length-N arrays.
    Returns somatic spike times per column, the final state and, if
    keep = (t0, t1), every 5th sample of (Vs, Vd) on [t0, t1] for the
    columns keep_cols."""
    y = np.array(y0, dtype=float)
    N = y.shape[1]
    sp = [[] for _ in range(N)]
    n_steps = int(round(t_end / dt))
    store_t, store = [], []
    for k in range(n_steps):
        if keep is not None and keep[0] <= k * dt <= keep[1] and k % 5 == 0:
            store_t.append(k * dt)
            store.append(y[:2, keep_cols].copy())
        k1 = f(y, *args)
        k2 = f(y + 0.5 * dt * k1, *args)
        k3 = f(y + 0.5 * dt * k2, *args)
        k4 = f(y + dt * k3, *args)
        yn = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        hit = (y[0] < V_SPIKE) & (yn[0] >= V_SPIKE)
        if hit.any():
            for j in np.flatnonzero(hit):
                sp[j].append(k * dt + dt * (V_SPIKE - y[0, j]) / (yn[0, j] - y[0, j]))
        y = yn
    out = {"spikes": [np.array(s) for s in sp], "state": y}
    if keep is not None:
        out["t"], out["V"] = np.array(store_t), np.array(store)
    return out


# --------------------------------------------------------------- analysis
def rest_state(I_s):
    """Root of the full steady-state equations; returns the 8-vector."""
    def full(Vs, Vd):
        a_m, b_m, a_h, b_h, a_n, b_n = soma_rates(Vs)
        h, n = a_h / (a_h + b_h), a_n / (a_n + b_n)
        a_s, b_s = dend_rates(Vd, 0.0)[:2]
        s = a_s / (a_s + b_s)
        Ca = -0.13 * g_Ca * s * s * (Vd - V_Ca) / 0.075
        a_s, b_s, a_c, b_c, a_q, b_q = dend_rates(Vd, Ca)
        return np.array([Vs, Vd, h, n, s, a_c / (a_c + b_c), a_q / (a_q + b_q), Ca])

    def F(v):
        return rhs(full(*v), I_s, G_C_STD)[:2]

    root = fsolve(F, [-4.6, -4.6], xtol=1e-12)
    return full(*root)


def describe(spikes, t_from=T_FROM):
    """Period statistics of a spike train on [t_from, T_END]."""
    s = spikes[spikes >= t_from]
    if len(s) < 3:
        return dict(kind="rest" if len(s) == 0 else "sparse", n=len(s), isi=np.nan,
                    per_burst=np.nan, period=np.nan, ratio=np.nan)
    isi = np.diff(s)
    ratio = isi.max() / isi.min()
    if ratio > BURST_RATIO:
        cut = np.sqrt(isi.max() * isi.min())
        starts = np.concatenate(([0], np.flatnonzero(isi > cut) + 1))
        sizes = np.diff(np.concatenate((starts, [len(s)])))
        full = sizes[1:-1] if len(sizes) > 2 else sizes
        period = np.diff(s[starts]).mean() if len(starts) > 1 else np.nan
        per_burst = full.mean()
        kind = "burst" if per_burst >= 1.5 else "mixed"
    else:
        period, per_burst, kind = isi.mean(), 1.0, "single"
    return dict(kind=kind, n=len(s), isi=isi.mean(), per_burst=per_burst,
                period=period, ratio=ratio)


def last_isi(spikes, k=3):
    return np.diff(spikes[-(k + 1):]).mean()


def reference(I_s, g_c, t_end=T_END):
    sol = solve_ivp(lambda t, y: rhs(y, I_s, g_c), (0.0, t_end), Y0, method="DOP853",
                    rtol=REF_TOL, atol=REF_TOL, dense_output=True,
                    events=lambda t, y: y[0] - V_SPIKE)
    ev = sol.t_events[0]
    up = sol.sol(ev - 1e-6)[0] < V_SPIKE
    return ev[up], sol


def jacobian(y, I_s, g_c, eps=1e-7):
    J = np.empty((8, 8))
    for k in range(8):
        e = np.zeros(8)
        e[k] = eps * max(1.0, abs(y[k]))
        J[:, k] = (rhs(y + e, I_s, g_c) - rhs(y - e, I_s, g_c)) / (2 * e[k])
    return J


def slope_and_extrapolation(g, T, T_lim, big=True):
    """Log-log slope of |T - T_lim| vs g (big: over g >= SLOPE_FROM, small:
    over g <= SLOPE_TO; outside those ranges the error is not yet linear in
    1/g or g), and the Richardson extrapolation
    of T to g -> infinity (big) or g -> 0 (not big) from the two most
    extreme points, assuming T = T_lim + A g^-1 (resp. + B g)."""
    err = np.abs(T - T_lim)
    sel = g >= SLOPE_FROM if big else g <= SLOPE_TO
    slope = np.polyfit(np.log(g[sel]), np.log(err[sel]), 1)[0]
    if big:
        i1, i2 = np.argsort(g)[-1], np.argsort(g)[-2]     # g1 = 2 g2
    else:
        i1, i2 = np.argsort(g)[0], np.argsort(g)[1]       # g1 = g2 / 2
    return slope, 2.0 * T[i1] - T[i2], err


# --------------------------------------------------------------- validation
def validate():
    R = {}
    # batch layout: map | big-g_c | small-g_c | rest
    cols_I, cols_g, tags = [], [], []
    for I in I_MAP:
        for g in GC_MAP:
            cols_I.append(I); cols_g.append(g); tags.append(("map", I, g))
    for I in I_MAP:
        for g in GC_BIG:
            cols_I.append(I); cols_g.append(g); tags.append(("big", I, g))
        for g in GC_SMALL:
            cols_I.append(I); cols_g.append(g); tags.append(("small", I, g))
    for I in I_MAP:
        cols_I.append(I); cols_g.append(0.0); tags.append(("iso", I, 0.0))
        cols_I.append(I); cols_g.append(0.0); tags.append(("one", I, np.inf))
    rests = [rest_state(I) for I in I_REST]
    for I in I_REST:
        for kick in (0.0, 1.0):
            cols_I.append(I); cols_g.append(G_C_STD); tags.append(("rest", I, kick))
    N = len(tags)
    y0 = np.repeat(Y0[:, None], N, axis=1)
    for j, t in enumerate(tags):
        if t[0] == "rest":
            y0[:, j] = rests[list(I_REST).index(t[1])]
            y0[:2, j] += t[2]
        if t[0] == "one":
            y0[1, j] = y0[0, j]
    one_mask = np.array([t[0] == "one" for t in tags])
    idx = {t: j for j, t in enumerate(tags)}
    show = [idx[("map", 0.75, G_C_STD)], idx[("map", 1.5, 10.5)],
            idx[("map", 0.75, GC_MAP[np.argmin(np.abs(GC_MAP - 0.5))])]]
    t0 = _time.time()
    run = simulate(rhs, y0, (np.array(cols_I), np.array(cols_g), one_mask),
                   keep=(T_FROM, T_FROM + 1500.0), keep_cols=show)
    print(f"    batch of {N} RK4 runs, {T_END:g} ms at dt = {DT} ms: "
          f"{_time.time() - t0:.1f} s")
    sp = run["spikes"]

    # 1. rest
    devs = []
    for I, yr in zip(I_REST, rests):
        lam = np.linalg.eigvals(jacobian(yr, I, G_C_STD))
        dd = []
        for kick in (0.0, 1.0):
            j = idx[("rest", I, kick)]
            yf = run["state"][:, j]
            dd.append(max(abs(yf[0] - yr[0]), abs(yf[1] - yr[1])))
            assert len(sp[j]) == 0
        slow = lam[np.argmax(lam.real)]
        print(f"[1] rest at I_s = {I:g}: root (Vs, Vd) = ({yr[0]:.9f}, {yr[1]:.9f}) mV; "
              f"run started on it: max |dev| {dd[0]:.2e} mV (tolerance {REST_TOL:g}); "
              f"started +1 mV off: {dd[1]:.2e} mV after {T_END:g} ms (tolerance "
              f"{REST_TOL_KICK:g}); slowest Jacobian eigenvalue {slow.real:.3e} /ms "
              f"(time constant {-1 / slow.real:.0f} ms), max Re = {lam.real.max():.3e}")
        assert dd[0] < REST_TOL and dd[1] < REST_TOL_KICK and lam.real.max() < 0
        devs.append(dd + [slow.real])
    R["rest"] = list(zip(I_REST, rests, devs))

    # 2. limits
    R["big"], R["small"] = [], []
    for i, I in enumerate(I_MAP):
        T_one = last_isi(sp[idx[("one", I, np.inf)]])
        T_big = np.array([last_isi(sp[idx[("big", I, g)]]) for g in GC_BIG])
        for g in GC_BIG:
            d = describe(sp[idx[("big", I, g)]])
            assert d["kind"] == "single", (I, g, d)
        sl, ext, err = slope_and_extrapolation(GC_BIG, T_big, T_one, big=True)
        rel = ext / T_one - 1
        print(f"[2] I_s = {I:g}, g_c -> inf: ISI at g_c = "
              f"{', '.join(f'{g:g}' for g in GC_BIG)}: "
              f"{', '.join(f'{x:.3f}' for x in T_big)} ms; one compartment {T_one:.3f} ms; "
              f"slope of |error| {sl:.3f} (tolerance -1 +/- {SLOPE_TOL}); Richardson "
              f"{ext:.3f} ms ({rel:+.3%}, tolerance {EXTRAP_TOL:.0%})")
        assert abs(sl + 1) < SLOPE_TOL and abs(rel) < EXTRAP_TOL
        R["big"].append(dict(I=I, T=T_big, T_lim=T_one, slope=sl, ext=ext, rel=rel, err=err))

        T_iso = last_isi(sp[idx[("iso", I, 0.0)]])
        T_sm = np.array([last_isi(sp[idx[("small", I, g)]]) for g in GC_SMALL])
        for g in GC_SMALL:
            d = describe(sp[idx[("small", I, g)]])
            assert d["kind"] == "single", (I, g, d)
        sl, ext, err = slope_and_extrapolation(GC_SMALL, T_sm, T_iso, big=False)
        rel = ext / T_iso - 1
        print(f"    I_s = {I:g}, g_c -> 0: ISI at g_c = "
              f"{', '.join(f'{g:g}' for g in GC_SMALL)}: "
              f"{', '.join(f'{x:.4f}' for x in T_sm)} ms; isolated soma {T_iso:.4f} ms; "
              f"slope of |error| {sl:.3f} (tolerance +1 +/- {SLOPE_TOL}); Richardson "
              f"{ext:.4f} ms ({rel:+.3%}, tolerance {EXTRAP_TOL:.0%})")
        assert abs(sl - 1) < SLOPE_TOL and abs(rel) < EXTRAP_TOL
        R["small"].append(dict(I=I, T=T_sm, T_lim=T_iso, slope=sl, ext=ext, rel=rel, err=err))

    # regime map
    M = {}
    for I in I_MAP:
        M[I] = [describe(sp[idx[("map", I, g)]]) for g in GC_MAP]
        kinds = [d["kind"] for d in M[I]]
        b = [g for g, k in zip(GC_MAP, kinds) if k == "burst"]
        line = " ".join(f"{g:.3g}:{k[0]}" + (f"{d['per_burst']:.2g}" if k != "single" else "")
                        for g, k, d in zip(GC_MAP, kinds, M[I]))
        print(f"    regimes at I_s = {I:g} (s single, b burst, m mixed, r rest): {line}")
        if b:
            print(f"      bursting for g_c in [{min(b):.3g}, {max(b):.3g}]")
    R["map"] = M

    # 3. reference solver
    R["ref"] = []
    for I, g in REF_RUNS:
        t0 = _time.time()
        ev, sol = reference(I, g)
        dr = describe(ev)
        dk = describe(sp[idx[("map", I, g)]])
        sk = sp[idx[("map", I, g)]]
        e_per = dk["period"] / dr["period"] - 1
        print(f"[3] I_s = {I:g}, g_c = {g:g}: spikes {len(sk)} (RK4) vs {len(ev)} (DOP853, "
              f"{_time.time() - t0:.1f} s); {dk['kind']} vs {dr['kind']}, "
              f"{dk['per_burst']:.2f} vs {dr['per_burst']:.2f} spikes per event; period "
              f"{dk['period']:.4f} vs {dr['period']:.4f} ms ({e_per:+.2e}, tolerance "
              f"{REF_REL_TOL:g}); max spike-time difference "
              f"{np.abs(sk - ev).max() if len(sk) == len(ev) else np.nan:.4f} ms")
        assert len(sk) == len(ev) and dk["kind"] == dr["kind"]
        assert dk["per_burst"] == dr["per_burst"] and abs(e_per) < REF_REL_TOL
        R["ref"].append(dict(I=I, g=g, sol=sol, ev=ev, dk=dk, dr=dr, e_per=e_per))

    # 4. paper, qualitative
    d_std = M[0.75][list(GC_MAP).index(G_C_STD)]
    t, V = run["t"], run["V"]
    Vd_peak = V[:, 1, 0].max()
    print(f"[4] g_c = {G_C_STD}, I_s = 0.75: {d_std['kind']}, {d_std['per_burst']:.2f} "
          f"spikes per burst, burst period {d_std['period']:.1f} ms, dendritic peak "
          f"{Vd_peak:.1f} mV")
    assert d_std["kind"] == "burst" and Vd_peak > 50.0
    for I in I_MAP:
        for g, d in zip(GC_MAP, M[I]):
            if g >= 10.5:
                assert d["kind"] == "single", (I, g, d)
    print("    single spikes at every g_c >= 10.5 for I_s = "
          f"{', '.join(f'{I:g}' for I in I_MAP)}")
    R.update(run_t=t, run_V=V, show=show, Vd_peak=Vd_peak, d_std=d_std)
    return R


# ----------------------------------------------------------------- plotting
KCOL = {"single": "tab:blue", "burst": "tab:red", "mixed": "tab:purple",
        "rest": "grey", "sparse": "grey"}


def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    t, V = R["run_t"], R["run_V"]
    labels = [f"I_s = 0.75, g_c = {G_C_STD} (standard)", "I_s = 1.5, g_c = 10.5",
              f"I_s = 0.75, g_c = {GC_MAP[np.argmin(np.abs(GC_MAP - 0.5))]:.3g}"]
    for a, k, lab in ((ax[0, 0], 0, labels[0]), (ax[0, 1], 1, labels[1])):
        sel = t <= T_FROM + 1000.0
        a.plot(t[sel] - T_FROM, V[sel, 0, k], color="k", lw=0.8, label="soma Vs (RK4)")
        a.plot(t[sel] - T_FROM, V[sel, 1, k], color="tab:orange", lw=0.8,
               label="dendrite Vd (RK4)")
        r = R["ref"][k]
        ev = r["ev"][(r["ev"] >= T_FROM) & (r["ev"] <= T_FROM + 1000.0)]
        a.plot(ev - T_FROM, np.full(len(ev), 105.0), "v", color="tab:green", ms=5,
               label="DOP853 spike times")
        d = r["dk"]
        a.set_title(f"{lab}: {d['kind']}, {d['per_burst']:.0f} spike(s)/event, "
                    f"period {d['period']:.1f} ms", fontsize=10)
        a.set_xlabel(f"time (ms, from t = {T_FROM:g})")
        a.set_ylabel("V (mV, leak reversal = 0)")
        a.set_ylim(-25, 115)
        a.legend(fontsize=7, loc="upper right")
        if k == 0:
            i_ev = ev[0] - T_FROM if len(ev) else 0.0
            ins = a.inset_axes([0.08, 0.45, 0.32, 0.4])
            z = (t - T_FROM >= i_ev - 5) & (t - T_FROM <= i_ev + 25)
            ins.plot(t[z] - T_FROM, V[z, 0, k], color="k", lw=0.8)
            ins.plot(t[z] - T_FROM, V[z, 1, k], color="tab:orange", lw=0.8)
            ins.set_title("burst, zoomed", fontsize=7)
            ins.tick_params(labelsize=6)

    a = ax[0, 2]
    sel = (t >= T_FROM) & (t <= T_FROM + 150.0)
    k0 = R["d_std"]
    a.plot(t[sel] - T_FROM, V[sel, 0, 2], color="k", lw=0.8, label="soma Vs")
    a.plot(t[sel] - T_FROM, V[sel, 1, 2], color="tab:orange", lw=0.8, label="dendrite Vd")
    a.set_title(labels[2] + ": weakly coupled soma fires alone", fontsize=10)
    a.set_xlabel(f"time (ms, from t = {T_FROM:g})")
    a.set_ylabel("V (mV)")
    a.legend(fontsize=7, loc="upper right")

    a = ax[1, 0]
    for I, mk in zip(I_MAP, ("o", "s", "^")):
        M = R["map"][I]
        y = [d["per_burst"] if d["kind"] in ("single", "burst", "mixed") else 0 for d in M]
        a.plot(GC_MAP, y, "-", color="grey", lw=0.5)
        for g, d, yy in zip(GC_MAP, M, y):
            a.plot(g, yy, mk, color=KCOL[d["kind"]], ms=5, mfc="none" if I != 0.75 else None)
        a.plot([], [], mk, color="k", mfc="none" if I != 0.75 else None, label=f"I_s = {I:g}")
    for kk in ("single", "burst", "mixed"):
        a.plot([], [], "s", color=KCOL[kk], label=kk)
    a.axvline(G_C_STD, color="tab:red", ls=":", lw=1)
    a.set_xscale("log")
    a.set_xlabel("coupling g_c (mS/cm²)")
    a.set_ylabel("spikes per event")
    a.set_title("Regime vs coupling (dotted: standard g_c = 2.1)", fontsize=10)
    a.legend(fontsize=7, ncol=2)

    a = ax[1, 1]
    cols = ["tab:blue", "tab:orange", "tab:red"]
    for r, c in zip(R["big"], cols):
        a.loglog(GC_BIG, r["err"], "o-", color=c, ms=4,
                 label=f"I_s = {r['I']:g}: slope {r['slope']:.2f}, extrap. {r['rel']:+.2%}")
    for r, c in zip(R["small"], cols):
        a.loglog(GC_SMALL, r["err"], "s--", color=c, ms=4, mfc="none",
                 label=f"g_c→0, I_s = {r['I']:g}: slope {r['slope']:.2f}, extrap. {r['rel']:+.2%}")
    gg = np.array([10.0, 60.0])
    a.loglog(gg, 3000.0 / gg, "k:", lw=1, label="∝ 1/g_c, ∝ g_c")
    gg = np.array([4e-4, 0.05])
    a.loglog(gg, 120.0 * gg, "k:", lw=1)
    a.set_xlabel("g_c (mS/cm²)")
    a.set_ylabel("|ISI − ISI of reduced model| (ms)")
    a.set_title("Limits: one compartment (g_c→∞), isolated soma (g_c→0)", fontsize=10)
    a.legend(fontsize=6)

    a = ax[1, 2]
    for r, c in zip(R["big"], cols):
        a.semilogx(GC_BIG, r["T"], "o-", color=c, ms=4, label=f"two-compartment, I_s = {r['I']:g}")
        a.axhline(r["T_lim"], color=c, ls="--", lw=1)
        a.plot([80.0], [r["ext"]], "x", color=c, ms=8)
    a.plot([], [], "k--", lw=1, label="one-compartment reduction")
    a.plot([], [], "kx", label="Richardson from g_c = 25, 50")
    a.set_xlabel("g_c (mS/cm²)")
    a.set_ylabel("interspike interval (ms)")
    a.set_title("Strong coupling: single spikes approach one compartment", fontsize=10)
    a.legend(fontsize=7)

    fig.suptitle("Pinsky–Rinzel two-compartment CA3 cell: RK4 dt = 0.02 ms; checks against "
                 "rest root, reduced limits, DOP853 and the paper's bursting at g_c = 2.1",
                 fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "pinsky_rinzel.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
