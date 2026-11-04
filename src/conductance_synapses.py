"""Conductance-based synapses: AMPA, NMDA with Mg2+ block, and GABA-A, with
exponential, alpha and bi-exponential kinetics, on a passive membrane.

Postsynaptic membrane (single passive compartment; Dayan & Abbott 2001,
secs. 5.8 and 5.9), V in mV, t in ms, conductances in mS/cm^2:

    C_m dV/dt = -g_L (V - E_L) - sum_k g_k(t) B_k(V) (V - E_k) + I_hold

with C_m = 1 uF/cm^2, g_L = 0.05 mS/cm^2 (tau_m = C_m/g_L = 20 ms),
E_L = -70 mV. B_k = 1 except for NMDA.

Synaptic conductance g_k(t) = g_max sum_spikes s(t - t_spike), with the
unit waveform s(t) (t >= 0) one of

    exponential:     s(t) = exp(-t/tau)
    alpha:           s(t) = (t/tau) exp(1 - t/tau)                (peak 1 at t = tau)
    bi-exponential:  s(t) = N [exp(-t/tau_d) - exp(-t/tau_r)],
                     N = 1 / [exp(-t_p/tau_d) - exp(-t_p/tau_r)],
                     t_p = tau_r tau_d / (tau_d - tau_r) ln(tau_d/tau_r)  (peak 1 at t_p)

each written as a linear two-state ODE so it can be integrated with V:
    exponential:  p' = -p/tau;                      g = g_max p;       spike: p += 1
    alpha:        p' = -p/tau + q,  q' = -q/tau;    g = g_max p;       spike: q += e/tau
    bi-exp:       p' = -p/tau_d,    q' = -q/tau_r;  g = g_max (p - q); spike: p += N, q += N

Receptor parameters (bi-exponential; typical values in the ranges quoted by
Destexhe, Mainen & Sejnowski 1998 and Dayan & Abbott 2001, sec. 5.8 --
chosen here, not fitted to data):
    AMPA:   tau_r = 0.2 ms, tau_d = 2 ms,  E = 0 mV
    NMDA:   tau_r = 2 ms,   tau_d = 80 ms, E = 0 mV
    GABA-A: tau_r = 0.5 ms, tau_d = 6 ms,  E = -80 mV

NMDA magnesium block (Jahr & Stevens 1990, eq. 4 as written in Dayan &
Abbott 2001, eq. 5.22, and Gerstner et al. 2014, sec. 3.1.2):

    B(V) = 1 / (1 + [Mg2+]/3.57 mM * exp(-0.062 V)),   [Mg2+] = 1 mM by default,

taken as instantaneous in V.

Integrator: classical RK4 on (V, all synaptic states) at dt = 0.01 ms.
Presynaptic spikes sit on the time grid and their jumps are applied exactly
between steps.

Validation (validate()):
1. Closed form, conductance: simulated unit waveforms for the three kinetics
   vs the formulas above; max |error| < 1e-6 (peak = 1) and below twice the
   RK4 truncation estimate amp * exp(-1) (dt/tau)^4 / 120 for the fastest tau.
2. Reference "solver" from the exact solution: for conductances without Mg
   block the membrane equation is linear in V with time-varying coefficient,
   so with u = V - E_L, a_k = g_k/C_m, A = int sum a_k:
       u(t) = int_0^t sum_k a_k(s) (E_k - E_L) exp(-(t-s)/tau_m - (A(t) - A(s))) ds
   with A(t) in closed form; evaluated by scipy.integrate.quad. AMPA, GABA-A,
   exponential and alpha PSPs and a 5-spike 100 Hz AMPA train (sublinear
   summation through the driving force) vs this solution; max error / PSP
   peak < 1e-7. RK4 order from the train at dt = 0.1 ... 0.0125 ms: 4 +- 0.3.
3. Closed form, PSP shape: an exponential conductance with tau_s = tau_m is,
   to first order in g_max, the alpha-function PSP
       u(t) = (E_s - E_L) (g_max/C_m) t exp(-t/tau_m),
   peak (E_s - E_L) g_max tau_m/(C_m e) at t = tau_m. Relative deviation
   max |u - u_alpha| / max u_alpha < 1% at g_max = 1e-4 mS/cm^2 and falling
   as g_max^1 (log-log slope 1 +- 0.05) over 1e-5 ... 1e-2; peak time within
   0.05 ms of tau_m at the smallest g_max.
4. Closed form, NMDA I-V (Jahr-Stevens): ideal voltage clamp at -100 ... +40
   mV for [Mg] = 0, 0.5, 1, 2 mM; peak current vs g_max B(V) (V - E_NMDA),
   max relative error < 1e-5; the I-V minimum at 1 mM (fine clamp grid,
   0.1 mV) vs the root of d/dV[B(V)(V - E)] = 0 (brentq), within 0.1 mV;
   half-block voltage at 1 mM from the simulated I/I(Mg=0) vs
   ln([Mg]/3.57)/0.062 within 0.05 mV; Boltzmann fit to the simulated block
   recovers K = 3.57 mM and delta = 0.062 /mV to 1e-4 relative; Mg = 0 I-V
   linear with reversal at E_NMDA (< 1e-6 mV).
5. Reference solver: NMDA EPSP and a 5-spike 50 Hz NMDA train with Mg block in
   current clamp, RK4 vs scipy solve_ivp (DOP853, rtol = atol = 1e-12) with
   the same jumps; max |dV| < 1e-6 mV.
6. GABA-A reversal: the IPSP amplitude as a function of the holding
   potential (set by I_hold = g_L (V_hold - E_L)) crosses zero at E_GABA;
   brentq on the simulated amplitude, |V_rev - E_GABA| < 1e-6 mV.

The model has no randomness, so no RNG is used. SciPy is used only for the
validation (quad, solve_ivp, brentq, curve_fit).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.integrate import quad, solve_ivp  # noqa: E402
from scipy.optimize import brentq, curve_fit  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Parameters
C_M = 1.0        # uF/cm^2
G_L = 0.05       # mS/cm^2
E_L = -70.0      # mV
TAU_M = C_M / G_L  # 20 ms
DT = 0.01        # ms

MG_DEFAULT = 1.0   # mM
K_MG = 3.57        # mM (Jahr & Stevens 1990)
DELTA_MG = 0.062   # 1/mV

RECEPTORS = {
    "AMPA": dict(shape="biexp", tau_r=0.2, tau_d=2.0, E=0.0, nmda=False),
    "NMDA": dict(shape="biexp", tau_r=2.0, tau_d=80.0, E=0.0, nmda=True),
    "GABA_A": dict(shape="biexp", tau_r=0.5, tau_d=6.0, E=-80.0, nmda=False),
}


# ---------------------------------------------------------------------------
# Model functions
def mg_block(V, mg):
    """Jahr-Stevens (1990) fraction of NMDA channels not blocked by Mg2+."""
    return 1.0 / (1.0 + (mg / K_MG) * np.exp(-DELTA_MG * V))


def biexp_peak(tau_r, tau_d):
    tp = tau_r * tau_d / (tau_d - tau_r) * np.log(tau_d / tau_r)
    norm = 1.0 / (np.exp(-tp / tau_d) - np.exp(-tp / tau_r))
    return tp, norm


def make_synapse(shape, g_max, E, tau=None, tau_r=None, tau_d=None, nmda=False,
                 mg=MG_DEFAULT, spikes=()):
    """Linear two-state representation: S' = M S, g = g_max (c . S), jump j."""
    if shape == "exp":
        M = np.array([[-1.0 / tau, 0.0], [0.0, 0.0]])
        c = np.array([1.0, 0.0])
        j = np.array([1.0, 0.0])
    elif shape == "alpha":
        M = np.array([[-1.0 / tau, 1.0], [0.0, -1.0 / tau]])
        c = np.array([1.0, 0.0])
        j = np.array([0.0, np.e / tau])
    elif shape == "biexp":
        _, norm = biexp_peak(tau_r, tau_d)
        M = np.array([[-1.0 / tau_d, 0.0], [0.0, -1.0 / tau_r]])
        c = np.array([1.0, -1.0])
        j = np.array([norm, norm])
    else:
        raise ValueError(shape)
    return dict(shape=shape, M=M, c=c, j=j, g_max=g_max, E=E, tau=tau,
                tau_r=tau_r, tau_d=tau_d, nmda=nmda, mg=mg,
                spikes=np.asarray(spikes, float))


def receptor(name, g_max, spikes=(0.0,), mg=MG_DEFAULT, shape=None, **kw):
    p = dict(RECEPTORS[name])
    if shape is not None:
        p["shape"] = shape
    p.update(kw)
    return make_synapse(p["shape"], g_max, p["E"], tau=p.get("tau"),
                        tau_r=p.get("tau_r"), tau_d=p.get("tau_d"),
                        nmda=p["nmda"], mg=mg, spikes=spikes)


def unit_closed(t, syn):
    """Closed-form unit waveform s(t) summed over the synapse's spikes."""
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for ts in syn["spikes"]:
        x = t - ts
        on = x >= 0
        xx = np.where(on, x, 0.0)
        if syn["shape"] == "exp":
            s = np.exp(-xx / syn["tau"])
        elif syn["shape"] == "alpha":
            s = (xx / syn["tau"]) * np.exp(1.0 - xx / syn["tau"])
        else:
            _, n = biexp_peak(syn["tau_r"], syn["tau_d"])
            s = n * (np.exp(-xx / syn["tau_d"]) - np.exp(-xx / syn["tau_r"]))
        out += np.where(on, s, 0.0)
    return out


def unit_integral(t, syn):
    """Closed-form integral of the unit waveform from 0 to t."""
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    for ts in syn["spikes"]:
        x = np.maximum(t - ts, 0.0)
        if syn["shape"] == "exp":
            tau = syn["tau"]
            out += tau * (1.0 - np.exp(-x / tau))
        elif syn["shape"] == "alpha":
            tau = syn["tau"]
            out += np.e * tau * (1.0 - (1.0 + x / tau) * np.exp(-x / tau))
        else:
            tr, td = syn["tau_r"], syn["tau_d"]
            _, n = biexp_peak(tr, td)
            out += n * (td * (1.0 - np.exp(-x / td)) - tr * (1.0 - np.exp(-x / tr)))
    return out


def syn_current(V, S, syns):
    """Total synaptic current density (uA/cm^2) and per-synapse conductances."""
    I = np.zeros_like(V)
    gs = []
    for k, sy in enumerate(syns):
        g = sy["g_max"] * (sy["c"] @ S[k])
        if sy["nmda"]:
            g = g * mg_block(V, sy["mg"])
        I = I + g * (V - sy["E"])
        gs.append(g)
    return I, gs


def simulate(syns, T, dt=DT, V0=E_L, I_hold=0.0, clamp=False, record_every=1):
    """RK4 on (V, synaptic states). V0, g_max, mg and I_hold may be arrays of
    shape (n,) to run n conditions at once. With clamp=True, V is held at V0
    and the synaptic current is recorded instead. Spikes must lie on the
    grid; their jumps are applied before the step that starts at t_spike."""
    nsteps = int(round(T / dt))
    V = np.array(np.broadcast_to(np.asarray(V0, float), np.broadcast(
        np.asarray(V0), *[np.asarray(s["g_max"]) for s in syns],
        *[np.asarray(s["mg"]) for s in syns], np.asarray(I_hold)).shape), dtype=float)
    V = np.atleast_1d(V).copy()
    S = np.zeros((len(syns), 2))
    jump_at = {}
    for k, sy in enumerate(syns):
        for ts in sy["spikes"]:
            i = int(round(ts / dt))
            if abs(i * dt - ts) > 1e-9:
                raise ValueError("spike times must lie on the dt grid")
            jump_at.setdefault(i, []).append(k)
    Ms = [sy["M"] for sy in syns]

    def fS(S_):
        return np.array([Ms[k] @ S_[k] for k in range(len(syns))])

    def fV(V_, S_):
        if clamp:
            return np.zeros_like(V_)
        I, _ = syn_current(V_, S_, syns)
        return (-G_L * (V_ - E_L) - I + I_hold) / C_M

    nrec = nsteps // record_every + 1
    t_rec = np.arange(nrec) * dt * record_every
    V_rec = np.empty((nrec, V.size))
    I_rec = np.empty((nrec, V.size))
    g_rec = np.empty((nrec, len(syns)))
    r = 0
    for i in range(nsteps + 1):
        for k in jump_at.get(i, ()):
            S[k] += syns[k]["j"]
        if i % record_every == 0:
            I, _ = syn_current(V, S, syns)
            V_rec[r], I_rec[r] = V, I
            g_rec[r] = [sy["c"] @ S[k] for k, sy in enumerate(syns)]
            r += 1
        if i == nsteps:
            break
        k1s, k1v = fS(S), fV(V, S)
        S2, V2 = S + 0.5 * dt * k1s, V + 0.5 * dt * k1v
        k2s, k2v = fS(S2), fV(V2, S2)
        S3, V3 = S + 0.5 * dt * k2s, V + 0.5 * dt * k2v
        k3s, k3v = fS(S3), fV(V3, S3)
        S4, V4 = S + dt * k3s, V + dt * k3v
        k4s, k4v = fS(S4), fV(V4, S4)
        S = S + dt / 6.0 * (k1s + 2 * k2s + 2 * k3s + k4s)
        V = V + dt / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return t_rec, V_rec, I_rec, g_rec


# ---------------------------------------------------------------------------
# Validation references
def exact_psp(t_eval, syns, V0=E_L):
    """Exact u(t) = V - E_L for Mg-free conductances, starting at rest, by
    quadrature of the integrating-factor solution."""
    assert V0 == E_L and not any(s["nmda"] for s in syns)
    pts = sorted({float(ts) for s in syns for ts in s["spikes"]})

    def A(t):
        return sum(s["g_max"] * unit_integral(t, s) for s in syns) / C_M

    out = []
    for t in np.atleast_1d(t_eval):
        At = A(t)

        def f(s):
            src = sum(sy["g_max"] * unit_closed(s, sy) * (sy["E"] - E_L) for sy in syns) / C_M
            return src * np.exp(-(t - s) / TAU_M - (At - A(s)))

        p = [x for x in pts if 0.0 < x < t]
        val, _ = quad(f, 0.0, t, points=p or None, limit=500, epsabs=1e-14, epsrel=1e-13)
        out.append(val)
    return np.array(out)


def reference_ivp(syns, T, t_eval, V0=E_L):
    """DOP853 on the same ODE, segment by segment between spikes."""
    ns = len(syns)

    def rhs(t, y):
        V = y[:1]
        S = y[1:].reshape(ns, 2)
        I, _ = syn_current(V, S, syns)
        dS = np.array([syns[k]["M"] @ S[k] for k in range(ns)])
        return np.concatenate([(-G_L * (V - E_L) - I) / C_M, dS.ravel()])

    events = sorted({float(ts) for s in syns for ts in s["spikes"]})
    y = np.concatenate([[V0], np.zeros(2 * ns)])
    edges = [0.0] + [e for e in events if 0 < e < T] + [T]
    Vout = np.full(len(t_eval), np.nan)
    for a, b in zip(edges[:-1], edges[1:]):
        S = y[1:].reshape(ns, 2)
        for k, sy in enumerate(syns):
            S[k] += sy["j"] * np.sum(np.isclose(sy["spikes"], a))
        y[1:] = S.ravel()
        m = (t_eval >= a) & (t_eval < b) if b < T else (t_eval >= a) & (t_eval <= b)
        sol = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-12, atol=1e-12,
                        t_eval=t_eval[m] if m.any() else None, dense_output=False)
        if m.any():
            Vout[m] = sol.y[0]
        y = solve_ivp(rhs, (a, b), y, method="DOP853", rtol=1e-12, atol=1e-12).y[:, -1]
    return Vout


def loglog_slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def iv_minimum(mg, E=0.0):
    """Root of d/dV [B(V)(V - E)] = 0."""
    def d(V):
        B = mg_block(V, mg)
        return B + (V - E) * DELTA_MG * B * (1 - B)
    return brentq(d, -80.0, -5.0, xtol=1e-12)


# ---------------------------------------------------------------------------
def validate():
    R = {}
    print("Integrator: RK4, dt = %.3g ms; tau_m = %.1f ms" % (DT, TAU_M))

    # 1. conductance waveforms -------------------------------------------
    print("\n1. Unit conductance waveforms vs closed form (spike at t = 1 ms)")
    shapes = [
        ("exponential, tau = 5 ms", make_synapse("exp", 1.0, 0.0, tau=5.0, spikes=[1.0])),
        ("alpha, tau = 5 ms", make_synapse("alpha", 1.0, 0.0, tau=5.0, spikes=[1.0])),
        ("bi-exp AMPA 0.2/2 ms", receptor("AMPA", 1.0, spikes=[1.0])),
        ("bi-exp NMDA 2/80 ms", receptor("NMDA", 1.0, spikes=[1.0], mg=0.0)),
        ("bi-exp GABA-A 0.5/6 ms", receptor("GABA_A", 1.0, spikes=[1.0])),
    ]
    R["shapes"] = []
    for name, sy in shapes:
        t, _, _, g = simulate([sy], 40.0, clamp=True, V0=-70.0)
        gc = unit_closed(t, sy)
        err = np.max(np.abs(g[:, 0] - gc))
        tpk = t[np.argmax(g[:, 0])] - 1.0
        # RK4 truncation on exp(-t/tau): global error ~ (t/tau) e^(-t/tau) (dt/tau)^4/120,
        # maximal at t = tau; summed over the (normalised) components
        taus = [sy["tau"]] if sy["shape"] != "biexp" else [sy["tau_d"], sy["tau_r"]]
        amp = 1.0 if sy["shape"] == "exp" else (np.e / sy["tau"] * sy["tau"] if sy["shape"] == "alpha"
                                                 else sy["j"][0])
        pred = amp * max(np.exp(-1.0) * (DT / tau) ** 4 / 120.0 for tau in taus)
        print(f"   {name:26s} max|err| = {err:.2e} (RK4 truncation estimate {pred:.2e}), "
              f"peak at {tpk:.2f} ms after spike")
        # (the 1e-12 floor covers round-off; for the alpha kernel the estimate
        # omits the t^2 term and is only indicative)
        assert err < 1e-6 and err < 2.0 * pred + 1e-12, \
            f"{name}: conductance error {err:.2e} (bound 1e-6, 2 x {pred:.2e})"
        R["shapes"].append((name, t, g[:, 0], gc))

    # 2. PSPs vs exact quadrature ----------------------------------------
    print("\n2. PSPs (no Mg block) vs exact integrating-factor solution (quad)")
    cases = [
        ("AMPA, g = 0.01", [receptor("AMPA", 0.01)], 60.0),
        ("GABA-A, g = 0.01", [receptor("GABA_A", 0.01)], 60.0),
        ("exp tau 5, g = 0.005", [make_synapse("exp", 0.005, 0.0, tau=5.0, spikes=[0.0])], 60.0),
        ("alpha tau 5, g = 0.005", [make_synapse("alpha", 0.005, 0.0, tau=5.0, spikes=[0.0])], 60.0),
        ("AMPA 5 x 100 Hz, g = 0.05", [receptor("AMPA", 0.05, spikes=np.arange(5) * 10.0)], 100.0),
    ]
    R["psp"] = {}
    for name, syns, T in cases:
        t, V, _, _ = simulate(syns, T)
        te = np.arange(0.5, T, 0.5)
        ue = exact_psp(te, syns)
        idx = np.round(te / DT).astype(int)
        u = V[idx, 0] - E_L
        pk = np.max(np.abs(V[:, 0] - E_L))
        err = np.max(np.abs(u - ue)) / pk
        print(f"   {name:26s} peak {np.sign(u[np.argmax(np.abs(u))]) * pk:+.4f} mV, "
              f"max err/peak = {err:.2e}")
        assert err < 1e-7, f"{name}: PSP error {err:.2e} > 1e-7"
        R["psp"][name] = (t, V[:, 0], te, ue + E_L, err)
    # sublinear summation in the train
    t1, V1, _, _ = simulate([receptor("AMPA", 0.05)], 100.0)
    tt, Vt, _, _ = simulate(cases[-1][1], 100.0)
    lin = sum(np.interp(tt - 10.0 * k, t1, V1[:, 0] - E_L, left=0.0) for k in range(5))
    ratio = np.max(Vt[:, 0] - E_L) / np.max(lin)
    print(f"   train peak {np.max(Vt[:, 0] - E_L):.3f} mV vs linear sum of single PSPs "
          f"{np.max(lin):.3f} mV (ratio {ratio:.3f}; driving-force sublinearity, not asserted)")
    R["train_lin"] = (tt, lin + E_L, ratio)

    # RK4 order
    syns = cases[-1][1]
    te = np.arange(0.5, 100.0, 0.5)
    ue = exact_psp(te, syns)
    dts = np.array([0.1, 0.05, 0.025, 0.0125])
    errs = []
    for d in dts:
        t, V, _, _ = simulate(syns, 100.0, dt=d)
        idx = np.round(te / d).astype(int)
        errs.append(np.max(np.abs(V[idx, 0] - E_L - ue)))
    errs = np.array(errs)
    order = loglog_slope(dts, errs)
    print(f"   RK4 order (AMPA train, dt = 0.1 ... 0.0125 ms): errors "
          + ", ".join(f"{e:.2e}" for e in errs) + f" mV; slope {order:.3f}")
    assert abs(order - 4.0) < 0.3, f"RK4 order {order:.3f} not 4 +- 0.3"
    R["order"] = (dts, errs, order)

    # 3. alpha-function PSP limit ----------------------------------------
    print("\n3. Exponential conductance with tau_s = tau_m vs alpha-function PSP (linear limit)")
    g0 = np.array([1e-5, 1e-4, 1e-3, 1e-2])
    sy = make_synapse("exp", g0, 0.0, tau=TAU_M, spikes=[0.0])
    t, V, _, _ = simulate([sy], 150.0, record_every=10)
    dE = 0.0 - E_L
    ua = dE * (g0[None, :] / C_M) * t[:, None] * np.exp(-t[:, None] / TAU_M)
    u = V - E_L
    rel = np.max(np.abs(u - ua), axis=0) / np.max(ua, axis=0)
    slope = loglog_slope(g0, rel)
    tpk = t[np.argmax(u, axis=0)]
    pk_pred = dE * g0 * TAU_M / (C_M * np.e)
    for k in range(len(g0)):
        print(f"   g = {g0[k]:.0e}: peak {u[:, k].max():.5g} mV (alpha {pk_pred[k]:.5g}), "
              f"at {tpk[k]:.2f} ms; max |u - u_alpha|/peak = {rel[k]:.3e}")
    print(f"   log-log slope of deviation vs g: {slope:.4f}")
    # finer peak time at smallest g
    t2, V2, _, _ = simulate([make_synapse("exp", 1e-5, 0.0, tau=TAU_M, spikes=[0.0])], 40.0)
    tpk_fine = t2[np.argmax(V2[:, 0])]
    print(f"   peak time at g = 1e-5 (dt grid): {tpk_fine:.2f} ms vs tau_m = {TAU_M:.2f} ms")
    assert rel[1] < 1e-2, f"alpha deviation {rel[1]:.2e} >= 1% at g = 1e-4"
    assert abs(slope - 1.0) < 0.05, f"deviation slope {slope:.3f} not 1 +- 0.05"
    assert abs(tpk_fine - TAU_M) < 0.05, "alpha peak time off by > 0.05 ms"
    R["alpha"] = (t, u, ua, g0, rel, slope)

    # 4. NMDA I-V -----------------------------------------------------------
    print("\n4. NMDA I-V under ideal voltage clamp vs Jahr-Stevens block")
    Vc = np.linspace(-100.0, 40.0, 29)
    mgs = np.array([0.0, 0.5, 1.0, 2.0])
    VV, MM = np.meshgrid(Vc, mgs)
    gN = 0.01
    syn = receptor("NMDA", gN, mg=MM.ravel())
    t, _, I, g = simulate([syn], 30.0, V0=VV.ravel(), clamp=True)
    ip = np.argmax(g[:, 0])
    Ipk = I[ip].reshape(VV.shape)
    Ith = gN * g[ip, 0] * mg_block(VV, MM) * (VV - 0.0)
    nz = np.abs(Ith) > 1e-12
    relI = np.max(np.abs(Ipk[nz] - Ith[nz]) / np.abs(Ith[nz]))
    tp, _ = biexp_peak(2.0, 80.0)
    print(f"   peak g at {t[ip]:.2f} ms (closed form {tp:.4f}), sampled value {g[ip, 0]:.9f}")
    print(f"   max relative error of peak current over 116 clamps: {relI:.2e}")
    assert relI < 1e-5, f"NMDA I-V error {relI:.2e} > 1e-5"
    # Mg = 0 linear, reversal
    pfit = np.polyfit(Vc, Ipk[0], 1)
    Vrev = -pfit[1] / pfit[0]
    lin_res = np.max(np.abs(np.polyval(pfit, Vc) - Ipk[0])) / np.max(np.abs(Ipk[0]))
    print(f"   Mg = 0: linear fit residual {lin_res:.1e}, reversal {Vrev:+.2e} mV (E_NMDA = 0)")
    assert abs(Vrev) < 1e-6 and lin_res < 1e-9, "Mg-free NMDA I-V not ohmic about E_NMDA"
    # fine grid minimum at 1 mM
    Vf = np.arange(-60.0, -9.95, 0.1)
    t, _, If, gf = simulate([receptor("NMDA", gN, mg=1.0)], 10.0, V0=Vf, clamp=True)
    ipf = np.argmax(gf[:, 0])
    Vmin_sim = Vf[np.argmin(If[ipf])]
    Vmin_th = iv_minimum(1.0)
    print(f"   I-V minimum at 1 mM: {Vmin_sim:.2f} mV (0.1 mV grid) vs {Vmin_th:.4f} mV (root of slope)")
    assert abs(Vmin_sim - Vmin_th) < 0.1, "I-V minimum off by > 0.1 mV"
    # negative slope conductance region upper end = minimum; report slope at -70
    # half-block voltage at 1 mM
    ratio = If[ipf] / (gN * gf[ipf, 0] * (Vf - 0.0))
    Vhalf_sim = np.interp(0.5, ratio, Vf)
    Vhalf_th = np.log(1.0 / K_MG) / DELTA_MG
    print(f"   half-block voltage at 1 mM: {Vhalf_sim:.4f} mV (interpolated) vs "
          f"ln(1/3.57)/0.062 = {Vhalf_th:.4f} mV")
    assert abs(Vhalf_sim - Vhalf_th) < 0.05, "half-block voltage off by > 0.05 mV"
    # Boltzmann fit recovering K and delta
    m = (MM > 0) & (np.abs(VV) > 1e-9)
    frac = Ipk[m] / np.broadcast_to(Ipk[0][None, :], MM.shape)[m]

    def bfun(X, K, d):
        v, mg = X
        return 1.0 / (1.0 + mg / K * np.exp(-d * v))

    (Kfit, dfit), _ = curve_fit(bfun, (VV[m], MM[m]), frac, p0=(2.0, 0.05))
    eK, ed = Kfit / K_MG - 1, dfit / DELTA_MG - 1
    print(f"   Boltzmann fit to simulated block: K = {Kfit:.5f} mM ({eK:+.1e}), "
          f"delta = {dfit:.6f} /mV ({ed:+.1e})")
    assert abs(eK) < 1e-4 and abs(ed) < 1e-4, "block fit does not recover Jahr-Stevens constants"
    B70 = mg_block(-70.0, 1.0)
    print(f"   unblocked fraction at 1 mM: {B70:.4f} at -70 mV, {mg_block(-40.0, 1.0):.4f} at -40, "
          f"{mg_block(0.0, 1.0):.4f} at 0 mV")
    R["iv"] = (Vc, mgs, Ipk, Vmin_th, Vhalf_th)

    # 5. NMDA in current clamp vs DOP853 ------------------------------------
    print("\n5. NMDA EPSPs with Mg block (1 mM), RK4 vs DOP853")
    R["nmda"] = {}
    for name, syns, T in [
        ("single, g = 0.01", [receptor("NMDA", 0.01)], 300.0),
        ("5 x 50 Hz, g = 0.01", [receptor("NMDA", 0.01, spikes=np.arange(5) * 20.0)], 400.0),
        ("5 x 50 Hz, g = 0.01, Mg = 0", [receptor("NMDA", 0.01, spikes=np.arange(5) * 20.0, mg=0.0)], 400.0),
        ("5 x 50 Hz, g = 0.05", [receptor("NMDA", 0.05, spikes=np.arange(5) * 20.0)], 400.0),
    ]:
        t, V, _, _ = simulate(syns, T)
        te = np.arange(1.0, T, 1.0)
        Vr = reference_ivp(syns, T, te)
        idx = np.round(te / DT).astype(int)
        err = np.max(np.abs(V[idx, 0] - Vr))
        print(f"   {name:28s} peak {V[:, 0].max() - E_L:.4f} mV at {t[np.argmax(V[:, 0])]:.1f} ms; "
              f"max|RK4 - DOP853| = {err:.2e} mV")
        assert err < 1e-6, f"NMDA {name}: RK4 vs DOP853 {err:.2e} > 1e-6 mV"
        R["nmda"][name] = (t, V[:, 0], te, Vr)
    # block relief in the train: effective B at the train peak
    for gx in (0.01, 0.05):
        _, V, _, _ = simulate([receptor("NMDA", gx, spikes=np.arange(5) * 20.0)], 400.0)
        Vpk = V[:, 0].max()
        print(f"   g = {gx}: B(V) at rest {mg_block(E_L, 1.0):.4f}, at train peak "
              f"({Vpk:.2f} mV) {mg_block(Vpk, 1.0):.4f} (not asserted)")

    # 6. GABA-A reversal ---------------------------------------------------
    print("\n6. GABA-A IPSP reversal vs E_GABA = -80 mV")

    def ipsp_amp(Vh):
        Ih = G_L * (Vh - E_L)
        t, V, _, _ = simulate([receptor("GABA_A", 0.01)], 30.0, V0=Vh, I_hold=Ih)
        d = V[:, 0] - Vh
        return d[np.argmax(np.abs(d))]

    Vh_grid = np.linspace(-95.0, -55.0, 9)
    amps = np.array([ipsp_amp(v) for v in Vh_grid])
    Vrev_g = brentq(ipsp_amp, -90.0, -70.0, xtol=1e-10)
    print("   IPSP amplitude at V_hold = " + ", ".join(f"{v:.0f}: {a:+.3f}" for v, a in zip(Vh_grid, amps)) + " mV")
    print(f"   reversal by brentq: {Vrev_g:.9f} mV, |V_rev - E_GABA| = {abs(Vrev_g + 80.0):.1e} mV")
    assert abs(Vrev_g + 80.0) < 1e-6, "GABA-A reversal off"
    R["gaba"] = (Vh_grid, amps, Vrev_g)

    print("\nAll assertions passed.")
    return R


# ---------------------------------------------------------------------------
def plot(R):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))

    # A: conductance waveforms
    ax = axes[0, 0]
    for k, (name, t, g, gc) in enumerate(R["shapes"]):
        c = f"C{k}"
        ax.plot(t[::40], g[::40], "o", ms=3, mfc="none", color=c)
        ax.plot(t, gc, color=c, lw=1.3, label=name)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("g / g_max")
    ax.set_title("Conductance waveforms: RK4 (o) vs closed form (—)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # B: PSPs vs exact
    ax = axes[0, 1]
    for k, name in enumerate(["AMPA, g = 0.01", "GABA-A, g = 0.01", "AMPA 5 x 100 Hz, g = 0.05"]):
        t, V, te, Ve, err = R["psp"][name]
        ax.plot(t, V, color=f"C{k}", lw=1.3, label=f"{name} (err/peak {err:.0e})")
        ax.plot(te[::4], Ve[::4], "k.", ms=3)
    tt, lin, ratio = R["train_lin"]
    ax.plot(tt, lin, "C2--", lw=1, label=f"linear sum of single PSPs (train/linear {ratio:.3f})")
    ax.axhline(E_L, color="0.6", lw=0.8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("PSPs: RK4 (—) vs exact quadrature (•)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # C: alpha-function limit
    ax = axes[0, 2]
    t, u, ua, g0, rel, slope = R["alpha"]
    for k in range(len(g0)):
        ax.plot(t, u[:, k] / ua[:, k].max(), color=f"C{k}", lw=1.3,
                label=f"g = {g0[k]:.0e}: dev {rel[k]:.1e}")
    ax.plot(t[::40], (ua[:, 0] / ua[:, 0].max())[::40], "ko", ms=3, mfc="none",
            label="alpha function (t/τ_m) e^{1−t/τ_m}")
    ax.axvline(TAU_M, color="0.6", lw=0.8)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("u / peak of linear prediction")
    ax.set_title("Exp. conductance, τ_s = τ_m = 20 ms → alpha-function PSP")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # D: NMDA I-V
    ax = axes[1, 0]
    Vc, mgs, Ipk, Vmin, Vhalf = R["iv"]
    Vf = np.linspace(-100, 40, 400)
    gpk = 0.01
    for k, mg in enumerate(mgs):
        ax.plot(Vc, -Ipk[k], "o", ms=4, mfc="none", color=f"C{k}")
        ax.plot(Vf, -gpk * mg_block(Vf, mg) * Vf, color=f"C{k}", lw=1.3, label=f"[Mg] = {mg:g} mM")
    ax.axvline(Vmin, color="C2", ls=":", lw=1, label=f"I–V minimum at 1 mM: {Vmin:.2f} mV")
    ax.axhline(0, color="0.6", lw=0.8)
    ax.set_xlabel("clamp V (mV)")
    ax.set_ylabel("peak inward current −I (µA/cm²)")
    ax.set_title("NMDA I–V: clamp simulation (o) vs Jahr–Stevens (—)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # E: NMDA EPSPs vs DOP853
    ax = axes[1, 1]
    for k, (name, (t, V, te, Vr)) in enumerate(R["nmda"].items()):
        ax.plot(t, V, color=f"C{k}", lw=1.3, label=name)
        ax.plot(te[::8], Vr[::8], "k.", ms=3)
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("NMDA EPSPs, [Mg] = 1 mM unless stated: RK4 (—) vs DOP853 (•)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # F: convergence and GABA reversal
    ax = axes[1, 2]
    dts, errs, order = R["order"]
    ax.loglog(dts, errs, "o-", color="C0", label=f"RK4 error vs exact, AMPA train: slope {order:.2f}")
    ax.loglog(dts, errs[-1] * (dts / dts[-1]) ** 4, "k--", lw=0.8, label="dt⁴")
    _, _, _, g0, rel, slope = R["alpha"]
    ax.loglog(g0 * 10, rel, "s-", color="C3",
              label=f"alpha-limit deviation vs g_max×10: slope {slope:.3f}")
    ax.set_xlabel("dt (ms)   /   10 × g_max (mS/cm²)")
    ax.set_ylabel("max error (mV)   /   relative deviation")
    Vh, amps, Vrev = R["gaba"]
    ins = ax.inset_axes([0.08, 0.08, 0.36, 0.3])
    ins.plot(Vh, amps, "o-", color="C4", ms=3)
    ins.axhline(0, color="0.6", lw=0.8)
    ins.axvline(-80, color="C4", ls=":", lw=1)
    ins.set_title(f"GABA-A IPSP vs V_hold; zero at {Vrev:.4f} mV", fontsize=7)
    ins.tick_params(labelsize=6)
    ax.set_title("Convergence, linear limit, GABA-A reversal")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "conductance_synapses.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print("No RNG: the model is deterministic.")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
