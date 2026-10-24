"""Morris-Lecar model: class I vs class II excitability.

Equations (Morris & Lecar 1981; in the reduced two-variable form of Rinzel &
Ermentrout 1998, with m at its steady state):

    C dV/dt = I - g_L (V - E_L) - g_Ca m_inf(V) (V - E_Ca) - g_K w (V - E_K)
    dw/dt   = phi (w_inf(V) - w) / tau_w(V)

    m_inf(V) = 0.5 [1 + tanh((V - V1)/V2)]
    w_inf(V) = 0.5 [1 + tanh((V - V3)/V4)]
    tau_w(V) = 1 / cosh((V - V3)/(2 V4))

V in mV, t in ms, C in uF/cm^2, conductances in mS/cm^2, I in uA/cm^2.
Shared parameters (Rinzel & Ermentrout 1998; Ermentrout & Terman 2010, sec.
3.1): C = 20, g_L = 2, E_L = -60, g_K = 8, E_K = -84, E_Ca = 120, V1 = -1.2,
V2 = 18. The two published sets differ in the Ca and w kinetics:

    "Hopf" set   (class II): g_Ca = 4.4, V3 = 2,  V4 = 30,   phi = 0.04
    "SNLC" set   (class I) : g_Ca = 4.0, V3 = 12, V4 = 17.4, phi = 1/15

Integrator: forward Euler with dt = 0.05 ms. A spike is an upward crossing of
0 mV, with the crossing time interpolated linearly within the step.

Validation (ROADMAP note: class I vs class II from the two parameter sets,
onset frequency near 0 Hz vs finite):

Fixed points lie on w = w_inf(V) at currents I = I_ss(V) = g_L(V - E_L) +
g_Ca m_inf(V)(V - E_Ca) + g_K w_inf(V)(V - E_K). The Jacobian there is

    J = [[(-g_L - g_Ca m_inf'(V)(V - E_Ca) - g_Ca m_inf - g_K w)/C, -g_K (V - E_K)/C],
         [phi cosh((V-V3)/(2V4)) w_inf'(V),                       -phi cosh((V-V3)/(2V4))]]

and det J = 0 exactly where dI_ss/dV = 0. Class I set: I_ss is not monotone,
the stable rest state and a saddle merge at the local maximum of I_ss
(saddle-node on the invariant circle), I_SN = 39.963. Near it the dynamics
reduce to the normal form dx/dt = alpha (I - I_SN) + beta x^2 along the null
eigenvector, with alpha = l.dF/dI and beta = (1/2) l.D^2F[r,r] (r, l right and
left null vectors of J, l.r = 1), so the inter-spike interval diverges as

    T(I) = pi / sqrt(alpha beta (I - I_SN)) + T0 + ...

and the rate starts continuously from 0 Hz. The script fits the simulated
ISIs to this law with I_SN, K = alpha beta and T0 free and asserts that the
fitted I_SN matches the closed form to 0.1% and K matches alpha*beta to 5%.
Class II set: I_ss is monotone (one fixed point) and the rest state loses
stability where trace J = 0 (Andronov-Hopf, I_H = 93.857 here), which the
script also finds by a perturbation sweep (tolerance 1%). Repetitive firing
starts below I_H, at the current I_LPC where the large spiking cycle is born
(found by a downward sweep from the cycle), with a finite frequency; between
I_LPC and I_H rest and spiking coexist. The script asserts f(I_LPC+) > 5 Hz
for class II and f < 2 Hz at 0.01 uA/cm^2 above I_SN for class I. Finally
Euler ISIs at eight currents are compared with solve_ivp (DOP853, rtol = atol
= 1e-10; tolerance 1%) at dt = 0.05 and 0.025 ms.

Second derivatives for beta are taken by central differences of the analytic
Jacobian (step 1e-4 along r); everything else is closed form. scipy is used
only for root finding, the reference solver and the curve fit.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, curve_fit

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Rinzel & Ermentrout 1998)
# ---------------------------------------------------------------------------
C, g_L, E_L, g_K, E_K, E_Ca, V1, V2 = 20.0, 2.0, -60.0, 8.0, -84.0, 120.0, -1.2, 18.0
SETS = {
    "class II (Hopf)": dict(g_Ca=4.4, V3=2.0, V4=30.0, phi=0.04),
    "class I (SNLC)": dict(g_Ca=4.0, V3=12.0, V4=17.4, phi=1.0 / 15.0),
}
P2, P1 = SETS["class II (Hopf)"], SETS["class I (SNLC)"]

dt = 0.05            # Euler step (ms)
V_SPIKE = 0.0        # spike = upward crossing of 0 mV
SEED = 0
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions, fixed points, Jacobian, closed forms
# ---------------------------------------------------------------------------
def m_inf(V):
    return 0.5 * (1.0 + np.tanh((V - V1) / V2))


def w_inf(V, p):
    return 0.5 * (1.0 + np.tanh((V - p["V3"]) / p["V4"]))


def inv_tau_w(V, p):
    return np.cosh((V - p["V3"]) / (2.0 * p["V4"]))


def rhs(V, w, I, p):
    dV = (I - g_L * (V - E_L) - p["g_Ca"] * m_inf(V) * (V - E_Ca) - g_K * w * (V - E_K)) / C
    dw = p["phi"] * (w_inf(V, p) - w) * inv_tau_w(V, p)
    return dV, dw


def I_ss(V, p):
    """Current at which V is a fixed point (w = w_inf(V))."""
    return g_L * (V - E_L) + p["g_Ca"] * m_inf(V) * (V - E_Ca) + g_K * w_inf(V, p) * (V - E_K)


def jacobian(V, w, p):
    u = (V - p["V3"]) / p["V4"]
    dm = 0.5 / V2 / np.cosh((V - V1) / V2) ** 2
    dwi = 0.5 / p["V4"] / np.cosh(u) ** 2
    J11 = (-g_L - p["g_Ca"] * dm * (V - E_Ca) - p["g_Ca"] * m_inf(V) - g_K * w) / C
    J12 = -g_K * (V - E_K) / C
    J21 = p["phi"] * (dwi * np.cosh(u / 2) + (w_inf(V, p) - w) * np.sinh(u / 2) / (2 * p["V4"]))
    J22 = -p["phi"] * np.cosh(u / 2)
    return np.array([[J11, J12], [J21, J22]])


def J_fp(V, p):
    return jacobian(V, w_inf(V, p), p)


def fixed_points(I, p):
    """All roots of I_ss(V) = I on [-90, 60] mV."""
    V = np.linspace(-90, 60, 3001)
    g = I_ss(V, p) - I
    idx = np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0]
    return np.array([brentq(lambda v: I_ss(v, p) - I, V[k], V[k + 1], xtol=1e-13) for k in idx])


def saddle_node(p):
    """Lower knee of I_ss (det J = 0): returns V_SN, I_SN."""
    V_sn = brentq(lambda v: np.linalg.det(J_fp(v, p)), -40, -20, xtol=1e-13)
    return V_sn, I_ss(V_sn, p)


def hopf(p, V_lo=-40.0, V_hi=-10.0):
    """trace J = 0 with det J > 0 on the rest branch: returns V_H, I_H, omega_H."""
    V_h = brentq(lambda v: np.trace(J_fp(v, p)), V_lo, V_hi, xtol=1e-13)
    J = J_fp(V_h, p)
    assert np.linalg.det(J) > 0
    return V_h, I_ss(V_h, p), np.sqrt(np.linalg.det(J))


def normal_form(p):
    """alpha, beta of dx/dt = alpha (I - I_SN) + beta x^2 at the saddle-node."""
    V_sn, I_sn = saddle_node(p)
    x0 = np.array([V_sn, w_inf(V_sn, p)])
    J = J_fp(V_sn, p)
    ev, R = np.linalg.eig(J)
    r = R[:, np.argmin(abs(ev))].real
    evl, L = np.linalg.eig(J.T)
    l = L[:, np.argmin(abs(evl))].real
    l = l / (l @ r)
    alpha = l[0] / C
    h = 1e-4
    xp, xm = x0 + h * r, x0 - h * r
    D2 = (jacobian(*xp, p) @ r - jacobian(*xm, p) @ r) / (2 * h)
    beta = 0.5 * l @ D2
    return alpha, beta, ev


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(I, p, T, V0, w0, dt=dt, record=False):
    """Forward Euler for one or many currents in lock-step.

    Returns (spike_times list, time, V, w); traces only if record=True."""
    I = np.atleast_1d(np.asarray(I, dtype=float))
    V = np.broadcast_to(np.asarray(V0, float), I.shape).copy()
    w = np.broadcast_to(np.asarray(w0, float), I.shape).copy()
    n = int(round(T / dt))
    spikes = [[] for _ in I]
    if record:
        Vt, wt = np.empty((n + 1,) + I.shape), np.empty((n + 1,) + I.shape)
        Vt[0], wt[0] = V, w
    for i in range(n):
        dV, dw = rhs(V, w, I, p)
        Vn = V + dt * dV
        w = w + dt * dw
        for j in np.nonzero((V < V_SPIKE) & (Vn >= V_SPIKE))[0]:
            spikes[j].append((i + (V_SPIKE - V[j]) / (Vn[j] - V[j])) * dt)
        V = Vn
        if record:
            Vt[i + 1], wt[i + 1] = V, w
    if record:
        return spikes, np.arange(n + 1) * dt, Vt, wt
    return spikes, None, V, w


def reference_isi(I, p, V0, w0, n_spikes=8):
    """solve_ivp (DOP853) ISI after transients: mean of last 3 spike intervals."""
    ev = lambda t, y: y[0] - V_SPIKE
    ev.direction = 1
    sol = solve_ivp(lambda t, y: rhs(y[0], y[1], I, p), (0, 12000), [V0, w0],
                    method="DOP853", rtol=1e-10, atol=1e-10, events=ev)
    ts = sol.t_events[0]
    assert len(ts) >= 5, f"reference fired only {len(ts)} spikes at I={I}"
    return np.mean(np.diff(ts[-4:]))


def last_isi(sp, n=3):
    s = np.asarray(sp)
    return np.mean(np.diff(s[-(n + 1):])) if len(s) >= n + 1 else np.nan


# on-cycle starting state for class II (taken from a run at I = 120)
def cycle_state(p, I=120.0):
    _, _, V, w = simulate(I, p, 2000.0, -60.0, w_inf(-60.0, p))
    return float(V[0]), float(w[0])


def perturbation_grows(I_values, p, T=4000.0):
    """Start 1e-3 mV from the rest fixed point; True if the deviation grows."""
    I_values = np.asarray(I_values, float)
    Vs = np.array([fixed_points(I, p)[0] for I in I_values])
    V0 = Vs + 1e-3 * np.sign(rng.standard_normal(I_values.shape))
    I_ = I_values
    V, w = V0.copy(), w_inf(Vs, p)
    n = int(T / dt)
    amp3 = np.zeros_like(V); amp4 = np.zeros_like(V)
    for i in range(n):
        dV, dw = rhs(V, w, I_, p)
        V = V + dt * dV; w = w + dt * dw
        dev = np.abs(V - Vs)
        if n // 2 <= i < 3 * n // 4:
            amp3 = np.maximum(amp3, dev)
        elif i >= 3 * n // 4:
            amp4 = np.maximum(amp4, dev)
    return (amp4 > amp3) | (amp4 > 1.0)


def keeps_firing(I_values, p, V0, w0, T=6000.0):
    """True if a run started on the spiking cycle still spikes in the last 1500 ms."""
    sp, _, _, _ = simulate(I_values, p, T, V0, w0)
    return np.array([len(s) > 0 and s[-1] > T - 1500.0 for s in sp])


def bisect_sweep(test, lo, hi, n_grid=24, n_refine=3):
    """Grid refinement for the first switch of a boolean test between lo and hi."""
    for _ in range(n_refine + 1):
        grid = np.linspace(lo, hi, n_grid)
        res = test(grid)
        ch = np.nonzero(res[1:] != res[:-1])[0]
        assert ch.size >= 1, f"no transition in [{lo}, {hi}]"
        lo, hi = grid[ch[0]], grid[ch[0] + 1]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def fI_curve(I_values, p, V0, w0, T=4000.0):
    sp, _, _, _ = simulate(I_values, p, T, V0, w0)
    out = []
    for s in sp:
        s = np.asarray(s)
        s = s[s > 1000.0]
        out.append(1000.0 / np.mean(np.diff(s)) if len(s) >= 3 else 0.0)
    return np.array(out)


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    R = {}

    # (a) rest state at I = 0 for both sets: simulation vs root of I_ss(V) = 0
    for name, p in SETS.items():
        Vs = fixed_points(0.0, p)[0]
        _, _, V, w = simulate(0.0, p, 3000.0, -40.0, 0.2)
        err = abs(V[0] - Vs)
        print(f"{name}: rest at I=0, root V* = {Vs:.6f} mV, simulated {V[0]:.6f} mV, "
              f"deviation {err:.1e} mV")
        assert err < 1e-4
        R.setdefault("rest", {})[name] = (Vs, V[0], err)

    # (b) class I: saddle-node closed form, normal form, ISI scaling near onset
    V_sn, I_sn = saddle_node(P1)
    alpha, beta, ev = normal_form(P1)
    K_nf = alpha * beta
    n_fp = [len(fixed_points(I, P1)) for I in (I_sn - 1, I_sn + 1)]
    print(f"Class I: saddle-node at V = {V_sn:.4f} mV, I_SN = {I_sn:.5f} uA/cm^2 "
          f"(fixed points: {n_fp[0]} below, {n_fp[1]} above); eigenvalues {ev[0]:.1e}, {ev[1]:.4f}")
    print(f"Class I normal form: alpha = {alpha:.5f}, beta = {beta:.6f}, alpha*beta = {K_nf:.4e} /ms^2")
    dI = np.geomspace(0.01, 0.5, 10)
    T_pred = np.pi / np.sqrt(K_nf * dI)
    V_rest = fixed_points(I_sn - 1.0, P1)[0]
    sp, _, _, _ = simulate(I_sn + dI, P1, float(5 * T_pred.max() + 500), V_rest, w_inf(V_rest, P1))
    T_sim = np.array([last_isi(s) for s in sp])
    law = lambda I, Isn, K, T0: np.pi / np.sqrt(K * np.clip(I - Isn, 1e-12, None)) + T0
    (I_fit, K_fit, T0_fit), _ = curve_fit(law, I_sn + dI, T_sim, p0=(I_sn - 1e-3, K_nf, 20.0))
    rel_Isn = abs(I_fit - I_sn) / I_sn
    rel_K = abs(K_fit - K_nf) / K_nf
    f_min = 1000.0 / T_sim[0]
    for d, ts, tp in zip(dI, T_sim, T_pred):
        print(f"   I - I_SN = {d:.4f}: ISI {ts:9.2f} ms ({1000/ts:6.3f} Hz); pi/sqrt(ab dI) = {tp:9.2f} ms")
    print(f"Class I fit T = pi/sqrt(K (I - I_SN)) + T0: I_SN = {I_fit:.5f} (closed form {I_sn:.5f}, "
          f"{100*rel_Isn:.4f}%), K = {K_fit:.4e} (alpha*beta {K_nf:.4e}, {100*rel_K:.2f}%), T0 = {T0_fit:.2f} ms")
    print(f"Class I rate at I_SN + 0.01: {f_min:.3f} Hz")
    assert rel_Isn < 1e-3, f"I_SN fit off by {100*rel_Isn:.3f}%"
    assert rel_K < 0.05, f"normal-form coefficient off by {100*rel_K:.2f}%"
    assert f_min < 2.0
    R["classI"] = dict(I_sn=I_sn, V_sn=V_sn, K_nf=K_nf, I_fit=I_fit, K_fit=K_fit, T0=T0_fit,
                       rel_Isn=rel_Isn, rel_K=rel_K, dI=dI, T_sim=T_sim, T_pred=T_pred, f_min=f_min)

    # (c) class II: Hopf closed form vs sweep; onset of firing; bistability
    V_h, I_h, om_h = hopf(P2)
    assert len(fixed_points(I_h, P2)) == 1 and np.all(np.diff(I_ss(np.linspace(-90, 60, 3001), P2)) > 0)
    I_h_sw, hw_h = bisect_sweep(lambda g: perturbation_grows(g, P2), 85.0, 105.0)
    rel_h = abs(I_h_sw - I_h) / I_h
    print(f"Class II: I_ss monotone (one fixed point). Hopf at V = {V_h:.4f} mV, I_H = {I_h:.5f}, "
          f"omega_H/2pi = {1000*om_h/(2*np.pi):.2f} Hz")
    print(f"Class II Hopf by perturbation sweep: {I_h_sw:.5f} +/- {hw_h:.1e}; relative error {100*rel_h:.3f}%")
    assert rel_h < 0.01
    Vc, wc = cycle_state(P2)
    I_lpc, hw_lpc = bisect_sweep(lambda g: keeps_firing(g, P2, Vc, wc), 70.0, I_h)
    I_on = I_lpc + 2 * hw_lpc
    sp, _, _, _ = simulate(I_on, P2, 6000.0, Vc, wc)
    f_on = 1000.0 / last_isi(sp[0])
    print(f"Class II lowest firing current (sweep down from the cycle): I_LPC = {I_lpc:.4f} +/- {hw_lpc:.1e}; "
          f"rate just above it {f_on:.2f} Hz")
    assert f_on > 5.0
    I_bi = 0.5 * (I_lpc + I_h)
    Vr = fixed_points(I_bi, P2)[0]
    sp_r, _, _, _ = simulate(I_bi, P2, 4000.0, Vr + 1.0, w_inf(Vr, P2))
    sp_c, _, _, _ = simulate(I_bi, P2, 4000.0, Vc, wc)
    f_bi = 1000.0 / last_isi(sp_c[0])
    print(f"Class II bistability at I = {I_bi:.2f}: from rest {len(sp_r[0])} spikes, "
          f"from the cycle {len(sp_c[0])} spikes ({f_bi:.2f} Hz)")
    assert len(sp_r[0]) == 0 and len(sp_c[0]) > 20
    R["classII"] = dict(I_h=I_h, V_h=V_h, om_h=om_h, I_h_sw=I_h_sw, hw_h=hw_h, rel_h=rel_h,
                        I_lpc=I_lpc, hw_lpc=hw_lpc, f_on=f_on, I_bi=I_bi, f_bi=f_bi, Vc=Vc, wc=wc)

    # (d) Euler vs solve_ivp ISIs, two timesteps
    cases = [("class I (SNLC)", P1, I_sn + d) for d in (0.1, 1.0, 10.0, 40.0)] + \
            [("class II (Hopf)", P2, I) for I in (89.0, 100.0, 150.0, 200.0)]
    rows = []
    for name, p, I in cases:
        if p is P1:
            V0, w0 = V_rest, w_inf(V_rest, P1)
        else:
            V0, w0 = Vc, wc
        T_ref = reference_isi(I, p, V0, w0)
        Tsim = 8 * T_ref + 500
        e = []
        for h in (dt, dt / 2):
            sp, _, _, _ = simulate(I, p, Tsim, V0, w0, dt=h)
            e.append((last_isi(sp[0]) - T_ref) / T_ref)
        rows.append((name, I, T_ref, e[0], e[1]))
        print(f"   {name:16s} I = {I:8.3f}: ISI ref {T_ref:8.3f} ms; Euler error {100*e[0]:+.3f}% (dt=0.05), "
              f"{100*e[1]:+.3f}% (dt=0.025), ratio {e[0]/e[1]:.2f}")
    max_err = max(abs(r[3]) for r in rows)
    print(f"Max |Euler - solve_ivp| ISI error at dt = 0.05 ms: {100*max_err:.3f}%")
    assert max_err < 0.01
    R["ref"] = rows
    return R


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(R):
    c1, c2 = R["classI"], R["classII"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    V_rest1 = fixed_points(c1["I_sn"] - 1.0, P1)[0]

    # Panel 1: class I traces
    a = ax[0, 0]
    for d, col in ((0.2, "tab:blue"), (5.0, "tab:orange"), (40.0, "tab:red")):
        _, t, V, _ = simulate(c1["I_sn"] + d, P1, 1500.0, V_rest1, w_inf(V_rest1, P1), record=True)
        a.plot(t, V[:, 0], color=col, lw=1, label=f"I = I_SN + {d:g}")
    a.set(xlabel="time (ms)", ylabel="V (mV)", title="Class I (SNLC set): firing starts at arbitrarily low rate")
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3)

    # Panel 2: class II traces
    a = ax[0, 1]
    for I, col in ((c2["I_lpc"] + 2 * c2["hw_lpc"], "tab:blue"), (100.0, "tab:orange"), (200.0, "tab:red")):
        _, t, V, _ = simulate(I, P2, 1500.0, c2["Vc"], c2["wc"], record=True)
        a.plot(t, V[:, 0], color=col, lw=1, label=f"I = {I:.2f}")
    a.set(xlabel="time (ms)", ylabel="V (mV)", title="Class II (Hopf set): firing starts at a finite rate")
    a.legend(fontsize=8, loc="lower right"); a.grid(alpha=0.3)

    # Panel 3: f-I curves with closed-form markers
    a = ax[0, 2]
    I1 = np.linspace(30, 90, 61)
    f1 = fI_curve(I1, P1, V_rest1, w_inf(V_rest1, P1))
    a.plot(I1, f1, "o", ms=3, color="tab:green", label="class I, Euler")
    Ix = np.linspace(c1["I_sn"], 50, 200)
    a.plot(Ix, 1000 / (np.pi / np.sqrt(c1["K_nf"] * np.clip(Ix - c1["I_sn"], 1e-9, None)) + c1["T0"]),
           color="tab:green", lw=1, label="1/(π/√(αβ ΔI) + T0)")
    I2 = np.linspace(70, 220, 76)
    f2_up = fI_curve(I2, P2, np.array([fixed_points(I, P2)[0] + 1.0 for I in I2]),
                     np.array([w_inf(fixed_points(I, P2)[0], P2) for I in I2]))
    f2_dn = fI_curve(I2, P2, c2["Vc"], c2["wc"])
    a.plot(I2, f2_dn, "s", ms=3, color="tab:purple", label="class II, Euler, start on cycle")
    a.plot(I2, f2_up, "x", ms=4, color="black", label="class II, Euler, start at rest + 1 mV")
    for r in R["ref"]:
        a.plot(r[1], 1000 / r[2], "D", mfc="none", mec="red", ms=7)
    a.plot([], [], "D", mfc="none", mec="red", label="solve_ivp reference")
    a.axvline(c1["I_sn"], color="tab:green", ls="--", lw=1, label=f"I_SN = {c1['I_sn']:.3f} (closed form)")
    a.axvline(c2["I_h"], color="tab:purple", ls="--", lw=1, label=f"I_H = {c2['I_h']:.3f} (Jacobian)")
    a.axvline(c2["I_h_sw"], color="black", ls=":", lw=1, label=f"I_H sweep = {c2['I_h_sw']:.3f}")
    a.axvline(c2["I_lpc"], color="tab:purple", ls="-.", lw=1, label=f"I_LPC sweep = {c2['I_lpc']:.3f}")
    a.set(xlabel="I (µA/cm²)", ylabel="rate (Hz)", title="f–I: continuous from 0 Hz (I) vs jump (II)")
    a.legend(fontsize=7, loc="upper left"); a.grid(alpha=0.3)

    # Panel 4: class I ISI scaling
    a = ax[1, 0]
    a.loglog(c1["dI"], c1["T_sim"], "o", color="tab:green", label="Euler ISI")
    a.loglog(c1["dI"], c1["T_pred"], "--", color="gray", label="π/√(αβ ΔI), normal form")
    a.loglog(c1["dI"], c1["T_pred"] + c1["T0"], "-", color="black", lw=1,
             label=f"π/√(αβ ΔI) + T0, T0 = {c1['T0']:.1f} ms")
    a.set(xlabel="I − I_SN (µA/cm²)", ylabel="ISI (ms)",
          title=f"Class I: fit K = {c1['K_fit']:.4e} vs αβ = {c1['K_nf']:.4e} ({100*c1['rel_K']:.2f}%)")
    a.legend(fontsize=8); a.grid(alpha=0.3, which="both")

    # Panel 5: class I phase plane near SN
    a = ax[1, 1]
    I = c1["I_sn"] + 0.5
    _, t, V, w = simulate(I, P1, 800.0, V_rest1, w_inf(V_rest1, P1), record=True)
    VV = np.linspace(-70, 50, 600)
    wV = (I - g_L * (VV - E_L) - P1["g_Ca"] * m_inf(VV) * (VV - E_Ca)) / (g_K * (VV - E_K))
    a.plot(VV, wV, color="red", lw=1.3, label="V-nullcline")
    a.plot(VV, w_inf(VV, P1), color="green", lw=1.3, label="w-nullcline")
    a.plot(V[:, 0], w[:, 0], color="black", lw=0.8, label=f"trajectory, I = I_SN + 0.5")
    a.plot(c1["V_sn"], w_inf(c1["V_sn"], P1), "o", mfc="orange", mec="black", ms=8, label="saddle-node point")
    a.set(xlim=(-70, 50), ylim=(-0.05, 0.6), xlabel="V (mV)", ylabel="w", title="Class I phase plane: SNIC")
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)

    # Panel 6: class II bistability
    a = ax[1, 2]
    I = c2["I_bi"]
    _, t, V, w = simulate(I, P2, 1500.0, c2["Vc"], c2["wc"], record=True)
    Vr = fixed_points(I, P2)[0]
    _, t, Vb, wb = simulate(I, P2, 1500.0, Vr + 1.0, w_inf(Vr, P2), record=True)
    wV = (I - g_L * (VV - E_L) - P2["g_Ca"] * m_inf(VV) * (VV - E_Ca)) / (g_K * (VV - E_K))
    a.plot(VV, wV, color="red", lw=1.3, label="V-nullcline")
    a.plot(VV, w_inf(VV, P2), color="green", lw=1.3, label="w-nullcline")
    a.plot(V[:, 0], w[:, 0], color="black", lw=0.8, label="start on cycle → keeps firing")
    a.plot(Vb[:, 0], wb[:, 0], color="tab:blue", lw=1.5, label="start at rest + 1 mV → decays")
    a.plot(Vr, w_inf(Vr, P2), "o", mfc="white", mec="black", ms=8, label="stable rest")
    a.set(xlim=(-70, 50), ylim=(-0.05, 0.6), xlabel="V (mV)", ylabel="w",
          title=f"Class II at I = {I:.2f}: rest and spiking coexist")
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "morris_lecar.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
