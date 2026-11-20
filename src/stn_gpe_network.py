"""Subthalamic nucleus - globus pallidus externus (STN-GPe) network of Terman et al. (2002).

Terman, Rubin, Yew & Wilson (2002), "Activity patterns in a model for the
subthalamopallidal network of the basal ganglia", J. Neurosci. 22, 2963-2976.
Single-compartment conductance-based cells, parameters from Tables 1 (STN)
and 2 (GPe). Units: v in mV, t in ms, g in nS/um^2, currents in pA/um^2,
C_m = 1 pF/um^2.

Both cell types:
    C_m dv/dt = -I_L - I_K - I_Na - I_T - I_Ca - I_AHP - I_syn + I_app
    I_L = g_L (v - v_L)               I_K  = g_K n^4 (v - v_K)
    I_Na = g_Na m_inf(v)^3 h (v - v_Na)
    I_Ca = g_Ca s_inf(v)^2 (v - v_Ca) I_AHP = g_AHP (v - v_K) Ca / (Ca + k1)
    STN: I_T = g_T a_inf(v)^3 b_inf(r)^2 (v - v_Ca),
         b_inf(r) = 1/(1 + exp(-(r - theta_b)/|sigma_b|)) - 1/(1 + exp(theta_b/|sigma_b|))
    GPe: I_T = g_T a_inf(v)^3 r (v - v_Ca)
    dX/dt = phi_X (X_inf(v) - X)/tau_X(v),  X = n, h, r
    X_inf(v) = 1/(1 + exp(-(v - theta_X)/sigma_X))
    tau_X(v) = tau0_X + tau1_X / (1 + exp(-(v - theta_tauX)/sigma_tauX))   (GPe: tau_r = 30 ms)
    dCa/dt = eps (-I_Ca - I_T - k_Ca Ca)
Synapses (presynaptic variable s_j of cell j):
    ds_j/dt = alpha H_inf(v_j - theta_g)(1 - s_j) - beta s_j,
    H_inf(x) = 1/(1 + exp(-(x - theta_gH)/sigma_gH))
    STN: I_G->S = g_GS (v - v_GS) sum_j s_j(GPe)
    GPe: I_S->G = g_SG (v - v_SG) sum_j s_j(STN),  I_G->G = g_GG (v - v_GG) sum_j s_j(GPe)
Sign convention: activation variables have sigma > 0, inactivation sigma < 0
(the tables print magnitudes; the signs used here are the standard ones).

Network: the paper's structured "off-centre" architecture (fig. 5), N = 16
cells of each type on a ring: STN i excites GPe i; GPe j inhibits STN j-2
and j+2 (skipping the three nearest) and GPe j-1, j+1. g_GS = 4.5, g_GG = 0.06,
g_SG = 0.56 (episodic clustering) and 0.72 (continuous clustering), with the
fig. 5 caption values v_GG = -85 mV and GPe synaptic decay beta = 0.04 /ms
(Table 2 gives -100 mV and 0.08). STN has no applied current. Striatal input
is, as in the paper, a constant current I_app common to all GPe cells; the
paper's network value is I_app = -1.0; "raising striatal inhibition" means
making I_app more negative.

Integrator: classical fourth-order Runge-Kutta, dt = 0.0125 ms. Initial
voltages uniform on [-60, -50] mV (gates at steady state, Ca uniform on
[0, 0.1]); RNG seed printed.

Validation (see ROADMAP.md note, and README for the outcome):
(1) Closed-form rest states: with no synapses and a hyperpolarising current
    (STN at I = -20, GPe at the paper's I_app = -1.2, which the paper states
    silences GPe), the fixed point is the root of a scalar equation in v
    (gates at steady state, Ca* = (-I_Ca - I_T)/k_Ca); a run from rest + 1 mV
    must return to it.
(2) Published single-cell numbers: STN fires spontaneously at about 3 Hz
    (paper, fig. 1 and text) and fires a rebound burst after release from
    hyperpolarisation.
(3) Reference solver: RK4 spike times in a 300 ms network run vs DOP853
    (rtol = atol = 1e-10).
(4) The roadmap's regime claim, with criteria fixed before the sweep: the
    STN population rhythm is "beta" if its power-spectrum peak (1-100 Hz)
    lies in 13-30 Hz AND >= 30% of the 1-100 Hz power is in 13-30 Hz. The
    claim to test: beta absent at the paper's I_app = -1.0 and present for
    some I_app <= -1.5. Reported, not asserted (see README Limitations).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq            # closed-form rest state only
from scipy.integrate import solve_ivp        # reference solver only
from scipy.signal import welch               # analysis only

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Terman et al. 2002, Tables 1 and 2)
# ---------------------------------------------------------------------------
STN = dict(gL=2.25, gK=45.0, gNa=37.5, gT=0.5, gCa=0.5, gAHP=9.0,
           vL=-60.0, vK=-80.0, vNa=55.0, vCa=140.0, vGS=-85.0,
           tau_h0=1.0, tau_h1=500.0, tau_n0=1.0, tau_n1=100.0, tau_r0=40.0, tau_r1=17.5,
           phi_h=0.75, phi_n=0.75, phi_r=0.2, k1=15.0, kCa=22.5, eps=3.75e-5,
           th_m=-30.0, s_m=15.0, th_h=-39.0, s_h=-3.1, th_n=-32.0, s_n=8.0,
           th_r=-67.0, s_r=-2.0, th_a=-63.0, s_a=7.8, th_b=0.4, s_b=-0.1, th_s=-39.0, s_s=8.0,
           tth_h=-57.0, ts_h=-3.0, tth_n=-80.0, ts_n=-26.0, tth_r=-68.0, ts_r=-2.2,
           th_gH=-39.0, s_gH=8.0, th_g=30.0, alpha=5.0, beta=1.0)
GPE = dict(gL=0.1, gK=30.0, gNa=120.0, gT=0.5, gCa=0.15, gAHP=30.0,
           vL=-55.0, vK=-80.0, vNa=55.0, vCa=120.0, vGG=-85.0, vSG=0.0,
           tau_h0=0.05, tau_h1=0.27, tau_n0=0.05, tau_n1=0.27, tau_r=30.0,
           phi_h=0.05, phi_n=0.05, phi_r=1.0, k1=30.0, kCa=20.0, eps=1e-4,
           th_m=-37.0, s_m=10.0, th_h=-58.0, s_h=-12.0, th_n=-50.0, s_n=14.0,
           th_r=-70.0, s_r=-2.0, th_a=-57.0, s_a=2.0, th_s=-35.0, s_s=2.0,
           tth_h=-40.0, ts_h=-12.0, tth_n=-40.0, ts_n=-12.0,
           th_gH=-57.0, s_gH=2.0, th_g=20.0, alpha=2.0, beta=0.04)

N = 16                       # cells per population
G_GS, G_GG = 4.5, 0.06       # fig. 5
G_SG_LIST = (0.56, 0.72)     # fig. 5c (episodic) and 5d (continuous clustering)
I_PAPER = -1.0               # fig. 5 striatal input to GPe
I_SWEEP = np.array([1.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0])
dt = 0.0125                  # ms (0.025 ms failed the reference check, see README)
T_NET, T_SKIP = 2500.0, 1000.0
V_SPIKE = -20.0              # mV, upward crossing = spike
SEED = 2002
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


def ring(offsets):
    W = np.zeros((N, N))
    for i in range(N):
        for o in offsets:
            W[i, (i + o) % N] = 1.0
    return W


W_GS = ring([-2, 2])         # STN i <- GPe i-2, i+2
W_SG = np.eye(N)             # GPe i <- STN i
W_GG = ring([-1, 1])         # GPe i <- GPe i-1, i+1
ZERO = np.zeros((N, N))


# ---------------------------------------------------------------------------
# Block 2: Model functions
# ---------------------------------------------------------------------------
def xinf(v, th, s):
    return 1.0 / (1.0 + np.exp(-(v - th) / s))


def taux(v, t0, t1, th, s):
    return t0 + t1 / (1.0 + np.exp(-(v - th) / s))


def b_inf(r):
    p = STN
    return xinf(r, p["th_b"], -p["s_b"]) - 1.0 / (1.0 + np.exp(-p["th_b"] / p["s_b"]))


def ionic(v, n, h, r, ca, p, stn):
    IL = p["gL"] * (v - p["vL"])
    IK = p["gK"] * n ** 4 * (v - p["vK"])
    INa = p["gNa"] * xinf(v, p["th_m"], p["s_m"]) ** 3 * h * (v - p["vNa"])
    avail = b_inf(r) ** 2 if stn else r
    IT = p["gT"] * xinf(v, p["th_a"], p["s_a"]) ** 3 * avail * (v - p["vCa"])
    ICa = p["gCa"] * xinf(v, p["th_s"], p["s_s"]) ** 2 * (v - p["vCa"])
    IAHP = p["gAHP"] * (v - p["vK"]) * ca / (ca + p["k1"])
    return IL + IK + INa + IT + ICa + IAHP, IT + ICa


def gates(v, n, h, r, p, stn):
    dn = p["phi_n"] * (xinf(v, p["th_n"], p["s_n"]) - n) / taux(v, p["tau_n0"], p["tau_n1"], p["tth_n"], p["ts_n"])
    dh = p["phi_h"] * (xinf(v, p["th_h"], p["s_h"]) - h) / taux(v, p["tau_h0"], p["tau_h1"], p["tth_h"], p["ts_h"])
    tr = taux(v, p["tau_r0"], p["tau_r1"], p["tth_r"], p["ts_r"]) if stn else p["tau_r"]
    dr = p["phi_r"] * (xinf(v, p["th_r"], p["s_r"]) - r) / tr
    return dn, dh, dr


def rhs(X, I_S, I_G, g_GS, g_SG, g_GG, W_GS, W_SG, W_GG):
    """X has shape (12, B, N): STN v n h r Ca s, then GPe v n h r Ca s.
    Currents and conductances broadcast against (B, N)."""
    v, n, h, r, ca, s = X[:6]
    u, nn, hh, rr, cc, ss = X[6:]
    p, q = STN, GPE
    I_ion, I_cal = ionic(v, n, h, r, ca, p, True)
    I_syn = g_GS * (v - p["vGS"]) * (ss @ W_GS.T)
    dv = -I_ion - I_syn + I_S
    dn, dh, dr = gates(v, n, h, r, p, True)
    dca = p["eps"] * (-I_cal - p["kCa"] * ca)
    ds = p["alpha"] * xinf(v - p["th_g"], p["th_gH"], p["s_gH"]) * (1 - s) - p["beta"] * s
    J_ion, J_cal = ionic(u, nn, hh, rr, cc, q, False)
    J_syn = g_SG * (u - q["vSG"]) * (s @ W_SG.T) + g_GG * (u - q["vGG"]) * (ss @ W_GG.T)
    du = -J_ion - J_syn + I_G
    dnn, dhh, drr = gates(u, nn, hh, rr, q, False)
    dcc = q["eps"] * (-J_cal - q["kCa"] * cc)
    dss = q["alpha"] * xinf(u - q["th_g"], q["th_gH"], q["s_gH"]) * (1 - ss) - q["beta"] * ss
    return np.array([dv, dn, dh, dr, dca, ds, du, dnn, dhh, drr, dcc, dss])


def steady(v, p, stn):
    """Rest state at voltage v (no synapses): gates at steady state, Ca from dCa/dt = 0."""
    n, h, r = xinf(v, p["th_n"], p["s_n"]), xinf(v, p["th_h"], p["s_h"]), xinf(v, p["th_r"], p["s_r"])
    _, I_cal = ionic(v, n, h, r, 0.0, p, stn)
    ca = -I_cal / p["kCa"]
    I_ion, _ = ionic(v, n, h, r, ca, p, stn)
    return np.array([v, n, h, r, ca, 0.0]), I_ion


def rest_voltage(I, p, stn, lo=-100.0, hi=-50.0):
    return brentq(lambda v: steady(v, p, stn)[1] - I, lo, hi, xtol=1e-13)


def initial_state(B, n):
    X = np.zeros((12, B, n))
    for off, p in ((0, STN), (6, GPE)):
        v = -60.0 + 10.0 * rng.random((B, n))
        X[off] = v
        X[off + 1] = xinf(v, p["th_n"], p["s_n"])
        X[off + 2] = xinf(v, p["th_h"], p["s_h"])
        X[off + 3] = xinf(v, p["th_r"], p["s_r"])
        X[off + 4] = 0.1 * rng.random((B, n))
    return X


# ---------------------------------------------------------------------------
# Block 3: Simulation
# ---------------------------------------------------------------------------
def simulate(X, T, args, record_every=1, h=None):
    """RK4 at step h (default dt). Returns final state, time grid and recorded STN and GPe
    voltages (n_rec, B, N)."""
    h = dt if h is None else h
    n_steps = int(round(T / h))
    n_rec = n_steps // record_every
    VS = np.empty((n_rec,) + X.shape[1:])
    VG = np.empty_like(VS)
    for k in range(n_steps):
        k1 = rhs(X, *args)
        k2 = rhs(X + 0.5 * h * k1, *args)
        k3 = rhs(X + 0.5 * h * k2, *args)
        k4 = rhs(X + h * k3, *args)
        X = X + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (k + 1) % record_every == 0:
            j = (k + 1) // record_every - 1
            VS[j], VG[j] = X[0], X[6]
    t = np.arange(1, n_rec + 1) * h * record_every
    return X, t, VS, VG


def spike_times(t, V):
    """Upward crossings of V_SPIKE; V is 1-D on a uniform grid t. The crossing is
    located on the cubic through the four samples around it (Newton from the linear
    estimate), so the interpolation error is O(h^4) rather than the O(h^2) of a chord."""
    k = np.nonzero((V[:-1] < V_SPIKE) & (V[1:] >= V_SPIKE))[0]
    h = t[1] - t[0]
    x = (V_SPIKE - V[k]) / (V[k + 1] - V[k])          # linear estimate, in units of h from t[k]
    ok = (k >= 1) & (k + 2 < V.size)
    kk = k[ok]
    if kk.size:
        y0, y1, y2, y3 = V[kk - 1], V[kk], V[kk + 1], V[kk + 2]
        # cubic through (-1, y0), (0, y1), (1, y2), (2, y3)
        c0 = y1
        c1 = -y0 / 3 - y1 / 2 + y2 - y3 / 6
        c2 = y0 / 2 - y1 + y2 / 2
        c3 = -y0 / 6 + y1 / 2 - y2 / 2 + y3 / 6
        z = x[ok]
        for _ in range(8):
            z = z - (c0 + z * (c1 + z * (c2 + z * c3)) - V_SPIKE) / (c1 + z * (2 * c2 + 3 * z * c3))
        x[ok] = z
    return t[k] + x * h


def single_args(I_S, I_G):
    return (I_S, I_G, 0.0, 0.0, 0.0, ZERO, ZERO, ZERO)


# ---------------------------------------------------------------------------
# Block 4: Validation
# ---------------------------------------------------------------------------
def validate():
    R = {}
    # One batch holds the striatal-input sweep (rows 0 .. n_c-1, coupled networks) and,
    # in its last row, 16 uncoupled STN and 16 uncoupled GPe cells for the single-cell checks.
    n_c = len(I_SWEEP) * len(G_SG_LIST)
    B = n_c + 1
    I_G = np.zeros((B, N))
    I_G[:n_c] = np.tile(I_SWEEP, len(G_SG_LIST))[:, None]
    I_G_fi = np.linspace(-1.2, 2.0, N)                # GPe col 0: rest check at the paper's -1.2
    I_G[n_c] = I_G_fi
    g_SG = np.zeros((B, 1))
    g_SG[:n_c, 0] = np.repeat(G_SG_LIST, len(I_SWEEP))
    g_GS = np.full((B, 1), G_GS)
    g_GS[n_c] = 0.0
    g_GG = np.full((B, 1), G_GG)
    g_GG[n_c] = 0.0
    I_S = np.zeros((B, N))
    I_S[n_c, 0] = -20.0                               # STN col 0: rest check
    I_S[n_c, 1] = -25.0                               # STN col 1: rebound (released at T_HOLD)
    T_HOLD = 1000.0

    v_stn = rest_voltage(-20.0, STN, True)
    v_gpe = rest_voltage(-1.2, GPE, False)
    X = initial_state(B, N)
    X[:6, n_c, 0] = steady(v_stn, STN, True)[0]
    X[6:, n_c, 0] = steady(v_gpe, GPE, False)[0]
    X[0, n_c, 0] += 1.0
    X[6, n_c, 0] += 1.0

    t0 = _time.time()
    X, t1, VS1, VG1 = simulate(X, T_HOLD, (I_S, I_G, g_GS, g_SG, g_GG, W_GS, W_SG, W_GG), record_every=2)
    I_S[n_c, 1] = 0.0
    X, t2, VS2, VG2 = simulate(X, T_NET - T_HOLD, (I_S, I_G, g_GS, g_SG, g_GG, W_GS, W_SG, W_GG), record_every=2)
    t = np.concatenate([t1, T_HOLD + t2])
    VS = np.concatenate([VS1, VS2])
    VG = np.concatenate([VG1, VG2])
    del VS1, VS2, VG1, VG2
    print(f"Main batch ({n_c} networks + 32 uncoupled cells, {T_NET / 1000:.1f} s, dt = {dt} ms) in {_time.time() - t0:.0f} s")

    # (1) closed-form rest states
    s0 = spike_times(t, VS[:, n_c, 0]).size + spike_times(t, VG[:, n_c, 0]).size
    d_stn, d_gpe = abs(X[0, n_c, 0] - v_stn), abs(X[6, n_c, 0] - v_gpe)
    print(f"Rest (no synapses), from v* + 1 mV: STN at I = -20: v* = {v_stn:.6f} mV, |v({T_NET / 1000:.1f} s) - v*| = {d_stn:.2e} mV; "
          f"GPe at I_app = -1.2: v* = {v_gpe:.6f} mV, |v - v*| = {d_gpe:.2e} mV; spikes {s0}")
    assert d_stn < 1e-2 and d_gpe < 1e-2 and s0 == 0
    R["rest"] = (v_stn, v_gpe, d_stn, d_gpe)

    # (2) STN spontaneous rate (cols 2..15), rebound (col 1), GPe f-I (cols 1..15)
    rates = []
    for i in range(2, N):
        st = spike_times(t, VS[:, n_c, i])
        st = st[st > 1000.0]
        rates.append(1000.0 / np.mean(np.diff(st)))
    rates = np.array(rates)
    f_stn = rates.mean()
    print(f"STN spontaneous rate (I_app = 0, {rates.size} cells from random starts, ISIs after 1 s): "
          f"{f_stn:.3f} Hz (range {rates.min():.3f}-{rates.max():.3f}); paper: about 3 Hz")
    assert abs(f_stn - 3.0) < 1.0
    T_s = (T_NET - T_SKIP) / 1000.0
    g_rates = np.array([np.sum(spike_times(t, VG[:, n_c, i]) > T_SKIP) / T_s for i in range(N)])
    I_on = I_G_fi[np.argmax(g_rates > 0)]
    print("GPe rate vs I_app (no synapses): " + ", ".join(f"{a:+.2f}: {b:.1f}" for a, b in zip(I_G_fi[::3], g_rates[::3]))
          + f" Hz; lowest firing current on the grid {I_on:+.3f}")
    assert g_rates[0] == 0.0
    R["stn_spont"] = (t, VS[:, n_c, 2], rates)
    R["gpe_fi"] = (I_G_fi, g_rates)

    st = spike_times(t, VS[:, n_c, 1])
    st = st[st > T_HOLD] - T_HOLD
    isi = np.diff(st)
    k_end = int(np.argmax(np.concatenate([isi, [np.inf]]) >= 50.0))
    burst = st[:k_end + 1]
    print(f"STN rebound after {T_HOLD:.0f} ms at I = -25: {burst.size} spikes in the burst (ISIs < 50 ms), "
          f"latency {burst[0]:.1f} ms, duration {burst[-1] - burst[0]:.1f} ms, min ISI {isi.min():.2f} ms "
          f"(tonic ISI {1000 / f_stn:.0f} ms); paper: rebound bursts lasting up to about 200 ms")
    assert burst.size >= 3 and isi.min() < 0.1 * 1000 / f_stn
    m = (t > T_HOLD - 100) & (t < T_HOLD + 500)
    R["rebound"] = (t[m] - T_HOLD, VS[m, n_c, 1], burst)

    # (3) striatal-inhibition sweep
    m = t > T_SKIP
    edges = np.arange(T_SKIP, T_NET + 1e-9, 1.0)       # 1 ms bins
    rows = []
    for c in range(n_c):
        st_s = [spike_times(t, VS[:, c, i]) for i in range(N)]
        st_g = [spike_times(t, VG[:, c, i]) for i in range(N)]
        pop = np.histogram(np.concatenate(st_s), edges)[0].astype(float)
        f, P = welch(pop, fs=1000.0, nperseg=750, detrend="constant")
        band = (f >= 1) & (f <= 100)
        if P[band].sum() > 0:
            f_pk = f[band][np.argmax(P[band])]
            frac = P[(f >= 13) & (f <= 30)].sum() / P[band].sum()
        else:
            f_pk, frac = np.nan, 0.0
        Vs = VS[m, c, :]
        chi = np.sqrt(np.var(Vs.mean(1)) / np.mean(np.var(Vs, axis=0)))
        r_s = np.mean([np.sum(a > T_SKIP) for a in st_s]) / T_s
        r_g = np.mean([np.sum(a > T_SKIP) for a in st_g]) / T_s
        beta = bool(13 <= f_pk <= 30 and frac >= 0.3)
        rows.append((I_G[c, 0], g_SG[c, 0], r_s, r_g, f_pk, frac, chi, beta, f, P, st_s, st_g))
        print(f"  g_SG = {g_SG[c, 0]:.2f}, I_app(GPe) = {I_G[c, 0]:+.1f}: STN {r_s:5.2f} Hz, GPe {r_g:5.2f} Hz, "
              f"STN population peak {f_pk:5.2f} Hz, 13-30 Hz fraction {frac:.3f}, chi(STN) {chi:.3f}, beta: {beta}")
    R["sweep"] = rows
    del VS, VG
    absent = [not r[7] for r in rows if r[0] == I_PAPER]
    present = [r[7] for r in rows if r[0] <= -1.5]
    ok = all(absent) and any(present)
    print(f"Roadmap claim (beta absent at I_app = {I_PAPER}, present for some I_app <= -1.5): "
          f"absent at baseline {all(absent)}, present when raised {any(present)} -> "
          f"{'MATCHES' if ok else 'DOES NOT MATCH'} (not asserted)")
    R["claim"] = ok

    # (4) network: RK4 vs DOP853 over 300 ms at the paper's I_app, g_SG = 0.72
    X0 = initial_state(1, N)
    args = (0.0, I_PAPER, G_GS, G_SG_LIST[1], G_GG, W_GS, W_SG, W_GG)
    Tr = 300.0
    fun = lambda tt, y: rhs(y.reshape(12, 1, N), *args).ravel()
    t0 = _time.time()
    sol = solve_ivp(fun, (0, Tr), X0.ravel(), method="DOP853", rtol=1e-10, atol=1e-10, dense_output=True)
    tr = np.arange(0.0, Tr, 0.002)
    Y = sol.sol(tr).reshape(12, 1, N, -1)
    print(f"DOP853 reference (rtol = atol = 1e-10): {sol.nfev} evaluations, {_time.time() - t0:.0f} s")
    errs = {}
    for h in (2 * dt, dt):
        _, tk, VSk, VGk = simulate(X0.copy(), Tr, args, h=h)
        e, n_tot = [], 0
        for i in range(N):
            for Vrk, Vref in ((VSk[:, 0, i], Y[0, 0, i]), (VGk[:, 0, i], Y[6, 0, i])):
                a, b = spike_times(tk, Vrk), spike_times(tr, Vref)
                assert a.size == b.size, f"spike counts differ between RK4 (dt = {h}) and DOP853"
                n_tot += a.size
                e.extend(np.abs(a - b))
        errs[h] = (np.max(e), np.median(e), n_tot)
        print(f"  RK4 dt = {h} ms: {n_tot} spikes in both, spike-time difference max {np.max(e):.4f} ms, "
              f"median {np.median(e):.5f} ms")
    err = errs[dt][0]
    assert err < 0.05, f"max spike-time error {err:.3f} ms"
    R["ref"] = errs
    return R


# ---------------------------------------------------------------------------
# Block 5: Plot
# ---------------------------------------------------------------------------
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    t, V, rates = R["stn_spont"]
    a = ax[0, 0]
    m = t <= 2000
    a.plot(t[m], V[m], color="C0", lw=0.7, label=f"spontaneous: {rates.mean():.2f} Hz (paper: ~3 Hz)")
    tr, Vr, burst = R["rebound"]
    a.plot(tr + 2200, Vr, color="C3", lw=0.7,
           label=f"released from I = -25 at t = 2200 ms: {burst.size}-spike rebound burst")
    a.set_xlabel("time (ms)")
    a.set_ylabel("v (mV)")
    a.set_title("Single STN cell (no synapses)")
    a.legend(fontsize=8, loc="lower left")

    a = ax[0, 1]
    I, gr = R["gpe_fi"]
    a.plot(I, gr, "o-", color="C2", ms=4)
    a.axvline(-1.2, color="k", ls="--", lw=1, label="paper's silencing I_app = -1.2")
    a.axvline(I_PAPER, color="C1", ls=":", lw=1, label=f"fig. 5 network I_app = {I_PAPER}")
    a.set_xlabel("I_app (pA/µm²)")
    a.set_ylabel("rate (Hz)")
    a.set_title(f"Isolated GPe cell; closed-form rest at -1.2: v* = {R['rest'][1]:.3f} mV")
    a.legend(fontsize=8)

    for j, I_show in enumerate((I_PAPER, -3.0)):
        a = ax[1, j]
        r = [r for r in R["sweep"] if r[0] == I_show and r[1] == G_SG_LIST[1]][0]
        for i in range(N):
            a.plot(r[10][i], np.full(r[10][i].size, i), "|", color="C0", ms=4)
            a.plot(r[11][i], np.full(r[11][i].size, i + N + 1), "|", color="C2", ms=4)
        a.set_xlim(T_SKIP, T_NET)
        a.set_yticks([N / 2, 1.5 * N + 1])
        a.set_yticklabels(["STN", "GPe"])
        a.set_xlabel("time (ms)")
        a.set_title(f"Network, g_SG = {G_SG_LIST[1]}, I_app(GPe) = {I_show:+.1f}\n"
                    f"STN {r[2]:.1f} Hz, GPe {r[3]:.1f} Hz, PSD peak {r[4]:.1f} Hz", fontsize=10)

    a = ax[0, 2]
    for k, gs in enumerate(G_SG_LIST):
        rows = [r for r in R["sweep"] if r[1] == gs]
        a.plot([r[0] for r in rows], [r[4] for r in rows], "o-", color=f"C{k}", label=f"peak freq, g_SG = {gs}")
    a.axhspan(13, 30, color="0.9", zorder=0, label="beta 13-30 Hz")
    a.axvline(I_PAPER, color="k", ls=":", lw=1)
    a.set_xlabel("I_app(GPe) (more negative = more striatal inhibition)")
    a.set_ylabel("STN population PSD peak (Hz)")
    a.set_ylim(0, 100)
    a2 = a.twinx()
    for k, gs in enumerate(G_SG_LIST):
        rows = [r for r in R["sweep"] if r[1] == gs]
        a2.plot([r[0] for r in rows], [r[5] for r in rows], "s--", color=f"C{k}", ms=4, alpha=0.6)
    a2.axhline(0.3, color="C3", lw=0.8, ls="--")
    a2.set_ylabel("13-30 Hz power fraction (squares; criterion 0.3)")
    a2.set_ylim(0, 1)
    a.set_title("Beta criterion vs striatal input" + ("" if R["claim"] else ": not met"))
    a.legend(fontsize=7, loc="upper left")

    a = ax[1, 2]
    for I_show, col in ((1.0, "C1"), (I_PAPER, "k"), (-3.0, "C0")):
        r = [r for r in R["sweep"] if r[0] == I_show and r[1] == G_SG_LIST[1]][0]
        a.semilogy(r[8], r[9] + 1e-12, color=col, label=f"I_app = {I_show:+.1f}: peak {r[4]:.0f} Hz")
    a.axvspan(13, 30, color="0.9", zorder=0)
    a.set_xlim(0, 100)
    a.set_xlabel("frequency (Hz)")
    a.set_ylabel("STN population PSD (spikes²/Hz)")
    a.set_title(f"STN population spectrum, g_SG = {G_SG_LIST[1]}")
    a.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "stn_gpe_network.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    plot(validate())
    print(f"Done in {_time.time() - t0:.1f} s")
