"""Multi-compartment Traub-style CA3 pyramidal neuron with active dendrites:
back-propagating action potentials along an unbranched apical dendrite.

Kinetics: the Traub et al. (1991, J. Neurophysiol. 66, 635-650) CA3 channel
set in the reduced form of Pinsky & Rinzel (1994, J. Comput. Neurosci. 1,
39-60, appendix), exactly as in src/pinsky_rinzel.py; voltages in mV
shifted so that the leak reversal is 0, t in ms, currents in uA/cm^2.
Every compartment k carries the same equation with its own densities:

  C_m dV_k/dt = -g_L (V_k - V_L) - I_Na - I_KDR - I_Ca - I_KC - I_KAHP
                + c_k,k-1 (V_k-1 - V_k) + c_k,k+1 (V_k+1 - V_k) + I_inj,k

  I_Na   = g_Na m_inf(V)^2 h (V - V_Na)     I_KDR  = g_KDR n (V - V_K)
  I_Ca   = g_Ca s^2 (V - V_Ca)              I_KAHP = g_KAHP q (V - V_K)
  I_KC   = g_KC c min(Ca/250, 1) (V - V_K)  dCa/dt = -0.13 I_Ca - 0.075 Ca
  dx/dt  = a_x (1 - x) - b_x x  for x = h, n, s, c, q (rates as in
           pinsky_rinzel.py)

with c_k,j = G_kj / A_k (axial conductance between the nodes / membrane
area of compartment k). C_m = 3 uF/cm^2, g_L = 0.1 mS/cm^2 (tau_m = 30 ms),
V_Na = 120, V_Ca = 140, V_K = -15, V_L = 0 mV.

Geometry ("Traub-style": a soma and a chain of cylindrical dendritic
compartments; dimensions chosen here, not taken from Traub et al. 1991):
soma a cylinder 20 um long and 20 um across (lateral area 1257 um^2), one
apical dendrite of diameter d = 3 um and length L = 600 um, sealed at the
far end, cut into N = L/h compartments (default h = 20 um, N = 30) with
centres at x_j = (j - 1/2) h from the soma. Axial resistivity R_a = 100
Ohm cm, so the dendrite has lambda = sqrt(d / (4 R_a g_L)) = 866 um and
L/lambda = 0.69. Dendritic compartments are joined by G = pi d^2/(4 R_a h);
the soma, taken as isopotential, is joined to the first dendritic centre
by 2G (half a segment).

Densities (mS/cm^2): soma g_Na = 30, g_KDR = 15 (Pinsky & Rinzel's somatic
values); dendrite g_Na,d = 0 (passive), 1, 2, 3, 4 or 6 with
g_KDR,d = g_Na,d / 2 (same ratio as the soma; the dendritic Na density is
the knob). g_Ca, g_KC, g_KAHP are zero in the bAP runs, so the dendritic
spike is a Na spike; they are used in check 4 (two-compartment limit).

Stimulus: a somatic current pulse of 3 nA for 0.5 ms from t = 1 ms, from
the rest state, which elicits one somatic action potential (more when a
dendritic spike re-excites the soma, see check 3); the response
is recorded for 25 ms.

Integrator: classical RK4 written out, fixed dt, vectorised over
parameter sets; the injected current is held constant over each step and
the pulse edges lie on the grid. The fastest mode is set by the axial
coupling, about 4 G/(A C_m) = 375 /ms at h = 20 um, so dt must scale as
h^2: dt = 0.005 ms at h = 20 um (RK4 limit about 0.0074 ms), 0.02 ms at
h = 40 um and above, 0.00125 ms at h = 10 um. A somatic spike is an
upward crossing of 20 mV. The bAP amplitude in a compartment is its peak
voltage minus its resting voltage. A constant somatic holding current of
-0.03 nA is on throughout the bAP runs: without it the Traub somatic Na
window current makes this cell fire spontaneously (42 spikes in 3 s in the
N = 1 version), as the isolated Pinsky-Rinzel soma does at I_s = 0; with
it the soma rests near -4.8 mV.

Validation (validate()):

1. Closed form, passive steady state. All active densities zero, a 0.05 nA
   DC current at the soma, 450 ms of RK4 (15 tau_m). The sealed finite
   cable with an isopotential soma gives
       V(x) = V0 cosh((L - x)/lambda) / cosh(L/lambda),
       R_in = 1 / (g_L A_soma + G_inf tanh(L/lambda)),
       G_inf = pi d lambda g_L.
   Asserted: max |V_k - V(x_k)| / V0 < 1e-3, |R_in,sim / R_in - 1| < 1e-3.
2. Closed form, passive bAP. With a passive dendrite the dendritic
   voltage (measured from its rest) is a linear functional of the somatic
   voltage, which is the boundary value at x = 0. In the Laplace domain
       V(x, s) = V_soma(s) cosh((L - x) q) / cosh(L q),
       q = sqrt(1 + s tau_m) / lambda,
   and the membrane-area mean over the dendrite is
       Vbar(s) = V_soma(s) tanh(L q) / (L q).
   The script takes the simulated somatic waveform, multiplies it by
   exp(-sigma t) (sigma = 0.4 /ms, which suppresses the wrap-around of the
   zero-padded FFT by exp(-30)), applies the transfer function at
   s = sigma + i omega, and transforms back. This is compared with every
   dendritic compartment for h = 600, 300, 150, 75, 40, 20, 10 um
   (N = 1 ... 60; N = 1 is a two-compartment soma + dendrite cell).
   Asserted: the max relative error of the bAP amplitude (peak minus rest)
   over all dendritic compartments falls as h^2 (fitted order over h = 40,
   20, 10 um within 2 +- 0.2) and is < 1% at h = 20 um. The max waveform
   error over 0-25 ms, divided by the somatic spike amplitude, is printed
   but not asserted: it peaks within h of the soma at the end of the
   current pulse, where the kink in the somatic waveform is not
   band-limited at the sampling step and the FFT transfer is inexact.
3. Reference solver, active dendrite. The g_Na,d = 3 run at h = 20 um is
   repeated with scipy's DOP853 (rtol = atol = 1e-9) on the same grid.
   Asserted: bAP amplitude in every compartment within 1e-4 relative.
   Also asserted: the bAP amplitude at the far end grows monotonically
   with g_Na,d (active dendrites reduce attenuation).
4. Limiting case, Pinsky-Rinzel. With one dendritic compartment, areas
   (p, 1 - p) = (0.5, 0.5) of a unit total, a coupling conductance g_c = 2.1
   in place of the geometric G, the somatic channels on the soma and
   g_Ca = 10, g_KAHP = 0.8, g_KC = 15 on the dendrite, the same code is the
   Pinsky-Rinzel model already validated in this repository. At
   I_s = 0.75 uA/cm^2 from the paper's initial state, RK4 dt = 0.02 ms,
   the burst period over 2000-4000 ms must equal the value printed by
   src/pinsky_rinzel.py (495.0190 ms, RK4; DOP853 495.0206 ms) within
   1e-6 relative, with the same 2 spikes per burst.
5. Comparison with the two-compartment cell (reported, not asserted):
   the N = 1 run of check 2 is the two-compartment reduction of the same
   soma and dendrite. Its dendritic bAP amplitude is compared with the
   membrane-mean bAP of the converged chain and with the closed form, and,
   for the active dendrites, with the bAP at each distance.

No random numbers are used. scipy is used only for validation (rest-state
root, reference solver).
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
C_m, g_L = 3.0, 0.1                               # uF/cm^2, mS/cm^2
V_Na, V_Ca, V_K, V_L = 120.0, 140.0, -15.0, 0.0   # mV (shifted)
R_a = 100.0                                       # Ohm cm
D_SOMA, L_SOMA = 20.0, 20.0                       # um
D_DEND, L_DEND = 3.0, 600.0                       # um
G_NA_SOMA, G_KDR_SOMA = 30.0, 15.0                # mS/cm^2
G_NA_DEND = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0])  # mS/cm^2, g_KDR,d = g_Na,d/2
H_STD = 20.0                                      # um, default compartment length
H_LIST = np.array([600.0, 300.0, 150.0, 75.0, 40.0, 20.0, 10.0])
H_FIT = (40.0, 20.0, 10.0)                        # asymptotic range for the order fit


def dt_for(h):
    """RK4 step for compartment length h (stability scales as h^2)."""
    return 0.02 if h >= 40.0 else (0.005 if h >= 20.0 else 0.00125)


I_PULSE, T_ON, T_DUR = 3.0, 1.0, 0.5              # nA, ms, ms
I_HOLD = -0.03                                    # nA, somatic holding current
T_BAP = 25.0                                      # ms recorded
I_DC, T_DC = 0.05, 450.0                          # nA, ms (passive steady state)
SIGMA = 0.4                                       # /ms, exponential window for the FFT
V_SPIKE = 20.0                                    # mV
REF_TOL = 1e-9
G_REF = 3.0                                       # g_Na,d of the DOP853 run
# Pinsky-Rinzel limit (values printed by src/pinsky_rinzel.py)
PR_P, PR_GC, PR_IS = 0.5, 2.1, 0.75
PR_Y0 = np.array([-4.6, -4.5, 0.999, 0.001, 0.009, 0.007, 0.01, 0.2])
PR_PERIOD_RK4, PR_PERIOD_DOP = 495.0190, 495.0206  # ms
PR_DT, PR_T_END, PR_T_FROM = 0.02, 4000.0, 2000.0

TOL_SS = 1e-3
TOL_ORDER = 0.2
TOL_BAP = 1e-2
TOL_REF = 1e-4
TOL_PR = 1e-6

LAMBDA = np.sqrt(D_DEND * 1e-4 / (4.0 * R_a * g_L * 1e-3)) * 1e4   # um
TAU_M = C_m / g_L                                                   # ms


# ------------------------------------------------------------------- model
def vtrap(x):
    """x / (1 - exp(-x)), continuous at x = 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-7
    return np.where(small, 1.0 + 0.5 * x, x / np.where(small, 1.0, -np.expm1(-x)))


def rates(V, Ca):
    a_m = 1.28 * vtrap((V - 13.1) / 4.0)
    b_m = 1.4 * vtrap((40.1 - V) / 5.0)
    a_h = 0.128 * np.exp((17.0 - V) / 18.0)
    b_h = 4.0 / (1.0 + np.exp((40.0 - V) / 5.0))
    a_n = 0.08 * vtrap((V - 35.1) / 5.0)
    b_n = 0.25 * np.exp(0.5 - 0.025 * V)
    a_s = 1.6 / (1.0 + np.exp(-0.072 * (V - 65.0)))
    b_s = 0.1 * vtrap((51.1 - V) / 5.0)
    e = np.exp((6.5 - V) / 27.0)
    low = V <= 50.0
    a_c = np.where(low, np.exp((V - 10.0) / 11.0 - (V - 6.5) / 27.0) / 18.975, 2.0 * e)
    b_c = np.where(low, 2.0 * e - a_c, 0.0)
    a_q = np.minimum(2e-5 * Ca, 0.01)
    return a_m, b_m, a_h, b_h, a_n, b_n, a_s, b_s, a_c, b_c, a_q, 0.001


def geometry(h=H_STD):
    """Soma + N dendritic compartments. Returns areas (cm^2), per-area
    coupling to the left/right neighbour (mS/cm^2) and node positions (um)."""
    N = int(round(L_DEND / h))
    A_s = np.pi * D_SOMA * L_SOMA * 1e-8
    A_d = np.pi * D_DEND * h * 1e-8
    G = 1e3 * np.pi * (D_DEND * 1e-4) ** 2 / 4.0 / (R_a * h * 1e-4)   # mS
    A = np.r_[A_s, np.full(N, A_d)]
    Gc = np.r_[2.0 * G, np.full(N - 1, G)]
    return dict(A=A, cR=np.r_[Gc / A[:-1], 0.0], cL=np.r_[0.0, Gc / A[1:]],
                x=np.r_[0.0, (np.arange(N) + 0.5) * h], h=h, N=N)


def pr_geometry():
    """Pinsky-Rinzel: unit total area split (p, 1 - p), coupling g_c."""
    A = np.array([PR_P, 1.0 - PR_P])
    return dict(A=A, cR=np.array([PR_GC / A[0], 0.0]), cL=np.array([0.0, PR_GC / A[1]]),
                x=np.array([0.0, 1.0]), h=np.nan, N=1)


def densities(geo, g_na_d, g_ca_d=0.0, g_kahp_d=0.0, g_kc_d=0.0):
    """Arrays of shape (N + 1, B) for a batch of dendritic Na densities."""
    g_na_d = np.atleast_1d(np.asarray(g_na_d, dtype=float))
    n, B = geo["N"] + 1, len(g_na_d)
    g = {k: np.zeros((n, B)) for k in ("Na", "KDR", "Ca", "KAHP", "KC")}
    g["Na"][0], g["KDR"][0] = G_NA_SOMA, G_KDR_SOMA
    g["Na"][1:], g["KDR"][1:] = g_na_d, g_na_d / 2.0
    g["Ca"][1:], g["KAHP"][1:], g["KC"][1:] = g_ca_d, g_kahp_d, g_kc_d
    return g


def rhs(y, geo, g, inj):
    """y = (V, h, n, s, c, q, Ca), each of shape (N + 1, B)."""
    V, h, n, s, c, q, Ca = y
    a_m, b_m, a_h, b_h, a_n, b_n, a_s, b_s, a_c, b_c, a_q, b_q = rates(V, Ca)
    m = a_m / (a_m + b_m)
    I_Ca = g["Ca"] * s * s * (V - V_Ca)
    I_ion = (g_L * (V - V_L) + g["Na"] * m * m * h * (V - V_Na) + g["KDR"] * n * (V - V_K)
             + I_Ca + g["KC"] * c * np.minimum(Ca / 250.0, 1.0) * (V - V_K)
             + g["KAHP"] * q * (V - V_K))
    ax = np.zeros_like(V)
    ax[:-1] += geo["cR"][:-1, None] * (V[1:] - V[:-1])
    ax[1:] += geo["cL"][1:, None] * (V[:-1] - V[1:])
    return np.array(((-I_ion + ax + inj) / C_m,
                     a_h * (1 - h) - b_h * h, a_n * (1 - n) - b_n * n,
                     a_s * (1 - s) - b_s * s, a_c * (1 - c) - b_c * c,
                     a_q * (1 - q) - b_q * q, -0.13 * I_Ca - 0.075 * Ca))


def steady_gates(V):
    a_m, b_m, a_h, b_h, a_n, b_n, a_s, b_s = rates(V, 0.0)[:8]
    return a_h / (a_h + b_h), a_n / (a_n + b_n), a_s / (a_s + b_s)


def rest_state(geo, g, inj):
    """Rest of every column under the constant injection `inj`: root of
    dV/dt = 0 with gates at steady state."""
    n, B = geo["N"] + 1, g["Na"].shape[1]
    y = np.zeros((7, n, B))
    for b in range(B):
        gb = {k: v[:, b:b + 1] for k, v in g.items()}

        def full(V):
            V = V[:, None]
            hh, nn, ss = steady_gates(V)
            Ca = -0.13 * gb["Ca"] * ss * ss * (V - V_Ca) / 0.075
            r = rates(V, Ca)
            cc, qq = r[8] / (r[8] + r[9]), r[10] / (r[10] + r[11])
            return np.array([V, hh, nn, ss, cc, qq * np.ones_like(V), Ca])

        root, info, ier, msg = fsolve(lambda V: rhs(full(V), geo, gb, inj)[0, :, 0],
                                      np.full(n, -3.0), xtol=1e-13, full_output=True)
        y[:, :, b] = full(root)[:, :, 0]
        assert ier == 1 and np.abs(info["fvec"]).max() < 1e-10, msg
    return y


def pulse(geo, amp_nA, t_on=T_ON, t_dur=T_DUR, hold=I_HOLD):
    """Somatic current density (uA/cm^2) as a function of the step start
    time: the holding current plus the pulse."""
    to_dens = 1e-3 / geo["A"][0]

    def f(t):
        inj = np.zeros((geo["N"] + 1, 1))
        inj[0] = hold * to_dens
        if t_on - 1e-9 <= t < t_on + t_dur - 1e-9:
            inj[0] += amp_nA * to_dens
        return inj
    return f


def simulate(geo, g, y0, t_end, dt, inj_fn, every=1):
    """RK4 for a batch. Returns sample times, V (n_t, N + 1, B) every
    `every` steps, somatic spike times per column and the final state."""
    y = np.array(y0, dtype=float)
    B = y.shape[2]
    n_steps = int(round(t_end / dt))
    ts, Vs, sp = [0.0], [y[0].copy()], [[] for _ in range(B)]
    for k in range(n_steps):
        t = k * dt
        inj = inj_fn(t)
        k1 = rhs(y, geo, g, inj)
        k2 = rhs(y + 0.5 * dt * k1, geo, g, inj)
        k3 = rhs(y + 0.5 * dt * k2, geo, g, inj)
        k4 = rhs(y + dt * k3, geo, g, inj)
        yn = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        hit = (y[0, 0] < V_SPIKE) & (yn[0, 0] >= V_SPIKE)
        for j in np.flatnonzero(hit):
            sp[j].append(t + dt * (V_SPIKE - y[0, 0, j]) / (yn[0, 0, j] - y[0, 0, j]))
        y = yn
        if (k + 1) % every == 0:
            ts.append((k + 1) * dt)
            Vs.append(y[0].copy())
    return dict(t=np.array(ts), V=np.array(Vs), spikes=[np.array(s) for s in sp], state=y)


def bap_run(h, g_na_d):
    geo = geometry(h)
    g = densities(geo, g_na_d)
    f_inj = pulse(geo, I_PULSE)
    y0 = rest_state(geo, g, f_inj(-1.0))
    dt = dt_for(h)
    every = int(round(0.005 / dt)) if dt < 0.005 else 1
    out = simulate(geo, g, y0, T_BAP, dt, f_inj, every=every)
    out.update(geo=geo, g=g, y0=y0, dt=dt, rest=y0[0])
    return out


# --------------------------------------------------------------- analysis
def cable_transfer(t, dVs, x, mean=False):
    """Dendritic response of a sealed passive cable (L_DEND, LAMBDA, TAU_M)
    to the somatic deviation dVs(t) (uniform samples, dVs(0) = 0), via the
    exponentially windowed FFT. x: positions in um (ignored if mean)."""
    n = len(t)
    dt = t[1] - t[0]
    P = 1 << int(np.ceil(np.log2(4 * n)))
    w = np.exp(-SIGMA * t)
    W = np.fft.rfft(dVs * w, P)
    s = SIGMA + 2j * np.pi * np.fft.rfftfreq(P, dt)
    q = np.sqrt(1.0 + s * TAU_M) / LAMBDA
    e2L = np.exp(-2.0 * L_DEND * q)
    if mean:
        H = ((1.0 - e2L) / (1.0 + e2L) / (L_DEND * q))[None, :]
    else:
        x = np.atleast_1d(x)[:, None]
        H = (np.exp(-x * q) + np.exp(-(2.0 * L_DEND - x) * q)) / (1.0 + e2L)
    return np.fft.irfft(W[None, :] * H, P, axis=1)[:, :n] / w


def describe(spikes, t_from):
    """Burst period and spikes per burst (same rule as pinsky_rinzel.py)."""
    s = spikes[spikes >= t_from]
    isi = np.diff(s)
    ratio = isi.max() / isi.min()
    if ratio <= 3.0:
        return dict(kind="single", period=isi.mean(), per_burst=1.0)
    cut = np.sqrt(isi.max() * isi.min())
    starts = np.concatenate(([0], np.flatnonzero(isi > cut) + 1))
    sizes = np.diff(np.concatenate((starts, [len(s)])))
    full = sizes[1:-1] if len(sizes) > 2 else sizes
    return dict(kind="burst" if full.mean() >= 1.5 else "mixed",
                period=np.diff(s[starts]).mean(), per_burst=full.mean())


# --------------------------------------------------------------- validation
def validate():
    R = {}
    print(f"    dendrite: d = {D_DEND} um, L = {L_DEND} um, lambda = {LAMBDA:.2f} um, "
          f"L/lambda = {L_DEND / LAMBDA:.4f}, tau_m = {TAU_M:g} ms")

    # 1. passive steady state
    t0 = _time.time()
    geo = geometry(H_STD)
    g = densities(geo, 0.0)
    for k in ("Na", "KDR"):
        g[k][:] = 0.0
    y0 = np.zeros((7, geo["N"] + 1, 1))
    dens = I_DC * 1e-3 / geo["A"][0]
    inj = np.zeros((geo["N"] + 1, 1))
    inj[0] = dens
    ss = simulate(geo, g, y0, T_DC, dt_for(H_STD), lambda t: inj, every=1000)
    Vss = ss["state"][0, :, 0]
    G_inf = np.pi * D_DEND * 1e-4 * LAMBDA * 1e-4 * g_L               # mS
    R_in = 1.0 / (g_L * geo["A"][0] + G_inf * np.tanh(L_DEND / LAMBDA))  # 1/mS = kOhm
    V0 = I_DC * R_in * 1e-3                                            # mV (nA * kOhm = uV)
    prof = V0 * np.cosh((L_DEND - geo["x"]) / LAMBDA) / np.cosh(L_DEND / LAMBDA)
    e_prof = np.abs(Vss - prof).max() / V0
    e_rin = Vss[0] / V0 - 1.0
    drift = np.abs(ss["V"][-1, :, 0] - ss["V"][-2, :, 0]).max()
    print(f"[1] passive steady state, {I_DC} nA DC, {T_DC:g} ms ({_time.time() - t0:.1f} s): "
          f"V_soma {Vss[0]:.6f} vs {V0:.6f} mV; R_in {Vss[0] / I_DC:.4f} vs "
          f"{V0 / I_DC:.4f} MOhm ({e_rin:+.2e}, tolerance {TOL_SS:g}); V(L) / V(0) = "
          f"{Vss[-1] / Vss[0]:.6f} (closed form at x = L - h/2: {prof[-1] / prof[0]:.6f}); max "
          f"|V - cosh profile| / V0 = {e_prof:.2e} (tolerance {TOL_SS:g}); change over the "
          f"last 5 ms {drift:.1e} mV")
    assert e_prof < TOL_SS and abs(e_rin) < TOL_SS
    R["ss"] = dict(x=geo["x"], V=Vss, prof=prof, V0=V0, e_prof=e_prof, e_rin=e_rin, R_in=V0 / I_DC)

    # 2. passive bAP vs closed-form transfer function, convergence in h
    R["conv"] = []
    for h in H_LIST:
        t0 = _time.time()
        out = bap_run(h, 0.0)
        t, V, rest = out["t"], out["V"][:, :, 0], out["rest"][:, 0]
        dV = V - rest
        amp_s = dV[:, 0].max()
        pred = cable_transfer(t, dV[:, 0], out["geo"]["x"][1:])
        pk, pk_pred = dV[:, 1:].max(axis=0), pred.max(axis=1)
        err = np.abs(pk / pk_pred - 1.0).max()
        err_wave = np.abs(dV[:, 1:] - pred.T).max() / amp_s
        mean_sim = dV[:, 1:].mean(axis=1)
        mean_pred = cable_transfer(t, dV[:, 0], None, mean=True)[0]
        err_mean = mean_sim.max() / mean_pred.max() - 1.0
        n_sp = len(out["spikes"][0])
        print(f"[2] h = {h:g} um (N = {out['geo']['N']}, dt = {out['dt']} ms, "
              f"{_time.time() - t0:.1f} s): {n_sp} somatic spike, amplitude {amp_s:.3f} mV; "
              f"far-end bAP {pk[-1]:.3f} vs closed form {pk_pred[-1]:.3f} mV; max relative "
              f"error of the bAP amplitude over {len(pk)} compartments {err:.3e}; max "
              f"|waveform error| / soma amplitude {err_wave:.3e}; membrane-mean bAP "
              f"{mean_sim.max():.3f} vs closed form {mean_pred.max():.3f} mV ({err_mean:+.2e})")
        assert n_sp == 1
        R["conv"].append(dict(h=h, N=out["geo"]["N"], err=err, err_wave=err_wave,
                              err_mean=err_mean, amp_s=amp_s, t=t, dV=dV, pk_pred=pk_pred,
                              pred=pred, x=out["geo"]["x"], mean_sim=mean_sim,
                              mean_pred=mean_pred, peaks=dV.max(axis=0)))
    hh = np.array([c["h"] for c in R["conv"]])
    ee = np.array([c["err"] for c in R["conv"]])
    sel = np.isin(hh, H_FIT)
    order = np.polyfit(np.log(hh[sel]), np.log(ee[sel]), 1)[0]
    e_std = ee[hh == H_STD][0]
    ew = np.array([c["err_wave"] for c in R["conv"]])
    order_w = np.polyfit(np.log(hh[sel]), np.log(ew[sel]), 1)[0]
    print(f"    bAP amplitude error: fitted order over h = {', '.join(f'{x:g}' for x in H_FIT)} "
          f"um {order:.3f} (tolerance 2 +/- {TOL_ORDER}); at h = {H_STD:g} um {e_std:.2e} "
          f"(tolerance {TOL_BAP:g}). Waveform error (not asserted): order {order_w:.3f}")
    assert abs(order - 2.0) < TOL_ORDER and e_std < TOL_BAP
    R["order"], R["order_w"] = order, order_w

    # 3. active dendrites, sweep and reference solver
    t0 = _time.time()
    act = bap_run(H_STD, G_NA_DEND)
    print(f"[3] active sweep g_Na,d = {', '.join(f'{x:g}' for x in G_NA_DEND)} mS/cm^2 "
          f"({_time.time() - t0:.1f} s)")
    peaks = act["V"].max(axis=0) - act["rest"]                    # (N + 1, B)
    for b, gd in enumerate(G_NA_DEND):
        i_mid = np.argmin(np.abs(act["geo"]["x"] - L_DEND / 2))
        print(f"    g_Na,d = {gd:g}: rest {act['rest'][0, b]:.3f} (soma) / "
              f"{act['rest'][-1, b]:.3f} mV (far end); bAP {peaks[0, b]:.2f} (soma), "
              f"{peaks[i_mid, b]:.2f} (x = {act['geo']['x'][i_mid]:g} um), "
              f"{peaks[-1, b]:.2f} mV (x = {act['geo']['x'][-1]:g} um); far/soma "
              f"{peaks[-1, b] / peaks[0, b]:.3f}; somatic spikes {len(act['spikes'][b])}")
    assert np.all(np.diff(peaks[-1]) > 0)
    j = list(G_NA_DEND).index(G_REF)
    geo, gref = act["geo"], {k: v[:, j:j + 1] for k, v in act["g"].items()}
    n = geo["N"] + 1
    f_inj = pulse(geo, I_PULSE)
    t0 = _time.time()
    y = act["y0"][:, :, j].ravel()
    segs = [(0.0, T_ON), (T_ON, T_ON + T_DUR), (T_ON + T_DUR, T_BAP)]
    Vref = []
    for a, b in segs:
        inj = f_inj(a + 1e-6)
        tt = act["t"][(act["t"] >= a - 1e-9) & (act["t"] <= b + 1e-9)]
        sol = solve_ivp(lambda t, z: rhs(z.reshape(7, n, 1), geo, gref, inj).ravel(),
                        (a, b), y, method="DOP853", rtol=REF_TOL, atol=REF_TOL, t_eval=tt)
        Vref.append(sol.y.reshape(7, n, -1)[0].T if not Vref else
                    sol.y.reshape(7, n, -1)[0].T[1:])
        y = sol.y[:, -1]
    Vref = np.concatenate(Vref)
    pk_ref = Vref.max(axis=0) - act["rest"][:, j]
    e_ref = np.abs(peaks[:, j] / pk_ref - 1.0).max()
    e_tr = np.abs(Vref - act["V"][:, :, j]).max()
    print(f"    DOP853 (rtol = atol = {REF_TOL:g}, {_time.time() - t0:.1f} s) at g_Na,d = "
          f"{G_REF:g}: max relative difference in bAP amplitude over {n} compartments "
          f"{e_ref:.2e} (tolerance {TOL_REF:g}); max |V_RK4 - V_DOP853| {e_tr:.2e} mV")
    assert e_ref < TOL_REF
    R["act"] = dict(t=act["t"], V=act["V"], rest=act["rest"], peaks=peaks, x=geo["x"],
                    Vref=Vref, j=j, e_ref=e_ref, e_tr=e_tr)

    # 4. two-compartment limit = Pinsky-Rinzel
    t0 = _time.time()
    geo = pr_geometry()
    g = densities(geo, 0.0, g_ca_d=10.0, g_kahp_d=0.8, g_kc_d=15.0)
    y0 = np.zeros((7, 2, 1))
    Vs0, Vd0, h0, n0, s0, c0, q0, Ca0 = PR_Y0
    y0[:, 0, 0] = (Vs0, h0, n0, s0, c0, q0, Ca0)
    y0[:, 1, 0] = (Vd0, h0, n0, s0, c0, q0, Ca0)
    inj = np.array([[PR_IS / PR_P], [0.0]])
    pr = simulate(geo, g, y0, PR_T_END, PR_DT, lambda t: inj, every=5)
    d = describe(pr["spikes"][0], PR_T_FROM)
    e_pr = d["period"] / PR_PERIOD_RK4 - 1.0
    print(f"[4] Pinsky-Rinzel limit (N = 1, p = {PR_P}, g_c = {PR_GC}, I_s = {PR_IS}; "
          f"{_time.time() - t0:.1f} s): {d['kind']}, {d['per_burst']:.2f} spikes per burst, "
          f"period {d['period']:.4f} ms vs {PR_PERIOD_RK4} (pinsky_rinzel.py RK4, "
          f"{e_pr:+.1e}, tolerance {TOL_PR:g}) and {PR_PERIOD_DOP} (DOP853, "
          f"{d['period'] / PR_PERIOD_DOP - 1:+.1e}); dendritic peak "
          f"{pr['V'][:, 1, 0].max():.1f} mV")
    assert d["kind"] == "burst" and d["per_burst"] == 2.0 and abs(e_pr) < TOL_PR
    R["pr"] = dict(t=pr["t"], V=pr["V"][:, :, 0], d=d, e=e_pr)

    # 5. comparison with the two-compartment reduction of the same cell
    two, ref = R["conv"][0], R["conv"][-1]
    print(f"[5] two-compartment reduction (h = {two['h']:g} um) vs chain (h = {ref['h']:g} "
          f"um), passive dendrite: somatic spike {two['amp_s']:.2f} vs {ref['amp_s']:.2f} mV; "
          f"dendritic bAP {two['peaks'][1]:.2f} mV (node at {two['x'][1]:g} um) vs chain at "
          f"that distance {np.interp(two['x'][1], ref['x'], ref['peaks']):.2f} mV, chain "
          f"membrane mean {ref['mean_sim'].max():.2f} mV, far end {ref['peaks'][-1]:.2f} mV; "
          f"closed-form mean for the two-compartment soma waveform "
          f"{two['mean_pred'].max():.2f} mV")
    t0 = _time.time()
    two_act = bap_run(L_DEND, G_NA_DEND)
    R["two_act"] = two_act["V"].max(axis=0) - two_act["rest"]
    mid = [np.interp(300.0, R["act"]["x"], R["act"]["peaks"][:, b])
           for b in range(len(G_NA_DEND))]
    print(f"    active dendrites, two-compartment ({_time.time() - t0:.1f} s): dendritic bAP "
          f"{', '.join(f'{x:.2f}' for x in R['two_act'][1])} mV for g_Na,d = "
          f"{', '.join(f'{x:g}' for x in G_NA_DEND)}; chain at x = 300 um "
          f"{', '.join(f'{x:.2f}' for x in mid)} mV")
    return R


# ----------------------------------------------------------------- plotting
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    cmap = plt.get_cmap("viridis")

    a = ax[0, 0]
    c = [r for r in R["conv"] if r["h"] == H_STD][0]
    for i, xx in enumerate((0.0, 100.0, 200.0, 300.0, 450.0, 590.0)):
        k = np.argmin(np.abs(c["x"] - xx))
        col = cmap(i / 6)
        a.plot(c["t"], c["dV"][:, k], color=col, lw=1.2, label=f"x = {c['x'][k]:g} um")
        if k > 0:
            a.plot(c["t"][::20], c["pred"][k - 1, ::20], "o", color=col, ms=2.5, mfc="none")
    a.plot([], [], "ko", ms=3, mfc="none", label="cable transfer of soma V (closed form)")
    a.set_xlim(0, 15)
    a.set_xlabel("time (ms)")
    a.set_ylabel("V − rest (mV)")
    a.set_title("Passive dendrite: bAP vs closed-form cable transfer", fontsize=10)
    a.legend(fontsize=7)

    a = ax[0, 1]
    A = R["act"]
    j = A["j"]
    for i, xx in enumerate((0.0, 100.0, 200.0, 300.0, 450.0, 590.0)):
        k = np.argmin(np.abs(A["x"] - xx))
        col = cmap(i / 6)
        a.plot(A["t"], A["V"][:, k, j] - A["rest"][k, j], color=col, lw=1.2,
               label=f"x = {A['x'][k]:g} um")
        a.plot(A["t"][::20], A["Vref"][::20, k] - A["rest"][k, j], "o", color=col, ms=2.5,
               mfc="none")
    a.plot([], [], "ko", ms=3, mfc="none", label="DOP853")
    a.set_xlim(0, 15)
    a.set_xlabel("time (ms)")
    a.set_ylabel("V − rest (mV)")
    a.set_title(f"Active dendrite (g_Na,d = {G_REF:g} mS/cm²): RK4 vs DOP853", fontsize=10)
    a.legend(fontsize=7)

    a = ax[0, 2]
    for b, gd in enumerate(G_NA_DEND):
        a.plot(A["x"], A["peaks"][:, b], "-", color=plt.get_cmap("plasma")(b / len(G_NA_DEND)),
               lw=1.5, label=f"g_Na,d = {gd:g}")
        a.plot(L_DEND / 2, R["two_act"][1, b], "s", ms=6, mfc="none",
               color=plt.get_cmap("plasma")(b / len(G_NA_DEND)))
    a.plot([], [], "ks", mfc="none", label="two-compartment (N = 1) node")
    a.set_xlabel("distance from soma (um)")
    a.set_ylabel("bAP amplitude (mV above rest)")
    a.set_title("bAP attenuation vs dendritic Na density (h = 20 um)", fontsize=10)
    a.legend(fontsize=7)

    a = ax[1, 0]
    hh = np.array([r["h"] for r in R["conv"]])
    ee = np.array([r["err"] for r in R["conv"]])
    a.loglog(hh, ee, "o-", color="k",
             label=f"bAP amplitude, max relative error (order {R['order']:.2f}, asserted)")
    ew = np.array([r["err_wave"] for r in R["conv"]])
    a.loglog(hh, ew, "s--", color="grey", mfc="none",
             label=f"waveform, max error / soma bAP (order {R['order_w']:.2f})")
    hx = np.array([8.0, 700.0])
    a.loglog(hx, ee[-1] * (hx / hh[-1]) ** 2, "k:", lw=1, label="∝ h²")
    for r in R["conv"]:
        a.annotate(f"N={r['N']}", (r["h"], r["err"]), fontsize=7, xytext=(4, -10),
                   textcoords="offset points")
    a.set_xlabel("compartment length h (um)")
    a.set_ylabel("error vs continuous cable")
    a.set_title("Passive bAP: convergence to the continuous cable", fontsize=10)
    a.legend(fontsize=7)

    a = ax[1, 1]
    for i, r in enumerate(R["conv"]):
        col = plt.get_cmap("cividis")(i / len(R["conv"]))
        a.plot(r["t"], r["mean_sim"], color=col, lw=1.2,
               label=f"N = {r['N']}" + (" (two-compartment)" if r["N"] == 1 else ""))
        a.plot(r["t"][::25], r["mean_pred"][::25], "o", ms=2, color=col, mfc="none")
    a.plot([], [], "ko", ms=3, mfc="none", label="closed form tanh(Lq)/(Lq)")
    a.set_xlim(0, 20)
    a.set_xlabel("time (ms)")
    a.set_ylabel("membrane-mean dendritic V − rest (mV)")
    a.set_title("Passive dendrite mean: two-compartment vs chain", fontsize=10)
    a.legend(fontsize=7)

    a = ax[1, 2]
    P = R["pr"]
    sel = (P["t"] >= PR_T_FROM) & (P["t"] <= PR_T_FROM + 1200.0)
    a.plot(P["t"][sel] - PR_T_FROM, P["V"][sel, 0], "k", lw=0.8, label="soma")
    a.plot(P["t"][sel] - PR_T_FROM, P["V"][sel, 1], color="tab:orange", lw=0.8,
           label="dendrite")
    a.set_xlabel(f"time (ms, from t = {PR_T_FROM:g})")
    a.set_ylabel("V (mV)")
    a.set_title(f"Same code, N = 1, Pinsky–Rinzel parameters:\nburst period "
                f"{P['d']['period']:.4f} ms vs {PR_PERIOD_RK4} ms ({P['e']:+.0e})", fontsize=10)
    a.legend(fontsize=7)

    fig.suptitle("Traub-style multi-compartment CA3 cell: back-propagating action potential "
                 "along a 600 um dendrite (RK4; checks: cable closed forms, DOP853, "
                 "Pinsky–Rinzel limit)", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "traub_multicompartment.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
