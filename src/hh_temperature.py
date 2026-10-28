"""Hodgkin-Huxley with temperature dependence (Q10 = 3).

Equations (Hodgkin & Huxley 1952, modern convention with rest at -65 mV, the
same parameters as src/hodgkin_huxley.py), with every gating rate multiplied
by the temperature factor of the 1952 paper (eq. 25 and the text before it):

    C_m dV/dt = I - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L)
    dx/dt     = phi(T) [alpha_x(V) (1 - x) - beta_x(V) x],   x = m, h, n
    phi(T)    = Q10^((T - 6.3)/10),  Q10 = 3

    alpha_m = 0.1 (V + 40) / (1 - exp(-(V + 40)/10)),  beta_m = 4 exp(-(V + 65)/18)
    alpha_h = 0.07 exp(-(V + 65)/20),                  beta_h = 1 / (1 + exp(-(V + 35)/10))
    alpha_n = 0.01 (V + 55) / (1 - exp(-(V + 55)/10)), beta_n = 0.125 exp(-(V + 65)/80)

C_m = 1 uF/cm^2, g_Na = 120, g_K = 36, g_L = 0.3 mS/cm^2, E_Na = 50,
E_K = -77, E_L = -54.4 mV; the rate functions are those fitted at 6.3 C.
Conductances, reversal potentials and C_m are taken as temperature
independent, as in the 1952 paper. Stimulus: a constant 10 uA/cm^2 from
t = 0, starting from the steady state at -65 mV (which does not depend on
T), for 300 ms; firing rate and spike width are measured on [150, 300] ms.
Temperatures: 6, 7, ..., 20 C plus the reference 6.3 C.

Integrator: classical RK4 written out, fixed dt = 0.005 ms, vectorised
over all temperatures at once. Spike times are upward crossings of 0 mV;
the spike width is the full width at half amplitude, the half level being
midway between the spike peak and the preceding after-hyperpolarisation
minimum. Crossings are located by cubic Hermite interpolation (V and dV/dt
at the grid nodes), peaks by a parabola through the three highest samples.

The paper's Q10 = 3 is a statement about the rate constants. The emergent
quantities do not have to scale by 3, because C_m dV/dt carries no
temperature factor. The validation separates the two:

1. Exact time-scaling control (closed form). If C_m is replaced by
   C_m / phi(T) as well, the whole system becomes y'(t) = phi f(y), so its
   solution is the 6.3 C solution with time compressed by phi:
   V_T(t) = V_6.3(phi t). Every rate must then scale exactly as phi and
   every width as 1/phi, i.e. Q10 = 3 exactly. Asserted: the control's
   f(T) / (phi f(6.3)) and w(T) phi / w(6.3) are 1 within 1e-4 at every T,
   and the Q10 from a log-linear fit is 3 within 1e-4.
2. Reference solver. At 6.3, 13 and 20 C the physical model is integrated
   with scipy's DOP853 (rtol = atol = 1e-11) and measured with the same
   code on the same grid. Asserted: equal spike counts, and mean ISI and
   mean width within 1e-4 relative.
3. Comparison with Q10 = 3. With C_m fixed (the physical model) the fitted
   Q10 of firing rate and of spike width are reported; asserted only that
   both lie strictly between 1 and 3 and that rate rises and width falls
   monotonically with T, since the paper states no Q10 for these
   emergent quantities.

No random numbers are used. scipy is used only for the reference solver.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
C_m = 1.0                            # uF/cm^2
g_Na, g_K, g_L = 120.0, 36.0, 0.3    # mS/cm^2
E_Na, E_K, E_L = 50.0, -77.0, -54.4  # mV
Q10 = 3.0
T_BASE = 6.3                         # C, temperature of the fitted rate functions

I_AMP = 10.0                         # uA/cm^2, constant from t = 0
T_END = 300.0                        # ms
T_MEAS = 150.0                       # ms, measurements on [T_MEAS, T_END]
DT = 0.005                           # ms, RK4 step
TEMPS = np.concatenate(([T_BASE], np.arange(6.0, 20.5, 1.0)))  # C
REF_TEMPS = [6.3, 13.0, 20.0]
REF_TOL = 1e-11

CTRL_TOL = 1e-4                      # scaled-C_m control vs exact phi scaling
REF_REL_TOL = 1e-4                   # RK4 vs DOP853, ISI and width


# ------------------------------------------------------------------- model
def phi(T):
    return Q10 ** ((np.asarray(T, dtype=float) - T_BASE) / 10.0)


def rates(V):
    am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def rhs(y, I, ph, C):
    """y = (V, m, h, n), each an array over runs; ph and C per run."""
    V, m, h, n = y
    am, bm, ah, bh, an, bn = rates(V)
    dV = (I - g_Na * m ** 3 * h * (V - E_Na) - g_K * n ** 4 * (V - E_K)
          - g_L * (V - E_L)) / C
    return np.array([dV, ph * (am * (1.0 - m) - bm * m),
                     ph * (ah * (1.0 - h) - bh * h), ph * (an * (1.0 - n) - bn * n)])


def resting_state(n_runs, V=-65.0):
    am, bm, ah, bh, an, bn = rates(np.array(V))
    y = np.array([V, am / (am + bm), ah / (ah + bh), an / (an + bn)])
    return np.repeat(y[:, None], n_runs, axis=1)


def simulate(ph, C, I=I_AMP, t_end=T_END, dt=DT):
    """RK4 for all runs at once. Returns t, V[time, run], dV/dt[time, run]."""
    ph = np.asarray(ph, dtype=float)
    C = np.asarray(C, dtype=float)
    n = int(round(t_end / dt))
    t = np.arange(n + 1) * dt
    y = resting_state(len(ph))
    Vs = np.empty((n + 1, len(ph)))
    dVs = np.empty((n + 1, len(ph)))
    for k in range(n):
        k1 = rhs(y, I, ph, C)
        Vs[k], dVs[k] = y[0], k1[0]
        k2 = rhs(y + 0.5 * dt * k1, I, ph, C)
        k3 = rhs(y + 0.5 * dt * k2, I, ph, C)
        k4 = rhs(y + dt * k3, I, ph, C)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    Vs[n], dVs[n] = y[0], rhs(y, I, ph, C)[0]
    return t, Vs, dVs


def reference(T, I=I_AMP, t_end=T_END, dt=DT):
    """DOP853 at tight tolerance, sampled on the RK4 grid."""
    ph = float(phi(T))

    def f(_, y):
        return rhs(y, I, ph, C_m)

    sol = solve_ivp(f, (0.0, t_end), resting_state(1)[:, 0], method="DOP853",
                    rtol=REF_TOL, atol=REF_TOL, dense_output=True)
    t = np.arange(int(round(t_end / dt)) + 1) * dt
    Y = sol.sol(t)
    return t, Y[0], rhs(Y, I, ph, C_m)[0], sol.nfev


# ------------------------------------------------------------- measurement
def hermite_cross(t0, h, v0, v1, d0, d1, level):
    """Root of the cubic Hermite interpolant minus level on [t0, t0 + h]."""
    def p(s):
        h00 = 2 * s ** 3 - 3 * s ** 2 + 1
        h10 = s ** 3 - 2 * s ** 2 + s
        h01 = -2 * s ** 3 + 3 * s ** 2
        h11 = s ** 3 - s ** 2
        return h00 * v0 + h10 * h * d0 + h01 * v1 + h11 * h * d1 - level
    a, b = 0.0, 1.0
    pa = p(a)
    for _ in range(60):
        c = 0.5 * (a + b)
        pc = p(c)
        if (pc > 0) == (pa > 0):
            a, pa = c, pc
        else:
            b = c
    return t0 + 0.5 * (a + b) * h


def crossings(t, V, dV, level, i0, i1, up=True):
    """All crossings of level between grid indices i0 and i1."""
    seg = V[i0:i1 + 1] - level
    if up:
        idx = np.where((seg[:-1] < 0) & (seg[1:] >= 0))[0] + i0
    else:
        idx = np.where((seg[:-1] >= 0) & (seg[1:] < 0))[0] + i0
    h = t[1] - t[0]
    return np.array([hermite_cross(t[i], h, V[i], V[i + 1], dV[i], dV[i + 1], level)
                     for i in idx]), idx


def measure(t, V, dV, t_from=T_MEAS):
    """Rate (Hz), mean width (ms), mean peak and trough (mV) on [t_from, end]."""
    ts, idx = crossings(t, V, dV, 0.0, 0, len(t) - 1, up=True)
    keep = np.where(ts >= t_from)[0]
    keep = keep[keep >= 1]
    widths, peaks, troughs = [], [], []
    for j in keep[:-1]:                 # need the next spike to bound this one
        lo, mid, hi = idx[j - 1], idx[j], idx[j + 1]
        i_tr = lo + np.argmin(V[lo:mid + 1])
        i_pk = mid + np.argmax(V[mid:hi + 1])
        a, b, c = V[i_pk - 1], V[i_pk], V[i_pk + 1]
        curv = a - 2.0 * b + c
        peak = b - (a - c) ** 2 / (8.0 * curv) if curv != 0 else b
        level = 0.5 * (peak + V[i_tr])
        tu, _ = crossings(t, V, dV, level, i_tr, i_pk, up=True)
        td, _ = crossings(t, V, dV, level, i_pk, hi, up=False)
        widths.append(td[0] - tu[-1])
        peaks.append(peak)
        troughs.append(V[i_tr])
    isi = np.diff(ts[keep])
    return {"rate": 1000.0 / isi.mean(), "isi": isi.mean(), "width": np.mean(widths),
            "peak": np.mean(peaks), "trough": np.mean(troughs), "n_spikes": len(ts),
            "spikes": ts}


def q10_fit(T, y):
    """Q10 from a least-squares fit of ln y against T."""
    slope = np.polyfit(T, np.log(y), 1)[0]
    return float(np.exp(10.0 * slope))


# --------------------------------------------------------------- validation
def validate():
    R = {}
    ph = phi(TEMPS)
    nT = len(TEMPS)
    # physical model (C_m fixed) and the scaled-capacitance control in one run
    t0 = _time.time()
    t, V, dV = simulate(np.concatenate([ph, ph]),
                        np.concatenate([np.full(nT, C_m), C_m / ph]))
    print(f"RK4 dt = {DT} ms, {2 * nT} runs of {T_END:.0f} ms in {_time.time() - t0:.1f} s")
    phys = [measure(t, V[:, j], dV[:, j]) for j in range(nT)]
    ctrl = [measure(t, V[:, nT + j], dV[:, nT + j]) for j in range(nT)]
    R.update(t=t, V=V[:, :nT], Vc=V[:, nT:], phys=phys, ctrl=ctrl, ph=ph)

    f_p = np.array([m["rate"] for m in phys])
    w_p = np.array([m["width"] for m in phys])
    f_c = np.array([m["rate"] for m in ctrl])
    w_c = np.array([m["width"] for m in ctrl])
    order = np.argsort(TEMPS)
    Ts = TEMPS[order]

    # 1. exact time-scaling control
    dev_f = f_c / (ph * f_c[0]) - 1.0
    dev_w = w_c * ph / w_c[0] - 1.0
    qf_c = q10_fit(TEMPS, f_c)
    qw_c = 1.0 / q10_fit(TEMPS, w_c)
    print(f"[1] Scaled-C_m control (C_m/phi): rate at 6.3 C {f_c[0]:.4f} Hz, width {w_c[0]:.5f} ms")
    print(f"    max |f(T)/(phi f(6.3)) - 1| = {np.abs(dev_f).max():.2e}, "
          f"max |w(T) phi/w(6.3) - 1| = {np.abs(dev_w).max():.2e} (tolerance {CTRL_TOL:g})")
    print(f"    fitted Q10: rate {qf_c:.6f}, 1/width {qw_c:.6f} (exact 3)")
    assert np.abs(dev_f).max() < CTRL_TOL and np.abs(dev_w).max() < CTRL_TOL
    assert abs(qf_c - 3.0) < 3.0 * CTRL_TOL and abs(qw_c - 3.0) < 3.0 * CTRL_TOL
    R.update(dev_f=dev_f, dev_w=dev_w, qf_c=qf_c, qw_c=qw_c)

    # 2. reference solver for the physical model
    R["ref"] = {}
    print(f"[2] Physical model, RK4 dt = {DT} ms vs DOP853 rtol = atol = {REF_TOL:g}:")
    for Tr in REF_TEMPS:
        j = int(np.argmin(np.abs(TEMPS - Tr)))
        tr, Vr, dVr, nfev = reference(Tr)
        mr = measure(tr, Vr, dVr)
        e_isi = phys[j]["isi"] / mr["isi"] - 1.0
        e_w = phys[j]["width"] / mr["width"] - 1.0
        e_sp = np.abs(phys[j]["spikes"] - mr["spikes"]).max() if phys[j]["n_spikes"] == mr["n_spikes"] else np.inf
        print(f"    T = {Tr:4.1f} C: spikes {phys[j]['n_spikes']} vs {mr['n_spikes']}, "
              f"ISI {phys[j]['isi']:.5f} vs {mr['isi']:.5f} ms ({e_isi:+.1e}), "
              f"width {phys[j]['width']:.5f} vs {mr['width']:.5f} ms ({e_w:+.1e}), "
              f"max spike-time diff {e_sp:.1e} ms, {nfev} RHS evals")
        assert phys[j]["n_spikes"] == mr["n_spikes"]
        assert abs(e_isi) < REF_REL_TOL and abs(e_w) < REF_REL_TOL
        R["ref"][Tr] = mr

    # 3. physical Q10 of rate and width
    fs, ws = f_p[order], w_p[order]
    loc_f = (fs[1:] / fs[:-1]) ** (10.0 / np.diff(Ts))
    loc_w = (ws[:-1] / ws[1:]) ** (10.0 / np.diff(Ts))
    qf = q10_fit(TEMPS, f_p)
    qw = 1.0 / q10_fit(TEMPS, w_p)
    j6, j16 = 0, int(np.argmin(np.abs(TEMPS - 16.0)))
    print("[3] Physical model (C_m fixed), I = 10 uA/cm^2:")
    print("    T (C)  phi    rate (Hz)  width (ms)  peak (mV)  AHP (mV)")
    for j in order:
        m = phys[j]
        print(f"    {TEMPS[j]:5.1f}  {ph[j]:5.3f}  {m['rate']:8.2f}  {m['width']:9.4f}  "
              f"{m['peak']:8.2f}  {m['trough']:8.2f}")
    print(f"    fitted Q10 over 6-20 C: rate {qf:.3f}, 1/width {qw:.3f} (rate constants: 3)")
    print(f"    local Q10 of rate {loc_f.min():.3f}-{loc_f.max():.3f}, of 1/width {loc_w.min():.3f}-{loc_w.max():.3f}")
    print(f"    6.3 -> 16 C: rate x{f_p[j16] / f_p[j6]:.3f}, width x{w_p[j16] / w_p[j6]:.3f}, phi x{ph[j16] / ph[j6]:.3f}")
    assert np.all(np.diff(fs) > 0) and np.all(np.diff(ws) < 0)
    assert 1.0 < qf < 3.0 and 1.0 < qw < 3.0
    R.update(qf=qf, qw=qw, loc_f=loc_f, loc_w=loc_w, Ts=Ts, fs=fs, ws=ws,
             fcs=f_c[order], wcs=w_c[order], order=order)
    return R


# --------------------------------------------------------------------- plot
def plot(R):
    t, V, Vc, ph = R["t"], R["V"], R["Vc"], R["ph"]
    Ts, order = R["Ts"], R["order"]
    show = [6.3, 13.0, 20.0]
    cols = {6.3: "tab:blue", 13.0: "tab:orange", 20.0: "tab:red"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    a = ax[0, 0]
    for Tr in show:
        j = int(np.argmin(np.abs(TEMPS - Tr)))
        sel = (t >= 250) & (t <= 300)
        a.plot(t[sel], V[sel, j], color=cols[Tr], lw=1, label=f"{Tr:g} °C")
    a.set_xlabel("time (ms)")
    a.set_ylabel("V (mV)")
    a.set_title("Physical model (C_m fixed), I = 10 µA/cm²")
    a.legend(fontsize=8, loc="upper right")

    a = ax[0, 1]
    for Tr in show:
        j = int(np.argmin(np.abs(TEMPS - Tr)))
        tsp = R["phys"][j]["spikes"]
        s0 = tsp[tsp >= 250][0]
        sel = (t >= s0 - 3) & (t <= s0 + 4.5)
        a.plot(t[sel] - s0, V[sel, j], color=cols[Tr], lw=1.5, label=f"physical {Tr:g} °C")
        tsc = R["ctrl"][j]["spikes"]
        s1 = tsc[tsc >= 250][0]
        selc = (t >= s1 - 3 / ph[j]) & (t <= s1 + 4.5 / ph[j])
        a.plot((t[selc] - s1) * ph[j] / ph[0], Vc[selc, j], color=cols[Tr], lw=1, ls="--",
               label=f"C_m/φ control {Tr:g} °C, time × φ")
    a.set_xlabel("time from 0 mV crossing (ms; control rescaled to 6.3 °C)")
    a.set_ylabel("V (mV)")
    a.set_title("Spike shape: control collapses exactly, physical narrows less")
    a.legend(fontsize=7, loc="upper right")

    a = ax[0, 2]
    for Tr in show:
        j = int(np.argmin(np.abs(TEMPS - Tr)))
        sel = (t >= 200) & (t <= 300)
        a.plot(V[sel, j], np.gradient(V[sel, j], t[1] - t[0]), color=cols[Tr], lw=1, label=f"{Tr:g} °C")
    a.set_xlabel("V (mV)")
    a.set_ylabel("dV/dt (mV/ms)")
    a.set_title("Phase plot: lower peak, faster repolarisation when warm")
    a.legend(fontsize=8)

    TT = np.linspace(Ts[0], Ts[-1], 100)
    j0 = 0
    a = ax[1, 0]
    a.semilogy(Ts, R["fs"], "o-", color="k", ms=4, label=f"physical, fitted Q10 = {R['qf']:.2f}")
    a.semilogy(Ts, R["fcs"], "s", color="tab:green", ms=5, mfc="none", label="C_m/φ control")
    a.semilogy(TT, R["phys"][j0]["rate"] * phi(TT), color="tab:green", lw=1, label="f(6.3)·3^((T−6.3)/10)")
    for Tr, mr in R["ref"].items():
        a.semilogy(Tr, mr["rate"], "x", color="tab:red", ms=9, mew=2)
    a.semilogy([], [], "x", color="tab:red", ms=9, mew=2, label="DOP853 reference")
    a.set_xlabel("temperature (°C)")
    a.set_ylabel("firing rate (Hz)")
    a.set_title("Firing rate vs temperature")
    a.legend(fontsize=8)

    a = ax[1, 1]
    a.semilogy(Ts, R["ws"], "o-", color="k", ms=4, label=f"physical, fitted Q10 of 1/width = {R['qw']:.2f}")
    a.semilogy(Ts, R["wcs"], "s", color="tab:green", ms=5, mfc="none", label="C_m/φ control")
    a.semilogy(TT, R["phys"][j0]["width"] / phi(TT), color="tab:green", lw=1, label="w(6.3)·3^(−(T−6.3)/10)")
    for Tr, mr in R["ref"].items():
        a.semilogy(Tr, mr["width"], "x", color="tab:red", ms=9, mew=2)
    a.semilogy([], [], "x", color="tab:red", ms=9, mew=2, label="DOP853 reference")
    a.set_xlabel("temperature (°C)")
    a.set_ylabel("spike width at half amplitude (ms)")
    a.set_title("Spike width vs temperature")
    a.legend(fontsize=8)

    a = ax[1, 2]
    mids = 0.5 * (Ts[1:] + Ts[:-1])
    a.plot(mids, R["loc_f"], "o-", color="k", ms=4, label="local Q10, rate (physical)")
    a.plot(mids, R["loc_w"], "^-", color="tab:purple", ms=4, label="local Q10, 1/width (physical)")
    a.axhline(3.0, color="tab:green", lw=1, label="Q10 of rate constants (HH 1952)")
    a2 = a.twinx()
    a2.semilogy(Ts, np.abs(R["dev_f"][order]) + 1e-16, "s", color="tab:green", ms=4, mfc="none")
    a2.semilogy(Ts, np.abs(R["dev_w"][order]) + 1e-16, "D", color="tab:olive", ms=4, mfc="none")
    a2.axhline(CTRL_TOL, color="gray", ls=":", lw=1)
    a2.set_ylabel("control: |ratio/φ − 1| (□ rate, ◇ width)")
    a.set_ylim(1.0, 3.2)
    a.set_xlabel("temperature (°C)")
    a.set_ylabel("Q10")
    a.set_title("Emergent Q10 vs the rate-constant Q10")
    a.legend(fontsize=8, loc="lower left")

    fig.suptitle("Hodgkin–Huxley with Q10 = 3 temperature scaling of the gating rates, 6–20 °C", fontsize=13)
    fig.tight_layout()
    out = FIGURES / "hh_temperature.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
