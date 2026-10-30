"""Wang-Buzsaki fast-spiking hippocampal interneuron.

Equations (Wang & Buzsaki 1996, J. Neurosci. 16(20), 6402-6413, single
cell part of the Methods), V in mV, t in ms:

    C_m dV/dt = I - I_Na - I_K - I_L
    I_Na = g_Na m_inf(V)^3 h (V - E_Na),   m_inf = alpha_m / (alpha_m + beta_m)
    I_K  = g_K n^4 (V - E_K)
    I_L  = g_L (V - E_L)
    dh/dt = phi [alpha_h (1 - h) - beta_h h]
    dn/dt = phi [alpha_n (1 - n) - beta_n n]

    alpha_m = -0.1 (V + 35) / (exp(-0.1 (V + 35)) - 1),  beta_m = 4 exp(-(V + 60)/18)
    alpha_h = 0.07 exp(-(V + 58)/20),                    beta_h = 1 / (exp(-0.1 (V + 28)) + 1)
    alpha_n = -0.01 (V + 34) / (exp(-0.1 (V + 34)) - 1), beta_n = 0.125 exp(-(V + 44)/80)

C_m = 1 uF/cm^2, g_Na = 35, g_K = 9, g_L = 0.1 mS/cm^2, E_Na = 55,
E_K = -90, E_L = -65 mV, phi = 5. Na activation is instantaneous, so the
state is (V, h, n). Stimulus: a constant current I from t = 0, starting
from the resting state at I = 0.

Integrator: classical RK4 written out, fixed dt = 0.01 ms, vectorised over
currents. Spike times are upward crossings of -20 mV. The spike width is
the full width at half amplitude, the half level midway between the peak
and the preceding after-hyperpolarisation minimum (the same definition as
src/hh_temperature.py). Crossings use cubic Hermite interpolation (V and
dV/dt at the grid nodes), peaks a parabola through the three top samples.

Validation (validate()):

1. Closed form, rest state. With h and n at steady state the membrane
   current is I_ss(V) = g_Na m^3 h (V-E_Na) + g_K n^4 (V-E_K) + g_L (V-E_L)
   (all gates at their infinity values). The simulated rest at I = 0 must
   equal the root of I_ss(V) = 0 to 1e-6 mV.
2. Closed form, onset. The model is class I: rest disappears in a saddle
   node at the local maximum of I_ss, I_SN. Near onset the period follows
   the saddle-node law T = pi / sqrt(K (I - I_SN)) + T0. Fitting that law
   to simulated ISIs at I_SN + 0.002 ... 0.128 uA/cm^2 with I_SN free must
   return the I_ss maximum within 0.1%. Asserted also: the rate just above
   onset is below 5 Hz (continuous from 0 Hz) and a cell started on the
   limit cycle stops firing 0.005 uA/cm^2 below I_SN (no bistability).
3. Reference solver. At I = 1, 5 and 20 uA/cm^2 the model is integrated
   with scipy's DOP853 (rtol = atol = 1e-10). Asserted: equal spike counts,
   mean ISI and mean spike width within 1e-4 relative.
4. Published number. Wang & Buzsaki (1996, fig. 1) show the f-I curve
   rising to about 400 Hz at I = 20 uA/cm^2; asserted within 10%
   (the value is read from the figure, hence the loose tolerance).

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
from scipy.optimize import brentq, minimize_scalar, curve_fit

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
C_m = 1.0                              # uF/cm^2
g_Na, g_K, g_L = 35.0, 9.0, 0.1        # mS/cm^2
E_Na, E_K, E_L = 55.0, -90.0, -65.0    # mV
PHI = 5.0

DT = 0.01                              # ms, RK4 step
V_SPIKE = -20.0                        # mV, spike detection level
T_FI = 1000.0                          # ms, f-I runs; measured on [T_FROM, T_FI]
T_FROM = 300.0
I_FI = np.concatenate((np.arange(0.0, 1.0, 0.05), np.arange(1.0, 20.01, 0.5)))
DI_ONSET = 0.002 * 2.0 ** np.arange(7)  # uA/cm^2 above I_SN
T_ONSET = 6000.0                       # ms, onset runs
DI_BELOW = 0.005                       # uA/cm^2 below I_SN, bistability test
REF_CURRENTS = [1.0, 5.0, 20.0]
REF_TOL = 1e-10
REF_REL_TOL = 1e-4
I_PUB, F_PUB, PUB_TOL = 20.0, 400.0, 0.10
REST_TOL = 1e-6
ISN_TOL = 1e-3
HH_WIDTH = 1.5056                      # ms, HH at 6.3 C and 10 uA/cm^2 (hh_temperature.py)


# ------------------------------------------------------------------- model
def vtrap(x):
    """x / (1 - exp(-x)), continuous at x = 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-7
    return np.where(small, 1.0 + 0.5 * x, x / np.where(small, 1.0, -np.expm1(-x)))


def rates(V):
    a_m = vtrap(0.1 * (V + 35.0))
    b_m = 4.0 * np.exp(-(V + 60.0) / 18.0)
    a_h = 0.07 * np.exp(-(V + 58.0) / 20.0)
    b_h = 1.0 / (np.exp(-0.1 * (V + 28.0)) + 1.0)
    a_n = 0.1 * vtrap(0.1 * (V + 34.0))
    b_n = 0.125 * np.exp(-(V + 44.0) / 80.0)
    return a_m, b_m, a_h, b_h, a_n, b_n


def rhs(V, h, n, I):
    a_m, b_m, a_h, b_h, a_n, b_n = rates(V)
    m = a_m / (a_m + b_m)
    dV = (I - g_Na * m ** 3 * h * (V - E_Na) - g_K * n ** 4 * (V - E_K)
          - g_L * (V - E_L)) / C_m
    return dV, PHI * (a_h * (1.0 - h) - b_h * h), PHI * (a_n * (1.0 - n) - b_n * n)


def steady_current(V):
    """I_ss(V): current needed to hold V with every gate at steady state."""
    a_m, b_m, a_h, b_h, a_n, b_n = rates(V)
    m, h, n = a_m / (a_m + b_m), a_h / (a_h + b_h), a_n / (a_n + b_n)
    return (g_Na * m ** 3 * h * (V - E_Na) + g_K * n ** 4 * (V - E_K)
            + g_L * (V - E_L))


def gates_inf(V):
    a_m, b_m, a_h, b_h, a_n, b_n = rates(V)
    return a_h / (a_h + b_h), a_n / (a_n + b_n)


def rest_voltage():
    return brentq(steady_current, -70.0, -62.0, xtol=1e-13)


def saddle_node():
    """Local maximum of I_ss (the knee of the N-shaped I-V curve)."""
    r = minimize_scalar(lambda v: -steady_current(v), bounds=(-63.0, -55.0),
                        method="bounded", options={"xatol": 1e-10})
    return float(r.x), float(-r.fun)


def simulate(I, t_end, y0=None, dt=DT, keep=None):
    """RK4 for an array of currents. Returns spike times per current, the
    final state and, if keep = (t_from, _), V stored from t_from on."""
    I = np.atleast_1d(np.asarray(I, dtype=float))
    if y0 is None:
        Vr = rest_voltage()
        h0, n0 = gates_inf(Vr)
        y0 = (Vr, h0, n0)
    V, h, n = (np.full_like(I, 0.0) + np.asarray(c, dtype=float) for c in y0)
    n_steps = int(round(t_end / dt))
    spikes = [[] for _ in I]
    store_t, store = [], []
    i_keep = None if keep is None else int(round(keep[0] / dt))
    for k in range(n_steps):
        if i_keep is not None and k >= i_keep:
            store_t.append(k * dt)
            store.append(V.copy())
        k1 = rhs(V, h, n, I)
        k2 = rhs(V + 0.5 * dt * k1[0], h + 0.5 * dt * k1[1], n + 0.5 * dt * k1[2], I)
        k3 = rhs(V + 0.5 * dt * k2[0], h + 0.5 * dt * k2[1], n + 0.5 * dt * k2[2], I)
        k4 = rhs(V + dt * k3[0], h + dt * k3[1], n + dt * k3[2], I)
        Vn = V + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        h = h + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        n = n + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        for j in np.where((V < V_SPIKE) & (Vn >= V_SPIKE))[0]:
            # linear interpolation for spike times; widths use Hermite below
            spikes[j].append(k * dt + dt * (V_SPIKE - V[j]) / (Vn[j] - V[j]))
        V = Vn
    out = {"spikes": [np.array(s) for s in spikes], "state": (V, h, n)}
    if keep is not None:
        out["t"] = np.array(store_t)
        out["V"] = np.array(store)
    return out


def reference(I, t_end=T_FI):
    """DOP853 run, sampled on the RK4 grid from T_FROM on."""
    Vr = rest_voltage()
    h0, n0 = gates_inf(Vr)
    f = lambda t, y: rhs(y[0], y[1], y[2], I)
    t_eval = np.arange(int(round(T_FROM / DT)), int(round(t_end / DT))) * DT
    sol = solve_ivp(f, (0.0, t_end), [Vr, h0, n0], method="DOP853",
                    rtol=REF_TOL, atol=REF_TOL, t_eval=t_eval)
    return sol.t, sol.y


# -------------------------------------------------------------- measurement
def hermite_cross(t0, h, v0, v1, d0, d1, level):
    """Root of the cubic Hermite interpolant minus level on [t0, t0 + h]."""
    def p(s):
        return ((2 * s ** 3 - 3 * s ** 2 + 1) * v0 + (s ** 3 - 2 * s ** 2 + s) * h * d0
                + (-2 * s ** 3 + 3 * s ** 2) * v1 + (s ** 3 - s ** 2) * h * d1 - level)
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
    seg = V[i0:i1 + 1] - level
    if up:
        idx = np.where((seg[:-1] < 0) & (seg[1:] >= 0))[0] + i0
    else:
        idx = np.where((seg[:-1] >= 0) & (seg[1:] < 0))[0] + i0
    h = t[1] - t[0]
    return np.array([hermite_cross(t[i], h, V[i], V[i + 1], dV[i], dV[i + 1], level)
                     for i in idx]), idx


def measure(t, V, h, n, I):
    """Mean ISI (ms), rate (Hz), mean width (ms), peak and trough (mV)."""
    dV = rhs(V, h, n, I)[0]
    ts, idx = crossings(t, V, dV, V_SPIKE, 0, len(t) - 1, up=True)
    widths, peaks, troughs = [], [], []
    for j in range(1, len(idx) - 1):
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
    isi = np.diff(ts)
    return {"isi": isi.mean(), "rate": 1000.0 / isi.mean(), "width": np.mean(widths),
            "peak": np.mean(peaks), "trough": np.mean(troughs), "n_spikes": len(ts),
            "spikes": ts}


def rk4_trajectory(I, t_end=T_FI):
    """Single-current RK4 run storing V, h, n from T_FROM on (for widths)."""
    Vr = rest_voltage()
    h, n = gates_inf(Vr)
    V = Vr
    n_steps = int(round(t_end / DT))
    i0 = int(round(T_FROM / DT))
    out = np.empty((3, n_steps - i0))
    for k in range(n_steps):
        if k >= i0:
            out[:, k - i0] = V, h, n
        k1 = rhs(V, h, n, I)
        k2 = rhs(V + 0.5 * DT * k1[0], h + 0.5 * DT * k1[1], n + 0.5 * DT * k1[2], I)
        k3 = rhs(V + 0.5 * DT * k2[0], h + 0.5 * DT * k2[1], n + 0.5 * DT * k2[2], I)
        k4 = rhs(V + DT * k3[0], h + DT * k3[1], n + DT * k3[2], I)
        V = V + DT / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        h = h + DT / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        n = n + DT / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    return np.arange(i0, n_steps) * DT, out


def sn_law(I, I_sn, K, T0):
    return np.pi / np.sqrt(K * np.clip(I - I_sn, 1e-12, None)) + T0


# --------------------------------------------------------------- validation
def validate():
    R = {}
    # 1. rest state
    V_rest = rest_voltage()
    run0 = simulate([0.0], 500.0)
    V_sim = float(run0["state"][0][0])
    dev_rest = abs(V_sim - V_rest)
    print(f"[1] rest: root of I_ss = {V_rest:.9f} mV, simulated (500 ms, I = 0) "
          f"{V_sim:.9f} mV, |dev| = {dev_rest:.2e} (tolerance {REST_TOL:g})")
    assert dev_rest < REST_TOL
    R["V_rest"] = V_rest

    # 2. saddle-node onset
    V_sn, I_sn = saddle_node()
    print(f"[2] I_ss maximum: I_SN = {I_sn:.6f} uA/cm^2 at V = {V_sn:.4f} mV")
    I_on = I_sn + DI_ONSET
    # bistability test: start below I_SN from a state on the limit cycle at I_SN + 0.05
    cyc = simulate([I_sn + 0.05], 400.0)["state"]
    I_all = np.concatenate((I_on, [I_sn - DI_BELOW]))
    Vr = rest_voltage()
    h0, n0 = gates_inf(Vr)
    y0 = (np.concatenate((np.full(len(I_on), Vr), cyc[0])),
          np.concatenate((np.full(len(I_on), h0), cyc[1])),
          np.concatenate((np.full(len(I_on), n0), cyc[2])))
    on = simulate(I_all, T_ONSET, y0=y0)
    T_on = []
    for s in on["spikes"][:-1]:
        T_on.append(np.diff(s[1:]).mean())   # drop the transient first interval
    T_on = np.array(T_on)
    n_below = len(on["spikes"][-1])
    late_below = int(np.sum(on["spikes"][-1] > 1000.0))
    (I_fit, K_fit, T0_fit), _ = curve_fit(sn_law, I_on, T_on, p0=[I_sn, 0.015, 0.0])
    rel_sn = (I_fit - I_sn) / I_sn
    resid = sn_law(I_on, I_fit, K_fit, T0_fit) / T_on - 1
    f_min = 1000.0 / T_on[0]
    print(f"    onset ISIs at I_SN + {', '.join(f'{d:g}' for d in DI_ONSET)}: "
          f"{', '.join(f'{x:.2f}' for x in T_on)} ms")
    print(f"    fit T = pi/sqrt(K (I - I_SN)) + T0: I_SN = {I_fit:.6f} "
          f"({rel_sn:+.2e} relative, tolerance {ISN_TOL:g}), K = {K_fit:.5f}, "
          f"T0 = {T0_fit:.2f} ms, max fit residual {np.abs(resid).max():.2e}")
    print(f"    lowest rate {f_min:.3f} Hz at I_SN + {DI_ONSET[0]:g}; "
          f"started on the cycle at I_SN - {DI_BELOW:g}: {n_below} spikes, "
          f"{late_below} after 1000 ms")
    assert abs(rel_sn) < ISN_TOL
    assert f_min < 5.0
    assert late_below == 0
    R.update(I_sn=I_sn, V_sn=V_sn, I_on=I_on, T_on=T_on, sn_fit=(I_fit, K_fit, T0_fit),
             f_min=f_min, n_below=n_below)

    # f-I curve
    fi = simulate(I_FI, T_FI)
    rate = np.array([1000.0 / np.diff(s[s >= T_FROM]).mean() if np.sum(s >= T_FROM) > 2
                     else 0.0 for s in fi["spikes"]])
    R.update(I_fi=I_FI, rate=rate)

    # 3. reference solver, and widths
    ref = []
    for I in REF_CURRENTS:
        tr, yr = reference(I)
        mr = measure(tr, yr[0], yr[1], yr[2], I)
        tk, yk = rk4_trajectory(I)
        mk = measure(tk, yk[0], yk[1], yk[2], I)
        e_isi = mk["isi"] / mr["isi"] - 1
        e_w = mk["width"] / mr["width"] - 1
        print(f"[3] I = {I:g}: spikes {mk['n_spikes']} (RK4) vs {mr['n_spikes']} (DOP853); "
              f"ISI {mk['isi']:.6f} vs {mr['isi']:.6f} ms ({e_isi:+.2e}); width "
              f"{mk['width']:.5f} vs {mr['width']:.5f} ms ({e_w:+.2e}); peak "
              f"{mk['peak']:.2f}, AHP {mk['trough']:.2f} mV (tolerance {REF_REL_TOL:g})")
        assert mk["n_spikes"] == mr["n_spikes"]
        assert abs(e_isi) < REF_REL_TOL and abs(e_w) < REF_REL_TOL
        ref.append(dict(I=I, rk=mk, ref=mr, tk=tk, yk=yk, e_isi=e_isi, e_w=e_w))
    R["ref"] = ref

    # 4. published number
    f20 = ref[-1]["rk"]["rate"]
    rel_pub = f20 / F_PUB - 1
    print(f"[4] rate at I = {I_PUB:g}: {f20:.2f} Hz vs ~{F_PUB:g} Hz (Wang & Buzsaki 1996, "
          f"fig. 1), {rel_pub:+.2%} (tolerance {PUB_TOL:.0%})")
    assert abs(rel_pub) < PUB_TOL
    w1 = ref[0]["rk"]["width"]
    print(f"    spike width {w1:.3f} ms at I = 1, {ref[-1]['rk']['width']:.3f} ms at I = 20; "
          f"HH at 10 uA/cm^2, 6.3 C, same definition: {HH_WIDTH} ms "
          f"(ratio {HH_WIDTH / w1:.1f}, not asserted)")
    R["f20"], R["rel_pub"] = f20, rel_pub
    R["trace_low"] = simulate([I_sn + DI_ONSET[2]], 3000.0, keep=(0.0, 1))
    return R


# ----------------------------------------------------------------- plotting
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    ref = R["ref"]
    cols = ["tab:blue", "tab:orange", "tab:red"]

    a = ax[0, 0]
    for r, c in zip(ref, cols):
        tk, V = r["tk"], r["yk"][0]
        sel = (tk >= 900) & (tk <= 950)
        a.plot(tk[sel] - 900, V[sel], color=c, lw=1, label=f"I = {r['I']:g} µA/cm²")
    a.set_xlabel("time (ms, from t = 900)")
    a.set_ylabel("V (mV)")
    a.set_title("Repetitive firing")
    a.legend(fontsize=8, loc="lower right")

    a = ax[0, 1]
    tl = R["trace_low"]
    a.plot(tl["t"], tl["V"][:, 0], color="k", lw=0.8)
    a.set_xlabel("time (ms)")
    a.set_ylabel("V (mV)")
    a.set_title(f"Near onset: I = I_SN + {DI_ONSET[2]:g} µA/cm²")

    a = ax[0, 2]
    a.plot(R["I_fi"], R["rate"], ".-", color="tab:blue", lw=1, ms=4, label="RK4 dt = 0.01 ms")
    a.plot([r["I"] for r in ref], [r["ref"]["rate"] for r in ref], "o", mfc="none",
           color="k", ms=8, label="DOP853 (tol 1e-10)")
    a.errorbar([I_PUB], [F_PUB], yerr=[PUB_TOL * F_PUB], fmt="s", color="tab:red",
               capsize=4, label="~400 Hz, W&B 1996 fig. 1 (±10%)")
    a.axvline(R["I_sn"], color="grey", ls=":", lw=1)
    a.set_xlabel("I (µA/cm²)")
    a.set_ylabel("rate (Hz)")
    a.set_title("f–I curve (class I, from 0 Hz)")
    a.legend(fontsize=8, loc="lower right")

    a = ax[1, 0]
    I_fit, K_fit, T0_fit = R["sn_fit"]
    dI = R["I_on"] - R["I_sn"]
    d = np.geomspace(dI[0] * 0.7, dI[-1] * 1.4, 200)
    a.loglog(dI, R["T_on"], "o", color="tab:blue", label="simulated ISI")
    a.loglog(d, sn_law(R["I_sn"] + d, I_fit, K_fit, T0_fit), "k-", lw=1,
             label=f"π/√(K(I−I_SN))+T0, fitted I_SN = {I_fit:.5f}")
    a.set_xlabel("I − I_SN (µA/cm²), I_SN from max of I_ss")
    a.set_ylabel("ISI (ms)")
    a.set_title(f"Saddle-node law; I_ss max = {R['I_sn']:.5f}")
    a.legend(fontsize=8)

    a = ax[1, 1]
    V = np.linspace(-80, -40, 400)
    a.plot(V, steady_current(V), "k-", lw=1)
    a.axhline(0, color="grey", lw=0.5)
    a.plot(R["V_sn"], R["I_sn"], "o", color="tab:red", label="saddle node (max of I_ss)")
    a.plot(R["V_rest"], 0.0, "o", color="tab:blue", label=f"rest {R['V_rest']:.3f} mV")
    a.set_ylim(-2, 1)
    a.set_xlabel("V (mV)")
    a.set_ylabel("I_ss (µA/cm²)")
    a.set_title("Steady-state I–V curve")
    a.legend(fontsize=8, loc="lower left")

    a = ax[1, 2]
    for r, c in zip(ref, cols):
        mk = r["rk"]
        tk, Vk = r["tk"], r["yk"][0]
        ts = mk["spikes"][len(mk["spikes"]) // 2]
        sel = (tk >= ts - 1.5) & (tk <= ts + 2.5)
        a.plot(tk[sel] - ts, Vk[sel], color=c, lw=1.2,
               label=f"I = {r['I']:g}: width {mk['width']:.3f} ms")
    a.set_xlabel("time from spike (ms)")
    a.set_ylabel("V (mV)")
    a.set_title(f"Spike shape (HH, same definition: {HH_WIDTH:.2f} ms)")
    a.legend(fontsize=8)

    fig.suptitle("Wang–Buzsáki interneuron: RK4 dt = 0.01 ms; checks against I_ss, "
                 "saddle-node law, DOP853 and the published ~400 Hz", fontsize=11)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "wang_buzsaki.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
