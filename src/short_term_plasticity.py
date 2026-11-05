"""Short-term synaptic plasticity: the Tsodyks-Markram model of depression and
facilitation, driving a passive membrane through an exponential current.

Synapse (Tsodyks & Markram 1997 for depression; Tsodyks, Pawelzik & Markram
1998 and Markram, Wang & Tsodyks 1998 for facilitation), t in ms:

    dx/dt = (1 - x)/tau_D - u+ x- delta(t - t_sp)       available resources
    du/dt = -u/tau_F + U (1 - u-) delta(t - t_sp)        utilisation (facilitating)

At each presynaptic spike u jumps first, u+ = u- + U (1 - u-), then the
fraction u+ x- of the resources is released and x+ = x- - u+ x-. The
synaptic efficacy of spike n is E_n = u+_n x-_n. For a purely depressing
synapse (tau_F -> 0) u+ = U at every spike, which is the 1997 model.

The released resource drives an exponentially decaying synaptic current and a
passive membrane (the 1997 paper's tau_in and tau_mem), V in mV, I in pA:

    dI/dt = -I/tau_s + A E_n delta(t - t_sp)
    tau_m dV/dt = -(V - E_L) + R_in I

Parameters (chosen here, in the ranges reported by Tsodyks & Markram 1997
and Markram, Wang & Tsodyks 1998; not fitted to data):
    depressing:   U = 0.5,  tau_D = 800 ms, no facilitation
    facilitating: U = 0.1,  tau_D = 100 ms, tau_F = 1000 ms
    A = 250 pA, tau_s = 3 ms, tau_m = 20 ms, R_in = 0.1 GOhm, E_L = -70 mV.

Integrator: classical RK4 on (x, u, I, V) at dt = 0.05 ms. Presynaptic spikes
lie on the grid and their jumps are applied between steps.

Closed forms (derived from the jump map, independent of the simulation). For
a regular train with interval D = 1000/f, e_D = exp(-D/tau_D),
e_F = exp(-D/tau_F) (e_F = 0 without facilitation):

    u+_n  = u_ss + (U - u_ss) [(1 - U) e_F]^(n-1),    u_ss = U / (1 - (1 - U) e_F)
    x-_ss = (1 - e_D) / (1 - (1 - u_ss) e_D)
    E_ss  = u_ss x-_ss

and, for any spike train, the exact event map
    u-_{n+1} = u+_n exp(-D_n/tau_F),  x-_{n+1} = 1 - (1 - x+_n) exp(-D_n/tau_D).

Single PSP for a current jump J: v(t) = R J tau_s/(tau_m - tau_s)
(exp(-t/tau_m) - exp(-t/tau_s)). In the periodic steady state the PSPs sum to
    V_ss(t) - E_L = R A E_ss tau_s/(tau_m - tau_s)
                    [exp(-t/tau_m)/(1 - q_m) - exp(-t/tau_s)/(1 - q_s)],  0 <= t < D,
q = exp(-D/tau). Its maximum is at
    t* = ln[tau_m (1 - q_m) / (tau_s (1 - q_s))] / (1/tau_s - 1/tau_m),
and the steady-state PSP amplitude as measured in a recording (peak minus the
trough just before the spike) is P_ss = V_ss(t*) - V_ss(0).

Validation (validate()):
1. Closed form, steady state vs frequency: E_ss and P_ss from long regular
   trains at f = 0.5 ... 100 Hz for both synapses vs the formulas above;
   max relative error < 1e-5 (P_ss peak located by quadratic interpolation
   on the dt grid). Train lengths are chosen so the transient has decayed
   below 1e-12.
2. Closed form / exact map, transient: E_n over 15-spike 20 Hz and 50 Hz
   trains and a recovery spike 500 ms later vs the exact event map; the
   depressing E_n also vs its geometric closed form
   x-_n = x_ss + (1 - x_ss) [(1 - U) e_D]^(n-1), and the facilitating u+_n vs
   the closed form for u+_n above; max relative error < 1e-8.
3. Exact solution, voltage: V(t) for an 8-spike 20 Hz train plus recovery
   spike vs the superposition of single-PSP kernels weighted by the exact
   E_n; max error < 1e-6 mV at dt = 0.05 ms, and RK4 order 4 +- 0.3 from
   dt = 0.4 ... 0.05 ms.
4. Reported, not asserted: the depressing synapse's 1/f regime,
   f E_ss tau_D -> 1 as f -> infinity (Tsodyks & Markram 1997), with the
   limiting frequency 1/(U tau_D); the frequency at which the facilitating
   synapse's E_ss peaks (scipy minimize_scalar on the closed form).

The model has no randomness, so no RNG is used. SciPy is used only for the
validation (minimize_scalar).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.optimize import minimize_scalar  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Parameters
DT = 0.05        # ms
A_SE = 250.0     # pA, absolute synaptic efficacy
TAU_S = 3.0      # ms, synaptic current decay (tau_in)
TAU_M = 20.0     # ms, membrane time constant
R_IN = 0.1       # GOhm (pA * GOhm = mV)
E_L = -70.0      # mV

SYNAPSES = {
    "depressing": dict(U=0.5, tau_D=800.0, tau_F=None),
    "facilitating": dict(U=0.1, tau_D=100.0, tau_F=1000.0),
}

FREQS = np.array([0.5, 1, 2, 4, 5, 8, 10, 16, 20, 25, 40, 50, 80, 100.0])  # D on the dt grid


# ---------------------------------------------------------------------------
# Model functions
def simulate(conds, spike_steps, n_steps, dt=DT, windows=None):
    """RK4 on (x, u, I, V) for C conditions at once.

    conds: list of dicts with U, tau_D, tau_F (None = pure depression).
    spike_steps: list of integer arrays, the grid indices of each condition's
    spikes (the jump is applied before the step starting there).
    windows: list of (start_step, length) per condition; V is recorded on
    steps start ... start + length. Returns per-spike efficacies, the
    recorded V windows, the (x, u) traces over the windows, and u+ at each
    spike.
    """
    C = len(conds)
    U = np.array([c["U"] for c in conds])
    inv_D = np.array([1.0 / c["tau_D"] for c in conds])
    inv_F = np.array([0.0 if c["tau_F"] is None else 1.0 / c["tau_F"] for c in conds])
    facil = np.array([c["tau_F"] is not None for c in conds])
    x = np.ones(C)
    u = np.where(facil, 0.0, U)
    I = np.zeros(C)
    V = np.full(C, E_L)

    big = n_steps + 10
    L = max(len(s) for s in spike_steps)
    sched = np.full((C, L + 1), big, dtype=np.int64)
    for c, s in enumerate(spike_steps):
        sched[c, :len(s)] = s
    ptr = np.zeros(C, dtype=np.int64)
    eff = [[] for _ in range(C)]
    usp = [[] for _ in range(C)]

    if windows is None:
        windows = [(0, n_steps)] * C
    w0 = np.array([w[0] for w in windows], dtype=np.int64)
    wl = np.array([w[1] for w in windows], dtype=np.int64)
    rec = np.full((C, wl.max() + 1), np.nan)
    rec_x = np.full_like(rec, np.nan)
    rec_u = np.full_like(rec, np.nan)
    cidx = np.arange(C)

    def f(x_, u_, I_, V_):
        return (1.0 - x_) * inv_D, -u_ * inv_F, -I_ / TAU_S, (-(V_ - E_L) + R_IN * I_) / TAU_M

    h = dt
    for k in range(n_steps + 1):
        nxt = sched[cidx, ptr]
        spk = nxt == k
        if spk.any():
            up = np.where(facil, u + U * (1.0 - u), U)
            e = up * x
            for c in np.nonzero(spk)[0]:
                eff[c].append(e[c])
                usp[c].append(up[c])
            x = np.where(spk, x - e, x)
            u = np.where(spk, up, u)
            I = np.where(spk, I + A_SE * e, I)
            ptr = ptr + spk
        j = k - w0
        m = (j >= 0) & (j <= wl)
        if m.any():
            rec[m, j[m]] = V[m]
            rec_x[m, j[m]] = x[m]
            rec_u[m, j[m]] = u[m]
        if k == n_steps:
            break
        a1 = f(x, u, I, V)
        a2 = f(x + 0.5 * h * a1[0], u + 0.5 * h * a1[1], I + 0.5 * h * a1[2], V + 0.5 * h * a1[3])
        a3 = f(x + 0.5 * h * a2[0], u + 0.5 * h * a2[1], I + 0.5 * h * a2[2], V + 0.5 * h * a2[3])
        a4 = f(x + h * a3[0], u + h * a3[1], I + h * a3[2], V + h * a3[3])
        x = x + h / 6.0 * (a1[0] + 2 * a2[0] + 2 * a3[0] + a4[0])
        u = u + h / 6.0 * (a1[1] + 2 * a2[1] + 2 * a3[1] + a4[1])
        I = I + h / 6.0 * (a1[2] + 2 * a2[2] + 2 * a3[2] + a4[2])
        V = V + h / 6.0 * (a1[3] + 2 * a2[3] + 2 * a3[3] + a4[3])
    return [np.array(e) for e in eff], rec, rec_x, rec_u, [np.array(v) for v in usp]


# ---------------------------------------------------------------------------
# Closed forms and exact references
def event_map(times, U, tau_D, tau_F):
    """Exact efficacies E_n = u+_n x-_n for arbitrary spike times."""
    x, u, out, tprev = 1.0, 0.0, [], None
    for t in times:
        if tprev is not None:
            Dn = t - tprev
            x = 1.0 - (1.0 - x) * np.exp(-Dn / tau_D)
            u = 0.0 if tau_F is None else u * np.exp(-Dn / tau_F)
        up = U if tau_F is None else u + U * (1.0 - u)
        out.append(up * x)
        x, u, tprev = x - up * x, up, t
    return np.array(out)


def u_closed(n, D, U, tau_F):
    if tau_F is None:
        return np.full(np.shape(n), U, float)
    eF = np.exp(-D / tau_F)
    uss = U / (1.0 - (1.0 - U) * eF)
    return uss + (U - uss) * ((1.0 - U) * eF) ** (np.asarray(n) - 1)


def steady_state(f, U, tau_D, tau_F):
    D = 1000.0 / f
    eD = np.exp(-D / tau_D)
    eF = 0.0 if tau_F is None else np.exp(-D / tau_F)
    uss = U / (1.0 - (1.0 - U) * eF)
    xss = (1.0 - eD) / (1.0 - (1.0 - uss) * eD)
    return uss, xss, uss * xss


def kernel(t):
    """Single PSP (mV) per pA of current jump, t >= 0."""
    t = np.asarray(t, float)
    return np.where(t >= 0, R_IN * TAU_S / (TAU_M - TAU_S)
                    * (np.exp(-np.maximum(t, 0) / TAU_M) - np.exp(-np.maximum(t, 0) / TAU_S)), 0.0)


def psp_ss(f, Ess):
    """Closed-form periodic steady-state PSP amplitude (peak - trough)."""
    D = 1000.0 / f
    qm, qs = np.exp(-D / TAU_M), np.exp(-D / TAU_S)
    pref = R_IN * A_SE * Ess * TAU_S / (TAU_M - TAU_S)

    def Vss(t):
        return pref * (np.exp(-t / TAU_M) / (1 - qm) - np.exp(-t / TAU_S) / (1 - qs))
    ts = np.log(TAU_M * (1 - qm) / (TAU_S * (1 - qs))) / (1 / TAU_S - 1 / TAU_M)
    assert 0 < ts < D, "steady-state PSP peak not inside the interval"
    return Vss(ts) - Vss(0.0), ts


def grid_peak(y, dt):
    """Maximum of a sampled curve, refined by a parabola through 3 points."""
    i = int(np.nanargmax(y))
    a, b, c = y[i - 1], y[i], y[i + 1]
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den
    return b - 0.25 * (a - c) * d, (i + d) * dt


def n_needed(f, p):
    """Spikes needed for the transient to decay below 1e-12."""
    D = 1000.0 / f
    U, tD, tF = p["U"], p["tau_D"], p["tau_F"]
    uss, _, _ = steady_state(f, U, tD, tF)
    rho = max((1 - U) * (0 if tF is None else np.exp(-D / tF)),
              (1 - uss) * np.exp(-D / tD), np.exp(-D / TAU_M), 1e-3)
    return max(6, int(np.ceil(np.log(1e-12) / np.log(rho))) + 3)


def demo_train(dt=DT):
    """8 spikes at 20 Hz from t = 20 ms, then a recovery spike 500 ms later."""
    times = list(20.0 + 50.0 * np.arange(8)) + [20.0 + 50.0 * 7 + 500.0]
    T = times[-1] + 100.0
    steps = np.array([int(round(t / dt)) for t in times])
    return np.array(times), steps, int(round(T / dt))


# ---------------------------------------------------------------------------
def validate():
    R = {}
    names = list(SYNAPSES)
    print(f"Integrator: RK4, dt = {DT} ms; tau_s = {TAU_S} ms, tau_m = {TAU_M} ms, "
          f"A = {A_SE} pA, R_in = {R_IN} GOhm")

    # 1. steady state vs frequency ------------------------------------------
    conds, spikes, wins, meta = [], [], [], []
    for name in names:
        p = SYNAPSES[name]
        for f in FREQS:
            D = 1000.0 / f
            sD = int(round(D / DT))
            assert abs(sD * DT - D) < 1e-9
            N = n_needed(f, p)
            st = np.arange(N + 1) * sD + int(round(10.0 / DT))
            conds.append(p)
            spikes.append(st)
            wins.append((st[-2], sD))  # V over the interval after spike N
            meta.append((name, f, N))
    n_steps = max(s[-1] for s in spikes) + 1
    t0 = _time.time()
    eff, rec, _, _, _ = simulate(conds, spikes, n_steps, windows=wins)
    print(f"\n1. Steady state: {len(conds)} trains, {n_steps} steps "
          f"({n_steps * DT / 1000:.1f} s simulated) in {_time.time() - t0:.1f} s")
    ss = {n: dict(f=[], E_sim=[], E_th=[], P_sim=[], P_th=[], tpk=[]) for n in names}
    errE, errP = [], []
    for c, (name, f, N) in enumerate(meta):
        p = SYNAPSES[name]
        _, _, Eth = steady_state(f, p["U"], p["tau_D"], p["tau_F"])
        Es = eff[c][N - 1]
        sD = int(round(1000.0 / f / DT))
        y = rec[c, :sD + 1]
        pk, tpk = grid_peak(y, DT)
        Ps = pk - y[0]
        Pth, tth = psp_ss(f, Eth)
        eE, eP = abs(Es / Eth - 1), abs(Ps / Pth - 1)
        errE.append(eE)
        errP.append(eP)
        d = ss[name]
        for k, v in zip(["f", "E_sim", "E_th", "P_sim", "P_th", "tpk"], [f, Es, Eth, Ps, Pth, tth]):
            d[k].append(v)
        print(f"   {name:12s} f = {f:5.1f} Hz  N = {N:4d}  E_ss {Es:.6f} vs {Eth:.6f} "
              f"(rel {eE:.1e})  P_ss {Ps:.5f} vs {Pth:.5f} mV (rel {eP:.1e})  t* {tth:.3f} ms")
    for name in names:
        for k in ss[name]:
            ss[name][k] = np.array(ss[name][k])
    mE, mP = max(errE), max(errP)
    print(f"   max relative error: E_ss {mE:.2e}, P_ss {mP:.2e} (tolerance 1e-5)")
    assert mE < 1e-5, f"steady-state efficacy error {mE:.2e} >= 1e-5"
    assert mP < 1e-5, f"steady-state PSP amplitude error {mP:.2e} >= 1e-5"
    R["ss"] = ss

    # 4. reported regimes --------------------------------------------------
    pd = SYNAPSES["depressing"]
    lam = 1000.0 / (pd["U"] * pd["tau_D"])
    d = ss["depressing"]
    prod = d["f"] / 1000.0 * d["E_sim"] * pd["tau_D"]
    print(f"\n4. Depressing: limiting frequency 1/(U tau_D) = {lam:.2f} Hz; "
          f"f E_ss tau_D = {prod[d['f'] == 10][0]:.4f} (10 Hz), {prod[d['f'] == 40][0]:.4f} (40 Hz), "
          f"{prod[-1]:.4f} (100 Hz) -> 1 (not asserted)")
    _, _, Einf = steady_state(1e5, pd["U"], pd["tau_D"], None)
    print(f"   closed form at 1e5 Hz: f E_ss tau_D = {100 * Einf * pd['tau_D']:.6f}")
    pf = SYNAPSES["facilitating"]
    res = minimize_scalar(lambda lf: -steady_state(10 ** lf, pf["U"], pf["tau_D"], pf["tau_F"])[2],
                          bounds=(-1, 3), method="bounded", options=dict(xatol=1e-10))
    fpk = 10 ** res.x
    Epk = -res.fun
    fs = ss["facilitating"]
    i = int(np.argmax(fs["E_sim"]))
    print(f"   Facilitating: closed-form E_ss peaks at {fpk:.3f} Hz (E_ss {Epk:.4f}, first-spike E = "
          f"{pf['U']}); largest simulated E_ss on the grid at {fs['f'][i]:g} Hz ({fs['E_sim'][i]:.4f}) "
          f"(not asserted)")
    R["lim"] = (lam, fpk, Epk)

    # 2. transient vs closed form / event map ----------------------------------
    print("\n2. Transients vs closed form and exact event map")
    conds, spikes, refs = [], [], []
    times_d, steps_d, n_d = demo_train()
    for name in names:
        p = SYNAPSES[name]
        for f in (20.0, 50.0):
            D = 1000.0 / f
            times = list(20.0 + D * np.arange(15)) + [20.0 + D * 14 + 500.0]
            conds.append(p)
            spikes.append(np.array([int(round(t / DT)) for t in times]))
            refs.append((name, f, np.array(times)))
    n_steps = max(s[-1] for s in spikes) + 1
    eff, _, _, _, usim = simulate(conds, spikes, n_steps, windows=[(0, 0)] * len(conds))
    worst = 0.0
    R["trans"] = []
    for c, (name, f, times) in enumerate(refs):
        p = SYNAPSES[name]
        Em = event_map(times, p["U"], p["tau_D"], p["tau_F"])
        n = np.arange(1, 16)
        uc = u_closed(n, 1000.0 / f, p["U"], p["tau_F"])
        e1 = np.max(np.abs(eff[c] / Em - 1))
        if p["tau_F"] is None:
            # pure depression: x-_n = x_ss + (1 - x_ss) [(1 - U) e_D]^(n-1)
            _, xss, _ = steady_state(f, p["U"], p["tau_D"], None)
            r = (1 - p["U"]) * np.exp(-1000.0 / f / p["tau_D"])
            Ec = p["U"] * (xss + (1 - xss) * r ** (n - 1))
            e2 = np.max(np.abs(eff[c][:15] / Ec - 1))
            lab = "E_n vs closed form"
        else:
            e2 = np.max(np.abs(usim[c][:15] / uc - 1))
            lab = "u+_n vs closed form"
        worst = max(worst, e1, e2)
        print(f"   {name:12s} {f:4.0f} Hz: E_1..E_15 {Em[0]:.4f} -> {Em[14]:.4f}, recovery "
              f"{Em[15]:.4f}; max rel error E_n vs map {e1:.1e}, {lab} {e2:.1e}")
        R["trans"].append((name, f, eff[c], Em))
    print(f"   max relative error {worst:.2e} (tolerance 1e-8)")
    assert worst < 1e-8, f"transient efficacy error {worst:.2e} >= 1e-8"

    # 3. voltage vs exact superposition; RK4 order ------------------------------
    print("\n3. V(t) vs exact superposition of PSP kernels")
    errs, dts = [], np.array([0.4, 0.2, 0.1, 0.05])
    R["demo"] = {}
    for dt in dts:
        times, steps, n = demo_train(dt)
        conds = [SYNAPSES[nm] for nm in names]
        eff, rec, rx, ru, _ = simulate(conds, [steps, steps], n, dt=dt)
        t = np.arange(n + 1) * dt
        e_dt = 0.0
        for c, nm in enumerate(names):
            p = SYNAPSES[nm]
            Em = event_map(times, p["U"], p["tau_D"], p["tau_F"])
            Vx = E_L + A_SE * sum(Em[k] * kernel(t - times[k]) for k in range(len(times)))
            e_dt = max(e_dt, np.max(np.abs(rec[c, :n + 1] - Vx)))
            if dt == DT:
                R["demo"][nm] = (t, rec[c, :n + 1], Vx, rx[c, :n + 1], ru[c, :n + 1], times, Em)
        errs.append(e_dt)
        print(f"   dt = {dt:5.3f} ms: max |V - V_exact| = {e_dt:.3e} mV")
    errs = np.array(errs)
    order = np.polyfit(np.log(dts), np.log(errs), 1)[0]
    print(f"   RK4 order {order:.3f} (tolerance 4 +- 0.3); error at dt = {DT}: {errs[-1]:.2e} mV "
          f"(tolerance 1e-6)")
    assert errs[-1] < 1e-6, "V error vs exact superposition >= 1e-6 mV"
    assert abs(order - 4.0) < 0.3, f"RK4 order {order:.3f} not 4 +- 0.3"
    R["order"] = (dts, errs, order)

    print("\nAll assertions passed.")
    return R


# ---------------------------------------------------------------------------
def plot(R):
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.5))
    cols = {"depressing": "C0", "facilitating": "C3"}

    # A: voltage traces
    ax = axes[0, 0]
    for nm, (t, V, Vx, _, _, times, Em) in R["demo"].items():
        ax.plot(t, V, color=cols[nm], lw=1.3, label=f"{nm} (RK4)")
        ax.plot(t[::40], Vx[::40], "k.", ms=2)
    ax.plot([], [], "k.", label="exact kernel superposition")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_title("8 spikes at 20 Hz + recovery spike 500 ms later")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # B: x and u
    ax = axes[0, 1]
    for nm, (t, _, _, x, u, times, Em) in R["demo"].items():
        ax.plot(t, x, color=cols[nm], lw=1.3, label=f"{nm}: x (resources)")
        ax.plot(t, u, color=cols[nm], lw=1.0, ls="--", label=f"{nm}: u (utilisation)")
        ax.plot(times, Em, "o", color=cols[nm], ms=4, mfc="none")
    ax.plot([], [], "ko", mfc="none", ms=4, label="efficacy u⁺x⁻ (exact map)")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("x, u, E_n")
    ax.set_title("Synaptic state")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # C: transient efficacies
    ax = axes[0, 2]
    for nm, f, E, Em in R["trans"]:
        n = np.arange(1, len(E) + 1)
        ls = "-" if f == 20 else ":"
        ax.plot(n, Em, color=cols[nm], ls=ls, lw=1.2, label=f"{nm}, {f:g} Hz (map)")
        ax.plot(n, E, "o", color=cols[nm], ms=3.5, mfc="none")
    ax.set_xlabel("spike number (16 = recovery spike after 500 ms)")
    ax.set_ylabel("efficacy E_n = u⁺x⁻")
    ax.set_title("Transients: RK4 (o) vs exact event map (—)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3)

    # D: steady-state efficacy vs f
    ax = axes[1, 0]
    ff = np.logspace(np.log10(0.3), np.log10(150), 300)
    for nm, d in R["ss"].items():
        p = SYNAPSES[nm]
        ax.loglog(ff, steady_state(ff, p["U"], p["tau_D"], p["tau_F"])[2], color=cols[nm], lw=1.3,
                  label=f"{nm}: closed form")
        ax.loglog(d["f"], d["E_sim"], "o", color=cols[nm], ms=4, mfc="none")
    lam, fpk, Epk = R["lim"]
    pd = SYNAPSES["depressing"]
    ax.loglog(ff, 1000.0 / (ff * pd["tau_D"]), "k--", lw=0.8, label="1/(f τ_D)")
    ax.axvline(lam, color="C0", ls=":", lw=1, label=f"1/(U τ_D) = {lam:.1f} Hz")
    ax.axvline(fpk, color="C3", ls=":", lw=1, label=f"facilitating peak {fpk:.1f} Hz")
    ax.set_ylim(5e-3, 1)
    ax.set_xlabel("presynaptic rate f (Hz)")
    ax.set_ylabel("steady-state efficacy E_ss")
    ax.set_title("Steady-state efficacy: RK4 (o) vs closed form (—)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # E: steady-state PSP amplitude vs f
    ax = axes[1, 1]
    for nm, d in R["ss"].items():
        p = SYNAPSES[nm]
        Pf = [psp_ss(f, steady_state(f, p["U"], p["tau_D"], p["tau_F"])[2])[0] for f in ff[ff <= 100]]
        ax.loglog(ff[ff <= 100], Pf, color=cols[nm], lw=1.3, label=f"{nm}: closed form P_ss")
        ax.loglog(d["f"], d["P_sim"], "o", color=cols[nm], ms=4, mfc="none")
        kp = A_SE * kernel(TAU_S * TAU_M / (TAU_M - TAU_S) * np.log(TAU_M / TAU_S))
        ax.loglog(d["f"], kp * d["E_th"], "x", color=cols[nm], ms=4,
                  label=f"{nm}: E_ss × single-PSP peak")
    ax.set_xlabel("presynaptic rate f (Hz)")
    ax.set_ylabel("steady-state PSP amplitude (mV)")
    ax.set_title("Steady-state PSP (peak − trough): RK4 (o) vs closed form (—)")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # F: errors and convergence
    ax = axes[1, 2]
    dts, errs, order = R["order"]
    ax.loglog(dts, errs, "o-", color="k", label=f"V vs exact superposition: slope {order:.2f}")
    ax.loglog(dts, errs[-1] * (dts / dts[-1]) ** 4, "k--", lw=0.8, label="dt⁴")
    for nm, d in R["ss"].items():
        ax.loglog(d["f"] / 100.0, np.abs(d["P_sim"] / d["P_th"] - 1) + 1e-16, "s", color=cols[nm],
                  ms=5 if nm == "depressing" else 3, mfc="none" if nm == "depressing" else cols[nm],
                  label=f"{nm}: |P_ss rel. error| vs f/100 Hz")
    ax.set_xlabel("dt (ms)   /   f / 100 Hz")
    ax.set_ylabel("max |error| (mV)   /   relative error")
    ax.set_title("RK4 convergence and steady-state PSP error")
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "short_term_plasticity.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print("No RNG: the model is deterministic.")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
