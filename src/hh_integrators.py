"""Hodgkin-Huxley with a proper integrator: forward Euler vs RK4 vs solve_ivp.

Equations (Hodgkin & Huxley 1952, modern convention with rest at -65 mV, the
same parameters as src/hodgkin_huxley.py):

    C_m dV/dt = I(t) - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K) - g_L (V - E_L)
    dx/dt     = alpha_x(V) (1 - x) - beta_x(V) x,        x = m, h, n

    alpha_m = 0.1 (V + 40) / (1 - exp(-(V + 40)/10)),  beta_m = 4 exp(-(V + 65)/18)
    alpha_h = 0.07 exp(-(V + 65)/20),                  beta_h = 1 / (1 + exp(-(V + 35)/10))
    alpha_n = 0.01 (V + 55) / (1 - exp(-(V + 55)/10)), beta_n = 0.125 exp(-(V + 65)/80)

C_m = 1 uF/cm^2, g_Na = 120, g_K = 36, g_L = 0.3 mS/cm^2, E_Na = 50,
E_K = -77, E_L = -54.4 mV. Stimulus: the protocol of hodgkin_huxley.py, a
10 uA/cm^2 step on [10, 110) ms, 150 ms in total, starting from the
steady state at -65 mV (7 spikes).

Integrators (both written out here, fixed step dt):
  * forward Euler:   y_{k+1} = y_k + dt f(t_k, y_k)                 (order 1)
  * classical RK4:   y_{k+1} = y_k + dt (k1 + 2 k2 + 2 k3 + k4) / 6  (order 4)
Every dt used is 0.1 / 2^j ms, so the step edges at 10 and 110 ms fall on
grid nodes and the current is constant inside each step (it is evaluated
at the step midpoint). Without this, the discontinuity inside a step would
cap RK4 at first order.

Validation (ROADMAP note: order-of-convergence plot, error vs dt on log-log
axes, slope ~1 for Euler and ~4 for RK4):

1. Reference solution. scipy's DOP853 with rtol = atol = 1e-13, integrated
   piecewise over [0, 10], [10, 110], [110, 150] so that no step straddles
   a current jump. Its accuracy is checked against two independent
   solvers (Radau, rtol = atol = 1e-13, and RK45, rtol = atol = 1e-12):
   the largest disagreement must be below 2e-8 mV, at least 20x below the
   smallest fixed-step error used in the fits.

2. Order of convergence. Error = max |V - V_ref| over 301 check points
   every 0.5 ms on [0, 150] ms. Euler for dt = 0.05 ... 0.0015625 ms and RK4
   for dt = 0.05 ... 0.003125 ms (RK4 is unstable at dt = 0.1). The
   least-squares slope of log(error) vs log(dt) must be within 0.15 of the
   formal order (1 and 4), and the slope between the two finest steps
   within 0.05 (Euler) and 0.1 (RK4).

3. Spike times. Upward crossings of 0 mV, located by cubic Hermite
   interpolation on each fixed-step trajectory and by event detection on
   the reference. At the repository's default dt = 0.01 ms the spike count
   must equal the reference (7) and the spike-time errors are reported,
   for comparison with the dt = 0.01 vs 0.001 Euler check in
   check_hh_timestep.py (max 0.019 ms).

Also reported, not asserted: the largest stable dt of each fixed-step
method on this protocol (bisection on "the run stays finite and |V| < 500
mV"), and the cost in right-hand-side evaluations of each method,
including solve_ivp RK45 and DOP853 at rtol = 1e-3 ... 1e-10 (atol = rtol
* 1e-3), as a work-precision diagram.

No random numbers are used. scipy is used only for the reference and the
adaptive comparison solvers.
"""

from pathlib import Path
import math
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
C_m = 1.0                        # uF/cm^2
g_Na, g_K, g_L = 120.0, 36.0, 0.3  # mS/cm^2
E_Na, E_K, E_L = 50.0, -77.0, -54.4  # mV

T = 150.0                        # ms
I_AMP, T_ON, T_OFF = 10.0, 10.0, 110.0  # uA/cm^2, ms, ms
PIECES = [(0.0, T_ON, 0.0), (T_ON, T_OFF, I_AMP), (T_OFF, T, 0.0)]
CHECK_EVERY = 0.5                # ms between error check points
T_CHECK = np.arange(0.0, T + 1e-9, CHECK_EVERY)

DT_EULER = [0.1 / 2 ** j for j in range(1, 7)]   # 0.05 ... 0.0015625
DT_RK4 = [0.1 / 2 ** j for j in range(1, 6)]     # 0.05 ... 0.003125
DT_DEFAULT = 0.01                # ms, the repository's default step (10 and 110 ms are still grid nodes)
ADAPTIVE_RTOL = [10.0 ** -k for k in range(3, 11)]

REF_TOL = 1e-13
REF_FLOOR_TOL = 2e-8             # mV, reference cross-check
SLOPE_FIT_TOL = 0.15
SLOPE_FINE_TOL = {"euler": 0.05, "rk4": 0.10}


# ------------------------------------------------------------------- model
def rates(V):
    am = 0.1 * (V + 40.0) / (1.0 - math.exp(-(V + 40.0) / 10.0))
    bm = 4.0 * math.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * math.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + math.exp(-(V + 35.0) / 10.0))
    an = 0.01 * (V + 55.0) / (1.0 - math.exp(-(V + 55.0) / 10.0))
    bn = 0.125 * math.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def rhs(V, m, h, n, I):
    am, bm, ah, bh, an, bn = rates(V)
    dV = (I - g_Na * m ** 3 * h * (V - E_Na) - g_K * n ** 4 * (V - E_K)
          - g_L * (V - E_L)) / C_m
    return (dV, am * (1.0 - m) - bm * m, ah * (1.0 - h) - bh * h,
            an * (1.0 - n) - bn * n)


def rhs_np(t, y, I):
    """Same right-hand side for solve_ivp; returns inf instead of raising."""
    V, m, h, n = y
    with np.errstate(all="ignore"):
        am = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0))
        bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
        ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
        bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        an = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0))
        bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
        dV = (I - g_Na * m ** 3 * h * (V - E_Na) - g_K * n ** 4 * (V - E_K)
              - g_L * (V - E_L)) / C_m
        return [dV, am * (1 - m) - bm * m, ah * (1 - h) - bh * h,
                an * (1 - n) - bn * n]


def current(t):
    return I_AMP if T_ON <= t < T_OFF else 0.0


def resting_state(V=-65.0):
    am, bm, ah, bh, an, bn = rates(V)
    return (V, am / (am + bm), ah / (ah + bh), an / (an + bn))


Y0 = resting_state()


def euler_step(y, I, dt):
    d = rhs(*y, I)
    return tuple(a + dt * b for a, b in zip(y, d))


def rk4_step(y, I, dt):
    k1 = rhs(*y, I)
    k2 = rhs(*(a + 0.5 * dt * b for a, b in zip(y, k1)), I)
    k3 = rhs(*(a + 0.5 * dt * b for a, b in zip(y, k2)), I)
    k4 = rhs(*(a + dt * b for a, b in zip(y, k3)), I)
    return tuple(a + dt / 6.0 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
                 for a, b1, b2, b3, b4 in zip(y, k1, k2, k3, k4))


STEPPERS = {"euler": euler_step, "rk4": rk4_step}
EVALS_PER_STEP = {"euler": 1, "rk4": 4}


# --------------------------------------------------------------- simulate
def simulate(method, dt, t_end=T, keep=True):
    """Fixed-step run from rest. Returns (t, Y) with Y of shape (N+1, 4),
    or None if the run overflows or leaves |V| < 500 mV."""
    step = STEPPERS[method]
    N = int(round(t_end / dt))
    y = Y0
    Y = np.empty((N + 1, 4)) if keep else None
    if keep:
        Y[0] = y
    try:
        for k in range(N):
            t = k * dt
            y = step(y, current(t + 0.5 * dt), dt)
            if not abs(y[0]) < 500.0:
                return None
            if keep:
                Y[k + 1] = y
    except (OverflowError, ZeroDivisionError):
        return None
    return (np.arange(N + 1) * dt, Y) if keep else True


def at_checks(t, Y, dt):
    stride = int(round(CHECK_EVERY / dt))
    assert abs(CHECK_EVERY / dt - stride) < 1e-9
    return Y[::stride, 0]


def reference(method="DOP853", rtol=REF_TOL, atol=REF_TOL, dense=False):
    """Piecewise solve_ivp over the three constant-current pieces."""
    y = np.array(Y0)
    V, nfev, sols = [], 0, []
    for a, b, I in PIECES:
        te = T_CHECK[(T_CHECK >= a - 1e-12) & (T_CHECK <= b + 1e-12)]
        s = solve_ivp(rhs_np, (a, b), y, method=method, rtol=rtol, atol=atol,
                      t_eval=te, args=(I,), dense_output=dense)
        if s.status != 0:
            return None, nfev, None
        nfev += s.nfev
        V.append(s.y[0] if a == 0.0 else s.y[0][1:])
        y = s.y[:, -1]
        sols.append(s)
    return np.concatenate(V), nfev, sols


def reference_spikes():
    def up(t, y, I):
        return y[0]
    up.direction = 1.0
    y = np.array(Y0)
    ts = []
    for a, b, I in PIECES:
        s = solve_ivp(rhs_np, (a, b), y, method="DOP853", rtol=REF_TOL,
                      atol=REF_TOL, events=up, args=(I,))
        ts.extend(s.t_events[0])
        y = s.y[:, -1]
    return np.array(ts)


def hermite_spikes(t, Y, dt):
    """Upward 0 mV crossings by cubic Hermite interpolation using dV/dt."""
    V = Y[:, 0]
    idx = np.where((V[:-1] < 0.0) & (V[1:] >= 0.0))[0]
    out = []
    for k in idx:
        I = current(t[k] + 0.5 * dt)
        d0 = rhs(*Y[k], I)[0] * dt
        d1 = rhs(*Y[k + 1], I)[0] * dt
        p0, p1 = V[k], V[k + 1]
        lo, hi = 0.0, 1.0
        for _ in range(60):
            s = 0.5 * (lo + hi)
            h00, h10 = 2 * s ** 3 - 3 * s ** 2 + 1, s ** 3 - 2 * s ** 2 + s
            h01, h11 = -2 * s ** 3 + 3 * s ** 2, s ** 3 - s ** 2
            if h00 * p0 + h10 * d0 + h01 * p1 + h11 * d1 < 0.0:
                lo = s
            else:
                hi = s
        out.append(t[k] + 0.5 * (lo + hi) * dt)
    return np.array(out)


def stability_limit(method, lo, hi, n_iter=16):
    """Largest dt with a finite run; lo must be stable, hi unstable."""
    assert simulate(method, lo, keep=False) and simulate(method, hi, keep=False) is None
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if simulate(method, mid, keep=False):
            lo = mid
        else:
            hi = mid
    return lo, hi


def slope(dts, errs):
    return np.polyfit(np.log(dts), np.log(errs), 1)[0]


# --------------------------------------------------------------- validate
def validate():
    R = {}

    # 1. reference and its cross-check
    V_ref, nfev_ref, _ = reference()
    floors = {}
    for name, meth, tol in [("Radau 1e-13", "Radau", 1e-13),
                            ("RK45 1e-12", "RK45", 1e-12)]:
        V2, n2, _ = reference(meth, tol, tol)
        floors[name] = np.abs(V2 - V_ref).max()
    floor = max(floors.values())
    print(f"[1] Reference DOP853 rtol=atol=1e-13: {nfev_ref} RHS evaluations")
    for k, v in floors.items():
        print(f"    max |V_ref - V_{k}| = {v:.2e} mV")
    assert floor < REF_FLOOR_TOL, f"reference cross-check {floor:.2e} >= {REF_FLOOR_TOL}"
    R.update(V_ref=V_ref, floor=floor, nfev_ref=nfev_ref)

    # 2. order of convergence
    conv = {}
    for method, dts in [("euler", DT_EULER), ("rk4", DT_RK4)]:
        errs, rms, nfev = [], [], []
        for dt in dts:
            out = simulate(method, dt)
            assert out is not None, f"{method} unstable at dt = {dt}"
            e = np.abs(at_checks(*out, dt) - V_ref)
            errs.append(e.max())
            rms.append(np.sqrt(np.mean(e ** 2)))
            nfev.append(EVALS_PER_STEP[method] * int(round(T / dt)))
        errs, rms = np.array(errs), np.array(rms)
        p_fit = slope(dts, errs)
        p_rms = slope(dts, rms)
        p_fine = math.log(errs[-2] / errs[-1]) / math.log(dts[-2] / dts[-1])
        local = np.log(errs[:-1] / errs[1:]) / np.log(2.0)
        order = 1 if method == "euler" else 4
        print(f"[2] {method.upper()}: dt (ms) -> max |V error| (mV)")
        for dt, e in zip(dts, errs):
            print(f"      {dt:<10.7g} {e:.4e}")
        print(f"    local slopes {np.array2string(local, precision=3)}; "
              f"LS slope {p_fit:.3f} (RMS error: {p_rms:.3f}); finest pair {p_fine:.3f}; "
              f"formal order {order}")
        assert abs(p_fit - order) < SLOPE_FIT_TOL, f"{method} LS slope {p_fit:.3f}"
        assert abs(p_fine - order) < SLOPE_FINE_TOL[method], f"{method} fine slope {p_fine:.3f}"
        assert errs.min() > 20 * floor, "fixed-step error too close to reference floor"
        conv[method] = dict(dts=np.array(dts), errs=errs, rms=rms, nfev=np.array(nfev),
                            p_fit=p_fit, p_fine=p_fine, local=local)
    R["conv"] = conv

    # 3. spike times at the default dt = 0.01 ms
    ts_ref = reference_spikes()
    R["ts_ref"] = ts_ref
    print(f"[3] Reference spike times (ms): {np.array2string(ts_ref, precision=4)}")
    traj = {}
    for method in ("euler", "rk4"):
        t, Y = simulate(method, DT_DEFAULT)
        ts = hermite_spikes(t, Y, DT_DEFAULT)
        assert len(ts) == len(ts_ref), f"{method} dt=0.01: {len(ts)} spikes vs {len(ts_ref)}"
        d = ts - ts_ref
        print(f"    {method.upper()} dt = 0.01 ms: {len(ts)} spikes, spike-time error "
              f"{np.array2string(d, precision=5)} ms, max |error| {np.abs(d).max():.2e} ms")
        traj[method] = dict(t=t, Y=Y, ts=ts, dts=d)
    t, Y = simulate("euler", 0.001)
    d = hermite_spikes(t, Y, 0.001) - ts_ref
    print(f"    EULER dt = 0.001 ms: max |spike-time error| {np.abs(d).max():.2e} ms")
    R["euler_0001_dts"] = np.abs(d).max()
    R["traj"] = traj
    tf = np.arange(0.0, T + 1e-9, DT_DEFAULT)
    Vf = []
    y = np.array(Y0)
    for a, b, I in PIECES:
        te = tf[(tf >= a - 1e-9) & (tf <= b + 1e-9)]
        s = solve_ivp(rhs_np, (a, b), y, method="DOP853", rtol=REF_TOL,
                      atol=REF_TOL, t_eval=te, args=(I,))
        Vf.append(s.y[0] if a == 0.0 else s.y[0][1:])
        y = s.y[:, -1]
    R["t_fine"], R["V_fine"] = tf, np.concatenate(Vf)

    # reported: stability limits
    stab = {"euler": stability_limit("euler", 0.05, 0.2),
            "rk4": stability_limit("rk4", 0.05, 0.1)}
    for k, (lo, hi) in stab.items():
        print(f"    {k.upper()} largest stable dt on this protocol: {lo:.6f} (stable) / {hi:.6f} ms (unstable)")
    R["stab"] = stab

    # reported: work-precision of adaptive solvers
    wp = {}
    for meth in ("RK45", "DOP853"):
        errs, nf = [], []
        for rtol in ADAPTIVE_RTOL:
            V2, n2, _ = reference(meth, rtol, rtol * 1e-3)
            errs.append(np.abs(V2 - V_ref).max() if V2 is not None else np.nan)
            nf.append(n2)
        wp[meth] = (np.array(nf), np.array(errs))
        print(f"    solve_ivp {meth}: rtol -> (RHS evaluations, max |V error| mV): " +
              ", ".join(f"{r:.0e}: ({n}, {e:.2e})" for r, n, e in zip(ADAPTIVE_RTOL, nf, errs)))
    R["wp"] = wp
    return R


# ------------------------------------------------------------------- plot
def plot(R):
    c = {"euler": "#c0392b", "rk4": "#2471a3", "RK45": "#7d3c98", "DOP853": "#1e8449"}
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))

    a = ax[0, 0]
    a.plot(R["t_fine"], R["V_fine"], color="k", lw=1.6, label="reference (DOP853, tol 1e-13)")
    for m, lab in (("euler", "Euler"), ("rk4", "RK4")):
        out = simulate(m, 0.05)
        a.plot(out[0], out[1][:, 0], color=c[m], lw=0.9, ls="--", label=f"{lab}, dt = 0.05 ms")
    a.set_xlim(8, 60)
    a.set_xlabel("time (ms)"); a.set_ylabel("V (mV)")
    a.set_title("10 µA/cm² step from 10 ms: coarse steps vs reference")
    a.legend(fontsize=8, loc="upper right")

    a = ax[0, 1]
    for m, lab, order in (("euler", "Euler", 1), ("rk4", "RK4", 4)):
        d = R["conv"][m]
        a.loglog(d["dts"], d["errs"], "o-", color=c[m],
                 label=f"{lab}: LS slope {d['p_fit']:.2f}")
        guide = d["errs"][-1] * (d["dts"] / d["dts"][-1]) ** order
        a.loglog(d["dts"], guide, ":", color=c[m], lw=1, label=f"slope {order}")
    a.axhline(R["floor"], color="gray", lw=0.8, ls="--", label="reference cross-check floor")
    a.set_xlabel("dt (ms)"); a.set_ylabel("max |V − V_ref| (mV)")
    a.set_title("Order of convergence")
    a.legend(fontsize=8)

    a = ax[0, 2]
    for m, lab in (("euler", "Euler"), ("rk4", "RK4")):
        d = R["conv"][m]
        a.loglog(d["nfev"], d["errs"], "o-", color=c[m], label=f"{lab} (fixed dt)")
    for m in ("RK45", "DOP853"):
        nf, e = R["wp"][m]
        ok = np.isfinite(e)
        a.loglog(nf[ok], e[ok], "s-", color=c[m], label=f"solve_ivp {m}, rtol 1e-3…1e-10")
    a.axhline(R["floor"], color="gray", lw=0.8, ls="--")
    a.set_xlabel("right-hand-side evaluations"); a.set_ylabel("max |V − V_ref| (mV)")
    a.set_title("Work–precision")
    a.legend(fontsize=8, loc="lower left")

    a = ax[1, 0]
    for m, lab in (("euler", "Euler"), ("rk4", "RK4")):
        tr = R["traj"][m]
        err = np.abs(tr["Y"][:, 0] - R["V_fine"])
        a.semilogy(tr["t"], np.maximum(err, 1e-14), color=c[m], lw=0.7, label=f"{lab}, dt = 0.01 ms")
    a.set_xlabel("time (ms)"); a.set_ylabel("|V − V_ref| (mV)")
    a.set_title("Pointwise error at the default dt = 0.01 ms")
    a.set_ylim(1e-10, 1e2)
    a.legend(fontsize=8)

    a = ax[1, 1]
    k = np.arange(1, len(R["ts_ref"]) + 1)
    for m, lab in (("euler", "Euler"), ("rk4", "RK4")):
        a.semilogy(k, np.abs(R["traj"][m]["dts"]), "o-", color=c[m], label=f"{lab}, dt = 0.01 ms")
    a.set_xlabel("spike number"); a.set_ylabel("|t_spike − t_ref| (ms)")
    a.set_title("Spike-time error vs reference")
    a.legend(fontsize=8)

    a = ax[1, 2]
    for m, lab in (("euler", "Euler"), ("rk4", "RK4")):
        d = R["conv"][m]
        a.semilogx(0.5 * (d["dts"][:-1] + d["dts"][1:]), d["local"], "o-", color=c[m], label=lab)
        a.axhline(1 if m == "euler" else 4, color=c[m], ls=":", lw=1)
    a.set_ylim(0, 5)
    a.set_xlabel("dt (ms), midpoint of each halving"); a.set_ylabel("local order log₂(e(dt)/e(dt/2))")
    a.set_title("Local order per halving of dt")
    a.legend(fontsize=8)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "hh_integrators.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
