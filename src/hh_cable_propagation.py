"""Propagating action potential on an active Hodgkin-Huxley cable.

Equations (Hodgkin & Huxley 1952, Part V, the propagated action potential;
modern convention with rest at -65 mV, same membrane as src/hodgkin_huxley.py):

    C_m dV/dt = (a / 2R_i) d2V/dx2 - g_Na m^3 h (V - E_Na) - g_K n^4 (V - E_K)
                - g_L (V - E_L) + I_stim(x, t)
    dx/dt     = phi [alpha_x(V) (1 - x) - beta_x(V) x],   x = m, h, n
    phi       = 3^((T - 6.3)/10)

    alpha_m = 0.1 (V + 40) / (1 - exp(-(V + 40)/10)),  beta_m = 4 exp(-(V + 65)/18)
    alpha_h = 0.07 exp(-(V + 65)/20),                  beta_h = 1 / (1 + exp(-(V + 35)/10))
    alpha_n = 0.01 (V + 55) / (1 - exp(-(V + 55)/10)), beta_n = 0.125 exp(-(V + 65)/80)

C_m = 1 uF/cm^2, g_Na = 120, g_K = 36, g_L = 0.3 mS/cm^2, E_Na = 50,
E_K = -77, E_L = -54.4 mV. The cable values are those of the 1952 paper's
propagation calculation: fibre radius a = 238 um, axoplasm resistivity
R_i = 35.4 Ohm cm, temperature T = 18.5 C (phi = 3.82). The paper solved
the travelling-wave ODE by trial and error and obtained a conduction
velocity of 18.8 m/s (measured on the same axon: 21.2 m/s). Here the full
PDE is solved instead, which the paper could not do by hand.

Geometry and stimulus: an unbranched cable of length L = 8 cm (about
7.5 length constants, lambda = sqrt(a R_m / 2R_i) = 1.06 cm) with sealed
ends (dV/dx = 0), starting from the resting state at -65 mV. A current of
1000 uA/cm^2 is injected over 0 <= x < 0.3 cm for 0.2 ms.

Discretisation: method of lines. Nodes x_i = i dx, i = 0..N, with the
second-order central difference for d2V/dx2 and mirror ghost nodes at the
sealed ends. Time integrator: classical RK4 written out, fixed step
dt = 0.25 dx^2 C_m / D with D = a / 2R_i (a quarter of the forward-Euler
diffusion limit, well inside the RK4 stability region), so dt scales with
dx^2 and the temporal error (order 4 in dt = order 8 in dx) is negligible
against the spatial error (order 2 in dx).

Velocity: the upward 0 mV crossing time of every node is found online by
linear interpolation between steps; the velocity is the least-squares
slope of x against crossing time over the middle of the cable,
0.3 L <= x <= 0.7 L, away from the stimulus and the sealed far end.

Validation:
1. Convergence under refinement. dx = 0.04, 0.02, 0.01, 0.005 cm (dt
   following dx^2). Asserted: observed order of the velocity error
   log2((c1 - c2)/(c2 - c3)) on the three finest grids within 2 +/- 0.5,
   the finest grid within 0.5% of the Richardson extrapolate, and halving
   dt alone at dx = 0.01 cm changes the velocity by less than 0.05%.
2. Published number. The Richardson-extrapolated velocity agrees with
   Hodgkin & Huxley's computed 18.8 m/s within 10%.
3. Diameter scaling. For radii a0 x {1/8, 1/4, 1/2, 1, 2} at fixed
   dx = 0.01 cm (cable length and stimulus region scaled by sqrt(a/a0)),
   the log-log slope of velocity against diameter is 0.5 +/- 0.05.

No random numbers are used. SciPy is not needed.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
C_m = 1.0                            # uF/cm^2
g_Na, g_K, g_L = 120.0, 36.0, 0.3    # mS/cm^2
E_Na, E_K, E_L = 50.0, -77.0, -54.4  # mV
V_REST = -65.0                       # mV
A0 = 0.0238                          # cm, fibre radius (238 um)
R_I = 35.4                           # Ohm cm, axoplasm resistivity
TEMP = 18.5                          # C
PHI = 3.0 ** ((TEMP - 6.3) / 10.0)

L0 = 8.0                             # cm, cable length at a = A0
STIM_AMP = 1000.0                    # uA/cm^2
STIM_X = 0.3                         # cm, stimulated region at a = A0
STIM_T = 0.2                         # ms
T_END = 6.0                          # ms, hard cap; runs stop once measured
CFL = 0.25                           # dt = CFL dx^2 C_m / D
V_CROSS = 0.0                        # mV
WINDOW = (0.3, 0.7)                  # fraction of L used for the velocity fit

DXS = [0.04, 0.02, 0.01, 0.005]      # cm, refinement study
DX_SCALE = 0.01                      # cm, diameter sweep
A_FACTORS = [0.125, 0.25, 0.5, 1.0, 2.0]

HH_VELOCITY = 18.8                   # m/s, Hodgkin & Huxley (1952) computed
HH_MEASURED = 21.2                   # m/s, measured on the same fibre
TOL_PUB = 0.10
TOL_ORDER = 0.5
TOL_RICH = 0.005
TOL_DT = 0.0005
TOL_SLOPE = 0.05


# ------------------------------------------------------------------- model
def vtrap(x, k):
    """x / (1 - exp(-x/k)), with the removable singularity at x = 0 filled."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-6
    xs = np.where(small, 1.0, x)
    return np.where(small, k * (1.0 + x / (2.0 * k)), xs / (1.0 - np.exp(-xs / k)))


def rates(V):
    am = 0.1 * vtrap(V + 40.0, 10.0)
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    an = 0.01 * vtrap(V + 55.0, 10.0)
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def diffusion_coefficient(a):
    """a / (2 R_i) in uA/cm^2 per mV/cm^2 (S -> mA, x1000 -> uA)."""
    return 1e3 * a / (2.0 * R_I)


def laplacian(V, dx):
    out = np.empty_like(V)
    out[1:-1] = V[:-2] - 2.0 * V[1:-1] + V[2:]
    out[0] = 2.0 * (V[1] - V[0])
    out[-1] = 2.0 * (V[-2] - V[-1])
    return out / dx ** 2


def rhs(y, D, dx, I_ext):
    V, m, h, n = y
    am, bm, ah, bh, an, bn = rates(V)
    I_ion = (g_Na * m ** 3 * h * (V - E_Na) + g_K * n ** 4 * (V - E_K)
             + g_L * (V - E_L))
    dV = (D * laplacian(V, dx) - I_ion + I_ext) / C_m
    return np.array([dV, PHI * (am * (1.0 - m) - bm * m),
                     PHI * (ah * (1.0 - h) - bh * h), PHI * (an * (1.0 - n) - bn * n)])


def resting_state(n_nodes):
    am, bm, ah, bh, an, bn = rates(np.array(V_REST))
    y = np.array([V_REST, am / (am + bm), ah / (ah + bh), an / (an + bn)])
    return np.repeat(y[:, None], n_nodes, axis=1)


def simulate(a=A0, dx=0.01, dt_factor=1.0, record=False):
    """RK4 method of lines. Returns dict with node positions, 0 mV crossing
    times, the fitted velocity (m/s) and optional recordings."""
    scale = np.sqrt(a / A0)
    L = L0 * scale
    N = int(round(L / dx))
    x = np.arange(N + 1) * dx
    D = diffusion_coefficient(a)
    dt = dt_factor * CFL * dx ** 2 * C_m / D
    stim = np.where(x < STIM_X * scale, STIM_AMP, 0.0)
    zero = np.zeros_like(x)
    lo, hi = WINDOW[0] * L, WINDOW[1] * L
    win = (x >= lo) & (x <= hi)
    last = np.flatnonzero(win)[-1]

    y = resting_state(N + 1)
    t_cross = np.full(N + 1, np.nan)
    rec_t, rec_V, snaps = [], [], []
    probes = [int(round(p * L / dx)) for p in (0.25, 0.5, 0.75)]
    rec_every = max(1, int(round(0.01 / dt)))
    snap_times = list(np.arange(0.5, T_END, 0.5))
    n_max = int(np.ceil(T_END / dt))
    t, k = 0.0, 0
    for k in range(n_max):
        I1 = stim if t < STIM_T else zero
        I2 = stim if t + 0.5 * dt < STIM_T else zero
        I4 = stim if t + dt < STIM_T else zero
        k1 = rhs(y, D, dx, I1)
        k2 = rhs(y + 0.5 * dt * k1, D, dx, I2)
        k3 = rhs(y + 0.5 * dt * k2, D, dx, I2)
        k4 = rhs(y + dt * k3, D, dx, I4)
        V_old = y[0]
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        new = np.isnan(t_cross) & (V_old < V_CROSS) & (y[0] >= V_CROSS)
        if new.any():
            t_cross[new] = t + dt * (V_CROSS - V_old[new]) / (y[0][new] - V_old[new])
        t += dt
        if record:
            if k % rec_every == 0:
                rec_t.append(t)
                rec_V.append(y[0][probes].copy())
            if snap_times and t >= snap_times[0]:
                snaps.append((snap_times.pop(0), y[0].copy()))
        elif not np.isnan(t_cross[last]):
            break
        if record and t > T_END - 1e-12:
            break
    assert not np.isnan(t_cross[win]).any(), "action potential did not traverse the window"
    slope, icpt = np.polyfit(t_cross[win], x[win], 1)
    resid = x[win] - (slope * t_cross[win] + icpt)
    out = dict(a=a, dx=dx, dt=dt, L=L, x=x, t_cross=t_cross, win=win,
               velocity=10.0 * slope, fit=(slope, icpt), resid=np.abs(resid).max(),
               steps=k + 1, peak=y[0].max())
    if record:
        out.update(rec_t=np.array(rec_t), rec_V=np.array(rec_V),
                   probes=x[probes], snaps=snaps)
    return out


# --------------------------------------------------------------- validation
def validate():
    R = {}
    print(f"Hodgkin-Huxley cable: a = {A0 * 1e4:.0f} um, R_i = {R_I} Ohm cm, "
          f"T = {TEMP} C (phi = {PHI:.4f}), L = {L0} cm, sealed ends")
    lam = np.sqrt(A0 * (1.0 / (g_L * 1e-3)) / (2.0 * R_I))
    print(f"length constant at rest (leak only) lambda = {lam:.3f} cm")

    print("[1] Refinement, dt = 0.25 dx^2 C_m / D:")
    runs = []
    for dx in DXS:
        t0 = _time.time()
        r = simulate(dx=dx, record=(dx == 0.01))
        runs.append(r)
        print(f"    dx = {dx:6.3f} cm  dt = {r['dt']:.3e} ms  c = {r['velocity']:.5f} m/s  "
              f"(fit residual {r['resid'] * 1e4:.2f} um, {r['steps']} steps, {_time.time() - t0:.1f} s)")
    c = np.array([r["velocity"] for r in runs])
    order = np.log2((c[-3] - c[-2]) / (c[-2] - c[-1]))
    c_inf = c[-1] + (c[-1] - c[-2]) / (2.0 ** 2 - 1.0)
    rich_err = abs(c[-1] - c_inf) / c_inf
    print(f"    observed order (three finest grids) = {order:.3f} (expected 2, tolerance {TOL_ORDER})")
    print(f"    Richardson extrapolate c_inf = {c_inf:.5f} m/s; finest grid off by {rich_err:.2e} "
          f"(tolerance {TOL_RICH})")
    assert abs(order - 2.0) < TOL_ORDER
    assert rich_err < TOL_RICH
    r_half = simulate(dx=0.01, dt_factor=0.5)
    j01 = DXS.index(0.01)
    dt_err = abs(r_half["velocity"] - c[j01]) / c[j01]
    print(f"    dx = 0.01 cm with dt halved: c = {r_half['velocity']:.5f} m/s, change {dt_err:.2e} "
          f"(tolerance {TOL_DT})")
    assert dt_err < TOL_DT

    pub_err = (c_inf - HH_VELOCITY) / HH_VELOCITY
    print(f"[2] c_inf = {c_inf:.3f} m/s vs Hodgkin & Huxley (1952) computed {HH_VELOCITY} m/s: "
          f"{100 * pub_err:+.2f}% (tolerance {100 * TOL_PUB:.0f}%); measured on the fibre {HH_MEASURED} m/s: "
          f"{100 * (c_inf - HH_MEASURED) / HH_MEASURED:+.2f}%")
    assert abs(pub_err) < TOL_PUB

    print(f"[3] Diameter scaling at dx = {DX_SCALE} cm:")
    sweep = []
    for f in A_FACTORS:
        t0 = _time.time()
        r = simulate(a=f * A0, dx=DX_SCALE)
        sweep.append(r)
        print(f"    a = {f * A0 * 1e4:7.2f} um  d = {2 * f * A0 * 1e4:7.1f} um  c = {r['velocity']:.4f} m/s  "
              f"({_time.time() - t0:.1f} s)")
    d = 2.0 * np.array(A_FACTORS) * A0
    v = np.array([r["velocity"] for r in sweep])
    slope, icpt = np.polyfit(np.log(d), np.log(v), 1)
    print(f"    log-log slope = {slope:.4f} (expected 0.5, tolerance {TOL_SLOPE})")
    assert abs(slope - 0.5) < TOL_SLOPE
    R.update(runs=runs, c=c, c_inf=c_inf, order=order, pub_err=pub_err,
             d=d, v=v, slope=slope, icpt=icpt, dt_err=dt_err)
    return R


# -------------------------------------------------------------------- plot
def plot(R):
    r = R["runs"][DXS.index(0.01)]
    fig, ax = plt.subplots(2, 3, figsize=(15, 8.5))

    a = ax[0, 0]
    for j, xp in enumerate(r["probes"]):
        a.plot(r["rec_t"], r["rec_V"][:, j], label=f"x = {xp:.0f} cm")
    a.set_xlabel("time (ms)"); a.set_ylabel("V (mV)")
    a.set_title("Membrane potential at three positions"); a.legend(frameon=False)

    a = ax[0, 1]
    cmap = plt.get_cmap("viridis")
    snaps = [s for s in r["snaps"] if s[0] <= 4.5]
    for i, (ts, Vs) in enumerate(snaps):
        a.plot(r["x"], Vs, color=cmap(i / max(1, len(snaps) - 1)), label=f"{ts:.1f} ms")
    a.set_xlabel("x (cm)"); a.set_ylabel("V (mV)")
    a.set_title("Voltage along the cable (dx = 0.01 cm)"); a.legend(frameon=False, fontsize=7, ncol=2)

    a = ax[0, 2]
    ok = ~np.isnan(r["t_cross"])
    a.plot(r["t_cross"][ok], r["x"][ok], ".", ms=2, color="0.5", label="0 mV crossing, all nodes")
    w = r["win"]
    s, b = r["fit"]
    tt = np.array([r["t_cross"][w].min(), r["t_cross"][w].max()])
    a.plot(tt, s * tt + b, "r-", lw=2, label=f"fit on window: {r['velocity']:.3f} m/s")
    a.axhspan(WINDOW[0] * r["L"], WINDOW[1] * r["L"], color="r", alpha=0.07)
    a.set_xlabel("crossing time (ms)"); a.set_ylabel("x (cm)")
    a.set_title("Wavefront position"); a.legend(frameon=False)

    a = ax[1, 0]
    dxs = np.array(DXS)
    a.plot(dxs * 1e4, R["c"], "o-", label="simulated")
    a.axhline(R["c_inf"], color="k", ls="--", lw=1, label=f"Richardson c_inf = {R['c_inf']:.3f} m/s")
    a.axhline(HH_VELOCITY, color="r", lw=1.5, label=f"HH 1952 computed {HH_VELOCITY} m/s")
    a.axhspan(HH_VELOCITY * (1 - TOL_PUB), HH_VELOCITY * (1 + TOL_PUB), color="r", alpha=0.08,
              label="±10% tolerance")
    a.axhline(HH_MEASURED, color="m", ls=":", lw=1.5, label=f"measured {HH_MEASURED} m/s")
    a.set_xscale("log"); a.set_xlabel("dx (µm)"); a.set_ylabel("conduction velocity (m/s)")
    a.set_title("Velocity under dx (and dt ∝ dx²) refinement"); a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]
    err = np.abs(R["c"] - R["c_inf"])
    a.loglog(dxs * 1e4, err, "o-", label=f"|c − c_inf|, order {R['order']:.2f}")
    a.loglog(dxs * 1e4, err[-1] * (dxs / dxs[-1]) ** 2, "k--", lw=1, label="slope 2")
    a.set_xlabel("dx (µm)"); a.set_ylabel("velocity error (m/s)")
    a.set_title("Spatial convergence"); a.legend(frameon=False)

    a = ax[1, 2]
    dd = R["d"] * 1e4
    a.loglog(dd, R["v"], "o", ms=7, label="simulated (dx = 0.01 cm)")
    a.loglog(dd, np.exp(R["icpt"]) * R["d"] ** R["slope"], "b-", lw=1,
             label=f"fit slope {R['slope']:.4f}")
    a.loglog(dd, R["c_inf"] * np.sqrt(R["d"] / (2 * A0)), "k--", lw=1, label="c_inf √(d/476 µm)")
    a.set_xlabel("diameter (µm)"); a.set_ylabel("conduction velocity (m/s)")
    a.set_title("Velocity ∝ √diameter"); a.legend(frameon=False)

    fig.suptitle("Propagating action potential on the Hodgkin–Huxley cable (a = 238 µm, R_i = 35.4 Ω·cm, 18.5 °C)")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "hh_cable_propagation.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
