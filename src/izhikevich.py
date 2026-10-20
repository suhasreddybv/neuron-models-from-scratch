"""Izhikevich (2003) simple spiking neuron model.

Equations (Izhikevich 2003, eqs. 1-3; "Simple model of spiking neurons",
IEEE Trans. Neural Networks 14(6), 1569-1572):

    dv/dt = 0.04 v^2 + 5 v + 140 - u + I
    du/dt = a (b v - u)
    if v >= 30 mV:  v <- c,  u <- u + d

v is the membrane potential in mV, u a recovery variable, t in ms and I the
injected current (dimensionless, same units as dv/dt in mV/ms). The quadratic
term and the constants 5 and 140 come from fitting the spike-initiation
dynamics of a cortical neuron so that v has mV scale (paper, sec. II).

Published parameter sets (paper, Fig. 2; the four the ROADMAP asks for):

    regular spiking        RS  a = 0.02  b = 0.2  c = -65  d = 8
    intrinsically bursting IB  a = 0.02  b = 0.2  c = -55  d = 4
    chattering             CH  a = 0.02  b = 0.2  c = -50  d = 2
    fast spiking           FS  a = 0.1   b = 0.2  c = -65  d = 2

The paper describes the regimes but prints no firing rates, so the "paper's
figure" comparison below is done in two ways: (1) each pattern is tested
against the paper's verbal description with explicit, quantitative criteria,
and (2) each trace is recomputed with the paper's own integration scheme
(forward Euler, dt = 1 ms for u and two 0.5 ms half-steps for v, as in the
MATLAB code printed in Fig. 1 of the paper) so the rates the paper's figure
was drawn from can be compared with the converged simulation.

Closed forms used for validation. With constant I the v-nullcline is
u = 0.04 v^2 + 5 v + 140 + I and the u-nullcline is u = b v, so fixed points
solve 0.04 v^2 + (5 - b) v + 140 + I = 0:

    v* = [-(5 - b) -/+ sqrt(D)] / 0.08,   D = (5 - b)^2 - 0.16 (140 + I).

For b = 0.2 and I = 0 this gives v_rest = -70 mV (the lower root, a stable
focus) and a saddle at -50 mV, as stated in the paper. The two merge in a
saddle-node at I_SN = (5 - b)^2 / 0.16 - 140 (= 4 for b = 0.2). The Jacobian
at the lower fixed point is [[0.08 v* + 5, -1], [a b, -a]]; since
0.08 v* + 5 = b - sqrt(D), its trace is b - a - sqrt(D) and its determinant
a sqrt(D) > 0, so the rest state loses stability (Andronov-Hopf) where
sqrt(D) = b - a:

    I_H = [(5 - b)^2 - (b - a)^2] / 0.16 - 140   (3.7975 for RS/IB/CH, 3.9375 for FS).

Integrator: forward Euler with dt = 0.01 ms for the converged runs, spike
time interpolated linearly across the v = 30 mV crossing before the reset;
the paper's coarse scheme is reproduced separately for comparison. RNG: a
seeded generator only jitters the perturbation direction in the stability
sweep; the seed is printed.

Validation (ROADMAP note: "reproduce the RS, IB, CH and FS regimes with the
published parameter sets; compare firing rates with the paper's figure"):
  (a) Rest potential at I = 0 for b = 0.2 settles on the closed-form root
      v* = -70 mV, u* = b v* = -14, to 1e-4 (no spikes).
  (b) Loss of rest-state stability found by a bisection sweep over I (does a
      1e-3 perturbation of the fixed point grow or decay over 2000 ms?)
      agrees with the closed-form I_H for the RS and FS parameter sets to
      within 1% of I_SN, and lies below the saddle-node current I_SN = 4.
  (c) Reference solver and convergence: spike times of each of the four
      regimes at I = 10 (500 ms) from Euler vs scipy.integrate.solve_ivp
      (DOP853, rtol = atol = 1e-10, terminal event at v = 30 then exact
      reset): same spike count and mean rate within 1% at dt = 0.01 ms, and
      the maximum spike-time deviation must fall by a factor in [1.6, 2.4]
      when dt is halved to 0.005 ms (first-order convergence; the deviation
      is a phase drift that accumulates by ~0.02-0.04 ms per spike at
      dt = 0.01 ms and is printed per regime).
  (d) Regime criteria, from the paper's descriptions (sec. III):
      RS  "a few spikes with short interspike period and then the period
          increases": last ISI > 1.2 x first ISI, no ISI < 10 ms after the
          first two spikes;
      IB  "a stereotypical burst of spikes followed by repetitive single
          spikes": an initial burst of >= 2 spikes (ISI < 10 ms) and every
          later spike a single (all later ISIs > 10 ms);
      CH  "stereotypical bursts of closely spaced spikes ... inter-burst
          frequency can be as high as 40 Hz": >= 3 bursts of >= 2 spikes with
          intra-burst ISI < 10 ms, inter-burst frequency <= 40 Hz;
      FS  "periodic trains ... extremely high frequency ... practically without
          any adaptation": CV of ISI < 0.05 (first ISI excluded) and a higher
          rate than RS.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters
# ---------------------------------------------------------------------------
dt = 0.01            # Euler time step, ms
V_PEAK = 30.0        # spike cut-off, mV (paper: "if v >= 30 mV")

# Published parameter sets, Izhikevich (2003) Fig. 2: (a, b, c, d)
REGIMES = {
    "RS": dict(a=0.02, b=0.2, c=-65.0, d=8.0, name="regular spiking"),
    "IB": dict(a=0.02, b=0.2, c=-55.0, d=4.0, name="intrinsically bursting"),
    "CH": dict(a=0.02, b=0.2, c=-50.0, d=2.0, name="chattering"),
    "FS": dict(a=0.1,  b=0.2, c=-65.0, d=2.0, name="fast spiking"),
}
I_STEP = 10.0        # step current for the regime traces
T_PRE = 50.0         # ms at rest before the step
T_RUN = 500.0        # ms of step current
I_SWEEP_SPAN = (3.0, 4.0)   # bracket for the stability sweep (below I_SN = 4)
T_SWEEP = 2000.0     # ms, length of each stability test
BURST_ISI = 10.0     # ms, intra-burst ISI threshold used in the criteria

SEED = 0
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions
# ---------------------------------------------------------------------------
def f_v(v, u, I):
    return 0.04 * v * v + 5.0 * v + 140.0 - u + I


def f_u(v, u, a, b):
    return a * (b * v - u)


def fixed_points(I, b):
    """Lower (rest) and upper (saddle) roots of 0.04 v^2 + (5 - b) v + 140 + I = 0."""
    D = (5.0 - b) ** 2 - 0.16 * (140.0 + I)
    if D < 0:
        return None, None
    v_lo = (-(5.0 - b) - np.sqrt(D)) / 0.08
    v_hi = (-(5.0 - b) + np.sqrt(D)) / 0.08
    return v_lo, v_hi


def saddle_node_current(b):
    return (5.0 - b) ** 2 / 0.16 - 140.0


def hopf_current(a, b):
    """Current where the trace b - a - sqrt(D) of the Jacobian at rest vanishes."""
    return ((5.0 - b) ** 2 - (b - a) ** 2) / 0.16 - 140.0


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(a, b, c, d, I, T, v0, u0, dt=dt):
    """Forward Euler. I may be a scalar or an array (t) of the same length as the run.

    Returns time, v(t), u(t) and the interpolated spike times. v is clipped at
    V_PEAK in the returned trace for display; the reset happens on the step the
    threshold is crossed.
    """
    n = int(round(T / dt))
    time = np.arange(n) * dt
    I_t = np.broadcast_to(np.asarray(I, dtype=float), (n,))
    v = np.empty(n)
    u = np.empty(n)
    v[0], u[0] = v0, u0
    spikes = []
    for i in range(1, n):
        vp, up = v[i - 1], u[i - 1]
        vn = vp + dt * f_v(vp, up, I_t[i - 1])
        un = up + dt * f_u(vp, up, a, b)
        if vn >= V_PEAK:
            spikes.append(time[i - 1] + dt * (V_PEAK - vp) / (vn - vp))
            vn, un = c, un + d
        v[i], u[i] = vn, un
    return time, v, u, np.array(spikes)


def simulate_paper_scheme(a, b, c, d, I, T, v0, u0):
    """The integration printed in Fig. 1 of the paper: dt = 1 ms, v in two
    0.5 ms Euler half-steps, u in one 1 ms step, reset tested once per ms."""
    n = int(round(T))
    time = np.arange(n, dtype=float)
    v = np.empty(n)
    u = np.empty(n)
    v[0], u[0] = v0, u0
    spikes = []
    for i in range(1, n):
        vp, up = v[i - 1], u[i - 1]
        if vp >= V_PEAK:
            vp, up = c, up + d
        vn = vp + 0.5 * f_v(vp, up, I)
        vn = vn + 0.5 * f_v(vn, up, I)
        un = up + f_u(vp, up, a, b)
        if vn >= V_PEAK:
            spikes.append(time[i])
        v[i], u[i] = vn, un
    return time, np.minimum(v, V_PEAK), u, np.array(spikes)


def simulate_reference(a, b, c, d, I, T, v0, u0):
    """solve_ivp (DOP853, tight tolerances) with a terminal event at v = 30 mV
    and an exact reset, as an integrator-independent reference."""
    def rhs(t, y):
        return [f_v(y[0], y[1], I), f_u(y[0], y[1], a, b)]

    def hit(t, y):
        return y[0] - V_PEAK
    hit.terminal = True
    hit.direction = 1

    t0, y0, spikes = 0.0, [v0, u0], []
    while t0 < T:
        sol = solve_ivp(rhs, (t0, T), y0, method="DOP853",
                        rtol=1e-10, atol=1e-10, events=hit)
        if sol.t_events[0].size == 0:
            break
        t_spk = sol.t_events[0][0]
        spikes.append(t_spk)
        u_spk = sol.y_events[0][0][1]
        t0, y0 = t_spk, [c, u_spk + d]
    return np.array(spikes)


def regime_trace(key, scheme="euler"):
    p = REGIMES[key]
    v0, _ = fixed_points(0.0, p["b"])
    u0 = p["b"] * v0
    T = T_PRE + T_RUN
    if scheme == "paper":
        # the paper scheme takes a scalar I; run the step alone and shift to t = T_PRE
        t, v, u, s = simulate_paper_scheme(p["a"], p["b"], p["c"], p["d"], I_STEP, T_RUN, v0, u0)
        return t + T_PRE, v, u, s + T_PRE
    I_t = np.where(np.arange(int(round(T / dt))) * dt >= T_PRE, I_STEP, 0.0)
    return simulate(p["a"], p["b"], p["c"], p["d"], I_t, T, v0, u0)


def mean_rate(spikes, window):
    """Spikes per second over the stimulus window (ms)."""
    return 1000.0 * spikes.size / window


def bursts(spikes, isi_max=BURST_ISI):
    """Split a spike train into bursts: consecutive spikes closer than isi_max."""
    if spikes.size == 0:
        return []
    out, cur = [], [spikes[0]]
    for s in spikes[1:]:
        if s - cur[-1] < isi_max:
            cur.append(s)
        else:
            out.append(np.array(cur))
            cur = [s]
    out.append(np.array(cur))
    return out


def stability_sweep(a, b, I_lo, I_hi, n_iter=14, eps=1e-3):
    """Bisection on I for the current at which a small perturbation of the
    rest state stops decaying. Vectorised over 16 currents per iteration."""
    def grows(I_vals):
        I_vals = np.asarray(I_vals)
        D = (5.0 - b) ** 2 - 0.16 * (140.0 + I_vals)
        v_star = (-(5.0 - b) - np.sqrt(D)) / 0.08
        u_star = b * v_star
        ang = rng.uniform(0, 2 * np.pi)
        v = v_star + eps * np.cos(ang)
        u = u_star + eps * np.sin(ang) * b
        n = int(round(T_SWEEP / dt))
        r0 = np.hypot(v - v_star, (u - u_star) / b)
        for _ in range(n):
            vn = v + dt * f_v(v, u, I_vals)
            un = u + dt * f_u(v, u, a, b)
            # a run that escapes to a spike is unstable; hold it at V_PEAK so
            # the quadratic term cannot overflow
            v, u = np.minimum(vn, V_PEAK), un
        r1 = np.hypot(v - v_star, (u - u_star) / b)
        return (r1 > r0) | ~np.isfinite(r1)
    lo, hi = I_lo, I_hi
    for _ in range(n_iter):
        grid = np.linspace(lo, hi, 16)
        g = grows(grid)
        if not g.any():
            return hi
        k = int(np.argmax(g))          # first current that grows
        if k == 0:
            return lo
        lo, hi = grid[k - 1], grid[k]
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    results = {}
    print("\n=== Izhikevich (2003) validation ===")

    # (a) rest state at I = 0 vs closed form
    p = REGIMES["RS"]
    v_cf, v_sad = fixed_points(0.0, p["b"])
    t, v, u, s = simulate(p["a"], p["b"], p["c"], p["d"], 0.0, 300.0, v_cf + 3.0, p["b"] * v_cf)
    dv, du = abs(v[-1] - v_cf), abs(u[-1] - p["b"] * v_cf)
    print(f"(a) rest at I=0: closed form v*={v_cf:.4f} mV (saddle {v_sad:.4f}), "
          f"simulated v={v[-1]:.6f}, u={u[-1]:.6f}; |dv|={dv:.2e}, |du|={du:.2e}, spikes={s.size}")
    assert dv < 1e-4 and du < 1e-4 and s.size == 0, "rest state does not match closed form"
    results["rest"] = (v_cf, v[-1], u[-1])

    # (b) stability sweep vs Hopf closed form
    results["hopf"] = {}
    for key in ("RS", "FS"):
        p = REGIMES[key]
        I_H = hopf_current(p["a"], p["b"])
        I_SN = saddle_node_current(p["b"])
        I_sweep = stability_sweep(p["a"], p["b"], *I_SWEEP_SPAN)
        err = abs(I_sweep - I_H) / I_SN
        print(f"(b) {key}: rest loses stability at I={I_sweep:.5f} (sweep) vs I_H={I_H:.5f} "
              f"(Jacobian), I_SN={I_SN:.4f}; |diff|/I_SN={100*err:.3f}%")
        assert err < 0.01, f"{key} Hopf current off by more than 1% of I_SN"
        assert I_sweep < I_SN
        results["hopf"][key] = (I_sweep, I_H, I_SN)

    # (c) regimes at I = 10 vs reference solver, (d) regime criteria
    results["traces"] = {}
    rates = {}
    for key, p in REGIMES.items():
        t, v, u, s = regime_trace(key)
        s_ref = simulate_reference(p["a"], p["b"], p["c"], p["d"], I_STEP, T_RUN,
                                   *fixed_points(0.0, p["b"])[:1], p["b"] * fixed_points(0.0, p["b"])[0]) + T_PRE
        tp, vp, up, sp = regime_trace(key, scheme="paper")
        r_eu, r_ref, r_pa = (mean_rate(x, T_RUN) for x in (s, s_ref, sp))
        rel = abs(r_eu - r_ref) / r_ref
        assert s.size == s_ref.size, f"{key}: spike count differs from reference solver"
        assert rel < 0.01, f"{key}: rate differs from reference by > 1%"
        # convergence: same run at dt and dt/2, deviation from the reference
        v0, _ = fixed_points(0.0, p["b"])
        dev = []
        for h in (dt, dt / 2):
            _, _, _, s_h = simulate(p["a"], p["b"], p["c"], p["d"], I_STEP, T_RUN, v0, p["b"] * v0, dt=h)
            n_cmp = min(s_h.size, s_ref.size)
            dev.append(np.max(np.abs(s_h[:n_cmp] + T_PRE - s_ref[:n_cmp])))
        ratio = dev[0] / dev[1]
        print(f"(c) {key}: Euler {s.size} spikes, {r_eu:.1f} Hz; solve_ivp {s_ref.size} spikes, "
              f"{r_ref:.1f} Hz; rate diff {100*rel:.3f}%; max |dt_spike| = {dev[0]:.4f} ms at dt={dt} "
              f"({1000*dev[0]/s.size:.1f} us/spike), {dev[1]:.4f} ms at dt={dt/2}, ratio {ratio:.2f}; "
              f"paper scheme (dt=0.5/1 ms) {sp.size} spikes, {r_pa:.1f} Hz")
        assert 1.6 < ratio < 2.4, f"{key}: spike-time error is not first order in dt"
        rates[key] = r_eu
        results["traces"][key] = dict(t=t, v=np.minimum(v, V_PEAK), u=u, s=s, s_ref=s_ref,
                                      tp=tp, vp=vp, sp=sp, rate=r_eu, rate_ref=r_ref, rate_paper=r_pa,
                                      dev=dev, ratio=ratio)

    isi = {k: np.diff(results["traces"][k]["s"]) for k in REGIMES}
    # RS
    rs = isi["RS"]
    print(f"(d) RS: first ISI {rs[0]:.2f} ms, last ISI {rs[-1]:.2f} ms, ratio {rs[-1]/rs[0]:.2f}; "
          f"min ISI after 2nd spike {rs[2:].min():.2f} ms")
    assert rs[-1] > 1.2 * rs[0] and rs[2:].min() > BURST_ISI, "RS adaptation criterion failed"
    # IB
    ib_b = bursts(results["traces"]["IB"]["s"])
    ib = isi["IB"]
    n0 = ib_b[0].size
    later = np.array([b.size for b in ib_b[1:]])
    print(f"(d) IB: initial burst of {n0} spikes (ISIs {', '.join(f'{x:.2f}' for x in ib[:n0-1])} ms), "
          f"then {len(later)} single spikes with ISIs {ib[n0-1:].min():.1f}-{ib[n0-1:].max():.1f} ms")
    assert n0 >= 2 and later.size >= 3 and later.max() == 1, "IB burst-then-single criterion failed"
    # CH
    ch_b = bursts(results["traces"]["CH"]["s"])
    sizes = np.array([b.size for b in ch_b])
    onsets = np.array([b[0] for b in ch_b])
    ibi = np.diff(onsets)
    f_burst = 1000.0 / ibi.mean()
    intra = np.concatenate([np.diff(b) for b in ch_b if b.size > 1])
    print(f"(d) CH: {len(ch_b)} bursts of {sizes.min()}-{sizes.max()} spikes, intra-burst ISI "
          f"{intra.min():.2f}-{intra.max():.2f} ms, inter-burst interval {ibi.mean():.1f} ms "
          f"= {f_burst:.1f} Hz (paper: as high as 40 Hz)")
    assert len(ch_b) >= 3 and sizes.min() >= 2 and f_burst <= 40.0, "CH chattering criterion failed"
    # FS
    fs = isi["FS"][1:]
    cv = fs.std() / fs.mean()
    print(f"(d) FS: rate {rates['FS']:.1f} Hz vs RS {rates['RS']:.1f} Hz, ISI CV (1st excluded) {cv:.4f}")
    assert cv < 0.05 and rates["FS"] > rates["RS"], "FS periodic high-rate criterion failed"
    results["criteria"] = dict(rs_ratio=rs[-1] / rs[0], ib_first=n0, ch_bursts=len(ch_b),
                               ch_fburst=f_burst, fs_cv=cv)
    print("All validation checks passed.")
    return results


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(results):
    FIGURES.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.5))
    axes = axes.ravel()
    for ax, key in zip(axes[:4], REGIMES):
        tr = results["traces"][key]
        p = REGIMES[key]
        ax.plot(tr["t"], tr["v"], color="C0", lw=0.9, label=f"Euler dt = {dt} ms, {tr['rate']:.1f} Hz")
        ax.plot(tr["tp"], tr["vp"], color="C1", lw=0.7, alpha=0.6,
                label=f"paper scheme dt = 0.5/1 ms, {tr['rate_paper']:.1f} Hz")
        ax.plot(tr["s_ref"], np.full_like(tr["s_ref"], 36.0), "v", color="k", ms=4,
                label=f"solve_ivp spikes, {tr['rate_ref']:.1f} Hz")
        ax.axvspan(T_PRE, T_PRE + T_RUN, color="0.92", zorder=0)
        ax.set_title(f"{key}: {p['name']}  (a={p['a']}, b={p['b']}, c={p['c']:.0f}, d={p['d']:.0f}), I = {I_STEP:.0f}")
        ax.set_ylabel("v (mV)")
        ax.set_ylim(-95, 42)
        ax.set_xlim(0, T_PRE + T_RUN)
        ax.legend(loc="lower right", fontsize=7)
    axes[2].set_xlabel("t (ms)")
    axes[3].set_xlabel("t (ms)")

    # phase plane with nullclines at I = 0 and I = 10 (RS)
    ax = axes[4]
    p = REGIMES["RS"]
    vv = np.linspace(-90, 35, 400)
    for I, col in ((0.0, "C2"), (I_STEP, "C3")):
        ax.plot(vv, 0.04 * vv ** 2 + 5 * vv + 140 + I, color=col, lw=1, label=f"v-nullcline, I = {I:.0f}")
    ax.plot(vv, p["b"] * vv, "k--", lw=1, label="u-nullcline u = b v")
    v_lo, v_hi = fixed_points(0.0, p["b"])
    ax.plot([v_lo, v_hi], [p["b"] * v_lo, p["b"] * v_hi], "o", color="C2", ms=5)
    tr = results["traces"]["RS"]
    ax.plot(tr["v"], tr["u"], color="C0", lw=0.5, alpha=0.7, label="RS trajectory")
    ax.set_xlim(-90, 35)
    ax.set_ylim(-20, 10)
    ax.set_xlabel("v (mV)")
    ax.set_ylabel("u")
    ax.set_title(f"RS phase plane: rest {v_lo:.0f} mV, saddle {v_hi:.0f} mV at I = 0; none at I = {I_STEP:.0f}")
    ax.legend(fontsize=7, loc="upper left")

    # stability sweep vs closed form
    ax = axes[5]
    for k, (key, col) in enumerate((("RS", "C0"), ("FS", "C3"))):
        I_sw, I_H, I_SN = results["hopf"][key]
        ax.plot([I_H], [k], "|", color=col, ms=22, mew=2, label=f"{key}: I_H Jacobian = {I_H:.4f}")
        ax.plot([I_sw], [k], "o", color=col, mfc="none", ms=10, label=f"{key}: sweep = {I_sw:.4f}")
    ax.axvline(4.0, color="k", ls=":", label="saddle-node I_SN = 4")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["RS (a = 0.02)", "FS (a = 0.1)"])
    ax.set_xlim(3.7, 4.05)
    ax.set_xlabel("I")
    ax.set_title("Loss of rest-state stability: bisection sweep vs trace(J) = 0")
    ax.legend(fontsize=7, loc="upper left")

    fig.suptitle("Izhikevich (2003) simple model: four published regimes, reference solver, closed-form rest/Hopf checks", fontsize=11)
    fig.tight_layout()
    out = FIGURES / "izhikevich.png"
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
