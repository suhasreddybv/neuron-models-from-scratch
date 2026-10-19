"""Quadratic integrate-and-fire (QIF) neuron and the theta neuron.

Equations. The QIF is the normal form of a saddle-node-on-invariant-circle
(SNIC) bifurcation, written here in the dimensionless canonical form of
Izhikevich (2007, sec. 3.3.8 and 8.1.3):

    dV/dt = V^2 + I,        spike when V reaches V_peak, then V <- V_reset

with V_reset = -V_peak. The theta neuron (Ermentrout & Kopell 1986, eq. 1.2;
Ermentrout 1996) is the same system after the change of variables

    V = tan(theta/2)   <=>   theta = 2 arctan V,

which maps the whole real line onto the circle (-pi, pi):

    dtheta/dt = (1 - cos theta) + (1 + cos theta) I,   spike when theta crosses pi.

Derivation: d theta/dt = 2 dV/dt / (1 + V^2) = 2 (V^2 + I) / (1 + V^2), and with
V = tan(theta/2), 2 V^2/(1 + V^2) = 1 - cos theta and 2/(1 + V^2) = 1 + cos theta.
A QIF with finite threshold V_peak and reset -V_peak is therefore exactly a
theta neuron with threshold theta_peak = 2 arctan V_peak and reset -theta_peak;
the standard theta neuron (theta_peak = pi) is the QIF with V_peak -> infinity.

Closed forms used for validation.
  * For I > 0 the QIF solution is V(t) = sqrt(I) tan(sqrt(I) t + arctan(V0/sqrt(I))),
    so the period from V = -V_peak to +V_peak is
        T(I) = (2/sqrt(I)) arctan(V_peak/sqrt(I)),
    and for the theta neuron (V_peak -> inf) this is T = pi/sqrt(I), i.e. the
    firing rate is f = sqrt(I)/pi (Ermentrout 1996; Izhikevich 2007 eq. 8.3).
    The rate rises continuously from 0 at I = 0: class I excitability.
  * For I < 0 there is no firing. The fixed points are V* = -/+ sqrt(-I)
    (stable node / saddle), i.e. theta* = -/+ 2 arctan sqrt(-I); they merge at
    I = 0 (SNIC).

Integrator: forward Euler with dt = 1e-3 (dimensionless time) for both models,
with the spike time interpolated linearly between the last sub-threshold and
first supra-threshold step. The repository's other models use the same
scheme. RNG: a seeded generator only jitters the initial condition of the f-I
runs so that no spike is aligned to the grid by construction; seed printed.

Validation (ROADMAP note: "f-I curve against the closed-form rate of the theta
neuron; show the two models are equivalent under the change of variables"):
  (a) theta-neuron f-I curve from simulation vs f = sqrt(I)/pi for 40 currents
      in [0.02, 2]: max relative error must be < 1%.
  (b) QIF (V_peak = 10) f-I curve vs its own closed form
      1/T = sqrt(I) / (2 arctan(V_peak/sqrt(I))): max relative error < 1%.
  (c) Equivalence: a theta neuron with theta_peak = 2 arctan(V_peak) fires at
      the same rate as the QIF with V_peak at every current (max relative
      difference < 1%), and one QIF trajectory mapped through theta = 2 arctan V
      lies on top of the theta-neuron trajectory and on the exact solution
      (max |delta theta| < 0.02 rad over one inter-spike interval).
  (d) For I = -0.25 both simulations settle on the stable fixed point
      V* = -sqrt(-I), theta* = -2 arctan sqrt(-I), to 1e-6.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters
# ---------------------------------------------------------------------------
dt = 1e-3            # Euler time step (dimensionless time)
V_peak = 10.0        # QIF spike threshold; reset is -V_peak
V_reset = -V_peak
THETA_PEAK = np.pi   # standard theta neuron: threshold at pi, reset to -pi

I_FI = np.linspace(0.02, 2.0, 40)   # currents for the f-I curves
T_FI = 200.0                        # run length per current (>= 9 spikes at I = 0.02)
I_DEMO = 0.5                        # current for the time-series / equivalence panels
I_REST = -0.25                      # sub-threshold current for the fixed-point check

SEED = 0
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions and closed forms
# ---------------------------------------------------------------------------
def f_qif(V, I):
    """Right-hand side of the QIF: dV/dt = V^2 + I."""
    return V ** 2 + I


def f_theta(theta, I):
    """Right-hand side of the theta neuron: dtheta/dt = (1 - cos theta) + (1 + cos theta) I."""
    return (1.0 - np.cos(theta)) + (1.0 + np.cos(theta)) * I


def v_to_theta(V):
    return 2.0 * np.arctan(V)


def theta_to_v(theta):
    return np.tan(theta / 2.0)


def qif_exact(t, I, V0):
    """Closed-form QIF solution for I > 0 starting from V0 at t = 0."""
    s = np.sqrt(I)
    return s * np.tan(s * t + np.arctan(V0 / s))


def period_qif(I, v_peak=V_peak):
    """Inter-spike interval of the QIF with threshold v_peak and reset -v_peak, I > 0."""
    s = np.sqrt(I)
    return 2.0 * np.arctan(v_peak / s) / s


def rate_theta_closed_form(I):
    """f = sqrt(I)/pi for I > 0, zero otherwise (class I: continuous from 0)."""
    I = np.asarray(I, dtype=float)
    return np.where(I > 0, np.sqrt(np.maximum(I, 0.0)) / np.pi, 0.0)


def rate_qif_closed_form(I, v_peak=V_peak):
    I = np.asarray(I, dtype=float)
    out = np.zeros_like(I)
    pos = I > 0
    out[pos] = 1.0 / period_qif(I[pos], v_peak)
    return out


def fixed_points(I):
    """(stable, saddle) fixed points of the QIF and theta neuron for I < 0."""
    assert I < 0
    r = np.sqrt(-I)
    return (-r, r), (v_to_theta(-r), v_to_theta(r))


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(model, I, T, x0, x_peak, x_reset, dt=dt):
    """Forward-Euler simulation of 'qif' or 'theta' with constant current I.

    I and x0 may be scalars or 1-D arrays of the same length (lock-step runs).
    Spike when x >= x_peak; the crossing time is linearly interpolated and x is
    reset to x_reset. Returns time, x(t) (shape (n,) or (n, len(I))) and a list
    of spike-time arrays, one per neuron.
    """
    rhs = f_qif if model == "qif" else f_theta
    I = np.asarray(I, dtype=float)
    x0 = np.broadcast_to(np.asarray(x0, dtype=float), I.shape)
    n = int(round(T / dt))
    time = np.arange(n) * dt
    x = np.zeros((n,) + I.shape)
    x[0] = x0
    spikes = [[] for _ in range(I.size)]
    for i in range(1, n):
        x_prev = x[i - 1]
        x_new = x_prev + dt * rhs(x_prev, I)
        fired = x_new >= x_peak
        if np.any(fired):
            frac = (x_peak - x_prev) / (x_new - x_prev)   # in (0, 1]
            t_cross = time[i - 1] + dt * frac
            for k in np.flatnonzero(np.atleast_1d(fired)):
                spikes[k].append(float(np.atleast_1d(t_cross)[k]))
            x_new = np.where(fired, x_reset, x_new)
        x[i] = x_new
    spikes = [np.array(s) for s in spikes]
    if I.ndim == 0:
        return time, x, spikes[0]
    return time, x, spikes


def rate_from_spikes(spikes):
    """1 / mean inter-spike interval (0 if fewer than two spikes)."""
    out = np.zeros(len(spikes))
    for k, s in enumerate(spikes):
        if s.size >= 2:
            out[k] = 1.0 / np.mean(np.diff(s))
    return out


def fi_curve(model, I_values, x_peak, x_reset):
    """Simulated firing rate for each current, from the mean ISI."""
    # Start just above reset with a seeded jitter so spikes are not grid-aligned.
    span = x_peak - x_reset
    x0 = x_reset + 0.01 * span * rng.random(I_values.shape)
    _, _, spikes = simulate(model, I_values, T_FI, x0, x_peak, x_reset)
    return rate_from_spikes(spikes)


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    results = {}
    theta_peak_equiv = v_to_theta(V_peak)   # theta threshold equivalent to V_peak

    # (a) theta neuron f-I vs sqrt(I)/pi
    f_theta_sim = fi_curve("theta", I_FI, THETA_PEAK, -THETA_PEAK)
    f_theta_cf = rate_theta_closed_form(I_FI)
    rel_a = np.abs(f_theta_sim - f_theta_cf) / f_theta_cf
    print(f"(a) theta neuron f-I vs f = sqrt(I)/pi over {I_FI.size} currents in "
          f"[{I_FI[0]:.2f}, {I_FI[-1]:.2f}]: max rel. error {100*rel_a.max():.3f}% "
          f"(at I = {I_FI[rel_a.argmax()]:.3f}), mean {100*rel_a.mean():.3f}%")
    print(f"    e.g. I = {I_FI[-1]:.2f}: simulated {f_theta_sim[-1]:.5f}, closed form {f_theta_cf[-1]:.5f}")
    assert rel_a.max() < 0.01, f"theta f-I mismatch {100*rel_a.max():.2f}% > 1%"

    # (b) QIF (finite V_peak) f-I vs its closed form
    f_qif_sim = fi_curve("qif", I_FI, V_peak, V_reset)
    f_qif_cf = rate_qif_closed_form(I_FI)
    rel_b = np.abs(f_qif_sim - f_qif_cf) / f_qif_cf
    print(f"(b) QIF (V_peak = {V_peak:g}) f-I vs 1/T, T = (2/sqrt(I)) arctan(V_peak/sqrt(I)): "
          f"max rel. error {100*rel_b.max():.3f}%, mean {100*rel_b.mean():.3f}%")
    assert rel_b.max() < 0.01, f"QIF f-I mismatch {100*rel_b.max():.2f}% > 1%"

    # (c) equivalence: theta neuron with theta_peak = 2 arctan V_peak vs QIF
    f_theta_eq = fi_curve("theta", I_FI, theta_peak_equiv, -theta_peak_equiv)
    rel_c = np.abs(f_theta_eq - f_qif_sim) / f_qif_sim
    print(f"(c) rates: theta neuron with theta_peak = 2 arctan({V_peak:g}) = {theta_peak_equiv:.5f} rad "
          f"vs QIF with V_peak = {V_peak:g}: max rel. difference {rel_c.max():.1e} ({100*rel_c.max():.4f}%)")
    assert rel_c.max() < 0.01, f"QIF/theta rate mismatch {100*rel_c.max():.2f}% > 1%"
    gap_inf = 100 * ((f_qif_cf - f_theta_cf) / f_theta_cf).max()
    print(f"    (closed forms: the QIF with finite V_peak = {V_peak:g} fires faster than the standard "
          f"theta neuron, V_peak -> inf, by at most {gap_inf:.3f}%, because the reset skips the "
          f"finite time spent beyond |V| = V_peak)")

    # (c') one trajectory: QIF mapped to theta vs theta neuron vs exact solution
    T_one = period_qif(I_DEMO)
    T_run = 0.995 * T_one   # stop just before the spike so no reset occurs
    t, V, _ = simulate("qif", I_DEMO, T_run, V_reset, V_peak, V_reset)
    _, th, _ = simulate("theta", I_DEMO, T_run, -theta_peak_equiv, theta_peak_equiv, -theta_peak_equiv)
    th_from_V = v_to_theta(V)
    th_exact = v_to_theta(qif_exact(t, I_DEMO, V_reset))
    dev_qif_theta = np.abs(th_from_V - th).max()
    dev_qif_exact = np.abs(th_from_V - th_exact).max()
    dev_theta_exact = np.abs(th - th_exact).max()
    print(f"(c') trajectory at I = {I_DEMO}: max |2 arctan V_QIF - theta| = {dev_qif_theta:.2e} rad; "
          f"vs exact solution: QIF {dev_qif_exact:.2e}, theta {dev_theta_exact:.2e} rad")
    assert dev_qif_theta < 0.02, f"trajectory mismatch {dev_qif_theta:.3f} rad > 0.02"
    assert dev_qif_exact < 0.02 and dev_theta_exact < 0.02

    # (d) I < 0: both settle on the stable fixed point
    (v_stable, v_saddle), (th_stable, th_saddle) = fixed_points(I_REST)
    _, V_rest, sp = simulate("qif", I_REST, 60.0, 0.9 * v_saddle, V_peak, V_reset)
    _, th_rest, sp2 = simulate("theta", I_REST, 60.0, 0.9 * th_saddle, THETA_PEAK, -THETA_PEAK)
    err_v = abs(V_rest[-1] - v_stable)
    err_th = abs(th_rest[-1] - th_stable)
    print(f"(d) I = {I_REST}: fixed points V* = {v_stable:.5f} (stable), {v_saddle:.5f} (saddle); "
          f"theta* = {th_stable:.5f}, {th_saddle:.5f} rad")
    print(f"    simulated rest after 60 time units: V = {V_rest[-1]:.6f} (dev {err_v:.1e}), "
          f"theta = {th_rest[-1]:.6f} (dev {err_th:.1e}); spikes: {sp.size}, {sp2.size}")
    assert sp.size == 0 and sp2.size == 0
    assert err_v < 1e-6 and err_th < 1e-6

    results.update(f_theta_sim=f_theta_sim, f_theta_cf=f_theta_cf, f_qif_sim=f_qif_sim,
                   f_qif_cf=f_qif_cf, f_theta_eq=f_theta_eq, rel_a=rel_a, rel_b=rel_b, rel_c=rel_c,
                   traj=(t, V, th, th_from_V, th_exact), devs=(dev_qif_theta, dev_qif_exact, dev_theta_exact),
                   rest=(V_rest, th_rest, v_stable, v_saddle, th_stable, th_saddle),
                   theta_peak_equiv=theta_peak_equiv)
    return results


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(results):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: time series at I_DEMO, QIF and theta neuron, 3 spikes
    ax = axes[0, 0]
    T_show = 3.2 * period_qif(I_DEMO)
    t, V, spk = simulate("qif", I_DEMO, T_show, V_reset, V_peak, V_reset)
    thp = results["theta_peak_equiv"]
    _, th, _ = simulate("theta", I_DEMO, T_show, -thp, thp, -thp)
    ax.plot(t, V, color="black", lw=1.2, label=f"QIF V(t), V_peak = {V_peak:g}")
    for s in spk:
        ax.axvline(s, color="gray", lw=0.6, ls=":")
    ax.set_ylabel("V")
    ax.set_xlabel("time (dimensionless)")
    ax2 = ax.twinx()
    ax2.plot(t, th, color="steelblue", lw=1.2, label="theta(t), theta_peak = 2 arctan V_peak")
    ax2.plot(t, v_to_theta(V), color="orange", lw=1.0, ls="--", label="2 arctan V_QIF(t)")
    ax2.set_ylabel("theta (rad)", color="steelblue")
    ax2.tick_params(axis="y", colors="steelblue")
    ax2.set_ylim(-np.pi * 1.05, np.pi * 1.05)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(f"QIF and theta neuron at I = {I_DEMO}: same spikes, V = tan(theta/2)")
    ax.grid(alpha=0.3)

    # Panel 2: f-I curves, simulation vs closed form
    ax = axes[0, 1]
    I_dense = np.linspace(-0.5, 2.0, 500)
    ax.plot(I_dense, rate_theta_closed_form(I_dense), color="red", lw=1.5,
            label="closed form, theta neuron: f = sqrt(I)/pi")
    ax.plot(I_dense, rate_qif_closed_form(I_dense), color="green", lw=1.5, ls="--",
            label=f"closed form, QIF V_peak = {V_peak:g}: sqrt(I) / (2 arctan(V_peak/sqrt(I)))")
    ax.plot(I_FI, results["f_theta_sim"], "o", ms=5, mfc="none", color="black",
            label="simulation, theta neuron (theta_peak = pi)")
    ax.plot(I_FI, results["f_qif_sim"], "s", ms=5, mfc="none", color="steelblue",
            label=f"simulation, QIF (V_peak = {V_peak:g})")
    ax.plot(I_FI, results["f_theta_eq"], "x", ms=5, color="orange",
            label="simulation, theta neuron with theta_peak = 2 arctan V_peak")
    ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlabel("injected current I")
    ax.set_ylabel("firing rate (spikes per unit time)")
    ax.set_title(f"f-I curves: max rel. error {100*results['rel_a'].max():.3f}% (theta), "
                 f"{100*results['rel_b'].max():.3f}% (QIF); class I onset at I = 0")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: phase circle for I < 0 and I > 0 with the trajectories
    ax = axes[1, 0]
    V_rest, th_rest, v_stable, v_saddle, th_stable, th_saddle = results["rest"]
    thg = np.linspace(-np.pi, np.pi, 400)
    ax.plot(thg, f_theta(thg, I_REST), color="black", lw=1.5, label=f"dtheta/dt, I = {I_REST}")
    ax.plot(thg, f_theta(thg, I_DEMO), color="steelblue", lw=1.5, label=f"dtheta/dt, I = {I_DEMO}")
    ax.plot(thg, f_theta(thg, 0.0), color="gray", lw=1.0, ls="--", label="dtheta/dt, I = 0 (SNIC)")
    ax.axhline(0, color="gray", lw=0.8)
    ax.plot(th_stable, 0, "o", color="white", mec="black", ms=9,
            label=f"stable node theta* = -2 arctan sqrt(-I) = {th_stable:.4f}")
    ax.plot(th_saddle, 0, "o", color="black", ms=9, label=f"saddle theta* = {th_saddle:.4f}")
    ax.plot(th_rest[-1], 0, "+", color="red", ms=14, mew=2,
            label=f"simulated rest {th_rest[-1]:.4f} (dev {abs(th_rest[-1]-th_stable):.1e})")
    ax.set_xlabel("theta (rad)")
    ax.set_ylabel("dtheta/dt")
    ax.set_xlim(-np.pi, np.pi)
    ax.set_title("Theta-neuron vector field: fixed points below I = 0, none above")
    ax.legend(fontsize=8, loc="upper center")
    ax.grid(alpha=0.3)

    # Panel 4: equivalence of one trajectory, residuals vs exact solution
    ax = axes[1, 1]
    t1, V1, th1, th_from_V, th_exact = results["traj"]
    d1, d2, d3 = results["devs"]
    ax.plot(t1, th_from_V - th1, color="orange", lw=1.2,
            label=f"2 arctan V_QIF - theta_sim   (max {d1:.1e})")
    ax.plot(t1, th_from_V - th_exact, color="black", lw=1.0, ls="--",
            label=f"2 arctan V_QIF - theta_exact   (max {d2:.1e})")
    ax.plot(t1, th1 - th_exact, color="steelblue", lw=1.0, ls=":",
            label=f"theta_sim - theta_exact   (max {d3:.1e})")
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xlabel("time since reset (dimensionless)")
    ax.set_ylabel("residual (rad)")
    ax.set_title(f"Change of variables, one inter-spike interval at I = {I_DEMO}, dt = {dt:g}")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "quadratic_if_theta.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
