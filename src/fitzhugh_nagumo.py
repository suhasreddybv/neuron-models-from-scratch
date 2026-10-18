"""FitzHugh-Nagumo model of an excitable membrane.

Equations (FitzHugh 1961, eq. 1; Nagumo, Arimoto & Yoshizawa 1962), in the
dimensionless form used by Izhikevich (2007, sec. 4.2.4):

    dv/dt = v - v^3/3 - w + I
    dw/dt = eps * (v + a - b*w)          with eps = 1/tau

v is a voltage-like fast variable, w a slow recovery variable, I the injected
current. Parameters are FitzHugh's: a = 0.7, b = 0.8, tau = 12.5 (eps = 0.08).

Integrator: forward Euler with dt = 0.005 (dimensionless time units), the same
scheme as the LIF and HH scripts in this repository.

Validation (see ROADMAP.md note): the nullclines and the fixed point follow
from the equations. The v-nullcline is w = v - v^3/3 + I and the w-nullcline
is w = (v + a)/b. Their intersection solves the cubic

    v^3/3 + (1/b - 1) v + a/b - I = 0,

which has exactly one real root here because 1/b - 1 = 0.25 > 0. The Jacobian
at the fixed point (v*, w*) is

    J = [[1 - v*^2, -1], [eps, -eps*b]],

trace = 1 - v*^2 - eps*b, det = eps*(1 - b*(1 - v*^2)) > 0. A Hopf
bifurcation occurs where the trace crosses zero, i.e. v*^2 = 1 - eps*b, which
gives two predicted currents I_H = v*^3/3 - v* + (v* + a)/b with
v* = -/+ sqrt(1 - eps*b): the rest state loses stability at the lower one and
the depolarised ("excitation block") state regains it at the upper one.

The script finds both onset currents independently by sweeping I in the
simulation (bisection on whether a small perturbation of the fixed point grows
or decays) and asserts that they agree with the Jacobian prediction to within
1%. It also asserts that the simulated resting potential at I = 0 matches the
cubic root to 1e-3.

Note on the sweep: forward Euler on a neutrally stable centre is weakly
unstable (growth ~ dt*omega^2/2 per unit time), which widens the detected
oscillatory window by about dt*omega^2/(d trace/dI) ~ 2e-4 at each end at
dt = 0.005 (lower onset found slightly low, upper slightly high). This is well
inside the stated tolerance and is reported, not hidden.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (FitzHugh 1961)
# ---------------------------------------------------------------------------
a = 0.7
b = 0.8
tau = 12.5
eps = 1.0 / tau

dt = 0.005           # Euler time step (dimensionless time)
SEED = 0             # RNG seed for the initial perturbation in the sweep
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions, nullclines, fixed point, Jacobian
# ---------------------------------------------------------------------------
def f(v, w, I):
    """Right-hand side of the FitzHugh-Nagumo equations."""
    dv = v - v ** 3 / 3.0 - w + I
    dw = eps * (v + a - b * w)
    return dv, dw


def v_nullcline(v, I):
    return v - v ** 3 / 3.0 + I


def w_nullcline(v):
    return (v + a) / b


def fixed_point(I):
    """Unique real root of v^3/3 + (1/b - 1) v + a/b - I = 0, and w* = (v*+a)/b."""
    roots = np.roots([1.0 / 3.0, 0.0, 1.0 / b - 1.0, a / b - I])
    real = roots[np.abs(roots.imag) < 1e-9].real
    assert real.size == 1, f"expected one real root, got {real}"
    v_star = float(real[0])
    return v_star, w_nullcline(v_star)


def jacobian(v_star):
    return np.array([[1.0 - v_star ** 2, -1.0], [eps, -eps * b]])


def hopf_currents_predicted():
    """Currents where trace(J) = 0, i.e. v*^2 = 1 - eps*b (lower, upper)."""
    v_h = np.sqrt(1.0 - eps * b)
    out = []
    for v_star in (-v_h, v_h):
        I_h = v_star ** 3 / 3.0 - v_star + (v_star + a) / b
        out.append(I_h)
    return tuple(out)


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(I, T=300.0, v0=None, w0=None, dt=dt):
    """Forward-Euler simulation of a single FHN cell with constant current I.

    I may be a scalar or a 1-D array; in the array case all currents are
    integrated in lock-step (used by the Hopf sweep). Returns time, v(t), w(t)
    with v, w of shape (n_steps,) or (n_steps, len(I)).
    """
    I = np.asarray(I, dtype=float)
    n = int(round(T / dt))
    time = np.arange(n) * dt
    shape = (n,) + I.shape
    v = np.zeros(shape)
    w = np.zeros(shape)
    if v0 is None:
        v0, w0 = fixed_point(0.0)
    v[0] = v0
    w[0] = w0
    for i in range(1, n):
        dv, dw = f(v[i - 1], w[i - 1], I)
        v[i] = v[i - 1] + dt * dv
        w[i] = w[i - 1] + dt * dw
    return time, v, w


def oscillates(I_values, T=3000.0):
    """For each current, start 1e-3 away from its fixed point and decide whether
    the perturbation grows (True) or decays (False) by comparing the peak-to-peak
    amplitude of v in the third and fourth quarters of the run."""
    I_values = np.asarray(I_values, dtype=float)
    v_star = np.array([fixed_point(I)[0] for I in I_values])
    w_star = w_nullcline(v_star)
    pert = 1e-3 * rng.standard_normal(I_values.shape)
    _, v, _ = simulate(I_values, T=T, v0=v_star + pert, w0=w_star)
    n = v.shape[0]
    q3 = v[n // 2: 3 * n // 4]
    q4 = v[3 * n // 4:]
    amp3 = q3.max(axis=0) - q3.min(axis=0)
    amp4 = q4.max(axis=0) - q4.min(axis=0)
    # growing perturbation, or an already-established limit cycle
    return (amp4 > amp3) | (amp4 > 0.05), amp4


def find_hopf_by_sweep(I_lo, I_hi, n_grid=48, n_refine=2):
    """Locate the current at which oscillation switches on/off between I_lo and
    I_hi by repeated grid refinement (vectorised bisection). Returns the
    midpoint of the final bracket and its half-width."""
    lo, hi = I_lo, I_hi
    for _ in range(n_refine + 1):
        grid = np.linspace(lo, hi, n_grid)
        osc, _ = oscillates(grid)
        change = np.nonzero(osc[1:] != osc[:-1])[0]
        assert change.size >= 1, f"no transition found in [{lo}, {hi}]"
        k = change[0]
        lo, hi = grid[k], grid[k + 1]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    results = {}

    # (a) resting state at I = 0: simulation vs cubic root
    v_star, w_star = fixed_point(0.0)
    _, v, w = simulate(0.0, T=400.0, v0=-1.5, w0=-0.5)
    v_sim, w_sim = v[-1], w[-1]
    err_fp = max(abs(v_sim - v_star), abs(w_sim - w_star))
    J = jacobian(v_star)
    eig = np.linalg.eigvals(J)
    print(f"Fixed point at I=0 (cubic root): v* = {v_star:.5f}, w* = {w_star:.5f}")
    print(f"Simulated rest after 400 time units: v = {v_sim:.5f}, w = {w_sim:.5f}; "
          f"max deviation {err_fp:.2e}")
    print(f"Jacobian eigenvalues at I=0: {eig[0]:.4f}, {eig[1]:.4f} "
          f"(trace {np.trace(J):.4f}, det {np.linalg.det(J):.4f})")
    assert err_fp < 1e-3, f"rest state deviates from cubic root by {err_fp:.2e}"
    results["fixed_point"] = (v_star, w_star, v_sim, w_sim, err_fp)

    # (b) Hopf currents: sweep vs Jacobian
    I_h_lo_pred, I_h_hi_pred = hopf_currents_predicted()
    I_h_lo_sweep, hw_lo = find_hopf_by_sweep(0.0, 0.8)
    I_h_hi_sweep, hw_hi = find_hopf_by_sweep(1.0, 1.8)
    rel_lo = abs(I_h_lo_sweep - I_h_lo_pred) / abs(I_h_lo_pred)
    rel_hi = abs(I_h_hi_sweep - I_h_hi_pred) / abs(I_h_hi_pred)
    print(f"Lower Hopf: Jacobian predicts I_H = {I_h_lo_pred:.5f}; "
          f"sweep finds {I_h_lo_sweep:.5f} +/- {hw_lo:.1e}; relative error {100*rel_lo:.3f}%")
    print(f"Upper Hopf: Jacobian predicts I_H = {I_h_hi_pred:.5f}; "
          f"sweep finds {I_h_hi_sweep:.5f} +/- {hw_hi:.1e}; relative error {100*rel_hi:.3f}%")
    assert rel_lo < 0.01, f"lower Hopf mismatch {100*rel_lo:.2f}% > 1%"
    assert rel_hi < 0.01, f"upper Hopf mismatch {100*rel_hi:.2f}% > 1%"
    results["hopf"] = (I_h_lo_pred, I_h_lo_sweep, rel_lo, I_h_hi_pred, I_h_hi_sweep, rel_hi)
    return results


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(results):
    I_h_lo_pred, I_h_lo_sweep, _, I_h_hi_pred, I_h_hi_sweep, _ = results["hopf"]

    # Time series and phase plane at a current inside the oscillatory window
    I_demo = 0.5
    time, v, w = simulate(I_demo, T=200.0, v0=-1.2, w0=-0.6)

    # Amplitude vs I for the bifurcation panel (coarse, steady-state amplitude)
    I_scan = np.linspace(0.0, 1.8, 91)
    osc, amp = oscillates(I_scan, T=1500.0)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: time series
    ax = axes[0]
    ax.plot(time, v, color="black", lw=1.2, label="v (fast)")
    ax.plot(time, w, color="steelblue", lw=1.2, label="w (slow)")
    ax.set_xlabel("time (dimensionless)")
    ax.set_ylabel("v, w")
    ax.set_title(f"FitzHugh-Nagumo, I = {I_demo}: relaxation oscillation")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: phase plane with nullclines
    ax = axes[1]
    vv = np.linspace(-2.5, 2.5, 400)
    ax.plot(vv, v_nullcline(vv, I_demo), color="red", lw=1.5, label="v-nullcline  w = v - v³/3 + I")
    ax.plot(vv, w_nullcline(vv), color="green", lw=1.5, label="w-nullcline  w = (v + a)/b")
    ax.plot(v, w, color="black", lw=0.9, alpha=0.8, label="trajectory")
    vs, ws = fixed_point(I_demo)
    ax.plot(vs, ws, "o", color="orange", ms=8, mec="black", label=f"fixed point ({vs:.2f}, {ws:.2f}), unstable")
    vs0, ws0 = fixed_point(0.0)
    ax.plot(vs0, ws0, "s", color="white", ms=7, mec="black", label=f"rest at I=0 ({vs0:.2f}, {ws0:.2f}), stable")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-1.0, 2.5)
    ax.set_xlabel("v")
    ax.set_ylabel("w")
    ax.set_title("Phase plane and nullclines")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: bifurcation diagram, sweep vs Jacobian prediction
    ax = axes[2]
    ax.plot(I_scan, amp, "o-", ms=3, color="steelblue", lw=1, label="peak-to-peak v (simulation)")
    for x, c, lab in ((I_h_lo_pred, "red", f"Jacobian I_H = {I_h_lo_pred:.4f}, {I_h_hi_pred:.4f}"),
                      (I_h_hi_pred, "red", None)):
        ax.axvline(x, color=c, ls="--", lw=1.5, label=lab)
    for x, lab in ((I_h_lo_sweep, f"sweep I_H = {I_h_lo_sweep:.4f}, {I_h_hi_sweep:.4f}"),
                   (I_h_hi_sweep, None)):
        ax.axvline(x, color="black", ls=":", lw=1.5, label=lab)
    # real part of the leading eigenvalue on a twin axis
    ax2 = ax.twinx()
    re_lead = []
    for I in I_scan:
        eig = np.linalg.eigvals(jacobian(fixed_point(I)[0]))
        re_lead.append(eig.real.max())
    ax2.plot(I_scan, re_lead, color="gray", lw=1, alpha=0.7)
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_ylabel("max Re(eigenvalue) of Jacobian", color="gray")
    ax2.tick_params(axis="y", colors="gray")
    ax.set_xlabel("injected current I")
    ax.set_ylabel("oscillation amplitude (peak-to-peak v)")
    ax.set_title("Hopf onset: sweep vs linearised prediction")
    ax.legend(fontsize=8, loc="center", framealpha=0.9)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "fitzhugh_nagumo.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
