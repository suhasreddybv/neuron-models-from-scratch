"""Wilson-Cowan excitatory-inhibitory population model.

Equations (Wilson & Cowan 1972, eqs. 11-12, with the refractory factor kept),
time measured in units of the population time constant tau = tau_e = tau_i:

    dE/dt = -E + (k_e - r_e E) S_e(c1 E - c2 I + P)
    dI/dt = -I + (k_i - r_i I) S_i(c3 E - c4 I + Q)

    S_j(x) = 1 / (1 + exp(-a_j (x - theta_j))) - 1 / (1 + exp(a_j theta_j))

E and I are the fractions of excitatory and inhibitory cells firing per unit
time, P and Q the external inputs. S_j is shifted so that S_j(0) = 0, and
k_j = 1 - 1/(1 + exp(a_j theta_j)) is its supremum. Parameters are the
limit-cycle set of Wilson & Cowan (1972), also used by Ermentrout & Terman
(2010, ch. 11): c1 = 16, c2 = 12, c3 = 15, c4 = 3, a_e = 1.3, theta_e = 4,
a_i = 2, theta_i = 3.7, r_e = r_i = 1, Q = 0. P is the bifurcation parameter.

Integrator: classical fourth-order Runge-Kutta, dt = 0.02 tau.

Validation (see ROADMAP.md note):
(1) Fixed points in closed form. On the I-nullcline I(E) is the unique root
    of -I + (k_i - I) S_i(c3 E - c4 I) = 0 (left side strictly decreasing in
    I); the E-equation then gives P as an explicit function of E,
        P(E) = S_e^-1(E / (k_e - E)) - c1 E + c2 I(E),
    so the whole fixed-point branch is a parametric curve (E, I(E), P(E))
    with no root finding in P. Folds are the extrema of P(E); the Jacobian
    J(E, I, P) is written out analytically.
(2) Hopf point. On the upper branch trace J = 0 with det J > 0 gives the
    Hopf input P_H and the linearised angular frequency omega_H = sqrt(det J).
(3) The simulation is checked against this three ways: (a) small (1e-9 on
    the growing side, 1e-6 on the decaying side) perturbations of the fixed point at 12 inputs on both sides of P_H decay
    or grow as exp(sigma t) cos(omega t), with sigma and omega measured from
    successive extrema and zero crossings and compared with Re, Im of the
    Jacobian eigenvalues; (b) the zero of the measured sigma(P) gives P_H
    from simulation; (c) the limit-cycle frequency at P = P_H - delta,
    delta = 0.0025 ... 0.08, extrapolated to delta -> 0 by a quadratic fit,
    is compared with omega_H / 2 pi, and the squared amplitude must scale as
    delta^1 (supercritical Hopf). (4) RK4 limit-cycle periods are compared
    with solve_ivp (DOP853, rtol = atol = 1e-12).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq          # validation / closed forms only
from scipy.integrate import solve_ivp      # reference solver only

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Wilson & Cowan 1972, limit-cycle set)
# ---------------------------------------------------------------------------
c1, c2, c3, c4 = 16.0, 12.0, 15.0, 3.0
a_e, theta_e = 1.3, 4.0
a_i, theta_i = 2.0, 3.7
r_e, r_i = 1.0, 1.0
Q = 0.0
s0_e = 1.0 / (1.0 + np.exp(a_e * theta_e))     # S_e offset so S_e(0) = 0
s0_i = 1.0 / (1.0 + np.exp(a_i * theta_i))
k_e = 1.0 - s0_e
k_i = 1.0 - s0_i

dt = 0.02            # RK4 step, in units of tau
P_DEMO = 1.25        # the input used for the time-series / phase-plane panels
SEED = 1972
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED} (used only for the directions of the initial perturbations)")


# ---------------------------------------------------------------------------
# Block 2: Model functions, nullclines, fixed points, Jacobian
# ---------------------------------------------------------------------------
def S(x, a, theta):
    return 1.0 / (1.0 + np.exp(-a * (x - theta))) - 1.0 / (1.0 + np.exp(a * theta))


def dS(x, a, theta):
    s = 1.0 / (1.0 + np.exp(-a * (x - theta)))
    return a * s * (1.0 - s)


def S_inv(y, a, theta, s0):
    return theta - np.log(1.0 / (y + s0) - 1.0) / a


def rhs(E, I, P):
    dE = -E + (k_e - r_e * E) * S(c1 * E - c2 * I + P, a_e, theta_e)
    dI = -I + (k_i - r_i * I) * S(c3 * E - c4 * I + Q, a_i, theta_i)
    return dE, dI


def E_nullcline(E, P):
    """I on the E-nullcline as an explicit function of E."""
    return (c1 * E + P - S_inv(E / (k_e - r_e * E), a_e, theta_e, s0_e)) / c2


def I_nullcline_E(I):
    """E on the I-nullcline as an explicit function of I."""
    return (S_inv(I / (k_i - r_i * I), a_i, theta_i, s0_i) + c4 * I - Q) / c3


def I_of_E(E):
    """Unique I on the I-nullcline for a given E."""
    return brentq(lambda I: -I + (k_i - r_i * I) * S(c3 * E - c4 * I + Q, a_i, theta_i),
                  -0.5, 1.0, xtol=1e-15, rtol=1e-15)


def P_of_E(E):
    """Input P at which (E, I(E)) is a fixed point."""
    I = I_of_E(E)
    return S_inv(E / (k_e - r_e * E), a_e, theta_e, s0_e) - c1 * E + c2 * I


def jacobian(E, I, P):
    xe = c1 * E - c2 * I + P
    xi = c3 * E - c4 * I + Q
    ge, gi = dS(xe, a_e, theta_e), dS(xi, a_i, theta_i)
    return np.array([
        [-1.0 - r_e * S(xe, a_e, theta_e) + (k_e - r_e * E) * ge * c1, -(k_e - r_e * E) * ge * c2],
        [(k_i - r_i * I) * gi * c3, -1.0 - r_i * S(xi, a_i, theta_i) - (k_i - r_i * I) * gi * c4],
    ])


def branch(E_max=0.45, n=4000):
    """Fixed-point branch as a parametric curve in E."""
    Es = np.linspace(1e-5, E_max, n)
    Is = np.array([I_of_E(e) for e in Es])
    Ps = np.array([P_of_E(e) for e in Es])
    return Es, Is, Ps


def folds():
    """Extrema of P(E): saddle-node points (E, P)."""
    Es, _, Ps = branch()
    d = np.diff(Ps)
    out = []
    for k in np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0]:
        dPdE = lambda e: (P_of_E(e + 1e-7) - P_of_E(e - 1e-7)) / 2e-7
        e = brentq(dPdE, Es[k], Es[k + 2])
        out.append((e, P_of_E(e)))
    return out


def fixed_point_upper(P):
    """Fixed point on the upper (E > second fold) branch at input P."""
    E_fold_hi = folds()[1][0]
    E = brentq(lambda e: P_of_E(e) - P, E_fold_hi, 0.45, xtol=1e-15, rtol=1e-15)
    return E, I_of_E(E)


def hopf_point():
    """Hopf point on the upper branch: trace J = 0. Returns E, I, P, omega."""
    E_fold_hi = folds()[1][0]

    def tr(e):
        return np.trace(jacobian(e, I_of_E(e), P_of_E(e)))

    E = brentq(tr, 0.2, 0.3, xtol=1e-15, rtol=1e-15)
    assert E > E_fold_hi
    I, P = I_of_E(E), P_of_E(E)
    det = np.linalg.det(jacobian(E, I, P))
    assert det > 0
    return E, I, P, np.sqrt(det)


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(P, T, E0, I0, dt=dt, record_every=1):
    """RK4 integration with constant input P (scalar or array, run in lock-step).
    Returns t, E(t), I(t) sampled every `record_every` steps."""
    P = np.asarray(P, dtype=float)
    E = np.broadcast_to(np.asarray(E0, dtype=float), P.shape).copy()
    I = np.broadcast_to(np.asarray(I0, dtype=float), P.shape).copy()
    n = int(round(T / dt))
    n_rec = n // record_every + 1
    tr = np.arange(n_rec) * dt * record_every
    Er = np.empty((n_rec,) + P.shape)
    Ir = np.empty((n_rec,) + P.shape)
    Er[0], Ir[0] = E, I
    h = 0.5 * dt
    for k in range(1, n + 1):
        a1, b1 = rhs(E, I, P)
        a2, b2 = rhs(E + h * a1, I + h * b1, P)
        a3, b3 = rhs(E + h * a2, I + h * b2, P)
        a4, b4 = rhs(E + dt * a3, I + dt * b3, P)
        E = E + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4)
        I = I + dt / 6.0 * (b1 + 2 * b2 + 2 * b3 + b4)
        if k % record_every == 0:
            Er[k // record_every], Ir[k // record_every] = E, I
    return tr, Er, Ir


def crossings(t, x, level):
    """Upward crossing times of x through level, linearly interpolated."""
    s = x - level
    k = np.nonzero((s[:-1] < 0) & (s[1:] >= 0))[0]
    return t[k] - s[k] * (t[k + 1] - t[k]) / (s[k + 1] - s[k])


def extrema(t, x):
    """Local maxima of x (parabolic interpolation): times and values."""
    k = np.nonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    y0, y1, y2 = x[k - 1], x[k], x[k + 1]
    den = y0 - 2 * y1 + y2
    off = 0.5 * (y0 - y2) / den
    h = t[1] - t[0]
    return t[k] + off * h, y1 - 0.25 * (y0 - y2) * off


def growth_and_frequency(t, x):
    """For x = A exp(sigma t) cos(omega t + phi): sigma from successive maxima,
    omega from successive upward zero crossings."""
    tz = crossings(t, x, 0.0)
    omega = 2 * np.pi / np.mean(np.diff(tz))
    tm, ym = extrema(t, x)
    ym = ym[ym > 0]
    tm = tm[-len(ym):] if len(ym) else tm
    sigma = np.polyfit(tm, np.log(ym), 1)[0]
    return sigma, omega


def cycle_stats(t, E):
    """Frequency (cycles per tau) and peak-to-peak amplitude of a converged cycle."""
    level = 0.5 * (E.max() + E.min())
    tc = crossings(t, E, level)
    return 1.0 / np.mean(np.diff(tc)), E.max() - E.min()


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    R = {}
    (E_sn1, P_sn1), (E_sn2, P_sn2) = folds()
    E_H, I_H, P_H, w_H = hopf_point()
    f_H = w_H / (2 * np.pi)
    R.update(P_sn=(P_sn1, P_sn2), hopf=(E_H, I_H, P_H, w_H))
    print(f"Folds of the fixed-point branch: P = {P_sn1:.6f} (E = {E_sn1:.5f}) and "
          f"P = {P_sn2:.6f} (E = {E_sn2:.5f}); three fixed points for {P_sn2:.4f} < P < {P_sn1:.4f}")
    print(f"Hopf (trace J = 0): P_H = {P_H:.6f}, (E, I) = ({E_H:.6f}, {I_H:.6f}), "
          f"omega_H = sqrt(det J) = {w_H:.6f}, f_H = {f_H:.6f} cycles/tau")

    # (1) simulated rest states vs closed-form fixed points
    P_rest = np.array([0.5, 2.2, 2.6])
    E_star = np.array([brentq(lambda e: P_of_E(e) - p, 1e-6, E_sn1) if p < P_sn2 else fixed_point_upper(p)[0]
                       for p in P_rest])
    I_star = np.array([I_of_E(e) for e in E_star])
    _, Es, Is = simulate(P_rest, 400.0, 0.1, 0.05, record_every=20000)
    err = np.max(np.abs(np.r_[Es[-1] - E_star, Is[-1] - I_star]))
    print(f"Rest states at P = {P_rest}: max |sim - closed form| = {err:.2e}")
    assert err < 1e-9, f"fixed points deviate by {err:.2e}"
    R["fp_err"] = err

    # (2) linear regime: sigma, omega from small perturbations vs Jacobian
    dP = np.array([-0.6, -0.3, -0.15, -0.08, -0.04, -0.02, 0.02, 0.04, 0.08, 0.15, 0.3, 0.6])
    P_lin = P_H + dP
    fp = np.array([fixed_point_upper(p) for p in P_lin])
    ang = rng.uniform(0, 2 * np.pi, P_lin.shape)
    # growing cases start smaller so that exp(sigma T) * eps0 stays below 1e-5
    eps0 = np.where(dP < 0, 1e-9, 1e-6)
    T_lin = 80.0
    t, Es, _ = simulate(P_lin, T_lin, fp[:, 0] + eps0 * np.cos(ang), fp[:, 1] + eps0 * np.sin(ang))
    sig_sim, om_sim, sig_th, om_th = [], [], [], []
    for j, p in enumerate(P_lin):
        s, w = growth_and_frequency(t, Es[:, j] - fp[j, 0])
        lam = np.linalg.eigvals(jacobian(fp[j, 0], fp[j, 1], p))
        lam = lam[np.argmax(lam.imag)]
        sig_sim.append(s); om_sim.append(w); sig_th.append(lam.real); om_th.append(lam.imag)
    sig_sim, om_sim, sig_th, om_th = map(np.array, (sig_sim, om_sim, sig_th, om_th))
    err_sig = np.max(np.abs(sig_sim - sig_th))
    err_om = np.max(np.abs(om_sim - om_th) / om_th)
    print("Linear regime (perturbation 1e-9 below P_H, 1e-6 above; T = 80):")
    for j in range(len(P_lin)):
        print(f"  P = {P_lin[j]:.4f}: sigma {sig_sim[j]:+.6f} (sim) vs {sig_th[j]:+.6f} (Jacobian); "
              f"omega {om_sim[j]:.6f} vs {om_th[j]:.6f}")
    print(f"  max |sigma error| = {err_sig:.2e}; max relative omega error = {err_om:.2e}")
    assert err_sig < 1e-4, f"growth-rate error {err_sig:.2e}"
    assert err_om < 1e-4, f"frequency error {err_om:.2e}"
    R["lin"] = (P_lin, sig_sim, om_sim, sig_th, om_th, err_sig, err_om)

    # (3) Hopf point from simulation: zero of measured sigma(P) (closest pair)
    k = np.nonzero(np.sign(sig_sim[1:]) != np.sign(sig_sim[:-1]))[0]
    assert k.size == 1
    k = k[0]
    P_H_sim = P_lin[k] - sig_sim[k] * (P_lin[k + 1] - P_lin[k]) / (sig_sim[k + 1] - sig_sim[k])
    rel_PH = abs(P_H_sim - P_H) / P_H
    print(f"Hopf from simulated sigma(P) (linear interpolation over +/-0.02): P_H = {P_H_sim:.6f} "
          f"vs {P_H:.6f}, relative difference {rel_PH:.2e}")
    assert rel_PH < 1e-4, f"Hopf point mismatch {rel_PH:.2e}"
    R["P_H_sim"] = P_H_sim

    # (4) limit-cycle frequency near the Hopf point vs omega_H / 2 pi
    deltas = np.array([0.0025, 0.005, 0.01, 0.02, 0.04, 0.08])
    P_lc = P_H - deltas
    fp = np.array([fixed_point_upper(p) for p in P_lc])
    T_lc, T_meas = 6000.0, 400.0
    t, Es, _ = simulate(P_lc, T_lc, fp[:, 0] + 0.1 * np.sqrt(deltas), fp[:, 1], record_every=1)
    m = t >= T_lc - T_meas
    f_lc, amp_lc = np.array([cycle_stats(t[m], Es[m, j]) for j in range(len(P_lc))]).T
    # convergence of the amplitude: compare with the previous window
    m2 = (t >= T_lc - 2 * T_meas) & (t < T_lc - T_meas)
    amp_prev = np.array([cycle_stats(t[m2], Es[m2, j])[1] for j in range(len(P_lc))])
    coef = np.polyfit(deltas, f_lc, 2)
    f0 = coef[-1]
    rel_f = abs(f0 - f_H) / f_H
    slope_amp = np.polyfit(np.log(deltas[:4]), np.log(amp_lc[:4] ** 2), 1)[0]
    print("Limit cycle at P = P_H - delta (T = 6000, last 400 measured):")
    for j in range(len(deltas)):
        print(f"  delta = {deltas[j]:.4f}: f = {f_lc[j]:.6f} cycles/tau, peak-to-peak E = {amp_lc[j]:.5f} "
              f"(previous window {amp_prev[j]:.5f})")
    print(f"  quadratic extrapolation f(delta -> 0) = {f0:.6f} vs omega_H/2pi = {f_H:.6f}; "
          f"relative error {rel_f:.2e}")
    print(f"  log-log slope of amplitude^2 vs delta (delta <= 0.02) = {slope_amp:.4f} (supercritical: 1)")
    print(f"  at delta = 0.0025 the cycle frequency differs from f_H by {abs(f_lc[0]-f_H)/f_H:.2e}")
    assert np.max(np.abs(amp_lc - amp_prev) / amp_lc) < 1e-3, "limit cycles not converged"
    assert rel_f < 1e-3, f"limit-cycle frequency at onset off by {rel_f:.2e}"
    assert abs(slope_amp - 1.0) < 0.05, f"amplitude scaling slope {slope_amp:.3f}"
    R["lc"] = (deltas, f_lc, amp_lc, coef, f0, rel_f, slope_amp)

    # (5) RK4 vs DOP853 limit-cycle period
    rk_T, ref_T = [], []
    for p in (P_DEMO, P_H - 0.04):
        t, Es, Is = simulate(p, 300.0, 0.3, 0.2)
        rk_T.append(1.0 / cycle_stats(t[t > 150], Es[t > 150])[0])
        sol = solve_ivp(lambda tt, y: rhs(y[0], y[1], p), (0, 300.0), [0.3, 0.2], method="DOP853",
                        rtol=1e-12, atol=1e-12, dense_output=True)
        tt = np.arange(150.0, 300.0, 0.002)
        ref_T.append(1.0 / cycle_stats(tt, sol.sol(tt)[0])[0])
    rk_T, ref_T = np.array(rk_T), np.array(ref_T)
    rel_T = np.max(np.abs(rk_T - ref_T) / ref_T)
    print(f"Limit-cycle period, RK4 dt = {dt} vs DOP853: P = {P_DEMO}: {rk_T[0]:.8f} vs {ref_T[0]:.8f}; "
          f"P = P_H - 0.04: {rk_T[1]:.8f} vs {ref_T[1]:.8f}; max relative difference {rel_T:.2e}")
    assert rel_T < 1e-6, f"RK4 period error {rel_T:.2e}"
    R["rk_T"] = (rk_T, ref_T, rel_T)
    return R


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(R):
    E_H, I_H, P_H, w_H = R["hopf"]
    f_H = w_H / (2 * np.pi)
    P_sn1, P_sn2 = R["P_sn"]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # Panel 1: time series at P_DEMO
    t, Es, Is = simulate(P_DEMO, 60.0, 0.05, 0.05)
    ax = axes[0, 0]
    ax.plot(t, Es, color="firebrick", lw=1.3, label="E")
    ax.plot(t, Is, color="steelblue", lw=1.3, label="I")
    ax.set_xlabel("time (units of τ)")
    ax.set_ylabel("fraction active")
    ax.set_title(f"Wilson–Cowan, P = {P_DEMO}: E–I oscillation")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: phase plane with nullclines
    ax = axes[0, 1]
    Eg = np.linspace(1e-4, 0.49, 800)
    ax.plot(Eg, E_nullcline(Eg, P_DEMO), color="firebrick", lw=1.5, label="E-nullcline")
    Ig = np.linspace(1e-5, 0.49, 800)
    ax.plot(I_nullcline_E(Ig), Ig, color="steelblue", lw=1.5, label="I-nullcline")
    m = t > 30
    ax.plot(Es[m], Is[m], color="black", lw=1.0, label="limit cycle")
    E0, I0 = fixed_point_upper(P_DEMO)
    ax.plot(E0, I0, "o", color="orange", mec="black", ms=8, label=f"fixed point ({E0:.3f}, {I0:.3f}), unstable")
    ax.set_xlim(-0.02, 0.5)
    ax.set_ylim(-0.02, 0.5)
    ax.set_xlabel("E")
    ax.set_ylabel("I")
    ax.set_title(f"Phase plane and nullclines, P = {P_DEMO}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: bifurcation diagram
    ax = axes[0, 2]
    Eb, Ib, Pb = branch()
    stab = np.array([np.linalg.eigvals(jacobian(e, i, p)).real.max() < 0 for e, i, p in zip(Eb, Ib, Pb)])
    Pm = np.where(stab, Pb, np.nan)
    Pu = np.where(~stab, Pb, np.nan)
    ax.plot(Pm, Eb, color="black", lw=2, label="stable fixed point")
    ax.plot(Pu, Eb, color="black", lw=1, ls="--", label="unstable fixed point")
    P_scan = np.linspace(0.8, 2.4, 81)
    t, Es, _ = simulate(P_scan, 400.0, 0.45, 0.1, record_every=5)
    mm = t > 300
    ax.plot(P_scan, Es[mm].max(0), "o", ms=3, color="firebrick", label="max / min E on attractor (sim)")
    ax.plot(P_scan, Es[mm].min(0), "o", ms=3, color="firebrick")
    ax.axvline(P_H, color="red", ls=":", lw=1.2, label=f"Hopf P_H = {P_H:.4f}")
    ax.axvline(P_sn1, color="gray", ls=":", lw=1.2, label=f"folds P = {P_sn2:.3f}, {P_sn1:.3f}")
    ax.axvline(P_sn2, color="gray", ls=":", lw=1.2)
    ax.set_xlim(0.8, 2.4)
    ax.set_ylim(-0.02, 0.5)
    ax.set_xlabel("external input P")
    ax.set_ylabel("E")
    ax.set_title("Bifurcation diagram in P")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 4: growth rate, sim vs Jacobian
    P_lin, sig_sim, om_sim, sig_th, om_th, err_sig, err_om = R["lin"]
    Pc = np.linspace(P_H - 0.65, P_H + 0.65, 200)
    lam = np.array([np.linalg.eigvals(jacobian(*fixed_point_upper(p), p)) for p in Pc])
    lam_r = lam.real.max(1)
    lam_i = np.abs(lam.imag).max(1)
    ax = axes[1, 0]
    ax.plot(Pc, lam_r, color="black", lw=1.2, label="Re λ (Jacobian)")
    ax.plot(P_lin, sig_sim, "o", color="firebrick", ms=5, mfc="none", label="σ from simulation")
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(P_H, color="red", ls=":", lw=1.2)
    ax.set_xlabel("P")
    ax.set_ylabel("growth rate (1/τ)")
    ax.set_title(f"Linear growth rate: max |error| {err_sig:.1e}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Panel 5: frequency vs P, linear prediction and limit cycles
    deltas, f_lc, amp_lc, coef, f0, rel_f, slope_amp = R["lc"]
    ax = axes[1, 1]
    ax.plot(Pc, lam_i / (2 * np.pi), color="black", lw=1.2, label="Im λ / 2π (Jacobian)")
    ax.plot(P_lin, om_sim / (2 * np.pi), "o", color="steelblue", ms=5, mfc="none",
            label="damped / growing oscillation (sim)")
    P_cyc = np.linspace(0.95, P_H - 0.0025, 60)
    t, Es, _ = simulate(P_cyc, 400.0, 0.45, 0.1, record_every=1)
    mm = t > 250
    f_cyc = []
    for j in range(len(P_cyc)):
        e = Es[mm, j]
        f_cyc.append(cycle_stats(t[mm], e)[0] if e.max() - e.min() > 1e-3 else np.nan)
    ax.plot(P_cyc, f_cyc, color="firebrick", lw=1.5, label="limit-cycle frequency (sim)")
    ax.plot(P_H - deltas, f_lc, "s", color="firebrick", ms=4)
    ax.plot(P_H, f_H, "*", color="red", ms=14, mec="black", label=f"ω_H/2π = {f_H:.5f}")
    ax.axvline(P_H, color="red", ls=":", lw=1.2)
    ax.set_xlabel("P")
    ax.set_ylabel("frequency (cycles per τ)")
    ax.set_title("Oscillation frequency vs linearised prediction")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    # Panel 6: near-onset zoom: f(delta) and amplitude^2
    ax = axes[1, 2]
    dd = np.linspace(0, deltas.max(), 100)
    ax.plot(deltas, f_lc, "s", color="firebrick", label="limit-cycle f (sim)")
    ax.plot(dd, np.polyval(coef, dd), color="firebrick", lw=1, label="quadratic fit")
    ax.plot(0, f_H, "*", color="red", ms=14, mec="black",
            label=f"ω_H/2π; extrapolation off by {rel_f:.1e}")
    ax.set_xlabel("δ = P_H − P")
    ax.set_ylabel("frequency (cycles per τ)")
    ax.set_title("Onset frequency and amplitude scaling")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.6, 1.0))
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(deltas, amp_lc ** 2, "o-", color="gray", ms=4, lw=0.8)
    ax2.set_ylabel(f"(peak-to-peak E)²  — slope vs δ {slope_amp:.3f}", color="gray")
    ax2.tick_params(axis="y", colors="gray")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "wilson_cowan.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
