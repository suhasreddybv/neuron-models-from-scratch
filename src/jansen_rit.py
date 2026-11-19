"""Jansen-Rit neural mass model of a cortical column (alpha rhythm of the EEG).

Equations (Jansen & Rit 1995, eqs. 6; notation of Grimbert & Faugeras 2006).
Three populations - pyramidal cells (PSP y0), excitatory interneurons (y1)
and inhibitory interneurons (y2) - each a second-order linear synapse driven
by a sigmoid of a membrane potential:

    y0'' = A a S(y1 - y2)              - 2a y0' - a^2 y0
    y1'' = A a [p + C2 S(C1 y0)]       - 2a y1' - a^2 y1
    y2'' = B b C4 S(C3 y0)             - 2b y2' - b^2 y2

    S(v) = 2 e0 / (1 + exp(r (v0 - v)))

written as six first-order ODEs (y3 = y0', y4 = y1', y5 = y2'). The EEG-like
output is the pyramidal membrane potential y = y1 - y2 (mV); p is the
extrinsic input rate (pulses/s). Parameters are the paper's standard set:
A = 3.25 mV, B = 22 mV, a = 100 /s, b = 50 /s, C = 135, C1 = C, C2 = 0.8 C,
C3 = C4 = 0.25 C, e0 = 2.5 /s, v0 = 6 mV, r = 0.56 /mV. "Reduced" here means
the single-column (lumped) form: no spatial extent, no inter-column coupling.

Stimulus. Jansen & Rit drove the column with p uniform on [120, 320]
pulses/s. The sample rate of that noise is not stated in a way that fixes
its spectral density, so here p is redrawn every 1 ms and held (zero-order
hold); the result is also reported for a 0.1 ms hold.

Integrator: classical fourth-order Runge-Kutta, dt = 0.1 ms, with p held
constant within each step (so a piecewise-constant input is integrated with
RK4 accuracy). Time is in seconds.

Validation (see ROADMAP.md note):
(1) Fixed points in closed form. With S invertible, every fixed point is a
    point of the curve y0 -> (y0, y1, y2, p) with v = y1 - y2 = S^-1(a y0/A),
    y2 = (B/b) C4 S(C3 y0), p = a y1/A - C2 S(C1 y0): no root finding in p.
    Folds are extrema of p(y0); the 6x6 Jacobian is written out
    analytically and Hopf points are where its leading complex pair crosses
    the imaginary axis.
(2) Linear regime. At 16 inputs within 25 pulses/s of each Hopf point the
    fixed point is perturbed by 1e-5 mV along the critical eigenvector; the
    growth rate sigma (log of successive peaks) and frequency (peak
    intervals) are compared with Re and Im of the Jacobian eigenvalue, and
    the zero of the simulated sigma(p) gives the Hopf inputs from simulation.
(3) Noise-driven spectrum in the linear regime. With small Gaussian input
    noise (SD 5 pulses/s, 1 ms hold) about the stable foci at p = 60 and
    p = 400, the Welch PSD of y must match the exact spectrum of the
    sampled linearised system, S(f) = 2 Dt sd^2 |c (e^{i 2 pi f Dt} - Phi)^-1 G|^2
    with Phi = expm(J Dt), G = J^-1 (Phi - I) b (Dt = 1 ms), and the output
    variance must match the discrete Lyapunov solution.
(4) Alpha rhythm. Under the paper's stimulus the PSD peak must lie in the
    alpha band (8-13 Hz; Jansen & Rit 1995 report alpha-like
    activity for this stimulus) and within 0.5 Hz of the
    frequency of the deterministic limit cycle at the mean input p = 220,
    whose RK4 period is checked against solve_ivp (DOP853, rtol = 1e-11).
(5) Bifurcation vs input rate: constant-p sweeps from a low and a high
    initial state give the attractors (min/max of y), overlaid on the
    closed-form fixed-point branch. The Hopf inputs found here (89.83 and
    315.70 pulses/s) are the values reported by Grimbert & Faugeras (2006).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq                   # closed forms only
from scipy.integrate import solve_ivp               # reference solver only
from scipy.linalg import expm, solve_discrete_lyapunov
from scipy.signal import welch                      # analysis only

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Jansen & Rit 1995)
# ---------------------------------------------------------------------------
A, B = 3.25, 22.0          # mV, maximal EPSP / IPSP amplitude
a, b = 100.0, 50.0         # 1/s, inverse synaptic time constants
C = 135.0
C1, C2, C3, C4 = C, 0.8 * C, 0.25 * C, 0.25 * C
e0, v0, r = 2.5, 6.0, 0.56  # 1/s, mV, 1/mV

dt = 1e-4                  # s, RK4 step
HOLD = 1e-3                # s, sample-and-hold interval of the input noise
P_LO, P_HI = 120.0, 320.0  # Jansen & Rit stimulus range, pulses/s
P_MEAN = 0.5 * (P_LO + P_HI)
SEED = 1995
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions, fixed points, Jacobian
# ---------------------------------------------------------------------------
def S(v):
    return 2.0 * e0 / (1.0 + np.exp(r * (v0 - v)))


def dS(v):
    s = S(v)
    return r * s * (1.0 - s / (2.0 * e0))


def S_inv(s):
    return v0 - np.log(2.0 * e0 / s - 1.0) / r


def rhs(y, p):
    """y has shape (6, ...); p broadcasts against y[0]."""
    y0, y1, y2, y3, y4, y5 = y
    return np.array([
        y3, y4, y5,
        A * a * S(y1 - y2) - 2 * a * y3 - a * a * y0,
        A * a * (p + C2 * S(C1 * y0)) - 2 * a * y4 - a * a * y1,
        B * b * C4 * S(C3 * y0) - 2 * b * y5 - b * b * y2,
    ])


Y0_MAX = A / a * 2 * e0          # supremum of y0 at a fixed point


def branch_point(y0):
    """Fixed point (y0, y1, y2) and the input p at which it exists."""
    v = S_inv(a * y0 / A)
    y2 = B / b * C4 * S(C3 * y0)
    y1 = v + y2
    p = a * y1 / A - C2 * S(C1 * y0)
    return np.array([y0, y1, y2, 0.0, 0.0, 0.0]), p


def jacobian(y):
    y0, y1, y2 = y[:3]
    J = np.zeros((6, 6))
    J[0, 3] = J[1, 4] = J[2, 5] = 1.0
    g = dS(y1 - y2)
    J[3, 0], J[3, 1], J[3, 2], J[3, 3] = -a * a, A * a * g, -A * a * g, -2 * a
    J[4, 0], J[4, 1], J[4, 4] = A * a * C2 * C1 * dS(C1 * y0), -a * a, -2 * a
    J[5, 0], J[5, 2], J[5, 5] = B * b * C4 * C3 * dS(C3 * y0), -b * b, -2 * b
    return J


B_IN = np.array([0, 0, 0, 0, A * a, 0.0])   # input direction of p
C_OUT = np.array([0, 1.0, -1.0, 0, 0, 0])   # output y = y1 - y2


def lead_pair(y):
    """Eigenvalue with largest real part among those with Im > 0."""
    ev = np.linalg.eigvals(jacobian(y))
    ev = ev[ev.imag > 1e-9]
    return ev[np.argmax(ev.real)]


def branch(n=6000):
    ys = np.linspace(1e-5, Y0_MAX * (1 - 1e-6), n)
    pts = [branch_point(y) for y in ys]
    P = np.array([q[1] for q in pts])
    stable = np.array([np.max(np.linalg.eigvals(jacobian(q[0])).real) < 0 for q in pts])
    V = np.array([q[0][1] - q[0][2] for q in pts])
    return ys, P, V, stable


def folds():
    ys, P, _, _ = branch()
    d = np.diff(P)
    out = []
    for k in np.nonzero(np.sign(d[1:]) != np.sign(d[:-1]))[0]:
        dp = lambda u: (branch_point(u + 1e-9)[1] - branch_point(u - 1e-9)[1]) / 2e-9
        u = brentq(dp, ys[k], ys[k + 2], xtol=1e-14)
        out.append((u, branch_point(u)[1]))
    return out


def hopf_points():
    """Zeros of Re(leading complex pair) along the branch."""
    ys = np.linspace(0.067, 0.15, 831)
    re = np.array([lead_pair(branch_point(u)[0]).real for u in ys])
    out = []
    for k in np.nonzero(np.sign(re[1:]) != np.sign(re[:-1]))[0]:
        u = brentq(lambda s: lead_pair(branch_point(s)[0]).real, ys[k], ys[k + 1], xtol=1e-15)
        y, p = branch_point(u)
        out.append((p, lead_pair(y).imag / (2 * np.pi), y))
    return out


def fixed_point_upper(p):
    """Fixed point on the branch y0 in (0.07, Y0_MAX) at input p (p(y0) is increasing there)."""
    u = brentq(lambda s: branch_point(s)[1] - p, 0.075, Y0_MAX * (1 - 1e-9), xtol=1e-15)
    return branch_point(u)[0]


def fixed_point_lower(p):
    """Fixed point on the low-activity branch y0 < 0.0208 (below the first fold, p < 113.6)."""
    u = brentq(lambda s: branch_point(s)[1] - p, 1e-6, 0.0208, xtol=1e-15)
    return branch_point(u)[0]


NOISE_POINTS = ((60.0, "lower", fixed_point_lower), (500.0, "upper", fixed_point_upper))


# ---------------------------------------------------------------------------
# Block 3: Simulation
# ---------------------------------------------------------------------------
def simulate(y_init, T, p_fn, record_every=10, dt=dt):
    """RK4. p_fn(k) returns the input for step k (held over the step).
    y_init has shape (6, n). Returns t, recorded y=y1-y2 (n_rec, n), final state."""
    y = np.array(y_init, dtype=float)
    n_steps = int(round(T / dt))
    rec = np.empty((n_steps // record_every + 1,) + y.shape[1:])
    rec[0] = y[1] - y[2]
    for k in range(n_steps):
        p = p_fn(k)
        k1 = rhs(y, p)
        k2 = rhs(y + 0.5 * dt * k1, p)
        k3 = rhs(y + 0.5 * dt * k2, p)
        k4 = rhs(y + dt * k3, p)
        y = y + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        if (k + 1) % record_every == 0:
            rec[(k + 1) // record_every] = y[1] - y[2]
    t = np.arange(rec.shape[0]) * dt * record_every
    return t, rec, y


def held_noise(n_steps, n, lo=None, hi=None, mean=None, sd=None, hold=HOLD):
    """Input sequence, one value per RK4 step, redrawn every `hold` seconds."""
    per = int(round(hold / dt))
    n_draw = n_steps // per + 1
    if sd is None:
        draws = rng.uniform(lo, hi, size=(n_draw, n))
    else:
        draws = mean + sd * rng.standard_normal((n_draw, n))
    return np.repeat(draws, per, axis=0)[:n_steps]


def peaks(t, x):
    """Local maxima with parabolic interpolation: times and values."""
    k = np.nonzero((x[1:-1] > x[:-2]) & (x[1:-1] >= x[2:]))[0] + 1
    y0_, y1_, y2_ = x[k - 1], x[k], x[k + 1]
    den = y0_ - 2 * y1_ + y2_
    off = 0.5 * (y0_ - y2_) / den
    h = t[1] - t[0]
    return t[k] + off * h, y1_ - 0.25 * (y0_ - y2_) * off


def linear_spectrum(y, sd, hold, f):
    """Exact one-sided PSD and variance of y sampled every `hold` for ZOH white input,
    linearised about the fixed point y."""
    J = jacobian(y)
    Phi = expm(J * hold)
    G = np.linalg.solve(J, (Phi - np.eye(6)) @ B_IN)
    z = np.exp(2j * np.pi * f * hold)
    H = np.array([C_OUT @ np.linalg.solve(zz * np.eye(6) - Phi, G) for zz in z])
    psd = 2 * hold * sd ** 2 * np.abs(H) ** 2
    Sig = solve_discrete_lyapunov(Phi, sd ** 2 * np.outer(G, G))
    return psd, C_OUT @ Sig @ C_OUT


# ---------------------------------------------------------------------------
# Block 4: Validation
# ---------------------------------------------------------------------------
def validate():
    R = {}
    # (1) closed-form branch, folds, Hopf points
    ys, P, V, stable = branch()
    R["branch"] = (P, V, stable)
    fl = folds()
    hp = hopf_points()
    R["folds"], R["hopf"] = fl, hp
    print("Folds of the fixed-point branch: " + ", ".join(f"p = {q:.4f} (y0 = {u:.5f})" for u, q in fl))
    for p_h, f_h, y in hp:
        print(f"Hopf (Jacobian): p_H = {p_h:.4f}, f_H = {f_h:.4f} Hz, v* = {y[1] - y[2]:.4f} mV")
    hp = [h for h in hp if h[0] > 0]          # the third crossing lies at p < 0 (unphysical)
    R["hopf"] = hp
    assert len(hp) == 2

    # fixed point is a fixed point of the simulated model
    y_fp = np.stack([fixed_point_lower(60.0), fixed_point_upper(60.0), fixed_point_upper(500.0)], axis=1)
    _, _, yT = simulate(y_fp, 0.5, lambda k: np.array([60.0, 60.0, 500.0]))
    err_fp = np.max(np.abs(yT - y_fp))
    print(f"Closed-form fixed points (p = 60 lower and upper, p = 500) held for 0.5 s: max |drift| = {err_fp:.2e}")
    assert err_fp < 1e-9

    # (2) linear regime near both Hopf points
    offs = np.array([-25, -12, -6, -3, 3, 6, 12, 25.0])
    p_lin = np.concatenate([hp[0][0] + offs, hp[1][0] + offs])
    y_init, sig_th, f_th = [], [], []
    for q in p_lin:
        y = fixed_point_upper(q)
        ev, vec = np.linalg.eig(jacobian(y))
        j = np.argmax(np.where(ev.imag > 1e-9, ev.real, -np.inf))
        d = vec[:, j].real
        d *= 1e-5 / abs(C_OUT @ d)
        y_init.append(y + d)
        sig_th.append(ev[j].real)
        f_th.append(ev[j].imag / (2 * np.pi))
    sig_th, f_th = np.array(sig_th), np.array(f_th)
    y_star = np.array([C_OUT @ fixed_point_upper(q) for q in p_lin])
    t, rec, _ = simulate(np.array(y_init).T, 6.0, lambda k: p_lin, record_every=1)
    sig_sim, f_sim = [], []
    for j in range(p_lin.size):
        tp, vp = peaks(t, rec[:, j] - y_star[j])
        m = tp > 1.0
        sig_sim.append(np.polyfit(tp[m], np.log(vp[m]), 1)[0])
        f_sim.append(1.0 / np.mean(np.diff(tp[m])))
    sig_sim, f_sim = np.array(sig_sim), np.array(f_sim)
    err_sig = np.max(np.abs(sig_sim - sig_th))
    err_f = np.max(np.abs(f_sim - f_th) / f_th)
    print("Linear regime (perturbation 1e-5 mV along the critical eigenvector, 6 s, fit t > 1 s):")
    for j in range(p_lin.size):
        print(f"  p = {p_lin[j]:8.3f}: sigma {sig_sim[j]:+.5f} (sim) vs {sig_th[j]:+.5f} /s; "
              f"f {f_sim[j]:.5f} vs {f_th[j]:.5f} Hz")
    print(f"  max |sigma error| = {err_sig:.2e} /s; max relative f error = {err_f:.2e}")
    assert err_sig < 5e-3, f"growth-rate error {err_sig:.2e}"
    assert err_f < 1e-3, f"frequency error {err_f:.2e}"
    p_H_sim = []
    for h in range(2):
        sl = slice(8 * h + 2, 8 * h + 6)        # p_H +/- 3 and +/- 6
        c = np.polyfit(p_lin[sl] - hp[h][0], sig_sim[sl], 2)
        rts = np.roots(c).real
        p_H_sim.append(hp[h][0] + rts[np.argmin(np.abs(rts))])
    rel_PH = max(abs(p_H_sim[h] - hp[h][0]) / hp[h][0] for h in range(2))
    print(f"  Hopf from simulated sigma(p) (quadratic fit over +/-3, +/-6): "
          f"{p_H_sim[0]:.4f}, {p_H_sim[1]:.4f} vs {hp[0][0]:.4f}, {hp[1][0]:.4f}; max rel. error {rel_PH:.2e}")
    assert rel_PH < 1e-4
    R["lin"] = (p_lin, sig_sim, sig_th, f_sim, f_th)

    # (3) small-noise spectra vs exact linearised spectrum
    T_n, n_tr, sd = 21.0, 32, 5.0
    p_lin_noise = np.array([q for q, _, _ in NOISE_POINTS])
    fps = [fn(q) for q, _, fn in NOISE_POINTS]
    n_steps = int(round(T_n / dt))
    y_start = np.repeat(np.stack(fps, 1), n_tr, axis=1)
    pn = held_noise(n_steps, 2 * n_tr, mean=np.repeat(p_lin_noise, n_tr), sd=sd)
    t, rec, _ = simulate(y_start, T_n, lambda k: pn[k], record_every=10)
    fs = 1.0 / (10 * dt)
    m = t >= 1.0
    R["spec"] = []
    for i, q in enumerate(p_lin_noise):
        x = rec[m][:, i * n_tr:(i + 1) * n_tr]
        f, Pxx = welch(x, fs=fs, nperseg=4096, axis=0, detrend="constant")
        Pxx = Pxx.mean(axis=1)
        psd_th, var_th = linear_spectrum(fps[i], sd, 10 * dt, f)
        lam = np.linalg.eigvals(jacobian(fps[i]))
        lam = lam[np.argmax(lam.real)]
        var_sim = np.mean(np.var(x, axis=0))
        n_v = x.shape[0] * n_tr
        band = (f >= 1) & (f <= 40)
        med = np.median(np.abs(Pxx[band] / psd_th[band] - 1))
        fpk_sim, fpk_th = f[np.argmax(Pxx)], f[np.argmax(psd_th)]
        print(f"Small-noise spectrum at p = {q:.0f}, {NOISE_POINTS[i][1]} branch (leading eigenvalue "
              f"{lam.real:.2f}{lam.imag:+.2f}i /s; input SD {sd}, 1 ms hold, {n_tr} x 20 s): "
              f"variance {var_sim:.4e} vs {var_th:.4e} mV^2 ({(var_sim / var_th - 1) * 100:+.2f}%); "
              f"PSD peak {fpk_sim:.3f} vs {fpk_th:.3f} Hz (bin {f[1]:.3f}); "
              f"median |PSD ratio - 1| over 1-40 Hz = {med:.3f}")
        assert abs(var_sim / var_th - 1) < 0.05
        if psd_th.max() > 2 * psd_th[np.argmin(np.abs(f - 1.0))]:   # peak check only for a resonance
            assert abs(fpk_sim - fpk_th) <= 2 * f[1] + 1e-12
        else:
            print("    (no resonance: spectrum flat to within 2x below its maximum, peak location not tested)")
        assert med < 0.05
        R["spec"].append((q, NOISE_POINTS[i][1], f, Pxx, psd_th))
        R.setdefault("spec_num", []).append((q, var_sim, var_th, fpk_sim, fpk_th, med))

    # (4) alpha rhythm: deterministic cycle at p = 220, RK4 vs DOP853
    y_c = fixed_point_upper(P_MEAN) + np.array([0, 0.1, 0, 0, 0, 0])
    t, rec, _ = simulate(y_c[:, None], 8.0, lambda k: P_MEAN, record_every=1)
    tp, _ = peaks(t, rec[:, 0])
    T_rk = np.mean(np.diff(tp[tp > 4.0]))
    sol = solve_ivp(lambda tt, yy: rhs(yy, P_MEAN), (0, 8.0), y_c, method="DOP853",
                    rtol=1e-11, atol=1e-11, max_step=1e-3, dense_output=True)
    tt = np.arange(0, 8.0, dt)
    yy = sol.sol(tt)
    tp2, _ = peaks(tt, yy[1] - yy[2])
    T_ref = np.mean(np.diff(tp2[tp2 > 4.0]))
    f_lc = 1.0 / T_rk
    rel_T = abs(T_rk - T_ref) / T_ref
    print(f"Limit cycle at constant p = {P_MEAN:.0f}: f = {f_lc:.5f} Hz (RK4) vs {1 / T_ref:.5f} Hz (DOP853), "
          f"rel. period error {rel_T:.2e}; peak-to-peak y = {np.ptp(rec[t > 4.0, 0]):.3f} mV")
    assert rel_T < 1e-5

    T_a, n_tr = 21.0, 16
    n_steps = int(round(T_a / dt))
    y_start = np.repeat(y_c[:, None], 2 * n_tr, axis=1)
    pa = np.concatenate([held_noise(n_steps, n_tr, lo=P_LO, hi=P_HI, hold=HOLD),
                         held_noise(n_steps, n_tr, lo=P_LO, hi=P_HI, hold=dt)], axis=1)
    t, rec, _ = simulate(y_start, T_a, lambda k: pa[k], record_every=10)
    m = t >= 1.0
    R["alpha"] = {"t": t, "trace": rec[:, 0], "f_lc": f_lc, "p_trace": pa[::10, 0]}
    for name, sl in (("1 ms hold", slice(0, n_tr)), ("0.1 ms hold", slice(n_tr, 2 * n_tr))):
        f, Pxx = welch(rec[m][:, sl], fs=fs, nperseg=4096, axis=0)
        Pxx = Pxx.mean(axis=1)
        k = np.argmax(Pxx[f > 1]) + np.sum(f <= 1)
        fpk = f[k]
        frac = Pxx[(f >= 8) & (f <= 13)].sum() / Pxx[(f >= 1) & (f <= 40)].sum()
        print(f"J&R stimulus (U[{P_LO:.0f}, {P_HI:.0f}], {name}, {n_tr} x 20 s): PSD peak {fpk:.3f} Hz "
              f"(limit cycle at p = 220: {f_lc:.3f} Hz, |diff| {abs(fpk - f_lc):.3f}); "
              f"fraction of 1-40 Hz power in 8-13 Hz = {frac:.3f}; SD of y = {rec[m][:, sl].std():.3f} mV")
        assert 8.0 <= fpk <= 13.0
        assert abs(fpk - f_lc) < 0.5
        R["alpha"][name] = (f, Pxx, fpk, frac)

    # (5) bifurcation sweep in p
    p_sw = np.linspace(-60.0, 420.0, 161)
    n = p_sw.size
    y_low = np.zeros((6, n))
    y_high = np.repeat(y_c[:, None], n, axis=1)
    t, rec, _ = simulate(np.concatenate([y_low, y_high], 1), 10.0,
                         lambda k: np.concatenate([p_sw, p_sw]), record_every=10)
    tail = rec[t >= 9.0]
    R["sweep"] = (p_sw, tail.min(0), tail.max(0))
    # oscillating at the end of the run: peak-to-peak over the last 1 s > 0.05 mV
    # (near the Hopf points |sigma| < 0.2 /s, so transients there have not died out by 9 s)
    osc = np.ptp(tail, axis=0) > 0.05
    R["sweep_osc"] = osc

    def rng_str(mask):
        return "none" if not mask.any() else f"[{p_sw[mask].min():.1f}, {p_sw[mask].max():.1f}]"
    print(f"Sweep, constant p on a {p_sw[1] - p_sw[0]:.1f} grid, 10 s: peak-to-peak > 0.05 mV after 9 s from the high "
          f"start for p in {rng_str(osc[n:])}, from rest for p in {rng_str(osc[:n])} "
          f"(Hopf points {hp[0][0]:.2f}, {hp[1][0]:.2f}; fold {fl[0][1]:.2f})")
    return R


# ---------------------------------------------------------------------------
# Block 5: Plot
# ---------------------------------------------------------------------------
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    al = R["alpha"]
    t = al["t"]
    m = (t >= 2.0) & (t <= 4.0)
    a0 = ax[0, 0]
    a0.plot(t[m] - 2.0, al["trace"][m], color="C0", lw=0.8)
    a0.set_xlabel("time (s)")
    a0.set_ylabel("y = y1 - y2 (mV)")
    a0.set_title("Jansen-Rit EEG, p ~ U[120, 320] /s (1 ms hold)")

    a1 = ax[0, 1]
    for name, c in (("1 ms hold", "C0"), ("0.1 ms hold", "C1")):
        f, Pxx, fpk, frac = al[name]
        a1.semilogy(f, Pxx, color=c, label=f"{name}: peak {fpk:.2f} Hz")
    a1.axvspan(8, 13, color="0.9", zorder=0, label="alpha band 8-13 Hz")
    a1.axvline(al["f_lc"], color="k", ls="--", lw=1, label=f"limit cycle at p = 220: {al['f_lc']:.2f} Hz")
    a1.set_xlim(0, 40)
    a1.set_xlabel("frequency (Hz)")
    a1.set_ylabel("PSD (mV$^2$/Hz)")
    a1.set_title("Alpha peak under the paper's stimulus")
    a1.legend(fontsize=8)

    a2 = ax[0, 2]
    P, V, st = R["branch"]
    for flag, ls, lab in ((True, "-", "stable fixed point"), (False, ":", "unstable fixed point")):
        Vm = np.where(st == flag, V, np.nan)
        a2.plot(P, Vm, "k", ls=ls, lw=1.2, label=lab)
    p_sw, ymin, ymax = R["sweep"]
    n = p_sw.size
    a2.plot(p_sw, ymax[n:], "o", ms=2.5, color="C3", label="sim max/min after 9 s (high start)")
    a2.plot(p_sw, ymin[n:], "o", ms=2.5, color="C3")
    a2.plot(p_sw, ymax[:n], "x", ms=3, color="C0", label="sim max/min after 9 s (start at 0)")
    a2.plot(p_sw, ymin[:n], "x", ms=3, color="C0")
    for p_h, f_h, y in R["hopf"]:
        a2.plot(p_h, y[1] - y[2], "s", color="C2", ms=6)
        a2.annotate(f"Hopf {p_h:.2f}\n{f_h:.2f} Hz", (p_h, y[1] - y[2]), textcoords="offset points",
                    xytext=(4, -24), fontsize=8, color="C2")
    for u, q in R["folds"]:
        yq = branch_point(u)[0]
        a2.plot(q, yq[1] - yq[2], "^", color="C4", ms=6)
    a2.set_xlim(-60, 420)
    a2.set_ylim(-3, max(14.0, np.nanmax(ymax) + 1.0))
    a2.set_xlabel("input rate p (pulses/s)")
    a2.set_ylabel("y (mV)")
    a2.set_title("Bifurcation vs input rate (closed-form branch, triangles = folds)")
    a2.legend(fontsize=7, loc="upper left")

    p_lin, ss, st_, fsim, fth = R["lin"]
    a3 = ax[1, 0]
    pp = np.linspace(60, 345, 300)
    sig_c, f_c = [], []
    for q in pp:
        lam = lead_pair(fixed_point_upper(q))
        sig_c.append(lam.real)
        f_c.append(lam.imag / (2 * np.pi))
    a3.plot(pp, sig_c, "k", lw=1, label="Re lambda (Jacobian)")
    a3.plot(p_lin, ss, "o", color="C3", ms=5, label="sigma from simulation")
    a3.axhline(0, color="0.6", lw=0.8)
    a3.set_xlabel("input rate p (pulses/s)")
    a3.set_ylabel("growth rate (1/s)")
    a3b = a3.twinx()
    a3b.plot(pp, f_c, color="C0", lw=1)
    a3b.plot(p_lin, fsim, "s", color="C0", ms=3)
    a3b.set_ylabel("frequency (Hz), blue", color="C0")
    a3.set_title("Linear regime near both Hopf points")
    a3.legend(fontsize=8, loc="lower center")

    for i, (q, br, f, Pxx, psd) in enumerate(R["spec"]):
        a = ax[1, 1 + i]
        a.semilogy(f, Pxx, color="C1", lw=1.2, label="simulated (Welch, 32 x 20 s)")
        a.semilogy(f, psd, "k--", lw=1, label="exact linearised spectrum")
        a.set_xlim(0, 60)
        a.set_ylim(psd[f <= 60].min() / 5, psd.max() * 5)
        a.set_xlabel("frequency (Hz)")
        a.set_ylabel("PSD (mV$^2$/Hz)")
        a.set_title(f"Small input noise (SD 5 /s) at p = {q:.0f}, {br} branch")
        a.legend(fontsize=8)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "jansen_rit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    plot(validate())
    print(f"Done in {_time.time() - t0:.1f} s")
