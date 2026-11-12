"""Ring attractor for head direction: threshold-linear rate ring with a
velocity-modulated asymmetric kernel.

Model (the ring model of Ben-Yishai, Bar-Or & Sompolinsky 1995, on the full
circle; velocity input by modulating an odd weight component as in Zhang 1996):

    tau dm_i/dt = -m_i + [ h_i ]_+ ,
    h_i = I0 + (1/N) sum_j J(theta_i - theta_j) m_j ,
    J(D) = J0 + J1 cos(D) + gamma(t) J1 sin(D),

m_i the rate (Hz) of the neuron with preferred direction theta_i = -pi +
2 pi i / N, [x]_+ = max(x, 0). The symmetric part J0 + J1 cos D (J0 < 0:
uniform inhibition, J1: cosine excitation) supports a bump at any position;
the odd part gamma J1 sin D, gamma proportional to angular head velocity,
moves it. Parameters: N = 512, tau = 10 ms, J0 = -8, J1 = 10, I0 = 32 Hz.
The bump position is read out as the population-vector angle
psi = arg sum_j m_j exp(i theta_j).

Integrator: classical fourth-order Runge-Kutta, dt = 0.5 ms (0.05 tau), the
recurrent sum evaluated as a circular convolution by FFT; gamma is held
constant within a step.

Closed forms used for validation (continuum limit N -> infinity, <.> = mean
over the circle):
(1) Stationary bump (Ben-Yishai et al. 1995). m(theta) = B [cos theta -
    cos theta_c]_+ is self-consistent iff the first harmonic reproduces
    itself, J1 g1(theta_c) = 1 with g1 = (theta_c - sin theta_c cos theta_c)
    / (2 pi), which fixes the half-width theta_c from J1 alone (requires
    J1 > 2), and the mean gives B = I0 / (-cos theta_c - J0 g0) with
    g0 = (sin theta_c - theta_c cos theta_c) / pi. Peak rate B (1 - cos theta_c).
(2) Linear stability. A perturbation dm = H(theta) f(theta) (H the
    indicator of the active arc) obeys tau d(dm)/dt = -dm + H dh. Three modes
    couple to the recurrent input: sin theta (translation), eigenvalue
    -1 + J1 <H sin^2> = 0 exactly; and the pair {1, cos theta} with
    tau lambda = -1 + eig [[J0 a0, J0 a1], [J1 a1, J1 a2]], a0 = theta_c / pi,
    a1 = sin theta_c / pi, a2 = (theta_c + sin theta_c cos theta_c) / (2 pi).
    Every other mode decays at -1/tau. A translation perturbation eps H sin
    theta shifts the bump by eps / B and does not decay.
(3) Velocity integration, exact for every gamma. Write the input as
    h = c + J1 |z| sqrt(1 + gamma^2) cos(theta - psi - atan gamma), z the
    first harmonic of m: the odd kernel rotates the input profile ahead of
    the bump by atan gamma. A leaky integrator driven by a profile rotating
    at angular speed omega delays its first harmonic by atan(omega tau) and
    scales it by 1/sqrt(1 + omega^2 tau^2). Self-consistency of the first
    harmonic therefore requires omega tau = gamma, and the same width and
    amplitude conditions as in (1): the rectified input [h]_+ travels with
    the static profile F(theta) = B [cos theta - cos theta_c]_+ and the rate
    profile is its low-pass, M - gamma M' = F, i.e. in the comoving frame
        M(phi) = sum_k c_k (cos k phi - k gamma sin k phi) / (1 + k^2 gamma^2),
    c_k the Fourier cosine coefficients of F (written out in
    static_fourier()). So the bump angle is psi(t) = psi(0) + (1/tau) int
    gamma dt: with gamma = tau omega_head the network integrates head
    velocity with unit gain.

Validation (ROADMAP note: bump stability; drift speed vs asymmetric input
matches integration of velocity): the simulated bump against (1) with a
convergence study in N; decay rates of small perturbations and the Jacobian
spectrum of the discrete network against (2); spontaneous drift of the bump
over 10 s from eight start positions; bump speed and travelling profile
against (3) for gamma = +-0.002 ... 2; and a 30 s random head-velocity
trajectory (Ornstein-Uhlenbeck angular velocity, sd 2 rad/s, correlation
time 0.3 s) against the integrated true heading.

RNG: NumPy PCG64 seeded with SEED (printed): the random initial rates and
start positions, and the head-velocity trajectory.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq          # closed forms only

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters
# ---------------------------------------------------------------------------
N = 512            # neurons on the ring
tau = 10.0         # ms
J0 = -8.0          # uniform (inhibitory) coupling
J1 = 10.0          # cosine (excitatory) coupling
I0 = 32.0          # Hz, uniform external input
dt = 0.5           # ms, RK4 step
SEED = 1996

# tolerances, fixed before the run
TOL_BUMP = 2e-3        # static profile: max |m - m_theory| / peak at N = 512
TOL_EIG = 0.02         # relative error of decay rates and translation shift
TOL_DRIFT = 0.05       # rad, max spontaneous drift over 10 s at gamma = 0
TOL_SPEED = 1e-3       # relative error of tau * omega / gamma
TOL_TRAVEL = 2e-3      # travelling profile: max |M - M_theory| / static peak
TOL_PATH = 0.05        # rad, max |psi - integrated heading| over 30 s
T_SHIFT = 20.0         # ms, time at which the translation shift is read

PI_T = 30000.0         # ms, path-integration run
PI_SD = 2.0            # rad/s, sd of head angular velocity
PI_TC = 300.0          # ms, its correlation time


def grid(n=N):
    return -np.pi + 2.0 * np.pi * np.arange(n) / n


# ---------------------------------------------------------------------------
# Block 2: Model functions and closed forms
# ---------------------------------------------------------------------------
def kernels_hat(n=N):
    """rfft of the symmetric and odd kernel columns, J(2 pi d / n) / n."""
    d = 2.0 * np.pi * np.arange(n) / n
    Ks = (J0 + J1 * np.cos(d)) / n
    Ka = (J1 * np.sin(d)) / n
    return np.fft.rfft(Ks), np.fft.rfft(Ka)


def rhs(m, Khat):
    n = m.shape[-1]
    h = np.fft.irfft(Khat * np.fft.rfft(m, axis=-1), n, axis=-1) + I0
    return (-m + np.maximum(h, 0.0)) / tau


def decode(m):
    th = grid(m.shape[-1])
    return np.angle(np.sum(m * np.exp(1j * th), axis=-1))


def theta_c_of_J1(j1=J1):
    return brentq(lambda x: x - np.sin(x) * np.cos(x) - 2.0 * np.pi / j1, 1e-9, np.pi)


def bump_constants():
    tc = theta_c_of_J1()
    g0 = (np.sin(tc) - tc * np.cos(tc)) / np.pi
    B = I0 / (-np.cos(tc) - J0 * g0)
    return dict(theta_c=tc, g0=g0, B=B, peak=B * (1 - np.cos(tc)), mean=B * g0)


def static_profile(phi, C):
    return C["B"] * np.maximum(np.cos(phi) - np.cos(C["theta_c"]), 0.0)


def static_fourier(C, K=20000):
    """Cosine coefficients c_k of F(phi) = B [cos phi - cos theta_c]_+,
    F = c_0 + sum_{k>=1} c_k cos k phi."""
    tc, B = C["theta_c"], C["B"]
    k = np.arange(K + 1, dtype=float)
    c = np.empty(K + 1)
    c[0] = B * C["g0"]
    c[1] = B / np.pi * (tc + np.sin(tc) * np.cos(tc) - 2 * np.cos(tc) * np.sin(tc))
    kk = k[2:]
    I1 = np.sin((kk - 1) * tc) / (kk - 1) + np.sin((kk + 1) * tc) / (kk + 1)
    c[2:] = B / np.pi * (I1 - np.cos(tc) * 2 * np.sin(kk * tc) / kk)
    return c


def travelling_profile(phi, gamma, C, c=None):
    """Closed-form rate profile of the bump moving at omega = gamma / tau, in
    the frame centred on its population-vector angle."""
    if c is None:
        c = static_fourier(C)
    k = np.arange(c.size)[:, None]
    x = phi[None, :] - np.arctan(gamma)       # first harmonic of M peaks at -atan(gamma)
    return np.sum(c[:, None] * (np.cos(k * x) - k * gamma * np.sin(k * x))
                  / (1 + (k * gamma) ** 2), axis=0)


def stability_theory(C):
    tc = C["theta_c"]
    a0, a1 = tc / np.pi, np.sin(tc) / np.pi
    a2 = (tc + np.sin(tc) * np.cos(tc)) / (2 * np.pi)
    Mx = np.array([[J0 * a0, J0 * a1], [J1 * a1, J1 * a2]])
    ev, V = np.linalg.eig(Mx)
    o = np.argsort(-ev.real)
    lam = -1.0 + ev.real[o]                   # in units of 1/tau
    lam_sin = -1.0 + J1 * (tc - np.sin(tc) * np.cos(tc)) / (2 * np.pi)
    return lam_sin, lam, V[:, o].real


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(m0, T, gamma=0.0, record_every=0, keep_rates=0):
    """RK4 with constant gamma. m0: (..., n) initial rates; gamma: scalar or an
    array matching the leading axes (several networks run in lock-step).
    Returns the final rates, the decoded angle every `record_every` steps
    (unwrapped) and, if keep_rates, the rates every `keep_rates` steps."""
    m = np.array(m0, dtype=float)
    Ks, Ka = kernels_hat(m.shape[-1])
    g = np.asarray(gamma, dtype=float)
    Khat = Ks + g[..., None] * Ka if g.ndim else Ks + float(g) * Ka
    steps = int(round(T / dt))
    psis, rates = [], []
    for k in range(steps):
        if record_every and k % record_every == 0:
            psis.append(decode(m))
            if keep_rates and k % keep_rates == 0:
                rates.append(m.copy())
        a = rhs(m, Khat)
        b = rhs(m + 0.5 * dt * a, Khat)
        c = rhs(m + 0.5 * dt * b, Khat)
        d = rhs(m + dt * c, Khat)
        m = m + dt / 6.0 * (a + 2 * b + 2 * c + d)
    if record_every:
        psis.append(decode(m))
        psis = np.unwrap(np.array(psis), axis=0)
    return m, psis, (np.array(rates) if keep_rates else None)


def simulate_velocity(m0, gamma_t, keep_every=40):
    """RK4 with a time-varying gamma (one value per step). Returns the decoded
    angle after every step (unwrapped) and rate snapshots."""
    m = np.array(m0, dtype=float)
    Ks, Ka = kernels_hat(m.size)
    psi = np.empty(gamma_t.size + 1)
    psi[0] = decode(m)
    snaps = []
    for k, g in enumerate(gamma_t):
        if k % keep_every == 0:
            snaps.append(m.copy())
        Khat = Ks + g * Ka
        a = rhs(m, Khat)
        b = rhs(m + 0.5 * dt * a, Khat)
        c = rhs(m + 0.5 * dt * b, Khat)
        d = rhs(m + dt * c, Khat)
        m = m + dt / 6.0 * (a + 2 * b + 2 * c + d)
        psi[k + 1] = decode(m)
    return np.unwrap(psi), np.array(snaps)


def settle(psi0, n=N, T=1000.0):
    th = grid(n)
    psi0 = np.atleast_1d(psi0)
    m0 = 10.0 * np.maximum(np.cos(th[None, :] - psi0[:, None]), 0.0)
    return simulate(m0, T)[0]


def fit_bump(m):
    """Least-squares c + a cos(theta - psi) on the active neurons (exact for a
    stationary threshold-linear bump)."""
    th = grid(m.size)
    act = m > 1e-9 * m.max()
    X = np.c_[np.ones(act.sum()), np.cos(th[act]), np.sin(th[act])]
    c0, cc, ss = np.linalg.lstsq(X, m[act], rcond=None)[0]
    a = np.hypot(cc, ss)
    return dict(peak=c0 + a, cos_tc=-c0 / a, psi=np.arctan2(ss, cc), active=act)


def wrap(x):
    return (x + np.pi) % (2 * np.pi) - np.pi


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    rng = np.random.default_rng(SEED)
    C = bump_constants()
    R = dict(C=C)
    th = grid()
    print(f"Closed form: theta_c = {C['theta_c']:.6f} rad ({np.degrees(C['theta_c']):.3f} deg), "
          f"B = {C['B']:.6f} Hz, peak {C['peak']:.6f} Hz, mean {C['mean']:.6f} Hz")

    # --- 1. stationary bump from random initial rates ----------------------
    print("\n1. Stationary bump from random initial rates vs closed form")
    m_init = rng.uniform(0.0, 5.0, N)
    _, psi_form, snaps = simulate(m_init, 1000.0, record_every=1, keep_rates=20)
    m_star, _, _ = simulate(m_init, 2000.0)
    fb = fit_bump(m_star)
    psi_star = fb["psi"]
    prof_err = np.max(np.abs(m_star - static_profile(wrap(th - psi_star), C))) / C["peak"]
    print(f"   formed at psi = {psi_star:+.4f} rad; peak {fb['peak']:.5f} vs {C['peak']:.5f} Hz "
          f"({(fb['peak'] / C['peak'] - 1) * 100:+.4f}%), cos theta_c {fb['cos_tc']:.6f} vs "
          f"{np.cos(C['theta_c']):.6f}, mean {m_star.mean():.5f} vs {C['mean']:.5f} Hz")
    print(f"   max |m - m_theory| / peak = {prof_err:.2e} (tolerance {TOL_BUMP:.0e})")
    assert prof_err < TOL_BUMP, f"static bump off by {prof_err:.2e}"
    Ns = [64, 128, 256, 512, 1024, 2048]
    errs_N = []
    for n in Ns:
        mm = settle(0.3, n=n, T=2000.0)[0]
        f = fit_bump(mm)
        errs_N.append(np.max(np.abs(mm - static_profile(wrap(grid(n) - f["psi"]), C))) / C["peak"])
    slope_N = np.polyfit(np.log(Ns), np.log(errs_N), 1)[0]
    print("   convergence in N: " + ", ".join(f"{n}: {e:.2e}" for n, e in zip(Ns, errs_N))
          + f"; log-log slope {slope_N:.2f}")
    R.update(m_init=m_init, snaps=snaps, m_star=m_star, psi_star=psi_star,
             prof_err=prof_err, Ns=Ns, errs_N=errs_N, slope_N=slope_N)

    # --- 2. linear stability -----------------------------------------------
    print("\n2. Stability: decay rates vs linearised theory (units of 1/tau)")
    lam_sin, lam, V = stability_theory(C)
    act = fb["active"]
    W = (J0 + J1 * np.cos(th[:, None] - th[None, :])) / N
    Jac = -np.eye(N) + act[:, None] * W
    ev = np.sort(np.linalg.eigvals(Jac).real)[::-1]
    ev_d = ev[[0, 1, -1]]
    print(f"   theory: translation {lam_sin:+.2e}, amplitude modes {lam[0]:+.5f}, {lam[1]:+.5f}; others -1")
    print(f"   discrete Jacobian (N = {N}): {ev_d[0]:+.2e}, {ev_d[1]:+.5f}, {ev_d[2]:+.5f}; "
          f"{np.sum(np.abs(ev + 1) < 1e-9)} eigenvalues at -1")
    phi = wrap(th - psi_star)
    H = act.astype(float)
    eps = 1e-4 * C["peak"]
    lam_sim, curves = [], []
    # perturbed and unperturbed networks run in lock-step; the difference
    # removes the slow pinning drift of the discrete bump (section 3)
    for j in range(2):
        f = H * (V[0, j] + V[1, j] * np.cos(phi))
        f *= eps / np.max(np.abs(f))
        _, _, rr = simulate(np.stack([m_star + f, m_star]), 40.0, record_every=1, keep_rates=1)
        tt = np.arange(rr.shape[0]) * dt
        nrm = np.linalg.norm(rr[:, 0] - rr[:, 1], axis=1)
        sel = tt >= 5.0
        lam_sim.append(np.polyfit(tt[sel], np.log(nrm[sel]), 1)[0] * tau)
        curves.append((tt, nrm / nrm[0]))
    f = eps * H * np.sin(phi)
    _, psi_tr, _ = simulate(np.stack([m_star + f, m_star]), 300.0, record_every=1)
    tt = np.arange(psi_tr.shape[0]) * dt
    dpsi = psi_tr[:, 0] - psi_tr[:, 1]
    shift_th = eps / C["B"]
    shift = dpsi[int(round(T_SHIFT / dt))]
    curves.append((tt, dpsi / shift_th))
    print(f"   translation shift at 300 ms / at {T_SHIFT:.0f} ms = {dpsi[-1] / shift:.4f} "
          f"(discrete pinning eigenvalue predicts {np.exp(ev_d[0] * (300 - T_SHIFT) / tau):.4f})")
    e_lam = [abs(a / b - 1) for a, b in zip(lam_sim, lam)]
    e_eig_d = [abs(ev_d[1] / lam[0] - 1), abs(ev_d[2] / lam[1] - 1)]
    e_shift = abs(shift / shift_th - 1)
    print(f"   simulated decay of the two amplitude modes: {lam_sim[0]:+.5f}, {lam_sim[1]:+.5f} "
          f"({e_lam[0] * 100:.2f}%, {e_lam[1] * 100:.2f}%)")
    print(f"   discrete Jacobian vs theory: {e_eig_d[0] * 100:.2f}%, {e_eig_d[1] * 100:.2f}%")
    print(f"   translation perturbation eps = {eps:.2e} Hz: shift at {T_SHIFT:.0f} ms {shift:.4e} vs eps/B "
          f"{shift_th:.4e} rad ({e_shift * 100:.2f}%); |translation eigenvalue| (discrete) "
          f"{abs(ev_d[0]):.1e}  (tolerance {TOL_EIG * 100:.0f}%)")
    assert max(e_lam + e_eig_d + [e_shift]) < TOL_EIG, "stability check failed"
    R.update(lam=lam, lam_sim=lam_sim, ev_d=ev_d, curves=curves, shift=shift,
             shift_th=shift_th, eps=eps)

    # --- 3. spontaneous drift ------------------------------------------------
    print("\n3. Spontaneous drift at gamma = 0, 8 start positions, 10 s")
    starts = rng.uniform(-np.pi, np.pi, 8)
    starts[:2] = [0.0, np.pi / N]            # on grid and half-way between
    m8 = settle(starts)
    _, psi8, _ = simulate(m8, 10000.0, record_every=200)
    drift = psi8 - psi8[0]
    t8 = np.arange(drift.shape[0]) * 200 * dt
    max_drift = np.max(np.abs(drift))
    print("   drift over 10 s (rad): " + ", ".join(f"{x:+.1e}" for x in drift[-1]))
    print(f"   max |drift| = {max_drift:.2e} rad (tolerance {TOL_DRIFT})")
    assert max_drift < TOL_DRIFT, f"bump drifts {max_drift:.3f} rad"
    R.update(t8=t8, drift=drift, max_drift=max_drift)

    # --- 4. bump speed and profile vs gamma ---------------------------------
    print("\n4. Bump speed vs asymmetric gain gamma: closed form omega = gamma / tau")
    gpos = np.array([0.002, 0.01, 0.05, 0.2, 0.5, 1.0, 2.0])
    gam = np.r_[-gpos[::-1], gpos]
    mg = np.repeat(m_star[None, :], gam.size, axis=0)
    mg, _, _ = simulate(mg, 1000.0, gamma=gam)
    T_meas = 1000.0
    mg_end, psig, _ = simulate(mg, T_meas, gamma=gam, record_every=20)
    omega = (psig[-1] - psig[0]) / T_meas               # rad/ms
    ratio = omega * tau / gam
    e_speed = np.max(np.abs(ratio - 1))
    c_k = static_fourier(C)
    e_prof = []
    profiles = {}
    for g, mm in zip(gam, mg_end):
        p = decode(mm)
        Mth = travelling_profile(wrap(th - p), g, C, c_k)
        e_prof.append(np.max(np.abs(mm - Mth)) / C["peak"])
        profiles[g] = (wrap(th - p), mm, Mth)
    e_prof = np.array(e_prof)
    for g, r, e in zip(gam, ratio, e_prof):
        print(f"   gamma = {g:+6.3f}: tau*omega/gamma = {r:.7f}, peak {profiles[g][1].max():7.3f} Hz "
              f"(theory {profiles[g][2].max():7.3f}), profile error {e:.1e}")
    print(f"   max |tau*omega/gamma - 1| = {e_speed:.2e} (tolerance {TOL_SPEED:.0e}); "
          f"max profile error {e_prof.max():.2e} (tolerance {TOL_TRAVEL:.0e})")
    assert e_speed < TOL_SPEED, f"speed off by {e_speed:.2e}"
    assert e_prof.max() < TOL_TRAVEL, f"travelling profile off by {e_prof.max():.2e}"
    R.update(gam=gam, omega=omega, ratio=ratio, e_speed=e_speed, e_prof=e_prof,
             profiles=profiles)

    # --- 5. path integration -------------------------------------------------
    print(f"\n5. Path integration: OU head velocity, sd {PI_SD} rad/s, correlation "
          f"{PI_TC:.0f} ms, {PI_T / 1000:.0f} s")
    steps = int(round(PI_T / dt))
    a = np.exp(-dt / PI_TC)
    w = np.empty(steps)
    w[0] = PI_SD * rng.standard_normal()
    xi = rng.standard_normal(steps)
    for k in range(1, steps):
        w[k] = a * w[k - 1] + PI_SD * np.sqrt(1 - a * a) * xi[k]
    gamma_t = tau * w / 1000.0                 # tau in ms, w in rad/s
    psi_pi, snaps_pi = simulate_velocity(m_star, gamma_t)
    true = psi_pi[0] + np.r_[0.0, np.cumsum(w * dt / 1000.0)]
    err = psi_pi - true
    t_pi = np.arange(steps + 1) * dt
    max_err, rms_err = np.max(np.abs(err)), np.sqrt(np.mean(err ** 2))
    print(f"   |omega| up to {np.max(np.abs(w)):.2f} rad/s (gamma up to {np.max(np.abs(gamma_t)):.4f}); "
          f"total turning {np.sum(np.abs(w)) * dt / 1000:.1f} rad, net {true[-1] - true[0]:+.2f} rad")
    print(f"   decoded vs integrated heading: max |error| {max_err:.2e} rad, RMS {rms_err:.2e} rad, "
          f"final {err[-1]:+.2e} rad (tolerance {TOL_PATH})")
    assert max_err < TOL_PATH, f"path integration error {max_err:.3f} rad"
    R.update(t_pi=t_pi, true=true, psi_pi=psi_pi, err=err, w=w, snaps_pi=snaps_pi,
             max_err=max_err, rms_err=rms_err)
    return R


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(R):
    C = R["C"]
    th = grid()
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    # A: bump formation from noise
    ax = axes[0, 0]
    sn = R["snaps"]
    for i, k in enumerate([0, 2, 5, 10, 25, len(sn) - 1]):
        ax.plot(np.degrees(th), sn[k], color=plt.cm.viridis(i / 5), lw=1,
                label=f"t = {k * 20 * dt:.0f} ms")
    ax.set_xlabel("preferred direction θ (deg)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title("A. Bump forms from random rates (γ = 0)")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    # B: steady bump vs closed form
    ax = axes[0, 1]
    phi = wrap(th - R["psi_star"])
    o = np.argsort(phi)
    ax.plot(np.degrees(phi[o]), R["m_star"][o], "o", ms=2.5, color="tab:blue", label=f"simulated, N = {N}")
    pp = np.linspace(-np.pi, np.pi, 2000)
    ax.plot(np.degrees(pp), static_profile(pp, C), color="k", lw=1,
            label=f"closed form, θc = {np.degrees(C['theta_c']):.2f}°")
    ax.set_xlabel("θ − ψ (deg)")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"B. Stationary bump: max error {R['prof_err']:.1e} of peak")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ins = ax.inset_axes([0.66, 0.6, 0.31, 0.33])
    ins.loglog(R["Ns"], R["errs_N"], "o-", ms=3)
    ins.set_title(f"error vs N, slope {R['slope_N']:.2f}", fontsize=7)
    ins.tick_params(labelsize=6)

    # C: perturbation decay
    ax = axes[0, 2]
    cols = ["tab:red", "tab:green"]
    for j in range(2):
        tt, y = R["curves"][j]
        ax.semilogy(tt, y, color=cols[j], lw=1.5,
                    label=f"amplitude mode {j + 1}: sim {R['lam_sim'][j]:+.3f}/τ")
        ax.semilogy(tt, np.exp(R["lam"][j] * tt / tau), "--", color="k", lw=0.8)
    ax.semilogy([], [], "--", color="k", lw=0.8, label="theory exp(λt)")
    ax.set_xlabel("time after perturbation (ms)")
    ax.set_ylabel("|δm| / |δm(0)|")
    ax.set_ylim(1e-3, 1.5)
    ax.set_title("C. Perturbation decay vs linearised eigenvalues")
    ax.grid(alpha=0.3, which="both")
    ax2 = ax.twinx()
    tt, y = R["curves"][2]
    ax2.plot(tt, y, color="tab:purple", lw=1.5, label="translation: shift / (ε/B)")
    ax2.set_ylim(0, 1.3)
    ax2.set_ylabel("translation shift / (ε/B)", color="tab:purple")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="lower left")

    # D: spontaneous drift
    ax = axes[0, 3]
    ax.plot(R["t8"] / 1000, R["drift"], lw=1)
    ax.axhline(TOL_DRIFT, color="gray", ls=":")
    ax.axhline(-TOL_DRIFT, color="gray", ls=":", label=f"tolerance ±{TOL_DRIFT} rad")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("ψ(t) − ψ(0) (rad)")
    ax.set_title(f"D. Drift at γ = 0, 8 starts: max {R['max_drift']:.1e} rad")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # E: speed vs gamma
    ax = axes[1, 0]
    gam = R["gam"]
    gg = np.linspace(gam.min(), gam.max(), 200)
    ax.plot(gg, gg / tau * 1000, color="k", lw=1, label="closed form ω = γ/τ")
    ax.plot(gam, R["omega"] * 1000, "o", color="tab:orange", label="simulated")
    ax.set_xlabel("asymmetric gain γ")
    ax.set_ylabel("bump angular speed (rad/s)")
    ax.set_title(f"E. Speed vs γ: max |τω/γ − 1| = {R['e_speed']:.1e}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # F: travelling profiles
    ax = axes[1, 1]
    for i, g in enumerate([0.002, 0.2, 0.5, 1.0, 2.0]):
        p, mm, Mth = R["profiles"][g]
        o = np.argsort(p)
        col = plt.cm.plasma(i / 5)
        ax.plot(np.degrees(p[o]), mm[o], "o", ms=1.8, color=col)
        ax.plot(np.degrees(p[o]), Mth[o], color=col, lw=0.9, label=f"γ = {g}")
    ax.set_xlabel("θ − ψ (deg), bump moving to the right")
    ax.set_ylabel("rate (Hz)")
    ax.set_title(f"F. Travelling bump: dots sim, lines closed form (err ≤ {R['e_prof'].max():.1e})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # G: path integration raster
    ax = axes[1, 2]
    sn = R["snaps_pi"]
    ts = np.arange(sn.shape[0]) * 40 * dt / 1000
    sel = ts <= 6.0
    ax.imshow(sn[sel].T, aspect="auto", origin="lower", cmap="Greys",
              extent=[0, ts[sel][-1], -180, 180])
    tp = R["t_pi"] / 1000
    s2 = tp <= 6.0
    ax.plot(tp[s2], np.degrees(wrap(R["true"][s2])), ".", ms=0.6, color="tab:red",
            label="integrated head velocity")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("preferred direction θ (deg)")
    ax.set_title("G. Rates during the random head-turn trajectory")
    ax.legend(fontsize=8, loc="upper right", markerscale=10)

    # H: path-integration error
    ax = axes[1, 3]
    ax.plot(tp, R["true"], color="tab:red", lw=1.5, label="∫ω dt (true heading)")
    ax.plot(tp, R["psi_pi"], "--", color="k", lw=1, label="decoded bump angle ψ")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("heading, unwrapped (rad)")
    ax.set_title(f"H. Path integration over {PI_T / 1000:.0f} s: max error {R['max_err']:.1e} rad")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    ax2 = ax.twinx()
    ax2.plot(tp, R["err"] * 1e3, color="tab:blue", lw=0.6)
    ax2.set_ylabel("ψ − ∫ω dt (mrad)", color="tab:blue")
    ax2.tick_params(axis="y", colors="tab:blue")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "ring_attractor.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print(f"RNG seed: {SEED} (PCG64)")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
