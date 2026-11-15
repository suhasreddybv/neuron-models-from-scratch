"""Kuramoto oscillators with a neural phase-response curve.

Model. N weakly coupled neural oscillators firing near f0 = 20 Hz are reduced
to phases theta_i (theta = 0 at the spike). Neuron j drives neuron i through a
synaptic waveform s(theta_j) seen through the phase-response curve Z(theta_i)
of neuron i (Ermentrout & Kopell 1991; Hansel, Mato & Meunier 1995;
Ermentrout 1996):

    dtheta_i/dt = omega_i + (K/N) sum_j Z(theta_i) s(theta_j)          (pulse)

Averaging over one cycle (weak coupling, |omega_i - omega0| << omega0) gives
the Kuramoto form with a neural interaction function H (Kuramoto 1984):

    dphi_i/dt = Delta omega_i + (K/N) sum_j H(phi_j - phi_i),          (phase)
    H(phi) = (1/2pi) int_0^2pi Z(t) s(t + phi) dt,

phi_i = theta_i - omega0 t the phase in the frame rotating at omega0. In
Fourier components, H_n = conj(Z_n) s_n. Two canonical PRCs are used:
    type I  (SNIC onset, theta neuron):  Z(theta) = 1 - cos theta,
    type II (Hopf onset, HH-like):       Z(theta) = -sin theta.
s is an alpha-function synapse (time constant tau_s = 4 ms) triggered at
theta = 0, written in phase units with rate beta = 1/(omega0 tau_s) and
normalised to a cycle mean of 1:
    s(theta) = sum_k 2 pi beta^2 (theta + 2 pi k) exp(-beta (theta + 2 pi k)),
    s_n = beta^2 / (beta + i n)^2.
Both PRCs contain only harmonics 0 and 1, so H = a0 + a1 sin phi + b1 cos phi
exactly, with a0 = Re H_0, a1 = -2 Im H_1, b1 = 2 Re H_1. The classic
Kuramoto model is H = sin phi (a0 = b1 = 0, a1 = 1).

Closed forms used for validation:
(1) Onset of synchrony from incoherence (Kuramoto 1975, 1984; Strogatz 2000):
    K_c = 2 / (pi g(0)) for H = sin, g the density of Delta omega. Only the
    odd first harmonic enters the linear stability of incoherence, so for a
    neural H, K_c = 2 / (pi g(0) a1). For type II with excitation a1 > 0
    when beta > 1; for type I a1 = -rho sin(2 delta) < 0 (rho = beta^2 /
    (1 + beta^2), delta = atan(1/beta)): excitation never synchronises, and
    inhibition (K < 0) synchronises past K_c = 2 / (pi g(0) a1) < 0
    (van Vreeswijk, Abbott & Ermentrout 1994).
(2) Lorentzian g (half-width gamma, g(0) = 1/(pi gamma)): the Ott-Antonsen
    reduction (Ott & Antonsen 2008) of the H = a0 + Im((a1 + i b1) e^{i phi})
    (Sakaguchi-Kuramoto) model is exact as N -> infinity:
        r^2 = 1 - K_c / K,  K_c = 2 gamma / a1,
        Omega = K a0 + K b1 (1 + r^2) / 2   (collective frequency, rotating frame).
(3) Gaussian g (sigma chosen so g(0) is the same as the Lorentzian): the
    self-consistency r = K r int_{-pi/2}^{pi/2} cos^2 x g(K r sin x) dx
    (Kuramoto 1984), solved by quadrature and root finding.

Natural frequencies are deterministic quantiles of g (omega_i = G^{-1}((i -
1/2)/N)), which removes sampling noise in g(0). Initial phases are all near 0
(r = 1) with uniform jitter of +-0.1 rad from a seeded RNG.

Integrators. Phase model: classical RK4, dt = 20 ms (locked oscillators move
by < 0.1 rad per step), in the frame rotating at omega0, mean field through
the order parameter z = (1/N) sum_j exp(i phi_j). Pulse model: forward
Euler, dt = 0.1 ms, absolute phases; a spike when theta crosses 2 pi; the
population synapse S = (1/N) sum_j s_j obeys tau_s dX/dt = -X, tau_s dS/dt =
X - S with X += (2 pi/omega0)/(N tau_s) per spike.

Validation (ROADMAP note: synchronisation transition at the analytical
critical coupling K_c = 2/(pi g(0))): H by quadrature vs its closed form;
K_c fitted from simulated r(K) against (1) for the classic model and for both
PRC types; r(K) against (2) and (3); collective frequency against (2); a dt
halving check; and the un-averaged pulse-coupled network against (1).
"""
import math
import time as _time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters
SEED = 20261115
F0 = 20.0                          # Hz, mean firing rate
OMEGA0 = 2 * np.pi * F0            # rad/s
GAMMA = 2 * np.pi * 0.1            # rad/s, Lorentzian half-width (0.1 Hz)
SIGMA = GAMMA * np.sqrt(np.pi / 2) # rad/s, Gaussian with the same g(0)
TAU_S = 0.004                      # s, alpha synapse
BETA = 1.0 / (OMEGA0 * TAU_S)      # synaptic rate in phase units
M_QUAD = 4096                      # quadrature points for H

N = 2000                           # phase model
DT = 0.02                          # s
T_SIM = 60.0                       # s
T_AVG = 20.0                       # s, averaging window at the end

N_FULL = 1000                      # pulse model
DT_FULL = 1e-4                     # s
T_FULL = 16.0                      # s
T_AVG_FULL = 4.0                   # s

K_MULT = np.array([0.25, 0.5, 0.75, 0.9, 1.1, 1.25, 1.4, 1.6, 1.8, 2.0, 2.3, 2.6, 3.0])
K_MULT_FULL = np.array([0.6, 1.3, 1.6, 2.0, 2.5, 3.0])

TOL_H = 1e-5        # H Fourier coefficients, quadrature vs closed form
TOL_KC = 0.03       # relative, fitted K_c vs closed form (phase model)
TOL_R = 0.02        # absolute, r vs OA / self-consistency for K >= 1.25 K_c
TOL_INCOH = 0.1     # r below 0.75 K_c and for type-I excitation
TOL_OMEGA = 0.03    # relative, collective frequency
TOL_DT = 2e-3       # r change on halving dt
TOL_KC_FULL = 0.10  # relative, pulse model K_c vs averaged prediction
R_FIT = 0.4         # rows with r above this enter the K_c fit


# ---------------------------------------------------------------- model functions
def prc_type1(th):
    return 1.0 - np.cos(th)


def prc_type2(th):
    return -np.sin(th)


def synapse(th, beta=BETA):
    """Periodised alpha function in phase units, cycle mean 1, th in [0, 2 pi)."""
    th = np.mod(th, 2 * np.pi)
    q = np.exp(-2 * np.pi * beta)
    return 2 * np.pi * beta**2 * np.exp(-beta * th) * (th / (1 - q) + 2 * np.pi * q / (1 - q) ** 2)


def interaction(Z, nmax=5, m=M_QUAD):
    """H on a grid and its Fourier coefficients H_0..H_nmax by quadrature."""
    th = 2 * np.pi * np.arange(m) / m
    Zf = np.fft.fft(Z(th)) / m
    sf = np.fft.fft(synapse(th)) / m
    Hf = np.conj(Zf) * sf
    Hgrid = np.real(np.fft.ifft(Hf) * m)       # H(phi_k) = mean_t Z(t) s(t + phi_k)
    return th, Hgrid, Hf[: nmax + 1]


def interaction_closed(kind, beta=BETA):
    s0, s1 = 1.0, beta**2 / (beta + 1j) ** 2
    Zm1, Z0 = (-0.5, 1.0) if kind == "I" else (-0.5j, 0.0)
    return np.array([Z0 * s0, Zm1 * s1])


def harmonics(H01):
    """a0, a1, b1 of H = a0 + a1 sin phi + b1 cos phi."""
    return H01[0].real, -2 * H01[1].imag, 2 * H01[1].real


def lorentz_freqs(n, gamma=GAMMA, wmax=None):
    if wmax is None:
        p = (np.arange(1, n + 1) - 0.5) / n
    else:                                   # quantiles of g truncated to |w| <= wmax
        F = 2 / np.pi * np.arctan(wmax / gamma)
        p = 0.5 + F * ((np.arange(1, n + 1) - 0.5) / n - 0.5)
    return gamma * np.tan(np.pi * (p - 0.5))


def gauss_freqs(n, sigma=SIGMA):
    x = np.linspace(-10, 10, 200001)
    cdf = 0.5 * (1 + np.array([math.erf(v / math.sqrt(2)) for v in x]))
    return sigma * np.interp((np.arange(1, n + 1) - 0.5) / n, cdf, x)


# ---------------------------------------------------------------- simulation
def simulate(omega, Ka0, Kc1, T=T_SIM, dt=DT, t_avg=T_AVG, seed=SEED, keep=5):
    """Phase model, rows = parameter sets. omega (rows, N); Ka0, Kc1 (rows, 1)
    with Kc1 = K (a1 + i b1). Returns time-averaged r, collective frequency,
    and a subsampled r(t)."""
    rng = np.random.default_rng(seed)
    rows, n = omega.shape
    phi = rng.uniform(-0.1, 0.1, (rows, n))

    def f(p):
        e = np.exp(1j * p)
        z = e.mean(axis=1, keepdims=True)
        return omega + Ka0 + (Kc1 * z * np.conj(e)).imag

    steps = int(round(T / dt))
    zs = np.empty((rows, steps + 1), complex)
    zs[:, 0] = np.exp(1j * phi).mean(1)
    for k in range(steps):
        k1 = f(phi)
        k2 = f(phi + 0.5 * dt * k1)
        k3 = f(phi + 0.5 * dt * k2)
        k4 = f(phi + dt * k3)
        phi = phi + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        zs[:, k + 1] = np.exp(1j * phi).mean(1)
    t = np.arange(steps + 1) * dt
    w = t >= T - t_avg
    r = np.abs(zs[:, w]).mean(1)
    ang = np.unwrap(np.angle(zs[:, w]), axis=1)
    Om = np.polyfit(t[w], ang.T, 1)[0]
    return dict(r=r, Omega=Om, t=t[::keep], rt=np.abs(zs[:, ::keep]))


def simulate_pulse(omega_abs, K, Z, T=T_FULL, dt=DT_FULL, t_avg=T_AVG_FULL, seed=SEED, rec=10):
    """Un-averaged pulse-coupled network, rows = coupling strengths K (rows, 1)."""
    rng = np.random.default_rng(seed)
    rows, n = K.shape[0], omega_abs.size
    th = np.mod(rng.uniform(-0.1, 0.1, (rows, n)), 2 * np.pi)
    X = np.zeros((rows, 1))
    S = np.zeros((rows, 1))
    jump = (2 * np.pi / OMEGA0) / (n * TAU_S)
    steps = int(round(T / dt))
    zs = []
    for k in range(steps):
        th += dt * (omega_abs + K * Z(th) * S)
        S += dt / TAU_S * (X - S)
        X -= dt / TAU_S * X
        spk = th >= 2 * np.pi
        if spk.any():
            th[spk] -= 2 * np.pi
            X += jump * spk.sum(1, keepdims=True)
        if k % rec == 0:
            zs.append(np.exp(1j * th).mean(1))
    zs = np.array(zs).T
    t = np.arange(zs.shape[1]) * rec * dt
    w = t >= T - t_avg
    return dict(r=np.abs(zs[:, w]).mean(1), t=t, rt=np.abs(zs))


# ---------------------------------------------------------------- theory
def r_oa(K, Kc):
    K = np.asarray(K, float)
    return np.sqrt(np.clip(1 - Kc / K, 0, None)) * (K / Kc > 1)


def r_gauss_sc(K, sigma=SIGMA):
    from scipy.integrate import quad
    from scipy.optimize import brentq
    g = lambda w: np.exp(-0.5 * (w / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
    F = lambda r: K * quad(lambda x: np.cos(x) ** 2 * g(K * r * np.sin(x)), -np.pi / 2, np.pi / 2,
                           epsabs=1e-13, epsrel=1e-12)[0] - 1
    return brentq(F, 1e-9, 1.0, xtol=1e-13) if F(1e-9) > 0 else 0.0


def fit_kc(K, r):
    """Fit r^2 = A - B/K (exact for Lorentzian g) on rows with r > R_FIT."""
    m = r > R_FIT
    p = np.polyfit(1 / K[m], r[m] ** 2, 1)
    return -p[0] / p[1], p[1], int(m.sum())


# ---------------------------------------------------------------- validation
def validate():
    R = {}
    print(f"f0 = {F0} Hz, gamma = {GAMMA:.4f} rad/s, sigma = {SIGMA:.4f} rad/s, "
          f"tau_s = {TAU_S*1e3:.0f} ms (beta = {BETA:.4f})")

    # --- 1. interaction functions ------------------------------------------
    print("\n1. Interaction function H by quadrature vs closed form H_n = conj(Z_n) s_n")
    coef = {}
    for kind, Z in (("I", prc_type1), ("II", prc_type2)):
        phi, Hg, Hf = interaction(Z)
        Hc = interaction_closed(kind)
        err01 = np.abs(Hf[:2] - Hc).max()
        err2 = np.abs(Hf[2:]).max()
        a0, a1, b1 = harmonics(Hf[:2])
        a0c, a1c, b1c = harmonics(Hc)
        coef[kind] = (a0c, a1c, b1c)
        R["H" + kind] = (phi, Hg)
        print(f"   type {kind:2s}: a0 = {a0:+.6f}, a1 = {a1:+.6f}, b1 = {b1:+.6f} "
              f"(closed form {a0c:+.6f}, {a1c:+.6f}, {b1c:+.6f}); "
              f"max |dH_n|, n<=1: {err01:.1e}; max |H_n|, n=2..5: {err2:.1e}")
        assert err01 < TOL_H and err2 < 1e-12, f"H type {kind} mismatch"
    R["coef"] = coef
    a1_II, a1_I = coef["II"][1], coef["I"][1]
    Kc0 = 2 * GAMMA                      # = 2 / (pi g(0)) for both distributions
    Kc_II = 2 * GAMMA / a1_II
    Kc_I = 2 * GAMMA / a1_I              # negative: inhibition
    print(f"   K_c = 2/(pi g(0) a1): classic {Kc0:.4f}, type II {Kc_II:.4f}, "
          f"type I {Kc_I:.4f} rad/s (inhibitory); type I excitation: no onset (a1 < 0)")

    # --- 2. one batch of phase-model runs ------------------------------------
    wl, wg = lorentz_freqs(N), gauss_freqs(N)
    groups = [("classic_L", wl, Kc0 * K_MULT, (0.0, 1.0, 0.0)),
              ("classic_G", wg, Kc0 * K_MULT, (0.0, 1.0, 0.0)),
              ("II", wl, Kc_II * K_MULT, coef["II"]),
              ("I_inh", wl, Kc_I * K_MULT, coef["I"]),
              ("I_exc", wl, abs(Kc_I) * np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0]), coef["I"])]
    om, Ka0, Kc1, idx = [], [], [], {}
    for name, w, Ks, (a0, a1, b1) in groups:
        idx[name] = slice(len(Ka0), len(Ka0) + len(Ks))
        for K in Ks:
            om.append(w); Ka0.append(K * a0); Kc1.append(K * (a1 + 1j * b1))
    t0 = _time.time()
    sim = simulate(np.array(om), np.array(Ka0)[:, None], np.array(Kc1)[:, None])
    print(f"\n   phase model: {len(Ka0)} runs x N = {N}, T = {T_SIM:.0f} s, dt = {DT*1e3:.0f} ms "
          f"({_time.time()-t0:.1f} s)")
    Ks = {name: g[2] for name, g in zip(idx, groups)}
    R.update(sim=sim, idx=idx, Ks=Ks, Kc0=Kc0, Kc_II=Kc_II, Kc_I=Kc_I)

    # --- 3. onset: fitted K_c vs 2/(pi g(0) a1) -------------------------------
    print("\n2. Synchronisation onset: K_c fitted from r^2 = A - B/K (rows with r > 0.4) "
          "vs 2/(pi g(0) a1)")
    fits = {}
    for name, Kc_th in (("classic_L", Kc0), ("II", Kc_II), ("I_inh", Kc_I)):
        K, r = Ks[name], sim["r"][idx[name]]
        Kc_fit, A, m = fit_kc(K, r)
        e = abs(Kc_fit / Kc_th - 1)
        fits[name] = (Kc_fit, Kc_th, e, A)
        print(f"   {name:9s}: K_c = {Kc_fit:+.4f} (fit, {m} points, A = {A:.4f}) vs {Kc_th:+.4f} "
              f"rad/s; error {100*e:.2f}%")
        assert e < TOL_KC, f"{name}: K_c off by {100*e:.2f}%"
    R["fits"] = fits

    # --- 4. r(K) vs Ott-Antonsen and Gaussian self-consistency ----------------
    print("\n3. Order parameter r(K) vs closed forms (K >= 1.25 K_c), incoherence below 0.75 K_c")
    rdev = {}
    for name, Kc_th in (("classic_L", Kc0), ("II", Kc_II), ("I_inh", Kc_I), ("classic_G", Kc0)):
        K, r = Ks[name], sim["r"][idx[name]]
        th = (np.array([r_gauss_sc(k) for k in K]) if name == "classic_G" else r_oa(K, Kc_th))
        hi, lo = K / Kc_th >= 1.25, K / Kc_th <= 0.75
        d = np.abs(r - th)[hi].max()
        rdev[name] = (d, r[lo].max())
        ref = "self-consistency" if name == "classic_G" else "OA sqrt(1 - K_c/K)"
        print(f"   {name:9s}: max |r - r_{ref}| = {d:.4f}; max r below 0.75 K_c = {r[lo].max():.4f}")
        assert d < TOL_R, f"{name}: r deviates by {d:.4f}"
        assert r[lo].max() < TOL_INCOH, f"{name}: not incoherent below K_c"
    r_exc = sim["r"][idx["I_exc"]]
    print(f"   I_exc    : type I with excitation, K up to {Ks['I_exc'][-1]:.3f} rad/s: "
          f"max r = {r_exc.max():.4f} (no synchrony; tolerance {TOL_INCOH})")
    assert r_exc.max() < TOL_INCOH, "type I excitation synchronised"
    R["rdev"] = rdev
    R["r_gauss_curve"] = (Kc0 * np.linspace(1.0, 3.1, 60),
                          np.array([r_gauss_sc(k) for k in Kc0 * np.linspace(1.0, 3.1, 60)]))

    # --- 5. collective frequency ---------------------------------------------
    print("\n4. Collective frequency vs Omega = K a0 + K b1 (1 + r^2)/2 (rotating frame)")
    omdev = {}
    for name, kind, Kc_th in (("II", "II", Kc_II), ("I_inh", "I", Kc_I)):
        a0, a1, b1 = coef[kind]
        K, r, Om = Ks[name], sim["r"][idx[name]], sim["Omega"][idx[name]]
        m = r > R_FIT
        rr = r_oa(K, Kc_th)
        Om_th = K * a0 + K * b1 * (1 + rr**2) / 2
        e = (np.abs(Om - Om_th) / np.abs(K * b1 * (1 + rr**2) / 2))[m].max()
        omdev[name] = e
        print(f"   {name:9s}: e.g. K = {K[m][-1]:+.3f}: Omega = {Om[m][-1]:+.5f} vs {Om_th[m][-1]:+.5f} "
              f"rad/s; max relative error of the b1 term {100*e:.2f}%")
        assert e < TOL_OMEGA, f"{name}: Omega off by {100*e:.2f}%"
    R["omdev"] = omdev

    # --- 6. timestep check ----------------------------------------------------
    sl = idx["classic_L"]
    sim_h = simulate(np.array(om)[sl], np.array(Ka0)[sl, None], np.array(Kc1)[sl, None], dt=DT / 2)
    ddt = np.abs(sim_h["r"] - sim["r"][sl]).max()
    print(f"\n5. dt = {DT*1e3:.0f} -> {DT*500:.0f} ms (classic, Lorentzian): max |dr| = {ddt:.1e}")
    assert ddt < TOL_DT, "not converged in dt"
    R["ddt"] = ddt

    # --- 7. un-averaged pulse-coupled network ---------------------------------
    wmax = OMEGA0 / 2
    F = 2 / np.pi * np.arctan(wmax / GAMMA)
    Kc_tr = 2 * GAMMA * F / a1_II             # truncated Lorentzian: g(0) = 1/(pi gamma F)
    Kf = Kc_tr * K_MULT_FULL
    t0 = _time.time()
    full = simulate_pulse(OMEGA0 + lorentz_freqs(N_FULL, wmax=wmax), Kf[:, None], prc_type2)
    Kc_full, A_full, m_full = fit_kc(Kf, full["r"])
    e_full = abs(Kc_full / Kc_tr - 1)
    print(f"\n6. Pulse-coupled type-II network (N = {N_FULL}, Euler dt = {DT_FULL*1e3:.1f} ms, "
          f"T = {T_FULL:.0f} s, {_time.time()-t0:.1f} s), Lorentzian truncated at |dw| <= omega0/2")
    print("   r = " + ", ".join(f"{x:.3f}" for x in full["r"]) + "  at K/K_c = "
          + ", ".join(f"{x:.1f}" for x in K_MULT_FULL))
    print(f"   K_c = {Kc_full:.4f} (fit, {m_full} points, A = {A_full:.4f}) vs averaged prediction "
          f"{Kc_tr:.4f} rad/s; error {100*e_full:.2f}% (tolerance {100*TOL_KC_FULL:.0f}%)")
    assert e_full < TOL_KC_FULL, f"pulse model K_c off by {100*e_full:.2f}%"
    R.update(full=full, Kf=Kf, Kc_tr=Kc_tr, fit_full=(Kc_full, e_full, A_full))
    return R


# ---------------------------------------------------------------- plot
def plot(R):
    sim, idx, Ks = R["sim"], R["idx"], R["Ks"]
    C = dict(L="#1f5fa8", G="#d9822b", II="#2a9d5c", I="#b23a48", full="#6a4c93", th="0.15")
    fig, ax = plt.subplots(3, 3, figsize=(15, 13))
    ax = ax.ravel()

    th = np.linspace(0, 2 * np.pi, 400)
    a = ax[0]
    a.plot(th, prc_type1(th), color=C["I"], label="type I PRC  1 − cos θ")
    a.plot(th, prc_type2(th), color=C["II"], label="type II PRC  −sin θ")
    a.plot(th, synapse(th) / synapse(th).max(), color="0.5", ls="--",
           label=f"alpha synapse s(θ), τ_s = {TAU_S*1e3:.0f} ms (scaled)")
    a.set(xlabel="phase θ (rad; spike at 0)", ylabel="Z(θ)", title="(a) PRCs and synaptic waveform")
    a.legend(fontsize=8)

    a = ax[1]
    for kind, col in (("I", C["I"]), ("II", C["II"])):
        phi, Hg = R["H" + kind]
        a0, a1, b1 = R["coef"][kind]
        a.plot(phi, Hg, color=col, lw=3, alpha=0.5, label=f"type {kind}: quadrature")
        a.plot(phi, a0 + a1 * np.sin(phi) + b1 * np.cos(phi), color=C["th"], lw=0.8, ls="--")
        a.text(0.02, 0.10 if kind == "I" else 0.04,
               f"type {kind}: a0 = {a0:+.3f}, a1 = {a1:+.3f}, b1 = {b1:+.3f}",
               transform=a.transAxes, fontsize=8, color=col, va="bottom")
    a.plot([], [], color=C["th"], ls="--", label="closed form")
    a.axhline(0, color="0.8", lw=0.5)
    a.set(xlabel="φ = θ_j − θ_i", ylabel="H(φ)", title="(b) Interaction function H = Z ⋆ s")
    a.legend(fontsize=8, loc="upper left")

    a = ax[2]
    sl = idx["classic_L"]
    for j, km in enumerate(K_MULT):
        if km in (0.5, 0.9, 1.1, 1.4, 2.0, 3.0):
            col = plt.cm.viridis(km / 3.2)
            a.plot(sim["t"], sim["rt"][sl][j], color=col, lw=1, label=f"K = {km:.1f} K_c")
            a.axhline(r_oa(km, 1.0), color=col, ls=":", lw=1)
    a.set(xlabel="time (s)", ylabel="r(t)", ylim=(0, 1.02),
          title="(c) Classic Kuramoto, Lorentzian g: r(t); dotted = OA")
    a.legend(fontsize=7, ncol=2)

    a = ax[3]
    Kc0 = R["Kc0"]
    Kd = np.linspace(0.01, 3.1, 400) * Kc0
    a.plot(Kd / Kc0, r_oa(Kd, Kc0), color=C["L"], lw=1, label="Lorentzian: OA √(1 − K_c/K)")
    Kg, rg = R["r_gauss_curve"]
    a.plot(np.r_[0, 1, Kg / Kc0], np.r_[0, 0, rg], color=C["G"], lw=1, label="Gaussian: self-consistency")
    a.plot(K_MULT, sim["r"][idx["classic_L"]], "o", color=C["L"], ms=6, mfc="none", label=f"simulated Lorentzian, N = {N}")
    a.plot(K_MULT, sim["r"][idx["classic_G"]], "s", color=C["G"], ms=5, mfc="none", label=f"simulated Gaussian, N = {N}")
    a.axvline(1, color="0.4", ls="--", lw=0.8)
    a.text(1.03, 0.45, "K_c = 2/(π g(0))", fontsize=8)
    a.set(xlabel="K / (2/(π g(0)))", ylabel="r", title="(d) Classic Kuramoto (H = sin φ): r vs K", ylim=(0, 1))
    a.legend(fontsize=8, loc="lower right")

    a = ax[4]
    for name, col, lab in (("classic_L", C["L"], "classic"), ("II", C["II"], "type II, excitatory"),
                           ("I_inh", C["I"], "type I, inhibitory")):
        K, r = Ks[name], sim["r"][idx[name]]
        Kc_fit, Kc_th, e, A = R["fits"][name]
        x = Kc_th / K
        m = r > R_FIT
        a.plot(x[m], r[m] ** 2, "o", color=col, mfc="none")
        xx = np.linspace(0, 1.05, 50)
        a.plot(xx, A - A * (Kc_fit / Kc_th) * xx, color=col, lw=1,
               label=f"{lab}: K_c fit/theory = {Kc_fit/Kc_th:.4f}")
    a.plot([0, 1], [1, 0], color=C["th"], ls="--", lw=0.8, label="theory r² = 1 − K_c/K")
    a.set(xlabel="K_c(theory) / K", ylabel="r²", xlim=(0, 1.05), ylim=(0, 1),
          title="(e) Onset: linear fit of r² vs 1/K")
    a.legend(fontsize=8)

    a = ax[5]
    Kc = R["Kc_II"]
    a.plot(Kd / Kc0, r_oa(Kd / Kc0 * Kc, Kc), color=C["II"], lw=1, label="OA, K_c = 2γ/a1")
    a.plot(K_MULT, sim["r"][idx["II"]], "o", color=C["II"], mfc="none", label="phase model (averaged)")
    Kf, full = R["Kf"], R["full"]
    a.plot(Kf / R["Kc_tr"], full["r"], "^", color=C["full"], mfc="none", ms=7,
           label=f"pulse-coupled network, N = {N_FULL}")
    a.axvline(1, color="0.4", ls="--", lw=0.8)
    Kc_full, e_full, _ = R["fit_full"]
    a.text(0.04, 0.9, f"type II: K_c = {Kc:.3f} rad/s\npulse-model fit: {Kc_full:.3f} ({100*e_full:.1f}%)",
           transform=a.transAxes, fontsize=8, va="top")
    a.set(xlabel="K / K_c", ylabel="r", ylim=(0, 1), title="(f) Type II PRC, excitatory alpha synapses")
    a.legend(fontsize=8, loc="lower right")

    a = ax[6]
    KcI = R["Kc_I"]
    Kn = np.linspace(-3.1, -0.01, 300) * abs(KcI)
    a.plot(Kn, r_oa(Kn, KcI), color=C["I"], lw=1, label="OA (inhibition), K_c = 2γ/a1 < 0")
    a.plot(Ks["I_inh"], sim["r"][idx["I_inh"]], "o", color=C["I"], mfc="none", label="simulated, inhibitory")
    a.plot(Ks["I_exc"], sim["r"][idx["I_exc"]], "s", color=C["I"], label="simulated, excitatory")
    a.plot([0, 3.1 * abs(KcI)], [0, 0], color=C["I"], lw=1, ls=":")
    a.axvline(KcI, color="0.4", ls="--", lw=0.8)
    a.axvline(0, color="0.8", lw=0.5)
    a.set(xlabel="K (rad/s; < 0 inhibitory)", ylabel="r", ylim=(-0.02, 1),
          title="(g) Type I PRC: only inhibition synchronises")
    a.legend(fontsize=8, loc="upper right")

    a = ax[7]
    for name, kind, col, Kc_th in (("II", "II", C["II"], R["Kc_II"]), ("I_inh", "I", C["I"], R["Kc_I"])):
        a0, a1, b1 = R["coef"][kind]
        K, r, Om = Ks[name], sim["r"][idx[name]], sim["Omega"][idx[name]]
        m = r > R_FIT
        KK = np.linspace(1.0, 3.1, 100) * Kc_th
        rr = r_oa(KK, Kc_th)
        a.plot(KK / Kc_th, (KK * a0 + KK * b1 * (1 + rr**2) / 2) / (2 * np.pi), color=col, lw=1,
               label=f"type {kind}: K a0 + K b1 (1 + r²)/2")
        a.plot((K / Kc_th)[m], Om[m] / (2 * np.pi), "o", color=col, mfc="none")
    a.set(xlabel="K / K_c", ylabel="(Ω − ω0)/2π  (Hz)", title="(h) Collective frequency shift (synchronous branch)")
    a.legend(fontsize=8)

    a = ax[8]
    j1, j2 = 0, np.argmin(np.abs(K_MULT_FULL - 2.0))
    a.plot(full["t"], full["rt"][j1], color="0.6", lw=0.6, label=f"K = {K_MULT_FULL[j1]:.1f} K_c")
    a.plot(full["t"], full["rt"][j2], color=C["full"], lw=0.6, label=f"K = {K_MULT_FULL[j2]:.1f} K_c")
    a.axhline(r_oa(2.0, 1.0), color=C["th"], ls="--", lw=0.8, label="OA at 2 K_c")
    a.set(xlabel="time (s)", ylabel="r(t)", ylim=(0, 1.02),
          title="(i) Pulse-coupled network: r(t) from a synchronous start")
    a.legend(fontsize=8)

    for a in ax:
        a.grid(alpha=0.25)
    fig.suptitle("Kuramoto oscillators with neural phase-response curves: onset at K_c = 2/(π g(0) a1)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = Path(__file__).resolve().parent.parent / "figures" / "kuramoto_prc.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print(f"RNG seed: {SEED} (PCG64)")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
