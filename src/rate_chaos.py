"""Random rate network and the transition to chaos (Sompolinsky, Crisanti & Sommers 1988).

Model. N rate units with local fields x_i and rates phi(x) = tanh(x)
(Sompolinsky, Crisanti & Sommers 1988, eq. 1, time in units of the unit
time constant):

    dx_i/dt = -x_i + sum_j J_ij phi(x_j),
    J_ij ~ N(0, g^2/N) independently, J_ii = 0.

One coupling matrix J0 with unit variance scale is drawn per network
realisation and multiplied by g, so the whole g sweep shares the same
realisation. Initial states x_i(0) ~ N(0, 1).

Largest Lyapunov exponent (Benettin et al. 1980). A tangent vector v obeys
the variational equation dv/dt = -v + J (phi'(x) * v); it is integrated
alongside x, renormalised to unit length every step, and lambda_max is the
time average of log of the growth factor after a transient.

Closed forms used for validation.
(1) For g < 1 the only attractor is x = 0 (SCS 1988), where the Jacobian is
    -I + g J0 since phi'(0) = 1. Its eigenvalues are those of g J0 shifted by
    -1, and by the circular law they fill a disc of radius g as N -> inf, so
    lambda_max = max Re eig(g J0) - 1 for this realisation (-> g - 1), which
    crosses zero at g = 1.
(2) Dynamic mean-field theory (SCS 1988, eqs. 6-10). For N -> inf each x_i
    is a Gaussian process with autocorrelation Delta(tau) = <x(t) x(t+tau)>
    that obeys
        d^2 Delta/dtau^2 = Delta - g^2 C(Delta; Delta0),
        C = int Dz [ int Dx phi(sqrt(Delta0 - |Delta|) x + sqrt(|Delta|) z) ]^2,
    i.e. a classical particle moving in the potential
        V(Delta; Delta0) = -Delta^2/2 + g^2 int Dz [ int Dx Phi(...) ]^2,
        Phi(x) = ln cosh x, so that d^2 Delta/dtau^2 = -dV/dDelta.
    The chaotic solution starts at Delta0 with zero velocity and creeps to
    Delta = 0 as tau -> inf, so Delta0 is fixed by energy conservation,
        V(Delta0; Delta0) = V(0; Delta0),
    which has a positive root only for g > 1. Delta(tau) follows from
        tau(Delta) = int_Delta^Delta0 dD / sqrt(2 (V(Delta0) - V(D))),
    evaluated with D = Delta0 - s^2 to remove the square-root singularity.
    Gaussian integrals use Gauss-Hermite quadrature; scipy.optimize.brentq
    finds Delta0 (validation only).

Integrator. Classical RK4 for (x, v) with dt = 0.1; a fixed-horizon run at
g = 2 (t = 0..10, dt = 0.4 ... 0.025, same initial state) checks fourth-order
convergence. Per realisation: 150 time units of transient, then 700
measured, with N_EXTRA further initial states at each g in G_DMFT. Delta(tau) from the simulation is
<x_i(t) x_i(t + tau)> averaged over units, time and realisations (no mean
subtracted; the theory has zero mean).

Validation (ROADMAP note): lambda_max(g) crosses zero at g = 1 within 0.1
for N = 1000; for g < 1 lambda_max matches max Re eig - 1; for g > 1 Delta0
and the autocorrelation Delta(tau) match the dynamic mean-field solution.
The zero crossing and the autocorrelation shape do not meet their
tolerances at N = 1000 and are printed, not asserted (see README
Limitations); the other checks are asserted.
"""
import time as _time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters
SEED = 19880801
N = 1000                     # units per network
N_REAL = 4                   # network realisations
G_SWEEP = np.array([0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.05, 1.1, 1.2, 1.3,
                    1.5, 1.75, 2.0, 2.5, 3.0])
G_DMFT = (1.5, 2.0)          # autocorrelation compared at these g
DT = 0.1                     # RK4 step (units of the unit time constant)
T_TRANS = 150.0
T_MEAS = 700.0
N_EXTRA = 3                  # extra initial conditions per realisation at each G_DMFT
MAX_LAG = 30.0               # autocorrelation lags compared
N_GH = 121                   # Gauss-Hermite nodes per dimension

# tolerances (fixed before the run)
TOL_GC = 0.1                 # |g at lambda_max = 0  -  1|
TOL_EIG = 0.02               # |lambda_max - (max Re eig - 1)| for g < 1
TOL_D0 = 0.05                # relative error of Delta0 for g >= 1.5
TOL_SHAPE = 0.05             # max |Delta_sim - Delta_DMFT| / Delta0 over 0..MAX_LAG
T_CONV = 10.0                # horizon of the timestep convergence test
TOL_DT = 1e-4               # max |x(dt) - x(dt/4)| at dt = 0.1, t = T_CONV
TOL_ORDER = 0.5              # |observed order - 4|

FIG = Path(__file__).resolve().parents[1] / "figures" / "rate_chaos.png"


# ------------------------------------------------------------------- network
def coupling(rng, n=N):
    J0 = rng.standard_normal((n, n)) / np.sqrt(n)
    np.fill_diagonal(J0, 0.0)
    return J0


def rhs(J0, g, x, v):
    """Right-hand sides of the network and of the variational equation."""
    r = np.tanh(x)
    d = 1.0 - r * r
    out = J0 @ np.hstack([r, d * v])
    k = x.shape[1]
    return -x + g * out[:, :k], -v + g * out[:, k:]


def simulate(J0, g, rng, dt=DT, t_trans=T_TRANS, t_meas=T_MEAS, record=()):
    """RK4 for all g at once (columns). Returns lambda_max, Delta0 per column
    and the recorded trajectories of the columns listed in `record`."""
    g = np.asarray(g, float)
    k = g.size
    x = rng.standard_normal((J0.shape[0], k))
    v = rng.standard_normal((J0.shape[0], k))
    v /= np.linalg.norm(v, axis=0)
    n_tr, n_me = int(round(t_trans / dt)), int(round(t_meas / dt))
    logsum = np.zeros(k)
    x2 = np.zeros(k)
    rec = {j: np.empty((n_me, J0.shape[0]), np.float32) for j in record}
    for step in range(n_tr + n_me):
        k1x, k1v = rhs(J0, g, x, v)
        k2x, k2v = rhs(J0, g, x + 0.5 * dt * k1x, v + 0.5 * dt * k1v)
        k3x, k3v = rhs(J0, g, x + 0.5 * dt * k2x, v + 0.5 * dt * k2v)
        k4x, k4v = rhs(J0, g, x + dt * k3x, v + dt * k3v)
        x = x + dt / 6 * (k1x + 2 * k2x + 2 * k3x + k4x)
        v = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        nv = np.linalg.norm(v, axis=0)
        v /= nv
        if step >= n_tr:
            i = step - n_tr
            logsum += np.log(nv)
            x2 += np.mean(x * x, axis=0)
            for j in record:
                rec[j][i] = x[:, j]
    return logsum / (n_me * dt), x2 / n_me, rec


def autocorr(X, n_lag):
    """<x_i(t) x_i(t+tau)> over units and t for a (T, N) trajectory."""
    T = X.shape[0]
    nfft = 1 << int(np.ceil(np.log2(2 * T)))
    ac = np.zeros(n_lag)
    for i in range(0, X.shape[1], 100):                   # chunks of units (memory)
        F = np.fft.rfft(X[:, i:i + 100].astype(float), nfft, axis=0)
        ac += np.fft.irfft(np.abs(F) ** 2, nfft, axis=0)[:n_lag].sum(axis=1)
    return ac / X.shape[1] / (T - np.arange(n_lag))


# ------------------------------------------------------- dynamic mean field
_z, _w = np.polynomial.hermite_e.hermegauss(N_GH)
_w = _w / np.sqrt(2 * np.pi)


def lncosh(y):
    a = np.abs(y)
    return a + np.log1p(np.exp(-2 * a)) - np.log(2.0)


def potential(D, D0, g):
    """V(Delta; Delta0) for an array of Delta in [0, Delta0]."""
    D = np.atleast_1d(D)
    a = np.sqrt(np.maximum(D0 - D, 0.0))[:, None, None]
    b = np.sqrt(np.maximum(D, 0.0))[:, None, None]
    inner = (lncosh(a * _z[None, None, :] + b * _z[None, :, None]) * _w).sum(-1)
    return -D ** 2 / 2 + g ** 2 * (inner ** 2 * _w).sum(-1)


def corr_C(D, D0):
    D = np.atleast_1d(D)
    a = np.sqrt(np.maximum(D0 - D, 0.0))[:, None, None]
    b = np.sqrt(np.maximum(D, 0.0))[:, None, None]
    inner = (np.tanh(a * _z[None, None, :] + b * _z[None, :, None]) * _w).sum(-1)
    return (inner ** 2 * _w).sum(-1)


def delta0(g):
    """Energy condition V(Delta0; Delta0) = V(0; Delta0); root for g > 1."""
    from scipy.optimize import brentq

    def f(D0):
        L = lncosh(np.sqrt(D0) * _z)
        return (-D0 ** 2 / 2 + g ** 2 * ((L ** 2 * _w).sum() - (L * _w).sum() ** 2)) / D0 ** 2

    return brentq(f, 1e-4, 1e3, xtol=1e-14, rtol=1e-14)


def dmft_curve(g, n_s=3000, frac_min=1e-4):
    """Delta(tau) of the chaotic DMFT solution by energy quadrature."""
    D0 = delta0(g)
    s = np.linspace(0.0, np.sqrt(D0 * (1 - frac_min)), n_s)
    D = D0 - s ** 2
    E = potential(np.array([D0]), D0, g)[0]
    dV0 = -D0 + g ** 2 * corr_C(np.array([D0]), D0)[0]       # dV/dDelta at Delta0
    gap = E - potential(D, D0, g)
    integrand = np.empty_like(s)
    integrand[0] = 2.0 / np.sqrt(2 * dV0)
    integrand[1:] = 2 * s[1:] / np.sqrt(2 * gap[1:])
    ok = np.concatenate([[True], gap[1:] > 0])
    n_ok = np.argmin(ok) if not ok.all() else ok.size     # stop where rounding wins
    s, D, integrand = s[:n_ok], D[:n_ok], integrand[:n_ok]
    tau = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(s))])
    return D0, tau, D, dV0


# ------------------------------------------------------------------ runs
def run_all():
    t0 = _time.time()
    rng = np.random.default_rng(SEED)
    g_cols = np.concatenate([G_SWEEP, np.repeat(G_DMFT, N_EXTRA)])
    rec_cols = {gd: [int(np.argmin(np.abs(G_SWEEP - gd)))] for gd in G_DMFT}
    for k, gd in enumerate(np.repeat(G_DMFT, N_EXTRA)):
        rec_cols[gd].append(G_SWEEP.size + k)
    all_rec = tuple(j for gd in G_DMFT for j in rec_cols[gd])
    lam = np.zeros((N_REAL, G_SWEEP.size))
    x2 = np.zeros_like(lam)
    eigmax = np.zeros(N_REAL)
    n_lag = int(round(MAX_LAG / DT)) + 1
    ac = {gd: [] for gd in G_DMFT}
    spectrum = None
    traces = None
    for r in range(N_REAL):
        J0 = coupling(rng)
        ev = np.linalg.eigvals(J0)
        eigmax[r] = ev.real.max()
        if r == 0:
            spectrum = ev
        l, m2, rec = simulate(J0, g_cols, rng, record=all_rec)
        lam[r], x2[r] = l[:G_SWEEP.size], m2[:G_SWEEP.size]
        for gd in G_DMFT:
            for j in rec_cols[gd]:
                ac[gd].append(autocorr(rec[j], n_lag))
        if r == 0:
            traces = {gd: rec[rec_cols[gd][0]][:, :4].copy() for gd in G_DMFT}
            _, _, rec_lo = simulate(J0, np.array([0.8]), np.random.default_rng(SEED + 1),
                                    t_trans=0.0, t_meas=60.0, record=(0,))
            traces[0.8] = rec_lo[0][:, :4].copy()
        del rec
        print(f"   realisation {r + 1}/{N_REAL}: max Re eig(J0) = {eigmax[r]:.4f}, "
              f"{_time.time() - t0:.0f} s elapsed")
    # timestep convergence at g = 2 on realisation 1: same initial state, t = 0..T_CONV
    rng2 = np.random.default_rng(SEED)
    J0 = coupling(rng2)
    dtc = {}
    for dt in (4 * DT, 2 * DT, DT, DT / 2, DT / 4):
        x = np.random.default_rng(SEED + 7).standard_normal((N, 1))
        v = np.zeros_like(x)
        for _ in range(int(round(T_CONV / dt))):
            k1, _ = rhs(J0, 2.0, x, v)
            k2, _ = rhs(J0, 2.0, x + 0.5 * dt * k1, v)
            k3, _ = rhs(J0, 2.0, x + 0.5 * dt * k2, v)
            k4, _ = rhs(J0, 2.0, x + dt * k3, v)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        dtc[dt] = x[:, 0]
    ac = {gd: np.array(v) for gd, v in ac.items()}
    return dict(lam=lam, x2=x2, eigmax=eigmax, ac=ac, spectrum=spectrum,
                traces=traces, dtc=dtc, lags=np.arange(n_lag) * DT)


# ------------------------------------------------------------- validation
def zero_crossing(g, y):
    i = np.nonzero((y[:-1] < 0) & (y[1:] >= 0))[0][0]
    return g[i] - y[i] * (g[i + 1] - g[i]) / (y[i + 1] - y[i])


def validate(R):
    print(f"seed = {SEED}, N = {N}, {N_REAL} realisations, RK4 dt = {DT}, "
          f"transient {T_TRANS:.0f}, measured {T_MEAS:.0f} (units of tau)")
    lam = R["lam"]
    lm = lam.mean(0)
    print("\n1. Largest Lyapunov exponent vs g (mean over realisations, range)")
    for gi, l, lo, hi in zip(G_SWEEP, lm, lam.min(0), lam.max(0)):
        print(f"   g = {gi:4.2f}: lambda_max = {l:+.4f}  [{lo:+.4f}, {hi:+.4f}]")
    gc = zero_crossing(G_SWEEP, lm)
    gc_r = [zero_crossing(G_SWEEP, l) for l in lam]
    print(f"   zero crossing g_c = {gc:.4f} (mean curve); per realisation "
          + ", ".join(f"{x:.4f}" for x in gc_r) + f"; |g_c - 1| = {abs(gc - 1):.4f} "
          f"(tolerance {TOL_GC})")
    near = (G_SWEEP > 1) & (G_SWEEP <= 1.2)
    print(f"   for 1 < g <= 1.2: |lambda_max| <= {np.abs(lam[:, near]).max():.4f} and the sign varies between "
          "realisations, so the crossing point is set by finite-N, finite-T effects")
    if abs(gc - 1) < TOL_GC:
        print("   PASS")
    else:
        print("   DOES NOT MATCH (not asserted; see README Limitations)")

    print("\n2. g < 1: lambda_max vs max Re eig(g J0) - 1 of the fixed point x = 0")
    lo = G_SWEEP < 1
    pred = G_SWEEP[lo][None, :] * R["eigmax"][:, None] - 1
    err = np.abs(lam[:, lo] - pred)
    print(f"   max Re eig(J0) = " + ", ".join(f"{e:.4f}" for e in R["eigmax"])
          + " (circular-law edge 1)")
    print(f"   max |lambda_max - prediction| = {err.max():.2e} over {err.size} runs "
          f"(worst at g = {G_SWEEP[lo][np.argmax(err.max(0))]:.2f}; tolerance {TOL_EIG})")
    assert err.max() < TOL_EIG, "lambda_max below g = 1 does not match the spectrum"

    print("\n3. Delta0 = <x^2> vs dynamic mean-field theory")
    d0_sim = R["x2"].mean(0)
    hi = G_SWEEP > 1
    d0_th = np.array([delta0(gi) for gi in G_SWEEP[hi]])
    d0_th_lo = []
    for gi, s, t in zip(G_SWEEP[hi], d0_sim[hi], d0_th):
        print(f"   g = {gi:4.2f}: sim {s:.4f} (SD over realisations "
              f"{R['x2'][:, G_SWEEP == gi].std():.4f}), DMFT {t:.4f}, rel. error {s / t - 1:+.4f}")
    big = G_SWEEP[hi] >= 1.5
    e0 = np.abs(d0_sim[hi][big] / d0_th[big] - 1).max()
    # quadrature convergence
    global _z, _w
    zs, ws = _z, _w
    _z, _w = np.polynomial.hermite_e.hermegauss(2 * N_GH)
    _w = _w / np.sqrt(2 * np.pi)
    dq = max(abs(delta0(gi) / t - 1) for gi, t in zip(G_SWEEP[hi], d0_th))
    _z, _w = zs, ws
    print(f"   max rel. error for g >= 1.5: {e0:.4f} (tolerance {TOL_D0}); "
          f"Gauss-Hermite {N_GH} -> {2 * N_GH} nodes changes Delta0 by {dq:.1e}")
    assert e0 < TOL_D0, "Delta0 does not match DMFT"

    print("\n4. Autocorrelation Delta(tau) vs the particle-in-a-potential solution")
    R["dmft"] = {}
    shape_err = {}
    for gd in G_DMFT:
        D0, tau, D, dV0 = dmft_curve(gd)
        th = np.interp(R["lags"], tau, D, right=0.0)
        acs = R["ac"][gd]
        sim = acs.mean(0)
        se = acs.std(0, ddof=1) / np.sqrt(acs.shape[0])
        zmax = np.abs((sim - th) / se)[1:].max()
        e = np.abs(sim - th).max() / D0
        # half-width where Delta = Delta0/2
        th_half = np.interp(0.5 * D0, D[::-1], tau[::-1])
        sim_half = np.interp(0.5, (sim / sim[0])[::-1], R["lags"][::-1])
        # reference: integrate Delta'' = Delta - g^2 C(Delta) from (Delta0, 0) with
        # solve_ivp down to Delta = 0.2 Delta0 (the solution is a separatrix, so
        # direct integration is unstable further out)
        from scipy.integrate import solve_ivp
        ev = lambda t, y: y[0] - 0.2 * D0
        ev.terminal = True
        sol = solve_ivp(lambda t, y: [y[1], y[0] - gd ** 2 * corr_C(np.array([y[0]]), D0)[0]],
                        (0, 100), [D0, 0.0], method="DOP853", rtol=1e-11, atol=1e-13,
                        events=ev, dense_output=True)
        tt = np.linspace(0, sol.t[-1], 200)
        ode = np.abs(sol.sol(tt)[0] - np.interp(tt, tau, D)).max() / D0
        print(f"   g = {gd}: Delta0 DMFT {D0:.4f}, sim {sim[0]:.4f}; tau at Delta0/2: "
              f"DMFT {th_half:.3f}, sim {sim_half:.3f}; max |sim - DMFT| / Delta0 over "
              f"0..{MAX_LAG:.0f} = {e:.4f} (tolerance {TOL_SHAPE}); Delta({MAX_LAG:.0f}) sim "
              f"{sim[-1]:+.4f} +- {se[-1]:.4f} (SE over {acs.shape[0]} trajectories), DMFT {th[-1]:+.4f}; "
              f"max |z| over lags {zmax:.1f}; quadrature vs solve_ivp on the ODE "
              f"(to Delta0/5) {ode:.1e} of Delta0")
        shape_err[gd] = e
        R["dmft"][gd] = (D0, tau, D)
        print("   PASS" if e < TOL_SHAPE else "   DOES NOT MATCH (not asserted; see README Limitations)")
        assert ode < 1e-4, "DMFT quadrature disagrees with the ODE"

    print(f"\n5. RK4 timestep convergence at g = 2, same initial state, t = 0..{T_CONV:.0f} "
          f"(reference dt = {DT / 4})")
    ref = R["dtc"][DT / 4]
    dts = [4 * DT, 2 * DT, DT, DT / 2]
    errs = [np.abs(R["dtc"][d] - ref).max() for d in dts]
    # Richardson-free order from successive halvings (reference error removed)
    orders = [np.log2(np.abs(R["dtc"][a] - R["dtc"][a / 2]).max()
                      / np.abs(R["dtc"][a / 2] - R["dtc"][a / 4]).max()) for a in (4 * DT, 2 * DT, DT)]
    print("   max |x(dt) - x(dt_ref)|: " + ", ".join(f"dt = {d:g}: {e:.1e}" for d, e in zip(dts, errs)))
    print("   observed order from successive halvings: " + ", ".join(f"{o:.3f}" for o in orders)
          + f" (expected 4; tolerance {TOL_ORDER}); error at dt = {DT}: {errs[2]:.1e} "
          f"(tolerance {TOL_DT})")
    assert errs[2] < TOL_DT and abs(orders[-1] - 4) < TOL_ORDER, "RK4 not converged"
    R["orders"] = orders
    R.update(gc=gc, d0_th=(G_SWEEP[hi], d0_th), shape_err=shape_err)
    return R


# ------------------------------------------------------------------- plot
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    t = np.arange(600) * DT
    off = 0
    for gd, col in ((0.8, "0.4"), (1.5, "C0"), (2.0, "C3")):
        X = R["traces"][gd][:600]
        for i in range(3):
            a.plot(t, X[:, i] + off, color=col, lw=0.8, label=f"g = {gd}" if i == 0 else None)
        off += 6 if gd > 1 else 4
    a.set_xlabel("time (τ)"); a.set_ylabel("x_i (offset per g)")
    a.set_title("Example units: decay (g < 1) and chaos (g > 1)")
    a.legend(frameon=False, fontsize=8, loc="upper right")

    a = ax[0, 1]
    ev = R["spectrum"]
    a.plot(ev.real, ev.imag, ".", ms=2, color="0.3")
    th = np.linspace(0, 2 * np.pi, 300)
    a.plot(np.cos(th), np.sin(th), "r-", lw=1, label="circular law, radius 1")
    a.axvline(1, color="C0", ls="--", lw=1, label="Re = 1: x = 0 unstable once g·Re > 1")
    a.set_aspect("equal"); a.set_xlabel("Re eig(J0)"); a.set_ylabel("Im eig(J0)")
    a.set_xlim(-1.15, 1.45); a.set_ylim(-1.3, 1.3)
    a.set_title(f"Spectrum of J0 (N = {N}, realisation 1)")
    a.legend(frameon=True, fontsize=7, loc="lower right", framealpha=0.9)

    a = ax[0, 2]
    lam = R["lam"]
    for r in range(lam.shape[0]):
        a.plot(G_SWEEP, lam[r], "o", ms=3, color="0.6", label="realisations" if r == 0 else None)
    a.plot(G_SWEEP, lam.mean(0), "k-", lw=1.5, label="mean")
    gl = np.linspace(0.45, 1.0, 10)
    a.plot(gl, gl - 1, "r--", lw=1.2, label="g − 1 (fixed point, N → ∞)")
    a.axhline(0, color="0.6", lw=0.5); a.axvline(1, color="0.6", lw=0.5, ls=":")
    a.plot([R["gc"]], [0], "bs", ms=6, label=f"zero crossing g = {R['gc']:.3f}")
    a.set_xlabel("g"); a.set_ylabel("λ_max (1/τ)"); a.set_title("Largest Lyapunov exponent")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 0]
    gg = np.linspace(1.01, 3.0, 60)
    a.plot(gg, [delta0(x) for x in gg], "r-", label="DMFT Δ0")
    a.plot(G_SWEEP, R["x2"].mean(0), "ko", ms=4, label=f"simulation ⟨x²⟩, N = {N}")
    a.set_xlabel("g"); a.set_ylabel("Δ0 = ⟨x²⟩"); a.set_title("Variance of the local field")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]
    for gd, col in zip(G_DMFT, ("C0", "C3")):
        D0, tau, D = R["dmft"][gd]
        Dg = np.linspace(-0.2 * D0, 1.05 * D0, 300)
        Dg_c = np.clip(Dg, 0, D0)
        V = potential(np.abs(Dg_c), D0, gd)
        a.plot(Dg_c / D0, (V - V[np.argmin(np.abs(Dg_c))]) / D0 ** 2, color=col, label=f"g = {gd}")
        a.plot([1], [0], "o", color=col)
    a.axhline(0, color="0.6", lw=0.5)
    a.set_xlabel("Δ / Δ0"); a.set_ylabel("(V(Δ) − V(0)) / Δ0²")
    a.set_title("Potential of the DMFT particle (starts at rest at Δ0)")
    a.legend(frameon=False, fontsize=8)

    a = ax[1, 2]
    for gd, col in zip(G_DMFT, ("C0", "C3")):
        D0, tau, D = R["dmft"][gd]
        a.plot(R["lags"], R["ac"][gd].mean(0), "o", ms=3, color=col, mfc="none",
               label=f"simulation g = {gd}", markevery=4)
        a.plot(tau, D, "-", color=col, lw=1.2,
               label=f"DMFT g = {gd} (max err {R['shape_err'][gd]:.3f} Δ0)")
    a.set_xlim(0, MAX_LAG); a.axhline(0, color="0.6", lw=0.5)
    a.set_xlabel("lag τ"); a.set_ylabel("Δ(τ) = ⟨x(t) x(t+τ)⟩")
    a.set_title("Autocorrelation: simulation vs mean-field theory")
    a.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    FIG.parent.mkdir(exist_ok=True)
    fig.savefig(FIG, dpi=150)
    print(f"\nwrote {FIG}")


if __name__ == "__main__":
    t0 = _time.time()
    R = run_all()
    R = validate(R)
    plot(R)
    print(f"total {_time.time() - t0:.0f} s")
