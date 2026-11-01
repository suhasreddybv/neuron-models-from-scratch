"""Passive cable: closed-form steady state and transient vs a compartmental
solver.

Cable equation (Rall 1959; Hodgkin & Rushton 1946; Dayan & Abbott 2001,
sec. 6.3, with no active currents), V in mV measured from rest,
x in mm, t in ms:

    c_m dV/dt = (a / (2 r_L)) d2V/dx2 - V / r_m + i_e(x, t)

with a the cable radius, r_L the intracellular resistivity, r_m the specific
membrane resistance and c_m the specific membrane capacitance. Dividing by
1/r_m gives the two constants

    lambda = sqrt(a r_m / (2 r_L))      (electrotonic length constant)
    tau_m  = r_m c_m                     (membrane time constant)

and the input resistance of a semi-infinite cable R_inf = r_L lambda / (pi a^2).

Parameters (Dayan & Abbott 2001, sec. 6.3): a = 2 um, r_L = 1 kOhm mm,
r_m = 1 MOhm mm^2, c_m = 10 nF/mm^2, so lambda = 1 mm, tau_m = 10 ms and
R_inf = 79.58 MOhm. Stimulus: a current step I_e = 0.1 nA switched on at
t = 0 and injected at the sealed end x = 0.

Closed forms used for validation (X = x/lambda, ell = L/lambda, T = t/tau_m):
- Finite cable of length L, sealed far end (dV/dx = 0 at x = L):
      V(x) = V0 cosh(ell - X) / cosh(ell),   R_in = R_inf coth(ell)
- Killed far end (V = 0 at x = L):
      V(x) = V0 sinh(ell - X) / sinh(ell),   R_in = R_inf tanh(ell)
- Semi-infinite: V(x) = I_e R_inf exp(-X); electrotonic length of a finite
  sealed cable from its end-to-end attenuation: ell = arccosh(V(0)/V(L)).
- Transient, semi-infinite cable, current step at the sealed end
  (Hodgkin & Rushton 1946; Jack, Noble & Tsien 1975, ch. 3):
      V(X, T) = (I_e R_inf / 2) [ e^-X erfc(X/(2 sqrt T) - sqrt T)
                                 - e^+X erfc(X/(2 sqrt T) + sqrt T) ],
  which reduces to V(0, T) = I_e R_inf erf(sqrt T).

Compartmental model (Dayan & Abbott 2001, sec. 6.4): the cable is cut into
N cylinders of length h = L/N with centres x_j = (j + 1/2) h. Each has
membrane area A = 2 pi a h, capacitance c_m A, leak conductance A/r_m, and is
coupled to its neighbours by the axial conductance g = pi a^2 / (r_L h):

    c_m A dV_j/dt = -A V_j / r_m + g (V_{j+1} - V_j) + g (V_{j-1} - V_j)
                    + I_e delta_{j0}

A sealed end has no neighbour; a killed end is the ghost V_N = -V_{N-1}
(V = 0 on the end face). In matrix form C dV/dt = -G V + I with G
tridiagonal; the steady state solves G V = I.

Integrators: backward Euler and Crank-Nicolson (CN), both implicit, with the
tridiagonal systems solved by the Thomas algorithm (the unbranched case of
Hines 1984), written in NumPy/Python. CN is started with four backward-Euler
half-steps over its first two steps (Rannacher 1984) to damp the stiff modes
excited by the current step. The transient run uses CN at dt = 0.01 ms on a
10 mm (10 lambda) cable with h = 5 um.

Validation (validate()):
1. Closed form, steady state: compartment voltages vs the cosh, sinh and
   exp solutions (L = lambda sealed and killed, L = 10 lambda as the
   semi-infinite cable, compared over 0 <= x <= 5 lambda, where the far-end
   reflection is below 1e-6 of V(0)) at h = 10 um; max error / V(0) < 1e-4.
2. Convergence in h: steady-state error falls as h^2 (fitted order 2 +- 0.1).
3. Closed form, input resistance V(0)/I_e vs R_inf coth(ell) and
   R_inf tanh(ell) for ell = 0.1 ... 4; max relative error < 1e-4.
4. Electrotonic length: lambda fitted to the simulated semi-infinite profile
   and ell = arccosh(V(0)/V(L)) from the L = lambda sealed cable (end values
   extrapolated from the compartment centres) vs the parameters; < 0.1%.
5. Closed form, transient: CN solution vs Hodgkin-Rushton at X = 0, 0.5,
   1, 2 for 0.5 <= t <= 30 ms; max error / (I_e R_inf) < 0.5%.
6. Reference solver, time convergence: on a fixed grid (L = 2 lambda,
   N = 200) both integrators vs the exact-in-time solution from the
   eigendecomposition of the compartment matrix (scipy.linalg.
   eigh_tridiagonal); fitted orders 1 +- 0.1 (backward Euler) and
   2 +- 0.15 (CN).

The model has no randomness, so no RNG is used. SciPy is used only for the
validation (erfc, eigh_tridiagonal, curve_fit).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Dayan & Abbott 2001, sec. 6.3). Units: mm, ms, mV,
# nA, MOhm, nF.  MOhm * nF = ms, nA * MOhm = mV.
# ---------------------------------------------------------------------------
a = 2e-3          # radius, mm (2 um)
r_L = 1e-3        # intracellular resistivity, MOhm mm (1 kOhm mm)
r_m = 1.0         # specific membrane resistance, MOhm mm^2
c_m = 10.0        # specific membrane capacitance, nF/mm^2
I_e = 0.1         # injected current, nA

lam = np.sqrt(a * r_m / (2.0 * r_L))      # mm
tau_m = r_m * c_m                          # ms
R_inf = r_L * lam / (np.pi * a**2)         # MOhm

H_SS = 0.01       # compartment length for steady-state checks, mm
H_TR = 0.005      # compartment length for the transient run, mm
DT_TR = 0.01      # CN timestep for the transient run, ms
T_TR = 30.0       # transient run length, ms
L_LONG = 10.0     # "semi-infinite" cable, mm (10 lambda)


# ---------------------------------------------------------------------------
# Block 2: Model — compartment matrices and the tridiagonal solver
# ---------------------------------------------------------------------------
def compartments(L, N, end="sealed"):
    """Return centres x, membrane area A and the tridiagonal conductance
    matrix G (lower, diag, upper; in uS) of an N-compartment cable."""
    h = L / N
    x = (np.arange(N) + 0.5) * h
    A = 2.0 * np.pi * a * h
    g = np.pi * a**2 / (r_L * h)
    diag = np.full(N, A / r_m + 2.0 * g)
    diag[0] -= g                       # sealed end at x = 0
    if end == "sealed":
        diag[-1] -= g
    elif end == "killed":
        diag[-1] += g                  # ghost V_N = -V_{N-1}: 3g on last
    else:
        raise ValueError(end)
    off = np.full(N - 1, -g)
    return x, A, (off, diag, off.copy())


def thomas_factor(lower, diag, upper):
    """LU factors of a tridiagonal matrix (no pivoting; G is diagonally
    dominant). Returns lists for fast scalar loops."""
    n = len(diag)
    lo, up = lower.tolist(), upper.tolist()
    d = diag.tolist()
    m = [0.0] * n       # multipliers
    piv = [0.0] * n     # pivots
    piv[0] = d[0]
    for i in range(1, n):
        m[i] = lo[i - 1] / piv[i - 1]
        piv[i] = d[i] - m[i] * up[i - 1]
    return m, piv, up


def thomas_solve(fac, rhs):
    m, piv, up = fac
    n = len(piv)
    y = rhs.tolist() if isinstance(rhs, np.ndarray) else list(rhs)
    for i in range(1, n):
        y[i] -= m[i] * y[i - 1]
    y[-1] /= piv[-1]
    for i in range(n - 2, -1, -1):
        y[i] = (y[i] - up[i] * y[i + 1]) / piv[i]
    return np.array(y)


def tri_matvec(lower, diag, upper, v):
    out = diag * v
    out[:-1] += upper * v[1:]
    out[1:] += lower * v[:-1]
    return out


def steady_state(L, N, end="sealed"):
    x, A, (lo, d, up) = compartments(L, N, end)
    rhs = np.zeros(N)
    rhs[0] = I_e
    return x, thomas_solve(thomas_factor(lo, d, up), rhs)


# ---------------------------------------------------------------------------
# Block 3: simulate() — current step from rest, implicit time stepping
# ---------------------------------------------------------------------------
def simulate(L, N, dt, T, method="cn", end="sealed", t_obs=None, rec_idx=None):
    """Integrate C dV/dt = -G V + I from V = 0 with a step I_e at x = 0 on
    from t = 0. Returns times, voltages of compartments rec_idx at every step,
    and full profiles at the times in t_obs."""
    x, A, (lo, d, up) = compartments(L, N, end)
    C = c_m * A
    I = np.zeros(N)
    I[0] = I_e
    n_steps = int(round(T / dt))
    rec_idx = [] if rec_idx is None else list(rec_idx)
    obs_steps = {} if t_obs is None else {int(round(t / dt)): t for t in t_obs}

    # backward Euler: (C/dt + G) V' = C/dt V + I
    def be_factor(step):
        return thomas_factor(lo, d + C / step, up)
    fac_be = be_factor(dt)
    fac_half = be_factor(dt / 2.0)
    # Crank-Nicolson: (C/dt + G/2) V' = (C/dt - G/2) V + I
    fac_cn = thomas_factor(0.5 * lo, 0.5 * d + C / dt, 0.5 * up)

    V = np.zeros(N)
    trace = np.zeros((n_steps + 1, len(rec_idx)))
    profiles = {}
    for n in range(1, n_steps + 1):
        if method == "be":
            V = thomas_solve(fac_be, C / dt * V + I)
        elif method == "cn":
            if n <= 2:   # Rannacher start: two BE half-steps per step
                for _ in range(2):
                    V = thomas_solve(fac_half, C / (dt / 2.0) * V + I)
            else:
                rhs = C / dt * V - 0.5 * tri_matvec(lo, d, up, V) + I
                V = thomas_solve(fac_cn, rhs)
        else:
            raise ValueError(method)
        if rec_idx:
            trace[n] = V[rec_idx]
        if n in obs_steps:
            profiles[obs_steps[n]] = V.copy()
    t = np.arange(n_steps + 1) * dt
    return x, t, trace, profiles


# ---------------------------------------------------------------------------
# Closed forms (validation only)
# ---------------------------------------------------------------------------
def v_sealed(x, L):
    ell, X = L / lam, x / lam
    return I_e * R_inf / np.tanh(ell) * np.cosh(ell - X) / np.cosh(ell)


def v_killed(x, L):
    ell, X = L / lam, x / lam
    return I_e * R_inf * np.tanh(ell) * np.sinh(ell - X) / np.sinh(ell)


def v_semi_inf(x):
    return I_e * R_inf * np.exp(-x / lam)


def v_hodgkin_rushton(x, t):
    from scipy.special import erfc
    X, T = x / lam, t / tau_m
    sT = np.sqrt(T)
    return 0.5 * I_e * R_inf * (np.exp(-X) * erfc(X / (2 * sT) - sT)
                                - np.exp(X) * erfc(X / (2 * sT) + sT))


def exact_in_time(L, N, t_obs):
    """Exact time solution of the semi-discrete system via the symmetric
    eigendecomposition of C^-1 G (uniform compartments, so symmetric)."""
    from scipy.linalg import eigh_tridiagonal
    x, A, (lo, d, up) = compartments(L, N, "sealed")
    C = c_m * A
    w, Q = eigh_tridiagonal(d / C, lo / C)
    s = np.zeros(N)
    s[0] = I_e / C
    c = Q.T @ s / w                    # modal steady-state amplitudes
    return {t: Q @ (c * (1.0 - np.exp(-w * t))) for t in t_obs}


def loglog_slope(h, e):
    return np.polyfit(np.log(h), np.log(e), 1)[0]


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
def validate():
    from scipy.optimize import curve_fit
    R = {}
    print(f"lambda = {lam:.4f} mm, tau_m = {tau_m:.2f} ms, R_inf = {R_inf:.3f} MOhm,"
          f" I_e R_inf = {I_e * R_inf:.4f} mV")

    # 1. Steady state vs closed forms
    print("\n1. Steady state vs closed form (h = 10 um)")
    ss = {}
    for name, L, end, f in (("sealed, L = lambda", lam, "sealed", lambda x: v_sealed(x, lam)),
                            ("killed, L = lambda", lam, "killed", lambda x: v_killed(x, lam)),
                            ("semi-infinite (L = 10 lambda)", L_LONG, "sealed", v_semi_inf)):
        N = int(round(L / H_SS))
        x, V = steady_state(L, N, end)
        m = x <= 5.0 * lam   # semi-infinite comparison: reflection off the
        #   far end at 10 lambda changes V by ~e^(-20+X), negligible for X <= 5
        err = np.max(np.abs(V[m] - f(x[m]))) / f(np.array([0.0]))[0]
        ss[name] = (x, V, err)
        print(f"   {name:30s} N = {N:5d}  V(x_0) = {V[0]:.5f} mV  max err / V(0) = {err:.2e}")
        assert err < 1e-4, f"steady state {name}: {err:.2e} > 1e-4"
    R["ss"] = ss

    # 2. Convergence in h (sealed and killed, L = lambda)
    print("\n2. Convergence in compartment length (L = lambda)")
    Ns = np.array([5, 10, 20, 40, 80, 160, 320, 640])
    conv = {}
    for end, f in (("sealed", v_sealed), ("killed", v_killed)):
        errs = []
        for N in Ns:
            x, V = steady_state(lam, N, end)
            errs.append(np.max(np.abs(V - f(x, lam))) / f(np.array([0.0]), lam)[0])
        errs = np.array(errs)
        p = loglog_slope(lam / Ns, errs)
        conv[end] = (lam / Ns, errs, p)
        print(f"   {end}: error {errs[0]:.2e} (N = {Ns[0]}) -> {errs[-1]:.2e} (N = {Ns[-1]}),"
              f" fitted order {p:.4f}")
        assert abs(p - 2.0) < 0.1, f"{end} spatial order {p:.3f} not 2 +- 0.1"
    R["conv"] = conv

    # 3. Input resistance vs coth / tanh
    print("\n3. Input resistance vs R_inf coth(ell) (sealed) and R_inf tanh(ell) (killed)")
    ells = np.array([0.1, 0.25, 0.5, 1.0, 2.0, 4.0])
    rin = {"sealed": [], "killed": []}
    worst = 0.0
    for ell in ells:
        L = ell * lam
        N = max(int(round(L / H_SS)), 20)
        row = []
        for end, th in (("sealed", R_inf / np.tanh(ell)), ("killed", R_inf * np.tanh(ell))):
            x, V = steady_state(L, N, end)
            # V(0) at the face: quadratic extrapolation from the first three centres
            V0 = np.polyval(np.polyfit(x[:3], V[:3], 2), 0.0)
            r = V0 / I_e
            rin[end].append(r)
            e = r / th - 1
            worst = max(worst, abs(e))
            row.append(f"{end} {r:9.3f} vs {th:9.3f} ({e:+.1e})")
        print(f"   ell = {ell:4.2f}: " + ";  ".join(row))
    print(f"   max relative error {worst:.2e}")
    assert worst < 1e-4, f"input resistance error {worst:.2e} > 1e-4"
    R["rin"] = (ells, {k: np.array(v) for k, v in rin.items()}, worst)

    # 4. Electrotonic length
    print("\n4. Electrotonic length")
    x, V, _ = ss["semi-infinite (L = 10 lambda)"]
    m = x <= 3.0 * lam
    (A0, lam_fit), _ = curve_fit(lambda xx, A0, l: A0 * np.exp(-xx / l), x[m], V[m],
                                 p0=(V[0], 0.8 * lam))
    e_lam = lam_fit / lam - 1
    x1, V1, _ = ss["sealed, L = lambda"]
    V_0 = np.polyval(np.polyfit(x1[:3], V1[:3], 2), 0.0)
    V_L = np.polyval(np.polyfit(x1[-3:], V1[-3:], 2), lam)
    ell_fit = np.arccosh(V_0 / V_L)
    e_ell = ell_fit - 1.0
    print(f"   lambda from exp fit to 0-3 lambda of the long cable: {lam_fit:.6f} mm"
          f" vs sqrt(a r_m / 2 r_L) = {lam:.6f} mm ({e_lam:+.2e})")
    print(f"   ell = arccosh(V(0)/V(L)) for L = 1 mm sealed: {ell_fit:.6f} vs 1 ({e_ell:+.2e});"
          f" V(L)/V(0) = {V_L / V_0:.5f} vs 1/cosh(1) = {1 / np.cosh(1):.5f}")
    assert abs(e_lam) < 1e-3 and abs(e_ell) < 1e-3, "electrotonic length off by > 0.1%"
    R["lam"] = (lam_fit, A0, ell_fit, V_0, V_L)

    # 5. Transient vs Hodgkin-Rushton
    print(f"\n5. Transient, current step at x = 0, CN dt = {DT_TR} ms, h = {H_TR * 1e3:.0f} um,"
          f" L = {L_LONG:.0f} lambda")
    t0 = _time.time()
    N = int(round(L_LONG / H_TR))
    Xs = np.array([0.0, 0.5, 1.0, 2.0])
    xc = (np.arange(N) + 0.5) * H_TR
    idx = [int(np.argmin(np.abs(xc - X * lam))) for X in Xs]
    x, t, trace, prof = simulate(L_LONG, N, DT_TR, T_TR, "cn", rec_idx=idx,
                                 t_obs=[1.0, 3.0, 10.0, 30.0])
    tr_err = []
    m = t >= 0.5
    for k, X in enumerate(Xs):
        ref = v_hodgkin_rushton(x[idx[k]], t[m])
        e = np.max(np.abs(trace[m, k] - ref)) / (I_e * R_inf)
        tr_err.append(e)
        print(f"   X = {X:3.1f} (centre at {x[idx[k]] * 1e3:6.1f} um): V(30 ms) = {trace[-1, k]:.4f} mV,"
              f" max |err| / (I_e R_inf) = {e:.2e}")
    # half-rise times, not asserted
    t_half = []
    for k in range(len(Xs)):
        vfin = v_hodgkin_rushton(x[idx[k]], 1e6)
        t_half.append(np.interp(0.5 * vfin, trace[:, k], t))
    print(f"   half-rise times {', '.join(f'{th:.3f}' for th in t_half)} ms at X = 0, 0.5, 1, 2"
          f" (not asserted); simulate() took {_time.time() - t0:.1f} s")
    worst_tr = max(tr_err)
    assert worst_tr < 5e-3, f"transient error {worst_tr:.2e} > 0.5%"
    R["tr"] = (Xs, x[idx], t, trace, tr_err, t_half, prof, x)

    # 6. Time convergence vs eigendecomposition reference
    print("\n6. Time convergence on L = 2 lambda, N = 200 vs exact-in-time eigendecomposition")
    t_obs = [1.0, 2.0, 5.0, 10.0]
    ref = exact_in_time(2.0 * lam, 200, t_obs)
    dts = np.array([0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])
    tconv = {}
    for meth in ("be", "cn"):
        errs = []
        for dt in dts:
            _, _, _, prof6 = simulate(2.0 * lam, 200, dt, 10.0, meth, t_obs=t_obs)
            errs.append(max(np.max(np.abs(prof6[tt] - ref[tt])) for tt in t_obs) / (I_e * R_inf))
        errs = np.array(errs)
        p = loglog_slope(dts, errs)
        tconv[meth] = (dts, errs, p)
        print(f"   {meth.upper()}: error {errs[0]:.2e} (dt = {dts[0]}) -> {errs[-1]:.2e}"
              f" (dt = {dts[-1]}), fitted order {p:.4f}")
    assert abs(tconv["be"][2] - 1.0) < 0.1, "backward Euler order not 1 +- 0.1"
    assert abs(tconv["cn"][2] - 2.0) < 0.15, "CN order not 2 +- 0.15"
    R["tconv"] = tconv

    print("\nAll assertions passed.")
    return R


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(R):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))

    # A: steady-state profiles
    ax = axes[0, 0]
    xx = np.linspace(0, lam, 200)
    x, V, e = R["ss"]["sealed, L = lambda"]
    ax.plot(xx, v_sealed(xx, lam), color="C0", lw=2, label="cosh closed form (sealed end)")
    ax.plot(x[::5], V[::5], "o", ms=4, mfc="none", color="C0", label=f"compartments, err {e:.1e}")
    x, V, e = R["ss"]["killed, L = lambda"]
    ax.plot(xx, v_killed(xx, lam), color="C3", lw=2, label="sinh closed form (killed end)")
    ax.plot(x[::5], V[::5], "s", ms=4, mfc="none", color="C3", label=f"compartments, err {e:.1e}")
    x, V, e = R["ss"]["semi-infinite (L = 10 lambda)"]
    m = x <= lam
    ax.plot(xx, v_semi_inf(xx), color="k", lw=1.5, ls="--", label="exp(-x/λ) (semi-infinite)")
    ax.plot(x[m][::5], V[m][::5], "^", ms=4, mfc="none", color="k", label=f"10λ cable, err {e:.1e}")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("steady-state V (mV above rest)")
    ax.set_title(f"Steady state, I_e = {I_e} nA at x = 0, L = λ = {lam:.0f} mm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # B: input resistance
    ax = axes[0, 1]
    ells, rin, worst = R["rin"]
    ee = np.logspace(-1.1, np.log10(4.5), 200)
    ax.loglog(ee, R_inf / np.tanh(ee), color="C0", lw=2, label="R_∞ coth(L/λ) (sealed)")
    ax.loglog(ee, R_inf * np.tanh(ee), color="C3", lw=2, label="R_∞ tanh(L/λ) (killed)")
    ax.loglog(ells, rin["sealed"], "o", color="C0", mfc="none", ms=7, label="compartments, sealed")
    ax.loglog(ells, rin["killed"], "s", color="C3", mfc="none", ms=7, label="compartments, killed")
    ax.axhline(R_inf, color="gray", ls=":", label=f"R_∞ = {R_inf:.1f} MΩ")
    ax.set_xlabel("electrotonic length L/λ")
    ax.set_ylabel("input resistance (MΩ)")
    ax.set_title(f"Input resistance vs closed form (max rel. err {worst:.1e})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # C: electrotonic length
    ax = axes[0, 2]
    lam_fit, A0, ell_fit, V_0, V_L = R["lam"]
    x, V, _ = R["ss"]["semi-infinite (L = 10 lambda)"]
    m = x <= 4.0
    ax.semilogy(x[m][::10], V[m][::10], "o", ms=3, color="k", mfc="none", label="10λ cable, compartments")
    ax.semilogy(x[m], A0 * np.exp(-x[m] / lam_fit), color="C2", lw=1.5,
                label=f"exp fit: λ = {lam_fit:.5f} mm (theory {lam:.5f})")
    for X in (1, 2, 3):
        ax.axvline(X * lam, color="gray", lw=0.5)
    ax.text(0.05, 0.08, f"sealed L = 1 mm: V(L)/V(0) = {V_L / V_0:.5f}\n"
            f"arccosh(V(0)/V(L)) = {ell_fit:.5f} (theory 1)", transform=ax.transAxes, fontsize=9,
            bbox=dict(fc="white", ec="gray"))
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("steady-state V (mV, log scale)")
    ax.set_title("Electrotonic length: e-fold decay every λ")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # D: transient
    ax = axes[1, 0]
    Xs, xr, t, trace, tr_err, t_half, prof, xall = R["tr"]
    tt = np.linspace(0.02, t[-1], 400)
    for k, X in enumerate(Xs):
        c = f"C{k}"
        ax.plot(t[::100], trace[::100, k], "o", ms=3.5, mfc="none", color=c)
        ax.plot(tt, v_hodgkin_rushton(xr[k], tt), color=c, lw=1.5,
                label=f"X = {X:.1f}: err {tr_err[k]:.1e}, t½ = {t_half[k]:.2f} ms")
    ax.set_xlabel("t (ms)")
    ax.set_ylabel("V (mV above rest)")
    ax.set_title("Charging after a current step: CN (circles) vs Hodgkin–Rushton")
    ax.legend(fontsize=8, title="err = max|ΔV|/(I_e R_∞), t ≥ 0.5 ms", title_fontsize=8)
    ax.grid(alpha=0.3)

    # E: profiles over time
    ax = axes[1, 1]
    m = xall <= 3.0
    xx = np.linspace(1e-4, 3.0, 300)
    for k, (tk, Vk) in enumerate(sorted(prof.items())):
        c = plt.cm.viridis(k / (len(prof)))
        ax.plot(xall[m][::30], Vk[m][::30], "o", ms=3.5, mfc="none", color=c)
        ax.plot(xx, v_hodgkin_rushton(xx, tk), color=c, lw=1.5, label=f"t = {tk:g} ms")
    ax.plot(xx, v_semi_inf(xx), "k--", lw=1, label="steady state")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("V (mV above rest)")
    ax.set_title("Voltage profile spreading along the cable")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # F: convergence
    ax = axes[1, 2]
    for end, mk, c in (("sealed", "o", "C0"), ("killed", "s", "C3")):
        h, e, p = R["conv"][end]
        ax.loglog(h * 1e3, e, mk + "-", color=c, mfc="none", label=f"space, {end}: order {p:.3f}")
    ax.set_xlabel("compartment length h (µm)  [space]   /   dt (µs)  [time]")
    for meth, mk, c in (("be", "v", "C1"), ("cn", "^", "C2")):
        dts, e, p = R["tconv"][meth]
        ax.loglog(dts * 1e3, e, mk + "--", color=c, mfc="none",
                  label=f"time, {'backward Euler' if meth == 'be' else 'Crank–Nicolson'}: order {p:.3f}")
    ax.set_ylabel("max error / reference V(0)")
    ax.set_title("Convergence: space vs closed form, time vs eigendecomposition")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "passive_cable.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nFigure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print("No RNG: the model is deterministic.")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
