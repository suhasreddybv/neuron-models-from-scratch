"""Hindmarsh-Rose model: bursting vs tonic firing, and the inter-burst interval vs r.

Equations (Hindmarsh & Rose 1984, in the standard three-variable form):

    dx/dt = y - a x^3 + b x^2 - z + I
    dy/dt = c - d x^2 - y
    dz/dt = r (s (x - x_R) - z)

x is a dimensionless membrane potential, y a fast recovery variable and z a
slow adaptation current; time is dimensionless. Parameters: a = 1, b = 3,
c = 1, d = 5, s = 4, x_R = -1.6 (the rounded value of -(1 + sqrt 5)/2 used in
the paper) and r = 0.001 unless stated; I is a constant current.

Integrator: classical fourth-order Runge-Kutta with dt = 0.02. Forward Euler
was tried first and rejected: at dt = 0.01 it shifts the time-averaged x on
the spiking cycle near the end of the burst by 0.13 (-0.766 vs -0.631 at
z - I = 0.0505), which moves the bursting/tonic boundary. A spike is an
upward crossing of x = 1, interpolated linearly within the step. A
"quiescent interval" is an inter-spike interval longer than 100, about twice
the longest period of the fast spiking cycle (48.5). No random numbers are used.

Validation (ROADMAP note: bursting vs tonic regimes as a function of I;
inter-burst interval vs the slow-variable rate r):

1. Rest state. Fixed points satisfy y = c - d x^2, z = s(x - x_R) and
   a x^3 + (d - b) x^2 + s x - c - s x_R - I = 0; with these parameters the
   derivative 3x^2 + 4x + 4 > 0, so there is one fixed point. The simulated
   rest state at I = 1 is compared with the cubic root (tolerance 1e-6).

2. Loss of rest. The 3x3 Jacobian at the fixed point has a complex pair
   crossing the imaginary axis at I_H (r = 0.001: I_H = 1.2699, a slow
   oscillation of period 2 pi / 0.0167 = 376). The growth rate of a small
   perturbation (1e-6 in z) is measured from simulations at 9 currents around I_H and
   its zero crossing compared with I_H (tolerance 0.1%), and each growth
   rate with Re(lambda) (tolerance 5e-5 absolute).

3. Fast-slow dissection (r -> 0). With u = z - I frozen, the (x, y)
   subsystem has equilibria on u = c + (b - d) x^2 - a x^3, whose lower
   branch ends at a fold at x_f = 2(b - d)/(3a) = -4/3, u_SN = -5/27, and a
   spiking cycle that exists for u below u_end (found here with solve_ivp
   by bisection). On the cycle the slow variable drifts at the averaged rate
   du/dt = r (s(<x>(u) - x_R) - I - u). Bursting needs this to be positive
   along the whole cycle branch, so the singular-limit bursting/tonic
   boundary is I_T = min_u [s(<x>(u) - x_R) - u]; rest/bursting is at the
   current that puts the fixed point at the fold, I_0 = s(x_f - x_R) - u_SN.
   The boundary seen in the r = 0.001 sweep is compared with I_T
   (tolerance 2%); the r = 0.0005 boundary is reported, not asserted
   (see README Limitations).

4. Inter-burst interval. During the quiescent phase the state follows the
   lower branch x_lo(u) from u_end down to the fold, so to leading order

       r T_q -> Q(I) = integral_{u_SN}^{u_end} du / (I + u - s(x_lo(u) - x_R)),

   i.e. T_q scales as 1/r. Slow passage through the fold adds a delay: in
   the normal form dX/dt = alpha mu + beta X^2 (alpha = 3/43, beta = 6/43
   here, from the null vectors of the fold Jacobian) with mu growing at
   rate rho = r (I - I_0), the escape time is Omega_0 (alpha beta rho)^(-1/3)
   with Omega_0 = 2.33811 the first zero of -Ai, so

       r T_q = Q + A r^(2/3) + O(r),   A = Omega_0 (alpha beta (I - I_0))^(-1/3).

   Simulations at I = 1.5, 2, 2.5 and r = 0.004 ... 0.00025 are compared
   with this with no free parameters: the log-log slope of r T_q - Q over
   r <= 0.001 must be within 0.1 of 2/3, and Q + A r^(2/3) must match r T_q
   to 5% at r = 0.00025. (A three-parameter fit of Q, A and the O(r) term
   was tried first; it is ill-conditioned over this range of r.) One interval
   (I = 2, r = 0.001) is also compared with solve_ivp (DOP853, rtol = atol
   = 1e-10; tolerance 0.1%).

scipy is used only for validation: root finding, quadrature, the Airy
zero, the reference solver and the fast-subsystem continuation.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq
from scipy.special import ai_zeros

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Hindmarsh & Rose 1984)
# ---------------------------------------------------------------------------
a, b, c, d, s, x_R = 1.0, 3.0, 1.0, 5.0, 4.0, -1.6
R_STD = 0.001          # slow rate used for the regime map
DT = 0.02              # RK4 step
THR = 1.0              # spike detection threshold on x
QUIET = 100.0          # quiescent interval: ISI > QUIET (about twice the longest fast-cycle period, 48.5)


# ---------------------------------------------------------------------------
# Block 2: Model
# ---------------------------------------------------------------------------
def rhs(x, y, z, I, r):
    return (y - a * x**3 + b * x**2 - z + I,
            c - d * x**2 - y,
            r * (s * (x - x_R) - z))


def fixed_point(I):
    """Unique fixed point (x*, y*, z*) from the cubic."""
    x = brentq(lambda x: a * x**3 + (d - b) * x**2 + s * x - c - s * x_R - I, -10, 10, xtol=1e-15)
    return x, c - d * x**2, s * (x - x_R)


def jacobian(I, r):
    x, _, _ = fixed_point(I)
    return np.array([[-3 * a * x**2 + 2 * b * x, 1.0, -1.0],
                     [-2 * d * x, -1.0, 0.0],
                     [r * s, 0.0, -r]])


def max_re_eig(I, r):
    return np.linalg.eigvals(jacobian(I, r)).real.max()


def simulate(I, r, T, x0, y0, z0, record_every=0):
    """Vectorised RK4 over arrays of currents and rates. Returns spike-time
    lists, final state, and (optionally) x, z sampled every record_every steps."""
    I = np.atleast_1d(np.asarray(I, float))
    r = np.asarray(r, float) * np.ones_like(I)
    x = np.asarray(x0, float) * np.ones_like(I)
    y = np.asarray(y0, float) * np.ones_like(I)
    z = np.asarray(z0, float) * np.ones_like(I)
    n = int(round(T / DT)); h = DT / 2
    spikes = [[] for _ in I]
    rec_x, rec_z = [], []
    for k in range(n):
        a1, b1, c1 = rhs(x, y, z, I, r)
        a2, b2, c2 = rhs(x + h * a1, y + h * b1, z + h * c1, I, r)
        a3, b3, c3 = rhs(x + h * a2, y + h * b2, z + h * c2, I, r)
        a4, b4, c4 = rhs(x + DT * a3, y + DT * b3, z + DT * c3, I, r)
        xn = x + DT / 6 * (a1 + 2 * a2 + 2 * a3 + a4)
        y = y + DT / 6 * (b1 + 2 * b2 + 2 * b3 + b4)
        z = z + DT / 6 * (c1 + 2 * c2 + 2 * c3 + c4)
        for i in np.nonzero((x < THR) & (xn >= THR))[0]:
            spikes[i].append((k + (THR - x[i]) / (xn[i] - x[i])) * DT)
        x = xn
        if record_every and k % record_every == 0:
            rec_x.append(x.copy()); rec_z.append(z.copy())
    out = dict(spikes=[np.array(v) for v in spikes], state=(x, y, z))
    if record_every:
        out["t"] = np.arange(0, n, record_every) * DT
        out["x"] = np.array(rec_x); out["z"] = np.array(rec_z)
    return out


def classify(sp, t_skip):
    """'rest', 'bursting' or 'tonic' from spikes after t_skip."""
    s2 = sp[sp > t_skip]
    if len(s2) < 3:
        return "rest", 0, np.array([])
    isi = np.diff(s2)
    q = isi[isi > QUIET]
    if len(q) == 0:
        return "tonic", 0, q
    return "bursting", len(s2) / len(q), q


# ---------------------------------------------------------------------------
# Block 3: Fast subsystem (validation only)
# ---------------------------------------------------------------------------
def _fast(t, v, u):
    x, y, _ = v
    return [y - a * x**3 + b * x**2 - u, c - d * x**2 - y, x]


def _ev(t, v, u):
    return v[0] - THR
_ev.direction = 1


def fast_cycle(u, v0, T=400.0):
    """Period, <x>, min and max x of the fast spiking cycle at frozen u."""
    sol = solve_ivp(_fast, (0, T), [v0[0], v0[1], 0.0], args=(u,), events=_ev,
                    rtol=1e-11, atol=1e-11, method="DOP853", dense_output=True)
    te, ye = sol.t_events[0], sol.y_events[0]
    P = te[-1] - te[-2]
    xs = sol.sol(np.linspace(te[-2], te[-1], 4000))[0]
    return P, (ye[-1][2] - ye[-2][2]) / P, xs.min(), xs.max(), ye[-1][:2]


def fast_cycle_end(v0):
    """Largest u at which the fast spiking cycle persists, by bisection."""
    def fires(u):
        sol = solve_ivp(_fast, (0, 1500), [v0[0], v0[1], 0.0], args=(u,), events=_ev,
                        rtol=1e-11, atol=1e-11, method="DOP853")
        te = sol.t_events[0]
        return len(te) > 2 and te[-1] > 1350
    lo, hi = 0.05, 0.15
    for _ in range(24):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if fires(m) else (lo, m)
    return lo, hi - lo


def x_fold():
    return 2 * (b - d) / (3 * a)


def u_branch(x):
    return c + (b - d) * x**2 - a * x**3


def x_lower(u):
    return brentq(lambda x: u_branch(x) - u, -20, x_fold(), xtol=1e-14)


def fold_normal_form():
    """alpha, beta of dX/dt = alpha mu + beta X^2 at the lower fold, mu = u_SN - u."""
    xf = x_fold()
    rv = np.array([1.0, -2 * d * xf])           # J r = 0
    lv = np.array([1.0, 1.0]) / (1 + rv[1])     # l J = 0, l.r = 1
    alpha = lv @ np.array([1.0, 0.0])
    d2 = np.array([(-6 * a * xf + 2 * b) * rv[0]**2, -2 * d * rv[0]**2])
    beta = 0.5 * lv @ d2
    return alpha, beta


# ---------------------------------------------------------------------------
# Block 4: Validation
# ---------------------------------------------------------------------------
def validate():
    R = {}
    xf = x_fold(); u_sn = u_branch(xf); I0 = s * (xf - x_R) - u_sn
    print(f"Fold of the lower branch: x_f = {xf:.6f}, u_SN = {u_sn:.6f}; singular rest boundary I_0 = {I0:.5f}")

    # --- fast subsystem continuation -------------------------------------
    t0 = _time.time()
    _, _, _, _, v0 = fast_cycle(-0.3, [1.5, -8.0])
    u_end, du = fast_cycle_end(v0)
    us = np.concatenate([np.linspace(-0.6, 0.08, 35), u_end - np.geomspace(3e-3, 1e-8, 6)])
    cyc = []; v = v0
    for u in us:
        P, xm, xmin, xmax, v = fast_cycle(u, v)
        cyc.append((u, P, xm, xmin, xmax))
    cyc = np.array(cyc)
    G = s * (cyc[:, 2] - x_R) - cyc[:, 0]
    I_T = G.min()
    print(f"Fast spiking cycle ends at u_end = {u_end:.8f} (bracket {du:.1e}); period there "
          f"{cyc[-1, 1]:.2f}; singular bursting/tonic boundary I_T = {I_T:.5f} at u = {cyc[np.argmin(G), 0]:.5f}"
          f"  [{_time.time() - t0:.1f} s]")
    R.update(u_sn=u_sn, u_end=u_end, I0=I0, I_T=I_T, cyc=cyc)

    # --- 1 + 3. regime map at r = 0.001 and boundary at r = 0.0005 ---------
    t0 = _time.time()
    I_map = np.round(np.concatenate([np.arange(1.0, 3.6001, 0.1), [1.25, 1.3], np.arange(3.205, 3.30, 0.01)]), 4)
    xs0 = -1.6
    out = simulate(I_map, R_STD, 20000.0, xs0, c - d * xs0**2, 0.0)
    cls = [classify(sp, 8000.0) for sp in out["spikes"]]
    i1 = int(np.argmin(np.abs(I_map - 1.0)))
    fp = np.array(fixed_point(1.0)); st = np.array([v[i1] for v in out["state"]])
    dev = np.abs(st - fp).max()
    print(f"[1] Rest at I = 1: simulated (x, y, z) = ({st[0]:.7f}, {st[1]:.7f}, {st[2]:.7f}), "
          f"cubic root ({fp[0]:.7f}, {fp[1]:.7f}, {fp[2]:.7f}); max deviation {dev:.1e}; "
          f"spikes after transient: {len(out['spikes'][i1][out['spikes'][i1] > 8000])}")
    assert cls[i1][0] == "rest" and dev < 1e-6
    order = np.argsort(I_map); Is = I_map[order]; lab = [cls[i][0] for i in order]
    last_rest = max(I for I, l in zip(Is, lab) if l == "rest")
    first_act = min(I for I, l in zip(Is, lab) if l != "rest")
    last_burst = max(I for I, l in zip(Is, lab) if l == "bursting")
    first_tonic = min(I for I, l in zip(Is, lab) if l == "tonic" and I > 2.5)
    assert last_rest < first_act and all(l == "tonic" for I, l in zip(Is, lab) if I > last_burst)
    assert all(l == "bursting" for I, l in zip(Is, lab) if first_act <= I <= 3.0)
    I_T_sim = 0.5 * (last_burst + first_tonic)
    err_T = (I_T_sim - I_T) / I_T
    print(f"[3] r = {R_STD}: rest up to I = {last_rest:.2f}, firing from {first_act:.2f}; bursting up to "
          f"{last_burst:.3f}, tonic from {first_tonic:.3f} -> boundary {I_T_sim:.4f} vs singular I_T = "
          f"{I_T:.4f} ({100 * err_T:+.2f}%; tolerance 2%)  [{_time.time() - t0:.1f} s]")
    assert abs(err_T) < 0.02
    R.update(I_map=I_map, cls=cls, I_T_sim=I_T_sim, last_burst=last_burst, first_tonic=first_tonic)

    t0 = _time.time()
    I_b = np.round(np.arange(3.12, 3.2601, 0.01), 3)
    out_b = simulate(I_b, 0.0005, 25000.0, xs0, c - d * xs0**2, 0.0)
    lab_b = [classify(sp, 9000.0)[0] for sp in out_b["spikes"]]
    lb = max(I for I, l in zip(I_b, lab_b) if l == "bursting")
    ft = min(I for I, l in zip(I_b, lab_b) if l == "tonic" and I > lb)
    I_T_b = 0.5 * (lb + ft)
    print(f"    r = 0.0005: bursting up to {lb:.2f}, tonic from {ft:.2f} -> boundary {I_T_b:.3f} "
          f"({100 * (I_T_b - I_T) / I_T:+.2f}% from I_T; reported, not asserted)  [{_time.time() - t0:.1f} s]")
    R.update(I_T_b=I_T_b)

    # --- 2. loss of rest: Jacobian vs measured growth rate ----------------
    t0 = _time.time()
    I_H = brentq(lambda I: max_re_eig(I, R_STD), 1.0, 1.5, xtol=1e-12)
    lam = np.linalg.eigvals(jacobian(I_H, R_STD)); om = np.abs(lam.imag).max()
    I_g = I_H + np.linspace(-0.02, 0.02, 9)
    fps = np.array([fixed_point(I) for I in I_g])
    every = 25
    og = simulate(I_g, R_STD, 6000.0, fps[:, 0], fps[:, 1], fps[:, 2] + 1e-6, record_every=every)
    win = 2 * np.pi / om
    tg = og["t"]; growth = []
    for j in range(len(I_g)):
        dv = og["x"][:, j] - fps[j, 0]
        edges = np.arange(1500.0, tg[-1] - win, win)
        rms = [np.sqrt(np.mean(dv[(tg >= e) & (tg < e + win)]**2)) for e in edges]
        growth.append(np.polyfit(edges + win / 2, np.log(rms), 1)[0])
    growth = np.array(growth)
    re_lam = np.array([max_re_eig(I, R_STD) for I in I_g])
    p = np.polyfit(I_g, growth, 1); I_H_sw = -p[1] / p[0]
    err_H = (I_H_sw - I_H) / I_H
    print(f"[2] Hopf of the rest state, r = {R_STD}: Jacobian I_H = {I_H:.6f} (omega = {om:.5f}, period "
          f"{2 * np.pi / om:.1f}); zero of the measured growth rate {I_H_sw:.6f} ({100 * err_H:+.4f}%; "
          f"tolerance 0.1%); max |growth - Re lambda| = {np.abs(growth - re_lam).max():.1e} "
          f"(Re lambda spans {re_lam.min():.1e} to {re_lam.max():.1e}; tolerance 5e-5)  [{_time.time() - t0:.1f} s]")
    I_H_lim = brentq(lambda I: max_re_eig(I, 1e-7), 1.0, 1.5, xtol=1e-12)
    print(f"    singular limit: I_H(r = 1e-7) = {I_H_lim:.5f} vs I_0 = {I0:.5f}")
    assert abs(err_H) < 1e-3 and np.abs(growth - re_lam).max() < 5e-5
    assert last_rest < I_H < first_act
    R.update(I_H=I_H, I_H_sw=I_H_sw, I_g=I_g, growth=growth, re_lam=re_lam)

    # --- 4. inter-burst interval vs r -------------------------------------
    t0 = _time.time()
    alpha, beta = fold_normal_form()
    Om0 = -ai_zeros(1)[0][0]
    I_q = [1.5, 2.0, 2.5]; r_q = [0.004, 0.002, 0.001, 0.0005, 0.00025]
    II = np.repeat(I_q, len(r_q)); RR = np.tile(r_q, len(I_q))
    xl0 = x_lower(u_sn + 0.05)
    oq = simulate(II, RR, 9000.0, xl0, c - d * xl0**2, II + u_sn + 0.05)
    Tq = np.zeros(len(II))
    for k, sp in enumerate(oq["spikes"]):
        kind, _, q = classify(sp, 0.0)
        assert kind == "bursting" and len(q) >= 3, (II[k], RR[k], kind, len(q))
        assert abs(q[-1] - q[-2]) / q[-1] < 1e-3
        Tq[k] = q[-1]
    Tq = Tq.reshape(len(I_q), len(r_q)); rq = np.array(r_q)
    fits = []
    for i, I in enumerate(I_q):
        Q = quad(lambda u: 1.0 / (I + u - s * (x_lower(u) - x_R)), u_sn, u_end, limit=200)[0]
        A = Om0 * (alpha * beta * (I - I0)) ** (-1.0 / 3.0)
        y = rq * Tq[i]; ex = y - Q
        slope = np.polyfit(np.log(rq[rq <= 0.001]), np.log(ex[rq <= 0.001]), 1)[0]
        miss = (y - (Q + A * rq ** (2 / 3))) / y
        miss0 = (y - Q) / y
        Af, Bf = np.linalg.lstsq(np.column_stack([rq ** (2 / 3), rq]), ex, rcond=None)[0]
        print(f"[4] I = {I}: r*T_q = " + ", ".join(f"{v:.4f}" for v in y) +
              f" at r = {', '.join(str(v) for v in r_q)}")
        print(f"    quadrature Q = {Q:.5f}; log-log slope of r*T_q - Q for r <= 0.001: {slope:.3f} "
              f"(law 2/3; tolerance 0.1); Q + A r^(2/3) with Airy A = {A:.3f} misses by "
              + ", ".join(f"{100 * v:.1f}" for v in miss) + f"% (tolerance 5% at r = {r_q[-1]}); "
              f"leading order Q alone misses by {100 * miss0[-1]:.1f}% at r = {r_q[-1]}; "
              f"fit with Q fixed: A = {Af:.2f} ({100 * (Af - A) / A:+.1f}%), B = {Bf:.1f}")
        assert abs(slope - 2 / 3) < 0.1 and abs(miss[-1]) < 0.05
        fits.append((I, Q, A, Af, Bf, slope, miss))
    print(f"    (alpha, beta) = ({alpha:.5f}, {beta:.5f}) = (3/43, 6/43); Omega_0 = {Om0:.5f}  [{_time.time() - t0:.1f} s]")
    R.update(I_q=I_q, r_q=rq, Tq=Tq, fits=fits)

    # reference solver for one interval
    t0 = _time.time()
    def f3(t, v):
        return rhs(v[0], v[1], v[2], 2.0, 0.001)
    ev = lambda t, v: v[0] - THR; ev.direction = 1
    sol = solve_ivp(f3, (0, 3000), [xl0, c - d * xl0**2, 2.0 + u_sn + 0.05], events=ev,
                    rtol=1e-10, atol=1e-10, method="DOP853")
    isi = np.diff(sol.t_events[0]); q_ref = isi[isi > QUIET][-1]
    q_rk = Tq[1, 2]
    print(f"    reference: T_q(I = 2, r = 0.001) = {q_rk:.4f} (RK4, dt = {DT}) vs {q_ref:.4f} (solve_ivp), "
          f"{100 * (q_rk - q_ref) / q_ref:+.4f}% (tolerance 0.1%)  [{_time.time() - t0:.1f} s]")
    assert abs(q_rk - q_ref) / q_ref < 1e-3
    R.update(q_ref=q_ref)
    return R


# ---------------------------------------------------------------------------
# Block 5: Plot
# ---------------------------------------------------------------------------
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    xs0 = -1.6

    # Panel 1: traces
    a_ = ax[0, 0]
    o = simulate([1.0, 2.0, 3.4], R_STD, 11000.0, xs0, c - d * xs0**2, 0.0, record_every=2)
    sel = o["t"] >= 8000
    for j, (I, off, col) in enumerate(((1.0, 0, "gray"), (2.0, 5, "tab:blue"), (3.4, 10, "tab:red"))):
        lbl = {0: "rest", 1: "bursting", 2: "tonic"}[j]
        a_.plot(o["t"][sel] - 8000, o["x"][sel, j] + off, color=col, lw=0.7, label=f"I = {I} ({lbl})")
    a_.set(xlabel="time", ylabel="x (offset by 5 per trace)", title=f"Hindmarsh–Rose, r = {R_STD}")
    a_.legend(fontsize=8, loc="upper right"); a_.grid(alpha=0.3)

    # Panel 2: fast-slow dissection at I = 2
    a_ = ax[0, 1]; I = 2.0
    xx = np.linspace(-2.2, 2.2, 800); uu = u_branch(xx)
    xf = x_fold()
    for m, st, lb in (((xx < xf), "-", "stable rest (lower branch)"), ((xx > xf) & (xx < 0), "--", "saddle"),
                      ((xx > 0), ":", "upper branch")):
        a_.plot(uu[m] + I, xx[m], "k", ls=st, lw=1.2, label=lb)
    cyc = R["cyc"]
    a_.plot(cyc[:, 0] + I, cyc[:, 3], color="tab:green", lw=1.2, label="spiking cycle min / max")
    a_.plot(cyc[:, 0] + I, cyc[:, 4], color="tab:green", lw=1.2)
    a_.plot(cyc[:, 0] + I, cyc[:, 2], color="tab:green", ls="--", lw=1.2, label="cycle average ⟨x⟩")
    xz = np.linspace(-2.2, 2.2, 10)
    a_.plot(s * (xz - x_R), xz, color="tab:orange", lw=1.2, label="z-nullcline z = s(x − x_R)")
    a_.plot(o["z"][sel, 1], o["x"][sel, 1], color="tab:blue", lw=0.4, alpha=0.8, label="trajectory, I = 2")
    a_.axvline(R["u_sn"] + I, color="purple", ls=":", lw=1); a_.axvline(R["u_end"] + I, color="purple", ls=":", lw=1)
    a_.set(xlim=(1.5, 2.4), ylim=(-2.0, 2.2), xlabel="z", ylabel="x",
           title="Fast–slow dissection (dotted: fold and cycle end)")
    a_.legend(fontsize=7, loc="lower left"); a_.grid(alpha=0.3)

    # Panel 3: regime map
    a_ = ax[0, 2]
    I_map = R["I_map"]; cls = R["cls"]
    for I, (kind, spb, q) in zip(I_map, cls):
        if kind == "bursting":
            a_.plot(I, spb, "o", color="tab:blue", ms=4)
        elif kind == "tonic":
            a_.plot(I, 1, "s", color="tab:red", ms=4)
        else:
            a_.plot(I, 0, "x", color="gray", ms=5)
    a_.plot([], [], "o", color="tab:blue", label="bursting: spikes per burst")
    a_.plot([], [], "s", color="tab:red", label="tonic (plotted at 1)")
    a_.plot([], [], "x", color="gray", label="rest (plotted at 0)")
    a_.axvline(R["I_H"], color="black", ls="--", lw=1, label=f"Hopf, Jacobian: {R['I_H']:.4f}")
    a_.axvline(R["I_T"], color="tab:green", ls="--", lw=1, label=f"I_T, fast–slow: {R['I_T']:.4f}")
    a_.axvline(R["I_T_sim"], color="tab:red", ls=":", lw=1, label=f"boundary, r = 0.001: {R['I_T_sim']:.4f}")
    a_.axvline(R["I_T_b"], color="tab:purple", ls=":", lw=1, label=f"boundary, r = 0.0005: {R['I_T_b']:.3f}")
    a_.set(xlabel="I", ylabel="spikes per burst", title="Regimes vs I")
    a_.legend(fontsize=7, loc="upper left"); a_.grid(alpha=0.3)

    # Panel 4: r * T_q vs r
    a_ = ax[1, 0]; rq = R["r_q"]; rr = np.geomspace(rq.min() / 2, rq.max() * 1.2, 100)
    cols = ("tab:blue", "tab:orange", "tab:green")
    for (I, Q, A, Af, Bf, slope, miss), Tq, col in zip(R["fits"], R["Tq"], cols):
        a_.semilogx(rq, rq * Tq, "o", color=col, label=f"I = {I}: simulated")
        a_.semilogx(rr, Q + A * rr ** (2 / 3), "-", color=col, lw=1)
        a_.axhline(Q, color=col, ls="--", lw=0.8)
    a_.plot([], [], "k-", lw=1, label="Q + Ω₀(αβ(I − I₀))^(−1/3) r^(2/3)")
    a_.plot([], [], "k--", lw=0.8, label="Q, singular limit (quadrature)")
    a_.set(xlabel="r", ylabel="r × inter-burst interval", title="Inter-burst interval ∝ 1/r, with fold delay")
    a_.legend(fontsize=7, loc="upper left"); a_.grid(alpha=0.3, which="both")

    # Panel 5: excess vs r
    a_ = ax[1, 1]
    for (I, Q, A, Af, Bf, slope, miss), Tq, col in zip(R["fits"], R["Tq"], cols):
        a_.loglog(rq, rq * Tq - Q, "o", color=col, label=f"I = {I}: r T_q − Q")
        a_.loglog(rr, A * rr ** (2 / 3), "-", color=col, lw=1)
        a_.loglog(rr, Af * rr ** (2 / 3) + Bf * rr, ":", color=col, lw=1)
    a_.plot([], [], "k-", lw=1, label="Airy delay A r^(2/3)")
    a_.plot([], [], "k:", lw=1, label="fit A r^(2/3) + B r (Q fixed)")
    a_.set(xlabel="r", ylabel="r T_q − Q", title="Slow passage through the fold: slope 2/3")
    a_.legend(fontsize=7, loc="upper left"); a_.grid(alpha=0.3, which="both")

    # Panel 6: Hopf growth rate
    a_ = ax[1, 2]
    Ig = np.linspace(R["I_g"].min(), R["I_g"].max(), 200)
    a_.plot(Ig, [max_re_eig(I, R_STD) for I in Ig], "k-", lw=1, label="Re λ, Jacobian")
    a_.plot(R["I_g"], R["growth"], "o", color="tab:red", label="growth rate from RK4 run")
    a_.axhline(0, color="gray", lw=0.8)
    a_.axvline(R["I_H"], color="black", ls="--", lw=1, label=f"I_H = {R['I_H']:.5f}")
    a_.axvline(R["I_H_sw"], color="tab:red", ls=":", lw=1, label=f"zero of measured rate = {R['I_H_sw']:.5f}")
    a_.set(xlabel="I", ylabel="growth rate", title=f"Loss of rest (r = {R_STD}): Hopf")
    a_.legend(fontsize=7, loc="upper left"); a_.grid(alpha=0.3)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "hindmarsh_rose.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print("Deterministic: no random numbers are used.")
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
