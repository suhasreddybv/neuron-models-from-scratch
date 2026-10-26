"""Connor-Stevens model: Hodgkin-Huxley plus a transient A-type K+ current.

Equations (Connor & Stevens 1971; Connor, Walter & McKown 1977; in the form
and parameters of Dayan & Abbott 2001, sec. 6.2 and eqs. 6.1-6.2):

    c_m dV/dt = -i_m + I_e/A
    i_m = g_L (V - E_L) + g_Na m^3 h (V - E_Na) + g_K n^4 (V - E_K)
          + g_A a^3 b (V - E_A)
    dz/dt = alpha_z(V) (1 - z) - beta_z(V) z          for z = m, h, n
    tau_z(V) dz/dt = z_inf(V) - z                       for z = a, b

    alpha_m = 0.38 (V + 29.7) / (1 - exp(-0.1 (V + 29.7)))
    beta_m  = 15.2 exp(-0.0556 (V + 54.7))
    alpha_h = 0.266 exp(-0.05 (V + 48))
    beta_h  = 3.8 / (1 + exp(-0.1 (V + 18)))
    alpha_n = 0.02 (V + 45.7) / (1 - exp(-0.1 (V + 45.7)))
    beta_n  = 0.25 exp(-0.0125 (V + 55.7))
    a_inf   = [0.0761 exp(0.0314 (V + 94.22)) / (1 + exp(0.0346 (V + 1.17)))]^(1/3)
    tau_a   = 0.3632 + 1.158 / (1 + exp(0.0497 (V + 55.96)))
    b_inf   = [1 / (1 + exp(0.0688 (V + 53.3)))]^4
    tau_b   = 1.24 + 2.678 / (1 + exp(0.0624 (V + 50)))

V in mV, t in ms, rates in 1/ms, c_m in nF/mm^2, conductances in mS/mm^2,
I_e/A in nA/mm^2. Parameters (Dayan & Abbott 2001, sec. 6.2): c_m = 10,
g_L = 0.003, g_Na = 1.2, g_K = 0.2, g_A = 0.477, E_L = -17, E_Na = 55,
E_K = -72, E_A = -75. The comparison model is Hodgkin-Huxley in the same
units and in this repository's convention (src/hodgkin_huxley.py): c_m = 10,
g_L = 0.003, g_Na = 1.2, g_K = 0.36, E_L = -54.4, E_Na = 50, E_K = -77
(1 uA/cm^2 = 10 nA/mm^2). A third model, Connor-Stevens with g_A = 0, isolates
the effect of the A-current.

Integrator: forward Euler with dt = 0.01 ms, all currents of a sweep in
lock-step. A spike is an upward crossing of -20 mV, with the crossing time
interpolated linearly within the step. The removable singularities of
alpha_m and alpha_n are evaluated with their limit (x/(1 - e^{-x/10}) -> 10).

Validation (ROADMAP note: class I f-I curve continuous from 0 Hz vs HH class
II; compare with Dayan & Abbott fig. 6.1):

Fixed points have every gate at its steady state and V solving
I_e/A = I_ss(V) = i_m(V, gates at steady state). For a conductance-based model
of this form det J = 0 exactly where dI_ss/dV = 0, so a saddle-node of the
rest state sits at a local maximum of I_ss. With the A-current, I_ss is not
monotone: the rest state merges with a saddle at I_SN = max I_ss on the rest
branch, and the Jacobian has no other eigenvalue with positive real part
there, so the bifurcation is a saddle-node on an invariant circle. Projecting
the 6-dimensional vector field F onto the null eigenvector r (left null
vector l, l.r = 1) gives the normal form dx/dt = alpha (I - I_SN) + beta x^2
with alpha = l.dF/dI = l_V / c_m and beta = (1/2) l.D^2F[r, r], so

    T(I) = pi / sqrt(alpha beta (I - I_SN)) + T0 + ...

and the rate rises continuously from 0 Hz. The script fits simulated ISIs
0.025-1 nA/mm^2 above onset to this law with I_SN, K and T0 free, and asserts
the fitted I_SN matches max I_ss to 0.1% and K matches alpha*beta to 5%.
For HH and for Connor-Stevens without the A-current I_ss is monotone (one
fixed point), and the rest state loses stability in an Andronov-Hopf
bifurcation where the leading complex pair of the Jacobian crosses the
imaginary axis. The script measures the growth rate of a 1e-3 mV
perturbation at 12 currents around that point (Euler, dt = 0.005 ms, which
shifts the crossing by dt omega^2 / (2 dRe(lambda)/dI)) and asserts the zero
crossing matches the Jacobian prediction to 1%; for HH it also asserts the
Jacobian value against the published I_H = 9.78 uA/cm^2 (Rinzel & Miller
1980; tolerance 0.1%). The lowest current that sustains firing, I_LPC, is
found by following the spiking cycle down in current with the state carried
over (0.5 nA/mm^2 resolution); for HH it is asserted against the published
6.26 uA/cm^2 (tolerance 1%). The rate just above I_LPC is asserted to be
> 20 Hz for both class II models, against < 2 Hz 0.025 nA/mm^2 above onset for
Connor-Stevens, and rest and spiking must coexist halfway between I_LPC and
I_H. Euler ISIs at seven currents are compared with solve_ivp (DOP853,
rtol = atol = 1e-10; tolerance 1%). Dayan & Abbott fig. 6.1 prints no
numbers, so the comparison with it is qualitative (shape of the f-I curves),
stated in the README. The independent checks run in parallel processes.

Jacobians and the second derivative for beta are central differences of the
analytic right-hand side; everything else is closed form. scipy is used only
for root finding, the curve fit and the reference solver.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, curve_fit

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Dayan & Abbott 2001, sec. 6.2; HH as in this repo)
# ---------------------------------------------------------------------------
c_m = 10.0  # nF/mm^2
CS = dict(name="Connor-Stevens", kind="cs", g_L=0.003, g_Na=1.2, g_K=0.2, g_A=0.477,
          E_L=-17.0, E_Na=55.0, E_K=-72.0, E_A=-75.0)
CS0 = dict(CS, name="Connor-Stevens, g_A = 0", g_A=0.0)
HH = dict(name="Hodgkin-Huxley", kind="hh", g_L=0.003, g_Na=1.2, g_K=0.36,
          E_L=-54.4, E_Na=50.0, E_K=-77.0)

dt = 0.01            # Euler step (ms)
V_SPIKE = -20.0      # spike = upward crossing of -20 mV
SEED = 0
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions, fixed points, Jacobian, closed forms
# ---------------------------------------------------------------------------
def vtrap(x):
    """x / (1 - exp(-x/10)) with its limit 10 + x/2 at x = 0."""
    x = np.asarray(x, dtype=float)
    small = np.abs(x) < 1e-7
    xs = np.where(small, 1.0, x)
    return np.where(small, 10.0 + 0.5 * x, xs / (1.0 - np.exp(-0.1 * xs)))


def rates(V, p):
    """alpha, beta of m, h, n."""
    if p["kind"] == "cs":
        return (0.38 * vtrap(V + 29.7), 15.2 * np.exp(-0.0556 * (V + 54.7)),
                0.266 * np.exp(-0.05 * (V + 48.0)), 3.8 / (1.0 + np.exp(-0.1 * (V + 18.0))),
                0.02 * vtrap(V + 45.7), 0.25 * np.exp(-0.0125 * (V + 55.7)))
    return (0.1 * vtrap(V + 40.0), 4.0 * np.exp(-(V + 65.0) / 18.0),
            0.07 * np.exp(-(V + 65.0) / 20.0), 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0)),
            0.01 * vtrap(V + 55.0), 0.125 * np.exp(-(V + 65.0) / 80.0))


def a_inf(V):
    return (0.0761 * np.exp(0.0314 * (V + 94.22)) / (1.0 + np.exp(0.0346 * (V + 1.17)))) ** (1 / 3)


def tau_a(V):
    return 0.3632 + 1.158 / (1.0 + np.exp(0.0497 * (V + 55.96)))


def b_inf(V):
    return (1.0 / (1.0 + np.exp(0.0688 * (V + 53.3)))) ** 4


def tau_b(V):
    return 1.24 + 2.678 / (1.0 + np.exp(0.0624 * (V + 50.0)))


def n_state(p):
    return 6 if p["kind"] == "cs" else 4


def steady_state(V, p):
    """State vector with every gate at its steady state at voltage V."""
    am, bm, ah, bh, an, bn = rates(V, p)
    X = [V, am / (am + bm), ah / (ah + bh), an / (an + bn)]
    if p["kind"] == "cs":
        X += [a_inf(V), b_inf(V)]
    return np.array(X, dtype=float)


def i_mem(X, p):
    """Membrane current in nA/mm^2."""
    V, m, h, n = X[0], X[1], X[2], X[3]
    i = p["g_L"] * (V - p["E_L"]) + p["g_Na"] * m**3 * h * (V - p["E_Na"]) \
        + p["g_K"] * n**4 * (V - p["E_K"])
    if p["kind"] == "cs":
        i = i + p["g_A"] * X[4] ** 3 * X[5] * (V - p["E_A"])
    return 1000.0 * i          # mS/mm^2 * mV = uA/mm^2 -> nA/mm^2


def rhs(X, I, p):
    V = X[0]
    am, bm, ah, bh, an, bn = rates(V, p)
    d = [(I - i_mem(X, p)) / c_m,
         am * (1 - X[1]) - bm * X[1], ah * (1 - X[2]) - bh * X[2], an * (1 - X[3]) - bn * X[3]]
    if p["kind"] == "cs":
        d += [(a_inf(V) - X[4]) / tau_a(V), (b_inf(V) - X[5]) / tau_b(V)]
    return np.array(d)


def I_ss(V, p):
    return i_mem(steady_state(V, p), p)


def jacobian(X, I, p, h=1e-6):
    X = np.asarray(X, float)
    k = X.size
    J = np.empty((k, k))
    for j in range(k):
        e = np.zeros(k)
        e[j] = h * max(1.0, abs(X[j]))
        J[:, j] = (rhs(X + e, I, p) - rhs(X - e, I, p)) / (2 * e[j])
    return J


def rest_V(I, p, lo=-130.0, hi=-20.0):
    """Lowest root of I_ss(V) = I (the rest state)."""
    V = np.linspace(lo, hi, 11001)
    g = I_ss(V, p) - I
    k = np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0][0]
    return brentq(lambda v: I_ss(v, p) - I, V[k], V[k + 1], xtol=1e-13)


def leading_eig(I, p):
    V = rest_V(I, p)
    ev = np.linalg.eigvals(jacobian(steady_state(V, p), I, p))
    return ev[np.argmax(ev.real)]


def saddle_node(p):
    """First local maximum of I_ss on the rest branch: V_SN, I_SN."""
    dI = lambda v: (I_ss(v + 1e-5, p) - I_ss(v - 1e-5, p)) / 2e-5
    V = np.linspace(-75, -40, 3501)
    s = dI(V)
    k = np.nonzero((s[:-1] > 0) & (s[1:] <= 0))[0][0]
    V_sn = brentq(dI, V[k], V[k + 1], xtol=1e-12)
    return V_sn, I_ss(V_sn, p)


def normal_form(p):
    """alpha, beta of dx/dt = alpha (I - I_SN) + beta x^2; also the spectrum at the SN."""
    V_sn, I_sn = saddle_node(p)
    x0 = steady_state(V_sn, p)
    J = jacobian(x0, I_sn, p)
    ev, R = np.linalg.eig(J)
    r = R[:, np.argmin(abs(ev))].real
    r = r / r[0]                               # unit V-component
    evl, L = np.linalg.eig(J.T)
    l = L[:, np.argmin(abs(evl))].real
    l = l / (l @ r)
    alpha = l[0] / c_m
    eps = 0.02
    D2 = (rhs(x0 + eps * r, I_sn, p) + rhs(x0 - eps * r, I_sn, p) - 2 * rhs(x0, I_sn, p)) / eps**2
    beta = 0.5 * l @ D2
    return alpha, beta, ev


def hopf(p, lo, hi):
    """Current where the leading eigenvalue of the rest state crosses Re = 0."""
    I_h = brentq(lambda I: leading_eig(I, p).real, lo, hi, xtol=1e-10)
    return I_h, abs(leading_eig(I_h, p).imag)


# ---------------------------------------------------------------------------
# Block 3: simulate()
# ---------------------------------------------------------------------------
def simulate(I, p, T, X0, dt=dt, record_every=0):
    """Forward Euler for one or many currents in lock-step.

    X0 has shape (n_state,) or (n_state, n_currents). Returns spike-time lists,
    final state and, if record_every > 0, (t, V) sampled every record_every steps."""
    I = np.atleast_1d(np.asarray(I, dtype=float))
    X = np.array(np.broadcast_to(np.asarray(X0, float).reshape(n_state(p), -1),
                                 (n_state(p), I.size)))
    n = int(round(T / dt))
    spikes = [[] for _ in I]
    rec_t, rec_V = [], []
    for i in range(n):
        if record_every and i % record_every == 0:
            rec_t.append(i * dt)
            rec_V.append(X[0].copy())
        Xn = X + dt * rhs(X, I, p)
        for j in np.nonzero((X[0] < V_SPIKE) & (Xn[0] >= V_SPIKE))[0]:
            spikes[j].append((i + (V_SPIKE - X[0, j]) / (Xn[0, j] - X[0, j])) * dt)
        X = Xn
    if record_every:
        return spikes, X, np.array(rec_t), np.array(rec_V)
    return spikes, X


def reference_spikes(I, p, X0, T):
    ev = lambda t, y: y[0] - V_SPIKE
    ev.direction = 1
    sol = solve_ivp(lambda t, y: rhs(y, I, p), (0, T), X0, method="DOP853",
                    rtol=1e-10, atol=1e-10, events=ev)
    return sol.t_events[0]


def last_isi(sp, n=3):
    s = np.asarray(sp)
    return np.mean(np.diff(s[-(n + 1):])) if len(s) >= n + 1 else np.nan


def rate(sp, t0):
    s = np.asarray(sp)
    s = s[s > t0]
    return 1000.0 / np.mean(np.diff(s)) if len(s) >= 3 else 0.0


def cycle_state(p, I=200.0):
    _, X = simulate(I, p, 300.0, steady_state(rest_V(0.0, p), p))
    return X[:, 0]


DT_GROWTH = 0.005    # smaller Euler step for the growth-rate runs (bias dt*omega^2/2)


def growth_rates(I_values, p, T=300.0, dt=DT_GROWTH):
    """Exponential growth rate of a 1e-3 mV perturbation of the rest state."""
    I_values = np.asarray(I_values, float)
    X0 = np.stack([steady_state(rest_V(I, p), p) for I in I_values], axis=1)
    Xs = X0.copy()
    X = X0.copy()
    X[0] += 1e-3
    n = int(round(T / dt))
    w = n // 4
    env = np.zeros((4, I_values.size))
    for i in range(n):
        X = X + dt * rhs(X, I_values, p)
        env[i // w] = np.maximum(env[i // w], np.abs(X[0] - Xs[0]))
    # fit log envelope over windows 2-4 (window 1 holds the transient)
    tc = (np.arange(4) + 0.5) * w * dt
    return np.polyfit(tc[1:], np.log(env[1:]), 1)[0]


def lowest_firing(p, I_start, coarse=3.0, fine=0.5):
    """Follow the spiking cycle down in current, carrying the state over (a jump of
    more than ~1 nA/mm^2 near the fold can land inside the unstable cycle and decay
    to rest). Coarse pass: 300 ms per current, steps of `coarse`, until a run stops
    firing. Fine pass: restart from the state two coarse steps above that current
    and step by `fine` with 800 ms per current, requiring a spike in the last 300 ms.
    Returns I_LPC (midpoint of the last firing and first silent current), half-width,
    and the state on the cycle at the last firing current."""
    X = cycle_state(p, I=I_start + 100.0)
    I, hist = I_start, []
    while True:
        sp, Xn = simulate(I, p, 300.0, X)
        if not (len(sp[0]) and sp[0][-1] > 200.0):
            break
        X = Xn[:, 0]
        hist.append((I, X))
        I -= coarse
    assert len(hist) >= 2
    I, X = hist[-2]
    I_last = None
    while True:
        sp, Xn = simulate(I, p, 800.0, X)
        if not (len(sp[0]) and sp[0][-1] > 500.0):
            break
        I_last, X = I, Xn[:, 0]
        I -= fine
    assert I_last is not None
    return I_last - fine / 2, fine / 2, X


# ---------------------------------------------------------------------------
# Block 4: validate()
# ---------------------------------------------------------------------------
# The four checks below are independent; validate() runs them in parallel
# processes (they share no state and use no random numbers) and prints their
# logs in order.
def check_rest():
    log, out = [], []
    for p, I0 in [(CS, 0.0), (HH, 0.0), (CS0, -100.0)]:
        Vs = rest_V(I0, p)
        X0 = steady_state(Vs, p)
        X0[0] += 1.0
        sp, X = simulate(I0, p, 250.0, X0)
        err = abs(X[0, 0] - Vs)
        log.append(f"{p['name']}: rest at I = {I0:g} nA/mm^2, root V* = {Vs:.6f} mV, simulated "
                   f"{X[0, 0]:.6f} mV (deviation {err:.1e}), spikes {len(sp[0])}")
        out.append((p["name"], I0, Vs, err, len(sp[0])))
    return dict(log=log, rest=out)


def check_saddle_node():
    """Connor-Stevens: saddle-node on the invariant circle and the ISI law."""
    log = []
    V_sn, I_sn = saddle_node(CS)
    alpha, beta, ev = normal_form(CS)
    K_nf = alpha * beta
    others = np.delete(ev, np.argmin(abs(ev)))
    log.append(f"CS saddle-node (max of I_ss on the rest branch): V_SN = {V_sn:.4f} mV, "
               f"I_SN = {I_sn:.5f} nA/mm^2; null eigenvalue {ev[np.argmin(abs(ev))].real:.1e}, "
               f"largest other Re = {others.real.max():.4f} /ms")
    log.append(f"CS normal form: alpha = {alpha:.6f}, beta = {beta:.6f}, "
               f"alpha*beta = {K_nf:.4e} /ms^2 per nA/mm^2")
    dI = np.geomspace(0.025, 1.0, 8)
    below = np.array([-0.5, -0.05])
    X_cyc = cycle_state(CS, I=100.0)
    sp, _ = simulate(I_sn + np.concatenate([dI, below]), CS, 7000.0, X_cyc)
    T_sim = np.array([last_isi(s, 2) for s in sp[:dI.size]])
    conv = max(np.ptp(np.diff(np.asarray(s)[-3:])) / last_isi(s, 2) for s in sp[:dI.size])
    n_late_below = [int(np.sum(np.asarray(s) > 2000.0)) for s in sp[dI.size:]]
    law = lambda I, Is, K, T0: np.pi / np.sqrt(K * np.clip(I - Is, 1e-12, None)) + T0
    (I_fit, K_fit, T0_fit), _ = curve_fit(law, I_sn + dI, T_sim, p0=[I_sn, K_nf, -50.0])
    rel_Isn, rel_K = abs(I_fit - I_sn) / I_sn, abs(K_fit - K_nf) / K_nf
    for d, ts in zip(dI, T_sim):
        log.append(f"   I - I_SN = {d:.4f}: ISI {ts:8.2f} ms ({1000 / ts:6.3f} Hz); "
                   f"pi/sqrt(ab dI) = {np.pi / np.sqrt(K_nf * d):8.2f} ms")
    log.append(f"   last ISIs converged to {conv:.1e} relative")
    log.append(f"CS fit T = pi/sqrt(K (I - I_SN)) + T0: I_SN = {I_fit:.5f} (closed form {I_sn:.5f}, "
               f"{rel_Isn:.1e} relative), K = {K_fit:.4e} ({100 * rel_K:.2f}% from alpha*beta), "
               f"T0 = {T0_fit:.1f} ms")
    log.append(f"CS from the spiking cycle at I_SN - 0.5 and I_SN - 0.05: {n_late_below} spikes after 2000 ms")
    f_min = 1000.0 / T_sim[0]
    log.append(f"CS rate at I_SN + {dI[0]:g}: {f_min:.3f} Hz")
    return dict(log=log, V_sn=V_sn, I_sn=I_sn, K_nf=K_nf, others_max=others.real.max(), dI=dI,
                T_sim=T_sim, conv=conv, n_late_below=n_late_below, I_fit=I_fit, K_fit=K_fit,
                T0_fit=T0_fit, rel_Isn=rel_Isn, rel_K=rel_K, f_min=f_min, X_cyc_cs=X_cyc)


def check_class_II(key):
    """Hopf by Jacobian vs perturbation growth; lowest firing current; bistability."""
    p, (lo, hi) = {"HH": (HH, (80.0, 110.0)), "CS0": (CS0, (-75.0, -50.0))}[key]
    log = []
    V = np.linspace(-130, -20, 11001)
    monotone = bool(np.all(np.diff(I_ss(V, p)) > 0))
    I_h, w_h = hopf(p, lo, hi)
    slope = (leading_eig(I_h + 0.05, p).real - leading_eig(I_h - 0.05, p).real) / 0.1
    span = 5e-3 / abs(slope)          # keep |Re lambda| <= 5e-3 /ms (linear regime)
    Ig = I_h + np.linspace(-span, span, 12)
    g = growth_rates(Ig, p)
    shift = DT_GROWTH * w_h**2 / 2 / slope   # predicted Euler shift of the zero crossing
    lam = np.array([leading_eig(I, p).real for I in Ig])
    k = np.nonzero(np.sign(g[1:]) != np.sign(g[:-1]))[0][0]
    I_h_sw = Ig[k] - g[k] * (Ig[k + 1] - Ig[k]) / (g[k + 1] - g[k])
    rel_h = abs(I_h_sw - I_h) / abs(I_h)
    log.append(f"{p['name']}: I_ss monotone = {monotone} (one fixed point); Hopf from Jacobian "
               f"I_H = {I_h:.5f} nA/mm^2, omega_H/2pi = {1000 * w_h / (2 * np.pi):.1f} Hz")
    log.append(f"   growth of a 1e-3 mV perturbation (Euler dt = {DT_GROWTH} ms) crosses zero at {I_h_sw:.5f} "
               f"({100 * rel_h:.4f}%; predicted Euler shift {-shift:+.4f}, i.e. {I_h - shift:.5f}); "
               f"max |growth - Re lambda| = {np.max(abs(g - lam)):.1e} /ms for |Re lambda| <= {np.max(abs(lam)):.1e}")
    I_lpc, hw, Xc = lowest_firing(p, I_h)
    I_on, I_mid = I_lpc + hw, 0.5 * (I_lpc + I_h)
    Xr = steady_state(rest_V(I_mid, p), p)
    Xr[0] += 1.0
    sp, _ = simulate([I_on, I_mid, I_mid], p, 1000.0, np.stack([Xc, Xc, Xr], axis=1))
    f_on, f_bi, n_rest = rate(sp[0], 300.0), rate(sp[1], 300.0), len(sp[2])
    if p is HH:
        log.append(f"   HH Hopf current {I_h / 10:.4f} uA/cm^2 vs published 9.78 (Rinzel & Miller 1980): "
                   f"{100 * abs(I_h / 10 - 9.78) / 9.78:.3f}%")
        log.append(f"   HH lowest firing current {I_lpc / 10:.3f} +/- {hw / 10:.3f} uA/cm^2 vs published 6.26 "
                   f"(Rinzel & Miller 1980): {100 * (I_lpc / 10 - 6.26) / 6.26:+.2f}%")
    log.append(f"   lowest firing current (cycle followed down) I_LPC = {I_lpc:.2f} +/- {hw:.2f}; "
               f"rate at {I_on:.2f}: {f_on:.1f} Hz")
    log.append(f"   bistable at I = {I_mid:.3f}: from rest + 1 mV {n_rest} spikes, from the cycle {f_bi:.1f} Hz")
    return dict(log=log, name=p["name"], monotone=monotone, I_h=I_h, w_h=w_h, Ig=Ig, g=g, lam=lam,
                I_h_sw=I_h_sw, rel_h=rel_h, shift=shift, I_lpc=I_lpc, hw=hw, f_on=f_on, I_on=I_on,
                f_bi=f_bi, n_rest=n_rest, Xc=Xc)


def check_reference():
    """Euler ISIs vs solve_ivp (Euler runs batched per model)."""
    log, out = [], []
    I_sn = saddle_node(CS)[1]
    cases = [(CS, [I_sn + 1.0, 100.0, 200.0], cycle_state(CS, 100.0), 1500.0),
             (HH, [100.0, 200.0], cycle_state(HH, 200.0), 600.0),
             (CS0, [-50.0, 100.0], cycle_state(CS0, 0.0), 600.0)]
    for p, Is, X0, T in cases:
        sp, _ = simulate(Is, p, T, X0)
        for I, s_eu in zip(Is, sp):
            T_ref, T_eu = last_isi(reference_spikes(I, p, X0, T)), last_isi(s_eu)
            e = (T_eu - T_ref) / T_ref
            log.append(f"   {p['name']:24s} I = {I:9.4f}: ISI solve_ivp {T_ref:9.4f} ms, "
                       f"Euler {T_eu:9.4f} ms ({100 * e:+.3f}%)")
            out.append((p["name"], I, T_ref, T_eu, e))
    max_err = max(abs(r[4]) for r in out)
    log.append(f"Max |Euler - solve_ivp| ISI error at dt = {dt} ms: {100 * max_err:.3f}%")
    return dict(log=log, ref=out, max_err=max_err)


def validate():
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=4) as ex:
        fut = {"sn": ex.submit(check_saddle_node), "HH": ex.submit(check_class_II, "HH"),
               "CS0": ex.submit(check_class_II, "CS0"), "ref": ex.submit(check_reference)}
        rest = check_rest()
        res = {k: f.result() for k, f in fut.items()}
    for r in [rest, res["sn"], res["HH"], res["CS0"], res["ref"]]:
        print("\n".join(r["log"]))

    # assertions (tolerances stated in the module docstring)
    for name, I0, Vs, err, n in rest["rest"]:
        assert err < 1e-4 and n == 0, f"{name}: rest state off by {err}"
    sn = res["sn"]
    assert sn["others_max"] < 0
    assert sn["rel_Isn"] < 1e-3, f"I_SN fit off by {100 * sn['rel_Isn']:.3f}%"
    assert sn["rel_K"] < 0.05, f"normal-form coefficient off by {100 * sn['rel_K']:.2f}%"
    assert sn["f_min"] < 2.0 and sn["conv"] < 1e-4 and sum(sn["n_late_below"]) == 0
    for key in ["HH", "CS0"]:
        c = res[key]
        assert c["monotone"]
        assert c["rel_h"] < 0.01, f"{key}: Hopf by sweep off by {100 * c['rel_h']:.3f}%"
        assert c["f_on"] > 20.0 and c["n_rest"] == 0 and c["f_bi"] > 20.0
    assert abs(res["HH"]["I_h"] / 10 - 9.78) / 9.78 < 1e-3
    assert abs(res["HH"]["I_lpc"] / 10 - 6.26) / 6.26 < 0.01
    assert res["ref"]["max_err"] < 0.01

    R = dict(sn)
    R["classII"] = {res[k]["name"]: res[k] for k in ["HH", "CS0"]}
    R["ref"] = res["ref"]["ref"]
    return R


# ---------------------------------------------------------------------------
# Block 5: plot()
# ---------------------------------------------------------------------------
def plot(R):
    I_sn = R["I_sn"]
    hh, cs0 = R["classII"][HH["name"]], R["classII"][CS0["name"]]
    col = {CS["name"]: "C0", HH["name"]: "C3", CS0["name"]: "C2"}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.5))

    # (a) Connor-Stevens traces
    ax = axs[0, 0]
    Is = np.array([I_sn + 1.0, 100.0, 200.0])
    sp, _, t, V = simulate(Is, CS, 500.0, R["X_cyc_cs"], record_every=10)
    f_lab = [1000.0 / R["T_sim"][-1], rate(sp[1], 0.0), rate(sp[2], 0.0)]
    for j, I in enumerate(Is):
        ax.plot(t, V[:, j] + 130 * (2 - j), lw=0.7, color=["C0", "C9", "C4"][j],
                label=f"I = {I:.2f} ({f_lab[j]:.1f} Hz)")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("V (mV), offset by 130 mV")
    ax.set_title("Connor–Stevens: rate starts near 0 Hz")
    ax.legend(fontsize=7, loc="upper right")

    # (b) class II models just above their lowest firing current
    ax = axs[0, 1]
    for k, (p, d) in enumerate([(HH, hh), (CS0, cs0)]):
        sp, _, t, V = simulate(d["I_on"], p, 300.0, d["Xc"], record_every=10)
        ax.plot(t, V[:, 0] + 130 * (1 - k), lw=0.7, color=col[p["name"]],
                label=f"{p['name']}, I = {d['I_on']:.2f} ({d['f_on']:.0f} Hz)")
    ax.set_xlabel("t (ms)"); ax.set_ylabel("V (mV), offset by 130 mV")
    ax.set_title("Class II: firing starts at a finite rate")
    ax.legend(fontsize=7, loc="upper right")

    # (c) f-I curves
    ax = axs[0, 2]
    Ic = np.linspace(0, 200, 41)
    sp, _ = simulate(Ic, CS, 700.0, R["X_cyc_cs"])
    ax.plot(Ic, [rate(s, 200.0) for s in sp], "o-", ms=3, color=col[CS["name"]], label="Connor–Stevens")
    ax.plot(I_sn + R["dI"], 1000.0 / R["T_sim"], "s", ms=3, color="C0", mfc="none", label="CS, 0.025–1 above I_SN")
    for p, d, Ir in [(HH, hh, np.linspace(0, 200, 41)), (CS0, cs0, np.linspace(-120, 200, 33))]:
        X0 = np.stack([steady_state(rest_V(I, p), p) for I in Ir], axis=1)
        X0[0] += 1.0
        X0 = np.concatenate([X0, np.repeat(d["Xc"][:, None], Ir.size, axis=1)], axis=1)
        sp, _ = simulate(np.concatenate([Ir, Ir]), p, 700.0, X0)
        sp_r, sp_c = sp[:Ir.size], sp[Ir.size:]
        ax.plot(Ir, [rate(s, 200.0) for s in sp_c], "-", color=col[p["name"]], label=f"{p['name']} (from cycle)")
        ax.plot(Ir, [rate(s, 200.0) for s in sp_r], "o", ms=3, mfc="none", color=col[p["name"]], label="  (from rest)")
        ax.axvline(d["I_h"], color=col[p["name"]], ls=":", lw=0.8)
    ax.axvline(I_sn, color="C0", ls=":", lw=0.8)
    for name, I, T_ref, _, _ in R["ref"]:
        ax.plot(I, 1000.0 / T_ref, "kx", ms=6)
    ax.plot([], [], "kx", label="solve_ivp")
    ax.set_xlabel("I_e/A (nA/mm²)"); ax.set_ylabel("rate (Hz)")
    ax.set_title("f–I curves (dotted: I_SN, Hopf currents)")
    ax.legend(fontsize=6.5, loc="upper left")

    # (d) ISI near onset vs saddle-node law
    ax = axs[1, 0]
    d = np.geomspace(0.01, 2, 200)
    ax.loglog(R["dI"], R["T_sim"], "o", color="C0", label="simulated (Euler)")
    ax.loglog(d, np.pi / np.sqrt(R["K_fit"] * (I_sn + d - R["I_fit"])) + R["T0_fit"], "-", color="C0", lw=1,
              label=f"fit: I_SN {R['I_fit']:.4f}, K {R['K_fit']:.3e}, T0 {R['T0_fit']:.0f} ms")
    ax.loglog(d, np.pi / np.sqrt(R["K_nf"] * d), "k--", lw=1, label=f"π/√(αβ ΔI), αβ = {R['K_nf']:.3e}")
    ax.set_xlabel("I − I_SN (nA/mm²)"); ax.set_ylabel("ISI (ms)")
    ax.set_title(f"CS: ISI ∝ (I − I_SN)^(−1/2), I_SN = {I_sn:.4f}")
    ax.legend(fontsize=7)

    # (e) steady-state I-V curves
    ax = axs[1, 1]
    V = np.linspace(-80, -35, 901)
    for p in [CS, CS0, HH]:
        ax.plot(V, I_ss(V, p), color=col[p["name"]], label=p["name"])
    ax.plot(R["V_sn"], I_sn, "ko", ms=5, label="saddle-node (max of I_ss)")
    for p, d in [(HH, hh), (CS0, cs0)]:
        ax.plot(rest_V(d["I_h"], p), d["I_h"], "k^", ms=5)
    ax.plot([], [], "k^", label="Hopf (Jacobian)")
    ax.set_ylim(-200, 300)
    ax.set_xlabel("V (mV)"); ax.set_ylabel("I_ss(V) (nA/mm²)")
    ax.set_title("Fixed points: I_e/A = I_ss(V)")
    ax.legend(fontsize=7)

    # (f) perturbation growth vs Jacobian at the Hopf points
    ax = axs[1, 2]
    for p, d in [(HH, hh), (CS0, cs0)]:
        x = d["Ig"] - d["I_h"]
        ax.plot(x, d["lam"], "-", color=col[p["name"]], label=f"{p['name']}: Re λ (Jacobian)")
        ax.plot(x, d["lam"] + DT_GROWTH * d["w_h"] ** 2 / 2, "--", lw=0.8, color=col[p["name"]],
                label="  Re λ + dt ω²/2 (Euler-shifted)")
        ax.plot(x, d["g"], "o", ms=4, mfc="none", color=col[p["name"]], label=f"  measured growth (Euler, dt = {DT_GROWTH})")
    ax.axhline(0, color="k", lw=0.5); ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("I − I_H (nA/mm²)"); ax.set_ylabel("growth rate (1/ms)")
    ax.set_title("Class II: rest state loses stability at the Hopf point")
    ax.legend(fontsize=7)

    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "connor_stevens.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
