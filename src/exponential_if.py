"""Exponential integrate-and-fire (EIF) neuron.

Equations (Fourcaud-Trocme, Hansel, van Vreeswijk & Brunel 2003, J. Neurosci.
23:11628, eq. 2):

    C dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_T)/Delta_T) + I,

with a spike when V reaches V_peak, after which V is held at V_r for an
absolute refractory period tau_ref. V_T is the voltage at which the
exponential term balances the leak (the minimum of dV/dt) and Delta_T is the
"slope factor" that sets how sharply the spike initiates: Delta_T -> 0 gives a
leaky integrate-and-fire neuron with a hard threshold at V_T, while a large
Delta_T gives a slow, graded upstroke.

Parameters are those fitted in Fourcaud-Trocme et al. (2003) to the
Wang-Buzsaki conductance-based model and used throughout that paper:
C = 1 uF/cm^2, g_L = 0.05 mS/cm^2 (tau_m = C/g_L = 20 ms), E_L = -65 mV,
V_T = -59.9 mV, Delta_T = 3.48 mV, V_r = -68 mV, tau_ref = 1.7 ms. V in mV,
t in ms, I in uA/cm^2 (so dV/dt is in mV/ms). The cut-off V_peak = 20 mV is
a numerical choice: the exact solution diverges in finite time and the time
spent between 0 mV and V_peak is ~1e-6 ms, so the rate does not depend on it
(the script prints that time).

Closed forms used for validation.
  * Nullcline / fixed points. With x = (V - V_T)/Delta_T the steady state
    solves e^x - x = k(I), k = [g_L (V_T - E_L) - I] / (g_L Delta_T). The
    right-hand side of the ODE is minimal at V = V_T (its derivative
    -g_L + g_L e^x vanishes there), so fixed points exist only while k >= 1,
    i.e. while I <= I_rh = g_L (V_T - E_L - Delta_T): the stable rest state
    and the saddle (threshold) merge at V_T in a saddle-node bifurcation
    at the rheobase I_rh = 0.05 * (5.1 - 3.48) = 0.081 uA/cm^2. The two
    roots are x = -k - W_0(-e^{-k}) (stable, x < 0) and x = -k - W_{-1}(-e^{-k})
    (saddle, x > 0), where W is the Lambert W function (scipy.special.lambertw).
  * Firing rate. For I > I_rh the inter-spike interval is the quadrature
        T(I) = tau_ref + integral_{V_r}^{V_peak} C dV / F(V),
    F(V) = -g_L (V - E_L) + g_L Delta_T e^{(V - V_T)/Delta_T} + I, evaluated
    here with scipy.integrate.quad (a reference solution of the same ODE by
    an independent method). The rate f = 1/T rises continuously from 0 Hz at
    I_rh (class I, saddle-node on an invariant circle).
  * LIF limit. As Delta_T -> 0 with the other parameters fixed, F(V) -> leak
    only below V_T and the ISI tends to the LIF value already validated in
    this repository (src/leaky_integrate_and_fire.py) with threshold V_T:
        T_LIF = tau_ref + tau_m ln[(V_inf - V_r)/(V_inf - V_T)],  V_inf = E_L + I/g_L.
    The EIF ISI exceeds it because the exponential term only takes over once
    e^x ~ a/Delta_T, a = V_inf - V_T, so the neuron spends extra time just
    above V_T. Linearising the leak about V_T, the excess is
        T_EIF - T_LIF ~ tau_m (Delta_T / a) ln(1 + a / Delta_T)
    (from int_0^inf dx / (1 + b e^x) = ln(1 + 1/b)): it vanishes as Delta_T -> 0
    but only like Delta_T ln(1/Delta_T), so halving Delta_T does not halve the
    deviation from the LIF rate.
  * Spike-initiation sharpness. In the phase plot (dV/dt against V) the
    slope of the upstroke, d(dV/dt)/dV = (g_L/C) (e^x - 1), is the "onset
    rapidness" of Naundorf, Wolf & Volgushev (2006). Measured where dV/dt
    first reaches a criterion Vdot_c, it is
        r(Vdot_c) = (g_L/C) (k_c + x_c - 1),  e^{x_c} - x_c = k_c,
        k_c = [C Vdot_c - I + g_L (V_T - E_L)] / (g_L Delta_T),
    with x_c on the upstroke branch, x_c = -k_c - W_{-1}(-e^{-k_c}). When the
    exponential term dominates (k_c >> 1) this is r ~ Vdot_c / Delta_T - g_L/C:
    the sharpness scales as 1/Delta_T.

Integrator: forward Euler with dt = 0.01 ms, spike time interpolated linearly
across the V_peak crossing (equivalently t_i + (V_peak - V_i)/F(V_i)), reset
to V_r and V held there for tau_ref. The exponent (V - V_T)/Delta_T is capped
at 60 before exponentiation; the cap only acts where dV/dt already exceeds
1e24 mV/ms (above V_peak for Delta_T >= 1.33 mV), so it cannot change a spike
time by more than one step. The rheobase sweep uses dt = 0.02 ms because the
dynamics it resolves (the slow passage near the saddle-node ghost) are on the
100 ms scale. RNG: a seeded generator only jitters the initial voltage of
the f-I runs by up to 1 mV so that no spike is aligned to the grid by
construction; seed printed.

Validation (ROADMAP note: "rheobase against the analytical value from the
nullcline; spike-initiation sharpness vs Delta_T"):
  (a) Rheobase found by a vectorised bisection sweep (does a neuron started at
      E_L fire within 4000 ms?) vs I_rh = g_L (V_T - E_L - Delta_T):
      relative error < 1%. Done for Delta_T = 3.48 mV and 1.0 mV.
  (b) Rest state at I = 0 and I = 0.06 uA/cm^2 vs the Lambert-W root of the
      nullcline: |dV| < 1e-4 mV, no spikes.
  (c) f-I curve from simulation vs the quadrature T(I) for 24 currents in
      [0.1, 3] uA/cm^2: max relative error < 1%.
  (d) LIF limit: at I = 1 uA/cm^2 the simulated excess ISI over the LIF closed
      form, T_EIF - T_LIF, for Delta_T = 1.6, 0.8, 0.4, 0.2 mV must shrink
      monotonically and lie within 10% of the asymptotic law
      tau_m (Delta_T/a) ln(1 + a/Delta_T) at every Delta_T; the same ratio for
      the quadrature ISI must approach 1 monotonically and be within 2% at
      Delta_T = 0.2 mV.
  (e) Onset rapidness at Vdot_c = 10 mV/ms measured from the simulated
      trajectory (secant of the finite-difference phase plot across the
      criterion crossing, one Euler step ~ 0.1 mV wide) vs the analytical
      slope (g_L/C)(e^x - 1) at the secant's midpoint voltage, for
      Delta_T in {0.5, 1, 2, 3.48, 6} mV: relative error < 2% at every Delta_T.
      The Lambert-W closed form r(Vdot_c) is reported alongside, and
      r * Delta_T / Vdot_c must be within 25% of 1 (the 1/Delta_T scaling).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Fourcaud-Trocme et al. 2003)
# ---------------------------------------------------------------------------
C = 1.0            # uF/cm^2
g_L = 0.05         # mS/cm^2  (tau_m = 20 ms)
E_L = -65.0        # mV
V_T = -59.9        # mV
Delta_T = 3.48     # mV
V_r = -68.0        # mV
tau_ref = 1.7      # ms
V_peak = 20.0      # mV, numerical cut-off
dt = 0.01          # ms, Euler step
tau_m = C / g_L

I_DEMO = 0.5                            # uA/cm^2, current for the trace / phase-plot panels
I_FI = np.linspace(0.1, 3.0, 24)        # currents for the f-I check
T_FI = 3000.0                           # ms per current (>= 7 ISIs at I = 0.1)
DELTA_T_SHARP = np.array([0.5, 1.0, 2.0, 3.48, 6.0])   # mV, sharpness sweep
DELTA_T_LIF = np.array([1.6, 0.8, 0.4, 0.2])            # mV, LIF-limit sweep
VDOT_C = 10.0                           # mV/ms, criterion for onset rapidness
I_REST = np.array([0.0, 0.06])          # sub-rheobase currents for the fixed-point check
T_SWEEP, DT_SWEEP = 4000.0, 0.02        # rheobase sweep: run length and step
X_CAP = 60.0                            # cap on (V - V_T)/Delta_T before exp

SEED = 0
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions and closed forms
# ---------------------------------------------------------------------------
def F(V, I, dT=Delta_T):
    """dV/dt of the EIF in mV/ms (eq. 2 of Fourcaud-Trocme et al. 2003)."""
    x = np.minimum((V - V_T) / dT, X_CAP)
    return (-g_L * (V - E_L) + g_L * dT * np.exp(x) + I) / C


def rheobase(dT=Delta_T):
    """Saddle-node current: F has its minimum at V_T, so I_rh makes F(V_T) = 0."""
    return g_L * (V_T - E_L - dT)


def fixed_points(I, dT=Delta_T):
    """Stable rest state and saddle from e^x - x = k via Lambert W; NaN above rheobase."""
    from scipy.special import lambertw
    k = (g_L * (V_T - E_L) - I) / (g_L * dT)
    if k < 1.0:
        return np.nan, np.nan
    arg = -np.exp(-k)
    x_stable = np.real(-k - lambertw(arg, 0))
    x_saddle = np.real(-k - lambertw(arg, -1))
    return V_T + dT * x_stable, V_T + dT * x_saddle


def isi_quadrature(I, dT=Delta_T):
    """Reference inter-spike interval T = tau_ref + int_{V_r}^{V_peak} C dV / F(V)."""
    from scipy.integrate import quad
    if I <= rheobase(dT):
        return np.inf
    val, _ = quad(lambda V: C / F(V, I, dT), V_r, V_peak, limit=400, epsabs=1e-12, epsrel=1e-10)
    return tau_ref + val


def rate_quadrature(I_values, dT=Delta_T):
    return np.array([1000.0 / isi_quadrature(I, dT) for I in I_values])


def rate_lif(I):
    """LIF closed form (src/leaky_integrate_and_fire.py) with threshold V_T, in Hz."""
    V_inf = E_L + I / g_L
    if V_inf <= V_T:
        return 0.0
    return 1000.0 / (tau_ref + tau_m * np.log((V_inf - V_r) / (V_inf - V_T)))


def phase_slope(V, dT):
    """d(dV/dt)/dV = (g_L/C)(e^x - 1) at voltage V, in 1/ms."""
    return (g_L / C) * (np.exp((V - V_T) / dT) - 1.0)


def onset_rapidness_closed_form(I, dT, vdot_c=VDOT_C):
    """Slope of the phase plot where dV/dt = vdot_c on the upstroke, in 1/ms."""
    from scipy.special import lambertw
    k_c = (C * vdot_c - I + g_L * (V_T - E_L)) / (g_L * dT)
    x_c = np.real(-k_c - lambertw(-np.exp(-k_c), -1))
    return (g_L / C) * (k_c + x_c - 1.0)


# ---------------------------------------------------------------------------
# Block 3: Simulation
# ---------------------------------------------------------------------------
def simulate(I, T, dT=Delta_T, dt=dt, V0=E_L, record=False):
    """Forward-Euler EIF for a vector of currents I (independent neurons).

    Returns spike-time lists (one per neuron) and, if record, the time grid and
    the voltage matrix (n_steps + 1, n_neurons)."""
    I = np.atleast_1d(np.asarray(I, dtype=float))
    n = I.size
    V = np.full(n, V0, dtype=float) if np.isscalar(V0) else np.array(V0, dtype=float)
    ref_until = np.full(n, -np.inf)
    spikes = [[] for _ in range(n)]
    n_steps = int(round(T / dt))
    Vrec = np.empty((n_steps + 1, n)) if record else None
    if record:
        Vrec[0] = V
    for i in range(n_steps):
        t = i * dt
        dV = F(V, I, dT)
        V_new = V + dt * dV
        refractory = t < ref_until
        V_new[refractory] = V_r
        fired = (V_new >= V_peak) & ~refractory
        if fired.any():
            idx = np.nonzero(fired)[0]
            t_sp = t + np.minimum((V_peak - V[idx]) / dV[idx], dt)
            for j, ts in zip(idx, t_sp):
                spikes[j].append(ts)
            ref_until[idx] = t_sp + tau_ref
            V_new[idx] = V_r
        V = V_new
        if record:
            Vrec[i + 1] = V
    if record:
        return spikes, np.arange(n_steps + 1) * dt, Vrec
    return spikes


def rate_from_spikes(spk):
    """Steady-state rate in Hz: 1 / mean ISI after discarding the first ISI."""
    spk = np.asarray(spk)
    if spk.size < 3:
        return 0.0
    return 1000.0 / np.mean(np.diff(spk[1:]))


def rheobase_sweep(dT=Delta_T, n_rounds=10, n_par=16):
    """Bisection on I: does a neuron started at E_L fire within T_SWEEP ms?"""
    I_pred = rheobase(dT)
    lo, hi = 0.5 * I_pred, 2.0 * I_pred
    for _ in range(n_rounds):
        I_grid = np.linspace(lo, hi, n_par + 2)[1:-1]
        spk = simulate(I_grid, T_SWEEP, dT, dt=DT_SWEEP)
        fired = np.array([len(s) > 0 for s in spk])
        if fired.all():            # transition lies between lo and the first grid point
            hi = I_grid[0]
        elif not fired.any():      # transition lies between the last grid point and hi
            lo = I_grid[-1]
        else:
            first = np.argmax(fired)
            assert fired[first:].all(), "firing is not monotone in I"
            lo = I_grid[first - 1] if first > 0 else lo
            hi = I_grid[first]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


def measure_onset_rapidness(t, V, vdot_c=VDOT_C):
    """Secant slope of the finite-difference phase plot across the first upstroke
    crossing of dV/dt = vdot_c, within the second inter-spike interval."""
    Vdot = np.diff(V) / (t[1] - t[0])
    Vmid = V[:-1]
    # take the second spike's upstroke (first is the transient from E_L)
    resets = np.nonzero(np.diff(V) < -20.0)[0]
    start = resets[0] + 1 if resets.size else 0
    seg = slice(start, resets[1] if resets.size > 1 else None)
    Vd, Vm = Vdot[seg], Vmid[seg]
    k = np.argmax(Vd >= vdot_c)
    slope = (Vd[k] - Vd[k - 1]) / (Vm[k] - Vm[k - 1])
    return slope, 0.5 * (Vm[k] + Vm[k - 1])


# ---------------------------------------------------------------------------
# Block 4: Validation
# ---------------------------------------------------------------------------
def validate():
    results = {}
    print(f"\nEIF, Fourcaud-Trocme et al. (2003) parameters: C = {C}, g_L = {g_L}, E_L = {E_L}, "
          f"V_T = {V_T}, Delta_T = {Delta_T}, V_r = {V_r}, tau_ref = {tau_ref}, V_peak = {V_peak}")
    from scipy.integrate import quad
    t_tail, _ = quad(lambda V: C / F(V, I_DEMO), 0.0, V_peak)
    print(f"Time spent between 0 mV and V_peak at I = {I_DEMO}: {t_tail:.2e} ms (cut-off irrelevant)")

    # (a) rheobase by sweep vs closed form, two Delta_T
    print("\n(a) Rheobase: bisection sweep vs I_rh = g_L (V_T - E_L - Delta_T)")
    results["rheo"] = []
    for dT in (Delta_T, 1.0):
        I_pred = rheobase(dT)
        I_sw, half = rheobase_sweep(dT)
        rel = abs(I_sw - I_pred) / I_pred
        print(f"    Delta_T = {dT:.2f} mV: sweep {I_sw:.6f} +/- {half:.1e}, closed form {I_pred:.6f} uA/cm^2, "
              f"rel. error {100*rel:.3f}%")
        assert rel < 0.01, f"rheobase mismatch {rel}"
        results["rheo"].append((dT, I_sw, half, I_pred, rel))

    # (b) rest state vs Lambert-W fixed point
    print("\n(b) Rest state vs Lambert-W root of the nullcline (500 ms from E_L)")
    spk, t, Vrec = simulate(I_REST, 500.0, record=True)
    results["rest"] = []
    for j, I in enumerate(I_REST):
        V_s, V_u = fixed_points(I)
        dev = abs(Vrec[-1, j] - V_s)
        print(f"    I = {I:.2f}: simulated {Vrec[-1, j]:.6f} mV, stable fixed point {V_s:.6f} mV "
              f"(saddle {V_u:.4f} mV), deviation {dev:.1e} mV, spikes {len(spk[j])}")
        assert dev < 1e-4 and len(spk[j]) == 0
        results["rest"].append((I, Vrec[-1, j], V_s, V_u, dev))

    # (c) f-I vs quadrature
    print("\n(c) f-I curve: Euler simulation vs quadrature of the ISI integral")
    V0 = E_L + rng.uniform(-1.0, 1.0, I_FI.size)
    spk = simulate(I_FI, T_FI, V0=V0)
    f_sim = np.array([rate_from_spikes(s) for s in spk])
    f_ref = rate_quadrature(I_FI)
    rel_c = np.abs(f_sim - f_ref) / f_ref
    for I, a, b, r in zip(I_FI[::4], f_sim[::4], f_ref[::4], rel_c[::4]):
        print(f"    I = {I:.3f}: sim {a:8.3f} Hz, quadrature {b:8.3f} Hz, rel. error {100*r:.3f}%")
    print(f"    max rel. error {100*rel_c.max():.3f}% (at I = {I_FI[np.argmax(rel_c)]:.3f}), "
          f"mean {100*rel_c.mean():.3f}%; rheobase {rheobase():.4f} uA/cm^2")
    assert rel_c.max() < 0.01, f"f-I mismatch {rel_c.max()}"
    results["fi"] = (f_sim, f_ref, rel_c)

    # (d) LIF limit as Delta_T -> 0
    I_L = 1.0
    f_L = rate_lif(I_L)
    T_L = 1000.0 / f_L
    a = E_L + I_L / g_L - V_T
    print(f"\n(d) LIF limit at I = {I_L} uA/cm^2: T_LIF (threshold V_T) = {T_L:.4f} ms = {f_L:.3f} Hz; "
          f"a = V_inf - V_T = {a:.2f} mV")
    ex_sim, ex_q, ex_as, devs = [], [], [], []
    for dT in DELTA_T_LIF:
        s = simulate(np.array([I_L]), 2000.0, dT)
        T_s = 1000.0 / rate_from_spikes(s[0])
        T_q = isi_quadrature(I_L, dT)
        asym = tau_m * (dT / a) * np.log(1.0 + a / dT)
        ex_sim.append(T_s - T_L); ex_q.append(T_q - T_L); ex_as.append(asym)
        devs.append(abs(1000.0 / T_s - f_L) / f_L)
        print(f"    Delta_T = {dT:.2f} mV: ISI sim {T_s:.4f} ms (quadrature {T_q:.4f}); excess over LIF "
              f"{T_s - T_L:.4f} ms, asymptotic tau_m (Delta_T/a) ln(1 + a/Delta_T) = {asym:.4f} ms, "
              f"ratio sim {(T_s - T_L)/asym:.4f}, quadrature {(T_q - T_L)/asym:.4f}; rate deviation {100*devs[-1]:.2f}%")
    ex_sim, ex_q, ex_as, devs = map(np.array, (ex_sim, ex_q, ex_as, devs))
    r_sim, r_q = ex_sim / ex_as, ex_q / ex_as
    print(f"    rate-deviation ratios on halving Delta_T: {np.round(devs[:-1] / devs[1:], 2)} "
          f"(not 2: the excess is O(Delta_T ln(1/Delta_T)))")
    assert np.all(np.diff(ex_sim) < 0), "excess ISI does not shrink with Delta_T"
    assert np.all(np.abs(r_sim - 1.0) < 0.10), f"simulated excess vs asymptotic law {r_sim}"
    assert np.all(np.diff(np.abs(r_q - 1.0)) < 0) and abs(r_q[-1] - 1.0) < 0.02, f"quadrature {r_q}"
    results["lif"] = (ex_sim, ex_q, ex_as, devs, f_L, a)

    # (e) onset rapidness vs Delta_T
    print(f"\n(e) Onset rapidness d(dV/dt)/dV at dV/dt = {VDOT_C} mV/ms, I = {I_DEMO}: simulation vs closed form")
    results["sharp"] = []
    results["traces"] = {}
    for dT in DELTA_T_SHARP:
        s, t, Vrec = simulate(np.array([I_DEMO]), 150.0, dT, record=True)
        r_meas, V_at = measure_onset_rapidness(t, Vrec[:, 0])
        r_mid = phase_slope(V_at, dT)
        r_cf = onset_rapidness_closed_form(I_DEMO, dT)
        rel = abs(r_meas - r_mid) / r_mid
        scaled = r_cf * dT / VDOT_C
        print(f"    Delta_T = {dT:.2f} mV: secant {r_meas:7.3f} /ms at V = {V_at:.2f} mV, analytical slope there "
              f"{r_mid:7.3f} /ms (rel. error {100*rel:.3f}%); r(Vdot_c) closed form {r_cf:7.3f} /ms, "
              f"r*Delta_T/Vdot_c = {scaled:.3f}")
        assert rel < 0.02, f"onset rapidness mismatch {rel}"
        assert abs(scaled - 1.0) < 0.25
        results["sharp"].append((dT, r_meas, r_cf, rel, V_at, r_mid))
        results["traces"][dT] = (t, Vrec[:, 0], s[0])
    print("\nAll checks passed.")
    return results


# ---------------------------------------------------------------------------
# Block 5: Figure
# ---------------------------------------------------------------------------
def plot(results):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # Panel 1: traces at I_DEMO for two Delta_T
    ax = axes[0, 0]
    for dT, col in ((Delta_T, "black"), (1.0, "steelblue")):
        t, V, s = results["traces"][dT]
        ax.plot(t, V, color=col, lw=1.0, label=f"Delta_T = {dT:g} mV, {len(s)} spikes in 150 ms")
        ax.vlines(s, V_r, V_peak, color=col, lw=0.8)   # spike drawn up to V_peak for display
    ax.axhline(V_T, color="gray", lw=0.8, ls="--", label=f"V_T = {V_T} mV")
    ax.axhline(V_r, color="gray", lw=0.8, ls=":", label=f"V_r = {V_r} mV")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("V (mV)")
    ax.set_ylim(-75, 25)
    ax.set_title(f"EIF at I = {I_DEMO} uA/cm^2 (Euler dt = {dt} ms)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    # Panel 2: phase plot for several Delta_T with the criterion and measured slopes
    ax = axes[0, 1]
    Vg = np.linspace(-70, -30, 800)
    cmap = plt.get_cmap("viridis")
    for i, (dT, r_meas, r_cf, rel, V_at, r_mid) in enumerate(results["sharp"]):
        col = cmap(i / (len(results["sharp"]) - 1))
        ax.plot(Vg, F(Vg, I_DEMO, dT), color=col, lw=1.5,
                label=f"Delta_T = {dT:g}: r = {r_meas:.2f} /ms (closed form {r_cf:.2f})")
        t, V, s = results["traces"][dT]
        Vdot = np.diff(V) / dt
        m = (Vdot > -5) & (V[:-1] < -30) & (t[:-1] > s[0] + tau_ref)
        ax.plot(V[:-1][m], Vdot[m], ".", color=col, ms=2)
        ax.plot(V_at, VDOT_C, "o", color="red", ms=5, mfc="none")
    ax.axhline(VDOT_C, color="red", lw=0.8, ls="--", label=f"criterion dV/dt = {VDOT_C:g} mV/ms")
    ax.axvline(V_T, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("dV/dt (mV/ms)")
    ax.set_ylim(-2, 25)
    ax.set_title("Phase plot: simulated points on the vector field; onset rapidness r at the criterion")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 3: sharpness vs Delta_T
    ax = axes[0, 2]
    dTs = np.array([s[0] for s in results["sharp"]])
    r_m = np.array([s[1] for s in results["sharp"]])
    r_c = np.array([s[2] for s in results["sharp"]])
    dT_dense = np.logspace(np.log10(0.3), np.log10(8), 100)
    ax.loglog(dT_dense, [onset_rapidness_closed_form(I_DEMO, d) for d in dT_dense], color="red", lw=1.5,
              label="closed form r(Vdot_c) via Lambert W")
    ax.loglog(dT_dense, VDOT_C / dT_dense, color="gray", lw=1.0, ls="--", label="Vdot_c / Delta_T")
    ax.loglog(dTs, r_m, "o", color="black", mfc="none", ms=7,
              label=f"secant from simulation (max rel. error vs slope at its midpoint "
                    f"{100*max(s[3] for s in results['sharp']):.2f}%)")
    ax.set_xlabel("Delta_T (mV)")
    ax.set_ylabel("onset rapidness r (1/ms)")
    ax.set_title(f"Spike-initiation sharpness vs Delta_T at dV/dt = {VDOT_C:g} mV/ms")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel 4: f-I simulation vs quadrature, LIF limit, rheobase
    ax = axes[1, 0]
    f_sim, f_ref, rel_c = results["fi"]
    I_dense = np.linspace(0.0, 3.0, 300)
    ax.plot(I_dense, rate_quadrature(I_dense), color="red", lw=1.5, label="quadrature of the ISI integral")
    ax.plot(I_FI, f_sim, "o", color="black", mfc="none", ms=6,
            label=f"Euler simulation (max rel. error {100*rel_c.max():.3f}%)")
    ax.plot(I_dense, [rate_lif(I) for I in I_dense], color="gray", lw=1.0, ls="--",
            label="LIF with threshold V_T (Delta_T -> 0 limit)")
    ax.axvline(rheobase(), color="steelblue", lw=1.0, ls=":",
               label=f"rheobase I_rh = {rheobase():.4f} (sweep {results['rheo'][0][1]:.4f})")
    ax.set_xlabel("I (uA/cm^2)")
    ax.set_ylabel("firing rate (Hz)")
    ax.set_title("f-I curve: class I onset at the saddle-node current")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Panel 5: fixed points vs I (nullcline) with rest states and sweep rheobase
    ax = axes[1, 1]
    for dT, col in ((Delta_T, "black"), (1.0, "steelblue")):
        I_fp = np.linspace(0, rheobase(dT), 200)
        fp = np.array([fixed_points(I, dT) for I in I_fp])
        ax.plot(I_fp, fp[:, 0], color=col, lw=1.5, label=f"stable rest, Delta_T = {dT:g}")
        ax.plot(I_fp, fp[:, 1], color=col, lw=1.5, ls="--", label=f"saddle, Delta_T = {dT:g}")
        rh = [r for r in results["rheo"] if r[0] == dT][0]
        ax.axvline(rh[3], color=col, lw=0.8, ls=":")
        ax.plot(rh[1], V_T, "x", color="red", ms=10, mew=2,
                label=f"sweep rheobase {rh[1]:.5f} vs {rh[3]:.5f} ({100*rh[4]:.3f}%)")
    for I, Vs, V_s, V_u, dev in results["rest"]:
        ax.plot(I, Vs, "+", color="red", ms=12, mew=2)
    ax.plot([], [], "+", color="red", ms=12, mew=2, label="simulated rest at I = 0, 0.06 (dev < 1e-4 mV)")
    ax.axhline(V_T, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("I (uA/cm^2)")
    ax.set_ylabel("fixed point V* (mV)")
    ax.set_title("Fixed points via Lambert W; saddle-node at V_T, I_rh = g_L(V_T - E_L - Delta_T)", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    # Panel 6: LIF limit, excess ISI vs the asymptotic law
    ax = axes[1, 2]
    ex_sim, ex_q, ex_as, devs, f_L, a = results["lif"]
    dT_dense = np.logspace(np.log10(0.1), np.log10(4), 100)
    T_L = 1000.0 / f_L
    ax.loglog(dT_dense, tau_m * (dT_dense / a) * np.log(1.0 + a / dT_dense), color="red", lw=1.5,
              label="asymptotic tau_m (Delta_T/a) ln(1 + a/Delta_T)")
    ax.loglog(dT_dense, [isi_quadrature(1.0, d) - T_L for d in dT_dense], color="gray", lw=1.0, ls="--",
              label="quadrature T_EIF - T_LIF")
    ax.loglog(DELTA_T_LIF, ex_sim, "o", color="black", mfc="none", ms=7,
              label=f"simulation T_EIF - T_LIF (ratio to asymptotic {np.round(ex_sim/ex_as, 3)})")
    ax.loglog(dT_dense, ex_as[-1] * dT_dense / DELTA_T_LIF[-1], color="steelblue", lw=0.8, ls=":",
              label="first order in Delta_T, for comparison")
    ax.set_xlabel("Delta_T (mV)")
    ax.set_ylabel("excess ISI over LIF (ms)")
    ax.set_title(f"LIF limit: T_EIF - T_LIF at I = 1.0 (T_LIF = {T_L:.2f} ms) ~ Delta_T ln(1/Delta_T)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "exponential_if.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
