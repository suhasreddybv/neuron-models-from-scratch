"""Adaptive exponential integrate-and-fire (AdEx) neuron.

Equations (Brette & Gerstner 2005, J. Neurophysiol. 94:3637; parameters and
firing patterns from Naud, Marcille, Clopath & Gerstner 2008, Biol. Cybern.
99:335, eqs. 1-3):

    C dV/dt = -g_L (V - E_L) + g_L Delta_T exp((V - V_T)/Delta_T) - w + I
    tau_w dw/dt = a (V - E_L) - w
    if V > V_peak:  V <- V_r,  w <- w + b

V is the membrane potential (mV), w an adaptation current (pA), I the
injected current (pA); C in pF, g_L and a in nS, t in ms, so dV/dt is in
mV/ms. The spike cut-off is V_peak = 0 mV as in Naud et al. (2008, sec. 2).
The model is the EIF of src/exponential_if.py plus one slow current: a sets
subthreshold adaptation (coupling of w to V), b spike-triggered adaptation.

Parameters: Naud et al. (2008) Table 1, the rows for Fig. 4. Six of the
eight rows are simulated here, those whose pattern label is unambiguous:

    pattern               fig   C    g_L  E_L  V_T  Delta_T  a    tau_w  b    V_r  I
    tonic spiking         4a   200   10   -70  -50  2        2     30    0   -58  500
    adapting              4b   200   12   -70  -50  2        2    300   60   -58  500
    initial burst         4c   130   18   -58  -50  2        4    150  120   -50  400
    regular bursting      4d   200   10   -58  -50  2        2    120  100   -46  210
    delayed accelerating  4e   200   12   -70  -50  2      -10    300    0   -58  300
    irregular spiking     4h   100   12   -60  -50  2      -11    130   30   -48  160

Stimulus: a current step of amplitude I at t = 50 ms after 50 ms at rest
(V = E_L, w = 0, as in the paper), held for 2000 ms.

Closed forms used for validation (derived from the equations; they are the
expressions printed in Naud et al. 2008, sec. 3.1). Fixed points satisfy
w* = a (V* - E_L) and I = (g_L + a)(V* - E_L) - g_L Delta_T e^{x*},
x* = (V* - V_T)/Delta_T. With tau_m = C/g_L the Jacobian is

    J = [[ (g_L/C)(e^x - 1),  -1/C   ],
         [  a/tau_w,          -1/tau_w ]].

  * Saddle-node (det J = 0, e^x = 1 + a/g_L), when a/g_L < tau_m/tau_w:
        I_SN = (g_L + a)[V_T - E_L - Delta_T + Delta_T ln(1 + a/g_L)].
  * Andronov-Hopf (tr J = 0, e^x = 1 + tau_m/tau_w), when a/g_L > tau_m/tau_w:
        I_AH = (g_L + a)[V_T - E_L + Delta_T ln(1 + tau_m/tau_w)]
               - g_L Delta_T (1 + tau_m/tau_w).

Adaptation index (Naud et al. 2008, sec. 2.3): with the first N = 20 spikes
and ISIs ISI_1 ... ISI_{N-1}, A = mean over i = k..N-1 of
(ISI_i - ISI_{i-1})/(ISI_i + ISI_{i-1}) with k = 4 (the early intervals are
discarded). A > 0.01 is classed as adapting (pyramidal cells have A ~ 0.015,
fast-spiking interneurons A ~ 0.005, values the paper quotes from Druckmann
et al. 2007); A < 0 means the rate accelerates. The paper's printed
normalisation 1/(N - k - 1) is replaced here by the number of terms, N - k.

Integrator: forward Euler with dt = 0.01 ms, spike time interpolated
linearly across the V_peak crossing, then the reset, after which the rest of
that step is integrated from (V_r, w + b) with one Euler sub-step (without it
every ISI is a multiple of dt and spike times drift by ~0.05 ms per spike). The exponent is capped
at 60 before exponentiation (only active far above V_peak). RNG: a seeded
generator draws the sign of the perturbation in the stability sweep; seed
printed.

Validation (ROADMAP note: "reproduce at least four firing patterns from Naud
et al. (2008) with their parameter table; spike-frequency adaptation index"):
  (a) Pattern criteria (chosen here, not from the paper) applied to each of
      the six regimes, using the paper's adaptation index A:
        tonic       A < 0.01 and ISI CV < 0.05 after the first 4 ISIs
        adapting    A > 0.01 and last ISI > 3 x first ISI
        init. burst first 2 ISIs < 10 ms, all later ISIs > 40 ms with CV < 0.05
        reg. burst  >= 5 bursts of >= 2 spikes (intra-burst ISI < 10 ms,
                    inter-burst > 50 ms), burst period CV < 0.05
        delayed acc. first-spike latency > 2 x final ISI and A < 0
        irregular   ISI CV > 0.3 after the first 500 ms of the step
  (b) Reference solver: spike times of the five non-chaotic regimes over the
      2000 ms step vs scipy.integrate.solve_ivp (DOP853, rtol = atol = 1e-10,
      terminal event at V_peak, exact reset): spike counts within one (a spike
      near the end of the window can fall either side of it, since Euler
      drifts by up to ~0.04 ms per spike), mean ISI within 1%, and the
      maximum spike-time deviation over the matched spikes must fall by a
      factor in [1.6, 2.4] when dt is halved to 0.005 ms (first order). The irregular regime is
      chaotic (Naud et al. 2008, Fig. 5); for it the script reports after how
      many spikes the Euler and reference trains first differ by > 1 ms.
  (c) Loss of rest-state stability, found by sweep, vs the closed forms above,
      within 1%. Saddle-node (4a, 4e): from the rest state at 0.9 I_pred, step
      to eight currents 0.3-12% above I_pred and record the first-spike
      latency, which near a saddle-node on an invariant circle follows the
      passage law T = T0 + K (I - I_SN)^(-1/2); I_SN is the fitted intercept
      (a bisection on "fires within a fixed window" fails here because the
      latency for 4e, tau_w = 300 ms, is 44 s at 0.3% above I_SN). The fit with
      a free exponent is reported as well. Andronov-Hopf (4b, 4d): bisection on
      whether a 1e-4 mV perturbation of the numerically found fixed point
      grows over 16 s.
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters (Naud et al. 2008, Table 1)
# ---------------------------------------------------------------------------
dt = 0.01          # ms, Euler step
V_PEAK = 0.0       # mV, spike cut-off (Naud et al. 2008)
X_CAP = 60.0       # cap on (V - V_T)/Delta_T before exp
T_PRE, T_STEP = 50.0, 2000.0    # ms at rest, ms of current step
N_A, K_A = 20, 4   # adaptation index: first N spikes, discard ISIs before k
SN_OFFSETS = np.array([0.003, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12])  # (I - I_pred)/I_pred, latency fit

KEYS = ("C", "g_L", "E_L", "V_T", "Delta_T", "a", "tau_w", "b", "V_r", "I")
REGIMES = {
    "tonic":       dict(fig="4a", p=(200, 10, -70, -50, 2,   2,  30,   0, -58, 500), name="tonic spiking"),
    "adapting":    dict(fig="4b", p=(200, 12, -70, -50, 2,   2, 300,  60, -58, 500), name="adapting"),
    "init_burst":  dict(fig="4c", p=(130, 18, -58, -50, 2,   4, 150, 120, -50, 400), name="initial burst"),
    "burst":       dict(fig="4d", p=(200, 10, -58, -50, 2,   2, 120, 100, -46, 210), name="regular bursting"),
    "delayed_acc": dict(fig="4e", p=(200, 12, -70, -50, 2, -10, 300,   0, -58, 300), name="delayed accelerating"),
    "irregular":   dict(fig="4h", p=(100, 12, -60, -50, 2, -11, 130,  30, -48, 160), name="irregular spiking"),
}
for r in REGIMES.values():
    r.update(dict(zip(KEYS, map(float, r["p"]))))
ORDER = list(REGIMES)

SEED = 0
rng = np.random.default_rng(SEED)
print(f"RNG seed: {SEED}")


# ---------------------------------------------------------------------------
# Block 2: Model functions and closed forms
# ---------------------------------------------------------------------------
def f_V(V, w, I, p):
    """dV/dt in mV/ms."""
    x = np.minimum((V - p["V_T"]) / p["Delta_T"], X_CAP)
    return (-p["g_L"] * (V - p["E_L"]) + p["g_L"] * p["Delta_T"] * np.exp(x) - w + I) / p["C"]


def f_w(V, w, p):
    """dw/dt in pA/ms."""
    return (p["a"] * (V - p["E_L"]) - w) / p["tau_w"]


def stack(names):
    """Parameter dict of arrays, one entry per regime name, for vectorised runs."""
    return {k: np.array([REGIMES[n][k] for n in names]) for k in KEYS}


def bifurcation(p):
    """Type and current of the rest-state bifurcation (Naud et al. 2008, sec. 3.1)."""
    gL, a, C, tw, EL, VT, dT = (p[k] for k in ("g_L", "a", "C", "tau_w", "E_L", "V_T", "Delta_T"))
    tau_m = C / gL
    if a / gL < tau_m / tw:
        return "saddle-node", (gL + a) * (VT - EL - dT + dT * np.log(1 + a / gL))
    return "Hopf", (gL + a) * (VT - EL + dT * np.log(1 + tau_m / tw)) - gL * dT * (1 + tau_m / tw)


def fixed_point(I, p):
    """Lower (rest) root of the steady-state I-V curve, by bracketing below its maximum."""
    gL, a, EL, VT, dT = (p[k] for k in ("g_L", "a", "E_L", "V_T", "Delta_T"))
    g = lambda V: (gL + a) * (V - EL) - gL * dT * np.exp((V - VT) / dT) - I
    V_max = VT + dT * np.log(1 + a / gL)          # maximum of the I-V curve
    V = brentq(g, EL - 60.0, V_max, xtol=1e-13)
    return V, a * (V - EL)


def adaptation_index(spk, N=N_A, k=K_A):
    """Naud et al. (2008) adaptation index from the first N spikes."""
    isi = np.diff(np.asarray(spk)[:N])
    if isi.size < k + 2:
        return np.nan
    i = np.arange(k, isi.size)          # 0-based isi[i] is ISI_{i+1}; ISI_k..ISI_{N-1}
    return np.mean((isi[i] - isi[i - 1]) / (isi[i] + isi[i - 1]))


# ---------------------------------------------------------------------------
# Block 3: Simulation
# ---------------------------------------------------------------------------
def simulate(names, dt=dt, record=False, T_pre=T_PRE, T_step=T_STEP):
    """Forward-Euler AdEx for several regimes at once; step current at T_pre."""
    p = stack(names)
    n = len(names)
    V, w = p["E_L"].copy(), np.zeros(n)
    spikes = [[] for _ in range(n)]
    n_steps = int(round((T_pre + T_step) / dt))
    if record:
        Vrec, wrec = np.empty((n_steps + 1, n)), np.empty((n_steps + 1, n))
        Vrec[0], wrec[0] = V, w
    for i in range(n_steps):
        t = i * dt
        I = p["I"] if t >= T_pre - 1e-9 else 0.0
        dV, dw = f_V(V, w, I, p), f_w(V, w, p)
        V_new, w = V + dt * dV, w + dt * dw
        fired = V_new >= V_PEAK
        if fired.any():
            idx = np.nonzero(fired)[0]
            frac = np.minimum((V_PEAK - V[idx]) / dV[idx], dt)
            for j, f in zip(idx, frac):
                spikes[j].append(t + f - T_pre)
            # reset, then integrate the rest of the step from (V_r, w + b) so that
            # spike times are not quantised to the grid
            w[idx] += p["b"][idx]
            pi = {k: v[idx] for k, v in p.items()}
            I_i = I[idx] if np.ndim(I) else I
            V_new[idx] = pi["V_r"] + (dt - frac) * f_V(pi["V_r"], w[idx], I_i, pi)
        V = V_new
        if record:
            Vrec[i + 1], wrec[i + 1] = np.minimum(V, V_PEAK), w
    spikes = [np.array(s) for s in spikes]
    if record:
        return spikes, np.arange(n_steps + 1) * dt - T_pre, Vrec, wrec
    return spikes


def simulate_reference(name, T_step=T_STEP):
    """solve_ivp (DOP853, rtol = atol = 1e-10) with a terminal event at V_peak and
    an exact reset; starts at the step onset from rest (V = E_L, w = 0)."""
    p = REGIMES[name]
    rhs = lambda t, y: [f_V(y[0], y[1], p["I"], p), f_w(y[0], y[1], p)]
    hit = lambda t, y: y[0] - V_PEAK
    hit.terminal, hit.direction = True, 1
    t0, y0, spikes = 0.0, [p["E_L"], 0.0], []
    while t0 < T_step:
        sol = solve_ivp(rhs, (t0, T_step), y0, method="DOP853", rtol=1e-10, atol=1e-10, events=hit)
        if sol.t_events[0].size == 0:
            break
        t0 = sol.t_events[0][0]
        spikes.append(t0)
        y0 = [p["V_r"], sol.y_events[0][0][1] + p["b"]]
    return np.array(spikes)


def first_spike_latency(p, I_grid, I_start, dt=0.02, T_max=80000.0):
    """Saddle-node test: start at the rest state for I_start < I_SN, step to each
    I in I_grid, and return the time to the first spike (ms)."""
    n = I_grid.size
    V0, w0 = fixed_point(I_start, p)
    V, w = np.full(n, V0), np.full(n, w0)
    first = np.full(n, np.nan)
    for i in range(int(round(T_max / dt))):
        V, w = V + dt * f_V(V, w, I_grid, p), w + dt * f_w(V, w, p)
        s = V >= V_PEAK
        if s.any():
            first[s & np.isnan(first)] = (i + 1) * dt
            V[s] = p["V_r"]
            if not np.isnan(first).any():
                break
    return first


def fit_saddle_node(I_grid, T):
    """Fit the saddle-node passage law T = T0 + K (I - I_SN)^(-1/2); also with a free exponent."""
    from scipy.optimize import curve_fit
    law = lambda I, T0, K, I_sn, g=0.5: T0 + K * np.clip(I - I_sn, 1e-12, None) ** (-g)
    p0 = [0.0, T[0] * np.sqrt(0.01 * I_grid[0]), 0.99 * I_grid[0]]
    p3, _ = curve_fit(law, I_grid, T, p0=p0, maxfev=20000)
    p4, _ = curve_fit(law, I_grid, T, p0=list(p3) + [0.5], maxfev=20000)
    return p3, p4


def unstable_rest(p, I_grid, dt=0.05, T=16000.0, eps=1e-4):
    """Hopf sweep: does a small perturbation of the fixed point grow? Compares the
    largest |V - V*| over [T/2, 3T/4] with that over [3T/4, T]; a spike counts as growth."""
    n = I_grid.size
    fp = np.array([fixed_point(I, p) for I in I_grid])
    V = fp[:, 0] + eps * rng.choice([-1.0, 1.0], n)
    w = fp[:, 1].copy()
    n_steps = int(round(T / dt))
    m1, m2, fired = np.zeros(n), np.zeros(n), np.zeros(n, bool)
    for i in range(n_steps):
        V, w = V + dt * f_V(V, w, I_grid, p), w + dt * f_w(V, w, p)
        s = V >= V_PEAK
        if s.any():
            fired |= s
            V[s] = p["V_r"]
            w[s] += p["b"]
        if i >= n_steps // 2:
            dev = np.abs(V - fp[:, 0])
            if i < 3 * n_steps // 4:
                m1 = np.maximum(m1, dev)
            else:
                m2 = np.maximum(m2, dev)
    return fired | (m2 > m1)


def bisect(test, p, lo, hi, n_rounds=6, n_par=16):
    """Vectorised bisection on I for the first current where test() is True."""
    for _ in range(n_rounds):
        I_grid = np.linspace(lo, hi, n_par + 2)[1:-1]
        res = test(p, I_grid)
        if res.all():
            hi = I_grid[0]
        elif not res.any():
            lo = I_grid[-1]
        else:
            first = np.argmax(res)
            assert res[first:].all(), "transition is not monotone in I"
            lo = I_grid[first - 1] if first > 0 else lo
            hi = I_grid[first]
    return 0.5 * (lo + hi), 0.5 * (hi - lo)


# ---------------------------------------------------------------------------
# Block 4: Validation
# ---------------------------------------------------------------------------
def bursts(spk, thr=10.0):
    """Split a spike train into bursts (groups with ISI < thr ms)."""
    groups, cur = [], [spk[0]]
    for a_, b_ in zip(spk[:-1], spk[1:]):
        if b_ - a_ < thr:
            cur.append(b_)
        else:
            groups.append(cur); cur = [b_]
    groups.append(cur)
    return groups


def classify(name, spk):
    """Pattern criteria (see module docstring). Returns (passed, description)."""
    isi = np.diff(spk)
    A = adaptation_index(spk)
    cv = lambda x: np.std(x) / np.mean(x)
    if name == "tonic":
        c = cv(isi[4:])
        return A < 0.01 and c < 0.05, f"A = {A:.4f}, ISI CV (after 4) = {c:.4f}, steady ISI {isi[-1]:.2f} ms"
    if name == "adapting":
        r = isi[-1] / isi[0]
        return A > 0.01 and r > 3, f"A = {A:.4f}, first ISI {isi[0]:.2f} ms, last {isi[-1]:.2f} ms (ratio {r:.2f})"
    if name == "init_burst":
        late = isi[2:]
        ok = np.all(isi[:2] < 10) and np.all(late > 40) and cv(late[2:]) < 0.05
        return ok, (f"A = {A:.4f}, first ISIs {isi[0]:.2f}, {isi[1]:.2f} ms, then {late.min():.1f}-{late.max():.1f} ms "
                    f"(CV {cv(late[2:]):.4f})")
    if name == "burst":
        g = bursts(spk)[1:]                 # the first group is the onset transient
        sizes = np.array([len(x) for x in g])
        starts = np.array([x[0] for x in g])
        period = np.diff(starts)
        intra = np.concatenate([np.diff(x) for x in g])
        gaps = np.array([g[i + 1][0] - g[i][-1] for i in range(len(g) - 1)])
        ok = (sizes >= 2).sum() >= 5 and np.all(sizes[:-1] >= 2) and intra.max() < 10 and gaps.min() > 50 \
            and cv(period) < 0.05
        return ok, (f"A = {A:.4f}, {len(g)} bursts of {sizes.min()}-{sizes.max()} spikes, intra-burst ISI "
                    f"{intra.min():.2f}-{intra.max():.2f} ms, burst period {period.mean():.1f} ms (CV {cv(period):.4f})")
    if name == "delayed_acc":
        lat = spk[0]
        return lat > 2 * isi[-1] and A < 0, (f"A = {A:.4f}, first-spike latency {lat:.1f} ms, first ISI "
                                             f"{isi[0]:.2f} ms, final ISI {isi[-1]:.2f} ms")
    if name == "irregular":
        late = isi[spk[1:] > 500.0]
        return cv(late) > 0.3, f"A = {A:.4f}, ISI CV after 500 ms = {cv(late):.3f}, ISI range {late.min():.1f}-{late.max():.1f} ms"
    raise KeyError(name)


def validate():
    res = {}
    print("\nAdEx, Naud et al. (2008) Table 1; step at t = 0 after "
          f"{T_PRE:g} ms at rest, {T_STEP:g} ms long; Euler dt = {dt} ms, V_peak = {V_PEAK} mV")

    # (a) patterns and adaptation index
    print("\n(a) Firing patterns and adaptation index A (k = 4, N = 20)")
    spk, t, Vrec, wrec = simulate(ORDER, record=True)
    res["traces"] = (t, Vrec, wrec, spk)
    res["A"] = {}
    for j, n in enumerate(ORDER):
        ok, desc = classify(n, spk[j])
        rate = spk[j].size / (T_STEP / 1000.0)
        print(f"    {REGIMES[n]['fig']} {REGIMES[n]['name']:<21s} {spk[j].size:4d} spikes ({rate:6.1f} Hz mean): "
              f"{desc} -> {'pass' if ok else 'FAIL'}")
        assert ok, f"pattern criterion failed for {n}"
        res["A"][n] = adaptation_index(spk[j])

    # (b) reference solver and convergence
    print("\n(b) Euler vs solve_ivp (DOP853, rtol = atol = 1e-10) over the 2000 ms step")
    spk_half = simulate(ORDER, dt=dt / 2)
    res["ref"] = {}
    for j, n in enumerate(ORDER):
        ref = simulate_reference(n)
        s1, s2 = spk[j], spk_half[j]
        if n == "irregular":
            m = min(s1.size, ref.size)
            k1 = np.argmax(np.abs(s1[:m] - ref[:m]) > 1.0)
            m2 = min(s2.size, ref.size)
            k2 = np.argmax(np.abs(s2[:m2] - ref[:m2]) > 1.0)
            print(f"    {REGIMES[n]['fig']} {n:<12s} chaotic: {s1.size} spikes (Euler dt = {dt}), {s2.size} "
                  f"(dt = {dt/2}), {ref.size} (solve_ivp); trains first differ by > 1 ms at spike {k1 + 1} "
                  f"(dt = {dt}) and {k2 + 1} (dt = {dt/2})")
            res["ref"][n] = (ref, k1, k2)
            continue
        m = min(s1.size, s2.size, ref.size)
        assert max(abs(s1.size - ref.size), abs(s2.size - ref.size)) <= 1, f"{n}: spike counts {s1.size}, {s2.size}, {ref.size}"
        isi_rel = abs(np.mean(np.diff(s1)) - np.mean(np.diff(ref))) / np.mean(np.diff(ref))
        e1, e2 = np.max(np.abs(s1[:m] - ref[:m])), np.max(np.abs(s2[:m] - ref[:m]))
        ratio = e1 / e2
        print(f"    {REGIMES[n]['fig']} {n:<12s} spikes {s1.size} (dt = {dt}), {s2.size} (dt = {dt/2}), {ref.size} "
              f"(solve_ivp; last at {ref[-1]:.1f} ms); mean ISI rel. diff {100*isi_rel:.3f}%; max |dt_spike| over the "
              f"first {m}: {e1:.4f} ms (dt = {dt}), {e2:.4f} ms (dt = {dt/2}), ratio {ratio:.2f}")
        assert isi_rel < 0.01, f"{n}: mean ISI mismatch {isi_rel}"
        assert 1.6 <= ratio <= 2.4, f"{n}: convergence ratio {ratio}"
        res["ref"][n] = (ref, e1, e2, ratio, s1.size, isi_rel)

    # (c) bifurcation current by sweep vs closed form
    print("\n(c) Loss of rest-state stability: sweep vs closed form")
    res["bif"] = {}
    for n in ("tonic", "delayed_acc", "adapting", "burst"):
        p = REGIMES[n]
        kind, I_pred = bifurcation(p)
        tau_m = p["C"] / p["g_L"]
        head = (f"    {p['fig']} {n:<12s} a/g_L = {p['a']/p['g_L']:+.3f}, tau_m/tau_w = {tau_m/p['tau_w']:.3f} "
                f"-> {kind}, closed form {I_pred:.5f} pA:")
        if kind == "saddle-node":
            I_grid = I_pred * (1.0 + SN_OFFSETS)
            T = first_spike_latency(p, I_grid, 0.9 * I_pred)
            assert not np.isnan(T).any(), f"{n}: no spike within T_max"
            p3, p4 = fit_saddle_node(I_grid, T)
            I_est, extra = p3[2], (I_grid, T, p3, p4)
            print(f"{head} latency fit T0 + K (I - I_SN)^-1/2 over I = I_pred x (1 + {SN_OFFSETS[0]:g}..{SN_OFFSETS[-1]:g}) "
                  f"(latencies {T[0]:.1f}..{T[-1]:.1f} ms) gives I_SN = {I_est:.3f} pA; free exponent "
                  f"{p4[3]:.3f} with I_SN = {p4[2]:.3f} pA")
        else:
            # upper end of the bracket below the I-V maximum, where the fixed point disappears
            I_max = (p["g_L"] + p["a"]) * (p["V_T"] - p["E_L"] - p["Delta_T"]
                                           + p["Delta_T"] * np.log(1 + p["a"] / p["g_L"]))
            I_est, half = bisect(unstable_rest, p, 0.8 * I_pred, min(1.2 * I_pred, I_max - 1e-6))
            extra = half
            print(f"{head} perturbation-growth sweep {I_est:.5f} +/- {half:.1e} pA")
        rel = abs(I_est - I_pred) / I_pred
        print(f"        rel. error {100*rel:.2e}%")
        assert rel < 0.01, f"{n}: bifurcation current mismatch {rel}"
        res["bif"][n] = (kind, I_est, extra, I_pred, rel)
    print("\nAll checks passed.")
    return res


# ---------------------------------------------------------------------------
# Block 5: Figure
# ---------------------------------------------------------------------------
def plot(res):
    t, Vrec, wrec, spk = res["traces"]
    fig = plt.figure(figsize=(19, 13))
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 1, 1.2])
    for j, n in enumerate(ORDER):
        ax = fig.add_subplot(gs[j // 2, j % 2])
        p = REGIMES[n]
        m = t <= 500.0
        ax.plot(t[m], Vrec[m, j], color="black", lw=0.7)
        sp = spk[j][spk[j] <= 500.0]
        ax.vlines(sp, p["V_r"], V_PEAK, color="black", lw=0.7)   # spikes drawn to V_peak for display
        ref = res["ref"][n][0]
        r = ref[ref <= 500.0]
        ax.plot(r, np.full(r.size, 5.0), "v", color="red", ms=4, label="solve_ivp spike times")
        A = res["A"][n]
        ax.set_title(f"Fig. {p['fig']} {p['name']} (I = {p['I']:g} pA): {spk[j].size} spikes / 2 s, A = {A:.3f}",
                     fontsize=10)
        ax.set_xlim(-50, 500)
        ax.set_ylim(-80, 10)
        ax.set_ylabel("V (mV)")
        if j >= 4:
            ax.set_xlabel("time from step onset (ms)")
        if j == 0:
            ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)

    # ISI sequences (right column, top)
    ax = fig.add_subplot(gs[0:2, 2])
    cmap = plt.get_cmap("tab10")
    for j, n in enumerate(ORDER):
        isi = np.diff(spk[j])
        ax.semilogy(np.arange(1, isi.size + 1)[:60], isi[:60], ".-", color=cmap(j), lw=0.8, ms=4,
                    label=f"{REGIMES[n]['fig']} {REGIMES[n]['name']} (A = {res['A'][n]:.3f})")
    ax.axvspan(K_A, N_A - 1, color="gray", alpha=0.12, label=f"ISIs used for A (k = {K_A}, N = {N_A})")
    ax.set_xlabel("ISI number")
    ax.set_ylabel("ISI (ms)")
    ax.set_title("Inter-spike intervals; adaptation index A > 0.01: adapting, A < 0: accelerating", fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(alpha=0.3, which="both")

    # Euler vs reference convergence
    ax = fig.add_subplot(gs[2, 2])
    names = [n for n in ORDER if n != "irregular"]
    e1 = [res["ref"][n][1] for n in names]
    e2 = [res["ref"][n][2] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, e1, 0.4, label=f"Euler dt = {dt} ms", color="gray")
    ax.bar(x + 0.2, e2, 0.4, label=f"Euler dt = {dt/2} ms", color="steelblue")
    for xi, n in zip(x, names):
        ax.text(xi, res["ref"][n][1] * 1.05, f"x{res['ref'][n][3]:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([REGIMES[n]["fig"] for n in names])
    ax.set_ylim(0, 1.15 * max(e1))
    ax.set_ylabel("max spike-time error vs solve_ivp (ms)")
    ax.set_title("Spike times vs solve_ivp: mean ISI within 0.5%, error halves with dt", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")

    # Phase plane of the bursting regime with nullclines
    ax = fig.add_subplot(gs[3, 0])
    j = ORDER.index("burst")
    p = REGIMES["burst"]
    Vg = np.linspace(-62, -42, 400)
    ax.plot(Vg, -p["g_L"] * (Vg - p["E_L"]) + p["g_L"] * p["Delta_T"] * np.exp((Vg - p["V_T"]) / p["Delta_T"]) + p["I"],
            color="red", lw=1.2, label="V-nullcline at I = 210 pA")
    ax.plot(Vg, p["a"] * (Vg - p["E_L"]), color="blue", lw=1.2, label="w-nullcline")
    m = (t > 300) & (Vrec[:, j] < -42)
    ax.plot(Vrec[m, j], wrec[m, j], ".", color="black", ms=0.6, label="trajectory (resets join at V_r)")
    ax.axvline(p["V_r"], color="gray", ls=":", lw=0.8)
    ax.set_xlabel("V (mV)")
    ax.set_ylabel("w (pA)")
    ax.set_ylim(0, 400)
    ax.set_title("Fig. 4d regular bursting: phase plane", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    # Bifurcation currents
    ax = fig.add_subplot(gs[3, 1])
    names = list(res["bif"])
    x = np.arange(len(names))
    pred = [res["bif"][n][3] for n in names]
    sw = [res["bif"][n][1] for n in names]
    ax.bar(x - 0.2, pred, 0.4, color="red", alpha=0.7, label="closed form (I_SN or I_AH)")
    ax.bar(x + 0.2, sw, 0.4, color="black", alpha=0.7, label="sweep")
    for xi, n in zip(x, names):
        kind, I_sw, _, I_pred, rel = res["bif"][n]
        ax.text(xi, max(I_sw, I_pred) * 1.03, f"{kind}\n{100*rel:.2f}%", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{REGIMES[n]['fig']}" for n in names])
    ax.set_ylabel("current (pA)")
    ax.set_ylim(0, 1.3 * max(pred))
    ax.set_title("Rest-state bifurcation current: sweep vs closed form", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    # Irregular: Euler vs reference divergence
    ax = fig.add_subplot(gs[3, 2])
    j = ORDER.index("irregular")
    ref, k1, k2 = res["ref"]["irregular"]
    s1 = spk[j]
    m = min(s1.size, ref.size)
    ax.semilogy(np.arange(1, m + 1), np.abs(s1[:m] - ref[:m]) + 1e-6, ".-", color="black", lw=0.8, ms=3)
    ax.axhline(1.0, color="red", ls="--", lw=0.8, label="1 ms")
    ax.set_xlabel("spike number")
    ax.set_ylabel("|t_Euler - t_solve_ivp| (ms)")
    ax.set_title(f"Fig. 4h irregular: Euler and solve_ivp trains part after spike {k1 + 1} (dt = {dt})", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "adex.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time as _time
    t0 = _time.time()
    results = validate()
    plot(results)
    print(f"Done in {_time.time() - t0:.1f} s")
