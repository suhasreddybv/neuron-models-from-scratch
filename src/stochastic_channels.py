"""Stochastic ion channels in the Hodgkin-Huxley model: Markov-chain Na/K
channels and the Fox-Lu diffusion approximation.

Membrane (Hodgkin & Huxley 1952, modern convention, the parameters of
src/hodgkin_huxley.py) with the deterministic gating products replaced by
the open fraction of a finite channel population:

    C_m dV/dt = I - g_Na (O_Na / N_Na)(V - E_Na) - g_K (O_K / N_K)(V - E_K)
                  - g_L (V - E_L)

Markov chain (Skaugen & Wallo:e 1979; Chow & White 1996). Each Na channel
has three m subunits and one h subunit and occupies one of 8 states
(i open m subunits, j open h subunit), open only in (3, 1). Each K channel
has four n subunits and 5 states, open only in state 4. Subunits flip
independently with the HH rates:

    x closed -> open at alpha_x(V),  open -> closed at beta_x(V)

    alpha_m = 0.1 (V + 40) / (1 - exp(-(V + 40)/10)),  beta_m = 4 exp(-(V + 65)/18)
    alpha_h = 0.07 exp(-(V + 65)/20),                  beta_h = 1 / (1 + exp(-(V + 35)/10))
    alpha_n = 0.01 (V + 55) / (1 - exp(-(V + 55)/10)), beta_n = 0.125 exp(-(V + 65)/80)

so a channel with i of s subunits open goes to the state with i +- 1 at
rate (s - i) alpha or i beta. Channel numbers come from the densities used by
Fox (1997) and Goldwyn & Shea-Brown (2011): N_Na = 60 A, N_K = 18 A for a
membrane patch of area A in um^2.

Fox-Lu diffusion approximation (Fox & Lu 1994; Fox 1997, the "subunit"
form): the gating variables obey Langevin equations

    dx = [alpha_x (1 - x) - beta_x x] dt + sqrt([alpha_x (1 - x) + beta_x x] / N_x) dW_x

with N_m = N_h = N_Na and N_n = N_K, x clipped to [0, 1], and O/N
replaced by m^3 h and n^4.

Integrators (dt = 0.01 ms). V: forward Euler. Markov chain: binomial /
multinomial update that is exact for V frozen over the step: a subunit
that is closed is open after dt with probability
a = x_inf (1 - exp(-dt/tau_x)) and an open one stays open with probability
b = x_inf + (1 - x_inf) exp(-dt/tau_x), so the channel-state transition
matrix is the distribution of Bin(i, b) + Bin(s - i, a) (a Kronecker
product of the m and h matrices for Na), and the channels in each state are
redistributed with one multinomial draw. The large-N limit of this scheme is
HH with exponential-Euler gating, which is run as the "same scheme,
N = infinity" deterministic control. Fox-Lu: Euler-Maruyama. The RNG is
NumPy's PCG64 seeded with SEED (printed); each experiment gets its own
spawned stream.

Validation:

1. Voltage clamp, stationary statistics (closed form). With V held at
   -60, -45, -30 mV and A = 100 um^2 (6000 Na, 1800 K channels), starting
   from all channels closed and running 100 ms at dt = 0.1 ms (the update is
   exact for frozen V at any dt), 1000 independent patches per voltage.
   Since the channels are independent, O_Na ~ Bin(N_Na, m_inf^3 h_inf) and
   O_K ~ Bin(N_K, n_inf^4) exactly. Asserted: sample mean and variance of
   the open counts within 4 standard errors of N p and N p (1 - p).
2. Voltage step, relaxation (closed form). Step from -65 to 0 mV, 2000
   patches, dt = 0.01 ms; the ensemble mean open fraction must follow
   n(t)^4 and m(t)^3 h(t) with x(t) = x_inf + (x_0 - x_inf) exp(-t/tau_x).
   Asserted: max |z| < 5 over 100 sample times, z using the exact
   binomial standard error at each t.
3. Convergence to deterministic HH (reference solver + limit). Constant
   20 uA/cm^2 from rest for 1000 ms (well above the Hopf current 9.78, so
   the limit cycle is the only attractor and is strongly attracting),
   A = 3 ... 10^5 um^2, 16 patches each; plus 400 patches x 5 ms per area
   for the first-spike time. Asserted: at A = 10^5 um^2 the Markov mean ISI
   is within 1% of the solve_ivp (DOP853, tol 1e-10) ISI of deterministic
   HH; the ISI CV and the standard deviation of the first spike time fall
   as N^(-1/2) (weak-noise limit, where phase diffusion and jitter are
   linear in the noise amplitude): log-log slope against N_Na over
   A >= 1000 um^2 within -0.5 +- 0.15.
4. Spontaneous firing at I = 0 vs area, Markov vs Fox-Lu, 16 patches x
   1000 ms per area. Reported; asserted only that the Markov rate falls
   from the smallest to the largest area.

scipy is used only for validation (reference solver, binomial pmf in the figure).
"""

from pathlib import Path
from math import comb
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.stats import binom

FIGURES = Path(__file__).resolve().parent.parent / "figures"

# ---------------------------------------------------------------- parameters
C_m = 1.0                            # uF/cm^2
g_Na, g_K, g_L = 120.0, 36.0, 0.3    # mS/cm^2
E_Na, E_K, E_L = 50.0, -77.0, -54.4  # mV
RHO_NA, RHO_K = 60.0, 18.0           # channels per um^2
DT = 0.01                            # ms
SEED = 20261029

CLAMP_V = [-60.0, -45.0, -30.0]      # mV
CLAMP_AREA, CLAMP_M = 100.0, 1000    # um^2, patches per voltage
CLAMP_DT, CLAMP_T = 0.1, 100.0       # ms
CLAMP_Z = 4.0

STEP_V0, STEP_V1 = -65.0, 0.0        # mV
STEP_M, STEP_T = 2000, 10.0          # patches, ms
STEP_Z = 5.0

I_DRIVE = 20.0                       # uA/cm^2
CONV_AREAS = np.array([3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 1e4, 3e4, 1e5])  # um^2
CONV_TRIALS, CONV_T, CONV_T0 = 16, 1000.0, 100.0   # patches, ms, ISIs measured after
JIT_TRIALS, JIT_T = 400, 5.0         # patches, ms (first-spike time)
ISI_TOL = 0.01
SLOPE_TOL = 0.15
SLOPE_MIN_AREA = 1000.0

SPONT_AREAS = np.array([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0])  # um^2
SPONT_TRIALS, SPONT_T = 16, 1000.0

V_ARM = -30.0                        # mV, spike detector re-arms below this


# ------------------------------------------------------------------- model
def rates(V):
    V = np.asarray(V, dtype=float)
    u = V + 40.0
    us = np.where(np.abs(u) < 1e-7, 1e-7, u)
    am = 0.1 * us / (1.0 - np.exp(-us / 10.0))
    bm = 4.0 * np.exp(-(V + 65.0) / 18.0)
    ah = 0.07 * np.exp(-(V + 65.0) / 20.0)
    bh = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
    w = V + 55.0
    ws = np.where(np.abs(w) < 1e-7, 1e-7, w)
    an = 0.01 * ws / (1.0 - np.exp(-ws / 10.0))
    bn = 0.125 * np.exp(-(V + 65.0) / 80.0)
    return am, bm, ah, bh, an, bn


def steady(V):
    am, bm, ah, bh, an, bn = rates(V)
    return am / (am + bm), ah / (ah + bh), an / (an + bn)


def subunit_probs(al, be, dt):
    """Closed->open probability a and open->open probability b over dt."""
    k = al + be
    xi = al / k
    e = np.exp(-k * dt)
    return xi * (1.0 - e), xi + (1.0 - xi) * e


def subunit_matrix(a, b, s):
    """T[c, i, j] = P(j of s subunits open after dt | i open now)."""
    C = a.shape[0]
    T = np.zeros((C, s + 1, s + 1))
    for i in range(s + 1):
        p = np.zeros((C, s + 1))
        p[:, 0] = 1.0
        for q, r in [(b, i), (a, s - i)]:
            for _ in range(r):
                new = p * (1.0 - q)[:, None]
                new[:, 1:] += p[:, :-1] * q[:, None]
                p = new
        T[:, i, :] = p
    return T


def binom_dist(x, s):
    """Stationary distribution of the number of open subunits, shape (C, s+1)."""
    x = np.atleast_1d(x)
    return np.stack([comb(s, i) * x ** i * (1.0 - x) ** (s - i) for i in range(s + 1)], axis=1)


def kron_na(Am, Ah):
    """Na state index 2 i + j for (i open m, j open h)."""
    if Am.ndim == 2:                  # distributions (C, 4) x (C, 2)
        return (Am[:, :, None] * Ah[:, None, :]).reshape(Am.shape[0], 8)
    C = Am.shape[0]
    return np.einsum("cik,cjl->cijkl", Am, Ah).reshape(C, 8, 8)


def transition_matrices(V, dt):
    am, bm, ah, bh, an, bn = rates(V)
    Tm = subunit_matrix(*subunit_probs(am, bm, dt), 3)
    Th = subunit_matrix(*subunit_probs(ah, bh, dt), 1)
    Tn = subunit_matrix(*subunit_probs(an, bn, dt), 4)
    PNa = kron_na(Tm, Th)
    PNa = np.clip(PNa, 0.0, None)
    PNa /= PNa.sum(axis=2, keepdims=True)
    Tn = np.clip(Tn, 0.0, None)
    Tn /= Tn.sum(axis=2, keepdims=True)
    return PNa, Tn


def stationary_counts(V, NNa, NK, rng):
    m, h, n = steady(V)
    piNa = kron_na(binom_dist(m, 3), binom_dist(h, 1))
    piK = binom_dist(n, 4)
    piNa /= piNa.sum(axis=1, keepdims=True)
    piK /= piK.sum(axis=1, keepdims=True)
    return rng.multinomial(NNa, piNa), rng.multinomial(NK, piK)


def markov_step(cNa, cK, PNa, PK, rng):
    return rng.multinomial(cNa, PNa).sum(axis=1), rng.multinomial(cK, PK).sum(axis=1)


def resting_potential(I=0.0):
    """Deterministic HH fixed point at current I (bisection on the I-V curve)."""
    def f(V):
        m, h, n = steady(V)
        return I - g_Na * m ** 3 * h * (V - E_Na) - g_K * n ** 4 * (V - E_K) - g_L * (V - E_L)
    lo, hi = -70.0, -60.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ------------------------------------------------------------- simulation
class _Spikes:
    """Spike detector: upward 0 mV crossings, re-armed below V_ARM."""

    def __init__(self, C):
        self.armed = np.ones(C, dtype=bool)
        self.cols, self.times = [], []

    def update(self, t, Vold, V):
        hit = self.armed & (V >= 0.0) & (Vold < 0.0)
        if hit.any():
            idx = np.nonzero(hit)[0]
            self.cols.append(idx)
            self.times.append(t + DT * (-Vold[idx]) / (V[idx] - Vold[idx]))
            self.armed[idx] = False
        self.armed |= V < V_ARM

    def per_column(self, C):
        if not self.cols:
            return [np.array([]) for _ in range(C)]
        c = np.concatenate(self.cols)
        t = np.concatenate(self.times)
        return [np.sort(t[c == j]) for j in range(C)]


def simulate(model, I, areas, t_end, rng, dt=DT, rec_cols=(), rec_every=10):
    """model in {'markov', 'foxlu', 'det'}; areas per column (um^2).

    Returns dict with spike times per column and a decimated V record for
    rec_cols."""
    areas = np.asarray(areas, dtype=float)
    C = len(areas)
    NNa = np.round(RHO_NA * areas).astype(np.int64)
    NK = np.round(RHO_K * areas).astype(np.int64)
    V = np.full(C, resting_potential(0.0))
    if model == "markov":
        cNa, cK = stationary_counts(V, NNa, NK, rng)
        oNa, oK = cNa[:, 7] / NNa, cK[:, 4] / NK
    else:
        m, h, n = (np.repeat(x, C) for x in steady(V[:1]))
        oNa, oK = m ** 3 * h, n ** 4
    nsteps = int(round(t_end / dt))
    rec_cols = list(rec_cols)
    rec = np.empty((nsteps // rec_every + 1, len(rec_cols)))
    sp = _Spikes(C)
    for k in range(nsteps):
        if k % rec_every == 0:
            rec[k // rec_every] = V[rec_cols]
        dV = (I - g_Na * oNa * (V - E_Na) - g_K * oK * (V - E_K) - g_L * (V - E_L)) / C_m
        if model == "markov":
            PNa, PK = transition_matrices(V, dt)
            cNa, cK = markov_step(cNa, cK, PNa, PK, rng)
            oNa, oK = cNa[:, 7] / NNa, cK[:, 4] / NK
        else:
            am, bm, ah, bh, an, bn = rates(V)
            if model == "det":           # exponential Euler = N -> infinity of the Markov scheme
                for x, al, be in ((m, am, bm), (h, ah, bh), (n, an, bn)):
                    a, b = subunit_probs(al, be, dt)
                    x[:] = x * b + (1.0 - x) * a
            else:                        # Fox-Lu, Euler-Maruyama
                sq = np.sqrt(dt)
                for x, al, be, N in ((m, am, bm, NNa), (h, ah, bh, NNa), (n, an, bn, NK)):
                    drift = al * (1.0 - x) - be * x
                    var = np.clip(al * (1.0 - x) + be * x, 0.0, None) / N
                    x += dt * drift + sq * np.sqrt(var) * rng.standard_normal(C)
                    np.clip(x, 0.0, 1.0, out=x)
            oNa, oK = m ** 3 * h, n ** 4
        Vold = V
        V = V + dt * dV
        sp.update(k * dt, Vold, V)
    if nsteps % rec_every == 0:
        rec[nsteps // rec_every] = V[rec_cols]
    return {"spikes": sp.per_column(C), "rec": rec, "rec_t": np.arange(rec.shape[0]) * dt * rec_every,
            "areas": areas}


def hh_reference(I, t_end):
    """Deterministic HH with DOP853; returns spike times (0 mV upward)."""
    def f(_, y):
        V, m, h, n = y
        am, bm, ah, bh, an, bn = rates(V)
        return [(I - g_Na * m ** 3 * h * (V - E_Na) - g_K * n ** 4 * (V - E_K) - g_L * (V - E_L)) / C_m,
                am * (1 - m) - bm * m, ah * (1 - h) - bh * h, an * (1 - n) - bn * n]

    def ev(_, y):
        return y[0]
    ev.direction = 1.0
    V0 = resting_potential(0.0)
    sol = solve_ivp(f, (0.0, t_end), [V0, *steady(V0)], method="DOP853", rtol=1e-10, atol=1e-10,
                    events=ev)
    return sol.t_events[0]


# ------------------------------------------------------------- validation
def check_clamp(rng):
    nV = len(CLAMP_V)
    V = np.repeat(CLAMP_V, CLAMP_M)
    C = len(V)
    NNa = int(RHO_NA * CLAMP_AREA)
    NK = int(RHO_K * CLAMP_AREA)
    cNa = np.zeros((C, 8), dtype=np.int64)
    cNa[:, 0] = NNa
    cK = np.zeros((C, 5), dtype=np.int64)
    cK[:, 0] = NK
    PNa, PK = transition_matrices(V, CLAMP_DT)
    for _ in range(int(round(CLAMP_T / CLAMP_DT))):
        cNa, cK = markov_step(cNa, cK, PNa, PK, rng)
    out = []
    for iv, v in enumerate(CLAMP_V):
        sl = slice(iv * CLAMP_M, (iv + 1) * CLAMP_M)
        m, h, n = steady(v)
        for name, x, N, p in (("Na", cNa[sl, 7], NNa, m ** 3 * h), ("K", cK[sl, 4], NK, n ** 4)):
            x = x.astype(float)
            M = len(x)
            mu, var = x.mean(), x.var(ddof=1)
            se_mu = np.sqrt(N * p * (1 - p) / M)
            mu4 = np.mean((x - mu) ** 4)
            se_var = np.sqrt(max(mu4 - var ** 2 * (M - 3) / (M - 1), 1e-30) / M)
            out.append(dict(V=v, ch=name, N=N, p=p, mean=mu, var=var, mean_th=N * p,
                            var_th=N * p * (1 - p), z_mean=(mu - N * p) / se_mu,
                            z_var=(var - N * p * (1 - p)) / se_var, counts=x))
    return out


def check_step(rng):
    M = STEP_M
    NNa = int(RHO_NA * CLAMP_AREA)
    NK = int(RHO_K * CLAMP_AREA)
    V0 = np.full(M, STEP_V0)
    cNa, cK = stationary_counts(V0, np.full(M, NNa), np.full(M, NK), rng)
    V = np.full(M, STEP_V1)
    PNa, PK = transition_matrices(V, DT)
    nsteps = int(round(STEP_T / DT))
    every = nsteps // 100
    ts, fNa, fK = [], [], []
    for k in range(1, nsteps + 1):
        cNa, cK = markov_step(cNa, cK, PNa, PK, rng)
        if k % every == 0:
            ts.append(k * DT)
            fNa.append(cNa[:, 7].mean() / NNa)
            fK.append(cK[:, 4].mean() / NK)
    ts = np.array(ts)
    am, bm, ah, bh, an, bn = rates(STEP_V1)
    x0 = steady(STEP_V0)
    xs = []
    for x00, al, be in zip(x0, (am, ah, an), (bm, bh, bn)):
        xi, tau = al / (al + be), 1.0 / (al + be)
        xs.append(xi + (x00 - xi) * np.exp(-ts / tau))
    pNa, pK = xs[0] ** 3 * xs[1], xs[2] ** 4
    zNa = (np.array(fNa) - pNa) / np.sqrt(pNa * (1 - pNa) / (NNa * M))
    zK = (np.array(fK) - pK) / np.sqrt(pK * (1 - pK) / (NK * M))
    return dict(t=ts, fNa=np.array(fNa), fK=np.array(fK), pNa=pNa, pK=pK, zNa=zNa, zK=zK)


def isi_stats(spikes, t0):
    isis = []
    for s in spikes:
        s = s[s >= t0]
        if len(s) > 1:
            isis.append(np.diff(s))
    isis = np.concatenate(isis) if isis else np.array([])
    return isis


def loglog_slope(x, y):
    return np.polyfit(np.log(x), np.log(y), 1)[0]


def check_convergence(rng_markov, rng_fox):
    areas = np.repeat(CONV_AREAS, CONV_TRIALS)
    rec_areas = [3.0, 30.0, 1000.0]
    rec_cols = [np.where(areas == a)[0][0] for a in rec_areas]
    det = simulate("det", I_DRIVE, [1.0], CONV_T, None, rec_cols=[0])
    det_sp = det["spikes"][0]
    det_isi = np.diff(det_sp[det_sp >= CONV_T0]).mean()
    ref_sp = hh_reference(I_DRIVE, CONV_T)
    ref_isi = np.diff(ref_sp[ref_sp >= CONV_T0]).mean()
    res = {"det_isi": det_isi, "ref_isi": ref_isi, "det_t1": det_sp[0], "ref_t1": ref_sp[0], "det": det}
    big = CONV_AREAS >= SLOPE_MIN_AREA
    jit_areas = np.repeat(CONV_AREAS[big], JIT_TRIALS)
    for model, rng in (("markov", rng_markov), ("foxlu", rng_fox)):
        t0 = _time.time()
        S = simulate(model, I_DRIVE, areas, CONV_T, rng, rec_cols=rec_cols)
        mean_isi, cv, nisi = [], [], []
        for a in CONV_AREAS:
            cols = np.where(areas == a)[0]
            isis = isi_stats([S["spikes"][c] for c in cols], CONV_T0)
            mean_isi.append(isis.mean())
            cv.append(isis.std(ddof=1) / isis.mean())
            nisi.append(len(isis))
        J = simulate(model, I_DRIVE, jit_areas, JIT_T, rng)
        t1 = np.array([x[0] if len(x) else np.nan for x in J["spikes"]])
        assert np.all(np.isfinite(t1)), "a patch did not fire within the first-spike window"
        t1_sd = np.array([t1[jit_areas == a].std(ddof=1) for a in CONV_AREAS[big]])
        t1_mean = np.array([t1[jit_areas == a].mean() for a in CONV_AREAS[big]])
        res[model] = dict(mean_isi=np.array(mean_isi), cv=np.array(cv), nisi=np.array(nisi),
                          t1_sd=t1_sd, t1_mean=t1_mean, rec=S["rec"], rec_t=S["rec_t"],
                          rec_areas=rec_areas, secs=_time.time() - t0)
        Na = RHO_NA * CONV_AREAS[big]
        res[model]["cv_slope"] = loglog_slope(Na, res[model]["cv"][big])
        res[model]["t1_slope"] = loglog_slope(Na, t1_sd)
    return res


def check_spontaneous(rng_markov, rng_fox):
    areas = np.repeat(SPONT_AREAS, SPONT_TRIALS)
    res = {}
    for model, rng in (("markov", rng_markov), ("foxlu", rng_fox)):
        t0 = _time.time()
        S = simulate(model, 0.0, areas, SPONT_T, rng, rec_cols=[np.where(areas == 4.0)[0][0]])
        counts = np.array([sum(len(S["spikes"][c]) for c in np.where(areas == a)[0]) for a in SPONT_AREAS])
        T_tot = SPONT_TRIALS * SPONT_T / 1000.0
        res[model] = dict(counts=counts, rate=counts / T_tot, err=np.sqrt(np.maximum(counts, 1)) / T_tot,
                          rec=S["rec"][:, 0], rec_t=S["rec_t"], secs=_time.time() - t0)
    return res


def fox_clamp_variance(rng):
    """Fox-Lu open-fraction variance at the clamp voltages vs the exact binomial value."""
    out = []
    NNa, NK = RHO_NA * CLAMP_AREA, RHO_K * CLAMP_AREA
    M = 400
    V = np.repeat(CLAMP_V, M)
    am, bm, ah, bh, an, bn = rates(V)
    m, h, n = (x.copy() for x in steady(V))
    sq = np.sqrt(DT)
    for _ in range(int(round(CLAMP_T / DT))):
        for x, al, be, N in ((m, am, bm, NNa), (h, ah, bh, NNa), (n, an, bn, NK)):
            x += DT * (al * (1 - x) - be * x) + sq * np.sqrt(np.clip(al * (1 - x) + be * x, 0, None) / N) \
                * rng.standard_normal(len(x))
            np.clip(x, 0.0, 1.0, out=x)
    for iv, v in enumerate(CLAMP_V):
        sl = slice(iv * M, (iv + 1) * M)
        mi, hi, ni = steady(v)
        pNa, pK = mi ** 3 * hi, ni ** 4
        out.append(dict(V=v, ratio_Na=np.var(m[sl] ** 3 * h[sl], ddof=1) / (pNa * (1 - pNa) / NNa),
                        ratio_K=np.var(n[sl] ** 4, ddof=1) / (pK * (1 - pK) / NK)))
    return out


def validate():
    ss = np.random.SeedSequence(SEED)
    r = [np.random.default_rng(s) for s in ss.spawn(7)]
    R = {}

    t0 = _time.time()
    R["clamp"] = check_clamp(r[0])
    print(f"\n1. Voltage clamp, A = {CLAMP_AREA:.0f} um^2, {CLAMP_M} patches, {CLAMP_T:.0f} ms from all closed "
          f"({_time.time() - t0:.1f} s)")
    print("   V (mV)  ch     N       p          mean (th)            var (th)          z_mean  z_var")
    for c in R["clamp"]:
        print(f"   {c['V']:6.1f}  {c['ch']:<3} {c['N']:5d}  {c['p']:.4e}  {c['mean']:9.3f} ({c['mean_th']:9.3f})"
              f"  {c['var']:8.3f} ({c['var_th']:8.3f})  {c['z_mean']:+6.2f} {c['z_var']:+6.2f}")
    zmax = max(max(abs(c["z_mean"]), abs(c["z_var"])) for c in R["clamp"])
    print(f"   max |z| = {zmax:.2f} (tolerance {CLAMP_Z})")
    R["clamp_zmax"] = zmax

    t0 = _time.time()
    R["fox_clamp"] = fox_clamp_variance(r[1])
    print(f"   Fox-Lu open-fraction variance / exact binomial variance ({_time.time() - t0:.1f} s):")
    for c in R["fox_clamp"]:
        print(f"   V = {c['V']:6.1f} mV: Na {c['ratio_Na']:.3f}, K {c['ratio_K']:.3f}")

    t0 = _time.time()
    R["step"] = s = check_step(r[2])
    zs = max(np.abs(s["zNa"]).max(), np.abs(s["zK"]).max())
    print(f"\n2. Step {STEP_V0:.0f} -> {STEP_V1:.0f} mV, {STEP_M} patches ({_time.time() - t0:.1f} s): "
          f"max |z| Na {np.abs(s['zNa']).max():.2f}, K {np.abs(s['zK']).max():.2f} (tolerance {STEP_Z}); "
          f"max |mean - closed form|: Na {np.abs(s['fNa'] - s['pNa']).max():.2e}, "
          f"K {np.abs(s['fK'] - s['pK']).max():.2e}")
    R["step_zmax"] = zs

    t0 = _time.time()
    R["conv"] = c = check_convergence(r[3], r[4])
    print(f"\n3. Convergence at I = {I_DRIVE} uA/cm^2, {CONV_TRIALS} patches x {CONV_T:.0f} ms "
          f"({_time.time() - t0:.1f} s)")
    print(f"   deterministic ISI: DOP853 {c['ref_isi']:.4f} ms, same-scheme (Euler V, exp. Euler gates) "
          f"{c['det_isi']:.4f} ms ({(c['det_isi'] / c['ref_isi'] - 1) * 100:+.3f}%)")
    print(f"   deterministic first spike: DOP853 {c['ref_t1']:.4f} ms, same scheme {c['det_t1']:.4f} ms")
    print("   A (um^2)    N_Na    | Markov: ISI (ms)  dev %    CV     | Fox-Lu: ISI  dev %    CV")
    for i, a in enumerate(CONV_AREAS):
        mk, fx = c["markov"], c["foxlu"]
        print(f"   {a:8.0f} {RHO_NA * a:9.0f}  | {mk['mean_isi'][i]:8.4f} {100 * (mk['mean_isi'][i] / c['ref_isi'] - 1):+7.3f}"
              f"  {mk['cv'][i]:.5f} | {fx['mean_isi'][i]:8.4f} "
              f"{100 * (fx['mean_isi'][i] / c['ref_isi'] - 1):+7.3f}  {fx['cv'][i]:.5f}")
    print(f"   first-spike time, {JIT_TRIALS} patches per area: A, Markov mean/SD (ms), Fox-Lu mean/SD (ms)")
    for i, a in enumerate(CONV_AREAS[CONV_AREAS >= SLOPE_MIN_AREA]):
        print(f"   {a:8.0f}  {c['markov']['t1_mean'][i]:.4f} / {c['markov']['t1_sd'][i]:.5f}   "
              f"{c['foxlu']['t1_mean'][i]:.4f} / {c['foxlu']['t1_sd'][i]:.5f}")
    for mdl in ("markov", "foxlu"):
        print(f"   {mdl}: log-log slopes vs N_Na over A >= {SLOPE_MIN_AREA:.0f}: ISI CV {c[mdl]['cv_slope']:.3f}, "
              f"first-spike SD {c[mdl]['t1_slope']:.3f} (target -0.5 +- {SLOPE_TOL})")
    R["isi_err"] = abs(c["markov"]["mean_isi"][-1] / c["ref_isi"] - 1)

    t0 = _time.time()
    R["spont"] = sp = check_spontaneous(r[5], r[6])
    print(f"\n4. Spontaneous firing at I = 0, {SPONT_TRIALS} x {SPONT_T:.0f} ms per area ({_time.time() - t0:.1f} s)")
    print("   A (um^2)  N_Na  N_K | Markov rate (Hz)  | Fox-Lu rate (Hz)")
    for i, a in enumerate(SPONT_AREAS):
        print(f"   {a:7.0f} {RHO_NA * a:6.0f} {RHO_K * a:4.0f} | {sp['markov']['rate'][i]:7.2f} +- "
              f"{sp['markov']['err'][i]:5.2f} ({sp['markov']['counts'][i]:4d}) | {sp['foxlu']['rate'][i]:7.2f} +- "
              f"{sp['foxlu']['err'][i]:5.2f} ({sp['foxlu']['counts'][i]:4d})")

    # assertions
    assert zmax < CLAMP_Z, f"clamp statistics off the binomial closed form: max |z| = {zmax:.2f}"
    assert zs < STEP_Z, f"step relaxation off the closed form: max |z| = {zs:.2f}"
    assert R["isi_err"] < ISI_TOL, f"Markov ISI at A = 1e5 um^2 off by {R['isi_err'] * 100:.2f}%"
    for key in ("cv_slope", "t1_slope"):
        sl = c["markov"][key]
        assert abs(sl + 0.5) < SLOPE_TOL, f"Markov {key} = {sl:.3f}, expected -0.5 +- {SLOPE_TOL}"
    assert sp["markov"]["rate"][0] > sp["markov"]["rate"][-1], "spontaneous rate does not fall with area"
    print("\nAll assertions passed.")
    return R


# ------------------------------------------------------------------- plot
def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    c = R["conv"]
    a = ax[0, 0]
    det = c["det"]
    tt = det["rec_t"]
    sel = tt <= 100
    for j, (A, col) in enumerate(zip(c["markov"]["rec_areas"], ("tab:red", "tab:orange", "tab:blue"))):
        a.plot(c["markov"]["rec_t"][sel], c["markov"]["rec"][sel, j], color=col, lw=0.8,
               label=f"Markov, A = {A:g} µm² ({RHO_NA * A:.0f} Na)")
    a.plot(tt[sel], det["rec"][sel, 0], color="k", lw=0.8, ls="--", label="deterministic HH")
    a.set_xlabel("time (ms)")
    a.set_ylabel("V (mV)")
    a.set_title(f"I = {I_DRIVE:g} µA/cm²: noise shrinks with patch area")
    a.legend(fontsize=7, loc="lower left")

    a = ax[0, 1]
    cl = [x for x in R["clamp"] if x["V"] == -45.0]
    for x, col in zip(cl, ("tab:red", "tab:blue")):
        k = np.arange(int(x["counts"].min()), int(x["counts"].max()) + 1)
        h = np.bincount(x["counts"].astype(int) - k[0], minlength=len(k)) / len(x["counts"])
        a.bar(k, h, width=1.0, color=col, alpha=0.4, label=f"{x['ch']} open count, simulated")
        a.plot(k, binom.pmf(k, x["N"], x["p"]), color=col, lw=1.5, label=f"Bin({x['N']}, {x['p']:.2e})")
    a.set_xlabel("open channels at −45 mV clamp")
    a.set_ylabel("probability")
    a.set_title(f"Stationary counts vs binomial (max |z| = {R['clamp_zmax']:.2f} over 3 V × 2 × mean/var)",
                fontsize=10)
    a.legend(fontsize=7)

    a = ax[0, 2]
    s = R["step"]
    a.plot(s["t"], s["fK"], "o", ms=3, color="tab:blue", label="Markov mean, K")
    a.plot(s["t"], s["pK"], color="k", lw=1, label="n(t)⁴ closed form")
    a.plot(s["t"], s["fNa"], "s", ms=3, color="tab:red", label="Markov mean, Na")
    a.plot(s["t"], s["pNa"], color="gray", lw=1, label="m(t)³h(t) closed form")
    a.set_xlabel("time after step −65 → 0 mV (ms)")
    a.set_ylabel("open fraction")
    a.set_title(f"Relaxation vs closed form (max |z| = {R['step_zmax']:.2f})")
    a.legend(fontsize=7)

    N = RHO_NA * CONV_AREAS
    a = ax[1, 0]
    for mdl, mk, col in (("markov", "o", "tab:blue"), ("foxlu", "^", "tab:green")):
        a.loglog(N, c[mdl]["cv"], mk + "-", color=col, label=f"ISI CV, {mdl} (slope {c[mdl]['cv_slope']:.2f})")
    big = CONV_AREAS >= SLOPE_MIN_AREA
    for mdl, mk, col in (("markov", "o", "tab:blue"), ("foxlu", "^", "tab:green")):
        a.loglog(N[big], c[mdl]["t1_sd"], mk + ":", color=col, mfc="none",
                 label=f"first-spike SD (ms), {mdl} (slope {c[mdl]['t1_slope']:.2f})")
    a.loglog(N[big], c["markov"]["cv"][big][0] * (N[big] / N[big][0]) ** -0.5, color="k", lw=2, alpha=0.4,
             label="∝ N^(−1/2)")
    a.set_xlabel("N_Na (N_K = 0.3 N_Na)")
    a.set_title(f"I = {I_DRIVE:g} µA/cm²: fluctuations vanish as N^(−1/2)")
    a.legend(fontsize=7)

    a = ax[1, 1]
    for mdl, mk, col in (("markov", "o", "tab:blue"), ("foxlu", "^", "tab:green")):
        a.semilogx(N, c[mdl]["mean_isi"], mk + "-", color=col, label=mdl)
    a.axhline(c["ref_isi"], color="k", ls="--", lw=1, label=f"deterministic HH, DOP853: {c['ref_isi']:.3f} ms")
    a.set_xlabel("N_Na")
    a.set_ylabel("mean ISI (ms)")
    err = R["isi_err"] * 100
    a.set_title(f"Mean ISI → deterministic ({err:.2f}% at N_Na = {N[-1]:.0f})")
    a.legend(fontsize=8)

    a = ax[1, 2]
    sp = R["spont"]
    for mdl, mk, col in (("markov", "o", "tab:blue"), ("foxlu", "^", "tab:green")):
        r = sp[mdl]["rate"]
        e = sp[mdl]["err"]
        pos = r > 0
        a.errorbar(SPONT_AREAS[pos], r[pos], yerr=e[pos], fmt=mk + "-", color=col, ms=5, capsize=2,
                   label=f"{mdl}")
        if (~pos).any():
            a.plot(SPONT_AREAS[~pos], 1.0 / (SPONT_TRIALS * SPONT_T / 1000) * np.ones((~pos).sum()), "v",
                   color=col, mfc="none", ms=7)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("patch area (µm²)")
    a.set_ylabel("spontaneous rate at I = 0 (Hz)")
    a.set_title("Spontaneous firing vs area (▽: no spikes, upper bound)")
    a.legend(fontsize=8)

    fig.suptitle("Hodgkin–Huxley with stochastic channels: Markov chain (exact frozen-V binomial update) "
                 "and Fox–Lu Langevin", fontsize=13)
    fig.tight_layout()
    out = FIGURES / "stochastic_channels.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    print(f"RNG seed: {SEED} (PCG64, one spawned stream per experiment)")
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
