"""Balanced network of binary neurons (van Vreeswijk & Sompolinsky 1996, 1998).

Model (van Vreeswijk & Sompolinsky 1998, sec. 2). Two populations k = E, I of
N binary neurons each, sigma_i in {0, 1}. Neuron i of population k is updated
at the times of its own Poisson clock of rate 1/tau_k (asynchronous update);
at an update it sets

    sigma_i = Theta(h_i),
    h_i = sum_l sum_j J_ij^kl sigma_j + sqrt(K) E_k m0 - theta_k ,

with J_ij^kl = J_kl / sqrt(K) if j (in population l) projects to i, 0
otherwise. Each of the N^2 pairs of every block (k, l) is connected
independently with probability p = K / N (so each neuron receives on average
K inputs from each population; no autapses are excluded explicitly, they
occur with probability p / N). External input is the constant
sqrt(K) E_k m0. Parameters (the set used by van Vreeswijk & Sompolinsky
1998): J_EE = 1, J_EI = -J_E = -2, J_IE = 1, J_II = -J_I = -1.8, E = 1,
I = 0.8, theta_E = 1, theta_I = 0.7; tau_E = 10 ms, tau_I = 9 ms; N = 20 000
per population; K = 125 ... 1000; m0 = 0.05 ... 0.3.

Integrator: event-exact update clocks on a time grid of dt = 0.1 ms. In each
step every neuron's clock fires with probability 1 - exp(-dt/tau_k) (two
firings of the same clock in one step give the same state, so this is exact
for the clocks); the neurons updated in a step all read the input at the
start of the step, which is the only approximation (simultaneity, O(dt);
checked against dt = 0.05 and 0.2 ms). Inputs are kept as running sums and
incremented only by the neurons that flip.

Theory used for validation:
(1) Balanced limit K -> infinity (vV&S 1998, eqs. 4.3-4.4). The O(sqrt K)
    part of the mean input must cancel: E_k m0 + sum_l J_kl m_l = 0, so the
    population rates are linear in the external input,
        m_E = (J_I E - J_E I) / (J_E - J_I) m0 ,
        m_I = (E - I) / (J_E - J_I) m0 ,
    (here both = m0), valid when E/I > J_E/J_I > 1.
(2) Finite-K mean field (vV&S 1998, eqs. 3.4-3.10, sec. 5). In the sparse
    limit the input to a neuron is Gaussian, with population mean
    u_k = sqrt(K)(E_k m0 + sum_l J_kl m_l) - theta_k and variance
    alpha_k = sum_l J_kl^2 (m_l - p q_l); its quenched (across neurons) part
    is beta_k = sum_l J_kl^2 (1 - p) q_l, q_l = <m_i^2> the second moment of
    the time-averaged rates. (The p terms are the finite-N corrections for
    Bernoulli connectivity; vV&S take p -> 0.) Then
        m_k = int Dz H(-(u_k + sqrt(beta_k) z) / sqrt(alpha_k - beta_k))
            = H(-u_k / sqrt(alpha_k)),
        q_k = int Dz H(-(u_k + sqrt(beta_k) z) / sqrt(alpha_k - beta_k))^2,
    H(x) = erfc(x / sqrt 2) / 2, Dz the standard Gaussian measure (Gauss-
    Hermite quadrature, 120 nodes); solved with fsolve. The population
    averages obey tau_k dm_k/dt = -m_k + H(-u_k(t) / sqrt(alpha_k(t)))
    (vV&S 1998, eq. 3.7).
(2') Exact-count mean field. The same independence assumptions without the
    Gaussian approximation: the numbers of active E and I inputs are
    Binomial(N, p m_l), and m_k = P(J_kE n_E + J_kI n_I + K E_k m0 -
    sqrt(K) theta_k > 0), summed over the integer lattice (count_F). At
    K m ~ 5-60 active inputs the input is visibly discrete and skewed, and
    this, not (2), is what the network should match. Its rate map drives
    the step-response ODE tau_k dm_k/dt = -m_k + F_k(m) (RK4, dt = 0.1 ms).
(3) Irregularity. If a neuron's input were independent between its updates,
    its state would be a two-state Markov chain and the intervals between
    its 0 -> 1 transitions would have CV^2 = r^2 + (1 - r)^2 = 1 - 2r(1 - r),
    r its time-averaged rate; temporal correlations of the input raise it.

Validation (ROADMAP note: firing rates scale as predicted when input is
increased; CV ~ 1): simulated population rates against (2') (asserted) and
(2) (reported) over K at m0 = 0.1 and over m0 at K = 1000, the slope dm/dm0
against (2') and (1), and the approach to (1) with K; the quenched rate variance q - m^2 (from the
covariance of the two halves of a long run, which removes the temporal
noise) against (2); the median CV of the inter-transition intervals; the
response to a step in m0 against the mean-field ODE (fast tracking); and a
dt check.

RNG: NumPy PCG64 seeded with SEED (printed): connectivity, initial states and
update clocks.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc, erfcinv      # closed forms only
from scipy.optimize import fsolve            # mean-field solution only
from scipy.stats import binom                # count mean field only

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Block 1: Parameters
# ---------------------------------------------------------------------------
N = 20000                       # neurons per population
J_E, J_I = 2.0, 1.8
JMAT = np.array([[1.0, -J_E],   # [post k, pre l]: rows E, I; columns E, I
                 [1.0, -J_I]])
EXT = np.array([1.0, 0.8])      # E, I
THETA = np.array([1.0, 0.7])    # theta_E, theta_I
TAU = np.array([10.0, 9.0])     # ms
DT = 0.1                        # ms
SEED = 1998

K_LIST = [125, 250, 500, 1000]
M0_LIST = [0.05, 0.1, 0.15, 0.2, 0.3]
K_MAIN = 1000
M0_MAIN = 0.1
T_TRANS = 200.0                 # ms discarded
T_MEAS = 1000.0                 # ms measured (sweeps)
T_LONG = 8000.0                 # ms measured (CV and rate distribution run)
K_DT = 250                      # dt check
DT_CHECK = [0.2, 0.05]
M0_STEP = (0.1, 0.15)           # step response at K_MAIN

# tolerances, fixed before the run
TOL_RATE = 0.03      # |m_sim / m_MF - 1|, every K, m0 and population
TOL_SLOPE = 0.05     # |slope_sim / slope_MF - 1| of m vs m0 at K = 1000
TOL_Q = 0.10         # |var_sim / var_MF - 1| of the quenched rate variance
CV_RANGE = (0.8, 1.2)  # median CV of inter-transition intervals
TOL_TRACK = 0.10     # RMS(sim - MF ODE) / step size, first 50 ms after step
TOL_DT = 0.02        # |m(dt) / m(0.1 ms) - 1|


# ---------------------------------------------------------------------------
# Block 2: Model
# ---------------------------------------------------------------------------
def H(x):
    return 0.5 * erfc(x / np.sqrt(2.0))


def Hinv(y):
    return np.sqrt(2.0) * erfcinv(2.0 * y)


def balanced_limit(m0):
    mE = (J_I * EXT[0] - J_E * EXT[1]) / (J_E - J_I) * m0
    mI = (EXT[0] - EXT[1]) / (J_E - J_I) * m0
    return np.array([mE, mI])


def build(K, rng):
    """Bernoulli(p = K/N) connectivity as 4 CSR blocks indexed by presynaptic
    neuron: blocks[(k, l)] = (indptr, targets in population k)."""
    p = K / N
    blocks = {}
    chunk = 1000
    for k in range(2):
        for l in range(2):
            ptr = np.zeros(N + 1, dtype=np.int64)
            tg = []
            for a in range(0, N, chunk):
                pre, post = np.nonzero(rng.random((chunk, N), dtype=np.float32) < p)
                ptr[a + 1:a + chunk + 1] = np.bincount(pre, minlength=chunk)
                tg.append(post.astype(np.int32))
            blocks[(k, l)] = (np.cumsum(ptr), np.concatenate(tg))
    return blocks


def gather(indptr, tg, idx):
    """Targets of presynaptic neurons idx, and the index of each source."""
    L = indptr[idx + 1] - indptr[idx]
    tot = int(L.sum())
    if tot == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.int64)
    start = np.repeat(indptr[idx] - np.concatenate(([0], np.cumsum(L)[:-1])), L)
    return tg[start + np.arange(tot)], np.repeat(np.arange(len(idx)), L)


# ---------------------------------------------------------------------------
# Block 3: Simulation
# ---------------------------------------------------------------------------
def simulate(blocks, K, m0, rng, T_meas=T_MEAS, dt=DT, m0_after=None, t_step=None,
             record=False, n_sample=200):
    """Asynchronous-update network. Returns population rates over the
    measurement window and, if record, per-neuron statistics.
    m0_after/t_step: change m0 at t_step (ms after the transient)."""
    W = JMAT / np.sqrt(K)
    s = [(rng.random(N) < m0).astype(np.int8) for _ in range(2)]
    # input from population l onto population k
    u = np.zeros((2, 2, N))
    for k in range(2):
        for l in range(2):
            t, _ = gather(*blocks[(k, l)], np.nonzero(s[l])[0])
            u[k, l] = W[k, l] * np.bincount(t, minlength=N)
    p_upd = 1.0 - np.exp(-dt / TAU)

    def ext(m0v):
        return np.sqrt(K) * EXT * m0v - THETA

    x = ext(m0)
    n_tr, n_me = int(round(T_TRANS / dt)), int(round(T_meas / dt))
    nst = n_tr + n_me
    pop = np.zeros((n_me, 2))
    cnt = np.array([s[0].sum(), s[1].sum()], dtype=np.int64)
    if record:
        half = n_me // 2
        acc = np.zeros((2, 2, N))              # [half, pop, neuron]
        up_t, up_i = [], []
        samp = np.zeros((n_me // 10 + 1, 4))    # mean E, I, ext and net input to E
        h_samp = []
    for it in range(nst):
        im = it - n_tr
        if t_step is not None and im == int(round(t_step / dt)):
            x = ext(m0_after)
        flips = [None, None]
        for k in range(2):
            upd = np.nonzero(rng.random(N) < p_upd[k])[0]
            if len(upd) == 0:
                continue
            new = (u[k, 0, upd] + u[k, 1, upd] + x[k] > 0).astype(np.int8)
            ch = new != s[k][upd]
            if ch.any():
                flips[k] = (upd[ch], new[ch])
        # every neuron updated in this step has read the input at its start;
        # now apply the flips of both populations
        for l in range(2):
            if flips[l] is None:
                continue
            idx, nv = flips[l]
            s[l][idx] = nv
            sg = 2.0 * nv - 1.0
            cnt[l] += int(sg.sum())
            for k in range(2):
                t, src = gather(*blocks[(k, l)], idx)
                if len(t):
                    u[k, l] += W[k, l] * np.bincount(t, weights=sg[src], minlength=N)
            if record and l == 0 and im >= 0:
                up = idx[nv == 1]
                up_t.append(np.full(len(up), im, dtype=np.int32))
                up_i.append(up.astype(np.int32))
        if im >= 0:
            pop[im] = cnt / N
            if record:
                acc[0 if im < half else 1, 0] += s[0]
                acc[0 if im < half else 1, 1] += s[1]
                if im % 10 == 0:
                    samp[im // 10] = (u[0, 0].mean(), u[0, 1].mean(), x[0],
                                      (u[0, 0] + u[0, 1]).mean() + x[0])
                    if im % 100 == 0:
                        h_samp.append(u[0, 0, :n_sample] + u[0, 1, :n_sample] + x[0])
    out = dict(pop=pop, m=pop.mean(0))
    if record:
        half_len = np.array([half, n_me - half])
        out["rates_h"] = acc / half_len[:, None, None]
        out["rates"] = acc.sum(0) / n_me
        out["up_t"] = np.concatenate(up_t) * dt
        out["up_i"] = np.concatenate(up_i)
        out["inputs"] = samp[:len(range(0, n_me, 10))].mean(0)
        out["h_samp"] = np.concatenate(h_samp)
    return out


def batch_se(x, nb=20):
    b = np.array_split(x, nb)
    mb = np.array([bb.mean(0) for bb in b])
    return mb.std(0, ddof=1) / np.sqrt(nb)


# ---------------------------------------------------------------------------
# Block 4: Mean-field theory
# ---------------------------------------------------------------------------
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(120)
GH_W = GH_W / GH_W.sum()


def mf_moments(K, m0, n=N):
    """Return m, q, u, alpha, beta (arrays over E, I) of the vV&S theory;
    n = np.inf gives the sparse limit p = 0 of vV&S."""
    p = K / n
    lim = balanced_limit(m0)

    def parts(v):
        m, q = v[:2], v[2:]
        u = np.sqrt(K) * (EXT * m0 + JMAT @ m) - THETA
        a = (JMAT ** 2) @ (m - p * q)
        b = (JMAT ** 2) @ ((1 - p) * q)
        return m, q, u, a, b

    def q_of(m, q):
        _, _, u, a, b = parts(np.concatenate([m, q]))
        b = np.maximum(b, 0.0)
        g = H(-(u[:, None] + np.sqrt(b)[:, None] * GH_X[None, :]) /
              np.sqrt(np.maximum(a - b, 1e-15))[:, None])
        return (g ** 2) @ GH_W

    # alternate: m from H(-u/sqrt(alpha)) = m at fixed q (2-d root), then q
    # from the quadrature at fixed m, until both stop changing
    def Fm_at(q):
        def Fm(mm):
            _, _, u, a, _ = parts(np.concatenate([mm, q]))
            return H(-u / np.sqrt(np.maximum(a, 1e-12))) - mm
        return Fm

    def solve_m(q, guess):
        Fm = Fm_at(q)
        for f in (1.0, 0.8, 0.6, 0.4, 0.2):
            mm, _, ier, _ = fsolve(Fm, f * guess, xtol=1e-14, full_output=True)
            if np.all(mm > 0) and np.max(np.abs(Fm(mm))) < 1e-12:
                return mm
        raise AssertionError("mean-field rate equation not solved")

    m = solve_m(lim ** 2, lim)
    q = 1.2 * m ** 2
    for _ in range(200):
        m_new = solve_m(q, m)
        q_new = q_of(m_new, q)
        done = np.max(np.abs(m_new - m)) < 1e-14 and np.max(np.abs(q_new - q)) < 1e-15
        m, q = m_new, q_new
        if done:
            break
    best = np.concatenate([m, q])
    _, _, u, a, b = parts(best)
    g = H(-(u[:, None] + np.sqrt(b)[:, None] * GH_X[None, :]) / np.sqrt(a - b)[:, None])
    resid = max(np.max(np.abs(g @ GH_W - m)), np.max(np.abs((g ** 2) @ GH_W - q)))
    assert resid < 1e-10 and np.all(m > 0), f"mean-field solution not found ({resid:.1e})"
    m, q, u, a, b = parts(best)
    return dict(m=m, q=q, u=u, alpha=a, beta=b)


def count_F(K, m0, n=N):
    """Rate map without the Gaussian approximation: the numbers of active E
    and I inputs are Binomial(n, p m_l) (independent inputs, each of the n
    presynaptic neurons connected with probability p and active with
    probability m_l), and m_k = P(J_kE n_E + J_kI n_I + K E_k m0 - sqrt(K)
    theta_k > 0), summed exactly over the integer lattice."""
    p = K / n

    def F(m):
        m = np.clip(m, 1e-12, 1.0)
        out = np.zeros(2)
        mu = n * p * m[0]
        nE = np.arange(0, int(mu + 15 * np.sqrt(mu + 1) + 20))
        pE = binom.pmf(nE, n, p * m[0])
        for k in range(2):
            x = (JMAT[k, 0] * nE + K * EXT[k] * m0 - THETA[k] * np.sqrt(K)) / (-JMAT[k, 1])
            out[k] = np.sum(pE * binom.cdf(np.ceil(x) - 1, n, p * m[1]))
        return out
    return F


def count_mf(K, m0, guess, n=N):
    F = count_F(K, m0, n)
    m = fsolve(lambda mm: F(mm) - mm, guess, xtol=1e-13, full_output=True)[0]
    assert np.max(np.abs(F(m) - m)) < 1e-10 and np.all(m > 0), "count mean field not solved"
    return m


def mf_rate_fn(K, m0, q_over_m2):
    p = K / N

    def Fm(m):
        u = np.sqrt(K) * (EXT * m0 + JMAT @ m) - THETA
        a = (JMAT ** 2) @ (m - p * q_over_m2 * m ** 2)
        return H(-u / np.sqrt(a))
    return Fm


def mf_ode(K, m_init, m0, T, dt):
    Fm = count_F(K, m0)
    f = lambda m: (-m + Fm(m)) / TAU
    n = int(round(T / dt))
    out = np.zeros((n + 1, 2))
    m = m_init.copy()
    out[0] = m
    for i in range(n):
        k1 = f(m); k2 = f(m + 0.5 * dt * k1); k3 = f(m + 0.5 * dt * k2); k4 = f(m + dt * k3)
        m = m + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        out[i + 1] = m
    return out


def mf_jacobian_eigs(K, m0, MF):
    Fm = mf_rate_fn(K, m0, MF["q"] / MF["m"] ** 2)
    eps = 1e-8
    Jc = np.zeros((2, 2))
    for j in range(2):
        d = np.zeros(2); d[j] = eps
        Jc[:, j] = (Fm(MF["m"] + d) - Fm(MF["m"] - d)) / (2 * eps)
    return np.linalg.eigvals((Jc - np.eye(2)) / TAU[:, None])


def cv_intervals(up_t, up_i, min_int=10):
    order = np.lexsort((up_t, up_i))
    t, i = up_t[order], up_i[order]
    cvs, nid = [], []
    bounds = np.nonzero(np.diff(i))[0] + 1
    for seg_t, seg_i in zip(np.split(t, bounds), np.split(i, bounds)):
        if len(seg_t) < min_int + 1:
            continue
        isi = np.diff(seg_t)
        cvs.append(isi.std() / isi.mean())
        nid.append(seg_i[0])
    return np.array(cvs), np.array(nid)


# ---------------------------------------------------------------------------
# Block 5: Validation
# ---------------------------------------------------------------------------
def validate():
    rng = np.random.default_rng(SEED)
    R = dict()
    lim = balanced_limit(M0_MAIN)
    print(f"Balanced limit (K -> inf): m_E = {lim[0]:.4f}, m_I = {lim[1]:.4f} at m0 = {M0_MAIN}; "
          f"slopes dm/dm0 = {balanced_limit(1.0)[0]:.4f}, {balanced_limit(1.0)[1]:.4f}")
    assert EXT[0] / EXT[1] > J_E / J_I > 1, "balance conditions violated"

    # --- 1. rates vs K at m0 = 0.1 -------------------------------------------
    print(f"\n1. Population rates vs K at m0 = {M0_MAIN} (N = {N} per population); "
          f"theory: vV&S Gaussian mean field (G) and exact-count mean field (C)")
    sims_K, mfs_K, cnt_K, errs_c, errs_g = {}, {}, {}, [], []
    blocks_main = None

    def report(label, S, MF, mc):
        se = batch_se(S["pop"])
        eg, ec = S["m"] / MF["m"] - 1, S["m"] / mc - 1
        errs_g.extend(np.abs(eg)); errs_c.extend(np.abs(ec))
        print(f"   {label}: m_E {S['m'][0]:.5f} +- {se[0]:.5f} vs C {mc[0]:.5f} ({ec[0] * 100:+.2f}%), "
              f"G {MF['m'][0]:.5f} ({eg[0] * 100:+.2f}%); m_I {S['m'][1]:.5f} +- {se[1]:.5f} vs "
              f"C {mc[1]:.5f} ({ec[1] * 100:+.2f}%), G {MF['m'][1]:.5f} ({eg[1] * 100:+.2f}%)")

    for K in K_LIST:
        t0 = _time.time()
        blocks = build(K, rng)
        MF = mf_moments(K, M0_MAIN)
        mc = count_mf(K, M0_MAIN, MF["m"])
        if K == K_MAIN:
            blocks_main = blocks
            S = simulate(blocks, K, M0_MAIN, rng, T_meas=T_LONG, record=True)
        else:
            S = simulate(blocks, K, M0_MAIN, rng)
        sims_K[K], mfs_K[K], cnt_K[K] = S, MF, mc
        eig = mf_jacobian_eigs(K, M0_MAIN, MF)
        report(f"K = {K:4d}", S, MF, mc)
        print(f"            mean active inputs K m: E {K * S['m'][0]:.1f}, I {K * S['m'][1]:.1f}; "
              f"MF eigenvalues {eig.real.max():.3f}, {eig.real.min():.3f} /ms [{_time.time() - t0:.1f} s]")
        if K != K_MAIN:
            del blocks
    R.update(sims_K=sims_K, mfs_K=mfs_K, cnt_K=cnt_K)

    # approach to the balanced limit
    Ks = np.array(K_LIST, float)
    mK = np.array([sims_K[K]["m"] for K in K_LIST])
    gap = lim[None, :] - mK
    sl = [np.polyfit(np.log(Ks), np.log(gap[:, j]), 1)[0] for j in range(2)]
    mg = np.array([mf_moments(K, M0_MAIN, n=np.inf)["m"] for K in (1e4, 1e5, 1e6)])
    c_inf = THETA + np.sqrt((JMAT ** 2) @ lim) * Hinv(lim)
    lead = np.linalg.solve(JMAT, c_inf)
    print(f"   gap to the balanced limit m_inf - m: E {gap[:, 0].round(4)}, I {gap[:, 1].round(4)}; "
          f"log-log slope vs K {sl[0]:.3f}, {sl[1]:.3f} (-0.5 only asymptotically)")
    print(f"   sqrt(K) (m - m_inf): E {(np.sqrt(Ks) * -gap[:, 0]).round(3)}, "
          f"I {(np.sqrt(Ks) * -gap[:, 1]).round(3)}; leading-order limit {lead.round(3)}; "
          f"G (N -> inf) at K = 1e4, 1e5, 1e6: m_E {mg[:, 0].round(4)}")
    R.update(gap=gap, gap_slope=sl, lead=lead)

    # --- 2. rates vs m0 at K = 1000 ------------------------------------------
    print(f"\n2. Population rates vs m0 at K = {K_MAIN}")
    sims_m0, mfs_m0, cnt_m0 = {}, {}, {}
    for m0 in M0_LIST:
        MF = mf_moments(K_MAIN, m0)
        mc = count_mf(K_MAIN, m0, MF["m"])
        S = sims_K[K_MAIN] if m0 == M0_MAIN else simulate(blocks_main, K_MAIN, m0, rng)
        sims_m0[m0], mfs_m0[m0], cnt_m0[m0] = S, MF, mc
        if m0 != M0_MAIN:
            report(f"m0 = {m0:.2f}", S, MF, mc)
    m0s = np.array(M0_LIST)
    ms = np.array([sims_m0[m]["m"] for m in M0_LIST])
    mcs = np.array([cnt_m0[m] for m in M0_LIST])
    mgs = np.array([mfs_m0[m]["m"] for m in M0_LIST])
    slope_s = [np.polyfit(m0s, ms[:, j], 1)[0] for j in range(2)]
    slope_c = [np.polyfit(m0s, mcs[:, j], 1)[0] for j in range(2)]
    slope_g = [np.polyfit(m0s, mgs[:, j], 1)[0] for j in range(2)]
    e_slope = max(abs(slope_s[j] / slope_c[j] - 1) for j in range(2))
    print(f"   slope dm/dm0 (linear fit over m0 = {M0_LIST[0]}-{M0_LIST[-1]}): E {slope_s[0]:.4f} "
          f"(C {slope_c[0]:.4f}, G {slope_g[0]:.4f}), I {slope_s[1]:.4f} (C {slope_c[1]:.4f}, "
          f"G {slope_g[1]:.4f}); balanced limit 1.0000; max |slope_sim/slope_C - 1| = {e_slope:.4f} "
          f"(tolerance {TOL_SLOPE})")
    e_rate, e_rate_g = max(errs_c), max(errs_g)
    print(f"   max |m_sim / m_C - 1| over all {len(errs_c)} rates = {e_rate:.4f} (tolerance {TOL_RATE})")
    print(f"   max |m_sim / m_G - 1| = {e_rate_g:.4f} (reported; Gaussian input assumed)")
    assert e_rate < TOL_RATE, f"rates off count mean field by {e_rate:.3f}"
    assert e_slope < TOL_SLOPE, f"slope off count mean field by {e_slope:.3f}"
    R.update(sims_m0=sims_m0, mfs_m0=mfs_m0, cnt_m0=cnt_m0, slope_s=slope_s, slope_c=slope_c,
             e_rate=e_rate, e_rate_g=e_rate_g)

    # --- 3. quenched rate variance, CV ---------------------------------------
    S, MF = sims_K[K_MAIN], mfs_K[K_MAIN]
    print(f"\n3. Long run, K = {K_MAIN}, m0 = {M0_MAIN}, {T_LONG / 1000:.0f} s")
    rh = S["rates_h"]
    var_s = np.array([np.mean((rh[0, j] - rh[0, j].mean()) * (rh[1, j] - rh[1, j].mean()))
                      for j in range(2)])
    var_t = MF["q"] - MF["m"] ** 2
    tot_s = S["rates"].var(axis=1)
    e_q = np.max(np.abs(var_s / var_t - 1))
    print(f"   quenched variance of time-averaged rates (split-half covariance): "
          f"E {var_s[0]:.3e} vs MF {var_t[0]:.3e}, I {var_s[1]:.3e} vs {var_t[1]:.3e}; "
          f"max rel. error {e_q:.4f} (tolerance {TOL_Q}); full-run variance incl. temporal noise "
          f"{tot_s[0]:.3e}, {tot_s[1]:.3e}")
    assert e_q < TOL_Q, f"quenched variance off by {e_q:.3f}"
    frac_silent = (S["rates"][0] < 1e-3).mean()
    cvs, nid = cv_intervals(S["up_t"], S["up_i"])
    r_i = S["rates"][0][nid]
    cv_markov = np.sqrt(1 - 2 * r_i * (1 - r_i))
    med = np.median(cvs)
    up_rate = len(S["up_t"]) / N / (T_LONG / 1000)
    print(f"   E neurons: mean 0->1 transition rate {up_rate:.2f} /s, {frac_silent * 100:.1f}% with rate < 1e-3; "
          f"{len(cvs)} neurons with >= 10 intervals")
    print(f"   CV of inter-transition intervals: median {med:.3f} (IQR {np.percentile(cvs, 25):.3f}-"
          f"{np.percentile(cvs, 75):.3f}); independent-input reference sqrt(1-2r(1-r)) median "
          f"{np.median(cv_markov):.3f}; median ratio {np.median(cvs / cv_markov):.3f}; "
          f"range asserted {CV_RANGE}")
    assert CV_RANGE[0] < med < CV_RANGE[1], f"median CV {med:.3f}"
    inp = S["inputs"]
    print(f"   mean input to E neurons: from E {inp[0]:+.3f}, from I {inp[1]:+.3f}, external "
          f"+ threshold {inp[2]:+.3f}, net {inp[3]:+.3f} (MF u_E {MF['u'][0]:+.3f}, "
          f"sd sqrt(alpha_E) {np.sqrt(MF['alpha'][0]):.3f}; sampled sd {S['h_samp'].std():.3f})")
    R.update(var_s=var_s, var_t=var_t, cvs=cvs, cv_markov=cv_markov, med_cv=med, e_q=e_q)

    # input components vs K
    comp = {}
    for K in K_LIST:
        MFk = mfs_K[K]
        comp[K] = (np.sqrt(K) * (EXT[0] * M0_MAIN + JMAT[0, 0] * sims_K[K]["m"][0]),
                   -np.sqrt(K) * J_E * sims_K[K]["m"][1], MFk["u"][0])
    R["comp"] = comp

    # --- 4. step response ----------------------------------------------------
    print(f"\n4. Step m0 {M0_STEP[0]} -> {M0_STEP[1]} at K = {K_MAIN} vs mean-field ODE")
    t_st, T_st = 100.0, 200.0
    S2 = simulate(blocks_main, K_MAIN, M0_STEP[0], rng, T_meas=T_st, m0_after=M0_STEP[1], t_step=t_st)
    MFa, MFb = mfs_m0[M0_STEP[0]], mfs_m0[M0_STEP[1]]
    m_pre = S2["pop"][:int(t_st / DT)].mean(0)
    ode = mf_ode(K_MAIN, m_pre, M0_STEP[1], T_st - t_st, DT)
    tt = np.arange(S2["pop"].shape[0]) * DT - t_st
    post = (tt >= 0) & (tt < 50)
    simp = S2["pop"][post]
    odep = ode[:post.sum()]
    step = count_mf(K_MAIN, M0_STEP[1], MFb["m"]) - count_mf(K_MAIN, M0_STEP[0], MFa["m"])
    e_track = np.sqrt(np.mean((simp - odep) ** 2, axis=0)) / np.abs(step)
    eig = mf_jacobian_eigs(K_MAIN, M0_STEP[1], MFb)
    t63 = [tt[post][np.argmax((simp[:, j] - m_pre[j]) / step[j] > 0.632)] for j in range(2)]
    t63o = [np.argmax((odep[:, j] - m_pre[j]) / step[j] > 0.632) * DT for j in range(2)]
    print(f"   step size (MF) {step.round(5)}; RMS(sim - ODE)/step over 0-50 ms: E {e_track[0]:.3f}, "
          f"I {e_track[1]:.3f} (tolerance {TOL_TRACK})")
    print(f"   63% rise time: sim {t63[0]:.1f}, {t63[1]:.1f} ms; ODE {t63o[0]:.1f}, {t63o[1]:.1f} ms; "
          f"MF eigenvalues {eig.real.round(3)} /ms (tau_E = {TAU[0]:.0f} ms)")
    assert e_track.max() < TOL_TRACK, "step response off mean-field ODE"
    R.update(step=dict(tt=tt, pop=S2["pop"], ode=ode, m_pre=m_pre, t63=t63, e=e_track))
    del blocks_main

    # --- 5. dt check ---------------------------------------------------------
    print(f"\n5. Time-grid check at K = {K_DT}, m0 = {M0_MAIN}")
    blocks = build(K_DT, rng)
    ref = simulate(blocks, K_DT, M0_MAIN, rng)
    e_dt = 0.0
    for dtc in DT_CHECK:
        Sd = simulate(blocks, K_DT, M0_MAIN, rng, dt=dtc)
        e = Sd["m"] / ref["m"] - 1
        e_dt = max(e_dt, np.abs(e).max())
        print(f"   dt = {dtc} ms: m = {Sd['m'].round(5)} vs {ref['m'].round(5)} at 0.1 ms "
              f"({(e * 100).round(2)}%; SE {(batch_se(Sd['pop']) / Sd['m'] * 100).round(2)}%)")
    print(f"   max |m(dt)/m(0.1) - 1| = {e_dt:.4f} (tolerance {TOL_DT})")
    assert e_dt < TOL_DT, "dt dependence"
    return R


# ---------------------------------------------------------------------------
# Block 6: Figure
# ---------------------------------------------------------------------------
def plot(R):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    S = R["sims_K"][K_MAIN]
    cE, cI = "tab:red", "tab:blue"

    ax = axes[0, 0]
    sel = R["sims_K"][K_MAIN]["up_i"] < 100
    tsel = R["sims_K"][K_MAIN]["up_t"][sel]
    win = tsel < 1000
    ax.plot(tsel[win], R["sims_K"][K_MAIN]["up_i"][sel][win], "|", color="k", ms=3)
    ax2 = ax.twinx()
    tp = np.arange(int(1000 / DT)) * DT
    ax2.plot(tp, S["pop"][:len(tp), 0], color=cE, lw=0.6, label="m_E(t)")
    ax2.plot(tp, S["pop"][:len(tp), 1], color=cI, lw=0.6, label="m_I(t)")
    ax2.set_ylim(0, 0.3)
    ax2.set_ylabel("fraction active")
    ax2.legend(fontsize=8, loc="upper right")
    ax.set_xlabel("time (ms)")
    ax.set_ylabel("E neuron (0→1 transitions)")
    ax.set_title(f"A. K = {K_MAIN}, m0 = {M0_MAIN}: transitions of 100 E neurons")

    ax = axes[0, 1]
    Kf = np.unique(np.round(np.geomspace(110, 1200, 60)))
    Ki = np.geomspace(100, 1e5, 50)
    mfK = np.array([mf_moments(K, M0_MAIN)["m"] for K in Kf])
    mcK = np.array([count_mf(int(K), M0_MAIN, g) for K, g in zip(Kf, mfK)])
    mfI = np.array([mf_moments(K, M0_MAIN, n=np.inf)["m"] for K in Ki])
    lim = balanced_limit(M0_MAIN)
    for j, (c, nm) in enumerate(((cE, "E"), (cI, "I"))):
        ax.plot(Kf, mfK[:, j], color=c, lw=1.5, label=f"Gaussian MF (vV&S), {nm}")
        ax.plot(Kf, mcK[:, j], color=c, lw=0.7, alpha=0.6, label=f"exact-count MF, {nm}")
        ax.plot(Ki, mfI[:, j], ":", color=c, lw=1, label=f"Gaussian MF, N → ∞, {nm}")
        ax.plot(K_LIST, [R["sims_K"][K]["m"][j] for K in K_LIST], "o", color=c, ms=7,
                mfc="none", mew=1.5, label=f"network, {nm}")
    ax.axhline(lim[0], color="k", ls="--", lw=1, label="balanced limit m_E = m_I = m0")
    ax.set_xscale("log")
    ax.set_xlabel("K (inputs per population)")
    ax.set_ylabel("population rate m")
    ax.set_title(f"B. Rates vs K at m0 = {M0_MAIN}: slow O(1/√K) approach to balance")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    m0f = np.linspace(0.045, 0.32, 30)
    mf0 = np.array([mf_moments(K_MAIN, m)["m"] for m in m0f])
    mc0 = np.array([count_mf(K_MAIN, m, g) for m, g in zip(m0f, mf0)])
    for j, (c, nm) in enumerate(((cE, "E"), (cI, "I"))):
        ax.plot(m0f, mf0[:, j], color=c, lw=1.5, label=f"Gaussian MF K = {K_MAIN}, {nm}")
        ax.plot(m0f, mc0[:, j], "--", color=c, lw=0.8, label=f"exact-count MF, {nm}")
        ax.plot(M0_LIST, [R["sims_m0"][m]["m"][j] for m in M0_LIST], "o", color=c, ms=7,
                mfc="none", mew=1.5, label=f"network, {nm} (slope {R['slope_s'][j]:.3f})")
    ax.plot(m0f, balanced_limit(m0f)[0], "k--", lw=1, label="balanced limit (slope 1)")
    ax.set_xlabel("external input m0")
    ax.set_ylabel("population rate m")
    ax.set_title(f"C. Rates vs input; all rates within {R['e_rate'] * 100:.1f}% of count MF")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[0, 3]
    Ks = np.array(K_LIST)
    ax.plot(Ks, [R["comp"][K][0] for K in K_LIST], "o-", color=cE, label="excitatory + external")
    ax.plot(Ks, [-R["comp"][K][1] for K in K_LIST], "s-", color=cI, label="|inhibitory|")
    ax.plot(Ks, [abs(R["comp"][K][2]) for K in K_LIST], "^-", color="k",
            label="|net input − θ_E| (mean field u_E)")
    ax.plot(Ks, 0.5 * np.sqrt(Ks / Ks[0]) * R["comp"][K_LIST[0]][0] * 2, ":", color="gray",
            label="∝ √K")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("K")
    ax.set_ylabel("mean input to E neurons")
    ax.set_title("D. Balance: O(√K) excitation and inhibition cancel")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1, 0]
    rh = S["rates_h"]
    MF = R["mfs_K"][K_MAIN]
    for j, (c, nm) in enumerate(((cE, "E"), (cI, "I"))):
        ax.hist(S["rates"][j], bins=np.linspace(0, 0.6, 61), density=True, histtype="step",
                color=c, lw=1.2, label=f"network {nm}, {T_LONG / 1000:.0f} s average")
        z = np.random.default_rng(0).standard_normal(400000)
        r = H(-(MF["u"][j] + np.sqrt(MF["beta"][j]) * z) / np.sqrt(MF["alpha"][j] - MF["beta"][j]))
        ax.hist(r, bins=np.linspace(0, 0.6, 61), density=True, histtype="stepfilled", alpha=0.2,
                color=c, label=f"MF distribution {nm}")
    ax.set_yscale("log")
    ax.set_xlabel("time-averaged rate m_i")
    ax.set_ylabel("density")
    ax.set_title(f"E. Rate distribution; quenched variance within {R['e_q'] * 100:.1f}% of MF")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(R["cvs"], bins=np.linspace(0.4, 2.0, 65), color="0.6", label="network (E)")
    ax.hist(R["cv_markov"], bins=np.linspace(0.4, 2.0, 65), histtype="step", color="k",
            label="independent-input reference √(1−2r(1−r))")
    ax.axvline(R["med_cv"], color=cE, lw=1.5, label=f"median {R['med_cv']:.3f}")
    ax.axvline(1, color="k", ls=":", lw=1)
    ax.set_xlabel("CV of intervals between 0→1 transitions")
    ax.set_ylabel("neurons")
    ax.set_title("F. Irregular activity: CV ≈ 1")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    st = R["step"]
    tt = st["tt"]
    ax.plot(tt, st["pop"][:, 0], color=cE, lw=0.6, alpha=0.8, label="network E")
    ax.plot(tt, st["pop"][:, 1], color=cI, lw=0.6, alpha=0.8, label="network I")
    to = np.arange(st["ode"].shape[0]) * DT
    ax.plot(to, st["ode"][:, 0], "k-", lw=1.5, label="mean-field ODE")
    ax.plot(to, st["ode"][:, 1], "k-", lw=1.5)
    ax.axvline(0, color="gray", ls=":")
    ax.axvline(TAU[0], color="gray", ls="--", lw=0.8, label="τ_E after step")
    ax.set_xlim(-30, 60)
    ax.set_xlabel("time from step (ms)")
    ax.set_ylabel("population rate")
    ax.set_title(f"G. Step m0 {M0_STEP[0]} → {M0_STEP[1]}: 63% rise {st['t63'][0]:.1f} ms (E)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 3]
    h = S["h_samp"]
    x = np.linspace(h.min(), h.max(), 300)
    ax.hist(h, bins=80, density=True, color="0.6", label="network: input to 200 E neurons")
    ax.plot(x, np.exp(-(x - MF["u"][0]) ** 2 / (2 * MF["alpha"][0])) / np.sqrt(2 * np.pi * MF["alpha"][0]),
            color=cE, lw=1.5, label="MF Gaussian N(u_E, α_E)")
    ax.axvline(0, color="k", ls=":", lw=1, label="threshold")
    ax.set_xlabel("net input h (threshold subtracted)")
    ax.set_ylabel("density")
    ax.set_title("H. Net input: O(1) mean below threshold, O(1) fluctuations")
    ax.legend(fontsize=8)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "balanced_network.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print(f"RNG seed: {SEED} (PCG64)")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
