"""Pair-based spike-timing-dependent plasticity (STDP): the Bi & Poo window,
weight drift and diffusion under Poisson input, the stationary weight
distribution, and competitive learning in a LIF neuron (Song, Miller &
Abbott 2000).

Rule (all-to-all pair interactions, implemented with traces as in Song,
Miller & Abbott 2000 and Morrison, Diesmann & Gerstner 2008), t in ms. Each
synapse keeps a presynaptic trace x and sees a postsynaptic trace y:

    dx/dt = -x/tau_+,   x -> x + 1 at each presynaptic spike
    dy/dt = -y/tau_-,   y -> y + 1 at each postsynaptic spike
    at a presynaptic spike:   w -> w - A_- f_-(w) y      (depression)
    at a postsynaptic spike:  w -> w + A_+ f_+(w) x      (potentiation)

so a single pair with lag s = t_post - t_pre changes w by the window

    W(s) =  A_+ exp(-s/tau_+)   for s > 0,
    W(s) = -A_- exp( s/tau_-)   for s < 0.

Additive rule: f_+ = f_- = 1, with hard bounds 0 <= w <= w_max (clipped).
Soft-bound (multiplicative) rule: f_+ = w_max - w, f_- = w (van Rossum, Bi &
Turrigiano 2000; Gütig et al. 2003 with mu = 1).

Integrator. The traces are linear, so between events they are advanced by
their exact exponential decay (event-driven; no discretisation error); this
is used for the pairing protocol and the open-loop Poisson experiments. The
closed-loop LIF neuron of Song et al. (2000),

    tau_m dV/dt = V_rest - V + g_ex (E_ex - V) + g_in (E_in - V),
    dg_ex/dt = -g_ex/tau_ex, g_ex -> g_ex + w_a at a spike of synapse a,
    dg_in/dt = -g_in/tau_in, g_in -> g_in + g_in_bar at an inhibitory spike,
    V >= V_th -> V = V_reset,

is integrated by forward Euler at dt = 0.1 ms, with input spikes on the grid
and the traces still decayed exactly. Within one step presynaptic spikes are
processed before the threshold test, so a pre and post spike in the same
step count as a pair with s = 0+ (potentiation).

Parameters.
  Bi & Poo window: tau_+ = 16.8 ms, tau_- = 33.7 ms (the exponential fits to
  the Bi & Poo 1998 data reported by Bi & Poo 2001); amplitudes chosen here,
  60 A_+ = 0.80 and 60 A_- = 0.35 (relative change after 60 pairs), with
  w0 = 1 and w_max = 2 for the soft-bound rule. Protocol as in Bi & Poo
  (1998): 60 pre/post pairs at 1 Hz, lag s = -100 ... +100 ms.
  Poisson experiments and LIF: Song et al. (2000): tau_+ = tau_- = 20 ms,
  A_+ = 0.005 w_max, A_-/A_+ = 1.05; 1000 excitatory synapses at 10 Hz with
  w_max = g_max = 0.015, 200 inhibitory at 10 Hz with g_in_bar = 0.05
  (conductances in units of the leak), tau_m = 20 ms, V_rest = -70,
  E_ex = 0, E_in = -70, V_th = -54, V_reset = -60 mV, tau_ex = tau_in = 5 ms.
  The open-loop stationary distribution uses A_+ = 0.002, A_-/A_+ = 1.01 (see
  check 4).

Closed forms used for validation (independent of the simulation code):

  Pairing protocol (pairs 1 s apart; cross-pair terms are below 3e-12):
    additive:      dw = 60 W(s)
    soft-bound:    s > 0: w_60 = w_max - (w_max - w0)(1 - A_+ e^(-s/tau_+))^60
                   s < 0: w_60 = w0 (1 - A_- e^(s/tau_-))^60
  Exact all-to-all pair sum for any trains (additive, unbounded):
    dw = sum_{post j} sum_{pre i} W(t_j - t_i).
  Independent Poisson pre (rate rho) and post (rate nu), additive and
  unbounded (Kempter, Gerstner & van Hemmen 1999 for the drift; the variance
  follows from the second moment of a bilinear Poisson functional):
    E[dw(T)] = rho nu [A_+ tau_+ (T - tau_+(1 - e^(-T/tau_+)))
                       - A_- tau_- (T - tau_-(1 - e^(-T/tau_-)))]
             -> mu T,  mu = rho nu Kbar,  Kbar = A_+ tau_+ - A_- tau_-
    Var[dw(T)] -> D T,  D = rho nu (A_+^2 tau_+ + A_-^2 tau_-)/2
                            + rho nu (rho + nu) Kbar^2.
  Hard bounds, diffusion (Fokker-Planck) limit: with mu < 0 the stationary
  density on [0, w_max] is the truncated exponential
    p(w) = exp(-w/lam) / (lam (1 - exp(-w_max/lam))),  lam = D / (2|mu|),
  mean lam - w_max/(exp(w_max/lam) - 1).

Validation (validate()):
1. Pairing protocol, s = -100 ... +100 ms in 2 ms steps (s != 0): the
   trace-based additive rule vs 60 W(s) and vs the exact pair sum, and the
   soft-bound rule vs its 60-fold map; max |error| < 1e-9 (relative change).
2. Trace implementation vs exact all-to-all pair sum on 5 independent
   20 Hz Poisson pre/post train pairs over 20 s; max relative error < 1e-9.
3. Drift and diffusion: 4000 independent synapses, rho = nu = 20 Hz,
   T = 100 s, additive unbounded with Song et al. amplitudes. Sample mean and
   variance of dw(T) vs E[dw(T)] and D T; |z| < 4 for both (standard
   errors from the sample: s/sqrt(M) and var sqrt(2/(M-1))).
4. Stationary distribution: 2000 independent synapses, hard bounds
   [0, 1], A_+ = 0.002, A_-/A_+ = 1.01, rho = nu = 20 Hz (lam = 0.101,
   about 50 jumps, so the diffusion limit should hold to a few %), started
   at w = 0.2. The slowest relaxation time of drifted Brownian motion with
   a reflecting wall is 2D/mu^2 (2525 s here); burn-in is 6 relaxation
   times, then 24 snapshots half a relaxation time apart are pooled
   (snapshots are taken at fixed event counts; Poisson events see time
   averages). Mean weight vs the Fokker-Planck mean within 5% (standard
   error from the per-synapse time averages) and Kolmogorov-Smirnov distance
   of the pooled sample from the truncated exponential < 0.03.
5. Closed loop, Song et al. (2000): 1000 synapses from uniform initial
   weights, 2000 s at 10 Hz. Asserted only qualitatively: the final weight
   distribution is bimodal in the sense that more than 60% of weights lie in
   the outer 20% of [0, g_max] (10% at each end; a uniform distribution has
   20%). Output rate and the fractions near each bound are reported.

RNG: NumPy PCG64 seeded with SEED (printed), one spawned stream per
experiment. SciPy is not needed (all references are closed forms or direct
sums); it is not imported.
"""

from pathlib import Path
import math
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- parameters
SEED = 1998

# Bi & Poo window (ms)
BP_TAU_P, BP_TAU_M = 16.8, 33.7
BP_N_PAIRS = 60
BP_A_P, BP_A_M = 0.80 / BP_N_PAIRS, 0.35 / BP_N_PAIRS
BP_W0, BP_WMAX = 1.0, 2.0
BP_PERIOD = 1000.0                                  # 1 Hz pairing
BP_LAGS = np.concatenate([np.arange(-100, 0, 2.0), np.arange(2, 101, 2.0)])

# Song, Miller & Abbott (2000)
TAU_P = TAU_M = 20.0                                # STDP time constants (ms)
A_P = 0.005                                         # in units of w_max
A_M = 1.05 * A_P
G_MAX = 0.015
N_E, R_E = 1000, 10.0                               # excitatory synapses, Hz
N_I, R_I, G_IN = 200, 10.0, 0.05                    # inhibitory synapses
TAU_MEM, V_REST, E_EX, E_IN = 20.0, -70.0, 0.0, -70.0
V_TH, V_RESET = -54.0, -60.0
TAU_EX = TAU_IN = 5.0
DT = 0.1                                            # ms, LIF Euler step
SONG_T = 2_000_000.0                                  # ms

# open-loop Poisson experiments (rates per ms)
RHO = NU = 20.0 / 1000.0
DRIFT_M, DRIFT_T = 4000, 100_000.0
STAT_M, STAT_AP, STAT_RATIO, STAT_W0 = 2000, 0.002, 1.01, 0.2
STAT_BURN_TR, STAT_SNAP_TR, STAT_N_SNAP = 6.0, 0.5, 24   # in relaxation times

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- model
def window(s, Ap, Am, tp, tm):
    """Pair window W(s), s = t_post - t_pre."""
    s = np.asarray(s, float)
    a = np.abs(s)
    return np.where(s > 0, Ap * np.exp(-a / tp), -Am * np.exp(-a / tm))


def stdp_trains(pre, post, w0, Ap, Am, tp, tm, rule="add", wmax=None):
    """Event-driven trace STDP for one synapse; returns the final weight."""
    ev = sorted([(t, 0) for t in pre] + [(t, 1) for t in post])
    w, x, y, tl = float(w0), 0.0, 0.0, 0.0
    for t, is_post in ev:
        d = t - tl
        x *= math.exp(-d / tp)
        y *= math.exp(-d / tm)
        tl = t
        if is_post:
            w += Ap * x * ((wmax - w) if rule == "soft" else 1.0)
            y += 1.0
        else:
            w -= Am * y * (w if rule == "soft" else 1.0)
            x += 1.0
        if rule == "add" and wmax is not None:
            w = min(max(w, 0.0), wmax)
    return w


def pair_sum(pre, post, Ap, Am, tp, tm):
    """Exact all-to-all pair sum (additive, unbounded)."""
    d = np.asarray(post)[:, None] - np.asarray(pre)[None, :]
    return Ap * np.exp(-d[d > 0] / tp).sum() - Am * np.exp(d[d < 0] / tm).sum()


def open_loop(rng, M, n_events, Ap, Am, tp, tm, rho, nu, w0=0.0, wmax=None,
              t_stop=None, snaps=()):
    """M independent synapses, each with independent Poisson pre (rho) and
    post (nu) trains, generated as one Poisson stream of rate rho + nu with
    each event labelled pre with probability rho/(rho + nu). Events are
    processed in order for all synapses at once (vectorised over synapses).
    Returns final w, event times and snapshots of w before the listed events."""
    lam, p_pre = rho + nu, rho / (rho + nu)
    w = np.full(M, float(w0))
    x = np.zeros(M)
    y = np.zeros(M)
    t = np.zeros(M)
    snaps = set(snaps)
    out = []
    block = 1000
    k = 0
    while k < n_events:
        nb = min(block, n_events - k)
        dts = rng.exponential(1.0 / lam, (nb, M))
        ispre = rng.random((nb, M)) < p_pre
        for j in range(nb):
            if k in snaps:
                out.append(w.copy())
            d = dts[j]
            t += d
            x *= np.exp(-d / tp)
            y *= np.exp(-d / tm)
            pre = ispre[j]
            dw = np.where(pre, -Am * y, Ap * x)
            if t_stop is not None:
                dw[t > t_stop] = 0.0
            w += dw
            if wmax is not None:
                np.clip(w, 0.0, wmax, out=w)
            x += pre
            y += ~pre
            k += 1
    return w, t, out


def song_lif(rng, T=SONG_T, dt=DT):
    """Closed-loop LIF with 1000 plastic excitatory synapses (Song et al. 2000)."""
    n = int(round(T / dt))
    w = rng.uniform(0.0, G_MAX, N_E)
    w_init = w.copy()
    xv = np.zeros(N_E)              # pre trace value at last pre spike
    tl = np.zeros(N_E)              # time of last pre spike
    y, t_post = 0.0, 0.0
    V, ge, gi = V_REST, 0.0, 0.0
    de, di = math.exp(-dt / TAU_EX), math.exp(-dt / TAU_IN)
    k_mem = dt / TAU_MEM
    amg, apg = A_M * G_MAX, A_P * G_MAX
    post_times = []
    rec_t, rec_w = [], []
    rec_every = int(round(1000.0 / dt))
    chunk = 200_000
    exp = math.exp
    for c0 in range(0, n, chunk):
        nc = min(chunk, n - c0)
        ce = rng.poisson(N_E * R_E * dt / 1000.0, nc)
        idx = rng.integers(0, N_E, ce.sum())
        ci = rng.poisson(N_I * R_I * dt / 1000.0, nc)
        p = 0
        for j in range(nc):
            s = c0 + j
            t = s * dt
            c = ce[j]
            if c:
                yy = y * exp(-(t - t_post) / TAU_M)
                for a in idx[p:p + c]:
                    wa = w[a]
                    ge += wa
                    wa -= amg * yy
                    w[a] = wa if wa > 0.0 else 0.0
                    xv[a] = xv[a] * exp(-(t - tl[a]) / TAU_P) + 1.0
                    tl[a] = t
                p += c
            if ci[j]:
                gi += G_IN * ci[j]
            V += k_mem * (V_REST - V + ge * (E_EX - V) + gi * (E_IN - V))
            ge *= de
            gi *= di
            if V >= V_TH:
                V = V_RESET
                w += apg * xv * np.exp(-(t - tl) / TAU_P)
                np.minimum(w, G_MAX, out=w)
                y = y * exp(-(t - t_post) / TAU_M) + 1.0
                t_post = t
                post_times.append(t)
            if s % rec_every == 0:
                rec_t.append(t)
                rec_w.append(w.mean())
    return dict(w=w, w_init=w_init, post=np.array(post_times),
                rec_t=np.array(rec_t), rec_w=np.array(rec_w))


# ---------------------------------------------------------------- closed forms
def drift_mean(T, rho, nu, Ap, Am, tp, tm):
    return rho * nu * (Ap * tp * (T - tp * (1 - math.exp(-T / tp)))
                       - Am * tm * (T - tm * (1 - math.exp(-T / tm))))


def drift_diffusion(rho, nu, Ap, Am, tp, tm):
    kbar = Ap * tp - Am * tm
    mu = rho * nu * kbar
    D = rho * nu * (Ap**2 * tp + Am**2 * tm) / 2 + rho * nu * (rho + nu) * kbar**2
    return mu, D


def trunc_exp_cdf(w, lam, wmax):
    return (1 - np.exp(-w / lam)) / (1 - math.exp(-wmax / lam))


def trunc_exp_mean(lam, wmax):
    return lam - wmax / math.expm1(wmax / lam)


# ---------------------------------------------------------------- simulate
def simulate():
    ss = np.random.SeedSequence(SEED)
    rng_trains, rng_drift, rng_stat, rng_song = [np.random.default_rng(s)
                                                 for s in ss.spawn(4)]
    R = {}
    tt = _time.time()
    # 1. pairing protocol
    pre = 100.0 + BP_PERIOD * np.arange(BP_N_PAIRS)
    add, soft, direct = [], [], []
    for s in BP_LAGS:
        post = pre + s
        add.append(stdp_trains(pre, post, BP_W0, BP_A_P, BP_A_M, BP_TAU_P, BP_TAU_M))
        soft.append(stdp_trains(pre, post, BP_W0, BP_A_P, BP_A_M, BP_TAU_P,
                                BP_TAU_M, rule="soft", wmax=BP_WMAX))
        direct.append(BP_W0 + pair_sum(pre, post, BP_A_P, BP_A_M, BP_TAU_P, BP_TAU_M))
    R["bp_add"], R["bp_soft"], R["bp_direct"] = map(np.array, (add, soft, direct))

    # 2. Poisson trains: trace vs direct sum
    errs, sums = [], []
    for _ in range(5):
        Tn = 20_000.0
        p_ = np.sort(rng_trains.uniform(0, Tn, rng_trains.poisson(RHO * Tn)))
        q_ = np.sort(rng_trains.uniform(0, Tn, rng_trains.poisson(NU * Tn)))
        wt = stdp_trains(p_, q_, 0.0, A_P, A_M, TAU_P, TAU_M)
        wd = pair_sum(p_, q_, A_P, A_M, TAU_P, TAU_M)
        sums.append(wd)
        errs.append(abs(wt - wd) / np.abs(window(q_[:, None] - p_[None, :], A_P,
                                                 A_M, TAU_P, TAU_M)).sum())
    R["train_err"], R["train_sum"] = np.array(errs), np.array(sums)
    print(f"  pairing + trains: {_time.time() - tt:.1f} s")

    # 3. drift and diffusion (unbounded additive)
    tt = _time.time()
    lam_ev = (RHO + NU) * DRIFT_T
    n_ev = int(lam_ev + 8 * math.sqrt(lam_ev)) + 10
    w, t, _ = open_loop(rng_drift, DRIFT_M, n_ev, A_P, A_M, TAU_P, TAU_M,
                        RHO, NU, t_stop=DRIFT_T)
    assert t.min() > DRIFT_T, "drift run did not reach T on every synapse"
    R["drift_w"] = w
    print(f"  drift/diffusion: {_time.time() - tt:.1f} s")

    # 4. stationary distribution (hard bounds)
    tt = _time.time()
    sap, sam = STAT_AP, STAT_RATIO * STAT_AP
    mu, D = drift_diffusion(RHO, NU, sap, sam, TAU_P, TAU_M)
    ev_tr = 2 * D / mu**2 * (RHO + NU)            # events per relaxation time
    snaps = [int(round((STAT_BURN_TR + i * STAT_SNAP_TR) * ev_tr))
             for i in range(STAT_N_SNAP)]
    n_ev = snaps[-1] + 1
    R["stat_snap_idx"] = np.array(snaps)
    w, t, out = open_loop(rng_stat, STAT_M, n_ev, sap, sam, TAU_P, TAU_M,
                          RHO, NU, w0=STAT_W0, wmax=1.0, snaps=snaps)
    R["stat_snaps"] = np.array(out)
    R["stat_t_end"] = t.mean()
    print(f"  stationary distribution: {_time.time() - tt:.1f} s")

    # 5. closed loop
    tt = _time.time()
    R["song"] = song_lif(rng_song)
    print(f"  Song et al. closed loop: {_time.time() - tt:.1f} s")
    return R


# ---------------------------------------------------------------- validate
def validate(R):
    V = {}
    print("\n1. Bi & Poo pairing protocol (60 pairs at 1 Hz)")
    lags = BP_LAGS
    cf_add = BP_W0 + BP_N_PAIRS * window(lags, BP_A_P, BP_A_M, BP_TAU_P, BP_TAU_M)
    eta_p = BP_A_P * np.exp(-lags / BP_TAU_P)
    eta_m = BP_A_M * np.exp(lags / BP_TAU_M)
    cf_soft = np.where(lags > 0,
                       BP_WMAX - (BP_WMAX - BP_W0) * (1 - eta_p) ** BP_N_PAIRS,
                       BP_W0 * (1 - eta_m) ** BP_N_PAIRS)
    e_add = np.abs(R["bp_add"] - cf_add).max()
    e_dir = np.abs(R["bp_add"] - R["bp_direct"]).max()
    e_soft = np.abs(R["bp_soft"] - cf_soft).max()
    V.update(cf_add=cf_add, cf_soft=cf_soft, e_bp_add=e_add, e_bp_dir=e_dir,
             e_bp_soft=e_soft)
    print(f"   additive vs 60 W(s): max |err| {e_add:.2e}; vs exact pair sum {e_dir:.2e}")
    print(f"   soft-bound vs 60-fold map: max |err| {e_soft:.2e}  (tolerance 1e-9)")
    i2, im2 = np.argmin(np.abs(lags - 2)), np.argmin(np.abs(lags + 2))
    print(f"   change at s = +2 ms: additive {100*(R['bp_add'][i2]-1):+.1f}%, "
          f"soft {100*(R['bp_soft'][i2]-1):+.1f}%; at s = -2 ms: additive "
          f"{100*(R['bp_add'][im2]-1):+.1f}%, soft {100*(R['bp_soft'][im2]-1):+.1f}%")
    assert max(e_add, e_dir, e_soft) < 1e-9

    print("\n2. Trace implementation vs exact all-to-all pair sum, 5 Poisson train pairs (20 Hz, 20 s)")
    e_tr = R["train_err"].max()
    V["e_trains"] = e_tr
    print(f"   dw = {np.array2string(R['train_sum'], precision=4)}; "
          f"max error / sum|W| {e_tr:.2e} (tolerance 1e-9)")
    assert e_tr < 1e-9

    print("\n3. Drift and diffusion, independent Poisson pre/post at 20 Hz, T = 100 s, M = 4000")
    w = R["drift_w"]
    M = w.size
    m_th = drift_mean(DRIFT_T, RHO, NU, A_P, A_M, TAU_P, TAU_M)
    mu, D = drift_diffusion(RHO, NU, A_P, A_M, TAU_P, TAU_M)
    v_th = D * DRIFT_T
    m_s, v_s = w.mean(), w.var(ddof=1)
    z_m = (m_s - m_th) / math.sqrt(v_s / M)
    z_v = (v_s - v_th) / (v_s * math.sqrt(2 / (M - 1)))
    V.update(m_th=m_th, v_th=v_th, m_s=m_s, v_s=v_s, z_m=z_m, z_v=z_v)
    print(f"   mean dw: {m_s:.5f} vs {m_th:.5f} ({100*(m_s/m_th-1):+.2f}%, z = {z_m:+.2f}); "
          f"mu T = {mu*DRIFT_T:.5f}")
    print(f"   var dw:  {v_s:.6f} vs D T = {v_th:.6f} ({100*(v_s/v_th-1):+.2f}%, z = {z_v:+.2f}); "
          f"cross term share {RHO*NU*(RHO+NU)*(A_P*TAU_P-A_M*TAU_M)**2/D:.1e}")
    assert abs(z_m) < 4 and abs(z_v) < 4

    print("\n4. Stationary distribution, hard bounds [0, 1], A+ = 0.002, A-/A+ = 1.01")
    sap, sam = STAT_AP, STAT_RATIO * STAT_AP
    mu, D = drift_diffusion(RHO, NU, sap, sam, TAU_P, TAU_M)
    lam = D / (2 * abs(mu))
    mean_th = trunc_exp_mean(lam, 1.0)
    pooled = R["stat_snaps"].ravel()
    snap_means = R["stat_snaps"].mean(axis=1)
    mean_s = pooled.mean()
    rel = mean_s / mean_th - 1
    se = R["stat_snaps"].mean(axis=0).std(ddof=1) / math.sqrt(R["stat_snaps"].shape[1])
    t_r = 2 * D / mu**2
    xs = np.sort(pooled)
    F = trunc_exp_cdf(xs, lam, 1.0)
    n = xs.size
    ks = max(np.max(np.arange(1, n + 1) / n - F), np.max(F - np.arange(n) / n))
    V.update(lam=lam, mean_th=mean_th, mean_s=mean_s, rel_stat=rel, ks=ks,
             snap_means=snap_means, se_stat=se)
    print(f"   mu = {mu*1000:.3e} /s, D = {D*1000:.3e} /s, lam = D/(2|mu|) = {lam:.4f} "
          f"({lam/sap:.1f} potentiation jumps); relaxation lam^2/D = {lam**2/D/1000:.0f} s")
    print(f"   relaxation time 2D/mu^2 = {t_r/1000:.0f} s; {STAT_N_SNAP} snapshots from "
          f"{R['stat_snap_idx'][0]/(RHO+NU)/1000:.0f} s to {R['stat_t_end']/1000:.0f} s; "
          f"snapshot means {snap_means.min():.4f}-{snap_means.max():.4f}")
    print(f"   mean weight {mean_s:.4f} +- {se:.4f} vs Fokker-Planck {mean_th:.4f} "
          f"({100*rel:+.2f}%, z = {(mean_s-mean_th)/se:+.2f}; tolerance 5%)")
    print(f"   KS distance {ks:.4f} (tolerance 0.03); fraction at w = 0: {np.mean(pooled == 0):.4f}")
    assert abs(rel) < 0.05 and ks < 0.03

    print("\n5. Closed loop (Song, Miller & Abbott 2000), 1000 synapses at 10 Hz, 2000 s")
    S = R["song"]
    wf = S["w"] / G_MAX
    lo, hi = np.mean(wf < 0.1), np.mean(wf > 0.9)
    post = S["post"]
    r_first = np.sum(post < 10_000) / 10.0
    r_last = np.sum(post > SONG_T - 500_000) / 500.0
    V.update(song_lo=lo, song_hi=hi, r_first=r_first, r_last=r_last)
    print(f"   final fraction w < 0.1 g_max: {lo:.3f}, w > 0.9 g_max: {hi:.3f}, "
          f"outer 20%: {lo+hi:.3f} (assert > 0.6; uniform start {np.mean((S['w_init']/G_MAX<0.1)|(S['w_init']/G_MAX>0.9)):.3f})")
    print(f"   mean weight {S['w_init'].mean()/G_MAX:.3f} -> {wf.mean():.3f} g_max; "
          f"output rate {r_first:.1f} Hz (first 10 s) -> {r_last:.1f} Hz (last 500 s)")
    assert lo + hi > 0.6
    print("\nAll checks passed.")
    return V


# ---------------------------------------------------------------- plot
def plot(R, V):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    lags = BP_LAGS
    sf = np.linspace(-100, 100, 2001)
    sf = sf[sf != 0]

    a = ax[0, 0]
    cf_line = 100 * BP_N_PAIRS * window(sf, BP_A_P, BP_A_M, BP_TAU_P, BP_TAU_M)
    a.plot(sf[sf < 0], cf_line[sf < 0], "k-", lw=1, label="closed form 60 W(s)")
    a.plot(sf[sf > 0], cf_line[sf > 0], "k-", lw=1)
    a.plot(lags, 100 * (R["bp_add"] - 1), "o", ms=4, color="C0", label="additive (traces)")
    eta = np.where(sf > 0, BP_A_P * np.exp(-sf / BP_TAU_P), BP_A_M * np.exp(sf / BP_TAU_M))
    soft_line = np.where(sf > 0, BP_WMAX - (BP_WMAX - 1) * (1 - eta) ** 60, (1 - eta) ** 60)
    a.plot(sf[sf < 0], 100 * (soft_line[sf < 0] - 1), "--", color="C3", lw=1, label="soft-bound 60-fold map")
    a.plot(sf[sf > 0], 100 * (soft_line[sf > 0] - 1), "--", color="C3", lw=1)
    a.plot(lags, 100 * (R["bp_soft"] - 1), "s", ms=3, mfc="none", color="C3", label="soft-bound (traces)")
    a.axhline(0, color="0.7", lw=0.6)
    a.axvline(0, color="0.7", lw=0.6)
    a.set_xlabel("spike timing s = t_post − t_pre (ms)")
    a.set_ylabel("weight change after 60 pairs at 1 Hz (%)")
    a.set_title(f"A  Bi & Poo protocol, τ+ = {BP_TAU_P}, τ− = {BP_TAU_M} ms")
    a.legend(fontsize=8, loc="upper left")

    a = ax[0, 1]
    a.semilogy(lags, np.abs(R["bp_add"] - V["cf_add"]) + 1e-18, "o", ms=3, label="additive vs 60 W(s)")
    a.semilogy(lags, np.abs(R["bp_add"] - R["bp_direct"]) + 1e-18, "x", ms=4, label="additive vs exact pair sum")
    a.semilogy(lags, np.abs(R["bp_soft"] - V["cf_soft"]) + 1e-18, "s", ms=3, mfc="none", label="soft-bound vs 60-fold map")
    a.axhline(1e-9, color="k", ls=":", lw=1, label="tolerance 1e-9")
    a.set_ylim(1e-18, 1e-6)
    a.set_xlabel("s (ms)")
    a.set_ylabel("|error| in w/w0")
    a.set_title(f"B  Protocol errors; Poisson trains vs pair sum {V['e_trains']:.1e}")
    a.legend(fontsize=8, loc="upper right")

    a = ax[0, 2]
    w = R["drift_w"]
    a.hist(w, bins=60, density=True, color="C0", alpha=0.6, label=f"simulated, M = {w.size}")
    g = np.linspace(w.min(), w.max(), 400)
    sd = math.sqrt(V["v_th"])
    a.plot(g, np.exp(-0.5 * ((g - V["m_th"]) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
           "k-", lw=1.2, label="N(E[Δw], D T) closed form")
    a.axvline(V["m_th"], color="k", ls=":", lw=1)
    a.axvline(0, color="0.7", lw=0.6)
    a.set_xlabel("Δw after T = 100 s (units of w_max)")
    a.set_ylabel("density")
    a.set_title(f"C  Drift/diffusion, 20 Hz Poisson: mean z = {V['z_m']:+.2f}, var z = {V['z_v']:+.2f}")
    a.legend(fontsize=8)

    a = ax[1, 0]
    pooled = R["stat_snaps"].ravel()
    a.hist(pooled, bins=np.linspace(0, 0.8, 81), density=True, color="C2", alpha=0.6,
           label=f"simulated ({STAT_N_SNAP} snapshots × {STAT_M})")
    g = np.linspace(0, 1, 500)
    lam = V["lam"]
    a.plot(g, np.exp(-g / lam) / (lam * (1 - math.exp(-1 / lam))), "k-", lw=1.2,
           label=f"Fokker–Planck, λ = D/2|μ| = {lam:.3f}")
    a.set_yscale("log")
    a.set_ylim(1e-3, 30)
    a.set_xlim(0, 0.8)
    a.set_xlabel("w / w_max")
    a.set_ylabel("density")
    a.set_title(f"D  Stationary, additive hard bounds: mean {V['mean_s']:.4f} vs {V['mean_th']:.4f}, KS {V['ks']:.3f}")
    a.legend(fontsize=8)

    S = R["song"]
    a = ax[1, 1]
    bins = np.linspace(0, 1, 41)
    a.hist(S["w_init"] / G_MAX, bins=bins, histtype="step", color="0.5", label="initial (uniform)")
    a.hist(S["w"] / G_MAX, bins=bins, color="C1", alpha=0.7, label=f"after {SONG_T/1000:.0f} s")
    a.set_xlabel("w / g_max")
    a.set_ylabel("number of synapses")
    a.set_title(f"E  LIF closed loop (Song et al. 2000): {100*(V['song_lo']+V['song_hi']):.0f}% in outer 20%")
    a.legend(fontsize=8)

    a = ax[1, 2]
    a.plot(S["rec_t"] / 1000, S["rec_w"] / G_MAX, color="C1")
    a.set_xlabel("time (s)")
    a.set_ylabel("mean w / g_max", color="C1")
    b = a.twinx()
    edges = np.arange(0, SONG_T + 1, 20_000.0)
    cnt, _ = np.histogram(S["post"], edges)
    b.plot(edges[:-1] / 1000 + 10, cnt / 20.0, color="C4", lw=1)
    b.set_ylabel("output rate (Hz, 20 s bins)", color="C4")
    a.set_title(f"F  Output rate {V['r_first']:.1f} → {V['r_last']:.1f} Hz")

    for a_ in ax.flat:
        a_.title.set_fontsize(10)
    fig.suptitle("Pair-based STDP: Bi & Poo window, Poisson drift/diffusion, stationary weights, "
                 "and competition in a LIF neuron", fontsize=13)
    fig.tight_layout()
    out = ROOT / "figures" / "stdp_pair.png"
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print(f"RNG seed: {SEED} (PCG64, one spawned stream per experiment)")
    t0 = _time.time()
    R = simulate()
    V = validate(R)
    plot(R, V)
    print(f"Runtime {_time.time() - t0:.1f} s")
