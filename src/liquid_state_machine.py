"""Liquid state machine: a column of leaky integrate-and-fire neurons with
dynamic synapses, read out linearly on a jittered spike-pattern task.

Reservoir (the generic microcircuit of Maass, Natschlaeger & Markram 2002,
"Real-time computing without stable states", Neural Computation 14,
2531-2560, Appendix), 135 LIF neurons on the integer points of a 15 x 3 x 3
grid, 20% of them inhibitory, assigned at random:

    tau_m dV/dt = -V + R_m (I_b + I_syn(t)),
    V >= V_thresh  ->  spike, V <- V_reset, held there for t_ref,

tau_m = 30 ms, R_m = 1 MOhm, V_thresh = 15 mV, V_reset = 13.5 mV,
I_b = 13.5 nA, t_ref = 3 ms (E) and 2 ms (I), V(0) ~ U[13.5, 15] mV
independently on every trial. Neuron a -> b is connected with probability
C * exp(-(D(a, b) / lambda)^2), D the Euclidean grid distance, lambda = 2,
C = 0.3 (EE), 0.2 (EI), 0.4 (IE), 0.1 (II). Synaptic currents are
exponential, tau_s = 3 ms (excitatory) and 6 ms (inhibitory). Transmission
delay 1.5 ms for E -> E and 0.8 ms otherwise. Each synapse is a
Tsodyks-Markram dynamic synapse (Markram, Wang & Tsodyks 1998): the n-th
spike of the presynaptic neuron produces a current jump A u_n R_n with

    u_1 = U, R_1 = 1,
    u_{n+1} = U + u_n (1 - U) exp(-Delta_n / F),
    R_{n+1} = 1 + (R_n - u_n R_n - 1) exp(-Delta_n / D),

Delta_n the n-th inter-spike interval. Mean (U, D [s], F [s]) =
(0.5, 1.1, 0.05) EE, (0.05, 0.125, 1.2) EI, (0.25, 0.7, 0.02) IE,
(0.32, 0.144, 0.06) II. Mean amplitudes A = 30 (EE), 60 (EI), -19 (IE),
-19 (II) nA. Here U, D, F are set to these means for every synapse of a
class (the paper draws them per synapse), and A is Gamma-distributed with
CV 0.5 around the mean (sign fixed by the presynaptic type).

Input: 4 spike trains. Each input channel projects to each reservoir neuron
with probability 0.3 through a static synapse (amplitude Gamma, CV 0.5,
mean 18 nA onto E and 9 nA onto I, tau_s = 3 ms).

Task (the segment classification task of Maass et al. 2002, Fig. 3): the
input is 200 ms long, cut into four 50 ms segments. For segment k there are
two fixed templates, 4 x 50 ms Poisson spike trains at 20 Hz; on each trial
every segment shows one of its two templates, chosen by an independent
fair coin b_k, and every spike is jittered by Gaussian noise, sd 4 ms.
Readouts are trained to report b_1 ... b_4 and the nonlinear b_3 XOR b_4
from the reservoir state at time t. The state is the reservoir spike trains
low-pass filtered with an exponential kernel, tau_r = 30 ms (as in the
paper), sampled every 10 ms from 0 to 250 ms. The readout is linear: ridge
regression of the +-1 label on the z-scored state plus a bias, sign as the
decision, ridge parameter chosen by 5-fold cross-validation on the training
set. 1000 training and 500 held-out test trials, independent jitter and
initial voltages.

Integrator: exponential Euler at dt = 0.1 ms. Synaptic currents decay
exactly over a step; V is propagated exactly over a step with the current
held at its start-of-step value; the threshold is tested on the grid. All
trials run in parallel as rows of one array.

Validation (ROADMAP note: readout accuracy above chance with a held-out set;
compare with a shuffled-label control):
(1) Held-out accuracy of each readout against chance (0.5) with an exact
    one-sided binomial test, asserted p < 1e-3 for all five tasks at
    t = 200 ms (end of the input).
(2) Shuffled-label control: the same readout pipeline, ridge CV included,
    trained on 200 random permutations of the training labels and scored on
    the true test labels. Permutation test (Ojala & Garriga 2010): p_perm =
    (1 + #{shuffled >= true}) / (1 + 200), asserted < 0.01 for every task;
    and the pooled shuffled mean must be within 0.01 of 0.5. (The spread of
    the shuffled accuracies is reported against the binomial sd 0.5 /
    sqrt(n_test) but not asserted: the binomial is the null for predictions
    fixed independently of the inputs, and a readout trained on shuffled
    labels is a random direction in a state space that encodes the labels,
    so its accuracy is more widely spread. See README Limitations.)
(3) An input-only control: the same readout on the 4 input channels
    filtered the same way (not asserted; shows what the reservoir adds).
(4) Components against closed forms: the neuron update on the grid against
    the LIF rate 1 / (t_ref + tau_m ln((V_inf - V_reset) / (V_inf -
    V_thresh))) (tolerance 2%; and the grid-rounded interval exactly), and
    the dynamic synapse under a regular train against the steady state
    u* = U / (1 - (1 - U) e^{-Delta/F}),
    R* = (1 - e^{-Delta/D}) / (1 - (1 - u*) e^{-Delta/D}) (tolerance 1e-9).

RNG: NumPy PCG64 seeded with SEED (printed): network, templates, jitter,
initial voltages, permutations.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import binom, binomtest       # validation only

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
SEED = 2002
DT = 0.1                      # ms
GRID = (15, 3, 3)
N = GRID[0] * GRID[1] * GRID[2]
FRAC_I = 0.2
TAU_M = 30.0                  # ms
R_M = 1.0                     # MOhm (nA * MOhm = mV)
V_TH, V_RESET = 15.0, 13.5    # mV
I_B = 13.5                    # nA
T_REF = {"E": 3.0, "I": 2.0}  # ms
TAU_S = {"E": 3.0, "I": 6.0}  # ms, by presynaptic type
LAMBDA = 2.0
C_CONN = {"EE": 0.3, "EI": 0.2, "IE": 0.4, "II": 0.1}
A_MEAN = {"EE": 30.0, "EI": 60.0, "IE": -19.0, "II": -19.0}   # nA
UDF = {"EE": (0.5, 1100.0, 50.0), "EI": (0.05, 125.0, 1200.0),
       "IE": (0.25, 700.0, 20.0), "II": (0.32, 144.0, 60.0)}  # U, D ms, F ms
DELAY = {"EE": 1.5, "EI": 0.8, "IE": 0.8, "II": 0.8}         # ms
W_CV = 0.5

N_IN = 4
P_IN = 0.3
A_IN = {"E": 18.0, "I": 9.0}  # nA

N_SEG, T_SEG = 4, 50.0        # ms
T_INPUT = N_SEG * T_SEG
T_TOTAL = 250.0
RATE_TEMPLATE = 20.0          # Hz
JITTER = 4.0                  # ms
TAU_R = 30.0                  # ms, readout filter
T_READ = np.arange(0.0, T_TOTAL + 1e-9, 10.0)
T_MAIN = 200.0

N_TRAIN, N_TEST = 1000, 500
N_PERM = 200
LAMBDAS = np.logspace(-2, 4, 13)
N_FOLD = 5
TASKS = ["segment 1", "segment 2", "segment 3", "segment 4", "XOR 3,4"]

P_MAX = 1e-3                  # (1) binomial p-value
TOL_SHUF_MEAN = 0.01          # (2)
P_PERM_MAX = 0.01             # (2)
TOL_RATE = 0.02               # (4)
TOL_TM = 1e-9                 # (4)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def gamma_weights(rng, mean, size):
    k = 1.0 / W_CV ** 2
    return np.sign(mean) * rng.gamma(k, abs(mean) / k, size)


def build_network(rng):
    """Grid positions, types (E first), weights, input weights, synapse params."""
    pos = np.array([(x, y, z) for x in range(GRID[0]) for y in range(GRID[1])
                    for z in range(GRID[2])], float)
    perm = rng.permutation(N)
    n_i = int(round(FRAC_I * N))
    n_e = N - n_i
    pos = pos[perm]                       # first n_e are E, rest I
    typ = np.array(["E"] * n_e + ["I"] * n_i)
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    W = np.zeros((N, N))                  # W[pre, post]
    for a in "EI":
        for b in "EI":
            pre = typ == a
            post = typ == b
            p = C_CONN[a + b] * np.exp(-(D[np.ix_(pre, post)] / LAMBDA) ** 2)
            conn = rng.random(p.shape) < p
            blk = np.where(conn, gamma_weights(rng, A_MEAN[a + b], p.shape), 0.0)
            W[np.ix_(pre, post)] = blk
    np.fill_diagonal(W, 0.0)
    W_in = np.zeros((N_IN, N))
    for b in "EI":
        post = typ == b
        conn = rng.random((N_IN, post.sum())) < P_IN
        W_in[:, post] = np.where(conn, gamma_weights(rng, A_IN[b], conn.shape), 0.0)
    # dynamic-synapse parameters per presynaptic neuron, one set per target type
    syn = {}
    for b in "EI":
        U = np.array([UDF[t + b][0] for t in typ])
        Dd = np.array([UDF[t + b][1] for t in typ])
        F = np.array([UDF[t + b][2] for t in typ])
        syn[b] = (U, Dd, F)
    t_ref = np.array([T_REF[t] for t in typ])
    return dict(pos=pos, typ=typ, n_e=n_e, W=W, W_in=W_in, syn=syn, t_ref=t_ref)


def lif_step(V, ref, I, t_ref, dt=DT):
    """One exponential-Euler step of the LIF neuron (shared by the network and
    the closed-form check). ref: remaining refractory time (ms)."""
    v_inf = R_M * (I_B + I)
    V = v_inf + (V - v_inf) * np.exp(-dt / TAU_M)
    refr = ref > 1e-9
    V = np.where(refr, V_RESET, V)
    ref = np.maximum(ref - dt, 0.0)
    spk = V >= V_TH
    V = np.where(spk, V_RESET, V)
    ref = np.where(spk, t_ref, ref)
    return V, ref, spk


def tm_update(spk, t, last, u, R, U, Dd, F):
    """Tsodyks-Markram efficacy u_n R_n for the spikes in spk at time t."""
    first = ~np.isfinite(last)
    delta = np.where(first, 1.0, t - last)
    u_new = np.where(first, U, U + u * (1 - U) * np.exp(-delta / F))
    R_new = np.where(first, 1.0, 1 + (R - u * R - 1) * np.exp(-delta / Dd))
    u = np.where(spk, u_new, u)
    R = np.where(spk, R_new, R)
    return np.where(spk, u * R, 0.0), u, R


def make_templates(rng):
    """templates[k][c] = list over channels of spike-time arrays in [0, T_SEG)."""
    tpl = []
    for k in range(N_SEG):
        pair = []
        for c in range(2):
            while True:
                tr = [np.sort(rng.uniform(0, T_SEG, rng.poisson(RATE_TEMPLATE * T_SEG / 1000)))
                      for _ in range(N_IN)]
                if sum(len(x) for x in tr) >= 2:
                    break
            pair.append(tr)
        tpl.append(pair)
    return tpl


def make_trials(rng, tpl, n):
    """Bits (n, N_SEG) and input spike indicator (n_steps, n, N_IN)."""
    n_steps = int(round(T_TOTAL / DT))
    bits = rng.integers(0, 2, (n, N_SEG))
    inp = np.zeros((n_steps, n, N_IN), bool)
    for i in range(n):
        for k in range(N_SEG):
            for ch, st in enumerate(tpl[k][bits[i, k]]):
                t = st + k * T_SEG + rng.normal(0, JITTER, len(st))
                t = t[(t >= 0) & (t < T_INPUT)]
                inp[(t / DT).astype(int), i, ch] = True
    return bits, inp


def labels(bits):
    y = np.column_stack([bits[:, 0], bits[:, 1], bits[:, 2], bits[:, 3],
                         bits[:, 2] ^ bits[:, 3]])
    return 2.0 * y - 1.0


def simulate(net, inp, rng, keep_raster=1):
    """Run all trials in parallel. Returns filtered reservoir state at T_READ
    (n_trials, len(T_READ), N), filtered input at T_READ, rasters of the first
    keep_raster trials, and spike counts."""
    n_steps, n, _ = inp.shape
    n_e = net["n_e"]
    W = net["W"]
    W_EE, W_EI = W[:n_e, :n_e], W[:n_e, n_e:]
    W_IE, W_II = W[n_e:, :n_e], W[n_e:, n_e:]
    d_ee = int(round(DELAY["EE"] / DT))
    d_oth = int(round(DELAY["EI"] / DT))
    L = max(d_ee, d_oth) + 1
    buf_e = np.zeros((L, n, N))
    buf_i = np.zeros((L, n, N))
    dec_e, dec_i = np.exp(-DT / TAU_S["E"]), np.exp(-DT / TAU_S["I"])
    dec_r = np.exp(-DT / TAU_R)
    V = rng.uniform(V_RESET, V_TH, (n, N))
    ref = np.zeros((n, N))
    I_e = np.zeros((n, N))
    I_i = np.zeros((n, N))
    last = np.full((n, N), -np.inf)
    st = {b: [np.zeros((n, N)), np.ones((n, N))] for b in "EI"}
    r = np.zeros((n, N))
    r_in = np.zeros((n, N_IN))
    read_steps = set(int(round(t / DT)) for t in T_READ)
    step_to_idx = {int(round(t / DT)): j for j, t in enumerate(T_READ)}
    X = np.zeros((n, len(T_READ), N))
    X_in = np.zeros((n, len(T_READ), N_IN))
    raster = [[] for _ in range(keep_raster)]
    count = np.zeros(N)
    for k in range(n_steps + 1):
        if k in read_steps:
            X[:, step_to_idx[k]] = r
            X_in[:, step_to_idx[k]] = r_in
        if k == n_steps:
            break
        t = k * DT
        slot = k % L
        I_e = I_e * dec_e + buf_e[slot]
        I_i = I_i * dec_i + buf_i[slot]
        buf_e[slot] = 0.0
        buf_i[slot] = 0.0
        x_in = inp[k].astype(float)
        if x_in.any():
            I_e += x_in @ net["W_in"]
        V, ref, spk = lif_step(V, ref, I_e + I_i, net["t_ref"])
        r_in = r_in * dec_r + x_in
        r = r * dec_r + spk
        if spk.any():
            count += spk.sum(0)
            for i in range(keep_raster):
                for j in np.flatnonzero(spk[i]):
                    raster[i].append((t + DT, j))
            eff = {}
            for b in "EI":
                U, Dd, F = net["syn"][b]
                eff[b], st[b][0], st[b][1] = tm_update(spk, t, last, st[b][0], st[b][1], U, Dd, F)
            last = np.where(spk, t, last)
            eE, eI = eff["E"], eff["I"]       # efficacy toward E / I targets
            buf_e[(k + d_ee) % L, :, :n_e] += eE[:, :n_e] @ W_EE
            buf_e[(k + d_oth) % L, :, n_e:] += eI[:, :n_e] @ W_EI
            buf_i[(k + d_oth) % L, :, :n_e] += eE[:, n_e:] @ W_IE
            buf_i[(k + d_oth) % L, :, n_e:] += eI[:, n_e:] @ W_II
    return X, X_in, raster, count


# ---------------------------------------------------------------------------
# Readout
# ---------------------------------------------------------------------------
def _zscore(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _ridge_preds(Xtr, Ytr, Xte, lambdas):
    """Predictions on Xte for every lambda and every column of Ytr.
    Bias handled by centring. Returns (len(lambdas), n_te, n_y)."""
    ym = Ytr.mean(0)
    Uu, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = Uu.T @ (Ytr - ym)
    XV = Xte @ Vt.T
    out = np.empty((len(lambdas), Xte.shape[0], Ytr.shape[1]))
    for li, lam in enumerate(lambdas):
        out[li] = XV @ ((s / (s ** 2 + lam))[:, None] * UtY) + ym
    return out


def train_test(Xtr, Ytr, Xte, rng):
    """Ridge with lambda chosen per label column by N_FOLD-fold CV.
    Returns predicted signs on Xte (n_te, n_y)."""
    Xtr, Xte = _zscore(Xtr, Xte)
    n = Xtr.shape[0]
    folds = np.array_split(rng.permutation(n), N_FOLD)
    cv_acc = np.zeros((len(LAMBDAS), Ytr.shape[1]))
    for f in folds:
        m = np.ones(n, bool)
        m[f] = False
        a, b = _zscore(Xtr[m], Xtr[f])
        P = _ridge_preds(a, Ytr[m], b, LAMBDAS)
        cv_acc += (np.sign(P) == Ytr[f][None]).mean(1)
    best = cv_acc.argmax(0)
    P = _ridge_preds(Xtr, Ytr, Xte, LAMBDAS)
    return np.sign(P[best, :, np.arange(Ytr.shape[1])].T), LAMBDAS[best]


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def check_lif():
    """Neuron update vs the closed-form LIF rate (constant drive, no synapses)."""
    I = np.linspace(2.0, 30.0, 15)
    T = 2000.0
    n = int(round(T / DT))
    V = np.full(len(I), V_RESET)
    ref = np.zeros(len(I))
    t_sp = [[] for _ in I]
    for k in range(n):
        V, ref, spk = lif_step(V, ref, I, T_REF["E"])
        for j in np.flatnonzero(spk):
            t_sp[j].append((k + 1) * DT)
    isi = np.array([np.mean(np.diff(s)) for s in t_sp])
    v_inf = R_M * (I_B + I)
    T_sub = TAU_M * np.log((v_inf - V_RESET) / (v_inf - V_TH))
    r_cf = 1000.0 / (T_REF["E"] + T_sub)
    r_sim = 1000.0 / isi
    # on the grid: refractory steps, then the first step at which V >= V_th
    n_ref = int(round(T_REF["E"] / DT))
    isi_grid = DT * (n_ref + np.ceil(T_sub / DT - 1e-9))
    return I, r_sim, r_cf, isi, isi_grid


def check_tm():
    """Dynamic synapse under a regular train vs closed-form steady state."""
    out = []
    for key in ["EE", "EI", "IE", "II"]:
        U, Dd, F = UDF[key]
        for f in [5.0, 20.0, 50.0]:
            dl = 1000.0 / f
            last = np.array([-np.inf])
            u, R = np.zeros(1), np.ones(1)
            for n in range(400):
                e, u, R = tm_update(np.array([True]), n * dl, last, u, R, U, Dd, F)
                last = np.array([n * dl])
            us = U / (1 - (1 - U) * np.exp(-dl / F))
            Rs = (1 - np.exp(-dl / Dd)) / (1 - (1 - us) * np.exp(-dl / Dd))
            out.append((key, f, e[0], us * Rs))
    return out


def validate():
    t0 = _time.time()
    rng = np.random.default_rng(SEED)
    print(f"seed {SEED}, dt = {DT} ms")
    R = {}

    # (4) components
    I, r_sim, r_cf, isi, isi_grid = check_lif()
    e_rate = np.max(np.abs(r_sim / r_cf - 1))
    e_grid = np.max(np.abs(isi - isi_grid))
    print(f"(4) LIF f-I, {len(I)} currents {I[0]:.0f}-{I[-1]:.0f} nA "
          f"({r_cf[0]:.1f}-{r_cf[-1]:.1f} Hz): max rel. error vs closed form "
          f"{e_rate:.4f} (tol {TOL_RATE}); vs grid-rounded interval {e_grid:.2e} ms")
    assert e_rate < TOL_RATE, "LIF rate off closed form"
    assert e_grid < 1e-6, "LIF interval off grid prediction"
    tm = check_tm()
    e_tm = max(abs(e - s) / s for _, _, e, s in tm)
    print(f"(4) Tsodyks-Markram steady state u*R*, 4 classes x 3 rates: "
          f"max rel. error {e_tm:.1e} (tol {TOL_TM})")
    assert e_tm < TOL_TM, "TM steady state"
    R.update(I=I, r_sim=r_sim, r_cf=r_cf, e_rate=e_rate, tm=tm)

    # network and data
    net = build_network(rng)
    tpl = make_templates(rng)
    n_syn = np.count_nonzero(net["W"])
    print(f"network: {net['n_e']} E + {N - net['n_e']} I, {n_syn} synapses "
          f"({n_syn / N:.1f} per neuron), {np.count_nonzero(net['W_in'])} input synapses")
    bits_tr, inp_tr = make_trials(rng, tpl, N_TRAIN)
    bits_te, inp_te = make_trials(rng, tpl, N_TEST)
    t1 = _time.time()
    X_tr, Xin_tr, ras_tr, cnt_tr = simulate(net, inp_tr, rng)
    X_te, Xin_te, ras_te, cnt_te = simulate(net, inp_te, rng)
    del inp_tr
    rate = (cnt_tr + cnt_te).sum() / ((N_TRAIN + N_TEST) * N * T_TOTAL / 1000)
    print(f"simulated {N_TRAIN + N_TEST} trials of {T_TOTAL:.0f} ms in "
          f"{_time.time() - t1:.1f} s; mean reservoir rate {rate:.1f} Hz")
    Y_tr, Y_te = labels(bits_tr), labels(bits_te)

    # (1) accuracy vs readout time, reservoir and input-only
    acc = np.zeros((len(T_READ), len(TASKS)))
    acc_in = np.zeros((len(T_READ), len(TASKS)))
    lam_main = None
    for j, t in enumerate(T_READ):
        P, lam = train_test(X_tr[:, j], Y_tr, X_te[:, j], rng)
        acc[j] = (P == Y_te).mean(0)
        P, _ = train_test(Xin_tr[:, j], Y_tr, Xin_te[:, j], rng)
        acc_in[j] = (P == Y_te).mean(0)
        if np.isclose(t, T_MAIN):
            lam_main = lam
    jm = int(np.argmin(np.abs(T_READ - T_MAIN)))
    print("    accuracy vs readout time (ms): " + ", ".join(TASKS))
    for j, t in enumerate(T_READ):
        print(f"    t = {t:5.0f}  " + "  ".join(f"{a:.3f}" for a in acc[j]))
    n_correct = np.round(acc[jm] * N_TEST).astype(int)
    pvals = np.array([binomtest(int(c), N_TEST, 0.5, alternative="greater").pvalue
                      for c in n_correct])
    print(f"(1) held-out accuracy at t = {T_MAIN:.0f} ms, n_test = {N_TEST}:")
    for k, name in enumerate(TASKS):
        print(f"    {name:10s} reservoir {acc[jm, k]:.3f} (p = {pvals[k]:.1e}, "
              f"ridge {lam_main[k]:.2g}), input-only {acc_in[jm, k]:.3f}")
    # training accuracy of the reservoir readout at T_MAIN, for overfitting
    P, _ = train_test(X_tr[:, jm], Y_tr, X_tr[:, jm], rng)
    acc_train = (P == Y_tr).mean(0)
    print("    training accuracy " + ", ".join(f"{a:.3f}" for a in acc_train))

    # (2) shuffled-label control at T_MAIN
    acc_shuf = np.zeros((N_PERM, len(TASKS)))
    for p in range(N_PERM):
        Ysh = Y_tr[rng.permutation(N_TRAIN)]
        P, _ = train_test(X_tr[:, jm], Ysh, X_te[:, jm], rng)
        acc_shuf[p] = (P == Y_te).mean(0)
    sd_binom = 0.5 / np.sqrt(N_TEST)
    m_sh, s_sh = acc_shuf.mean(), acc_shuf.std()
    q95 = binom.ppf(0.95, N_TEST, 0.5) / N_TEST
    n_ge = (acc_shuf >= acc[jm][None, :] - 1e-12).sum(0)
    p_perm = (1 + n_ge) / (1 + N_PERM)
    print(f"(2) shuffled labels, {N_PERM} permutations x {len(TASKS)} tasks: "
          f"pooled mean {m_sh:.4f} (tol |.-0.5| < {TOL_SHUF_MEAN}); sd {s_sh:.4f} "
          f"vs binomial {sd_binom:.4f} (ratio {s_sh / sd_binom:.2f}, not asserted)")
    for k, name in enumerate(TASKS):
        print(f"    {name:10s} shuffled mean {acc_shuf[:, k].mean():.3f}, sd "
              f"{acc_shuf[:, k].std():.3f}, max {acc_shuf[:, k].max():.3f}; true "
              f"{acc[jm, k]:.3f}; p_perm = {p_perm[k]:.4f}")
    R.update(net=net, tpl=tpl, bits_te=bits_te, inp_te=inp_te, ras=ras_te[0],
             acc=acc, acc_in=acc_in, acc_shuf=acc_shuf, pvals=pvals, jm=jm,
             rate=rate, sd_binom=sd_binom, q95=q95, acc_train=acc_train,
             p_perm=p_perm)
    print(f"validation done in {_time.time() - t0:.1f} s")

    assert np.all(pvals < P_MAX), f"a readout is not above chance: p = {pvals}"
    assert abs(m_sh - 0.5) < TOL_SHUF_MEAN, "shuffled control not at chance"
    assert np.all(p_perm < P_PERM_MAX), f"permutation test fails: {p_perm}"
    return R


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
def plot(R):
    FIGURES.mkdir(exist_ok=True)
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    cols = plt.cm.viridis(np.linspace(0, 0.85, 4))

    # (a) input of test trial 0 with segment templates
    a = ax[0, 0]
    inp = R["inp_te"][:, 0, :]
    b0 = R["bits_te"][0]
    for k in range(N_SEG):
        a.axvspan(k * T_SEG, (k + 1) * T_SEG, color=cols[k], alpha=0.08)
        a.text((k + 0.5) * T_SEG, N_IN - 0.35, f"b{k + 1}={b0[k]}", ha="center", fontsize=9)
        for ch, st in enumerate(R["tpl"][k][b0[k]]):
            a.plot(st + k * T_SEG, np.full(len(st), ch + 0.25), "|", color="0.6", ms=8)
    s, c = np.nonzero(inp)
    a.plot(s * DT, c, "|", color="k", ms=10)
    a.set(xlim=(0, T_TOTAL), ylim=(-0.6, N_IN), yticks=range(N_IN),
          xlabel="time (ms)", ylabel="input channel",
          title="Input, test trial 1 (grey: templates, black: jittered)")

    # (b) reservoir raster
    a = ax[0, 1]
    if R["ras"]:
        t, j = np.array(R["ras"]).T
        ne = R["net"]["n_e"]
        e = j < ne
        a.plot(t[e], j[e], ".", color="C0", ms=2.5, label="E")
        a.plot(t[~e], j[~e], ".", color="C3", ms=2.5, label="I")
        a.axhline(ne - 0.5, color="0.7", lw=0.5)
    a.set(xlim=(0, T_TOTAL), ylim=(-1, N), xlabel="time (ms)", ylabel="neuron",
          title=f"Reservoir, same trial (mean rate {R['rate']:.1f} Hz)")
    a.legend(loc="upper right", markerscale=4, fontsize=8)

    # (c) accuracy vs readout time
    a = ax[0, 2]
    for k in range(N_SEG):
        a.plot(T_READ, R["acc"][:, k], "o-", color=cols[k], ms=3, label=TASKS[k])
        a.axvspan(k * T_SEG, (k + 1) * T_SEG, color=cols[k], alpha=0.06)
    a.plot(T_READ, R["acc"][:, 4], "s-", color="C3", ms=3, label=TASKS[4])
    a.axhspan(1 - R["q95"], R["q95"], color="0.85", label="chance, 90% binomial band")
    a.axhline(0.5, color="0.5", lw=0.8)
    a.axvline(T_MAIN, color="k", ls=":", lw=0.8)
    a.set(xlabel="readout time t (ms)", ylabel="held-out accuracy", ylim=(0.35, 1.02),
          title="Readout accuracy vs time (fading memory)")
    a.legend(fontsize=8, loc="lower right")

    # (d) bars at T_MAIN
    a = ax[1, 0]
    x = np.arange(len(TASKS))
    jm = R["jm"]
    a.bar(x - 0.27, R["acc"][jm], 0.27, label="reservoir", color="C0")
    a.bar(x, R["acc_in"][jm], 0.27, label="input only", color="C1")
    sh = R["acc_shuf"]
    a.bar(x + 0.27, sh.mean(0), 0.27, yerr=2 * sh.std(0), capsize=3,
          label="shuffled labels (mean ± 2 sd)", color="0.6")
    a.axhline(R["q95"], color="k", ls="--", lw=0.8, label="binomial 95% point")
    for k in x:
        a.text(k - 0.27, R["acc"][jm, k] + 0.01, f"{R['acc'][jm, k]:.2f}",
               ha="center", fontsize=8)
    a.set(xticks=x, xticklabels=TASKS, ylim=(0.3, 1.22), ylabel="held-out accuracy",
          title=f"Accuracy at t = {T_MAIN:.0f} ms, {N_TEST} test trials")
    a.legend(fontsize=8, loc="upper center", ncol=2)

    # (e) shuffled distribution per task vs true-label accuracy
    a = ax[1, 1]
    rj = np.random.default_rng(0)
    for k in range(len(TASKS)):
        a.plot(k + rj.uniform(-0.25, 0.25, sh.shape[0]), sh[:, k], ".", color="0.6",
               ms=3, label="shuffled-label readouts" if k == 0 else None)
        a.plot(k, R["acc"][jm, k], "*", color="C0", ms=14,
               label="true-label readout" if k == 0 else None)
        a.text(k, 1.02, f"p={R['p_perm'][k]:.3f}", ha="center", fontsize=8)
    a.axhspan(1 - R["q95"], R["q95"], color="0.85", zorder=0,
              label="binomial 90% band (fixed predictions)")
    a.axhline(0.5, color="0.5", lw=0.8)
    a.set(xticks=range(len(TASKS)), xticklabels=TASKS, ylim=(0.2, 1.07),
          ylabel="held-out accuracy",
          title=f"Shuffled-label control, {sh.shape[0]} permutations (p = permutation test)")
    a.legend(fontsize=8, loc="lower left")

    # (f) LIF f-I and TM
    a = ax[1, 2]
    a.plot(R["I"], R["r_cf"], "k-", label="closed form")
    a.plot(R["I"], R["r_sim"], "o", color="C0", ms=4, label="lif_step, dt = 0.1 ms")
    a.set(xlabel="constant input current above I_b (nA)", ylabel="rate (Hz)",
          title=f"Neuron update vs LIF closed form (max err {100 * R['e_rate']:.2f}%)")
    a.legend(fontsize=8, loc="upper left")
    ins = a.inset_axes([0.58, 0.1, 0.38, 0.38])
    for key, mk in zip(["EE", "EI", "IE", "II"], "osd^"):
        rows = [r for r in R["tm"] if r[0] == key]
        ins.plot([r[1] for r in rows], [r[3] for r in rows], "-", color="0.6", lw=0.8)
        ins.plot([r[1] for r in rows], [r[2] for r in rows], mk, ms=3, label=key)
    ins.set_xscale("log")
    ins.set_title("TM steady state u*R*", fontsize=7)
    ins.tick_params(labelsize=6)
    ins.set_xlabel("Hz", fontsize=6)
    ins.legend(fontsize=5)

    fig.suptitle("Liquid state machine (Maass, Natschläger & Markram 2002): 135 LIF neurons, "
                 "dynamic synapses, linear readout of jittered spike templates", fontsize=12)
    fig.tight_layout()
    out = FIGURES / "liquid_state_machine.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    t0 = _time.time()
    R = validate()
    plot(R)
    print(f"total {_time.time() - t0:.1f} s")
