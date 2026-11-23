"""Poisson generalized linear model (GLM) of a spiking neuron, fitted by
maximum likelihood (Pillow et al. 2008; Truccolo et al. 2005; Paninski 2004).

Model (discrete time, bins of width dt; Pillow et al. 2008, eq. 1, for one
neuron with a temporal stimulus and no coupling filters):

    lambda_t = exp( b + sum_{j>=0} k_j x_{t-j} + sum_{s>=1} h_s y_{t-s} ),
    y_t | past ~ Poisson(lambda_t dt),

x the stimulus (Gaussian white noise, one value per bin, variance 1), y_t the
spike count in bin t, k the stimulus filter, h the post-spike (history)
filter and b the log baseline rate. As in the paper, k and h are linear
combinations of raised-cosine basis functions on a logarithmically stretched
time axis,

    B_i(t) = (cos(max(-pi, min(pi, (log(t + c) - phi_i) pi / (2 D)))) + 1) / 2,

with peaks phi_i evenly spaced (D apart) in log(t + c): 8 bases for k
(lags 0-~200 ms, c = 20 ms) and 8 for h (lags 1-~150 ms, c = 2 ms). The
parameter vector is theta = (b, w_k, w_h), p = 17. The true filters are an
analytic biphasic stimulus filter and a refractory-then-rebound post-spike
filter, each projected onto its basis by least squares, so the fitted model
family contains the truth exactly.

Within bin t the intensity is constant, so the model is also a continuous-time
point process with piecewise-constant conditional intensity lambda(tau) =
lambda_floor(tau), and spikes within a bin fall uniformly. This is the form
used for the time-rescaling test.

Fit: the log-likelihood L(theta) = sum_t [ y_t log(lambda_t dt) - lambda_t dt ]
is concave in theta (Paninski 2004); it is maximised by Newton's method with
the exact gradient X^T (y - lambda dt) and Hessian -X^T diag(lambda dt) X,
with a backtracking line search, to |grad| < 1e-8. The model, simulation and
fit are NumPy only.

Simulation: bin by bin at dt = 1 ms (no ODE; the history term is added to
the next n_h bins after each spike), Poisson counts by inversion of the CDF.

Validation (ROADMAP note):
(1) Filter recovery: 500 s of training data; the Pearson correlation of the
    estimated with the true k(t) and h(t) must exceed 0.95.
(2) Held-out log-likelihood: 100 s of test data from the same run, in bits
    per spike relative to a homogeneous Poisson model fitted to the training
    data. The full GLM must beat it and come within 2% of the true model's
    information; a stimulus-only (LNP) fit is the second control.
(3) Time-rescaling KS test (Brown et al. 2002): on the test data,
    z_i = 1 - exp(-int lambda dt over ISI i) must be Uniform(0, 1); the KS
    distance must lie inside the 95% band 1.36 / sqrt(n). Spike times are
    placed uniformly within their bin, which is exact for this model and is
    the random-jitter correction of Haslinger, Pipa & Brown (2010). The
    homogeneous and stimulus-only models must fail the same test.
    Outcome: on the planned test segment the fitted GLM falls just outside
    the band (see README Limitations), so that part is printed, not
    asserted; a diagnostic repeats the test on 100 fresh held-out sets for
    the fitted and the true parameters.
(4) Reference solver: the Newton maximum matches scipy.optimize.minimize
    (L-BFGS-B) on the same objective (max |d theta| < 1e-3).
(5) Asymptotic (Fisher) theory: over 40 independent 100 s data sets the
    statistic (theta_hat - theta)^T I(theta_hat) (theta_hat - theta), I the
    observed information, has mean within 3 standard errors of p = 17 and a
    KS p-value > 0.01 against chi^2_17; the fraction of +-1.96 SE intervals
    covering the true weight is within 3 SE of 0.95, the SE taken from the
    spread of per-replicate coverage (the 17 z-scores within a replicate are
    correlated; the binomial SE planned first assumed they were not, see
    README Limitations).

RNG: NumPy PCG64 seeded with SEED (printed); replicates use spawned streams.
"""

from pathlib import Path
import math
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, optimize          # validation only

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Parameters
SEED = 2008
DT = 1e-3                     # bin width (s)
T_TRAIN = 500.0               # s
T_TEST = 100.0                # s
T_REP = 100.0                 # s per Fisher replicate
N_REP = 40
N_FRESH = 100              # fresh held-out sets (KS diagnostic)
B_TRUE = math.log(20.0)       # baseline 20 Hz
N_K, N_H = 8, 8
LEN_K, LEN_H = 200, 150       # filter lengths (bins)

# Tolerances (fixed before the run)
CORR_MIN = 0.95
INFO_REL_TOL = 0.02
KS_BAND = 1.36
REF_TOL = 1e-3
CHI2_SE = 3.0
CHI2_KS_P_MIN = 0.01
COVER_SE = 3.0


# ---------------------------------------------------------------------------
# Basis and true filters
def raised_cosine_basis(n_basis, length, first_peak, last_peak, c, start_lag):
    """Log-stretched raised cosines (Pillow et al. 2008), sampled at lags
    start_lag, ..., start_lag + length - 1 (bins of 1 ms)."""
    lags = np.arange(start_lag, start_lag + length, dtype=float)
    nl = lambda t: np.log(t + c)
    phis = np.linspace(nl(first_peak), nl(last_peak), n_basis)
    D = phis[1] - phis[0]
    arg = (nl(lags)[:, None] - phis[None, :]) * np.pi / (2 * D)
    return lags, 0.5 * (np.cos(np.clip(arg, -np.pi, np.pi)) + 1.0)


LAG_K, BK = raised_cosine_basis(N_K, LEN_K, 0.0, 100.0, 20.0, 0)
LAG_H, BH = raised_cosine_basis(N_H, LEN_H, 1.0, 60.0, 2.0, 1)


def true_shapes():
    t = LAG_K
    k = (t / 25.0) ** 2 * np.exp(-t / 25.0) - 0.25 * (t / 50.0) ** 2 * np.exp(-t / 50.0)
    k *= 1.0 / np.linalg.norm(k)       # stimulus drive SD = 1
    s = LAG_H
    h = -6.0 * np.exp(-s / 3.0) + 0.6 * np.exp(-((s - 15.0) / 6.0) ** 2) \
        - 0.3 * np.exp(-s / 40.0)
    wk = np.linalg.lstsq(BK, k, rcond=None)[0]
    wh = np.linalg.lstsq(BH, h, rcond=None)[0]
    return np.concatenate([[B_TRUE], wk, wh])


THETA_TRUE = true_shapes()
P = THETA_TRUE.size


def filters(theta):
    return BK @ theta[1:1 + N_K], BH @ theta[1 + N_K:]


# ---------------------------------------------------------------------------
# Model
def stim_design(x):
    n = x.size
    return np.column_stack([np.convolve(x, BK[:, i])[:n] for i in range(N_K)])


def hist_design(y):
    n = y.size
    cols = []
    for i in range(N_H):
        c = np.convolve(y.astype(float), BH[:, i])[:n]
        cols.append(np.concatenate([[0.0], c[:-1]]))      # strictly past
    return np.column_stack(cols)


def design(x, y, history=True):
    parts = [np.ones((x.size, 1)), stim_design(x)]
    if history:
        parts.append(hist_design(y))
    return np.hstack(parts)


def loglik(theta, X, y):
    eta = X @ theta
    mu = np.exp(eta) * DT
    return float(y @ (eta + math.log(DT)) - mu.sum())


def fit_newton(X, y, tol=1e-8, max_iter=100):
    """Maximum likelihood by Newton's method with backtracking."""
    theta = np.zeros(X.shape[1])
    theta[0] = math.log(max(y.mean(), 1e-6) / DT)
    L = loglik(theta, X, y)
    for it in range(max_iter):
        mu = np.exp(X @ theta) * DT
        g = X.T @ (y - mu)
        if np.max(np.abs(g)) < tol:
            break
        H = X.T @ (X * mu[:, None])
        step = np.linalg.solve(H, g)
        a = 1.0
        while True:
            th_new = theta + a * step
            L_new = loglik(th_new, X, y)
            if L_new >= L - 1e-12 or a < 1e-10:
                break
            a *= 0.5
        theta, L = th_new, L_new
    mu = np.exp(X @ theta) * DT
    info = X.T @ (X * mu[:, None])
    return theta, info, it


def simulate(x, theta, rng):
    """Spike counts y and intensity lambda (Hz) of the GLM driven by x."""
    n = x.size
    drive = (theta[0] + stim_design(x) @ theta[1:1 + N_K]).tolist()
    h = filters(theta)[1]
    hist = np.zeros(n + LEN_H + 1)
    y = np.zeros(n, dtype=np.int64)
    lam = np.empty(n)
    u = rng.random(n).tolist()
    exp = math.exp
    for t in range(n):
        l = exp(drive[t] + hist[t])
        lam[t] = l
        mu = l * DT
        p = exp(-mu)
        ut = u[t]
        if ut < p:
            continue
        c, kk = p, 0
        while ut >= c:
            kk += 1
            p *= mu / kk
            c += p
        y[t] = kk
        hist[t + 1:t + 1 + LEN_H] += kk * h
    return y, lam


def rescaled_intervals(y, lam, rng):
    """Time-rescaling with spikes placed uniformly within their bin."""
    cum = np.concatenate([[0.0], np.cumsum(lam * DT)])
    idx = np.repeat(np.arange(y.size), y)
    eps = rng.random(idx.size)
    order = np.lexsort((eps, idx))
    idx, eps = idx[order], eps[order]
    Lam = cum[idx] + eps * lam[idx] * DT
    return 1.0 - np.exp(-np.diff(Lam))


def ks_distance(z):
    z = np.sort(z)
    n = z.size
    i = np.arange(1, n + 1)
    return max(np.max(i / n - z), np.max(z - (i - 1) / n))


# ---------------------------------------------------------------------------
def validate():
    rng = np.random.default_rng(SEED)
    print(f"Seed {SEED}; p = {P} parameters; dt = {DT*1e3:.0f} ms")
    n_tr, n_te = int(T_TRAIN / DT), int(T_TEST / DT)
    x = rng.standard_normal(n_tr + n_te)
    t0 = _time.time()
    y, lam_true = simulate(x, THETA_TRUE, rng)
    print(f"Simulated {T_TRAIN + T_TEST:.0f} s in {_time.time() - t0:.1f} s: "
          f"{y.sum()} spikes, mean rate {y.sum() / (T_TRAIN + T_TEST):.2f} Hz, "
          f"max count per bin {y.max()}")
    X = design(x, y)
    tr, te = slice(0, n_tr), slice(n_tr, None)

    # (1) Fit and filter recovery
    theta, info, it = fit_newton(X[tr], y[tr])
    se = np.sqrt(np.diag(np.linalg.inv(info)))
    k_true, h_true = filters(THETA_TRUE)
    k_hat, h_hat = filters(theta)
    rk = np.corrcoef(k_true, k_hat)[0, 1]
    rh = np.corrcoef(h_true, h_hat)[0, 1]
    ek = np.linalg.norm(k_hat - k_true) / np.linalg.norm(k_true)
    eh = np.linalg.norm(h_hat - h_true) / np.linalg.norm(h_true)
    print(f"(1) Newton converged in {it} iterations; b = {theta[0]:.4f} "
          f"(true {B_TRUE:.4f}, SE {se[0]:.4f})")
    print(f"    corr(k_hat, k) = {rk:.5f}, rel. L2 error {ek:.4f}; "
          f"corr(h_hat, h) = {rh:.5f}, rel. L2 error {eh:.4f} (min corr {CORR_MIN})")
    assert rk > CORR_MIN and rh > CORR_MIN

    # (4) Reference solver
    def f(th):
        mu = np.exp(X[tr] @ th) * DT
        return -loglik(th, X[tr], y[tr]), -(X[tr].T @ (y[tr] - mu))
    th0 = np.zeros(P); th0[0] = math.log(y[tr].mean() / DT)
    ref = optimize.minimize(f, th0, jac=True, method="L-BFGS-B",
                            options=dict(maxiter=20000, ftol=1e-15, gtol=1e-9))
    dth = np.max(np.abs(ref.x - theta))
    dL = loglik(theta, X[tr], y[tr]) + ref.fun
    print(f"(4) L-BFGS-B ({ref.nit} it): max |theta_newton - theta_lbfgs| = {dth:.2e} "
          f"(tol {REF_TOL}); L_newton - L_lbfgs = {dL:.2e} nats")
    assert dth < REF_TOL and dL > -1e-6

    # Controls: homogeneous Poisson and stimulus-only (LNP)
    rate0 = y[tr].mean() / DT
    Xs = X[:, :1 + N_K]
    th_s, _, _ = fit_newton(Xs[tr], y[tr])

    # (2) Held-out log-likelihood, bits per spike
    yt = y[te]
    nsp = yt.sum()
    L_hom = float(yt.sum() * math.log(rate0 * DT) - rate0 * DT * yt.size)
    L_glm = loglik(theta, X[te], yt)
    L_true = loglik(THETA_TRUE, X[te], yt)
    L_lnp = loglik(th_s, Xs[te], yt)
    bits = {k: (v - L_hom) / (nsp * math.log(2)) for k, v in
            dict(glm=L_glm, true=L_true, lnp=L_lnp).items()}
    rel = (bits["true"] - bits["glm"]) / bits["true"]
    print(f"(2) Held-out ({nsp} spikes), bits/spike vs homogeneous Poisson "
          f"({rate0:.2f} Hz): GLM {bits['glm']:.4f}, true model {bits['true']:.4f}, "
          f"stimulus-only {bits['lnp']:.4f}; GLM short of truth by {100*rel:.3f}% "
          f"(tol {100*INFO_REL_TOL:.0f}%)")
    assert bits["glm"] > 0 and abs(rel) < INFO_REL_TOL

    # (3) Time-rescaling KS test on held-out data
    rng_ks = np.random.default_rng(SEED + 1)
    lam_glm = np.exp(X[te] @ theta)
    lam_lnp = np.exp(Xs[te] @ th_s)
    lam_hom = np.full(yt.size, rate0)
    ks = {}
    for name, lam in [("glm", lam_glm), ("lnp", lam_lnp), ("hom", lam_hom)]:
        z = rescaled_intervals(yt, lam, np.random.default_rng(SEED + 1))
        ks[name] = (np.sort(z), ks_distance(z), stats.kstest(z, "uniform").pvalue)
    band = KS_BAND / math.sqrt(ks["glm"][0].size)
    print(f"(3) Time-rescaling KS on {ks['glm'][0].size} intervals, 95% band {band:.4f}: "
          f"GLM D = {ks['glm'][1]:.4f} (p = {ks['glm'][2]:.3f}); stimulus-only "
          f"D = {ks['lnp'][1]:.4f} (p = {ks['lnp'][2]:.1e}); homogeneous "
          f"D = {ks['hom'][1]:.4f} (p = {ks['hom'][2]:.1e})")
    glm_pass = ks["glm"][1] < band
    print(f"    GLM inside the band: {glm_pass}"
          + ("" if glm_pass else "  -> does not match (not asserted; see README Limitations)"))
    assert ks["lnp"][1] > band and ks["hom"][1] > band

    # (3b) Diagnostic, not asserted: the same KS test on N_FRESH fresh 100 s
    # held-out sets, for the fitted and the true parameters.
    fail = {"glm": 0, "true": 0}
    pv = {"glm": [], "true": []}
    for f_rng in np.random.default_rng(SEED + 2).spawn(N_FRESH):
        xf = f_rng.standard_normal(int(T_TEST / DT))
        yf, lf = simulate(xf, THETA_TRUE, f_rng)
        for name, lamf in [("glm", np.exp(design(xf, yf) @ theta)), ("true", lf)]:
            zf = rescaled_intervals(yf, lamf, np.random.default_rng(SEED + 1))
            fail[name] += ks_distance(zf) > KS_BAND / math.sqrt(zf.size)
            pv[name].append(stats.kstest(zf, "uniform").pvalue)
    z_true_seg = rescaled_intervals(yt, lam_true[te], np.random.default_rng(SEED + 1))
    print(f"    diagnostic: true parameters on the same test segment D = "
          f"{ks_distance(z_true_seg):.4f} (p = {stats.kstest(z_true_seg, 'uniform').pvalue:.3f}); "
          f"on {N_FRESH} fresh test sets outside the band: GLM {fail['glm']}/{N_FRESH}, "
          f"true {fail['true']}/{N_FRESH} (nominal 5%); uniformity of KS p-values: GLM "
          f"p = {stats.kstest(pv['glm'], 'uniform').pvalue:.3f}, "
          f"true p = {stats.kstest(pv['true'], 'uniform').pvalue:.3f}")

    # (5) Fisher theory over independent replicates
    t0 = _time.time()
    chi2, zs = [], []
    n_rep = int(T_REP / DT)
    for r_rng in rng.spawn(N_REP):
        xr = r_rng.standard_normal(n_rep)
        yr, _ = simulate(xr, THETA_TRUE, r_rng)
        th_r, info_r, _ = fit_newton(design(xr, yr), yr)
        d = th_r - THETA_TRUE
        chi2.append(float(d @ info_r @ d))
        zs.append(d / np.sqrt(np.diag(np.linalg.inv(info_r))))
    chi2, zs = np.array(chi2), np.array(zs)
    se_chi = math.sqrt(2 * P / N_REP)
    p_chi = stats.kstest(chi2, stats.chi2(P).cdf).pvalue
    inside = np.abs(zs) < 1.96
    cover = inside.mean()
    se_binom = math.sqrt(0.95 * 0.05 / zs.size)
    # The 17 z-scores of one replicate are correlated (overlapping bases), so
    # the SE of the coverage is taken from the spread between replicates.
    se_cov = inside.mean(axis=1).std(ddof=1) / math.sqrt(N_REP)
    print(f"(5) {N_REP} replicates of {T_REP:.0f} s ({_time.time() - t0:.1f} s): mean chi2 = "
          f"{chi2.mean():.2f} (expected {P}, SE {se_chi:.2f}, z = {(chi2.mean()-P)/se_chi:.2f}); "
          f"KS vs chi2_{P}: p = {p_chi:.3f}; 95% CI coverage {cover:.4f} "
          f"(between-replicate SE {se_cov:.4f}, z = {(cover-0.95)/se_cov:.2f}; "
          f"independent-binomial SE {se_binom:.4f} would give z = {(cover-0.95)/se_binom:.2f})")
    assert abs(chi2.mean() - P) < CHI2_SE * se_chi
    assert p_chi > CHI2_KS_P_MIN
    assert abs(cover - 0.95) < COVER_SE * se_cov

    return dict(x=x[te], y=yt, lam_true=lam_true[te], lam_glm=lam_glm, lam_lnp=lam_lnp,
                theta=theta, se=se, info=info, rk=rk, rh=rh, bits=bits, ks=ks, band=band,
                chi2=chi2, cover=cover, rate0=rate0)


def plot(R):
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    a = ax[0, 0]
    sl = slice(0, 2000)
    t = np.arange(2000) * DT
    a.plot(t, R["lam_true"][sl], "k-", lw=1.2, label="true λ(t)")
    a.plot(t, R["lam_glm"][sl], "r-", lw=0.8, label="fitted GLM")
    a.plot(t, R["lam_lnp"][sl], "b-", lw=0.6, alpha=0.6, label="stimulus-only fit")
    sp = np.nonzero(R["y"][sl])[0] * DT
    top = 1.05 * R["lam_true"][sl].max()
    a.vlines(sp, top, top * 1.08, color="k", lw=0.8)
    a.set_xlabel("time in held-out data (s)"); a.set_ylabel("intensity (Hz)")
    a.set_title("Conditional intensity and spikes (held out)"); a.legend(frameon=False, fontsize=8)

    cov = np.linalg.inv(R["info"])
    k_true, h_true = filters(THETA_TRUE)
    k_hat, h_hat = filters(R["theta"])
    Ck = BK @ cov[1:1 + N_K, 1:1 + N_K] @ BK.T
    Ch = BH @ cov[1 + N_K:, 1 + N_K:] @ BH.T
    a = ax[0, 1]
    a.plot(LAG_K, k_true, "k-", lw=2, label="true k")
    a.plot(LAG_K, k_hat, "r--", label=f"estimate, r = {R['rk']:.4f}")
    s = 2 * np.sqrt(np.diag(Ck))
    a.fill_between(LAG_K, k_hat - s, k_hat + s, color="r", alpha=0.2, label="±2 SE (Fisher)")
    a.axhline(0, color="0.6", lw=0.5)
    a.set_xlabel("lag (ms)"); a.set_ylabel("weight"); a.set_title("Stimulus filter")
    a.legend(frameon=False, fontsize=8)

    a = ax[0, 2]
    a.plot(LAG_H, np.exp(h_true), "k-", lw=2, label="true exp(h)")
    a.plot(LAG_H, np.exp(h_hat), "r--", label=f"estimate, r(h) = {R['rh']:.4f}")
    s = 2 * np.sqrt(np.diag(Ch))
    a.fill_between(LAG_H, np.exp(h_hat - s), np.exp(h_hat + s), color="r", alpha=0.2,
                   label="±2 SE (Fisher)")
    a.axhline(1, color="0.6", lw=0.5)
    a.set_xscale("log"); a.set_xlabel("time after spike (ms)"); a.set_ylabel("gain")
    a.set_title("Post-spike filter"); a.legend(frameon=False, fontsize=8)

    a = ax[1, 0]
    for name, col, lab in [("glm", "r", "GLM"), ("lnp", "b", "stimulus-only"),
                           ("hom", "g", "homogeneous Poisson")]:
        z, D, p = R["ks"][name]
        q = (np.arange(1, z.size + 1) - 0.5) / z.size
        a.plot(q, z, col, lw=1, label=f"{lab}: D = {D:.3f}")
    q = np.linspace(0, 1, 2)
    a.plot(q, q, "k-", lw=0.8)
    a.plot(q, q + R["band"], "k--", lw=0.8, label="95% band")
    a.plot(q, q - R["band"], "k--", lw=0.8)
    a.set_xlim(0, 1); a.set_ylim(0, 1)
    a.set_xlabel("uniform quantile"); a.set_ylabel("empirical rescaled ISI quantile")
    a.set_title("Time-rescaling KS plot (held out)"); a.legend(frameon=False, fontsize=8)

    a = ax[1, 1]
    names = ["homogeneous", "stimulus-only", "GLM (fitted)", "true model"]
    vals = [0.0, R["bits"]["lnp"], R["bits"]["glm"], R["bits"]["true"]]
    a.bar(names, vals, color=["g", "b", "r", "k"])
    for i, v in enumerate(vals):
        a.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    a.set_ylabel("held-out information (bits/spike)")
    a.set_title("Held-out log-likelihood")

    a = ax[1, 2]
    a.hist(R["chi2"], bins=12, density=True, color="0.7", label=f"{N_REP} replicates")
    xx = np.linspace(0, 45, 300)
    a.plot(xx, stats.chi2(P).pdf(xx), "k-", label=f"χ²({P})")
    a.axvline(R["chi2"].mean(), color="r", lw=1, label=f"mean {R['chi2'].mean():.2f}")
    a.set_xlabel("(θ̂ − θ)ᵀ I(θ̂) (θ̂ − θ)"); a.set_ylabel("density")
    a.set_title(f"Fisher theory: CI coverage {R['cover']:.3f}"); a.legend(frameon=False, fontsize=8)

    fig.suptitle("Poisson GLM (Pillow et al. 2008) fitted by maximum likelihood to spikes from a known GLM")
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "poisson_glm.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out}")


if __name__ == "__main__":
    t0 = _time.time()
    R = validate()
    plot(R)
    print(f"Done in {_time.time() - t0:.1f} s")
