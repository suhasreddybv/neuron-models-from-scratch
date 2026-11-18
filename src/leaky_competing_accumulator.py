"""Leaky competing accumulator (LCA) of Usher & McClelland (2001) and its
reduction to the drift-diffusion model.

Model (two alternatives; Usher & McClelland 2001, eq. 4, in the notation of
Bogacz, Brown, Moehlis, Holmes & Cohen 2006, sec. "Leaky competing
accumulator"):

    dx1 = (-k x1 - w x2 + I1) dt + c' dW1,
    dx2 = (-k x2 - w x1 + I2) dt + c' dW2,      x1, x2 >= 0,

k the leak (decay), w the mutual inhibition, I1, I2 the inputs, c' the noise
of each unit and W1, W2 independent Wiener processes. Activities are
rectified at zero after every step (Usher & McClelland's threshold-linear
units). Two stopping rules are used:
  - absolute criterion (Usher & McClelland): stop when either x_i reaches Z;
  - difference criterion (Bogacz et al. 2006): stop when |x1 - x2| reaches z.

Reduction. In the coordinates y = x1 - x2 and s = x1 + x2 the linear system
decouples exactly (Bogacz et al. 2006, eqs. 20-22):

    dy = (A + lambda y) dt + c dW_y,      A = I1 - I2,  lambda = w - k,
    ds = (I1 + I2 - (k + w) s) dt + c dW_s,

with c = sqrt(2) c' and W_y, W_s independent. So, while no unit is at zero,
the difference criterion turns the LCA into the Ornstein-Uhlenbeck (OU)
model, and into the pure DDM dy = A dt + c dW exactly when the leak equals
the inhibition (lambda = 0), for any k + w. Under the absolute criterion the
sum must also be pinned: x1 >= Z is y >= 2Z - s, so as K = k + w -> infinity
with s* = (I1 + I2)/K fixed, s -> s* and the LCA tends to the DDM with
bound z = 2Z - s*. For finite K the sum fluctuates with SD c / sqrt(2K) and
correlation time 1/K, during which y moves by ~ c / sqrt(K); both are the same
order, so the LCA should act like a DDM with bound z - delta, delta ~ c/sqrt(K)
(scaling argument made here; the constant is not derived).

Parameters: c = 0.1 (so y has the DDM script's noise), c' = c/sqrt(2),
z = 0.06, s* = 0.12, Z = (z + s*)/2 = 0.09; given A, K and lambda:
I1,2 = (K s* +- A)/2, k = (K - lambda)/2, w = (K + lambda)/2. Both units start
at the sum fixed point, x1 = x2 = s*/2 (equivalent to a long pre-stimulus
period with equal inputs). Units: x dimensionless, t in s.

Integrator: the linear part is advanced exactly in distribution in (y, s)
(OU transition with mean e^{-r dt} and variance c^2 (1 - e^{-2 r dt})/(2r),
r = K for s and r = -lambda for y; r = 0 is Brownian motion with drift), then
x1, x2 are formed and rectified at zero. The only discretisation error is in
the boundary test and the rectification. A Brownian-bridge test catches
crossings inside a step, as in drift_diffusion.py: probability
exp(-2 (b - u_n)(b - u_{n+1}) / (sigma^2 dt)) with sigma = c for the
difference y and sigma = c' for a unit x_i. dt = 0.5 ms for the difference
criterion; for the absolute criterion dt = min(0.5 ms, 0.02/K) so that the
sum relaxes by at most 2% per step.

Closed forms used for validation:
(1) Pure DDM at x0 = 0 (Bogacz et al. 2006): P(correct) = 1/(1 + e^{-2Az/c^2}),
    mean decision time (z/A) tanh(Az/c^2), = z^2/c^2 at A = 0.
(2) First-passage-time densities of the DDM at each bound, large-time series
    (Feller 1968; Navarro & Fuss 2009), for KS tests (as in drift_diffusion.py).
(3) OU model with absorbing bounds +-z: scale density
    psi(y) = exp(-(2/c^2)(A y + lambda y^2/2)); P(+z | y0) = int_{-z}^{y0} psi /
    int_{-z}^{z} psi (Busemeyer & Townsend 1993; Bogacz et al. 2006);
    mean exit time T solves (c^2/2) T'' + (A + lambda y) T' = -1, T(+-z) = 0,
    whose solution is T(y) = C0 Psi(y) - (2/c^2) int_{-z}^{y} psi(u) Phi(u) du,
    Psi = int psi, Phi = int 1/psi, C0 fixed by T(z) = 0 (evaluated by
    trapezoidal quadrature on 200 001 points, checked against scipy quad and
    against (1) at lambda = 0).

Validation (ROADMAP note: reduces to the DDM in the appropriate limit):
 1. difference criterion, k = w, K = 40, 160, 640, 5 drifts, 40 000 trials:
    accuracy and mean DT against (1) as z-scores, DT distributions by KS
    against (2);
 2. difference criterion, k != w (lambda = -4 ... 4, K = 160), against (3);
 3. absolute criterion, k = w, A = 0.1, K = 6.25 ... 400: accuracy and mean DT
    against (1); effective bounds z_eff inverted from the error rate and from
    the mean DT must agree (the LCA behaves as a DDM) for K >= 100, and
    delta = z - z_eff must fall as K^-1/2 (fitted slope within 0.15 of -0.5
    for K >= 50).

RNG: NumPy PCG64 seeded with SEED (printed).
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats, integrate, optimize          # validation only

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "figures"

# ---------------------------------------------------------------------------
# Parameters
SEED = 2001
C = 0.1                       # noise of y = x1 - x2 (and of s = x1 + x2)
C_UNIT = C / np.sqrt(2.0)     # noise of each unit, c'
Z_DIFF = 0.06                 # difference criterion |x1 - x2| >= z
S_STAR = 0.12                 # fixed point of the sum, (I1 + I2)/(k + w)
Z_ABS = 0.5 * (Z_DIFF + S_STAR)   # absolute criterion x_i >= Z (0.09)
T_MAX = 10.0
DT_DIFF = 5e-4
K_BAL = np.array([40.0, 160.0, 640.0])
DRIFTS = np.array([0.0, 0.05, 0.1, 0.2, 0.4])
N_TRIALS = 40_000
K_OU = 160.0
LAMS = np.array([-4.0, -2.0, 2.0, 4.0])
A_OU = np.array([0.1, 0.2])
K_CONV = np.array([6.25, 12.5, 25.0, 50.0, 100.0, 200.0, 400.0])
A_CONV = 0.1
KDT = 0.02                    # K dt for the absolute criterion
K_SERIES = 400
N_GRID = 200_001

# Tolerances (fixed before the run)
Z_TOL = 4.0
KS_P_MIN = 1e-4
FORM_TOL = 1e-8
K_FIT_MIN = 50.0
K_CONS_MIN = 100.0
SLOPE_TOL = 0.15


# ---------------------------------------------------------------------------
# Closed forms
def ddm_p(A, z=Z_DIFF, c=C):
    """P(correct) of the pure DDM from x0 = 0, closed form (1)."""
    return 1.0 / (1.0 + np.exp(-2.0 * np.asarray(A, float) * z / c**2))


def ddm_t(A, z=Z_DIFF, c=C):
    """Mean decision time of the pure DDM from x0 = 0, closed form (1)."""
    A = np.asarray(A, float)
    As = np.where(A == 0.0, 1.0, A)
    return np.where(A == 0.0, z**2 / c**2, z / As * np.tanh(As * z / c**2))


def ou_forms(A, lam, z=Z_DIFF, c=C, n=N_GRID):
    """(P(+z), mean exit time) of the OU model from y0 = 0, closed form (3)."""
    y = np.linspace(-z, z, n)
    e = -(2.0 / c**2) * (A * y + 0.5 * lam * y**2)
    e -= e.mean()
    psi = np.exp(e)
    Psi = integrate.cumulative_trapezoid(psi, y, initial=0.0)
    Phi = integrate.cumulative_trapezoid(1.0 / psi, y, initial=0.0)
    J = integrate.cumulative_trapezoid(psi * Phi, y, initial=0.0)
    c0 = (2.0 / c**2) * J[-1] / Psi[-1]
    T = c0 * Psi - (2.0 / c**2) * J
    mid = n // 2
    return Psi[mid] / Psi[-1], T[mid]


def ou_p_quad(A, lam, z=Z_DIFF, c=C):
    """P(+z) of the OU model by adaptive quadrature (cross-check of ou_forms)."""
    f = lambda u: np.exp(-(2.0 / c**2) * (A * u + 0.5 * lam * u**2))
    num = integrate.quad(f, -z, 0.0, epsabs=0, epsrel=1e-13)[0]
    den = num + integrate.quad(f, 0.0, z, epsabs=0, epsrel=1e-13)[0]
    return num / den


def series(A, boundary, K=K_SERIES, z=Z_DIFF, c=C):
    """Large-time series (2) for the DDM first-passage density, x0 = 0."""
    a = 2.0 * z / c
    v = A / c
    w = 0.5
    if boundary > 0:
        v, w = -v, 1.0 - w
    k = np.arange(1, K + 1, dtype=float)
    lam = 0.5 * v**2 + k**2 * np.pi**2 / (2.0 * a**2)
    coef = np.pi / a**2 * np.exp(-v * a * w) * k * np.sin(k * np.pi * w)
    return coef, lam


def fpt_density(t, A, boundary):
    coef, lam = series(A, boundary)
    t = np.asarray(t, float)
    g = np.exp(-np.outer(np.maximum(t, 1e-3), lam)) @ coef
    return np.where(t < 1e-3, 0.0, g)


def conditional_cdf(A, boundary):
    p = float(ddm_p(A))
    pb = p if boundary > 0 else 1.0 - p
    coef, lam = series(A, boundary)
    tg = np.linspace(0.0, T_MAX, 20_001)
    tail = np.exp(-np.outer(np.maximum(tg, 1e-3), lam)) @ (coef / lam)
    F = np.clip(np.where(tg < 1e-3, 0.0, (pb - tail) / pb), 0.0, 1.0)
    return lambda t: np.interp(t, tg, F)


# ---------------------------------------------------------------------------
# Model
def lca_params(A, K, lam):
    """Inputs and couplings (I1, I2, k, w) for drift A, K = k + w, lambda = w - k."""
    return 0.5 * (K * S_STAR + A), 0.5 * (K * S_STAR - A), 0.5 * (K - lam), 0.5 * (K + lam)


def ou_step_coeffs(r, dt):
    """Exact OU step dX = (a - r X) dt + C dW: (decay, gain for a, noise SD)."""
    r = np.asarray(r, float)
    rs = np.where(r == 0.0, 1.0, r)
    decay = np.exp(-r * dt)
    gain = np.where(r == 0.0, dt, -np.expm1(-r * dt) / rs)
    sd = C * np.sqrt(np.where(r == 0.0, dt, -np.expm1(-2.0 * r * dt) / (2.0 * rs)))
    return decay, gain, sd


def simulate(A, K, lam, rng, criterion, dt, t_max=T_MAX, rectify=True):
    """Simulate independent LCA trials (one per entry of A, K, lam).

    criterion "difference": stop at |x1 - x2| >= Z_DIFF; "absolute": stop at
    x_i >= Z_ABS. Returns choice (+1 unit 1, -1 unit 2, 0 undecided), decision
    time (s) and a flag per trial that rectification acted at least once."""
    A, K, lam = (a.ravel().astype(float) for a in np.broadcast_arrays(A, K, lam))
    n = A.size
    I1, I2, k, w = lca_params(A, K, lam)
    ds_, gs_, ss_ = ou_step_coeffs(k + w, dt)
    dy_, gy_, sy_ = ou_step_coeffs(k - w, dt)
    P = np.stack([ds_, gs_ * (I1 + I2), ss_, dy_, gy_ * (I1 - I2), sy_], axis=1)
    x1 = np.full(n, 0.5 * S_STAR)
    x2 = np.full(n, 0.5 * S_STAR)
    choice = np.zeros(n, np.int8)
    t_dec = np.full(n, np.nan)
    rect = np.zeros(n, bool)
    idx = np.arange(n)
    diff = criterion == "difference"
    b = Z_DIFF if diff else Z_ABS
    inv = 2.0 / ((C if diff else C_UNIT)**2 * dt)
    step = 0
    while idx.size and step * dt < t_max:
        m = idx.size
        s = x1 + x2
        y = x1 - x2
        sn = P[:, 0] * s + P[:, 1] + P[:, 2] * rng.standard_normal(m)
        yn = P[:, 3] * y + P[:, 4] + P[:, 5] * rng.standard_normal(m)
        x1n = 0.5 * (sn + yn)
        x2n = 0.5 * (sn - yn)
        if rectify:
            neg = (x1n < 0.0) | (x2n < 0.0)
            if neg.any():
                rect[idx[neg]] = True
                np.maximum(x1n, 0.0, out=x1n)
                np.maximum(x2n, 0.0, out=x2n)
        if diff:
            u0, u1, v0, v1 = y, x1n - x2n, -y, x2n - x1n
        else:
            u0, u1, v0, v1 = x1, x1n, x2, x2n
        up = u1 >= b
        lo = v1 >= b
        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
            fu = (b - u0) / (u1 - u0)
            fl = (b - v0) / (v1 - v0)
            pu = np.exp(-inv * (b - u0) * (b - u1))
            pl = np.exp(-inv * (b - v0) * (b - v1))
        both = up & lo
        up = up & ~(both & (fl < fu))
        lo = lo & ~(both & ~(fl < fu))
        inside = ~(up | lo)
        uu = rng.random(m)
        bu = inside & (uu < pu)
        bl = inside & ~bu & (uu < pu + pl)
        frac = np.where(up, fu, np.where(lo, fl, 0.5))
        up |= bu
        lo |= bl
        done = up | lo
        if done.any():
            d = idx[done]
            choice[d] = np.where(up[done], 1, -1)
            t_dec[d] = (step + frac[done]) * dt
            keep = ~done
            idx, x1, x2, P = idx[keep], x1n[keep], x2n[keep], P[keep]
        else:
            x1, x2 = x1n, x2n
        step += 1
    return choice, t_dec, rect


def sample_paths(A, K, n, rng, dt=2e-4, t_max=1.5):
    """A few absolute-criterion trajectories for display (grid test only)."""
    I1, I2, k, w = lca_params(A, K, 0.0)
    steps = int(round(t_max / dt))
    X = np.full((n, 2, steps + 1), np.nan)
    hit = np.zeros(n, int)
    for i in range(n):
        x = np.array([0.5 * S_STAR, 0.5 * S_STAR])
        X[i, :, 0] = x
        for j in range(steps):
            x = x + (np.array([I1, I2]) - k * x - w * x[::-1]) * dt \
                + C_UNIT * np.sqrt(dt) * rng.standard_normal(2)
            x = np.maximum(x, 0.0)
            X[i, :, j + 1] = x
            if x.max() >= Z_ABS:
                hit[i] = 1 if x[0] >= x[1] else -1
                break
    return np.arange(steps + 1) * dt, X, hit


# ---------------------------------------------------------------------------
def summarise(choice, t, p_th, m_th):
    n = choice.size
    p_sim = np.mean(choice == 1)
    se_p = np.sqrt(p_th * (1.0 - p_th) / n)
    sd = t.std(ddof=1)
    se_m = sd / np.sqrt(n)
    return dict(n=n, p_th=p_th, p_sim=p_sim, se_p=se_p, z_p=(p_sim - p_th) / se_p,
                m_th=m_th, m_sim=t.mean(), se_m=se_m, z_m=(t.mean() - m_th) / se_m)


def z_from_t(m):
    """Bound z for which the DDM mean decision time at A_CONV equals m."""
    return optimize.brentq(lambda z: float(ddm_t(A_CONV, z)) - m, 1e-4, 0.3, xtol=1e-14)


def validate():
    rng = np.random.default_rng(SEED)
    R = {}

    # 0. closed forms against each other
    errs = []
    for A in DRIFTS:
        p, t = ou_forms(A, 0.0)
        errs += [abs(p - ddm_p(A)) / ddm_p(A), abs(t - ddm_t(A)) / ddm_t(A)]
    for lam in LAMS:
        for A in A_OU:
            p = ou_forms(A, lam)[0]
            errs.append(abs(p - ou_p_quad(A, lam)) / p)
    R["form_err"] = max(errs)
    print(f"0. OU closed form (3) vs DDM (1) at lambda = 0 and vs quad: max relative "
          f"difference {R['form_err']:.1e} (tolerance {FORM_TOL:.0e})")

    # 1. difference criterion, k = w: exactly the DDM
    t1 = _time.time()
    KK, AA = np.meshgrid(K_BAL, DRIFTS, indexing="ij")
    Kt = np.repeat(KK.ravel(), N_TRIALS)
    At = np.repeat(AA.ravel(), N_TRIALS)
    ch, td, rc = simulate(At, Kt, 0.0, rng, "difference", DT_DIFF)
    assert (ch != 0).all(), "undecided trials at t_max"
    bal = []
    for K, A in zip(KK.ravel(), AA.ravel()):
        sel = (Kt == K) & (At == A)
        r = summarise(ch[sel], td[sel], float(ddm_p(A)), float(ddm_t(A)))
        r.update(K=K, A=A, n_rect=int(rc[sel].sum()), ks={})
        for b in (1, -1):
            tb = td[sel][ch[sel] == b]
            if tb.size >= 100:
                r["ks"][b] = stats.kstest(tb, conditional_cdf(A, b)).pvalue
        bal.append(r)
    R["bal"] = bal
    R["hist"] = dict(A=0.2, choice=ch[(Kt == 160) & (At == 0.2)], t=td[(Kt == 160) & (At == 0.2)])
    print(f"1. Difference criterion, k = w, {N_TRIALS} trials per condition, dt = "
          f"{DT_DIFF * 1e3:.1f} ms, {_time.time() - t1:.1f} s")
    print("   K     A     P(1) sim  DDM      z     | mean DT sim  DDM (s)   z     | KS p (1, 2)  | rectified trials")
    for r in bal:
        print(f"   {r['K']:4.0f}  {r['A']:.2f}  {r['p_sim']:.4f}  {r['p_th']:.4f}  {r['z_p']:+5.2f} | "
              f"{r['m_sim']:.4f}  {r['m_th']:.4f}  {r['z_m']:+5.2f} | "
              f"{r['ks'].get(1, np.nan):.3f}, {r['ks'].get(-1, np.nan):.3f} | {r['n_rect']}")

    # 2. difference criterion, k != w: the OU model
    t1 = _time.time()
    LL, AA = np.meshgrid(LAMS, A_OU, indexing="ij")
    Lt = np.repeat(LL.ravel(), N_TRIALS)
    At = np.repeat(AA.ravel(), N_TRIALS)
    ch, td, rc = simulate(At, K_OU, Lt, rng, "difference", DT_DIFF)
    assert (ch != 0).all(), "undecided trials at t_max"
    ou = []
    for lam, A in zip(LL.ravel(), AA.ravel()):
        sel = (Lt == lam) & (At == A)
        p_th, m_th = ou_forms(A, lam)
        r = summarise(ch[sel], td[sel], p_th, m_th)
        r.update(lam=lam, A=A, n_rect=int(rc[sel].sum()),
                 p_ddm=float(ddm_p(A)), m_ddm=float(ddm_t(A)))
        ou.append(r)
    R["ou"] = ou
    print(f"2. Difference criterion, K = {K_OU:.0f}, lambda = w - k != 0, {_time.time() - t1:.1f} s")
    print("   lambda  A     P(1) sim  OU       z     | mean DT sim  OU (s)    z     | (DDM P, DT) | rectified")
    for r in ou:
        print(f"   {r['lam']:+5.1f}  {r['A']:.2f}  {r['p_sim']:.4f}  {r['p_th']:.4f}  {r['z_p']:+5.2f} | "
              f"{r['m_sim']:.4f}  {r['m_th']:.4f}  {r['z_m']:+5.2f} | ({r['p_ddm']:.4f}, "
              f"{r['m_ddm']:.4f}) | {r['n_rect']}")

    # 3. absolute criterion (Usher & McClelland), k = w: DDM only as K -> infinity
    t1 = _time.time()
    conv = []
    p_th, m_th = float(ddm_p(A_CONV)), float(ddm_t(A_CONV))
    for K in K_CONV:
        dt = min(DT_DIFF, KDT / K)
        ch, td, rc = simulate(np.full(N_TRIALS, A_CONV), K, 0.0, rng, "absolute", dt)
        assert (ch != 0).all(), "undecided trials at t_max"
        r = summarise(ch, td, p_th, m_th)
        er = 1.0 - r["p_sim"]
        se_er = np.sqrt(er * (1.0 - er) / r["n"])
        zer = C**2 / (2.0 * A_CONV) * np.log((1.0 - er) / er)
        se_zer = C**2 / (2.0 * A_CONV) * se_er / (er * (1.0 - er))
        se_t = td.std(ddof=1) / np.sqrt(td.size)
        zdt = z_from_t(td.mean())
        h = 1e-6
        dTdz = (float(ddm_t(A_CONV, zdt + h)) - float(ddm_t(A_CONV, zdt - h))) / (2 * h)
        se_zdt = se_t / dTdz
        r.update(K=K, dt=dt, rect_frac=rc.mean(), z_er=zer, se_zer=se_zer, z_dt=zdt, se_zdt=se_zdt,
                 z_cons=(zer - zdt) / np.hypot(se_zer, se_zdt),
                 delta=Z_DIFF - zdt, kappa=(Z_DIFF - zdt) * np.sqrt(K) / C)
        conv.append(r)
    R["conv"] = conv
    print(f"3. Absolute criterion x_i >= {Z_ABS}, k = w, A = {A_CONV}, {N_TRIALS} trials per K, "
          f"{_time.time() - t1:.1f} s (DDM limit P = {p_th:.4f}, DT = {m_th:.4f} s)")
    print("   K       dt (ms)  P(1) sim  z(DDM)  | mean DT  z(DDM)   | z_eff from ER   from DT       "
          "z(agree) | delta     delta sqrt(K)/c | rectified")
    for r in conv:
        print(f"   {r['K']:6.2f}  {r['dt'] * 1e3:.3f}    {r['p_sim']:.4f}  {r['z_p']:+6.2f}  | "
              f"{r['m_sim']:.4f}  {r['z_m']:+7.2f} | {r['z_er']:.5f}±{r['se_zer']:.5f}  "
              f"{r['z_dt']:.5f}±{r['se_zdt']:.5f}  {r['z_cons']:+5.2f}  | {r['delta']:.5f}  "
              f"{r['kappa']:.3f}          | {r['rect_frac']:.4f}")
    Ks = np.array([r["K"] for r in conv])
    dl = np.array([r["delta"] for r in conv])
    fit = Ks >= K_FIT_MIN
    R["slope"] = np.polyfit(np.log(Ks[fit]), np.log(dl[fit]), 1)[0]
    print(f"   delta vs K for K >= {K_FIT_MIN:.0f}: log-log slope {R['slope']:.3f} "
          f"(scaling argument -0.5, tolerance {SLOPE_TOL})")

    # ---------------- assertions ----------------
    zs = [v for r in bal + ou for v in (r["z_p"], r["z_m"])]
    R["max_z"] = max(abs(np.array(zs)))
    R["n_z"] = len(zs)
    kss = [p for r in bal for p in r["ks"].values()]
    R["min_ks"] = min(kss)
    R["n_ks"] = len(kss)
    R["n_rect12"] = sum(r["n_rect"] for r in bal + ou)
    cons = [abs(r["z_cons"]) for r in conv if r["K"] >= K_CONS_MIN]
    R["max_cons"] = max(cons)
    print(f"Summary: checks 1-2 max |z| {R['max_z']:.2f} over {R['n_z']} comparisons (tolerance "
          f"{Z_TOL}); min KS p {R['min_ks']:.4f} over {R['n_ks']} tests (tolerance {KS_P_MIN:.0e}); "
          f"rectified trials {R['n_rect12']}")
    print(f"         check 3 z_eff agreement max |z| {R['max_cons']:.2f} for K >= {K_CONS_MIN:.0f}; "
          f"slope {R['slope']:.3f}")
    assert R["form_err"] < FORM_TOL, "closed forms disagree"
    assert R["max_z"] < Z_TOL, f"LCA (difference criterion) off DDM/OU closed form, |z| = {R['max_z']:.2f}"
    assert R["min_ks"] > KS_P_MIN, f"KS test rejects DDM distribution, p = {R['min_ks']:.2e}"
    assert R["max_cons"] < Z_TOL, "absolute-criterion LCA does not act as a DDM with one effective bound"
    assert abs(R["slope"] + 0.5) < SLOPE_TOL, f"delta does not scale as K^-1/2 (slope {R['slope']:.3f})"
    print("All LCA checks passed.")

    R["paths"] = sample_paths(0.1, 25.0, 8, rng)
    return R


# ---------------------------------------------------------------------------
def plot(R):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    cols = {40.0: "tab:blue", 160.0: "tab:green", 640.0: "tab:purple"}

    # A: sample paths
    ax = axes[0, 0]
    t, X, hit = R["paths"]
    for i in range(X.shape[0]):
        ax.plot(t, X[i, 0], color="tab:blue", lw=0.8)
        ax.plot(t, X[i, 1], color="tab:red", lw=0.8)
    ax.axhline(Z_ABS, color="k", lw=1, label=f"threshold Z = {Z_ABS}")
    ax.axhline(0.5 * S_STAR, color="0.6", lw=0.6, ls=":", label="start s*/2")
    ax.set_xlabel("time from stimulus onset (s)")
    ax.set_ylabel("activity")
    ax.set_title("A. LCA paths, k = w = 12.5, A = 0.1 (x₁ blue, x₂ red)")
    ax.legend(fontsize=8)

    # B, C: balanced, difference criterion vs DDM
    Ad = np.linspace(0, 0.42, 200)
    for ax, key, th, lab in ((axes[0, 1], "p", ddm_p, "P(choose 1)"),
                             (axes[0, 2], "m", ddm_t, "mean decision time (s)")):
        ax.plot(Ad, th(Ad), color="k", lw=1.2, label="pure DDM closed form")
        for j, K in enumerate(K_BAL):
            rr = [r for r in R["bal"] if r["K"] == K]
            ax.errorbar([r["A"] + 0.006 * (j - 1) for r in rr], [r[f"{key}_sim"] for r in rr],
                        yerr=[2 * r[f"se_{key}"] for r in rr], fmt="o", ms=4, capsize=2,
                        color=cols[K], label=f"LCA k = w = {K / 2:.0f} ± 2 SE")
        ax.set_xlabel("input difference A = I₁ − I₂ (s⁻¹)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 1].set_title("B. Leak = inhibition, |x₁ − x₂| ≥ z: accuracy")
    axes[0, 2].set_title("C. Leak = inhibition: mean decision time")

    # D: DT distribution vs DDM series
    ax = axes[0, 3]
    H = R["hist"]
    edges = np.arange(0, 1.6 + 1e-9, 0.02)
    n = H["t"].size
    for b, col, sgn, lab in ((1, "tab:blue", 1, "choice 1"), (-1, "tab:red", -1, "choice 2")):
        tb = H["t"][H["choice"] == b]
        cnt, _ = np.histogram(tb, edges)
        ax.bar(edges[:-1], sgn * cnt / (n * np.diff(edges)), width=np.diff(edges), align="edge",
               color=col, alpha=0.35, label=f"LCA {lab} ({tb.size})")
    tg = np.linspace(0, 1.6, 641)
    ax.plot(tg, fpt_density(tg, H["A"], 1), color="tab:blue", lw=1.5, label="DDM series g₊")
    ax.plot(tg, -fpt_density(tg, H["A"], -1), color="tab:red", lw=1.5, label="DDM series −g₋")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("decision time (s)")
    ax.set_ylabel("density (choice 2 mirrored)")
    ax.set_title(f"D. k = w = 80, A = 0.2; KS p ≥ {R['min_ks']:.3f} in {R['n_ks']} tests")
    ax.legend(fontsize=8)

    # E, F: OU model
    lg = np.linspace(-5, 5, 81)
    for ax, i, lab in ((axes[1, 0], 0, "P(choose 1)"), (axes[1, 1], 1, "mean decision time (s)")):
        for A, col in zip(A_OU, ("tab:orange", "tab:cyan")):
            ax.plot(lg, [ou_forms(A, l, n=20_001)[i] for l in lg], color=col, lw=1.2,
                    label=f"OU closed form, A = {A}")
            rr = [r for r in R["ou"] if r["A"] == A]
            key = "p" if i == 0 else "m"
            ax.errorbar([r["lam"] for r in rr], [r[f"{key}_sim"] for r in rr],
                        yerr=[2 * r[f"se_{key}"] for r in rr], fmt="o", ms=5, capsize=2, color=col,
                        label=f"LCA, A = {A} ± 2 SE")
            th0 = ddm_p(A) if i == 0 else ddm_t(A)
            rb = [r for r in R["bal"] if r["K"] == K_OU and r["A"] == A][0]
            ax.errorbar([0.0], [rb[f"{key}_sim"]], yerr=[2 * rb[f"se_{key}"]], fmt="s", ms=5,
                        capsize=2, mfc="none", color=col)
            ax.plot([0.0], [th0], "k+", ms=10)
        ax.axvline(0, color="0.6", lw=0.6, ls=":")
        ax.set_xlabel("λ = w − k (s⁻¹)  (< 0 leak-dominated, > 0 inhibition-dominated)")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[1, 0].set_title(f"E. k ≠ w, K = {K_OU:.0f}: accuracy vs OU model (+ DDM at λ = 0)")
    axes[1, 1].set_title("F. k ≠ w: mean decision time vs OU model")

    # G: absolute criterion, effective bound
    Cv = R["conv"]
    Ks = np.array([r["K"] for r in Cv])
    ax = axes[1, 2]
    ax.errorbar(Ks / 1.04, [r["z_er"] for r in Cv], yerr=[2 * r["se_zer"] for r in Cv], fmt="o",
                ms=5, capsize=2, color="tab:blue", label="z_eff from error rate")
    ax.errorbar(Ks * 1.04, [r["z_dt"] for r in Cv], yerr=[2 * r["se_zdt"] for r in Cv], fmt="s",
                ms=5, capsize=2, color="tab:red", label="z_eff from mean DT")
    ax.axhline(Z_DIFF, color="k", lw=1, label=f"DDM limit z = 2Z − s* = {Z_DIFF}")
    ax.set_xscale("log")
    ax.set_xlabel("K = k + w (s⁻¹), k = w")
    ax.set_ylabel("effective DDM bound")
    ax.set_title(f"G. Absolute threshold x_i ≥ Z, A = {A_CONV}: DDM-equivalent bound")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # H: delta scaling
    ax = axes[1, 3]
    dl = np.array([r["delta"] for r in Cv])
    se = np.array([r["se_zdt"] for r in Cv])
    ax.errorbar(Ks, dl, yerr=2 * se, fmt="o", ms=5, capsize=2, color="tab:red",
                label="z − z_eff (from mean DT) ± 2 SE")
    fit = Ks >= K_FIT_MIN
    pf = np.polyfit(np.log(Ks[fit]), np.log(dl[fit]), 1)
    kg = np.geomspace(Ks.min(), Ks.max(), 50)
    ax.plot(kg, np.exp(np.polyval(pf, np.log(kg))), "k-", lw=1,
            label=f"fit K ≥ {K_FIT_MIN:.0f}: slope {pf[0]:.3f}")
    kap = np.mean([r["kappa"] for r in Cv if r["K"] >= K_FIT_MIN])
    ax.plot(kg, kap * C / np.sqrt(kg), "k--", lw=1, label=f"{kap:.2f} c/√K")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K = k + w (s⁻¹)")
    ax.set_ylabel("bound shift δ")
    ax.set_title("H. Approach to the DDM: δ ∝ K^(−1/2)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)

    plt.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / "leaky_competing_accumulator.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Figure written to {out.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    t0 = _time.time()
    print(f"RNG seed: {SEED} (PCG64)")
    results = validate()
    plot(results)
    print(f"Runtime {_time.time() - t0:.1f} s")
