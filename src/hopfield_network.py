"""Hopfield network: storage capacity and retrieval overlap of the
Hebbian associative memory.

Model (Hopfield 1982). N binary neurons s_i = +/-1 store p random patterns
xi^mu (i.i.d. +/-1 with probability 1/2) in symmetric Hebbian couplings

    J_ij = (1/N) sum_mu xi_i^mu xi_j^mu,    J_ii = 0,

and evolve by zero-temperature asynchronous dynamics: a randomly chosen
neuron is set to s_i = sgn(h_i), h_i = sum_j J_ij s_j (a neuron with
h_i = 0 keeps its state). The energy E = -1/2 sum_ij J_ij s_i s_j falls by
2|h_i| at every flip, so the dynamics stops at a fixed point. The load is
alpha = p/N.

Integrator. There is no time step: the dynamics is the discrete map itself.
Updates of stable neurons do nothing, so random asynchronous updating is
simulated by its jump chain: at each step one of the currently unstable
neurons (s_i h_i < 0) is chosen uniformly at random and flipped, and the
fields are updated as h <- h + 2 s_i J_:,i. Couplings and fields are kept
as integers (N J_ij and N h_i), so the dynamics and the tie rule are exact.
Time is reported in sweeps, flips / N.

Protocol. Every retrieval run starts exactly at a stored pattern and is
relaxed to a fixed point; the result is the final overlap
m = (1/N) sum_i xi_i^mu s_i. Each network is tested from PER_NET of its
patterns. Retrieval counts as a success if m > M_SUCCESS = 0.7 (beyond
capacity the state falls to m ~ 0.3).

Closed forms used for validation:

  (a) One-step stability of a stored bit (derived here). At s = xi^1,
      xi_i^1 N h_i = (N - 1) + S_i with S_i a sum of K = (N - 1)(p - 1)
      terms xi_i^1 xi_i^mu xi_j^1 xi_j^mu (j != i, mu != 1). For fixed i
      these are independent and uniform on +/-1, so S_i ~ 2 Bin(K, 1/2) - K
      exactly, and the bit is unstable with probability
          P_flip = P[Bin(K, 1/2) < (K - N + 1)/2],
      which tends to (1/2) erfc(1/sqrt(2 alpha)) (Hopfield 1982; Hertz,
      Krogh & Palmer 1991, eq. 2.13).
  (b) Retrieval overlap at T = 0, replica-symmetric theory of Amit,
      Gutfreund & Sompolinsky (1985, 1987): m = erf(y) with y the largest
      root of
          erf(y) = y (sqrt(2 alpha) + (2/sqrt(pi)) exp(-y^2)),
      which has a solution only for alpha <= alpha_c = 0.1379, where
      m_c = 0.9674 (AGS 1987; Hertz, Krogh & Palmer 1991, sec. 2.5).

Validation (validate()):
1. Implementation: along one run beyond capacity, E computed directly from
   J falls at every flip by exactly 2|h_i| (integer arithmetic).
2. Fraction of unstable bits at the stored patterns, every network, vs the
   exact binomial (a): |z| < 4 for every (N, alpha) and chi^2/dof < 2.
3. Error fraction eps = (1 - m)/2 of successful retrievals at N = 8000 vs
   the RS value (b) at alpha = 0.08, 0.10, 0.12: relative error < 20%.
4. Capacity. For each N the success fraction against alpha is fitted with a
   logistic function (maximum likelihood); its midpoint alpha_50(N) is
   extrapolated as alpha_50 = alpha_inf + a / sqrt(N) (finite-size form
   chosen here) and alpha_inf must lie within 5% of the published 0.138.

RNG: NumPy PCG64 seeded with SEED (printed), one spawned stream per
(N, alpha) condition and one more for the energy run; the bootstrap uses
SeedSequence([SEED, 1]). SciPy is used only for the binomial distribution,
erf, root finding and the fits.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erf, erfc
from scipy.stats import binom
from scipy.optimize import brentq, minimize

# ---------------------------------------------------------------- parameters
SEED = 1985
N_LIST = [500, 1000, 2000, 4000, 8000]
ALPHAS_C = np.round(np.arange(0.120, 0.1801, 0.005), 3)    # capacity scan, every N
ALPHAS_M = np.round(np.arange(0.06, 0.1151, 0.01), 3)      # extra overlap scan, N_M only
N_M = 8000
TRIALS = {500: 400, 1000: 300, 2000: 200, 4000: 120, 8000: 100}   # retrievals per (N, alpha)
PER_NET = 20                         # patterns tested per network
M_SUCCESS = 0.7
ALPHA_C_PUBLISHED = 0.138            # Amit, Gutfreund & Sompolinsky (1985, 1987)
TRAJ = [(8000, 0.10), (8000, 0.17)]  # recorded overlap trajectories
ENERGY_RUN = (1000, 0.17)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- model
def make_network(rng, N, p):
    """Patterns xi (p x N, int8) and integer couplings N*J (int16 or int32)."""
    xi = rng.choice(np.array([-1, 1], dtype=np.int8), size=(p, N))
    xf = xi.astype(np.float32)
    J = xf.T @ xf                              # exact: integers |.| <= p < 2^24
    J = J.astype(np.int16 if p < 32000 else np.int32)
    np.fill_diagonal(J, 0)
    return xi, J


def fields(xi, s):
    """N*h = sum_j N*J_ij s_j, computed from the patterns (exact integers)."""
    p = xi.shape[0]
    xf = xi.astype(np.float64)
    return np.rint(xf.T @ (xf @ s) - p * s).astype(np.int64)


def relax(J, xi, s, rng, record=None, energy=False):
    """Zero-temperature asynchronous dynamics until no neuron is unstable.

    Jump chain: flip a uniformly chosen unstable neuron; h updated in place.
    record: pattern vector -> returns overlap after every flip.
    energy: also returns the directly computed N*E after every flip and the
    expected decrements 2|N h_i|."""
    s = s.astype(np.int64).copy()
    h = fields(xi, s)
    N = s.size
    ms, Es, dE = [], [], []
    if record is not None:
        ms.append(record @ s / N)
    if energy:
        Es.append(-0.5 * s @ (J.astype(np.int64) @ s))
    nflip = 0
    while True:
        u = np.flatnonzero(s * h < 0)
        if u.size == 0:
            break
        i = u[rng.integers(u.size)]
        if energy:
            dE.append(2 * abs(h[i]))
        s[i] = -s[i]
        h += 2 * s[i] * J[i]
        nflip += 1
        if record is not None:
            ms.append(record @ s / N)
        if energy:
            Es.append(-0.5 * s @ (J.astype(np.int64) @ s))
    return s, nflip, np.array(ms), np.array(Es), np.array(dE)


def p_flip_exact(N, p):
    K = (N - 1) * (p - 1)
    # unstable iff (N-1) + S < 0, S = 2B - K  <=>  B < (K - N + 1)/2
    thr = (K - N + 1) / 2.0
    kmax = np.ceil(thr) - 1                    # largest integer B strictly below thr
    return binom.cdf(kmax, K, 0.5)


def rs_overlap(alpha):
    """T = 0 replica-symmetric retrieval overlap (AGS); NaN above alpha_c."""
    g = lambda y: erf(y) - y * (np.sqrt(2 * alpha) + 2 / np.sqrt(np.pi) * np.exp(-y ** 2))
    y_c = 1.5146                               # near the maximiser of the alpha(y) curve
    if g(y_c) < 0:
        return np.nan
    return erf(brentq(g, y_c, max(20.0, 2.0 / np.sqrt(2 * alpha))))


def rs_alpha_c():
    y = np.linspace(0.5, 4.0, 400001)
    a = ((erf(y) - 2 * y / np.sqrt(np.pi) * np.exp(-y ** 2)) / (np.sqrt(2) * y)) ** 2
    k = a.argmax()
    return a[k], erf(y[k])


# ---------------------------------------------------------------- simulate
def run_condition(rng, N, alpha, trials, traj=False):
    p = int(round(alpha * N))
    nets = int(np.ceil(trials / PER_NET))
    ms, nfl, punst = [], [], []
    tr = None
    for k in range(nets):
        xi, J = make_network(rng, N, p)
        # one-step stability of every stored bit
        xf = xi.astype(np.float64)
        H = xf.T @ (xf @ xf.T) - p * xf.T       # N*h at s = xi^mu, column mu
        punst.append(np.mean(H * xf.T < 0))
        for mu in range(min(PER_NET, p)):
            rec = xi[mu].astype(np.int64) if (traj and k == 0 and mu == 0) else None
            s, nf, mtr, _, _ = relax(J, xi, xi[mu], rng, record=rec)
            ms.append(s @ xi[mu] / N)
            nfl.append(nf)
            if rec is not None:
                tr = mtr
        del J
    return dict(N=N, alpha=alpha, p=p, m=np.array(ms), nflip=np.array(nfl),
                punst=np.array(punst), traj=tr)


def simulate():
    ss = np.random.SeedSequence(SEED)
    conds = [(N, a) for N in N_LIST for a in ALPHAS_C] + [(N_M, a) for a in ALPHAS_M]
    streams = ss.spawn(len(conds) + 1)
    res = []
    for (N, a), st in zip(conds, streams[:-1]):
        t0 = _time.time()
        r = run_condition(np.random.default_rng(st), N, a, TRIALS[N], traj=(N, a) in TRAJ)
        res.append(r)
        print(f"N = {N:5d}, alpha = {a:.3f} (p = {r['p']:4d}): {len(r['m'])} retrievals, "
              f"success {np.mean(r['m'] > M_SUCCESS):.3f}, median m {np.median(r['m']):.4f}, "
              f"{time_str(t0)}")
    # energy run
    rng = np.random.default_rng(streams[-1])
    N, a = ENERGY_RUN
    xi, J = make_network(rng, N, int(round(a * N)))
    _, nf, _, Es, dE = relax(J, xi, xi[0], rng, energy=True)
    return dict(res=res, energy=dict(N=N, alpha=a, E=Es, dE=dE, nflip=nf))


def time_str(t0):
    return f"{_time.time() - t0:.1f} s"


# ---------------------------------------------------------------- validate
def logistic_mle(alphas, k, n):
    """Fit f(alpha) = 1/(1 + exp((alpha - a50)/w)) by maximum likelihood."""
    def nll(th):
        a50, lw = th
        f = 1 / (1 + np.exp((alphas - a50) / np.exp(lw)))
        f = np.clip(f, 1e-12, 1 - 1e-12)
        return -np.sum(k * np.log(f) + (n - k) * np.log(1 - f))
    f = k / n
    a0 = alphas[np.argmin(np.abs(f - 0.5))]
    best = minimize(nll, [a0, np.log(0.005)], method="Nelder-Mead",
                    options=dict(xatol=1e-7, fatol=1e-9, maxiter=4000))
    return best.x[0], np.exp(best.x[1])


def validate(R):
    res = R["res"]
    out = {}
    a_c, m_c = rs_alpha_c()
    print(f"\nRS theory (AGS): alpha_c = {a_c:.5f}, m_c = {m_c:.5f}")
    out["a_c"], out["m_c"] = a_c, m_c

    print("\n1. Energy along one run (N = %d, alpha = %.2f), directly from J" % (R["energy"]["N"], R["energy"]["alpha"]))
    E, dE = R["energy"]["E"], R["energy"]["dE"]
    steps = -np.diff(E)
    ok = np.all(steps > 0) and np.array_equal(steps, dE)
    print(f"   {R['energy']['nflip']} flips; N*E from {E[0]:.0f} to {E[-1]:.0f}; "
          f"every decrement > 0 and equal to 2|N h_i|: {ok}")
    assert ok
    out["energy_ok"] = ok

    print("\n2. Fraction of unstable bits at stored patterns vs exact binomial")
    zs, rows = [], []
    for r in res:
        pe = p_flip_exact(r["N"], r["p"])
        nb = r["N"] * r["p"]
        mean = r["punst"].mean()
        se = np.sqrt(pe * (1 - pe) / (nb * len(r["punst"])))
        se_net = r["punst"].std(ddof=1) / np.sqrt(len(r["punst"])) if len(r["punst"]) > 1 else se
        z = (mean - pe) / max(se, se_net)
        zs.append(z)
        rows.append((r["N"], r["alpha"], r["p"], mean, pe, z))
    zs = np.array(zs)
    for (N, a, p, mean, pe, z) in rows:
        if a in (0.06, 0.10, 0.12, 0.15, 0.18) and N in (500, 2000, 8000):
            print(f"   N = {N:5d}, alpha = {a:.3f}: sim {mean:.5f}   exact {pe:.5f}   "
                  f"Gaussian {0.5*erfc(1/np.sqrt(2*a)):.5f}   z = {z:+.2f}")
    chi2 = np.mean(zs ** 2)
    print(f"   {len(zs)} conditions: max |z| {np.abs(zs).max():.2f} (tolerance 4), "
          f"chi^2/dof {chi2:.2f} (tolerance 2)")
    assert np.abs(zs).max() < 4 and chi2 < 2
    out["stab_rows"], out["stab_z"], out["stab_chi2"] = rows, zs, chi2

    print(f"\n3. Error fraction (1 - m)/2 of successful retrievals at N = {N_M} vs RS theory")
    mrows = []
    for r in sorted([r for r in res if r["N"] == N_M], key=lambda r: r["alpha"]):
        ok_ = r["m"] > M_SUCCESS
        if ok_.sum() < 10:
            print(f"   alpha = {r['alpha']:.3f}: {ok_.sum()}/{ok_.size} retrieved, not compared")
            continue
        eps = (1 - r["m"][ok_]) / 2
        mrs = rs_overlap(r["alpha"])
        eps_rs = (1 - mrs) / 2 if np.isfinite(mrs) else np.nan
        se = eps.std(ddof=1) / np.sqrt(eps.size) if eps.size > 1 else np.nan
        mrows.append((r["alpha"], eps.mean(), se, eps_rs, ok_.mean()))
        rel = (eps.mean() - eps_rs) / eps_rs if np.isfinite(eps_rs) else np.nan
        print(f"   alpha = {r['alpha']:.3f}: sim {eps.mean():.3e} +/- {se:.1e} "
              f"({ok_.sum()}/{ok_.size} retrieved)   RS {eps_rs:.3e}   rel {100*rel:+.1f}%")
    mrows = np.array(mrows)
    out["mrows"] = mrows
    rels = []
    for a in (0.08, 0.10, 0.12):
        row = mrows[np.isclose(mrows[:, 0], a)][0]
        rels.append((row[1] - row[3]) / row[3])
    rels = np.array(rels)
    out["eps_rel"] = rels
    print(f"   asserted at alpha = 0.08, 0.10, 0.12: rel {', '.join(f'{100*x:+.1f}%' for x in rels)} "
          f"(tolerance 20%)")

    print("\n4. Capacity: logistic midpoint alpha_50(N), extrapolated in 1/sqrt(N)")
    a50, wid, a50se = [], [], []
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 1]))      # bootstrap stream
    fits = {}
    for N in N_LIST:
        rr = sorted([r for r in res if r["N"] == N and r["alpha"] in ALPHAS_C], key=lambda r: r["alpha"])
        al = np.array([r["alpha"] for r in rr])
        n = np.array([r["m"].size for r in rr])
        k = np.array([np.sum(r["m"] > M_SUCCESS) for r in rr])
        c, w = logistic_mle(al, k, n)
        # parametric bootstrap for the SE of the midpoint
        f = 1 / (1 + np.exp((al - c) / w))
        bs = [logistic_mle(al, rng.binomial(n, f), n)[0] for _ in range(200)]
        a50.append(c); wid.append(w); a50se.append(np.std(bs))
        fits[N] = (al, k, n, c, w)
        print(f"   N = {N:5d}: alpha_50 = {c:.4f} +/- {np.std(bs):.4f}, width {w:.4f}")
    a50, a50se = np.array(a50), np.array(a50se)
    x = 1 / np.sqrt(np.array(N_LIST, float))
    W = np.diag(1 / a50se ** 2)
    X = np.column_stack([np.ones_like(x), x])
    cov = np.linalg.inv(X.T @ W @ X)
    beta = cov @ X.T @ W @ a50
    resid = (a50 - X @ beta) / a50se
    a_inf, a_inf_se = beta[0], np.sqrt(cov[0, 0])
    rel_c = (a_inf - ALPHA_C_PUBLISHED) / ALPHA_C_PUBLISHED
    print(f"   alpha_50 = {a_inf:.4f} (+/- {a_inf_se:.4f}) + {beta[1]:.3f}/sqrt(N); "
          f"chi^2/dof {np.sum(resid**2)/(len(x)-2):.2f}")
    # comparison: 1/N form (reported only)
    X1 = np.column_stack([np.ones_like(x), x ** 2])
    b1 = np.linalg.solve(X1.T @ W @ X1, X1.T @ W @ a50)
    chi1 = np.sum(((a50 - X1 @ b1) / a50se) ** 2) / (len(x) - 2)
    print(f"   (1/N form, reported only: alpha_inf = {b1[0]:.4f}, chi^2/dof {chi1:.1f})")
    print(f"   alpha_inf vs published {ALPHA_C_PUBLISHED}: {100*rel_c:+.2f}% (tolerance 5%); "
          f"RS alpha_c {a_c:.4f}")
    out.update(a50=a50, a50se=a50se, wid=np.array(wid), fits=fits, beta=beta, a_inf=a_inf,
               a_inf_se=a_inf_se, rel_c=rel_c, b1=b1, chi_sqrt=np.sum(resid**2)/(len(x)-2), chi_1N=chi1)

    assert np.all(np.abs(rels) < 0.20), "error fraction does not match RS theory"
    assert abs(rel_c) < 0.05, "extrapolated capacity does not match 0.138"
    print("\nAll checks passed.")
    return out


# ---------------------------------------------------------------- plot
def plot(R, V):
    res = R["res"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))
    ag = np.linspace(0.03, V["a_c"] - 1e-6, 400)
    mg = np.array([rs_overlap(a) for a in ag])

    a = ax[0, 0]
    for j, N in enumerate(N_LIST):
        rr = sorted([r for r in res if r["N"] == N], key=lambda r: r["alpha"])
        al = np.array([r["alpha"] for r in rr])
        a.plot(al, [r["m"].mean() for r in rr], "o-", ms=3, lw=0.9, color=f"C{j}", label=f"N = {N}")
    a.plot(ag, mg, "k-", lw=1.4, label="RS theory (AGS)")
    a.plot([V["a_c"], V["a_c"]], [0, V["m_c"]], "k:", lw=1)
    a.set_xlabel("load α = p/N")
    a.set_ylabel("final overlap m (mean over runs)")
    a.set_xlim(0.055, 0.185)
    a.set_ylim(0, 1.02)
    a.legend(fontsize=8, loc="lower left")
    a.set_title(f"A  Overlap after relaxing from a stored pattern;\nRS α_c = {V['a_c']:.4f}, m_c = {V['m_c']:.4f}")

    a = ax[0, 1]
    mr = V["mrows"]
    a.semilogy(ag[ag > 0.04], (1 - mg[ag > 0.04]) / 2, "k-", lw=1.4, label="RS theory (1 − m)/2")
    a.errorbar(mr[:, 0], mr[:, 1], yerr=2 * mr[:, 2], fmt="o", color="C4", ms=4,
               label=f"N = {N_M}, successful runs (±2 SE)")
    a.set_xlabel("load α = p/N")
    a.set_ylabel("fraction of wrong bits at the fixed point")
    a.set_xlim(0.055, 0.15)
    a.set_ylim(5e-6, 3e-2)
    a.legend(fontsize=8)
    a.set_title("B  Retrieval errors vs replica-symmetric theory;\n"
                f"rel. error {', '.join(f'{100*x:+.0f}%' for x in V['eps_rel'])} at α = 0.08, 0.10, 0.12")

    a = ax[0, 2]
    g = np.linspace(0.115, 0.185, 300)
    for j, N in enumerate(N_LIST):
        al, k, n, c, w = V["fits"][N]
        a.plot(al, k / n, "o", ms=3.5, color=f"C{j}", label=f"N = {N}")
        a.plot(g, 1 / (1 + np.exp((g - c) / w)), "-", lw=0.9, color=f"C{j}")
    a.axvline(ALPHA_C_PUBLISHED, color="k", ls=":", lw=1, label="α_c = 0.138")
    a.set_xlabel("load α = p/N")
    a.set_ylabel(f"fraction of runs with m > {M_SUCCESS}")
    a.legend(fontsize=8)
    a.set_title("C  Retrieval success vs load, logistic fits")

    a = ax[1, 0]
    x = 1 / np.sqrt(np.array(N_LIST, float))
    a.errorbar(x, V["a50"], yerr=2 * V["a50se"], fmt="o", color="C0", label="α_50(N) (±2 SE)")
    xg = np.linspace(0, x.max() * 1.05, 50)
    a.plot(xg, V["beta"][0] + V["beta"][1] * xg, "C0--", lw=1,
           label=f"fit: {V['a_inf']:.4f} + {V['beta'][1]:.2f}/√N")
    a.plot(xg, V["b1"][0] + V["b1"][1] * xg ** 2, "C7:", lw=1,
           label=f"1/N form (reported only): {V['b1'][0]:.4f}")
    a.axhline(ALPHA_C_PUBLISHED, color="k", ls=":", lw=1, label="published 0.138")
    a.errorbar([0], [V["a_inf"]], yerr=[2 * V["a_inf_se"]], fmt="s", color="C3", ms=5)
    a.set_xlabel("1/√N")
    a.set_ylabel("capacity α_50(N)")
    a.legend(fontsize=8)
    a.set_title(f"D  Finite-size extrapolation: α_∞ = {V['a_inf']:.4f} ± {V['a_inf_se']:.4f}\n"
                f"({100*V['rel_c']:+.1f}% from 0.138)")

    a = ax[1, 1]
    g = np.linspace(0.05, 0.19, 200)
    a.semilogy(g, 0.5 * erfc(1 / np.sqrt(2 * g)), "k-", lw=1, label="Gaussian ½ erfc(1/√(2α))")
    for j, N in enumerate(N_LIST):
        rows = [r for r in V["stab_rows"] if r[0] == N]
        rows.sort(key=lambda r: r[1])
        a.plot([r[1] for r in rows], [r[3] for r in rows], "o", ms=3.5, color=f"C{j}", label=f"sim N = {N}")
        a.plot([r[1] for r in rows], [r[4] for r in rows], "_", ms=9, color="k")
    a.plot([], [], "k_", ms=9, label="exact binomial")
    a.set_xlabel("load α = p/N")
    a.set_ylabel("fraction of unstable bits at ξ^μ")
    a.legend(fontsize=7, ncol=2)
    a.set_title(f"E  One-step stability vs exact binomial:\nmax |z| {np.abs(V['stab_z']).max():.2f}, "
                f"χ²/dof {V['stab_chi2']:.2f} over {len(V['stab_z'])} conditions")

    a = ax[1, 2]
    for (N, al), col in zip(TRAJ, ["C2", "C3"]):
        r = [r for r in res if r["N"] == N and np.isclose(r["alpha"], al)][0]
        tr = r["traj"]
        a.semilogx(np.arange(1, tr.size + 1) / N, tr, color=col, lw=1.4,
                   label=f"N = {N}, α = {al:.2f}: {tr.size - 1} flips, final m = {tr[-1]:.4f}")
    a.set_xlabel("time (flips / N, log scale; first point = start)")
    a.set_ylabel("overlap with the starting pattern")
    a.set_ylim(0, 1.02)
    a.legend(fontsize=8)
    a.set_title("F  Relaxation from a stored pattern below and above capacity")

    for a_ in ax.flat:
        a_.title.set_fontsize(10)
    fig.suptitle("Hopfield network: Hebbian storage, retrieval overlap and capacity α_c ≈ 0.138", fontsize=13)
    fig.tight_layout()
    out = ROOT / "figures" / "hopfield_network.png"
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print(f"RNG seed: {SEED} (PCG64, one spawned stream per condition)")
    t0 = _time.time()
    R = simulate()
    V = validate(R)
    plot(R, V)
    print(f"Runtime {_time.time() - t0:.1f} s")
