"""Rate-based Hebbian learning with Oja's rule: a single linear neuron learns
the first principal component of its input.

Model (Oja 1982). A linear neuron with weight vector w in R^N receives
zero-mean inputs x drawn i.i.d. from N(0, C) and responds y = w . x. Plain
Hebbian learning, dw = eta y x, grows w without bound; Oja's rule adds a
normalising decay term,

    w_{n+1} = w_n + eta y_n (x_n - y_n w_n),     y_n = w_n . x_n,

which, for small eta, makes w converge to +/- the unit eigenvector e_1 of C
with the largest eigenvalue lambda_1 (Oja 1982; Hertz, Krogh & Palmer 1991,
sec. 8.2).

Integrator. The rule is itself a discrete map: one update per input sample,
learning rate eta (default ETA = 1e-3). "Time" is t = eta n, the time of the
averaged ODE below. Many independent chains (learners) run in parallel.

Input. N = 8 dimensions, eigenvalues LAMBDA = (4, 2, 1.5, 1, 0.7, 0.5, 0.3,
0.2) in a random orthonormal basis Q (QR of a seeded Gaussian matrix), so
C = Q diag(LAMBDA) Q^T and x = Q diag(sqrt(LAMBDA)) z with z ~ N(0, I).
Every chain starts from the same w_0, a random direction with |w_0| = 0.5.

Closed forms used for validation (independent of the learning code):

  (a) Averaged ODE. For small eta the mean trajectory follows
          dw/dt = C w - (w^T C w) w.
      Writing w = v/s with v(t) = e^{Ct} w_0 gives (s^2)' = (v^T v)', so
          w(t) = e^{Ct} w_0 / sqrt(1 - |w_0|^2 + |e^{Ct} w_0|^2),
      derived here; it is checked against scipy solve_ivp (DOP853).
  (b) Stationary misalignment. Linearising the map about w = e_1, in the
      eigenbasis the transverse components obey the AR(1) process
      d_k <- (1 - a_k) d_k + eta x_1 x_k with a_k = eta (lambda_1 - lambda_k)
      and Var(x_1 x_k) = lambda_1 lambda_k, so to leading order in eta
          E[sin^2 theta] = sum_{k>1} eta^2 lambda_1 lambda_k / (2 a_k - a_k^2)
                        ~ (eta/2) sum_{k>1} lambda_1 lambda_k/(lambda_1 - lambda_k),
      where theta is the angle between w and e_1 (derived here; the eta/2
      scaling is the standard small-gain result for stochastic approximation).

Validation (validate()):
1. Learned w vs the top eigenvector of C from numpy.linalg.eigh, after
   t = 20 at eta = 1e-3 (reported) and a further t = 5 at eta = 1e-4
   (asserted): |cos theta| > 0.99 in every chain (the roadmap criterion) and
   > 0.999 on average; |w| = 1 within 0.01 on average. At fixed eta the
   per-chain misalignment does not vanish (closed form (b)), which is why
   the learning rate is annealed for this check. Also reported against the
   top eigenvector of the sample covariance.
2. Averaged-ODE closed form vs solve_ivp (rtol 1e-12): max |error| < 1e-8.
   Ensemble mean of M = 8000 stochastic learners vs the closed form, max
   |error| in any component over 0 <= t <= 10, at eta = 4e-3 ... 5e-4: the
   deviation is a mean-field bias that must fall at order 1 +/- 0.25 in eta
   (log-log fit).
3. Stationary E[sin^2 theta] vs closed form (b) at eta = 4e-3 ... 2.5e-4:
   relative error < 5% at every eta, |z| < 4 at the smallest, and log-log
   slope 1 +/- 0.05.

RNG: NumPy PCG64 seeded with SEED (printed), one spawned stream per
experiment. SciPy is used only for the reference ODE solver.
"""

from pathlib import Path
import time as _time

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------- parameters
SEED = 1982
N = 8
LAMBDA = np.array([4.0, 2.0, 1.5, 1.0, 0.7, 0.5, 0.3, 0.2])
W0_NORM = 0.5
ETA = 1e-3                                  # default learning rate
T_LEARN = 20.0                              # time units (eta * steps)
ETA_FINE, T_FINE = 1e-4, 5.0                # annealing phase for check 1
M_LEARN = 400                               # chains for checks 1-2
T_ODE_CMP = 10.0
ODE_ETAS = np.array([4e-3, 2e-3, 1e-3, 5e-4])  # bias convergence, check 2
M_ODE = 8000
ETAS = np.array([4e-3, 2e-3, 1e-3, 5e-4, 2.5e-4])
M_STAT = 400                                # chains per eta, check 3
STAT_BURN, STAT_RUN = 8.0, 30.0             # in time units t = eta n
STAT_EVERY = 0.05                           # sampling interval (time units)

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- model
def make_input(rng):
    Q, R = np.linalg.qr(rng.standard_normal((N, N)))
    Q = Q * np.sign(np.diag(R))             # unique orthonormal basis
    C = Q @ np.diag(LAMBDA) @ Q.T
    A = Q * np.sqrt(LAMBDA)                 # x = A z
    return Q, C, A


def oja_step(w, x, eta):
    y = np.einsum("mi,mi->m", w, x)
    return w + eta * y[:, None] * (x - y[:, None] * w)


def run_chains(rng, A, w0, eta, T, M, record_every):
    """M independent Oja learners; returns record times and w snapshots."""
    n_steps = int(round(T / eta))
    rec = max(1, int(round(record_every / eta)))
    w = np.tile(w0, (M, 1)) if w0.ndim == 1 else w0.copy()
    ts, ws = [0.0], [w.copy()]
    for n in range(1, n_steps + 1):
        x = rng.standard_normal((M, N)) @ A.T
        w = oja_step(w, x, eta)
        if n % rec == 0:
            ts.append(n * eta)
            ws.append(w.copy())
    return np.array(ts), np.array(ws)       # ws: (T, M, N)


def ode_closed_form(t, C, w0):
    lam, U = np.linalg.eigh(C)
    c0 = U.T @ w0
    v = (U[None, :, :] * (np.exp(np.outer(t, lam)) * c0)[:, None, :]).sum(-1)
    s = np.sqrt(1.0 - w0 @ w0 + (v ** 2).sum(1))
    return v / s[:, None]


def sin2_closed_form(eta):
    l1, lk = LAMBDA[0], LAMBDA[1:]
    a = eta * (l1 - lk)
    return np.sum(eta ** 2 * l1 * lk / (2 * a - a ** 2))


# ---------------------------------------------------------------- simulate
def simulate():
    ss = np.random.SeedSequence(SEED)
    s_in, s_w0, s_learn, s_fine, s_samp, *s_rest = ss.spawn(5 + len(ETAS) + len(ODE_ETAS))
    s_stat, s_ode = s_rest[:len(ETAS)], s_rest[len(ETAS):]
    Q, C, A = make_input(np.random.Generator(np.random.PCG64(s_in)))
    g = np.random.Generator(np.random.PCG64(s_w0))
    w0 = g.standard_normal(N)
    w0 *= W0_NORM / np.linalg.norm(w0)

    t0 = _time.time()
    ts, ws = run_chains(np.random.Generator(np.random.PCG64(s_learn)), A, w0,
                        ETA, T_LEARN, M_LEARN, record_every=0.05)
    print(f"learning run: eta = {ETA}, {int(T_LEARN/ETA)} steps x {M_LEARN} chains "
          f"({_time.time()-t0:.1f} s)")

    t0 = _time.time()
    _, wfine = run_chains(np.random.Generator(np.random.PCG64(s_fine)), A, ws[-1],
                          ETA_FINE, T_FINE, M_LEARN, record_every=T_FINE)
    print(f"annealing run: eta = {ETA_FINE}, {int(T_FINE/ETA_FINE)} steps x {M_LEARN} chains "
          f"({_time.time()-t0:.1f} s)")

    Xs = np.random.Generator(np.random.PCG64(s_samp)).standard_normal((20000, N)) @ A.T

    ode = []
    for eta, s in zip(ODE_ETAS, s_ode):
        t0 = _time.time()
        tt, ww = run_chains(np.random.Generator(np.random.PCG64(s)), A, w0, eta,
                            T_ODE_CMP, M_ODE, record_every=0.1)
        ode.append(dict(eta=eta, t=tt, mean=ww.mean(1),
                        se=ww.std(1, ddof=1) / np.sqrt(M_ODE)))
        print(f"mean-field run: eta = {eta:g}, {int(T_ODE_CMP/eta)} steps x {M_ODE} chains "
              f"({_time.time()-t0:.1f} s)")

    stat = []
    for eta, s in zip(ETAS, s_stat):
        t0 = _time.time()
        tt, ww = run_chains(np.random.Generator(np.random.PCG64(s)), A, Q[:, 0], eta,
                            STAT_BURN + STAT_RUN, M_STAT, STAT_EVERY)
        keep = tt > STAT_BURN
        cos = np.einsum("tmi,i->tm", ww[keep], Q[:, 0]) / np.linalg.norm(ww[keep], axis=2)
        sin2 = 1 - cos ** 2                 # (samples, chains)
        stat.append(dict(eta=eta, chain_mean=sin2.mean(0),
                         norm=np.linalg.norm(ww[keep], axis=2),
                         d2=np.einsum("tmi,i->tm", ww[keep], Q[:, 1])))
        print(f"stationary run: eta = {eta:g}, {int((STAT_BURN+STAT_RUN)/eta)} steps x "
              f"{M_STAT} chains ({_time.time()-t0:.1f} s)")
    return dict(Q=Q, C=C, w0=w0, ts=ts, ws=ws, wfine=wfine[-1], Xs=Xs, stat=stat, ode=ode)


# ---------------------------------------------------------------- validate
def validate(R):
    V = {}
    Q, C, ws, ts = R["Q"], R["C"], R["ws"], R["ts"]
    lam, U = np.linalg.eigh(C)
    e1 = U[:, -1]
    print("\n1. Learned weights vs top eigenvector of C (numpy.linalg.eigh)")
    print(f"   eigh: lambda_1 = {lam[-1]:.6f}, gap to lambda_2 = {lam[-1]-lam[-2]:.6f}")
    Cs = np.cov(R["Xs"].T)
    es = np.linalg.eigh(Cs)[1][:, -1]
    for label, wf, eta in ((f"after t = {T_LEARN:g} at eta = {ETA:g}", ws[-1], ETA),
                           (f"after a further t = {T_FINE:g} at eta = {ETA_FINE:g}", R["wfine"], ETA_FINE)):
        nf = np.linalg.norm(wf, axis=1)
        cos = np.abs(wf @ e1) / nf
        print(f"   {label}: |cos theta| mean {cos.mean():.5f}, min {cos.min():.5f} over {len(cos)} chains; "
              f"mean sin^2 {np.mean(1-cos**2):.2e} (closed form (b) {sin2_closed_form(eta):.2e}); "
              f"|w| mean {nf.mean():.5f}")
    cos_s = np.abs(wf @ es) / nf
    V.update(cos_coarse=np.abs(ws[-1] @ e1) / np.linalg.norm(ws[-1], axis=1), cos=cos, nf=nf)
    print(f"   vs sample-covariance eigenvector (20000 samples): mean |cos| {cos_s.mean():.5f}")
    sgn = np.sign(wf @ e1)
    print(f"   sign: {np.sum(sgn > 0)} chains converged to +e1, {np.sum(sgn < 0)} to -e1")
    print("   assert (annealed): min |cos| > 0.99, mean > 0.999, |mean |w| - 1| < 0.01")
    assert cos.min() > 0.99 and cos.mean() > 0.999 and abs(nf.mean() - 1) < 0.01

    print("\n2. Averaged ODE dw/dt = Cw - (w'Cw)w")
    w0 = R["w0"]
    t_eval = ts[ts <= T_ODE_CMP + 1e-9]
    cf = ode_closed_form(t_eval, C, w0)
    sol = solve_ivp(lambda t, w: C @ w - (w @ C @ w) * w, (0, T_ODE_CMP), w0,
                    method="DOP853", rtol=1e-12, atol=1e-14, t_eval=t_eval)
    e_ode = np.abs(sol.y.T - cf).max()
    print(f"   closed form vs solve_ivp (DOP853, rtol 1e-12): max |err| {e_ode:.2e} (tolerance 1e-8)")
    errs = []
    for D in R["ode"]:
        cfD = ode_closed_form(D["t"], C, w0)
        e = np.abs(D["mean"] - cfD).max()
        errs.append(e)
        print(f"   ensemble mean of {M_ODE} chains, eta = {D['eta']:<6g} vs closed form, t <= {T_ODE_CMP:g}: "
              f"max |err| {e:.4f} (max SE of the mean {D['se'].max():.4f})")
    errs = np.array(errs)
    order = np.polyfit(np.log(ODE_ETAS), np.log(errs), 1)[0]
    V.update(t_eval=t_eval, cf=cf, e_ode=e_ode, ode_errs=errs, ode_order=order,
             e_ens=errs[ODE_ETAS == ETA][0])
    print(f"   order of the mean-field bias in eta: {order:.3f} (tolerance 1 +/- 0.25)")
    assert e_ode < 1e-8 and abs(order - 1) < 0.25

    print("\n3. Stationary misalignment E[sin^2 theta] vs linear-noise closed form")
    rows = []
    for S in R["stat"]:
        m = S["chain_mean"].mean()
        se = S["chain_mean"].std(ddof=1) / np.sqrt(len(S["chain_mean"]))
        cf3 = sin2_closed_form(S["eta"])
        rows.append((S["eta"], m, se, cf3))
        print(f"   eta = {S['eta']:<7g} sim {m:.4e} +/- {se:.1e}   closed form {cf3:.4e}   "
              f"rel {100*(m/cf3-1):+.2f}%  z {(m-cf3)/se:+.2f}   mean |w| {S['norm'].mean():.5f}")
    rows = np.array(rows)
    rel = rows[:, 1] / rows[:, 3] - 1
    z_last = (rows[-1, 1] - rows[-1, 3]) / rows[-1, 2]
    slope = np.polyfit(np.log(rows[:, 0]), np.log(rows[:, 1]), 1)[0]
    V.update(stat_rows=rows, rel=rel, slope=slope, z_last=z_last)
    print(f"   max |rel| {100*np.abs(rel).max():.2f}% (tolerance 5%); z at smallest eta {z_last:+.2f} "
          f"(tolerance 4); log-log slope {slope:.4f} (tolerance 1 +/- 0.05)")
    assert np.abs(rel).max() < 0.05 and abs(z_last) < 4 and abs(slope - 1) < 0.05
    print("\nAll checks passed.")
    return V


# ---------------------------------------------------------------- plot
def plot(R, V):
    Q, ws, ts = R["Q"], R["ws"], R["ts"]
    e1 = Q[:, 0]
    fig, ax = plt.subplots(2, 3, figsize=(16, 9.5))

    a = ax[0, 0]
    cosall = np.abs(np.einsum("tmi,i->tm", ws, e1)) / np.linalg.norm(ws, axis=2)
    for m in range(5):
        a.plot(ts, cosall[:, m], lw=0.7, color="0.6")
    cfc = np.abs(V["cf"] @ e1) / np.linalg.norm(V["cf"], axis=1)
    a.plot(ts, cosall.mean(1), color="C0", lw=1.5, label=f"mean of {ws.shape[1]} chains")
    a.plot(V["t_eval"], cfc, "k--", lw=1.2, label="averaged ODE, closed form")
    a.axhline(0.99, color="C3", lw=0.8, ls=":", label="0.99 criterion")
    a.set_xlabel("time t = η n")
    a.set_ylabel("|cos θ| between w and e₁")
    a.set_ylim(0, 1.02)
    a.legend(loc="lower right", fontsize=8)
    a.set_title(f"A  |cos θ| to e₁: mean {V['cos_coarse'].mean():.4f} at η = {ETA:g},\n"
                f"{V['cos'].mean():.5f} (min {V['cos'].min():.4f}) after annealing to η = {ETA_FINE:g}")

    a = ax[0, 1]
    D = [d for d in R["ode"] if d["eta"] == ETA][0]
    c = D["mean"] @ Q                         # eigenbasis components
    cfq = ode_closed_form(D["t"], R["C"], R["w0"]) @ Q
    for k in range(N):
        a.plot(D["t"], c[:, k], color=f"C{k}", lw=1.4, label=f"λ = {LAMBDA[k]:g}")
        a.plot(D["t"], cfq[:, k], "k--", lw=0.7)
    a.set_xlabel("time t = η n")
    a.set_ylabel("ensemble-mean w in eigenbasis")
    a.legend(fontsize=7, ncol=2)
    a.set_title(f"B  Mean of {M_ODE} chains (η = {ETA:g}) vs ODE closed form\n(dashed), max err {V['e_ens']:.4f}")

    a = ax[0, 2]
    p = R["Xs"][:3000] @ Q[:, :2]
    a.plot(p[:, 0], p[:, 1], ".", ms=1.5, color="0.7")
    wf = ws[-1, :20] @ Q[:, :2]
    for v in wf:
        s = np.sign(v[0])
        a.annotate("", xy=(4 * s * v[0], 4 * s * v[1]), xytext=(0, 0),
                   arrowprops=dict(arrowstyle="->", color="C0", lw=1.2, alpha=0.6))
    a.annotate("", xy=(0, 4), xytext=(0, 0), arrowprops=dict(arrowstyle="->", color="0.3", lw=1, ls="--"))
    a.text(4.1, 0.3, "learned w ×4", color="C0", fontsize=9)
    a.text(0.2, 4.0, "e₂ ×4", color="0.3", fontsize=9)
    a.set_aspect("equal")
    a.set_xlabel("x · e₁")
    a.set_ylabel("x · e₂")
    a.set_title("C  Inputs projected on (e₁, e₂) (λ = 4, 2);\nlearned w of 20 chains, sign-aligned")

    a = ax[1, 0]
    rows = V["stat_rows"]
    ef = np.geomspace(rows[0, 0] * 1.3, rows[-1, 0] / 1.3, 50)
    a.loglog(ef, [sin2_closed_form(e) for e in ef], "k-", lw=1, label="linear-noise closed form")
    a.errorbar(rows[:, 0], rows[:, 1], yerr=2 * rows[:, 2], fmt="o", color="C0",
               label="simulation (±2 SE)")
    a.set_xlabel("learning rate η")
    a.set_ylabel("stationary E[sin² θ]")
    a.legend(fontsize=8)
    a.set_title(f"D  Misalignment ∝ η: slope {V['slope']:.3f}, max rel err {100*np.abs(V['rel']).max():.2f}%")

    a = ax[1, 1]
    a.loglog(ODE_ETAS, V["ode_errs"], "o-", color="C0", label="max |mean w − ODE| (t ≤ 10)")
    a.loglog(ODE_ETAS, V["ode_errs"][0] * ODE_ETAS / ODE_ETAS[0], "k--", lw=0.8, label="slope 1")
    a.loglog(ODE_ETAS, [d["se"].max() for d in R["ode"]], "s:", color="0.5", ms=4,
             label="max SE of the mean")
    a.set_xlabel("learning rate η")
    a.set_ylabel("deviation from averaged ODE")
    a.legend(fontsize=8)
    a.set_title(f"E  Mean-field bias is O(η): order {V['ode_order']:.3f}; "
                f"closed form vs DOP853 {V['e_ode']:.1e}")

    a = ax[1, 2]
    S = R["stat"][2]
    sd = np.sqrt(S["eta"] ** 2 * LAMBDA[0] * LAMBDA[1] / (2 * S["eta"] * (LAMBDA[0] - LAMBDA[1])
                                                          - (S["eta"] * (LAMBDA[0] - LAMBDA[1])) ** 2))
    a.hist(S["d2"].ravel(), bins=80, density=True, color="C2", alpha=0.7, label="simulation")
    g = np.linspace(-5 * sd, 5 * sd, 200)
    a.plot(g, np.exp(-g ** 2 / (2 * sd ** 2)) / (sd * np.sqrt(2 * np.pi)), "k--", lw=1,
           label="Gaussian, AR(1) variance")
    a.set_xlabel("stationary w · e₂")
    a.set_ylabel("density")
    a.legend(fontsize=8)
    a.set_title(f"F  Transverse fluctuation along e₂ at η = {S['eta']:g}: sd {S['d2'].std():.4f} vs {sd:.4f}")

    for a_ in ax.flat:
        a_.title.set_fontsize(10)
    fig.suptitle("Oja's rule: a linear Hebbian neuron converges to the first principal component",
                 fontsize=13)
    fig.tight_layout()
    out = ROOT / "figures" / "oja_rule.png"
    fig.savefig(out, dpi=150)
    print(f"Figure written to {out.relative_to(ROOT)}")


if __name__ == "__main__":
    print(f"RNG seed: {SEED} (PCG64, one spawned stream per experiment)")
    t0 = _time.time()
    R = simulate()
    V = validate(R)
    plot(R, V)
    print(f"Runtime {_time.time() - t0:.1f} s")
