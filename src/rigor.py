"""
Scientific-completeness layer  (responses to review points 1-12)
=================================================================
This module exists because a domain review correctly flagged that several
headline numbers in the first revision mixed timescales or compared processes
acting on different state variables.  Everything below is the corrected,
self-consistent version.  Where a previous claim is withdrawn, it is said so
explicitly (see docs/ANALYSIS.md, "Addendum: scientific completeness").

Point-by-point:
  1  half-life coefficient: model.t_half is now the half-life of the std sigma
     (the plotted quantity); model.t_half_var keeps the variance half-life.
  2  no mixed quantiles: all channels are compared as per-generation log-RATES
     (channel_rates), never half-life vs mean-waiting-time.
  3  sigma's repair is modelled explicitly and is EXOGENOUS (eta).  Generational
     turnover lambda repairs competence (comp), NOT sigma.  Demonstrated in
     lambda_does_not_repair_sigma().
  4  q is clipped to [0,1] inside model.F, so fixed-point analysis stays valid.
  5  model.F is documented as the FRESH-meat static approximation; the
     atrophied / dynamic shares are F_atrophied / F_dynamic here.
  6  fixed_point_2d solves the joint (e, comp) fixed point with turnover lambda.
  7  w_opt is re-derived as the MMSE (minimum-MSE) linear-combination weight with
     the full covariance between the relayed answer and a direct query.
  8  I(e;a|fluency), I(e;a) ex-ante and I(e;a|theta) ex-post are computed
     numerically with a histogram MI estimator (mi_trio).
  9  a social utility that treats lost diversity as a loss (utility).
 10  Dunbar layer q values are exposed as an explicit assumption and swept
     (dunbar_sensitivity).
 11  (documentation) the rate-distortion "coin" is labelled a metaphorical
     application in the docs and figure captions.
 12  multi-seed means with 95% CI (multiseed), parameter sweeps, a pytest suite
     (tests/) and a CI workflow (.github/workflows/ci.yml).
"""
import json
from pathlib import Path

import numpy as np

import model as M

REPO = Path(__file__).resolve().parents[1]


# ================================================================ (2,3) rates
def channel_rates(e_bar, lam=M.LAM, rho=M.RHO, kappa=M.KAPPA,
                  dtau=M.DTAU, q=0.05, eta=M.SIGMA_REFRESH):
    """Per-generation log-rates for every channel, on ONE common timescale.

    sigma : damaged by self-ingestion at -ln(1-kappa(1-e)); repaired ONLY by
            exogenous fresh data at rate eta (toward SIG0).  lambda does not act.
    comp  : damaged by atrophy at rho(1-e); repaired by use rho*e AND by
            generational turnover lam (fresh meat has comp=1).
    tau   : learns at 2*dtau*q (proportional to audit frequency).
    """
    s = 1.0 - kappa * (1.0 - e_bar)
    sigma_damage = float(-np.log(s)) if 0 < s < 1 else (0.0 if s >= 1 else np.inf)
    return dict(
        sigma_damage=sigma_damage,
        sigma_repair=float(eta),                 # exogenous only
        comp_damage=float(rho * (1.0 - e_bar)),
        comp_repair=float(lam + rho * e_bar),
        tau_learn=float(2.0 * dtau * q),
    )


def sigma_convergent(eta, kappa=M.KAPPA, e_bar=0.0):
    """sigma stabilises iff exogenous refresh beats self-ingestion."""
    return eta > kappa * (1.0 - e_bar)


def comp_convergent(lam, rho=M.RHO, e_bar=0.0):
    """comp stabilises iff turnover+use beats atrophy:  lam > rho(1-2e)."""
    return lam > rho * (1.0 - 2.0 * e_bar)


def sigma_equilibrium(eta, kappa=M.KAPPA, e_bar=0.0, sigma0=M.SIG0):
    """Fixed point of sigma <- sigma(1-kappa(1-e)) + eta(SIG0-sigma)."""
    denom = kappa * (1.0 - e_bar) + eta
    if denom <= 0:
        return sigma0
    return float(eta * sigma0 / denom)


def lambda_does_not_repair_sigma(lams=(0.0, 0.067, 0.30), q_ext=0.05, T=400):
    """Empirical demonstration (point 3): terminal sigma is invariant to lambda."""
    out = []
    for lam in lams:
        r = M.run_endogenous(q_ext, 1.0, T=T, lam=lam, seed=7)
        out.append(dict(lam=lam,
                        sigma_terminal=round(float(np.mean([x["sigma"] for x in r[-50:]])), 5),
                        comp_terminal=round(float(np.mean([x["comp"] for x in r[-50:]])), 4)))
    return out


# ================================================================ (5) F variants
def F_atrophied(q, comp):
    """Verifying share among meat at competence `comp` (static approximation)."""
    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    thr = q * M.PHI * M.B / (1.0 + M.GAMMA * (1.0 - comp))
    return np.clip((thr - M.C0_LO) / (M.C0_HI - M.C0_LO), 0.0, 1.0)


def F_dynamic_note():
    return ("model.F(q) is the FRESH-meat (comp=1) static approximation; the "
            "time-resolved verifying share is what M.run/M.run_endogenous produce.")


# ================================================================ (6) 2-D fixed point
def fixed_point_2d(q_ext, beta=1.0, lam=M.LAM, iters=4000):
    """Joint fixed point (e*, comp*) with turnover lambda and endogenous audit.

    e    = clip((thr(comp) - C0_LO)/(C0_HI-C0_LO),0,1),  thr = q*PHI*B/(1+GAMMA(1-comp))
    comp = (1-lam)*(comp + RHO*(e-comp)) + lam*1
    q    = clip(q_ext + beta*e, 0, 1)
    """
    e, comp = 0.0, 1.0
    for _ in range(iters):
        q = float(np.clip(q_ext + beta * e, 0.0, 1.0))
        thr = q * M.PHI * M.B / (1.0 + M.GAMMA * (1.0 - comp))
        e_new = float(np.clip((thr - M.C0_LO) / (M.C0_HI - M.C0_LO), 0.0, 1.0))
        comp_new = (1 - lam) * (comp + M.RHO * (e_new - comp)) + lam * 1.0
        if abs(e_new - e) < 1e-12 and abs(comp_new - comp) < 1e-12:
            e, comp = e_new, comp_new
            break
        e, comp = e_new, comp_new
    return round(e, 5), round(comp, 5)


# ================================================================ (7) MMSE weight
def w_opt_mmse(e, comp=1.0, sigma=M.SIG_MIN):
    """Optimal (MMSE) weight on the relayed answer a when the recipient linearly
    combines it with their own direct query m, using the full covariance.

    a-theta = (1-e)(B+eps) + nu_r + e*nu_v ;  m-theta = B+eps
    var_M = sigma^2 + B^2 ;  var_a = (1-e)^2 var_M + DELTA_R^2 + e^2 (ETA/comp)^2
    cov(a,m) = (1-e) var_M
    w* = argmin_w E[(w(a-theta)+(1-w)(m-theta))^2] = (var_M - cov)/(var_a+var_M-2cov)
    """
    var_M = sigma ** 2 + M.B ** 2
    var_a = ((1 - e) ** 2 * var_M + M.DELTA_R ** 2
             + e ** 2 * (M.ETA / max(comp, 1e-6)) ** 2)
    cov = (1 - e) * var_M
    denom = var_a + var_M - 2 * cov
    w = 0.0 if denom <= 0 else (var_M - cov) / denom
    return dict(w_opt=float(np.clip(w, 0.0, 1.0)), var_a=float(var_a),
                var_M=float(var_M), cov=float(cov))


# ================================================================ (8) mutual info
def _mi(e, x, bins=80):
    """Histogram estimator of I(e; x) for binary e, continuous x."""
    e = np.asarray(e).astype(int)
    x = np.asarray(x, float)
    lo, hi = np.percentile(x, [0.2, 99.8])
    xb = np.clip(np.digitize(x, np.linspace(lo, hi, bins)), 0, bins - 1)
    joint = np.zeros((2, bins))
    for ei in (0, 1):
        m = e == ei
        for b in range(bins):
            joint[ei, b] = np.mean(m & (xb == b))
    pe = joint.sum(axis=1); px = joint.sum(axis=0)
    I = 0.0
    for ei in (0, 1):
        for b in range(bins):
            p = joint[ei, b]
            if p > 0 and pe[ei] > 0 and px[b] > 0:
                I += p * np.log(p / (pe[ei] * px[b]))
    return float(I / np.log(2))


def mi_trio(n=400000, seed=0, e_bar=0.5):
    """I(e;fluency) , I(e;a) ex-ante (theta unknown) , I(e;a|theta) ex-post."""
    rng = np.random.default_rng(seed)
    e = (rng.random(n) < e_bar).astype(float)
    theta = rng.normal(0, 1, n)
    eps = rng.normal(0, M.SIG_MIN, n)
    m = theta + M.B + eps
    a = (1 - e) * m + e * theta + rng.normal(0, M.DELTA_R, n) \
        + e * rng.normal(0, M.ETA, n)
    fluency = rng.normal(0.95, 0.001, n)          # identical for e=0 and e=1
    return dict(
        I_e_fluency=round(_mi(e, fluency), 5),
        I_e_a_exante=round(_mi(e, a), 5),          # theta unknown to recipient
        I_e_a_given_theta=round(_mi(e, a - theta), 5),  # ex-post, truth revealed
        note="ex-ante the recipient's usable statistic is fluency (I~0); the "
             "information about e is in a only once theta is known (ex-post).")


# ================================================================ (9) utility
def utility(e_bar, sigma, comp=1.0, omega=0.5, c_L=0.1):
    """Social utility in which lost diversity IS a loss (option value).

    U = -[ accuracy_loss + omega * diversity_loss + c_L * latency ]
    diversity_loss = -ln(sigma)  (sigma->0 makes it diverge: collapse is costly)
    """
    var_M = sigma ** 2 + M.B ** 2
    sd_M = float(np.sqrt(2 / np.pi) * np.sqrt(var_M))
    sd_V = float(np.sqrt(2 / np.pi) * M.ETA / max(comp, 1e-6))
    acc = (1 - e_bar) * sd_M + e_bar * sd_V
    div = -float(np.log(max(sigma, 1e-9)))
    lat = M.L0 + e_bar * M.DL
    U = -(acc + omega * div + c_L * lat)
    return float(U), dict(acc=round(acc, 4), div=round(div, 4), lat=round(lat, 3))


# ================================================================ (10) Dunbar sensitivity
DUNBAR_Q_BASE = [0.80, 0.50, 0.25, 0.10, 0.05]   # ASSUMED profile (support..beyond)


def dunbar_sensitivity(scales=(0.5, 1.0, 2.0), layers=None):
    """S1 gap(d) under scaled assumptions of the layer-q profile."""
    layers = layers or ["support", "sympathy", "affinity", "Dunbar", "beyond"]
    out = {}
    for sc in scales:
        rows = []
        for name, q in zip(layers, DUNBAR_Q_BASE):
            q = float(np.clip(q * sc, 0, 1))
            e = float(M.F(q))
            var_M = M.SIG_MIN ** 2 + M.B ** 2
            var_P = (1 - e) ** 2 * var_M + M.DELTA_R ** 2 + e ** 2 * M.ETA ** 2
            w_opt = float(np.clip(1 - var_P / var_M, -1, 1))
            rows.append(dict(layer=name, q=round(q, 3), e_bar=round(e, 3),
                             gap=round(0.95 - w_opt, 3)))
        out[f"q_profile_x{sc}"] = rows
    return out


# ================================================================ (12) multi-seed
def multiseed(fn, seeds=range(24)):
    vals = [fn(s) for s in seeds]
    v = np.array(vals, float)
    ci = 1.96 * v.std(ddof=1) / np.sqrt(len(v))
    return dict(mean=round(float(v.mean()), 4), ci95=round(float(ci), 4),
                n=len(v), min=round(float(v.min()), 4), max=round(float(v.max()), 4))


def parameter_sweep():
    """Sweeps over kappa, lambda, q_ext for the key observables (point 12)."""
    out = {"kappa": [], "lambda": [], "q_ext": []}
    for kappa in (0.01, 0.02, 0.05, 0.10):
        r = M.run_endogenous(0.05, 1.0, T=400, kappa=kappa, seed=7)
        out["kappa"].append(dict(kappa=kappa,
                                 sigma=round(float(np.mean([x["sigma"] for x in r[-50:]])), 5)))
    for lam in (0.0, 0.03, 0.067, 0.15, 0.30):
        r = M.run_endogenous(0.05, 1.0, T=400, lam=lam, seed=7)
        out["lambda"].append(dict(lam=lam,
                                  comp=round(float(np.mean([x["comp"] for x in r[-50:]])), 4),
                                  sigma=round(float(np.mean([x["sigma"] for x in r[-50:]])), 5)))
    for q in (0.0, 0.05, 0.2, 0.4, 0.6, 0.9):
        r = M.run_endogenous(q, 1.0, T=400, seed=7)
        out["q_ext"].append(dict(q_ext=q,
                                 e_bar=round(float(np.mean([x["e_bar"] for x in r[-100:]])), 4),
                                 gap=round(float(np.mean([x["gap"] for x in r[-100:]])), 4)))
    return out


# ================================================================ report
def report():
    R = {}
    eb = 0.21
    cr = channel_rates(eb)
    R["P1_P2_timescales"] = dict(
        rates=({k: round(v, 5) for k, v in cr.items()}),
        sigma_half_life_gen=round(M.t_half(M.KAPPA, eb), 2),
        sigma_half_life_VAR_gen_withdrawn=round(M.t_half_var(M.KAPPA, eb), 2),
        sigma_convergent_at_eta0=sigma_convergent(0.0, M.KAPPA, eb),
        comp_convergent_at_defaults=comp_convergent(M.LAM, M.RHO, eb),
        comp_convergence_condition="lam > rho(1-2e) = "
            f"{M.RHO*(1-2*eb):.4f}",
        withdrawn="the earlier R = t_repair/t_damage = 1.74 compared a sigma "
                  "half-life (variance, factor 2) against a mean waiting time "
                  "1/lam, AND attributed sigma-repair to lambda. Both wrong; "
                  "see channel_rates.")
    R["P3_lambda_does_not_repair_sigma"] = lambda_does_not_repair_sigma()
    R["P3_sigma_equilibrium_vs_eta"] = [
        dict(eta=eta, sigma_star=round(sigma_equilibrium(eta, M.KAPPA, eb), 4))
        for eta in (0.0, 0.01, 0.02, 0.04, 0.08)]
    R["P5_F_variants"] = dict(
        F_fresh_q09=round(float(M.F(0.9)), 4),
        F_atrophied_q09_comp0=round(float(F_atrophied(0.9, 0.0)), 4),
        F_atrophied_q09_comp1=round(float(F_atrophied(0.9, 1.0)), 4),
        note=F_dynamic_note())
    R["P6_fixed_point_2d"] = [
        dict(q_ext=q, lam=lam, fp=fixed_point_2d(q, 1.0, lam))
        for q in (0.05, 0.3, 0.6, 0.9) for lam in (0.0, M.LAM, 0.3)]
    R["P7_w_opt_mmse"] = [dict(e=e, **w_opt_mmse(e)) for e in (0.0, 0.21, 0.5, 1.0)]
    R["P8_mi_trio"] = mi_trio()
    R["P9_utility"] = {}
    for name, (eb_, sg) in [("RegimeI_e0_sigma_floor", (0.0, M.SIG_MIN)),
                            ("RegimeI_e0_sigma05", (0.0, 0.5)),
                            ("verified_e05_sigma05", (0.5, 0.5)),
                            ("verified_e05_sigma_floor", (0.5, M.SIG_MIN))]:
        U, parts = utility(eb_, sg)
        R["P9_utility"][name] = dict(U=round(U, 4), **parts)
    R["P10_dunbar_sensitivity"] = dunbar_sensitivity()
    R["P12_multiseed_e_bar_q05"] = multiseed(
        lambda s: float(np.mean([x["e_bar"] for x in M.run_endogenous(0.05, 1.0, T=300, seed=s)[-100:]])))
    R["P12_multiseed_gap_q05"] = multiseed(
        lambda s: float(np.mean([x["gap"] for x in M.run_endogenous(0.05, 1.0, T=300, seed=s)[-100:]])))
    R["P12_parameter_sweep"] = parameter_sweep()
    return R


if __name__ == "__main__":
    R = report()
    (REPO / "results").mkdir(exist_ok=True)
    with open(REPO / "results" / "rigor.json", "w") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)
    print(json.dumps(R, ensure_ascii=False, indent=2))
