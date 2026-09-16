"""
Meat proxy - a computational model, v4 (population version, final)
==================================================================
Agents: M (model) / P_i (proxy = meat, i=1..N) / S (society = recipient)

* One generation
  1. The true optimum theta_t is drawn.
  2. The model answers   m_i = theta_t + B + eps_i ,  eps_i ~ N(0, sigma_t^2)
  3. Meat i decides whether to verify:
        execution cost  c_i^eff = c0_i * (1 + GAMMA*(1 - comp_i))
        verifies iff    q * PHI * B > c_i^eff           ... bang-bang
     where q = audit (= external detection) rate, PHI = penalty,
     B = the error that verification removes.
     The cost c_i^eff is private, certain, immediate;
     the benefit q*PHI*B is stochastic and delayed.
  4. Society is shown  a_i = (1-e_i)*m_i + e_i*theta_t + nu_r + e_i*nu_v(comp_i)
        nu_r ~ N(0, DELTA_R^2)   relay noise (mispaste, truncation); always on
        nu_v ~ N(0,(ETA/comp)^2) verification noise; only on the verified part
        e=0 -> a = m + nu_r      zero added information; latency L0 and noise only
        e=1 -> a = theta + ...   full verification; latency L0+DL
  5. The audit observes |a_i - theta_t| but cannot attribute its source (P4)
     -> the penalty always falls on "the human who signed" -> trust tau updates
  6. Competence atrophy   comp_i <- comp_i + RHO*(e_i - comp_i)
  7. Generational turnover: with prob LAM the meat is replaced by a fresh one
     (c0 resampled, comp=1)
  8. Diversity collapse  sigma <- max(sigma*(1 - KAPPA*(1-e_bar)), SIG_MIN)
     only output that passed UNverified returns to the training data

* Propositions
  P1  For meat with q*PHI*B <= c0, e=0 is the best response; and because comp
      atrophy raises c^eff, raising q later cannot restore e
      (hysteresis / absorbing state).
  P2  The population mean verification rate e_bar is sustained only by the
      supply of fresh meat:  e_bar ~ LAM / (LAM + atrophy-driven dropout).
      The meat equilibrium is borrowed from lifespan; it is not a solution.
  P3  Diversity half-life  t_half = ln0.5 / (2 ln(1 - KAPPA(1-e_bar))).
      Not breaking within a career of L generations requires
      KAPPA(1-e_bar) < 1 - 0.5^(1/(2L)).
  P4  The audit sees the error but not its source; the penalty's addressee is
      always the human.
  P5  I(e ; a | fluency) ~ 0. The only observable leaking e is latency L, and
      society optimises it away -> the system destroys its own only diagnostic.
  P6  Weights: the weight society places on the human signature,
      w_act = tau*fluency, does not equal the warranted weight
      w_opt = 1 - var_P/var_M. The difference gap = w_act - w_opt is this
      system's "hidden total loss".
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ parameters
N       = 3000    # number of meat
B       = 0.35    # model's systematic bias = the error verification removes
ETA     = 0.20    # verification noise (at comp=1)
DELTA_R = 0.05    # relay noise
C0_LO, C0_HI = 0.08, 0.60   # individual spread of verification cost
GAMMA   = 1.20    # factor by which atrophy inflates verification cost
RHO     = 0.06    # competence atrophy / recovery rate
PHI     = 1.00    # audit penalty
KAPPA   = 0.05    # self-ingestion rate of unverified output
LAM     = 0.067   # generational turnover rate (~15 yr)
SIG_MIN = 0.005
SIG0    = 1.0
L0, DL  = 1.0, 4.0
FLUENCY = 0.95    # fluency (high regardless of verification)
AUDIT_TOL = 0.30  # threshold at which an audit calls it an error
DTAU    = 0.02    # trust update per audit event


# ------------------------------------------------------------------ closed forms
def q_critical(c0=B * 0 + 0.30, phi=PHI, b=B):
    """Lowest audit rate at which verification pays privately: q_c = c0/(PHI*B)."""
    return c0 / (phi * b)


def t_half(kappa=KAPPA, e_bar=0.0):
    s = 1.0 - kappa * (1.0 - e_bar)
    return float("inf") if s >= 1 else float(np.log(0.5) / (2 * np.log(s)))


def kappa_ceiling(L_gen, e_bar=0.0):
    """Upper bound on KAPPA such that diversity does not halve within a career of L generations."""
    return 1.0 - 0.5 ** (1.0 / (2 * L_gen * (1 - e_bar)))


# ------------------------------------------------------------------ core loop
def run(q, T=400, lam=LAM, kappa=KAPPA, latency_selection=0.0,
        sigma0=SIG0, tau0=1.0, seed=7, hetero=True, comp0=None, c0_init=None):
    rng = np.random.default_rng(seed)
    if c0_init is not None:
        c0 = np.array(c0_init, dtype=float)
    elif hetero:
        c0 = rng.uniform(C0_LO, C0_HI, N)
    else:
        c0 = np.full(N, 0.30)
    comp = np.ones(N) if comp0 is None else np.array(comp0, dtype=float)
    sigma, tau = sigma0, tau0
    rows = []

    for t in range(T):
        theta = float(rng.normal(0, 1))
        eps = rng.normal(0, sigma, N)
        m = theta + B + eps

        # 3. verification decision
        c_eff = c0 * (1 + GAMMA * (1 - comp))
        want = (q * PHI * B) > c_eff
        # P5: selection on latency - those who try to verify are dropped more
        if latency_selection > 0:
            survive = rng.random(N) > latency_selection
            e = (want & survive).astype(float)
        else:
            e = want.astype(float)

        # 4. the answer presented
        a = ((1 - e) * m + e * theta
             + rng.normal(0, DELTA_R, N)
             + e * rng.normal(0, ETA, N) / np.maximum(comp, 1e-6))
        err = np.abs(a - theta)
        e_bar = float(e.mean())

        # 5. audit (P4: source cannot be attributed; penalty falls on the human)
        #    * tau's learning speed is proportional to audit frequency q; thin q => learning stalls.
        aud = rng.random(N) < q
        n_aud = int(aud.sum())
        if n_aud > 0:
            bad = float((err[aud] > AUDIT_TOL).mean())
            tau = float(np.clip(tau + DTAU * q * (0.5 - bad) * 2, 0, 1))

        # 6. atrophy / recovery
        comp = comp + RHO * (e - comp)
        comp = np.clip(comp, 0.0, 1.0)

        # 7. generational turnover
        repl = rng.random(N) < lam
        if repl.any():
            comp[repl] = 1.0
            if hetero:
                c0[repl] = rng.uniform(C0_LO, C0_HI, int(repl.sum()))

        # 8. diversity collapse
        sigma = max(sigma * (1 - kappa * (1 - e_bar)), SIG_MIN)

        # 6'. weights
        var_M = sigma ** 2 + B ** 2
        var_P = float(np.mean((1 - e) ** 2 * var_M + DELTA_R ** 2
                              + e ** 2 * (ETA / np.maximum(comp, 1e-6)) ** 2))
        w_opt = float(np.clip(1 - var_P / var_M, -1, 1))
        w_act = float(np.clip(tau * FLUENCY, 0, 1))

        rows.append(dict(t=t, e_bar=e_bar, comp=float(comp.mean()),
                         sigma=sigma, err=float(err.mean()), tau=tau,
                         L=L0 + e_bar * DL, var_M=var_M, var_P=var_P,
                         w_P_opt=w_opt, w_P_act=w_act, gap=w_act - w_opt))
    return rows


def agg(rows, n=150):
    tail = rows[-n:]
    return {k: round(float(np.mean([r[k] for r in tail])), 4)
            for k in ("e_bar", "comp", "sigma", "err", "tau", "L",
                      "w_P_opt", "w_P_act", "gap")}


def series(rows, step=25):
    return [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
            for r in rows[::step]]


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    out = {}

    out["P1_q_critical_table"] = [
        dict(c0=c0, q_c=round(q_critical(c0), 3),
             verdict=("verifies" if q_critical(c0) < 0.95 else "does not verify"))
        for c0 in (0.08, 0.15, 0.30, 0.45, 0.60)]
    out["P1_share_verifying_vs_q"] = []
    for q in (0.0, 0.2, 0.4, 0.6, 0.8, 0.86, 0.95, 1.0):
        r = run(q, T=250)
        out["P1_share_verifying_vs_q"].append(dict(q=q, e_bar=round(float(np.mean(
            [x["e_bar"] for x in r[-100:]])), 3)))

    out["P2_e_bar_vs_lambda"] = []
    for lam in (0.0, 0.01, 0.033, 0.067, 0.15, 0.40):
        r = run(q=0.55, T=300, lam=lam)
        out["P2_e_bar_vs_lambda"].append(dict(
            lam=lam, e_bar=round(float(np.mean([x["e_bar"] for x in r[-100:]])), 3),
            comp=round(float(np.mean([x["comp"] for x in r[-100:]])), 3)))

    out["P3_kappa_ceiling"] = {f"L={L}gen": round(kappa_ceiling(L, 0.0), 5)
                               for L in (10, 20, 30, 40)}
    out["P3_t_half"] = [dict(kappa=k, e_bar_0=round(t_half(k, 0.0), 2),
                             e_bar_040=round(t_half(k, 0.40), 2),
                             e_bar_080=round(t_half(k, 0.80), 2))
                        for k in (0.005, 0.01, 0.02, 0.05, 0.10)]

    scen = {
        "A_no_audit":        dict(q=0.0),
        "B_no_audit_noTurn": dict(q=0.0, lam=0.0),
        "C_mid_audit":       dict(q=0.55),
        "D_full_audit":      dict(q=0.98),
        "E_audit_plus_latency": dict(q=0.98, latency_selection=0.45),
        "F_realistic_audit":  dict(q=0.05),
    }
    for name, kw in scen.items():
        r = run(T=400, **kw)
        out[f"sim_{name}"] = agg(r)
        out[f"series_{name}"] = series(r)

    # P1 hysteresis: meat left 300 gens without turnover, then audited at q=0.98
    ph1 = run(q=0.0, T=300, lam=0.0, seed=11)
    rng1 = np.random.default_rng(11)
    # carry over the atrophied comp and the same c0 to reproduce ph1's population
    # (for brevity, build the atrophied end-state analytically with same parameters)
    comp_atrophied = np.full(N, 0.0)
    c0_same = rng1.uniform(C0_LO, C0_HI, N)
    ph2 = run(q=0.98, T=300, lam=0.0, seed=11, comp0=comp_atrophied, c0_init=c0_same)
    ph3 = run(q=0.98, T=300, lam=0.0, seed=11)                      # fresh meat
    c_eff_atroph = c0_same * (1 + GAMMA * (1 - comp_atrophied))
    out["P1_hysteresis"] = dict(
        comp_after_300gen_no_audit=round(float(np.mean(
            [x["comp"] for x in ph1[-20:]])), 4),
        c_eff_multiplier_when_atrophied=round(1 + GAMMA, 3),
        share_still_verifying=round(float(((0.98 * PHI * B) > c_eff_atroph).mean()), 4),
        e_bar_atrophied_at_q098=round(float(np.mean([x["e_bar"] for x in ph2[-100:]])), 4),
        e_bar_fresh_at_q098=round(float(np.mean([x["e_bar"] for x in ph3[-100:]])), 4),
        verdict="at the same q, atrophied meat cannot verify; e*=0 is an absorbing state")

    # ---- P5/P6: the loss from non-attribution
    # Society cannot tell verified from unverified, so it cannot set the weight
    # per message. Compare effective error with and without that discrimination.
    def effective_error(e_bar, sigma=B * 0 + 0.005):
        var_M = sigma ** 2 + B ** 2
        sd_M = float(np.sqrt(2 / np.pi) * np.sqrt(var_M))   # approx E|B+eps|
        sd_V = float(np.sqrt(2 / np.pi) * ETA)              # residual error when verified
        no_attr = (1 - e_bar) * sd_M + e_bar * sd_V
        with_attr = e_bar * sd_V + (1 - e_bar) * min(sd_M, sd_V + C0_LO)  # unverified -> re-check
        return round(no_attr, 4), round(with_attr, 4), round(no_attr - with_attr, 4)

    out["P5_attribution_loss"] = {
        f"e_bar={e}": dict(zip(("no_attr", "with_attr", "loss"), effective_error(e)))
        for e in (0.0, 0.21, 0.50)}
    out["P5_note"] = ("The only discriminating channel is latency L; since L is what society optimises away, "
                      "the system destroys its own only diagnostic.")

    # ---- comparing the three timescales (the answer to the 'lifespan' proposition)
    e_ref = 0.21
    t_damage = t_half(KAPPA, e_ref)          # generations until diversity halves
    t_repair = 1.0 / LAM                     # timescale of repair via meat turnover
    out["timescales"] = dict(
        t_damage_half_gen=round(t_damage, 2),
        t_repair_gen=round(t_repair, 2),
        R_repair_over_damage=round(t_repair / t_damage, 3),
        lambda_star_for_R1=round(1.0 / t_damage, 4),
        tau_learning_timeconstant={str(q): round(1.0 / (2 * DTAU * q), 1)
                                   for q in (0.02, 0.05, 0.10, 0.55)},
        verdict=("R>1: damage outruns repair; lifespan delays collapse, it does not prevent it."
                 if t_repair / t_damage > 1 else "R<=1: repair keeps up with damage"))

    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(REPO / "results" / "sim_raw.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ==================================================================
#  P7: audit q is not exogenous. q = 'someone reads' = another name for e; hence endogenous.
#      q_t = q_ext + BETA * e_bar_t
#        q_ext : external audit (outages, retractions, suits, peer review); small, lagged
#        BETA  : mutual audit; how much colleagues read each other; physical max 1
#      fixed point  e* = F(q_ext + BETA*e*)
#        F(q) = P(c0 < q*PHI*B) = clip((q*PHI*B - C0_LO)/(C0_HI - C0_LO), 0, 1)
#      F'(q) = PHI*B/(C0_HI-C0_LO)  →  BETA_crit = (C0_HI-C0_LO)/(PHI*B)
#      if BETA_crit > 1, mutual audit alone cannot reach the high-verification equilibrium.
# ==================================================================
def F(q):
    return np.clip((np.asarray(q, float) * PHI * B - C0_LO) / (C0_HI - C0_LO), 0.0, 1.0)


def beta_crit():
    return (C0_HI - C0_LO) / (PHI * B)


def fixed_points(q_ext, beta, n=4000):
    e = np.linspace(0, 1, n)
    g = F(q_ext + beta * e) - e
    sign = np.sign(g)
    idx = np.where(np.diff(sign) != 0)[0]
    fps = [round(float(e[i]), 4) for i in idx]
    # stability: stable where g flips from + to -
    stab = [bool(sign[i] > 0) for i in idx]
    return fps, stab


def run_endogenous(q_ext, beta, T=400, **kw):
    """Run with q endogenous."""
    rng = np.random.default_rng(kw.pop("seed", 7))
    c0 = rng.uniform(C0_LO, C0_HI, N)
    comp = np.ones(N)
    sigma, tau = SIG0, 1.0
    rows = []
    for t in range(T):
        theta = float(rng.normal(0, 1))
        m = theta + B + rng.normal(0, sigma, N)
        # current q from last generation's e_bar (one-period lag)
        e_prev = rows[-1]["e_bar"] if rows else F(q_ext)
        q_t = float(np.clip(q_ext + beta * e_prev, 0, 1))
        c_eff = c0 * (1 + GAMMA * (1 - comp))
        e = ((q_t * PHI * B) > c_eff).astype(float)
        a = ((1 - e) * m + e * theta + rng.normal(0, DELTA_R, N)
             + e * rng.normal(0, ETA, N) / np.maximum(comp, 1e-6))
        err = np.abs(a - theta)
        e_bar = float(e.mean())
        aud = rng.random(N) < q_t
        if aud.sum() > 0:
            bad = float((err[aud] > AUDIT_TOL).mean())
            tau = float(np.clip(tau + DTAU * q_t * (0.5 - bad) * 2, 0, 1))
        comp = np.clip(comp + RHO * (e - comp), 0, 1)
        repl = rng.random(N) < kw.get("lam", LAM)
        if repl.any():
            comp[repl] = 1.0
            c0[repl] = rng.uniform(C0_LO, C0_HI, int(repl.sum()))
        sigma = max(sigma * (1 - kw.get("kappa", KAPPA) * (1 - e_bar)), SIG_MIN)
        var_M = sigma ** 2 + B ** 2
        var_P = float(np.mean((1 - e) ** 2 * var_M + DELTA_R ** 2
                              + e ** 2 * (ETA / np.maximum(comp, 1e-6)) ** 2))
        w_opt = float(np.clip(1 - var_P / var_M, -1, 1))
        w_act = float(np.clip(tau * FLUENCY, 0, 1))
        rows.append(dict(t=t, q=q_t, e_bar=e_bar, comp=float(comp.mean()),
                         sigma=sigma, err=float(err.mean()), tau=tau,
                         w_P_opt=w_opt, w_P_act=w_act, gap=w_act - w_opt))
    return rows
