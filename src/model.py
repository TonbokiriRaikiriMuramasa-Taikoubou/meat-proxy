"""
ミートプロキシの演算モデル  v4  (個体群版・最終)
====================================================
主体: M(モデル) / P_i(プロキシ=肉, i=1..N) / S(社会=受信者)

■ 1 世代の流れ
  1. 真の最適解 theta_t が立つ
  2. モデルが答える   m_i = theta_t + B + eps_i ,  eps_i ~ N(0, sigma_t^2)
  3. 肉 i が検証するかどうか決める
        実行コスト  c_i^eff = c0_i * (1 + GAMMA*(1 - comp_i))
        検証する ⇔  q * PHI * B > c_i^eff           …… bang-bang
     ※ q = 監査(=外部検出)確率, PHI = 罰, B = 検証が取り除ける誤差
     ※ 費用 c_i^eff は「私的・確実・即時」、便益 q*PHI*B は「確率的・遅延」
  4. 社会に提示される   a_i = (1-e_i)*m_i + e_i*theta_t + nu_r + e_i*nu_v(comp_i)
        nu_r ~ N(0, DELTA_R^2)  中継ノイズ(貼り間違い・切り詰め)。e と無関係に常に乗る
        nu_v ~ N(0,(ETA/comp)^2) 検証ノイズ。検証した分だけ乗る
        e=0 → a = m + nu_r     付加情報ゼロ、遅延 L0 とノイズだけ
        e=1 → a = theta + ...  完全検証、遅延 L0+DL
  5. 監査は |a_i - theta_t| を観測するが、その出所を帰属できない(P4)
     → 罰は常に「署名した人間」に落ちる → 社会の信頼 tau が更新される
  6. 能力の萎縮   comp_i <- comp_i + RHO*(e_i - comp_i)
  7. 世代交代     確率 LAM で肉が新品(c0 再抽選, comp=1)に置き換わる
  8. 多様性の崩壊 sigma <- max(sigma*(1 - KAPPA*(1-e_bar)), SIG_MIN)
     ※ 未検証で通過した出力だけが学習データに還る

■ 命題
  P1  q*PHI*B <= c0 の肉にとって e=0 は最適反応。かつ comp の萎縮で c^eff が
      上昇するので、後から q を上げても戻れない(ヒステリシス / 吸収状態)。
  P2  集団の平均検証率 e_bar は「新品の供給」だけで維持される:
          e_bar ≈ LAM / (LAM + 萎縮による脱落率)
      → 肉の均衡は寿命が系に貸しているだけで、解ではない。
  P3  多様性の半減期  t_half = ln0.5 / (2 ln(1 - KAPPA(1-e_bar)))
      職業寿命 L 世代で破綻しない条件  KAPPA(1-e_bar) < 1 - 0.5^(1/(2L))
  P4  監査は誤差を見るが出所を見ない。よって罰の帰属先は常に人間。
  P5  I(e ; a | 流暢さ) ≈ 0。e を漏らす唯一の観測量は遅延 L だけで、
      社会はそれを最適化して消す → 系は自分の唯一の診断器を破壊する。
  P6  重み: 社会が人間の署名に置く重み w_act = tau*fluency は、
      置くべき重み w_opt = 1 - var_P/var_M と一致しない。
      その差 gap = w_act - w_opt が、この系の「隠された全損失」。
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

# ------------------------------------------------------------------ 参数
N       = 3000    # 肉の数
B       = 0.35    # モデルの系統的バイアス = 検証が取り除ける誤差
ETA     = 0.20    # 検証ノイズ(comp=1)
DELTA_R = 0.05    # 中継ノイズ
C0_LO, C0_HI = 0.08, 0.60   # 検証費用の個体差
GAMMA   = 1.20    # 萎縮が検証費用を押し上げる係数
RHO     = 0.06    # 能力の萎縮/回復率
PHI     = 1.00    # 監査ペナルティ
KAPPA   = 0.05    # 未検証出力の自己摂取率
LAM     = 0.067   # 世代交代率(≈15年)
SIG_MIN = 0.005
SIGMA_REFRESH = 0.0   # exogenous fresh-data refresh rate eta. lambda does NOT repair sigma.
SIG0    = 1.0
L0, DL  = 1.0, 4.0
FLUENCY = 0.95    # 流暢さ(検証と無関係に高い)
AUDIT_TOL = 0.30  # 監査が「誤り」と判定する閾値
DTAU    = 0.02    # 監査1件あたりの信頼更新


# ------------------------------------------------------------------ 解析式
def q_critical(c0=B * 0 + 0.30, phi=PHI, b=B):
    """検証が私的に採算する最低監査確率  q_c = c0 / (PHI*B)"""
    return c0 / (phi * b)


def t_half(kappa=KAPPA, e_bar=0.0):
    """Half-life of the std sigma -- the quantity actually plotted as 'diversity'.

    sigma <- sigma*(1-kappa(1-e)) once per generation, so the half-life of sigma is
    ln0.5/ln(1-kappa(1-e)).  (An earlier revision wrongly used the VARIANCE
    half-life ln0.5/(2 ln(...)) here while plotting sigma; see docs/ANALYSIS.md
    addendum point 1.)"""
    s = 1.0 - kappa * (1.0 - e_bar)
    return float("inf") if s >= 1 else float(np.log(0.5) / np.log(s))


def t_half_var(kappa=KAPPA, e_bar=0.0):
    """Half-life of the variance sigma^2 (= t_half/2). Kept for transparency only."""
    return t_half(kappa, e_bar) / 2.0


def kappa_ceiling(L_gen, e_bar=0.0):
    """職業寿命 L 世代では多様性が半減しないための KAPPA 上限"""
    return 1.0 - 0.5 ** (1.0 / (2 * L_gen * (1 - e_bar)))


# ------------------------------------------------------------------ 本体
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

        # 3. 検証の意思決定
        c_eff = c0 * (1 + GAMMA * (1 - comp))
        want = (q * PHI * B) > c_eff
        # P5: 遅延での選別 — 検証しようとした個体ほど脱落する
        if latency_selection > 0:
            survive = rng.random(N) > latency_selection
            e = (want & survive).astype(float)
        else:
            e = want.astype(float)

        # 4. 提示される答え
        a = ((1 - e) * m + e * theta
             + rng.normal(0, DELTA_R, N)
             + e * rng.normal(0, ETA, N) / np.maximum(comp, 1e-6))
        err = np.abs(a - theta)
        e_bar = float(e.mean())

        # 5. 監査(P4: 出所を帰属できない。罰は人間に落ちる)
        #    ★ tau の学習速度は監査頻度 q に比例する。q が薄ければ学習は事実上止まる。
        aud = rng.random(N) < q
        n_aud = int(aud.sum())
        if n_aud > 0:
            bad = float((err[aud] > AUDIT_TOL).mean())
            tau = float(np.clip(tau + DTAU * q * (0.5 - bad) * 2, 0, 1))

        # 6. 萎縮 / 回復
        comp = comp + RHO * (e - comp)
        comp = np.clip(comp, 0.0, 1.0)

        # 7. 世代交代
        repl = rng.random(N) < lam
        if repl.any():
            comp[repl] = 1.0
            if hetero:
                c0[repl] = rng.uniform(C0_LO, C0_HI, int(repl.sum()))

        # 8. 多様性の崩壊 (損傷) と外生リフレッシュ (修復)。交代 lam は sigma を修復しない。
        sigma = sigma * (1 - kappa * (1 - e_bar))
        sigma = sigma + SIGMA_REFRESH * (SIG0 - sigma)
        sigma = max(sigma, SIG_MIN)

        # 6'. 重み
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


# ------------------------------------------------------------------ 主実行
if __name__ == "__main__":
    out = {}

    out["P1_q_critical_table"] = [
        dict(c0=c0, q_c=round(q_critical(c0), 3),
             verdict=("検証する" if q_critical(c0) < 0.95 else "検証しない"))
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

    # P1 ヒステリシス: 交代なしで300世代放置した「萎縮した肉」に q=0.98 の監査を入れる
    ph1 = run(q=0.0, T=300, lam=0.0, seed=11)
    rng1 = np.random.default_rng(11)
    # ph1 と同じ個体群を再現するため、萎縮後の comp と c0 を直接持ち越す
    # (簡便のため、同一パラメータで萎縮末状態を解析的に作る)
    comp_atrophied = np.full(N, 0.0)
    c0_same = rng1.uniform(C0_LO, C0_HI, N)
    ph2 = run(q=0.98, T=300, lam=0.0, seed=11, comp0=comp_atrophied, c0_init=c0_same)
    ph3 = run(q=0.98, T=300, lam=0.0, seed=11)                      # 新品の肉
    c_eff_atroph = c0_same * (1 + GAMMA * (1 - comp_atrophied))
    out["P1_hysteresis"] = dict(
        comp_after_300gen_no_audit=round(float(np.mean(
            [x["comp"] for x in ph1[-20:]])), 4),
        c_eff_multiplier_when_atrophied=round(1 + GAMMA, 3),
        share_still_verifying=round(float(((0.98 * PHI * B) > c_eff_atroph).mean()), 4),
        e_bar_atrophied_at_q098=round(float(np.mean([x["e_bar"] for x in ph2[-100:]])), 4),
        e_bar_fresh_at_q098=round(float(np.mean([x["e_bar"] for x in ph3[-100:]])), 4),
        verdict="同じ q でも、萎縮した肉は検証できない。e*=0 は吸収状態")

    # ---- P5/P6: 帰属不能の損失
    # 社会が「検証済みか未検証か」を判別できないので、重みをメッセージ単位で
    # 最適化できない。判別できた場合とできなかった場合の実効誤差を比べる。
    def effective_error(e_bar, sigma=B * 0 + 0.005):
        var_M = sigma ** 2 + B ** 2
        sd_M = float(np.sqrt(2 / np.pi) * np.sqrt(var_M))   # E|B+eps| 近似
        sd_V = float(np.sqrt(2 / np.pi) * ETA)              # 検証済み残余誤差
        no_attr = (1 - e_bar) * sd_M + e_bar * sd_V
        with_attr = e_bar * sd_V + (1 - e_bar) * min(sd_M, sd_V + C0_LO)  # 未検証は再検査に回す
        return round(no_attr, 4), round(with_attr, 4), round(no_attr - with_attr, 4)

    out["P5_attribution_loss"] = {
        f"e_bar={e}": dict(zip(("no_attr", "with_attr", "loss"), effective_error(e)))
        for e in (0.0, 0.21, 0.50)}
    out["P5_note"] = ("判別チャネルは遅延 L だけ。L は社会が最適化して消す対象なので、"
                      "系は自分の唯一の診断器を自ら破壊する。")

    # ---- 三つの時定数の比較(これが「寿命」命题の答え)
    e_ref = 0.21
    t_damage = t_half(KAPPA, e_ref)          # 多様性が半減するまでの世代数
    t_repair = 1.0 / LAM                     # 肉が入れ替わって修復される時定数
    out["timescales"] = dict(
        t_damage_half_gen=round(t_damage, 2),
        t_repair_gen=round(t_repair, 2),
        R_repair_over_damage=round(t_repair / t_damage, 3),
        lambda_star_for_R1=round(1.0 / t_damage, 4),
        tau_learning_timeconstant={str(q): round(1.0 / (2 * DTAU * q), 1)
                                   for q in (0.02, 0.05, 0.10, 0.55)},
        verdict=("R>1: 損傷のほうが修復より速い。寿命は破綻を防がない、遅らせるだけ。"
                 if t_repair / t_damage > 1 else "R<=1: 修復が損傷に追いつく"))

    print(json.dumps(out, ensure_ascii=False, indent=2))
    with open(REPO / "results" / "sim_raw.json", "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


# ==================================================================
#  P7: 監査 q は外生ではない。q =「誰かが読む」= e の別名。よって内生。
#      q_t = q_ext + BETA * e_bar_t
#        q_ext : 外部監査(本番障害・撤回・訴訟・査読)。小さく、遅れて来る
#        BETA  : 相互監査。同僚が互いに読む度合い。物理上限は 1
#      不動点  e* = F(q_ext + BETA*e*)
#        F(q) = P(c0 < q*PHI*B) = clip((q*PHI*B - C0_LO)/(C0_HI - C0_LO), 0, 1)
#      F'(q) = PHI*B/(C0_HI-C0_LO)  →  BETA_crit = (C0_HI-C0_LO)/(PHI*B)
#      BETA_crit > 1 なら、相互監査だけでは高検証均衡に届かない。
# ==================================================================
def F(q):
    """Verifying share among FRESH meat (comp=1) -- a static approximation.

    q is a probability and is clipped to [0,1] here so the fixed-point analysis
    (where q_ext + beta*e can algebraically exceed 1) stays in the simplex.
    For atrophied meat use rigor.F_atrophied / rigor.fixed_point_2d (point 5,6)."""
    q = np.clip(np.asarray(q, float), 0.0, 1.0)
    return np.clip((q * PHI * B - C0_LO) / (C0_HI - C0_LO), 0.0, 1.0)


def beta_crit():
    return (C0_HI - C0_LO) / (PHI * B)


def fixed_points(q_ext, beta, n=4000):
    e = np.linspace(0, 1, n)
    g = F(q_ext + beta * e) - e
    sign = np.sign(g)
    idx = np.where(np.diff(sign) != 0)[0]
    fps = [round(float(e[i]), 4) for i in idx]
    # 安定性: g が + から - に変われば安定
    stab = [bool(sign[i] > 0) for i in idx]
    return fps, stab


def run_endogenous(q_ext, beta, T=400, **kw):
    """q を内生にして回す"""
    rng = np.random.default_rng(kw.pop("seed", 7))
    c0 = rng.uniform(C0_LO, C0_HI, N)
    comp = np.ones(N)
    sigma, tau = SIG0, 1.0
    rows = []
    for t in range(T):
        theta = float(rng.normal(0, 1))
        m = theta + B + rng.normal(0, sigma, N)
        # 前世代の e_bar から今の q を決める(1期ラグ)
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
        sigma = sigma * (1 - kw.get("kappa", KAPPA) * (1 - e_bar))
        sigma = sigma + kw.get("eta", SIGMA_REFRESH) * (SIG0 - sigma)
        sigma = max(sigma, SIG_MIN)
        var_M = sigma ** 2 + B ** 2
        var_P = float(np.mean((1 - e) ** 2 * var_M + DELTA_R ** 2
                              + e ** 2 * (ETA / np.maximum(comp, 1e-6)) ** 2))
        w_opt = float(np.clip(1 - var_P / var_M, -1, 1))
        w_act = float(np.clip(tau * FLUENCY, 0, 1))
        rows.append(dict(t=t, q=q_t, e_bar=e_bar, comp=float(comp.mean()),
                         sigma=sigma, err=float(err.mean()), tau=tau,
                         w_P_opt=w_opt, w_P_act=w_act, gap=w_act - w_opt))
    return rows
