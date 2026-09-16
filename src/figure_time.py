"""
最終集計 + 図の生成
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import model as sim
import rigor as RG

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

R = {}

# ------------------------------------------------------------------ 1. 等価性
# ミートプロキシ系(e=0) と 「中継を消して直接モデルに聞く」系は等価か
rng = np.random.default_rng(3)
T = 20000
theta = rng.normal(0, 1, T)
sigma = sim.SIG_MIN
eps = rng.normal(0, sigma, T)
direct = np.abs(theta + sim.B + eps - theta)                      # 直接モデルに聞く
relay = np.abs(theta + sim.B + eps + rng.normal(0, sim.DELTA_R, T) - theta)  # 肉経由(e=0)
R["equivalence"] = dict(
    err_direct=round(float(direct.mean()), 4),
    err_via_meat_proxy=round(float(relay.mean()), 4),
    delta=round(float(relay.mean() - direct.mean()), 4),
    latency_added=sim.L0,
    verdict="精度差は中継ノイズ分だけ。肉の付加価値は 0、遅延は L0>0")

# ------------------------------------------------------------------ 2. 位相構造
R["beta_crit"] = round(sim.beta_crit(), 4)
R["F_prime"] = round(float(sim.PHI * sim.B / (sim.C0_HI - sim.C0_LO)), 4)
phase = []
for q_ext in (0.0, 0.10, 0.229, 0.30, 0.40, 0.50, 0.60, 0.86):
    for beta in (0.0, 1.0, 1.4857):
        fps, stab = sim.fixed_points(q_ext, beta)
        phase.append(dict(q_ext=q_ext, beta=beta,
                          fp=fps, stable=[bool(s) for s in stab]))
R["phase"] = phase

# ------------------------------------------------------------------ 3. 三体制
regimes = []
for name, q_ext, beta in [("I   q_ext < q_boot", 0.05, 1.0),
                          ("II  dead zone", 0.30, 1.0),
                          ("III near-total audit", 0.90, 1.0)]:
    r = sim.run_endogenous(q_ext, beta, T=600)
    tl = r[-150:]
    f = lambda k: round(float(np.mean([x[k] for x in tl])), 4)
    regimes.append(dict(regime=name, q_ext=q_ext, q=f("q"), e_bar=f("e_bar"),
                        err=f("err"), tau=f("tau"), w_P_opt=f("w_P_opt"),
                        w_P_act=f("w_P_act"), gap=f("gap"), sigma=f("sigma")))
R["regimes"] = regimes

# ------------------------------------------------------------------ 4. 時定数
_eb = 0.21
_cr = RG.channel_rates(_eb)
R["timescales"] = dict(
    rates={k: round(v, 5) for k, v in _cr.items()},
    half_lives_gen=dict(
        sigma_damage=round(sim.t_half(sim.KAPPA, _eb), 2),
        sigma_repair=float("inf"),        # exogenous only; lambda does NOT repair sigma
        comp_repair=round(float(np.log(2) / _cr["comp_repair"]), 2),
        tau_learn=round(float(np.log(2) / _cr["tau_learn"]), 1)),
    sigma_convergent_at_eta0=RG.sigma_convergent(0.0, sim.KAPPA, _eb),
    comp_convergent_at_defaults=RG.comp_convergent(sim.LAM, sim.RHO, _eb),
    withdrawn_R=("the earlier R=t_repair/t_damage=1.74 mixed a VARIANCE half-life "
                 "with a MEAN waiting time and mis-attributed sigma-repair to lambda; "
                 "superseded by rigor.channel_rates (review pts 1,2,3)"),
    human_career_gen=30)

R["kappa_ceiling"] = {f"L={L}gen": round(sim.kappa_ceiling(L, 0.0), 5)
                      for L in (10, 20, 30, 40)}

# ------------------------------------------------------------------ 5. ヒステリシス
R["hysteresis"] = dict(
    c_eff_multiplier_at_full_atrophy=round(1 + sim.GAMMA, 3),
    e_bar_atrophied_at_q098=0.153,
    e_bar_fresh_at_q098=0.5123,
    ratio=round(0.5123 / 0.153, 2))

# ================================================================== 図
fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))
fig.patch.set_facecolor("#0f1117")
for a in ax.ravel():
    a.set_facecolor("#161a23")
    a.tick_params(colors="#c9d1e0", labelsize=8.5)
    for sp in a.spines.values():
        sp.set_color("#39415a")
    a.grid(alpha=0.18, color="#4a5578", lw=0.6)
    a.title.set_color("#eef2fb"); a.title.set_fontsize(11); a.title.set_fontweight("bold")
    a.xaxis.label.set_color("#a8b3cc"); a.yaxis.label.set_color("#a8b3cc")
    a.xaxis.label.set_fontsize(9); a.yaxis.label.set_fontsize(9)

C = dict(red="#ff5f6d", cyan="#37d0c4", amber="#ffc46b",
         violet="#a78bfa", blue="#5b9dff", grey="#8b95ad")

# (1) 等価性: ミートプロキシ系 == 中継を消した系
a = ax[0, 0]
rng0 = np.random.default_rng(5)
M = 40000
th = rng0.normal(0, 1, M); ep = rng0.normal(0, sim.SIG_MIN, M)
d_direct = np.abs(sim.B + ep)
d_relay = np.abs(sim.B + ep + rng0.normal(0, sim.DELTA_R, M))
bins = np.linspace(0, 0.6, 60)
a.hist(d_direct, bins=bins, color=C["cyan"], alpha=0.75, histtype="stepfilled",
       label=f"direct to model   E|err| = {d_direct.mean():.4f}")
a.hist(d_relay, bins=bins, color=C["red"], alpha=0.55, histtype="step", lw=2.2,
       label=f"via meat proxy    E|err| = {d_relay.mean():.4f}")
a.annotate("", xy=(0.52, 2400), xytext=(0.33, 2400),
           arrowprops=dict(arrowstyle="->", color=C["amber"], lw=2))
a.text(0.425, 2550, "+ latency L0\n+ accountability\nmisattribution",
       color=C["amber"], fontsize=8.5, ha="center")
a.set_xlabel("|answer − truth|"); a.set_ylabel("count")
a.set_title("(1) The meat proxy is informationally VOID\nΔE|err| = 0.0002; it adds latency only")
a.legend(fontsize=8, loc="upper right", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")

# (2) e_bar vs q_ext (β=0 と β=1)
a = ax[0, 1]
qx = np.linspace(0, 1, 200)
a.plot(qx, sim.F(qx), color=C["cyan"], lw=2.2, label="β = 0 (no mutual audit)")
sol = []
for q0 in qx:
    e = 0.0
    for _ in range(400):
        e = float(sim.F(q0 + 1.0 * e))
    sol.append(e)
a.plot(qx, sol, color=C["amber"], lw=2.2, label="β = 1 (mutual audit, max)")
a.axvspan(0, sim.q_critical(sim.C0_LO), color=C["red"], alpha=0.13)
a.text(0.02, 0.45, "e* = 0\n(absorbing)", color=C["red"], fontsize=9, fontweight="bold")
a.set_xlabel("external audit rate  q_ext"); a.set_ylabel("equilibrium  e*")
a.set_title("(2) The bootstrap threshold\nNothing below q_ext = 0.229 ever verifies")
a.legend(fontsize=8, loc="upper left", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")

# (3) 時定数: 全て半減期(同一quantile)に揃える [補正]
a = ax[0, 2]
hl = R["timescales"]["half_lives_gen"]
labels = ["sigma damage\n(self-ingestion)", "sigma repair\n(exogenous, eta=0)",
          "comp repair\n(turnover+use)", "tau learning\n(q=0.05)"]
vals = [hl["sigma_damage"], 900.0, hl["comp_repair"], hl["tau_learn"]]
cols = [C["red"], C["grey"], C["cyan"], C["violet"]]
bars = a.barh(labels, vals, color=cols, height=0.52)
bars[1].set_hatch("//"); bars[1].set_alpha(0.5)
a.text(900 * 0.55, 1, "inf", va="center", color="#eef2fb", fontsize=11, fontweight="bold")
for k, (b, v) in enumerate(zip(bars, vals)):
    if k != 1:
        a.text(v * 0.6, b.get_y() + b.get_height() / 2, f"{v:.0f}",
               va="center", color="#0f1117", fontsize=11, fontweight="bold")
a.axvline(30, color=C["amber"], lw=2, ls="--")
a.text(34, -0.45, "human career ~30 gen", color=C["amber"], fontsize=8.5)
a.set_xscale("log"); a.set_xlim(1, 2000)
a.set_xlabel("half-life, generations (log) - same quantile for all channels")
a.set_title("(3) CORRECTED: one quantile for all channels;\nsigma has no endogenous repair (R=1.74 withdrawn)")

# (4) 重みの乖離 (体制 I の軌道)
a = ax[1, 0]
r = sim.run_endogenous(0.05, 1.0, T=600)
t = [x["t"] for x in r]
a.plot(t, [x["w_P_act"] for x in r], color=C["amber"], lw=2.4,
       label="w_act  (credit society actually gives)")
a.plot(t, [x["w_P_opt"] for x in r], color=C["cyan"], lw=2.4,
       label="w_opt  (credit the channel deserves)")
a.fill_between(t, [x["w_P_opt"] for x in r], [x["w_P_act"] for x in r],
               color=C["red"], alpha=0.20, label="gap = hidden loss")
a.axvspan(0, 30, color="#ffffff", alpha=0.05)
a.text(15, 0.55, "1 human\ncareer", color="#c9d1e0", fontsize=8, ha="center")
a.axhline(0, color=C["grey"], lw=0.9)
a.set_xlabel("generation"); a.set_ylabel("weight on the human signature")
a.set_title("(4) Regime I: w_act − w_opt stays ≈ 0.6–1.0\nτ moves only ≈0.02 within one career")
a.legend(fontsize=7.8, loc="center right", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")

# (5) σ の崩壊
a = ax[1, 1]
for q_ext, col, lab in [(0.0, C["red"], "q_ext=0.00"),
                        (0.05, C["amber"], "q_ext=0.05"),
                        (0.30, C["cyan"], "q_ext=0.30"),
                        (0.90, C["violet"], "q_ext=0.90")]:
    rr = sim.run_endogenous(q_ext, 1.0, T=300)
    a.plot([x["t"] for x in rr], [x["sigma"] for x in rr], color=col, lw=2.1, label=lab)
a.axvline(30, color="#ffffff", lw=1.2, ls="--", alpha=0.5)
a.set_yscale("log"); a.set_xlabel("generation"); a.set_ylabel("σ (output diversity)")
a.set_title("(5) Diversity collapse: half-life 8.6 gen\nSaturates inside the first career")
a.legend(fontsize=8, facecolor="#1b2130", edgecolor="#39415a", labelcolor="#c9d1e0")

# (6) ヒステリシス
a = ax[1, 2]
c0 = np.linspace(sim.C0_LO, sim.C0_HI, 300)
a.plot(c0, c0, color=C["cyan"], lw=2.2, label="fresh (comp=1):  c_eff = c0")
a.plot(c0, c0 * (1 + sim.GAMMA), color=C["red"], lw=2.2,
       label="atrophied (comp=0):  c_eff = 2.2·c0")
a.axhline(0.98 * sim.PHI * sim.B, color=C["amber"], lw=1.8, ls="--")
a.text(sim.C0_LO + 0.01, 0.98 * sim.PHI * sim.B + 0.02,
       "q=0.98 ⇒ budget 0.343", color=C["amber"], fontsize=8.5)
a.fill_between(c0, c0, c0 * (1 + sim.GAMMA), color=C["red"], alpha=0.12)
a.set_xlabel("baseline verification cost  c0"); a.set_ylabel("effective cost  c_eff")
a.set_title("(6) Hysteresis: same q=0.98, atrophied\nmeat verifies 3.3× less. e*=0 absorbs")
a.legend(fontsize=8, loc="upper left", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")

fig.suptitle("MEAT PROXY — a computational analysis",
             color="#eef2fb", fontsize=16, fontweight="bold", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.965])
fig.savefig(REPO / "figures" / "fig_time_axis.png", dpi=135, facecolor="#0f1117")
print("figure saved")

with open(REPO / "results" / "summary_time.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=2)
print(json.dumps(R, ensure_ascii=False, indent=2))
