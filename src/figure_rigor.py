"""
Corrected-analyses figure (review points 1-10).  -> figures/fig_rigor.png
Supersedes the timescale panel of fig_time_axis.png and the convergence panel
of fig_scale_axis.png, both of which mixed timescales / mis-attributed repair.
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import model as M
import rigor as RG

plt.rcParams["font.family"] = "DejaVu Sans"
C = dict(red="#ff5f6d", cyan="#37d0c4", amber="#ffc46b",
         violet="#a78bfa", blue="#5b9dff", grey="#8b95ad")

R = json.loads((REPO / "results" / "rigor.json").read_text())

fig, ax = plt.subplots(2, 3, figsize=(16.5, 9.2))
fig.patch.set_facecolor("#0f1117")
for a in ax.ravel():
    a.set_facecolor("#161a23")
    a.tick_params(colors="#c9d1e0", labelsize=8.5)
    for sp in a.spines.values():
        sp.set_color("#39415a")
    a.grid(alpha=0.18, color="#4a5578", lw=0.6)
    a.title.set_color("#eef2fb"); a.title.set_fontsize(10.5); a.title.set_fontweight("bold")
    a.xaxis.label.set_color("#a8b3cc"); a.yaxis.label.set_color("#a8b3cc")
    a.xaxis.label.set_fontsize(9); a.yaxis.label.set_fontsize(9)

# (1) per-channel rates on ONE timescale (points 1,2)
a = ax[0, 0]
cr = R["P1_P2_timescales"]["rates"]
ch = ["sigma\n(diversity)", "comp\n(competence)", "tau\n(trust)"]
dam = [cr["sigma_damage"], cr["comp_damage"], 0.0]
rep = [cr["sigma_repair"], cr["comp_repair"], cr["tau_learn"]]
x = np.arange(3)
a.bar(x - 0.19, dam, 0.36, color=C["red"], label="damage / drift rate")
a.bar(x + 0.19, rep, 0.36, color=C["cyan"], label="repair / learning rate")
a.set_xticks(x); a.set_xticklabels(ch, fontsize=8.5)
a.set_ylabel("per-generation log-rate")
a.text(0 - 0.19, 0.042, "0.040", ha="center", color="#eef2fb", fontsize=8)
a.text(0 + 0.19, 0.002, "0 (exogenous\nonly)", ha="center", color=C["cyan"], fontsize=7.5)
a.set_title("(1) One timescale for all channels (pts 1,2)\nsigma has NO endogenous repair; withdrawn R=1.74")
a.legend(fontsize=8, loc="upper right", facecolor="#1b2130", edgecolor="#39415a", labelcolor="#c9d1e0")

# (2) lambda does not repair sigma; sigma* needs exogenous eta (point 3)
a = ax[0, 1]
ls = R["P3_lambda_does_not_repair_sigma"]
a.plot([d["lam"] for d in ls], [d["sigma_terminal"] for d in ls], "o-",
       color=C["red"], lw=2.4, ms=9, label="terminal sigma  (flat: lambda does nothing)")
a2 = a.twinx()
a2.plot([d["lam"] for d in ls], [d["comp_terminal"] for d in ls], "s--",
        color=C["cyan"], lw=2.0, ms=8, label="terminal comp  (lambda repairs THIS)")
a2.set_ylabel("terminal comp", color=C["cyan"]); a2.tick_params(colors="#c9d1e0")
a.set_xlabel("turnover rate  lambda"); a.set_ylabel("terminal sigma", color=C["red"])
a.tick_params(colors="#c9d1e0")
a.set_ylim(0, 0.02); a2.set_ylim(0, 1)
a.set_title("(2) lambda repairs competence, NOT diversity (pt 3)\nsigma* = eta*SIG0/(kappa(1-e)+eta); needs eta>0.0395")
a.legend(fontsize=7.6, loc="center left", facecolor="#1b2130", edgecolor="#39415a", labelcolor="#c9d1e0")

# (3) 2-D fixed point (e*, comp*) incl. comp and lambda (point 6)
a = ax[0, 2]
for lam, col, mk in [(0.0, C["red"], "o"), (M.LAM, C["amber"], "s"), (0.3, C["cyan"], "^")]:
    qs = np.linspace(0, 1, 40)
    es, cs = zip(*[RG.fixed_point_2d(q, 1.0, lam) for q in qs])
    a.plot(qs, es, color=col, lw=2.2, marker=mk, ms=4, label=f"e*  (lambda={lam})")
a.plot(qs, [RG.fixed_point_2d(q, 1.0, M.LAM)[1] for q in qs], color=C["grey"],
       lw=1.4, ls=":", label="comp* (lambda=0.067)")
a.axhline(float(M.F(0.6)), color=C["violet"], lw=1.2, ls="--")
a.text(0.02, float(M.F(0.6)) + 0.02, "1-D fresh-meat F(0.6) overstates e*",
       color=C["violet"], fontsize=8)
a.set_xlabel("q_ext"); a.set_ylabel("fixed point")
a.set_title("(3) Fixed point with comp and lambda (pt 6)\nlambda=0 collapses comp and drags e* down")
a.legend(fontsize=7.6, loc="upper left", facecolor="#1b2130", edgecolor="#39415a", labelcolor="#c9d1e0")

# (4) MMSE weight vs ad-hoc (point 7)
a = ax[1, 0]
ee = np.linspace(0, 1, 60)
mmse = [RG.w_opt_mmse(e)["w_opt"] for e in ee]
adhoc = [float(np.clip(1 - ((1 - e) ** 2 * (M.SIG_MIN ** 2 + M.B ** 2) + M.DELTA_R ** 2
                            + e ** 2 * M.ETA ** 2) / (M.SIG_MIN ** 2 + M.B ** 2), -1, 1))
         for e in ee]
a.plot(ee, mmse, color=C["cyan"], lw=2.6, label="w_opt from MMSE + covariance (pt 7)")
a.plot(ee, adhoc, color=C["grey"], lw=1.6, ls="--", label="earlier ad-hoc 1-var_P/var_M")
a.axhline(0.95, color=C["amber"], lw=1.8, ls=":")
a.text(0.55, 0.97, "w_act = 0.95 (credence extended)", color=C["amber"], fontsize=8.5)
a.set_xlabel("verification effort  e"); a.set_ylabel("weight on relayed answer")
a.set_title("(4) w_opt derived from the MSE decision problem (pt 7)\nat e=0 the MMSE weight is exactly 0")
a.legend(fontsize=8, loc="center right", facecolor="#1b2130", edgecolor="#39415a", labelcolor="#c9d1e0")

# (5) mutual informations (point 8)
a = ax[1, 1]
mi = R["P8_mi_trio"]
labels = ["I(e; fluency)\nex-ante usable", "I(e; a)\nex-ante, theta unknown", "I(e; a|theta)\nex-post, truth known"]
vals = [mi["I_e_fluency"], mi["I_e_a_exante"], mi["I_e_a_given_theta"]]
bars = a.bar(labels, vals, color=[C["grey"], C["amber"], C["cyan"]], width=0.55)
for b, v in zip(bars, vals):
    a.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center",
           color="#eef2fb", fontsize=10, fontweight="bold")
a.set_ylabel("bits")
a.set_title("(5) I(e;a|fluency) computed, not asserted (pt 8)\nex-ante ~0 bits; the info is ex-post only")
a.tick_params(axis="x", labelsize=8)

# (6) utility with diversity priced as a loss (point 9)
a = ax[1, 2]
names = list(R["P9_utility"].keys())
Us = [R["P9_utility"][n]["U"] for n in names]
short = ["e=0\nsigma=floor", "e=0\nsigma=0.5", "e=0.5\nsigma=0.5", "e=0.5\nsigma=floor"]
cols = [C["red"] if "floor" in n else C["cyan"] for n in names]
bars = a.bar(short, Us, color=cols, width=0.55)
for b, v in zip(bars, Us):
    a.text(b.get_x() + b.get_width() / 2, v - 0.18, f"{v:.2f}", ha="center",
           color="#eef2fb", fontsize=9, fontweight="bold")
a.axhline(0, color=C["grey"], lw=0.9)
a.set_ylabel("social utility  U")
a.set_title("(6) Lost diversity IS a loss (pt 9)\ncollapse (sigma->floor) dominates every accuracy gain")
a.tick_params(axis="x", labelsize=8)

fig.suptitle("MEAT PROXY — corrected analyses (scientific-completeness layer)",
             color="#eef2fb", fontsize=15, fontweight="bold", y=0.985)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(REPO / "figures" / "fig_rigor.png", dpi=135, facecolor="#0f1117")
print("fig_rigor.png saved")
