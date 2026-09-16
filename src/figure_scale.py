"""
The scale axis, v1
==================
Where fig_time_axis.png was the TIME axis, this is the SCALE axis.

Three questions.
  S1  Up to what social distance does a signature / trace retain force?
      -> Dunbar layers
  S2  When a many-dimensional trace is shaved down to the "coin of choice"
      (1 bit), what survives?   [METAPHORICAL application - see docs/ANALYSIS.md]
  S3  Does individual-scale collision really converge at generational scale?

S1 Dunbar horizon
   Audit q is a function of social distance: repeated games and cheap
   reputation make q high on the inside.  (layer q values are ASSUMPTIONS;
   see rigor.dunbar_sensitivity for sensitivity.)
     layer 5   (support)   q~0.80
     layer 15  (sympathy)  q~0.50
     layer 50  (affinity)  q~0.25
     layer 150 (Dunbar)    q~0.10
     beyond    (strangers) q~q_ext=0.05
   Report e_bar(d)=F(q(d)) and gap(d)=w_act-w_opt per layer.
   -> The force of a trace/signature stands only inside the Dunbar horizon;
      outside it the meat-proxy regime (Regime I) holds by structure, not choice.

S2 The coin of choice (rate-distortion)  [metaphorical]
   The trace T is k-dimensional (form, scale, medium, intent, duration,
   audience...). Coding each dimension at distortion D/sigma^2 = 0.125 costs
   R_dim = 0.5*log2(1/0.125) = 1.5 bit, so H(T) = 1.5k bit. The coin is 1 bit.
   Fraction of the trace the coin can carry = 1/H(T) = 1/(1.5k).
   -> At k=6 that is ~11%; 89% is discarded by the shave.
   Yet exactly one thing travels losslessly: the binary fact that a signer
   chose = e itself.  The coin = e made public.

S3 Condition for generational convergence  [corrected]
   competence channel converges iff lambda > rho(1-2e);
   diversity channel converges iff exogenous eta > kappa(1-e) (lambda never).
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
import model as sim

plt.rcParams["font.family"] = "DejaVu Sans"
C = dict(red="#ff5f6d", cyan="#37d0c4", amber="#ffc46b",
         violet="#a78bfa", blue="#5b9dff", grey="#8b95ad")

OUT = {}

# ---------------------------------------------------------------- S1 Dunbar
LAYERS = [("5\nsupport", 5, 0.80), ("15\nsympathy", 15, 0.50),
          ("50\naffinity", 50, 0.25), ("150\nDunbar", 150, 0.10),
          ("beyond\nstrangers", 1500, 0.05)]
s1 = []
for name, n, q in LAYERS:
    e = float(sim.F(q))
    var_M = sim.SIG_MIN ** 2 + sim.B ** 2
    var_P = (1 - e) ** 2 * var_M + sim.DELTA_R ** 2 + e ** 2 * sim.ETA ** 2
    w_opt = float(np.clip(1 - var_P / var_M, -1, 1))
    w_act = 0.95
    s1.append(dict(layer=name, n=n, q=q, e_bar=round(e, 3),
                   w_opt=round(w_opt, 3), w_act=w_act,
                   gap=round(w_act - w_opt, 3)))
OUT["S1_dunbar"] = s1

# ---------------------------------------------------------------- S2 the coin
R_DIM = 0.5 * np.log2(1 / 0.125)          # 1.5 bit / dim
ks = np.arange(1, 13)
retained = 1.0 / (ks * R_DIM)
OUT["S2_coin"] = dict(R_dim_bit=round(float(R_DIM), 3),
                      retained_at_k6=round(float(1 / (6 * R_DIM)), 4),
                      curve={int(k): round(float(1 / (k * R_DIM)), 4) for k in ks},
                      lossless_part="the binary fact that a signer chose = e")

# ---------------------------------------------------------------- S3 generational convergence
e_ref = 0.21
lam_c = sim.RHO * (1 - 2 * e_ref)
OUT["S3_generational"] = dict(
    comp_channel=dict(convergence_condition=f"lambda > rho(1-2e) = {lam_c:.4f}",
                      convergent_at_defaults=bool(sim.LAM > lam_c)),
    sigma_channel=dict(damage_rate=round(float(-np.log(1 - sim.KAPPA * (1 - e_ref))), 5),
                       repair_rate_exogenous=sim.SIGMA_REFRESH,
                       convergent_at_eta0=False,
                       note="lambda does NOT repair sigma; only exogenous fresh data (eta) does"),
    withdrawn=("the earlier R=(1/lambda)/t_half and lambda*=1/t_half mixed timescales "
               "and mis-attributed sigma-repair to lambda (review pts 1,2,3)"))

# ================================================================= figures
fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.4))
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

# (1) Dunbar horizon
a = ax[0]
xs = np.arange(len(s1))
e_bars = a.bar(xs - 0.19, [d["e_bar"] for d in s1], width=0.36,
               color=C["cyan"], label="verifying share  e")
g_bars = a.bar(xs + 0.19, [d["gap"] for d in s1], width=0.36,
               color=C["red"], alpha=0.85, label="hidden loss  gap")
a.axvline(3.5, color=C["amber"], lw=2, ls="--")
a.text(3.45, 0.97, "Dunbar horizon\n(≈150)", color=C["amber"],
       fontsize=8.5, ha="right", va="top")
a.set_xticks(xs); a.set_xticklabels([d["layer"] for d in s1], fontsize=8)
a.set_ylim(0, 1.05)
a.set_ylabel("share / weight")
a.set_title("(1) The signature only has force inside\nthe Dunbar horizon; beyond it, meat-proxy by structure")
a.legend(fontsize=8, loc="upper left", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")
for x, d in zip(xs, s1):
    a.text(x - 0.19, d["e_bar"] + 0.02, f'{d["e_bar"]:.2f}', ha="center",
           color="#eef2fb", fontsize=7.5)

# (2) the coin of choice
a = ax[1]
a.plot(ks, retained, color=C["violet"], lw=2.6, marker="o", ms=4,
       label="fraction of trace the 1-bit coin can carry")
a.fill_between(ks, retained, 1.0, color=C["red"], alpha=0.14)
a.text(4.2, 0.55, "discarded by the collapse\nto the optimal token",
       color=C["red"], fontsize=9)
a.axvline(6, color=C["amber"], lw=1.6, ls="--")
a.plot([6], [1 / (6 * R_DIM)], "o", color=C["amber"], ms=9, mec="#0f1117", mew=1.5)
a.annotate(f"k=6 → {1/(6*R_DIM):.0%} carried\n89% shaved off",
           xy=(6, 1 / (6 * R_DIM)), xytext=(7.0, 0.34),
           color=C["amber"], fontsize=9, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C["amber"], lw=1.6))
a.text(1.15, 0.06, "carried losslessly: the binary fact\nthat a signer chose  (= e)",
       color=C["cyan"], fontsize=8.5)
a.set_xlabel("trace dimensionality  k  (form·scale·medium·intent·duration·audience…)")
a.set_ylabel("retained fraction")
a.set_ylim(0, 1.0); a.set_xlim(1, 12)
a.set_title("(2) The coin of choice: 1 bit vs H(T)=1.5k bits\nWhat survives is not content but the fact-of-choice")
a.legend(fontsize=8, loc="upper right", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")

# (3) generational convergence [corrected]: comp-channel condition / sigma exogenous
a = ax[2]
lam_grid = np.linspace(0.0, 0.40, 200)
e_ref = 0.21
a.plot(lam_grid, lam_grid + sim.RHO * e_ref, color=C["cyan"], lw=2.6,
       label="comp repair = lambda + rho*e")
a.plot(lam_grid, np.full_like(lam_grid, sim.RHO * (1 - e_ref)), color=C["red"],
       lw=2.6, label="comp damage = rho*(1-e)")
lam_c = sim.RHO * (1 - 2 * e_ref)
a.axvline(lam_c, color=C["amber"], lw=2, ls="--")
a.text(lam_c + 0.006, 0.30, f"lambda_c = rho(1-2e) = {lam_c:.3f}",
       color=C["amber"], fontsize=9, fontweight="bold")
a.fill_between(lam_grid, 0, 0.42, where=(lam_grid > lam_c), color=C["cyan"], alpha=0.12)
a.plot([sim.LAM], [sim.LAM + sim.RHO * e_ref], "o", color=C["cyan"], ms=10,
       mec="#0f1117", mew=1.6)
a.annotate("now lambda=0.067:\ncomp channel converges",
           xy=(sim.LAM, sim.LAM + sim.RHO * e_ref), xytext=(0.17, 0.15),
           color=C["cyan"], fontsize=8.5, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C["cyan"], lw=1.5))
a.text(0.015, 0.40, "sigma channel: repair = eta (exogenous) = 0 < damage 0.040\n"
                    "-> diversity needs fresh external data, NOT lifespan",
       color=C["red"], fontsize=8.5, va="top")
a.set_xlabel("generational turnover rate  lambda"); a.set_ylabel("per-generation rate")
a.set_ylim(0, 0.42); a.set_xlim(0, 0.40)
a.set_title("(3) CORRECTED convergence: competence converges for\n"
            "lambda>rho(1-2e); diversity only via exogenous eta")
a.legend(fontsize=8, loc="lower right", facecolor="#1b2130",
         edgecolor="#39415a", labelcolor="#c9d1e0")

fig.suptitle("MEAT PROXY — the scale axis: signature, the coin of choice, generational convergence",
             color="#eef2fb", fontsize=14.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(REPO / "figures" / "fig_scale_axis.png", dpi=135, facecolor="#0f1117")

with open(REPO / "results" / "summary_scale.json", "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print(json.dumps(OUT, ensure_ascii=False, indent=2))
print("scale.png saved")
