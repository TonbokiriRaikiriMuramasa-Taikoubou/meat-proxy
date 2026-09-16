"""
スケール軸の演算  v1
======================
第一の図(meatproxy.png)が「時間軸」だったのに対し、こちらは「スケール軸」。

問いは三つ。
  S1  署名/痕跡はどの社会的距離まで効力を持つか  → ダンバー層
  S2  多次の痕跡が「選択のコイン」(1 bit) に削ぎ落とされるとき、何が残るか
  S3  個人スケールの衝突は、世代スケールで本当に収束するか

S1 ダンバー地平
   監査 q は社会的距離の関数。反复ゲームと評判が安い内側ほど q は高い。
     layer 5   (support)   q≈0.80
     layer 15  (sympathy)  q≈0.50
     layer 50  (affinity)  q≈0.25
     layer 150 (Dunbar)    q≈0.10
     beyond    (strangers) q≈q_ext=0.05
   e_bar(d)=F(q(d))、gap(d)=w_act-w_opt を層ごとに出す。
   → 痕跡/署名の効力はダンバー地平の内側だけで立ち、外側では構造的に
     ミートプロキシ体制(体制I)になる。選択ではなく構造。

S2 選択のコイン (rate-distortion)
   痕跡 T は k 次元(形式・スケール・媒体・意図・持続・受手…)。
   各次元を歪率 D/σ²=0.125 で符号化するには R_dim = 0.5*log2(1/0.125) = 1.5 bit。
   よって H(T) = 1.5k bit。コインは 1 bit。
   コインが運べる痕跡の割合 = 1 / H(T) = 1/(1.5k)。
   → k=6 で ≈11%。89% は削ぎ落としで棄損する。
   しかしコインが*無損失で*運ぶものが一つだけある:
   「署名者が選択をした」という二値の事実 = e そのもの。
   コイン = e を公開したもの。

S3 世代収束の条件
   R = t_repair / t_damage = (1/λ) / t_half。
   R<1 ⇔ λ > λ* = 1/t_half のときだけ、世代スケールは収束する。
   現在 λ=0.067 → R=1.74 (発散側)。λ=0.15 → R=0.78 (収束側)。
   → 「世代的には収束を得る」は能力(comp)チャネルでは真、
     多様性(σ)チャネルでは λ>λ* の条件下でのみ真。無条件ではない。
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

# ---------------------------------------------------------------- S1 ダンバー
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

# ---------------------------------------------------------------- S2 コイン
R_DIM = 0.5 * np.log2(1 / 0.125)          # 1.5 bit / dim
ks = np.arange(1, 13)
retained = 1.0 / (ks * R_DIM)
OUT["S2_coin"] = dict(R_dim_bit=round(float(R_DIM), 3),
                      retained_at_k6=round(float(1 / (6 * R_DIM)), 4),
                      curve={int(k): round(float(1 / (k * R_DIM)), 4) for k in ks},
                      lossless_part="the binary fact that a signer chose = e")

# ---------------------------------------------------------------- S3 世代収束
lams = np.linspace(0.01, 0.40, 200)
t_half = sim.t_half(sim.KAPPA, 0.21)
R_curve = (1 / lams) / t_half
lam_star = 1 / t_half
OUT["S3_generational"] = dict(
    t_half=round(t_half, 2), lam_star=round(lam_star, 4),
    R_at_current_lambda=round((1 / sim.LAM) / t_half, 3),
    R_at_lambda_015=round((1 / 0.15) / t_half, 3),
    verdict="comp チャネルは常に交代で回復するが、σ チャネルは λ>λ* でのみ収束")

# ================================================================= 図
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

# (1) ダンバー地平
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

# (2) 選択のコイン
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

# (3) 世代収束
a = ax[2]
a.plot(lams, R_curve, color=C["blue"], lw=2.6)
a.axhline(1, color=C["grey"], lw=1.4, ls=":")
a.axvline(lam_star, color=C["amber"], lw=2, ls="--")
a.fill_between(lams, 0, 1, where=(R_curve < 1), color=C["cyan"], alpha=0.16)
a.fill_between(lams, 1, 4, where=(R_curve > 1), color=C["red"], alpha=0.14)
a.text(lam_star + 0.008, 3.4, f"λ* = {lam_star:.3f}", color=C["amber"],
       fontsize=9.5, fontweight="bold")
a.text(0.30, 0.55, "convergent\ngenerationally", color=C["cyan"],
       fontsize=9, ha="center", fontweight="bold")
a.text(0.055, 2.6, "divergent even\ngenerationally", color=C["red"],
       fontsize=9, ha="center", fontweight="bold")
a.plot([sim.LAM], [(1 / sim.LAM) / t_half], "o", color=C["red"], ms=10,
       mec="#0f1117", mew=1.6)
a.annotate("now\nλ=0.067, R=1.74", xy=(sim.LAM, (1 / sim.LAM) / t_half),
           xytext=(0.10, 2.0), color=C["red"], fontsize=8.5, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C["red"], lw=1.5))
a.plot([0.15], [(1 / 0.15) / t_half], "o", color=C["cyan"], ms=10,
       mec="#0f1117", mew=1.6)
a.annotate("λ=0.15 → R=0.78", xy=(0.15, (1 / 0.15) / t_half),
           xytext=(0.185, 0.9), color=C["cyan"], fontsize=8.5, fontweight="bold",
           arrowprops=dict(arrowstyle="->", color=C["cyan"], lw=1.5))
a.set_xlabel("generational turnover rate  λ"); a.set_ylabel("R = t_repair / t_damage")
a.set_ylim(0, 4); a.set_xlim(0.01, 0.40)
a.set_title("(3) 'Generationally it converges' is conditional:\ntrue only for λ > λ*; the competence channel, always")

fig.suptitle("MEAT PROXY — the scale axis: signature, the coin of choice, generational convergence",
             color="#eef2fb", fontsize=14.5, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(REPO / "figures" / "fig_scale_axis.png", dpi=135, facecolor="#0f1117")

with open(REPO / "results" / "summary_scale.json", "w") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=2)
print(json.dumps(OUT, ensure_ascii=False, indent=2))
print("scale.png saved")
