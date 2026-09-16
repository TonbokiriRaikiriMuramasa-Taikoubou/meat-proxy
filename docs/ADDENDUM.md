# Addendum: responses to the scientific-completeness review (12 points)

> A domain review flagged that several first-revision headline numbers mixed
> timescales and mis-attributed repair. This addendum records the response to all
> 12 points and every withdrawal / correction.
> Implementation: `src/rigor.py`; figure: `figures/fig_rigor.png`; tests:
> `tests/test_rigor.py`; CI: `.github/workflows/ci.yml`; numbers: `results/rigor.json`.

---

## Withdrawals (most important)

**The first revision's `R = t_repair / t_damage = 1.74 > 1` ("damage outruns repair")
and `λ* = 0.116` are WITHDRAWN.** Two reasons, both as the review stated:

1. **Mixed scales (points 1,2).** The numerator `t_damage` was the *variance*
   half-life `ln0.5/(2ln(1−κ(1−e)))=8.6` (the factor 2 comes from squaring), while the
   denominator `t_repair=1/λ` was a *mean waiting time*. A half-life and a mean
   waiting time are different quantiles and cannot be compared. Moreover the figure
   plotted σ (the std) as "diversity", so the correct σ half-life is
   `ln0.5/ln(1−κ(1−e))=17.2`.
2. **Mis-attributed repair (point 3).** Turnover `λ` repairs **comp (competence)**
   only, **not σ (diversity)**. σ is repaired only by exogenous data `η`.
   Demonstrated: `rigor.lambda_does_not_repair_sigma` — σ-terminal = 0.005 for
   λ=0 / 0.067 / 0.3 (invariant), while comp-terminal = 0.00 / 0.55 / 0.88.

**Corrected statement (per-channel rates, one common scale, point 2):**

| Channel | damage /gen | repair /gen | convergence condition | at defaults |
|---|---|---|---|---|
| σ (diversity) | 0.040 | **0** (exogenous η only) | `η > κ(1−e)` | **non-convergent** (η=0) |
| comp (competence) | 0.047 | 0.080 (=λ+ρe) | `λ > ρ(1−2e)=0.035` | convergent (λ=0.067) |
| τ (trust) | — | 0.002 (=2·DTAU·q) | ∝ q | extremely slow |

Hence "lifespan hides the collapse" is corrected to **"lifespan guards only the
atrophy of competence (the absorbing state); it does not guard the collapse of
diversity."** The only route that prevents diversity collapse is exogenous fresh
data η.

---

## Point-by-point

| # | Point | Response |
|---|---|---|
| 1 | half-life factor 2 | `model.t_half` now the std-σ half-life (17.2); the variance version kept explicitly as `t_half_var`. |
| 2 | mixing mean-wait and half-life | all channels reported as per-generation log-rates (`rigor.channel_rates`); figures use one quantile (half-lives). |
| 3 | model how λ repairs σ | σ repair modelled as exogenous `η` (`SIGMA_REFRESH`); λ↛σ demonstrated numerically (table, fig_rigor(2)). |
| 4 | clip q in fixed-point analysis | `model.F` clips `q∈[0,1]`; test asserts fixed points stay in the unit square under absurd `q_ext,β`. |
| 5 | F(q) is fresh-meat static approx | stated in docstring; atrophied variant `rigor.F_atrophied`, dynamic via `run/run_endogenous`. e.g. F(0.9): fresh 0.452 / atrophied 0.122. |
| 6 | include comp and λ in fixed points | `rigor.fixed_point_2d` (joint (e,comp) fixed point with turnover). e.g. q=0.6: λ=0 → e*=0.051 (comp collapse), λ=0.067 → 0.256, λ=0.3 → 0.471. The 1-D F overstates. |
| 7 | derive w_opt from MSE + covariance | `rigor.w_opt_mmse`: the MMSE weight when the recipient linearly combines the relayed answer a with their own direct query m, full covariance included. At e=0, **w_opt = 0 exactly** (a is strictly worse than m). |
| 8 | compute I(e;a｜fluency) | `rigor.mi_trio` (histogram MI estimator). **I(e;fluency)=0.0001, I(e;a) ex-ante=0.022, I(e;a｜θ) ex-post=0.734 bits.** Ex-ante the information about e is ~0; it exists only ex-post. P5 is now measured, not asserted. |
| 9 | utility where lost diversity is a loss | `rigor.utility` (`U = −[acc + ω·(−ln σ) + c_L·L]`). σ→floor regimes score U=−3.03, dominating any accuracy gain by an order of magnitude. |
| 10 | state Dunbar q as assumption + sensitivity | `DUNBAR_Q_BASE` exposed as an assumption; `rigor.dunbar_sensitivity` (×0.5/×1/×2). **Beyond-horizon gap=0.97 is invariant across all scalings (robust); inside values are assumption-dependent** (support: 0.76/0.40/0.29). |
| 11 | label rate-distortion as metaphorical | stated in prose and figure captions: the trace is not literally a Gaussian source and the 1-bit coin is conceptual; S2 is illustrative. |
| 12 | multi-seed, CI, tests, sweeps | `rigor.multiseed` (24 seeds, 95% CI: gap=0.831±0.0004), `parameter_sweep` (κ, λ, q_ext), `tests/test_rigor.py` (12 invariant tests), `.github/workflows/ci.yml` (py3.11/3.12, pytest + figure regeneration). |

---

## Conclusions that do NOT change

The withdrawals concern timescales and attribution, not the core:

- The meat's informational value-add is 0 (`ΔE|err| = 0.0002`).
- `e*=0` is an absorbing state (hysteresis 3.35×).
- `β_crit = 1.49 > 1` → no incremental internal reform.
- Misallocated credence `gap ≈ 0.65–0.97` frozen on a lifespan scale (τ learning 0.002/gen is the slowest channel).
- Beyond the Dunbar horizon, meat-proxy by structure (gap=0.97, robust to sensitivity).
- The answer is presented; the weight is hidden (I(e;fluency)≈0 now measured).

## Conclusions that DO change

- "Damage outruns repair (R=1.74)" → **withdrawn**. Correctly: σ has zero endogenous repair (non-convergent); comp converges.
- "Lifespan hides the collapse" → **"lifespan guards competence-atrophy only, not diversity-collapse."**
- "Generational convergence iff λ>0.116" → **comp channel converges for λ>0.035 (satisfied now); σ channel never converges via λ and needs η.**
