# Meat Proxy — a computational analysis (EN)

> This is not a thought experiment; it is a dynamical system. Definition → model →
> numbers → interpretation, carried through to "what should the weights be" and
> "how does the answer emerge, and is it presented or hidden".
> Reproduce: `make figures` (= `python3 src/model.py && python3 src/figure_time.py`)
> → `figures/fig_time_axis.png` / `results/summary_time.json`

---

## 0.0 Translator's note — a glossary of term-values

Translating a coinage is an act of pricing: each word must be given a value in the
target language, and the price chosen changes what survives. The values used here:

| JA | EN value chosen | why |
|---|---|---|
| ミートプロキシ | **meat proxy** | the term under study; kept verbatim |
| 肉 | **meat** (or, in neutral prose, *the proxy / the relay*) | the slang is deliberately blunt; softening it would delete its argument |
| 検証努力 `e` | **verification effort**, `e` | the proxy's only value-add |
| 検証能力 `comp` | **competence**, `comp` | the capacity to verify; atrophies with disuse |
| 監査 `q` | **audit rate**, `q`; `q_ext` external audit; `β` mutual audit | "audit" carries the sense of *someone actually reading* |
| 吸収状態 | **absorbing state** | standard Markov terminology |
| 萎縮 | **atrophy** | disuse-decay of competence |
| 世代交代 `λ` | **generational turnover**, `λ` | the repair channel |
| 多様性 `σ` / 自己摂取 | **output diversity**, `σ` / **self-ingestion** | the model-collapse channel |
| 重み `w` | **weight**; `w_act` = *credence extended*, `w_opt` = *credence warranted*; `gap` = *misallocated credence* ("the hidden loss") | "credence" is the epistemic sense of weight |
| 痕跡 | **trace** | the only observable of `e` |
| 署名 | **signature** | what the trace constitutes |
| 選択のコイン | **the coin of choice** | the single token the optimisation leaves |
| 削ぎ落とし | **the shave** | collapse of a k-dimensional trace to one token |
| （選択したという）事実 | **fact-of-choice** | the binary content the coin carries losslessly |
| ダンバー地平 | **the Dunbar horizon** | the social distance beyond which a signature loses force |
| 損傷 / 修復 / 学習 | **damage / repair / learning** | the three timescales |
| 体制 I / II / III | **Regime I / II / III** | phase labels |
| 説明責任 | **accountability** | |
| 流暢さ | **fluency** | high regardless of verification |
| 目撃可能 | **witnessable** (auditable) | the normative target: make the weight witnessable |
| 未決定性 | **underdetermination** | why an answer emerges anyway |
| 寿命 | **lifespan** | the turnover-bounded repair channel |

---

## 0. Fixing the terms

**Meat proxy** — a human who relays AI output to others without reading, understanding,
or verifying it (Gruhn, 2026-08-03). Formalised as:

- Agents: `M` = model, `P` = proxy (meat), `S` = society (recipient).
- `theta_t` … the true optimum.
- `m_t = theta_t + B + eps_t` … model output. `B` is a **systematic bias independent
  of fluency**; `eps ~ N(0, sigma_t^2)` is diversity.
- `e ∈ [0,1]` … the proxy's verification effort; the meat's only value-add.
- `a_t = (1-e)·m_t + e·theta_t + nu_r + e·nu_v` … **the answer presented**.
  - `e=0` → `a = m + nu_r` (pure relay)
  - `e=1` → `a = theta + nu_r + nu_v` (full verification)
- `q` … audit (= someone reads) rate. **Not an exogenous parameter but another name
  for `e`** (§4).
- `w_act` … the weight society actually places on the human signature;
  `w_opt` … the weight warranted (inverse-variance).

---

## 1. The paradox, stated exactly

The term is self-contradictory.

1. **A proxy is by definition an instrument of verification.** A relay is worth being
   a relay only because it *selects* what passes; a relay that does not select is not
   a relay but **plumbing**.
2. **Yet the reward for being meat accrues from being plumbing.** The cost of
   verification `c_e` is private, certain, immediate; its benefit `q·Φ·B` is
   stochastic, delayed, and diffused onto others. Private incentive therefore always
   points at `e=0`.
3. **And `e=0` is unobservable.** A verified answer and a relayed answer are
   indistinguishable along the axis of fluency (§6).

The term therefore denotes a **verifier that does not verify**, and carries no
equilibrium inside its definition. This is not metaphor: it emerges below as an
*absorbing state*.

---

## 2. The model (dynamics)

| # | Equation | Meaning |
|---|---|---|
| D1 | `e = 1{ q·Φ·B > c_eff }` | verification is bang-bang: if cost exceeds benefit, none at all |
| D2 | `c_eff = c0·(1 + Γ(1-comp))`, `comp ← comp + ρ(e-comp)` | **unused competence atrophies and inflates verification cost** |
| D3 | `sigma ← sigma·(1 − κ(1-e))` | only output that passed unverified returns to training data, killing diversity |
| D4 | with prob `λ` the meat is replaced by a fresh one | generational turnover = the only repair path |
| D5 | `q_t = q_ext + β·e_bar_{t-1}` | **audit is endogenous**: mutual audit `β≤1` plus external audit `q_ext` |
| D6 | `tau ← tau + DTAU·q·(0.5−bad)·2` | society's trust `tau` learns at a rate **proportional to audit frequency** |

`w_opt = 1 − var_P/var_M` (inverse-variance), `w_act = tau·fluency`, `gap = w_act − w_opt`.

---

## 3. Result 1 — the meat is informationally void

Comparing the `e=0` system against **deleting the relay and asking the model
directly**, over 40,000 samples:

```
E|err|  direct to model : 0.3500
E|err|  via meat proxy  : 0.3502
delta                   : 0.0002        latency = +L0 (>0)
```

**The meat's value-add is 0; the difference is latency and noise only.** Gruhn's
"adds nothing but latency" is exactly this equation. The meat proxy's social utility
is therefore **negative** (latency + misattributed accountability) and has no positive
term.

---

## 4. Result 2 — the breaking point is set by three timescales, not one threshold

The breaking point is not a single number but the **ordering of three speeds**:
damage, repair, learning.

| Channel | Timescale | Value (generations) |
|---|---|---|
| σ damage (self-ingestion) | `−ln(1−κ(1−e))` | **0.040** /gen |
| σ repair | exogenous refresh `η` only (**`λ` does NOT repair σ**) | **0** (at η=0) |
| comp damage (atrophy) | `ρ(1−e)` | **0.047** /gen |
| comp repair (use + turnover) | `λ+ρe` | **0.080** /gen |
| τ learning | `2·DTAU·q` | **0.002** /gen |
| (ref) half-life equivalents σ-dmg / comp-repair / τ-learn | `ln2/rate` | 17.2 / 8.7 / 346 gen |
| (ref) a human career | — | ≈30 |

```
WITHDRAWN: the first revision's R = t_repair/t_damage = 1.74 and lambda*=0.116 are
retracted. (a) They mixed a VARIANCE half-life (factor 2) with a MEAN waiting time
1/lambda; (b) they mis-attributed sigma-repair to lambda. The correct statement is
the per-channel rate table above: sigma has no endogenous repair (non-convergent at
eta=0); comp converges for lambda > rho(1-2e)=0.035 (satisfied at 0.067).
See docs/ADDENDUM.md points 1-3.
```

- **σ is not repaired endogenously.** Self-ingestion 0.040/gen against repair 0
  (exogenous only). Diversity therefore collapses to the floor unless fresh external
  data η arrives, independent of lifespan. comp, by contrast, converges
  (repair 0.080 > damage 0.047); only comp is sustained by lifespan (turnover).
- **Learning of the weight takes 500 generations.** Within a 30-generation career
  `tau` moves ≈0.02. The weight is effectively **frozen on a human timescale** (§6).
- Diversity collapse can be judged by an upper bound on `κ`: not halving within a
  30-generation career requires `κ < 0.0115`. Real self-ingestion rates far exceed
  this, so **collapse saturates inside the first career**.

---

## 5. "As long as humans have a lifespan, it does not break" — half right, half wrong

**The right half.** Lifespan (= turnover rate `λ>0`) is the *only* escape from the
absorbing state `e*=0`. With `λ=0` (the same human stays; the role is automated or
institutionalised), `comp→0`, `c_eff→2.2·c0`, and **even raising audit to `q=0.98`
yields verification of only 0.153** (fresh meat: 0.512; ratio 3.35×). Lifespan
guarantees the *existence* of repair.

**The wrong half (corrected).** Lifespan (turnover λ) repairs **competence (comp)
only**, not diversity (σ). No amount of λ stops σ-collapse (demonstrated:
`rigor.lambda_does_not_repair_sigma` — σ-terminal = 0.005 for λ=0 / 0.067 / 0.3);
only exogenous data η stops it. The correct statement is therefore: **lifespan
guards against the atrophy of competence (the absorbing state), but not against the
collapse of diversity.** Lifespan's protection of competence is conditional on meat
continuing to exist: automate the role (λ→0) and the absorbing state surfaces.

---

## 6. What the weights should be

The weight society places on the human signature does not equal the weight warranted.
Under a realistic audit rate `q_ext=0.05` (**Regime I = the observed world**), terminal
values:

```
e_bar  = 0.000     (nobody verifies; below q_boot = 0.229)
w_opt  = -0.020    (warranted epistemic weight of the meat's signature; slightly
                    *negative*, by the relay-noise term)
w_act  = 0.95 -> 0.63   (weight actually placed; set by fluency and signature alone)
gap    = +0.65 .. +0.97   <- this system's "hidden total loss"
```

The warranted weight is inverse-variance, `w_i ∝ 1/var_i`. Meat that does not verify
shrinks no variance, so `w_opt ≈ 0` (strictly, negative by relay noise). Yet society
places `w_act ≈ fluency ≈ 0.95`. **The weight is set not by the variance of content
but by surface fluency and the bare fact that a human signed** — that is the manner of
the mis-weighting, and its difference `gap` is the system's entire loss.

And per §4, `tau`'s learning rate is proportional to `q`, so `gap` is not corrected on
a lifespan scale. **The weight does not fail to converge to its warranted value; it
converges more slowly than a human lifetime.**

---

## 7. How the answer emerges, and what is presented versus hidden

### 7.1 How it emerges
The presented answer `a` is **a single draw from a mixture** of the verified and
unverified populations. At `e_bar=0.21`: 21% at `theta+small noise`, 79% at
`theta+B+eps`. The recipient has **no means of knowing which draw arrived** (fluency
is identical for both). The answer therefore emerges not as "the optimum" but as
**"the optimum plus an unattributable mixture label"**. Effective error is pinned to
the prior expectation `E|a−theta| = (1−e)·E|B+eps| + e·E|nu|`, unimprovable per message.

### 7.2 Presented / hidden
- **Presented:** the answer (argmax), and the human signature.
- **Hidden:** `e` (whether it was verified), `w` (the weight), `sigma` (remaining
  diversity), and **the objective function itself**.

This is not conspiracy but **structure**. The hidden variable `e` is precisely the
variable whose exercise leaves no trace: `I(e ; a | fluency) ≈ 0`. The only observable
leaking `e` is **latency `L`** — but society optimises response time, so it deletes
`L`, i.e. **the system destroys its own only diagnostic** (D5, latency selection).

### 7.3 Proof that it cannot be reformed from within (phase structure)
Making audit endogenous (`q = q_ext + β·e_bar`) and solving for fixed points:

```
F'(q) = PHI*B / (c0_hi - c0_lo) = 0.673
beta_crit = 1 / F'(0) = 1.486   <- physical maximum of mutual audit is 1
```

Since `β_crit = 1.49 > 1`, **mutual audit alone cannot reach the critical mass of
verification; there is no interior tipping point.** Bootstrapping requires external
audit `q_ext ≥ q_boot = 0.229` as an *initial condition*, and full reform requires
`q_ext ≳ 0.86` (near-total audit). The intermediate `q_ext ≈ 0.3` is a **dead zone**:
`tau→0` (trust in the signature is destroyed) while `e_bar` reaches only 0.13 —
"losing trust without gaining verification", the worst region.

**Conclusion: a meat-proxy culture cannot be reformed by incremental internal change.
The bootstrap must come from outside.**

---

## 8. Conclusions

1. The meat proxy is not a thought experiment but **a dynamical system with an
   absorbing state**; the term's self-contradiction appears exactly as the stability
   of `e*=0`.
2. The meat's informational value-add is 0 (ΔE|err| = 0.0002); social utility negative.
3. The breaking point is not one threshold but the ordering of **per-channel
   rates**: σ has zero endogenous repair (non-convergent), comp converges, and τ
   learning (0.002) is the slowest channel. The first revision's `R=1.74` is
   withdrawn (docs/ADDENDUM.md).
4. "It does not break as long as there is lifespan" is right that lifespan guarantees
   the *existence* of repair, wrong that repair's *speed* beats damage. Lifespan hides
   collapse; it does not prevent it.
5. The weight is placed at `w_act≈0.95` where `w_opt≈0` is warranted; `gap≈0.7–1.0` is
   frozen permanently, because correction speed (500 gens) exceeds human time (30 gens).
6. **The answer is presented; the weight is hidden.** Hidden not by intent but because
   `e` is a variable that leaves no trace when exercised; and because the system
   optimises away its only diagnostic (latency), the hiding is self-maintaining.

---

# Appendix: the scale axis — signature, the coin of choice, generational convergence

> Where the first figure (`figures/fig_time_axis.png`) was the time axis, this is the
> scale axis. Reproduce: `python3 src/figure_scale.py` →
> `figures/fig_scale_axis.png` / `results/summary_scale.json`

## S1. Over what social distance does a signature retain force (the Dunbar horizon)

Audit `q` is a function of social distance (repeated games and cheap reputation make
`q` high on the inside). Per layer, `e_bar=F(q)` and `gap`:

| layer | n | q | e_bar | gap |
|---|---|---|---|---|
| support | 5 | 0.80 | 0.39 | 0.40 |
| sympathy | 15 | 0.50 | 0.18 | 0.65 |
| affinity | 50 | 0.25 | 0.01 | 0.94 |
| Dunbar | 150 | 0.10 | 0.00 | 0.97 |
| beyond | strangers | 0.05 | 0.00 | 0.97 |

**A signature/trace retains force only inside the Dunbar horizon; outside it, Regime I
(meat proxy) holds by structure.** Beyond the horizon it is not a lack of virtue but
*structure* that makes you a meat proxy. "Assembling the Dunbar number" is the design
act of choosing the scale at which your trace still has force.

## S2. The coin of choice (rate-distortion)

The trace `T` is k-dimensional (form, scale, medium, intent, duration, audience…).
Coding each dimension at distortion 12.5% costs `R_dim = 0.5·log2(1/0.125) = 1.5 bit`,
so `H(T)=1.5k bit`. The coin is 1 bit, so the share of the trace it can carry is
`1/(1.5k)`.

```
k=6  ->  carried 11%  /  discarded 89%
```

**Exactly one thing travels losslessly: the binary fact that a signer chose = `e`
itself.** Placing the coin is therefore not handing over the optimum but **handing
over accountability**. The shave to one dimension destroys the content channel but not
the accountability channel — indeed the coin's entire content is accountability.

## S3. Generational convergence is conditional

The competence channel converges iff `λ > ρ(1−2e) = 0.035`. The diversity channel is
non-convergent in λ regardless (repair = exogenous η=0 < damage 0.040).

```
comp : lambda=0.067 > 0.035   convergent
sigma: eta=0        < 0.040   non-convergent (lifespan cannot save it)
```

- **Competence (comp) channel:** turnover always resets `comp=1`, so it converges
  generationally unconditionally.
- **Diversity (σ) channel:** does **not** converge via λ; converges only for
  exogenous `η > κ(1−e)`.

So "individually they collide, generationally they converge" is a theorem for
competence and a **conditional theorem** for diversity.

---

*Files: `src/model.py` (model), `src/figure_time.py` (aggregation + figures),
`results/summary_time.json` / `results/sim_raw.json` (numbers),
`figures/fig_time_axis.png` (6-panel figure).*
