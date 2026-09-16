# meat-proxy (EN)

**Modelling the meat proxy — the human who relays AI output without verifying it —
as a dynamical system with an absorbing state, and computing where it breaks,
what its weights are, and over what range a signature retains force.**

This is the English companion of the Japanese repository. The code, figures and
numerical results are **identical** (verified: all numerics in `results/*.json`
match the JA repo bit-for-bit; only human-readable verdict strings are translated).
The full derivation lives in [`docs/ANALYSIS.md`](docs/ANALYSIS.md); the response to the 12-point scientific-completeness review, including all withdrawals, is in [`docs/ADDENDUM.md`](docs/ADDENDUM.md).

> **On term-values.** Translating a coinage forces choices about what each word is
> worth in the target language. The choices made here are listed openly in
> [`docs/ANALYSIS.md` §0.0](docs/ANALYSIS.md) — a glossary of term-values. Where the
> slang is blunt ("meat"), it is kept blunt; where a neutral register serves the
> argument, "the proxy" / "the relay" is used instead.

---

## Headline results

| # | Result | Value |
|---|---|---|
| 1 | Informational value of the meat (vs. asking the model directly) | `ΔE\|err\| = 0.0002` (= 0; adds latency only) |
| 2 | `e*=0` is an absorbing state (hysteresis ratio) | at the same audit `q=0.98`: atrophied 0.153 vs fresh 0.512 (**3.35×**) |
| 3 | Per-channel rates (/gen) σ-dmg / σ-repair / comp-dmg / comp-repair / τ-learn | **0.040 / 0 (exogenous only) / 0.047 / 0.080 / 0.002** → σ has no endogenous repair; comp converges; τ slowest |
| 4 | Impossibility of internal reform (mutual-audit critical point) | `β_crit = 1.49 > 1` (physical max is 1) → no interior tipping point |
| 5 | Misallocated credence (Regime I, `q_ext=0.05`) | `w_opt ≈ -0.02` vs `w_act ≈ 0.95` → `gap ≈ 0.65–0.97`, frozen |
| 6 | Range of a signature's force (Dunbar horizon) | inside `gap=0.40` → outside `gap=0.97` (outside = meat-proxy by structure) |
| 7 | Share of a trace the coin of choice (1 bit) can carry | at `k=6`: **11%** (89% shaved; what survives losslessly is the fact-of-choice = `e`) |
| 8 | Condition for generational convergence | comp: `λ > ρ(1−2e)=0.035` (satisfied at 0.067). σ: needs `η > κ(1−e)=0.040` (λ is powerless) |
| 9 | `I(e;a｜fluency)` measured | **0.0001 bit** (ex-ante `I(e;a)=0.022`; ex-post `I(e;a｜θ)=0.734`) |
| 10 | `w_opt` derived from MMSE + covariance | **exactly 0** at `e=0` (the relayed answer is strictly worse than a direct query) |

---

## Figures

### Time axis — equivalence · bootstrap · timescales · weights · collapse · hysteresis
![time axis](figures/fig_time_axis.png)

### Scale axis — range of a signature · the coin of choice · generational convergence
![scale axis](figures/fig_scale_axis.png)

### Corrected axis — one-scale rates · λ↛σ demonstrated · 2-D fixed point · MMSE weight · measured MI · utility pricing diversity
![rigor](figures/fig_rigor.png)

> **The rate-distortion part of the coin of choice (S2) is a metaphorical application** (point 11): the trace is not literally a Gaussian source and the 1-bit coin is conceptual.

---

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

make figures      # = python3 src/model.py && python3 src/figure_time.py && python3 src/figure_scale.py
```

Outputs: `figures/*.png`, `results/*.json`

---

## Repository layout

```
.
├── README.md                  <- this file
├── docs/
│   └── ANALYSIS.md            <- full text (derivation, interpretation, appendix: scale axis)
├── src/
│   ├── model.py               <- the dynamical system (population, endogenous audit, fixed points)
│   ├── figure_time.py         <- time-axis aggregation and the 6-panel figure
│   ├── figure_scale.py        <- scale-axis aggregation and the 3-panel figure
│   └── rigor.py               <- scientific-completeness layer (corrections, MI, MMSE, sensitivity, multiseed)
├── figures/
│   ├── fig_time_axis.png
│   ├── fig_scale_axis.png
│   └── fig_rigor.png
├── results/
│   ├── sim_raw.json           <- raw trajectories
│   ├── summary_time.json      <- time-axis aggregates
│   ├── summary_scale.json     <- scale-axis aggregates
│   └── rigor.json             <- corrected-analysis aggregates
├── tests/
│   └── test_rigor.py          <- 12 invariant tests (pytest)
├── .github/workflows/ci.yml   <- CI (py3.11/3.12, pytest + figure regeneration)
├── Makefile
├── requirements.txt
└── LICENSE
```

---

## The model in one screen

Agents: `M` (model) / `P_i` (proxy = meat) / `S` (society = recipient).

```
m   = theta + B + eps                     # model output; B = bias independent of fluency
a   = (1-e)*m + e*theta + nu_r + e*nu_v   # the answer presented; e = verification effort
e   = 1{ q*PHI*B > c0*(1+GAMMA*(1-comp)) }   # verification is bang-bang
comp<- comp + RHO*(e-comp)                # unused competence atrophies
sig <- sig*(1-kappa*(1-e_bar))            # self-ingestion of unverified output kills diversity
q   = q_ext + beta*e_bar                  # audit is endogenous (= "someone reads" = another name for e)
tau <- tau + DTAU*q*(0.5-bad)*2           # society's trust learns at a rate proportional to audit frequency
w_opt = 1 - var_P/var_M ;  w_act = tau*fluency ;  gap = w_act - w_opt
```

Parameter table and propositions P1–P7: see [`src/model.py`](src/model.py) docstring
and [`docs/ANALYSIS.md`](docs/ANALYSIS.md).

---

## Note on scope

Every number here is **under this model's default parameters**. These are not
measurements of the world; they are conditional claims: *if* the meat proxy is read
as this dynamical system, *then* its breaking points fall here. Varying parameters
to probe sensitivity is an intended use.

## License

MIT — see [LICENSE](LICENSE).
