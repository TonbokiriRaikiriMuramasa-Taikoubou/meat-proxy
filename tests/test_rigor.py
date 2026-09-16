"""Invariant tests for the meat-proxy model and its scientific-completeness layer.

Run:  pytest -q
These guard the corrections from the domain review (points 1-12):
timescale consistency, q-clipping, the fresh-meat approximation, the 2-D fixed
point, the MMSE weight, the computed mutual informations, and reproducibility.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import model as M          # noqa: E402
import rigor as RG         # noqa: E402


# ---- point 1: half-life defined on the std; variance half-life is exactly half
def test_half_life_consistency():
    for kappa in (0.01, 0.05, 0.10):
        for e in (0.0, 0.21, 0.5):
            assert M.t_half_var(kappa, e) == pytest.approx(M.t_half(kappa, e) / 2)
    # std half-life must be the LARGER (slower) of the two
    assert M.t_half(0.05, 0.21) > M.t_half_var(0.05, 0.21)


# ---- point 2: all channels reported on one rate scale; no mixed quantiles
def test_channel_rates_single_scale():
    cr = RG.channel_rates(0.21)
    assert set(cr) == {"sigma_damage", "sigma_repair", "comp_damage",
                       "comp_repair", "tau_learn"}
    assert cr["sigma_damage"] > 0
    assert cr["sigma_repair"] == M.SIGMA_REFRESH


# ---- point 3: lambda repairs comp, NOT sigma
def test_lambda_does_not_repair_sigma():
    rows = RG.lambda_does_not_repair_sigma(lams=(0.0, 0.067, 0.30))
    sig = [r["sigma_terminal"] for r in rows]
    comp = [r["comp_terminal"] for r in rows]
    assert max(sig) - min(sig) < 1e-9, "sigma must be invariant to lambda"
    assert comp[-1] > comp[0], "comp must increase with lambda"
    assert not RG.sigma_convergent(0.0)
    assert RG.comp_convergent(M.LAM, M.RHO, 0.21)


# ---- point 4: q clipped to [0,1] everywhere, incl. fixed-point analysis
def test_q_clipped():
    assert 0.0 <= M.F(-3.0) <= 1.0
    assert 0.0 <= M.F(7.0) <= 1.0
    e, c = RG.fixed_point_2d(5.0, beta=5.0, lam=0.3)   # absurd q_ext, beta
    assert 0.0 <= e <= 1.0 and 0.0 <= c <= 1.0


# ---- point 5: F is the fresh-meat approximation; atrophy lowers the share
def test_F_fresh_vs_atrophied():
    q = 0.9
    assert M.F(q) == pytest.approx(RG.F_atrophied(q, 1.0))
    assert RG.F_atrophied(q, 0.0) < RG.F_atrophied(q, 1.0)


# ---- point 6: 2-D fixed point in the unit square and monotone in q_ext
def test_fixed_point_2d_sane():
    prev = -1
    for q in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
        e, c = RG.fixed_point_2d(q, 1.0, M.LAM)
        assert 0.0 <= e <= 1.0 and 0.0 <= c <= 1.0
        assert e >= prev - 1e-9
        prev = e


# ---- point 7: MMSE weight in [0,1]; exactly 0 at e=0; recovers inverse-variance at e=1
def test_w_opt_mmse():
    assert RG.w_opt_mmse(0.0)["w_opt"] == pytest.approx(0.0, abs=1e-9)
    for e in (0.2, 0.5, 0.8, 1.0):
        assert 0.0 <= RG.w_opt_mmse(e)["w_opt"] <= 1.0
    w1 = RG.w_opt_mmse(1.0)
    # at e=1 cov=0 so w* = var_M/(var_a+var_M)
    assert w1["w_opt"] == pytest.approx(w1["var_M"] / (w1["var_a"] + w1["var_M"]))


# ---- point 8: mutual informations ordered  fluency ~ 0  <  ex-ante  <<  ex-post
def test_mi_trio():
    mi = RG.mi_trio(n=200000, seed=1)
    assert mi["I_e_fluency"] < 0.01
    assert mi["I_e_a_exante"] < 0.10
    assert mi["I_e_a_given_theta"] > 0.4
    assert mi["I_e_fluency"] < mi["I_e_a_exante"] < mi["I_e_a_given_theta"]


# ---- point 9: collapsed diversity is a loss in the utility
def test_utility_prices_diversity():
    U_floor, _ = RG.utility(0.0, M.SIG_MIN)
    U_half, _ = RG.utility(0.0, 0.5)
    assert U_floor < U_half


# ---- point 10: Dunbar gaps monotone in layer and robust beyond the horizon
def test_dunbar_sensitivity():
    sens = RG.dunbar_sensitivity()
    for key, rows in sens.items():
        gaps = [r["gap"] for r in rows]
        assert gaps[-1] == pytest.approx(0.97, abs=0.01), "beyond-horizon gap robust"
        assert gaps[0] <= gaps[-1]


# ---- point 12: reproducibility (same seed -> identical trajectories)
def test_reproducible():
    a = M.run_endogenous(0.05, 1.0, T=120, seed=42)
    b = M.run_endogenous(0.05, 1.0, T=120, seed=42)
    assert [r["e_bar"] for r in a] == [r["e_bar"] for r in b]
    assert [round(r["sigma"], 12) for r in a] == [round(r["sigma"], 12) for r in b]


# ---- headline equivalence still holds
def test_equivalence_void():
    rng = np.random.default_rng(3)
    n = 50000
    eps = rng.normal(0, M.SIG_MIN, n)
    direct = np.abs(M.B + eps)
    relay = np.abs(M.B + eps + rng.normal(0, M.DELTA_R, n))
    assert abs(relay.mean() - direct.mean()) < 0.01
