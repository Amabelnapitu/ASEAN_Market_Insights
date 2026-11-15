"""Tests for the attribution, valuation, home-market, scenario and robustness calculations."""

import numpy as np
import pandas as pd
import pytest

from asean_insights import metrics
from asean_insights.data import from_usd, to_usd
from asean_insights.report import with_ci


def fridays(n, start="2020-01-03"):
    return pd.date_range(start, periods=n, freq="W-FRI")


def test_ols_recovers_coefficients_and_sensible_standard_errors():
    rng = np.random.default_rng(1)
    n = 2000
    x = rng.normal(0, 1, n)
    y = 0.5 + 2.0 * x + rng.normal(0, 1, n)
    fit = metrics.ols(y, np.column_stack([np.ones(n), x]))
    assert fit["coef"] == pytest.approx([0.5, 2.0], abs=0.1)
    # With independent noise of variance 1, the slope's standard error is about 1/sqrt(n).
    assert fit["se"][1] == pytest.approx(1 / np.sqrt(n), rel=0.2)
    assert fit["resid_var"] == pytest.approx(1.0, rel=0.1)


def test_return_attribution_adds_up_to_the_usd_return():
    idx = pd.bdate_range("2021-01-01", periods=3)
    price = pd.DataFrame({"Bank": [100.0, 105.0, 110.0]}, index=idx)  # +10% price
    total_local = pd.DataFrame({"Bank": [100.0, 108.0, 115.0]}, index=idx)  # +15% with dividends
    usd = total_local * pd.Series([0.50, 0.48, 0.45], index=idx).to_frame("Bank").to_numpy()  # FX -10%
    out = metrics.return_attribution(price, total_local, usd).loc["Bank"]
    assert out["Price"] == pytest.approx(0.10)
    assert out["Dividends"] == pytest.approx(0.05)
    assert out["Total in USD"] == pytest.approx(1.15 * 0.9 - 1)
    assert out["Price"] + out["Dividends"] + out["Currency"] == pytest.approx(out["Total in USD"])


def test_trailing_dividend_yield_only_counts_the_last_year():
    idx = pd.bdate_range("2021-01-01", "2022-06-30")
    prices = pd.DataFrame({"A": 50.0}, index=idx)
    dividends = pd.DataFrame({"A": 0.0}, index=idx)
    dividends.loc["2021-03-01", "A"] = 9.0  # more than a year before the end: excluded
    dividends.loc["2021-09-01", "A"] = 1.0
    dividends.loc["2022-03-01", "A"] = 1.5
    assert metrics.trailing_dividend_yield(dividends, prices)["A"] == pytest.approx(2.5 / 50)


def test_valuation_table_implies_missing_roe_and_measures_premium_to_the_line():
    fundamentals = pd.DataFrame({
        "P/B": [1.0, 2.0, 3.0, 4.0],
        "P/E": [10.0, 10.0, 10.0, 10.0],
        "Reported ROE": [np.nan, 0.20, 0.30, 0.40],
    }, index=["A", "B", "C", "D"])
    out = metrics.valuation_table(fundamentals, pd.Series(0.05, index=fundamentals.index))
    assert out.loc["A", "ROE"] == pytest.approx(0.10)  # 1.0 / 10.0
    assert out.loc["A", "ROE source"] == "implied"
    # All four sit exactly on P/B = 10 * ROE, so nobody trades at a premium or discount.
    assert out["P/B vs peers"].abs().max() == pytest.approx(0, abs=1e-9)
    assert out.attrs["fit"]["slope"] == pytest.approx(10.0)


def test_current_state_reports_distance_from_peak_and_percentile():
    idx = fridays(120)
    path = np.concatenate([np.linspace(100, 200, 100), np.linspace(200, 150, 20)])
    prices = pd.DataFrame({"A": path}, index=idx)
    market = pd.Series(np.linspace(100, 120, 120), index=idx)
    row = metrics.current_state(prices, market).loc["A"]
    assert row["Off peak"] == pytest.approx(150 / 200 - 1)
    assert row["Last 52 weeks"] == pytest.approx(150 / prices["A"].iloc[-53] - 1)
    assert row["52-week percentile"] == 0  # the latest year is the worst one-year window


def test_home_vs_us_attributes_moves_to_the_right_market():
    rng = np.random.default_rng(2)
    idx = fridays(300)
    home = pd.DataFrame({"SG": rng.normal(0, 0.02, 300)}, index=idx)
    us = pd.Series(rng.normal(0, 0.02, 300), index=idx)
    bank = pd.DataFrame({"Bank": 0.002 + 1.2 * home["SG"] + rng.normal(0, 0.002, 300)}, index=idx)
    row = metrics.home_vs_us(bank, home, us, {"Bank": "SG"}).loc["Bank"]
    assert row["R² home only"] > 0.95
    assert row["R² US only"] < 0.1
    assert row["Home beta"] == pytest.approx(1.2, abs=0.05)
    assert abs(row["US beta"]) < row["US beta ±"] + 0.02
    assert row["Alpha a year"] == pytest.approx(0.002 * 52, abs=row["Alpha a year ±"])


def test_scenario_prediction_matches_the_fitted_line_and_widens_for_bigger_shocks():
    rng = np.random.default_rng(3)
    idx = fridays(400)
    spx = pd.Series(rng.normal(0, 0.02, 400), index=idx)
    fx = pd.DataFrame({"SG": rng.normal(0, 0.005, 400)}, index=idx)
    rate = pd.Series(rng.normal(0, 0.1, 400), index=idx)
    bank = pd.DataFrame({"Bank": 0.5 * spx + 1.5 * fx["SG"] - 0.01 * rate + rng.normal(0, 0.01, 400)})
    models = metrics.scenario_model(bank, spx, fx, rate, {"Bank": "SG"})
    small = metrics.scenario_predict(models, -0.02, 0.0, 0.0).loc["Bank"]
    big = metrics.scenario_predict(models, -0.20, -0.05, 0.5).loc["Bank"]
    assert big["Expected"] == pytest.approx(0.5 * -0.20 + 1.5 * -0.05 - 0.01 * 0.5, abs=0.02)
    assert big["High"] - big["Low"] > small["High"] - small["Low"]
    assert small["Low"] < small["Expected"] < small["High"]


def test_rolling_betas_track_a_change_in_sensitivity():
    rng = np.random.default_rng(4)
    idx = fridays(208)
    market = pd.Series(rng.normal(0, 0.02, 208), index=idx)
    rate = pd.Series(rng.normal(0, 0.1, 208), index=idx)
    beta = np.where(np.arange(208) < 104, 0.2, 1.0)  # sensitivity jumps halfway
    bank = pd.DataFrame({"Bank": beta * market + rng.normal(0, 0.002, 208)}, index=idx)
    out = metrics.rolling_betas(bank, market, rate, window=52)
    assert out["Market beta"]["Bank"].iloc[0] == pytest.approx(0.2, abs=0.05)
    assert out["Market beta"]["Bank"].iloc[-1] == pytest.approx(1.0, abs=0.05)
    assert (out["Market beta ±"]["Bank"] > 0).all()


def test_regime_test_reports_the_annualised_difference():
    idx = fridays(8)
    returns = pd.DataFrame({"A": [0.01, 0.02, 0.01, 0.02, -0.01, 0.00, -0.01, 0.00]}, index=idx)
    flag = pd.Series([False] * 4 + [True] * 4, index=idx)
    row = metrics.regime_test(returns, flag).loc["A"]
    assert row["Difference"] == pytest.approx((-0.005 - 0.015) * 52)
    assert row["Difference ±"] > 0


def test_from_usd_reverses_to_usd():
    idx = pd.bdate_range("2021-01-01", periods=3)
    usd = pd.DataFrame({"ADR": [30.0, 31.0, 29.0]}, index=idx)
    fx = pd.DataFrame({"PHPUSD=X": [0.02, 0.021, 0.019]}, index=idx)
    local = from_usd(usd, fx, {"ADR": "PHPUSD=X"})
    assert local["ADR"].tolist() == pytest.approx([1500, 31 / 0.021, 29 / 0.019])
    assert to_usd(local, fx, {"ADR": "PHPUSD=X"})["ADR"].tolist() == pytest.approx(usd["ADR"].tolist())


def test_with_ci_turns_estimate_and_half_width_into_a_range():
    df = pd.DataFrame({"Beta": [0.5], "Beta ±": [0.2], "Weeks": [100]}, index=["A"])
    out = with_ci(df, {"Beta": "{:.2f}", "Weeks": "{:.0f}"})
    assert out.loc["A", "Beta"] == "0.50 (0.30 to 0.70)"
    assert out.loc["A", "Weeks"] == "100"
    assert list(out.columns) == ["Beta", "Weeks"]
