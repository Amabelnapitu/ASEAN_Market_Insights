import numpy as np
import pandas as pd
import pytest

from asean_insights import metrics
from asean_insights.data import to_usd


def days(n, start="2021-01-01"):
    return pd.bdate_range(start, periods=n)


def test_rebase_starts_where_every_column_has_a_price():
    # The first row is a holiday for B. Dividing by the first row would turn B into all-NaN.
    prices = pd.DataFrame({"A": [10.0, 11.0, 12.0], "B": [np.nan, 50.0, 55.0]}, index=days(3))
    out = metrics.rebase(prices)
    assert out.index[0] == prices.index[1]
    assert out.iloc[0].tolist() == [100.0, 100.0]
    assert out["B"].iloc[-1] == pytest.approx(110.0)


def test_weekly_returns_are_friday_to_friday():
    idx = days(15, "2021-01-04")  # three full weeks, Monday to Friday
    prices = pd.DataFrame({"A": np.arange(1.0, 16.0)}, index=idx)
    out = metrics.weekly_returns(prices)
    assert all(d.day_name() == "Friday" for d in out.index)
    assert out["A"].tolist() == pytest.approx([10 / 5 - 1, 15 / 10 - 1])


def test_drawdown_measures_fall_from_peak():
    prices = pd.DataFrame({"A": [100.0, 120.0, 90.0, 120.0, 60.0]}, index=days(5))
    dd = metrics.drawdown(prices)["A"]
    assert dd.tolist() == pytest.approx([0, 0, -0.25, 0, -0.5])


def test_performance_table_on_a_steady_grower():
    idx = pd.date_range("2020-01-03", periods=105, freq="W-FRI")  # two years of Fridays
    prices = pd.DataFrame({"A": 100 * 1.002 ** np.arange(105)}, index=idx)
    row = metrics.performance_table(prices).loc["A"]
    assert row["Total return"] == pytest.approx(1.002**104 - 1)
    assert row["Annualised return"] == pytest.approx(1.002**52 - 1, rel=0.01)
    assert row["Annualised volatility"] == pytest.approx(0, abs=1e-9)
    assert row["Max drawdown"] == 0


def test_factor_betas_recover_known_coefficients():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2020-01-03", periods=300, freq="W-FRI")
    market = pd.Series(rng.normal(0, 0.02, 300), index=idx)
    rates = pd.Series(rng.normal(0, 0.1, 300), index=idx)
    bank = 0.001 + 0.8 * market - 0.05 * rates + rng.normal(0, 0.001, 300)
    out = metrics.factor_betas(pd.DataFrame({"Bank": bank}), market, rates).loc["Bank"]
    assert out["Market beta"] == pytest.approx(0.8, abs=0.02)
    assert out["Rate beta"] == pytest.approx(-0.05, abs=0.005)
    assert out["R squared"] > 0.95


def test_regime_returns_split_by_flag_and_annualise():
    idx = pd.date_range("2020-01-03", periods=4, freq="W-FRI")
    returns = pd.DataFrame({"A": [0.01, 0.01, -0.02, -0.02]}, index=idx)
    flag = pd.Series([False, False, True, True], index=idx)
    out = metrics.regime_returns(returns, flag, ("Normal", "Inverted"))
    assert out.loc["A", "Normal"] == pytest.approx(0.52)
    assert out.loc["A", "Inverted"] == pytest.approx(-1.04)
    assert out.attrs["weeks"] == {"Normal": 2, "Inverted": 2}


def test_stress_weeks_flags_the_largest_vix_rises():
    idx = pd.date_range("2020-01-03", periods=21, freq="W-FRI")
    vix = pd.Series(20.0, index=idx)
    vix.iloc[10] = 40.0  # one spike, then back down
    stress = metrics.stress_weeks(vix, quantile=0.9)
    assert stress.sum() == 1 and stress.idxmax() == idx[10]


def test_to_usd_carries_fx_over_currency_holidays():
    idx = days(3)
    local = pd.DataFrame({"SG Bank": [10.0, 10.0, 10.0], "ADR": [5.0, 6.0, 7.0]}, index=idx)
    fx = pd.DataFrame({"SGDUSD=X": [0.70, 0.80]}, index=[idx[0], idx[2]])  # no quote on day 2
    out = to_usd(local, fx, {"SG Bank": "SGDUSD=X", "ADR": None})
    assert out["SG Bank"].tolist() == pytest.approx([7.0, 7.0, 8.0])
    assert out["ADR"].tolist() == [5.0, 6.0, 7.0]
