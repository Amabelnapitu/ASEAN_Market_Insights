"""Return, risk and sensitivity calculations. Pure functions: DataFrames in, DataFrames out."""

from __future__ import annotations

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


def rebase(prices: pd.DataFrame, base: float = 100.0) -> pd.DataFrame:
    """Index every column to `base` on the first date where all columns have a price."""
    filled = prices.ffill().dropna()
    return filled / filled.iloc[0] * base


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Friday-to-Friday simple returns.

    Weekly rather than daily, because these markets close at different hours: a US move shows
    up in Asian prices the next day, which makes daily correlations look weaker than they are.
    """
    weekly = prices.ffill().resample("W-FRI").last()
    return weekly.pct_change(fill_method=None).dropna(how="all")


def drawdown(prices: pd.DataFrame) -> pd.DataFrame:
    """Fall from the running peak, as a negative fraction."""
    filled = prices.ffill()
    return filled / filled.cummax() - 1


def performance_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Total return, annualised return and volatility, return per unit of risk, worst drawdown."""
    rows = {}
    for col in prices.columns:
        p = prices[col].dropna()
        years = (p.index[-1] - p.index[0]).days / 365.25
        total = p.iloc[-1] / p.iloc[0] - 1
        cagr = (1 + total) ** (1 / years) - 1
        vol = weekly_returns(p.to_frame())[col].std() * np.sqrt(WEEKS_PER_YEAR)
        rows[col] = {
            "Total return": total,
            "Annualised return": cagr,
            "Annualised volatility": vol,
            "Return / risk": cagr / vol,
            "Max drawdown": drawdown(p.to_frame())[col].min(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def factor_betas(returns: pd.DataFrame, market: pd.Series, rate_change: pd.Series) -> pd.DataFrame:
    """Regress each column on the market return and the change in the 10-year yield.

    return = alpha + market_beta * market + rate_beta * rate_change + error

    `rate_change` is in percentage points, so rate_beta is the return for a 1pp rise in yields
    with the equity market held fixed.
    """
    rows = {}
    for col in returns.columns:
        df = pd.concat([returns[col], market, rate_change], axis=1, keys=["y", "mkt", "rate"]).dropna()
        x = np.column_stack([np.ones(len(df)), df["mkt"], df["rate"]])
        coef, *_ = np.linalg.lstsq(x, df["y"].to_numpy(), rcond=None)
        resid = df["y"].to_numpy() - x @ coef
        r2 = 1 - resid.var() / df["y"].var(ddof=0)
        rows[col] = {"Market beta": coef[1], "Rate beta": coef[2], "R squared": r2, "Weeks": len(df)}
    return pd.DataFrame.from_dict(rows, orient="index")


def regime_returns(returns: pd.DataFrame, flag: pd.Series, labels: tuple[str, str]) -> pd.DataFrame:
    """Annualised mean weekly return when `flag` is False (labels[0]) and True (labels[1])."""
    flag = flag.reindex(returns.index).ffill().fillna(False).astype(bool)
    out = pd.DataFrame(
        {labels[0]: returns[~flag].mean(), labels[1]: returns[flag].mean()}
    ) * WEEKS_PER_YEAR
    out.attrs["weeks"] = {labels[0]: int((~flag).sum()), labels[1]: int(flag.sum())}
    return out


def stress_weeks(vix: pd.Series, quantile: float = 0.9) -> pd.Series:
    """True in weeks where the VIX rose by more than its `quantile` weekly change."""
    change = vix.ffill().resample("W-FRI").last().pct_change(fill_method=None).dropna()
    return change > change.quantile(quantile)


def stress_returns(returns: pd.DataFrame, stress: pd.Series) -> pd.DataFrame:
    """Average weekly return in stress weeks and in all other weeks (not annualised)."""
    stress = stress.reindex(returns.index).fillna(False).astype(bool)
    return pd.DataFrame({"Stress weeks": returns[stress].mean(), "Other weeks": returns[~stress].mean()})
