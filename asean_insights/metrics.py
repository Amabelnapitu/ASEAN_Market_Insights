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
    with the equity market held fixed. The ± columns are half-widths of 95% intervals using
    Newey-West standard errors (see `ols`).
    """
    rows = {}
    for col in returns.columns:
        df = pd.concat([returns[col], market, rate_change], axis=1, keys=["y", "mkt", "rate"]).dropna()
        fit = ols(df["y"].to_numpy(), _design(df["mkt"], df["rate"]))
        rows[col] = {"Market beta": fit["coef"][1], "Market beta ±": Z95 * fit["se"][1],
                     "Rate beta": fit["coef"][2], "Rate beta ±": Z95 * fit["se"][2],
                     "R squared": fit["r2"], "Weeks": fit["n"]}
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


# --- Statistical helpers -------------------------------------------------------------------

Z95 = 1.96  # two-sided 95% normal critical value


def ols(y: np.ndarray, x: np.ndarray, lags: int = 4) -> dict:
    """Least squares with Newey-West standard errors.

    `x` must already include a constant column. Newey-West errors stay valid when the noise
    is larger in some periods than others (2020 versus 2023, say) and when consecutive weeks
    are mildly correlated, both of which hold for weekly stock returns. Plain OLS errors would
    make the estimates look more precise than they are.
    """
    n, k = x.shape
    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ coef
    xtx_inv = np.linalg.pinv(x.T @ x)
    xe = x * resid[:, None]
    meat = xe.T @ xe
    for lag in range(1, min(lags, n - 1) + 1):
        weight = 1 - lag / (lags + 1)
        cross = xe[lag:].T @ xe[:-lag]
        meat += weight * (cross + cross.T)
    cov = xtx_inv @ meat @ xtx_inv * n / max(n - k, 1)
    r2 = 1 - resid.var() / y.var() if y.var() > 0 else np.nan
    return {"coef": coef, "se": np.sqrt(np.diag(cov)), "cov": cov, "r2": r2,
            "resid_var": resid @ resid / max(n - k, 1), "n": n}


def _design(*columns: pd.Series | np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(columns[0])), *columns])


def mean_difference(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Mean of `b` minus mean of `a`, and the half-width of its 95% interval (Welch)."""
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return np.nan, np.nan
    se = np.sqrt(a.var() / len(a) + b.var() / len(b))
    return b.mean() - a.mean(), Z95 * se


# --- 1. Where the return came from ---------------------------------------------------------

def return_attribution(prices_local: pd.DataFrame, total_local: pd.DataFrame,
                       total_usd: pd.DataFrame) -> pd.DataFrame:
    """Split each column's US-dollar total return into price, dividends and currency.

    Steps, each measured on the same dates:
      Price      = change in the local share price, dividends excluded
      Dividends  = local total return (dividends reinvested) minus the price change
      Currency   = US-dollar total return minus the local total return
    The three add up exactly to the US-dollar total return. The currency step includes the
    small cross effect of the currency moving on the dividends and price gains too.
    """
    frames = [prices_local, total_local, total_usd]
    common = frames[0].ffill().dropna().index
    for f in frames[1:]:
        common = common.intersection(f.ffill().dropna().index)
    first, last = common[0], common[-1]

    def growth(f: pd.DataFrame) -> pd.Series:
        f = f.ffill()
        return f.loc[last] / f.loc[first] - 1

    price, local, usd = (growth(f) for f in frames)
    out = pd.DataFrame({
        "Price": price,
        "Dividends": local - price,
        "Currency": usd - local,
        "Total in USD": usd,
    })
    out.attrs["period"] = (first, last)
    return out


# --- 2. Valuation and where each bank stands today ------------------------------------------

def trailing_dividend_yield(dividends: pd.DataFrame, prices: pd.DataFrame, days: int = 365) -> pd.Series:
    """Cash dividends paid in the last `days` as a share of the latest price.

    Uses ex-dividend dates, so it can include one-off special dividends.
    """
    last = prices.ffill().iloc[-1]
    end = prices.index[-1]
    paid = dividends.loc[end - pd.Timedelta(days=days) + pd.Timedelta(days=1):end].sum()
    return paid / last


def valuation_table(fundamentals: pd.DataFrame, dividend_yield: pd.Series) -> pd.DataFrame:
    """Price-to-book, price-to-earnings, return on equity and dividend yield, per bank.

    Where Yahoo reports no ROE, it is implied from the two multiples: ROE = earnings / book
    = (price / book) / (price / earnings).
    """
    out = fundamentals[["P/B", "P/E"]].copy()
    implied = fundamentals["P/B"] / fundamentals["P/E"]
    reported = fundamentals.get("Reported ROE", pd.Series(np.nan, index=fundamentals.index))
    out["ROE"] = reported.fillna(implied)
    out["ROE source"] = np.where(reported.notna(), "reported", "implied")
    out["Dividend yield"] = dividend_yield.reindex(out.index)
    fit = pb_roe_fit(out)
    out["P/B vs peers"] = fit["premium"]
    out.attrs["fit"] = {k: v for k, v in fit.items() if k != "premium"}
    return out


def pb_roe_fit(valuation: pd.DataFrame) -> dict:
    """Straight line of P/B on ROE across the banks, and each bank's premium to that line.

    The premium is actual P/B divided by the P/B the line predicts for that bank's ROE, minus
    one. A bank priced 30% above the line costs 30% more per unit of book than peers with the
    same profitability. With a handful of banks this is a rough guide, not a fair value.
    """
    v = valuation[["P/B", "ROE"]].dropna()
    if len(v) < 3:
        return {"slope": np.nan, "intercept": np.nan, "r2": np.nan,
                "premium": pd.Series(np.nan, index=valuation.index)}
    slope, intercept = np.polyfit(v["ROE"], v["P/B"], 1)
    predicted = intercept + slope * valuation["ROE"]
    resid = v["P/B"] - (intercept + slope * v["ROE"])
    r2 = 1 - resid.var() / v["P/B"].var()
    return {"slope": slope, "intercept": intercept, "r2": r2,
            "premium": valuation["P/B"] / predicted - 1}


def current_state(prices: pd.DataFrame, market: pd.Series, window: int = 52) -> pd.DataFrame:
    """Where each column stands now compared with its own history in the sample.

    - Off peak: today's fall from the highest price in the sample
    - Last 4 weeks, Last 52 weeks: recent returns
    - 52-week percentile: share of all past rolling 52-week returns below today's
      (90% means the last year was better than 90% of one-year windows in the sample)
    - Beta now / Beta average: one-year rolling beta to `market` today and over the sample
    """
    weekly = prices.ffill().resample("W-FRI").last()
    rets = weekly.pct_change(fill_method=None)
    mkt = market.ffill().resample("W-FRI").last().pct_change(fill_method=None)
    rows = {}
    for col in prices.columns:
        w = weekly[col].dropna()
        yearly = (w / w.shift(window) - 1).dropna()
        df = pd.concat([rets[col], mkt], axis=1, keys=["y", "m"]).dropna()
        rolling = df["y"].rolling(window).cov(df["m"]) / df["m"].rolling(window).var()
        rows[col] = {
            "Off peak": w.iloc[-1] / w.max() - 1,
            "Last 4 weeks": w.iloc[-1] / w.iloc[-5] - 1 if len(w) > 4 else np.nan,
            "Last 52 weeks": yearly.iloc[-1] if len(yearly) else np.nan,
            "52-week percentile": (yearly < yearly.iloc[-1]).mean() if len(yearly) > 1 else np.nan,
            "Beta now": rolling.iloc[-1] if rolling.notna().any() else np.nan,
            "Beta average": df["y"].cov(df["m"]) / df["m"].var() if len(df) > 2 else np.nan,
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# --- 3. Home market versus the US market ------------------------------------------------------

def home_vs_us(returns: pd.DataFrame, home: pd.DataFrame, us: pd.Series,
               home_of: dict[str, str]) -> pd.DataFrame:
    """How much of each bank's weekly move its home market and the US market explain.

    For each bank, three regressions on weekly returns:
      US only:    bank = a + b * S&P 500
      Home only:  bank = a + b * home market
      Both:       bank = alpha + b_home * home market + b_us * S&P 500
    Reports the R squared of each, the home and US betas from the joint model, and alpha
    (the bank's average weekly return not explained by either market, annualised) with its
    95% interval. An interval that spans zero means no reliable bank-specific outperformance.
    """
    rows = {}
    for col in returns.columns:
        df = pd.concat([returns[col], home[home_of[col]], us], axis=1, keys=["y", "home", "us"]).dropna()
        if len(df) < 20:
            continue
        y = df["y"].to_numpy()
        us_only = ols(y, _design(df["us"]))
        home_only = ols(y, _design(df["home"]))
        both = ols(y, _design(df["home"], df["us"]))
        rows[col] = {
            "R² US only": us_only["r2"],
            "R² home only": home_only["r2"],
            "R² both": both["r2"],
            "Home beta": both["coef"][1],
            "Home beta ±": Z95 * both["se"][1],
            "US beta": both["coef"][2],
            "US beta ±": Z95 * both["se"][2],
            "Alpha a year": both["coef"][0] * WEEKS_PER_YEAR,
            "Alpha a year ±": Z95 * both["se"][0] * WEEKS_PER_YEAR,
            "Weeks": len(df),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# --- 4. Scenarios ------------------------------------------------------------------------------

SCENARIO_FACTORS = ("S&P 500", "Home currency", "US 10Y change")


def scenario_model(returns_usd: pd.DataFrame, us: pd.Series, fx_returns: pd.DataFrame,
                   rate_change: pd.Series, fx_of: dict[str, str]) -> dict[str, dict]:
    """Fit, for each bank: USD return = a + b1 * S&P 500 + b2 * home currency + b3 * 10Y change.

    `fx_returns` holds weekly changes in each home currency's USD value, so a positive number
    means the currency strengthened. Returns the fitted `ols` result per bank.
    """
    models = {}
    for col in returns_usd.columns:
        df = pd.concat([returns_usd[col], us, fx_returns[fx_of[col]], rate_change], axis=1,
                       keys=["y", *SCENARIO_FACTORS]).dropna()
        if len(df) < 20:
            continue
        fit = ols(df["y"].to_numpy(), _design(*(df[f] for f in SCENARIO_FACTORS)))
        fit["factors"] = df[list(SCENARIO_FACTORS)]
        models[col] = fit
    return models


def scenario_predict(models: dict[str, dict], us: float, fx: float, rate: float) -> pd.DataFrame:
    """Expected return for a shock, and the 95% range of outcomes around it.

    The range combines the uncertainty in the estimated betas with the bank's own week-to-week
    noise, so it shows where a single week's return could plausibly land, not just the average.
    Shocks well outside what happened in the sample are extrapolations of a straight line.
    """
    x = np.array([1.0, us, fx, rate])
    rows = {}
    for name, m in models.items():
        expected = float(x @ m["coef"])
        spread = Z95 * np.sqrt(x @ m["cov"] @ x + m["resid_var"])
        rows[name] = {"Expected": expected, "Low": expected - spread, "High": expected + spread,
                      "S&P 500 beta": m["coef"][1], "Currency beta": m["coef"][2],
                      "Rate beta": m["coef"][3], "R squared": m["r2"]}
    return pd.DataFrame.from_dict(rows, orient="index")


# --- 5. How stable the sensitivities are ---------------------------------------------------------

def rolling_betas(returns: pd.DataFrame, market: pd.Series, rate_change: pd.Series,
                  window: int = 52) -> dict[str, pd.DataFrame]:
    """One-year rolling version of `factor_betas`: market beta, rate beta and their ± 95%."""
    out = {k: {} for k in ("Market beta", "Market beta ±", "Rate beta", "Rate beta ±")}
    for col in returns.columns:
        df = pd.concat([returns[col], market, rate_change], axis=1, keys=["y", "mkt", "rate"]).dropna()
        series = {k: {} for k in out}
        for end in range(window, len(df) + 1):
            chunk = df.iloc[end - window:end]
            fit = ols(chunk["y"].to_numpy(), _design(chunk["mkt"], chunk["rate"]))
            when = chunk.index[-1]
            series["Market beta"][when] = fit["coef"][1]
            series["Market beta ±"][when] = Z95 * fit["se"][1]
            series["Rate beta"][when] = fit["coef"][2]
            series["Rate beta ±"][when] = Z95 * fit["se"][2]
        for k, values in series.items():
            out[k][col] = pd.Series(values, dtype=float)
    return {k: pd.DataFrame(v) for k, v in out.items()}


def regime_test(returns: pd.DataFrame, flag: pd.Series) -> pd.DataFrame:
    """Annualised difference in mean weekly return when `flag` is True versus False, ± 95%."""
    flag = flag.reindex(returns.index).ffill().fillna(False).astype(bool)
    rows = {}
    for col in returns.columns:
        diff, half = mean_difference(returns.loc[~flag, col], returns.loc[flag, col])
        rows[col] = {"Difference": diff * WEEKS_PER_YEAR, "Difference ±": half * WEEKS_PER_YEAR}
    return pd.DataFrame.from_dict(rows, orient="index")
