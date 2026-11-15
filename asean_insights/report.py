"""Run the full analysis and write the figures and a markdown summary."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib
import pandas as pd
from matplotlib.ticker import PercentFormatter

from . import charts, metrics
from .config import BANK_NAMES, BANKS, COUNTRY_OF
from .data import MarketData


@dataclass
class Analysis:
    rebased: pd.DataFrame
    performance: pd.DataFrame
    drawdowns: pd.DataFrame
    correlation: pd.DataFrame
    betas: pd.DataFrame
    curve: pd.DataFrame  # returns with the yield curve normal vs inverted
    stress: pd.DataFrame  # returns in VIX-spike weeks vs other weeks
    start: pd.Timestamp
    end: pd.Timestamp
    attribution: pd.DataFrame  # USD return split into price, dividends and currency
    now: pd.DataFrame  # where each bank stands today versus its own history
    home: pd.DataFrame  # home market versus US market regressions
    curve_test: pd.DataFrame  # inverted minus normal, with a 95% interval
    rolling: dict[str, pd.DataFrame]  # one-year rolling betas and their intervals
    scenarios: dict[str, dict]  # fitted scenario model per bank (USD returns)
    dividend_yield: pd.Series  # trailing 12-month cash yield


def analyse(data: MarketData, usd: MarketData | None = None) -> Analysis:
    """Every table and series the report and dashboard show.

    `data.banks_usd` is the return series analysed, which the dashboard may swap for local
    currency. `usd`, when given, is the same selection kept in US dollars, for the parts that
    only make sense for a dollar investor (attribution and scenarios). It defaults to `data`.
    """
    local_mode = usd is not None and usd is not data
    usd = usd or data
    names = list(data.banks_usd.columns)
    banks = data.banks_usd.ffill().dropna()
    rebased = metrics.rebase(banks)
    bank_ret = metrics.weekly_returns(banks)
    factor_ret = metrics.weekly_returns(data.factors.drop(columns="VIX"))
    weekly_yields = data.yields.ffill().resample("W-FRI").last()
    rate_change = weekly_yields["US 10Y"].diff()

    everything = pd.concat([bank_ret, factor_ret], axis=1)
    everything["VIX change"] = data.factors["VIX"].ffill().resample("W-FRI").last().pct_change(fill_method=None)
    everything["US 10Y change"] = rate_change
    with_spx = pd.concat([bank_ret, factor_ret[["S&P 500"]]], axis=1)
    inverted = weekly_yields["Spread"] < 0

    # Home markets in the same currency as the bank returns being analysed.
    home_prices = data.home / data.fx.reindex(data.home.index).ffill() if local_mode else data.home
    home_ret = metrics.weekly_returns(home_prices)

    usd_ret = metrics.weekly_returns(usd.banks_usd.ffill().dropna())
    fx_ret = metrics.weekly_returns(data.fx)

    return Analysis(
        rebased=rebased,
        performance=metrics.performance_table(pd.concat([banks, data.factors[["S&P 500"]]], axis=1)),
        drawdowns=metrics.drawdown(banks),
        correlation=everything.dropna().corr(),
        betas=metrics.factor_betas(bank_ret, factor_ret["S&P 500"], rate_change),
        curve=metrics.regime_returns(with_spx, inverted, ("Curve normal", "Curve inverted")),
        stress=metrics.stress_returns(with_spx, metrics.stress_weeks(data.factors["VIX"])),
        start=banks.index[0],
        end=banks.index[-1],
        attribution=metrics.return_attribution(
            data.prices_local[names], data.banks_local[names], usd.banks_usd[names]),
        now=metrics.current_state(pd.concat([banks, data.factors[["S&P 500"]]], axis=1),
                                  data.factors["S&P 500"]),
        home=metrics.home_vs_us(bank_ret, home_ret, factor_ret["S&P 500"], COUNTRY_OF),
        curve_test=metrics.regime_test(with_spx, inverted),
        rolling=metrics.rolling_betas(bank_ret, factor_ret["S&P 500"], rate_change),
        scenarios=metrics.scenario_model(usd_ret, factor_ret["S&P 500"], fx_ret, rate_change, COUNTRY_OF),
        dividend_yield=metrics.trailing_dividend_yield(data.dividends[names], data.prices_local[names]),
    )


def country(name: str) -> str:
    return next((b.country for b in BANKS if b.name == name), "US")


def _table(df: pd.DataFrame, formats: dict[str, str], index_name: str) -> str:
    head = "| " + " | ".join([index_name, *df.columns]) + " |"
    rule = "|---|" + "---:|" * len(df.columns)
    body = [
        "| " + " | ".join([str(idx), *(formats.get(c, "{}").format(row[c]) for c in df.columns)]) + " |"
        for idx, row in df.iterrows()
    ]
    return "\n".join([head, rule, *body])


def with_ci(df: pd.DataFrame, fmt: dict[str, str]) -> pd.DataFrame:
    """Merge each "X" and "X ±" column pair into one text column "X (95% interval)"."""
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col.endswith(" ±"):
            continue
        f = fmt.get(col, "{}")
        if f"{col} ±" in df:
            lo, hi = df[col] - df[f"{col} ±"], df[col] + df[f"{col} ±"]
            out[col] = [f"{f.format(v)} ({f.format(a)} to {f.format(b)})" for v, a, b in zip(df[col], lo, hi)]
        else:
            out[col] = [f.format(v) for v in df[col]]
    return out


def summary_markdown(a: Analysis, valuation: pd.DataFrame | None = None) -> str:
    pct, num = "{:+.1%}", "{:.2f}"
    perf = a.performance.sort_values("Total return", ascending=False)
    bank_corr = a.correlation.loc[BANK_NAMES, BANK_NAMES]
    pairs = bank_corr.where(~pd.DataFrame(
        [[i >= j for j in range(len(BANK_NAMES))] for i in range(len(BANK_NAMES))],
        index=BANK_NAMES, columns=BANK_NAMES)).stack()
    weeks = a.curve.attrs["weeks"]
    attr = a.attribution.sort_values("Total in USD", ascending=False)
    curve = pd.concat([a.curve, a.curve_test], axis=1)
    parts = [
        "# ASEAN Market Insights: results",
        (f"Sample: {a.start:%d %b %Y} to {a.end:%d %b %Y}. Bank returns are in US dollars and "
        "include dividends. Ranges in brackets are 95% intervals using Newey-West standard errors. "
        "Generated by `python -m asean_insights`; do not edit by hand."),
        "## Performance",
        _table(perf, {"Total return": pct, "Annualised return": pct, "Annualised volatility": "{:.1%}",
                      "Return / risk": num, "Max drawdown": "{:.1%}"}, "Asset"),
        "## Where the return came from",
        ("Total return in US dollars split into the change in the local share price, reinvested "
         "dividends, and the currency. The three add up to the total. For BPI the peso price is "
         "implied from the US-dollar ADR."),
        _table(attr, {c: pct for c in attr.columns}, "Bank"),
    ]
    if valuation is not None and valuation[["P/B", "ROE"]].notna().any().any():
        fit = valuation.attrs.get("fit", {})
        parts += [
            "## Valuation today",
            (f"Snapshot from Yahoo Finance on {valuation.attrs.get('as_of', 'the run date')}. Dividend "
             "yield is cash paid in the last 12 months over today's price (may include special "
             "dividends). ROE is reported where Yahoo has it, otherwise P/B divided by P/E. "
             "\"P/B vs peers\" compares each bank's P/B with a straight line of P/B on ROE across the "
             f"seven banks (R squared {fit.get('r2', float('nan')):.2f}): +30% means paying 30% more per "
             "unit of book than the line implies for that ROE. BPI's multiples come from its ADR, "
             "and Yahoo's ADR dividend record misses some payments, so its yield is understated."),
            _table(valuation.drop(columns="ROE source"),
                   {"P/B": "{:.2f}", "P/E": "{:.1f}", "ROE": "{:.1%}", "Dividend yield": "{:.1%}",
                    "P/B vs peers": "{:+.0%}"}, "Bank"),
        ]
    parts += [
        "## Where each bank stands now",
        ("Off peak: fall from the highest price in the sample. 52-week percentile: share of all "
         "rolling one-year returns in the sample that were lower than the latest one. Beta now: "
         "S&P 500 beta over the last 52 weeks; Beta average: over the whole sample."),
        _table(a.now, {"Off peak": "{:.0%}", "Last 4 weeks": pct, "Last 52 weeks": pct,
                       "52-week percentile": "{:.0%}", "Beta now": num, "Beta average": num}, "Asset"),
        "## Home market versus the US market",
        ("Weekly regressions of each bank on its country's iShares ETF (in USD) and the S&P 500. "
         "R squared is the share of weekly moves explained. Alpha is the average return a year "
         "not explained by either market. Each bank is itself a large holding of its country ETF, "
         "which raises the home-market R squared."),
        _table(with_ci(a.home.drop(columns="Weeks"),
                       {"R² US only": "{:.2f}", "R² home only": "{:.2f}", "R² both": "{:.2f}",
                        "Home beta": num, "US beta": num, "Alpha a year": "{:+.0%}"}), {}, "Bank"),
        "## Sensitivity to the US market and US rates",
        ("Weekly regression: bank return = alpha + market beta × S&P 500 return + rate beta × change "
        "in the US 10-year yield (percentage points)."),
        _table(with_ci(a.betas, {"Market beta": num, "Rate beta": "{:+.3f}", "R squared": num,
                                 "Weeks": "{:.0f}"}), {}, "Bank"),
        "## Yield-curve regimes",
        (f"Annualised average weekly return while the US 10Y-2Y spread was positive "
        f"({weeks['Curve normal']} weeks) and negative ({weeks['Curve inverted']} weeks), and the "
        "difference (inverted minus normal) with its 95% interval."),
        _table(with_ci(curve, {c: pct for c in curve.columns}), {}, "Asset"),
        "## Risk-off weeks",
        "Average weekly return in the 10% of weeks with the largest VIX rise, and in all other weeks.",
        _table(a.stress, {c: "{:+.2%}" for c in a.stress.columns}, "Asset"),
        "## Correlations (weekly returns)",
        (f"Average pairwise correlation between the banks: {pairs.mean():.2f}. "
        f"Most correlated pair: {' / '.join(pairs.idxmax())} ({pairs.max():.2f}). "
        f"Least: {' / '.join(pairs.idxmin())} ({pairs.min():.2f})."),
        _table(a.correlation.loc[BANK_NAMES, [c for c in a.correlation.columns if c not in BANK_NAMES]],
               {c: num for c in a.correlation.columns}, "Bank"),
    ]
    return "\n\n".join(parts) + "\n"


def write_report(data: MarketData, out_dir: Path | str = "reports",
                 fundamentals: pd.DataFrame | None = None) -> Path:
    """Save every figure as PNG and the tables as summary.md. Returns the summary path."""
    matplotlib.use("Agg")
    out = Path(out_dir)
    figures = out / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    a = analyse(data)
    valuation = None
    if fundamentals is not None:
        valuation = metrics.valuation_table(fundamentals, a.dividend_yield)
        valuation.attrs["as_of"] = fundamentals.attrs.get("as_of")
    made = {
        "performance": charts.performance_chart(a.rebased),
        "attribution": charts.attribution_chart(a.attribution),
        "drawdowns": charts.drawdown_chart(a.drawdowns),
        "correlations": charts.correlation_heatmap(a.correlation, "Weekly return correlations"),
        "macro": charts.macro_chart(data.yields, data.factors),
        "yield_curve_regimes": charts.regime_chart(
            a.curve, "Annualised return by yield-curve regime (USD)"),
        "yield_curve_difference": charts.interval_chart(
            a.curve_test, "Difference", "Difference ±",
            "Inverted minus normal yield curve: difference in annualised return"),
        "home_vs_us": charts.regime_chart(
            a.home[["R² US only", "R² home only"]].rename(
                columns={"R² US only": "S&P 500", "R² home only": "Home market"}),
            "Share of weekly moves explained (R squared): US market vs home market", fmt="{:.0%}"),
        "alpha": charts.interval_chart(
            a.home, "Alpha a year", "Alpha a year ±",
            "Return a year not explained by the home or US market (alpha)", colours=charts.BANK_COLOURS),
        "rate_betas": charts.interval_chart(
            a.betas, "Rate beta", "Rate beta ±",
            "Return for a 1pp rise in the US 10-year yield, S&P 500 held fixed", fmt="{:+.1%}",
            axis_fmt=PercentFormatter(1, decimals=0), colours=charts.BANK_COLOURS),
        "rolling_betas": charts.rolling_beta_chart(
            a.rolling["Market beta"], a.rolling["Market beta ±"], a.betas["Market beta"],
            "S&P 500 beta, trailing 52 weeks"),
    }
    if valuation is not None:
        made["valuation"] = charts.valuation_chart(valuation)
    for name, fig in made.items():
        fig.savefig(figures / f"{name}.png", dpi=160)
    summary = out / "summary.md"
    summary.write_text(summary_markdown(a, valuation), encoding="utf-8")
    return summary
