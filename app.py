"""Interactive dashboard: streamlit run app.py"""

from datetime import date

import pandas as pd
import streamlit as st
from matplotlib.ticker import PercentFormatter

from asean_insights import charts, metrics
from asean_insights.config import BANK_NAMES
from asean_insights.data import MarketData, load_fundamentals, load_market_data
from asean_insights.report import analyse, country, with_ci

st.set_page_config(page_title="ASEAN Market Insights", page_icon="📊", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading market data…")
def get_data() -> MarketData:
    return load_market_data(cache_dir=None)


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading valuation data…")
def get_fundamentals() -> pd.DataFrame:
    return load_fundamentals(cache_dir=None)


def clip(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df.loc[str(start):str(end)]


def select(data: MarketData, banks: pd.DataFrame, chosen: list[str], start: date, end: date) -> MarketData:
    """The chosen banks and period. `banks` is the series analysed (USD or local currency)."""
    return MarketData(
        banks_local=clip(data.banks_local[chosen], start, end),
        banks_usd=clip(banks[chosen], start, end),  # analyse() reads this field
        factors=clip(data.factors, start, end),
        yields=clip(data.yields, start, end),
        prices_local=clip(data.prices_local[chosen], start, end),
        dividends=clip(data.dividends[chosen], start, end),
        fx=clip(data.fx, start, end),
        home=clip(data.home, start, end),
    )


def pct(cols, fmt="{:+.1%}") -> dict:
    return dict.fromkeys(cols, fmt)


data = get_data()
first, last = data.banks_usd.index[0].date(), data.banks_usd.index[-1].date()

with st.sidebar:
    st.header("Filters")
    start, end = st.slider("Period", first, last, (first, last), format="MMM YYYY")
    chosen = st.multiselect("Banks", BANK_NAMES, default=BANK_NAMES)
    currency = st.radio("Returns in", ["USD", "Local currency"], horizontal=True)
    st.caption("Prices: Yahoo Finance, dividend-adjusted. Yields: FRED. Refreshed every 6 hours.")

st.title("ASEAN Market Insights")
st.caption("Seven ASEAN banks against global equity, rates and risk indicators.")

if not chosen or (end - start).days < 365:
    st.info("Pick at least one bank and a period of one year or more.")
    st.stop()

usd_view = select(data, data.banks_usd, chosen, start, end)
if currency == "USD":
    view, a = usd_view, analyse(usd_view)
else:
    view = select(data, data.banks_local, chosen, start, end)
    a = analyse(view, usd=usd_view)
label = "USD" if currency == "USD" else "local currency"

best = a.performance.loc[chosen, "Total return"].idxmax()
deepest = a.performance.loc[chosen, "Max drawdown"].idxmin()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best performer", best, f"{a.performance.at[best, 'Total return']:+.0%}")
k2.metric("S&P 500, same period", f"{a.performance.at['S&P 500', 'Total return']:+.0%}")
k3.metric("Deepest drawdown", deepest, f"{a.performance.at[deepest, 'Max drawdown']:.0%}")
k4.metric("US 10Y minus 2Y", f"{view.yields['Spread'].dropna().iloc[-1]:+.2f} pp")

tabs = st.tabs(["Now", "Performance", "Risk", "Drivers", "Scenarios", "Correlations", "Macro"])
now_tab, perf_tab, risk_tab, drivers_tab, scenario_tab, corr_tab, macro_tab = tabs

with now_tab:
    st.subheader(f"Where each bank stands at {a.end:%d %b %Y}")
    st.caption("Compared with its own history in the selected period. 52-week percentile: share of "
               "all one-year windows in the period with a lower return than the latest year. "
               "Beta now: S&P 500 beta over the last 52 weeks, against the period average.")
    st.dataframe(a.now.style.format({**pct(["Last 4 weeks", "Last 52 weeks"]), "Off peak": "{:.0%}",
                                     "52-week percentile": "{:.0%}", "Beta now": "{:.2f}",
                                     "Beta average": "{:.2f}"}))

    st.subheader("Valuation today")
    fundamentals = get_fundamentals()
    full_yield = metrics.trailing_dividend_yield(data.dividends, data.prices_local)
    all_banks = metrics.valuation_table(fundamentals, full_yield)
    valuation = all_banks.loc[chosen]
    valuation.attrs["fit"] = all_banks.attrs["fit"]  # the peer line always uses all seven banks
    if valuation[["P/B", "ROE"]].isna().all().all():
        st.warning("Yahoo Finance did not return valuation data just now. Try again later.")
    else:
        left, right = st.columns([3, 2])
        left.pyplot(charts.valuation_chart(valuation))
        right.markdown(
            "Banks are usually valued on **price-to-book (P/B)** against **return on equity (ROE)**: "
            "a bank that earns more on its equity deserves a higher multiple. The dashed line is "
            "what the seven banks pay on average for each level of ROE. Banks above it are priced "
            "at a premium to peers with the same profitability; banks below it at a discount.\n\n"
            "A discount is not automatically a bargain. It can be the market charging for "
            "currency risk, slower growth or weaker governance. Check the Performance tab for how "
            "much currency has cost each bank before reading a discount as cheap.")
        as_of = fundamentals.attrs.get("as_of", "today")
        right.caption(f"Snapshot from Yahoo Finance, {as_of}. Ignores the period "
                      f"slider. Peer line R squared: {valuation.attrs['fit']['r2']:.2f}, so a rough "
                      "guide only. ROE is P/B divided by P/E where Yahoo reports none. BPI's "
                      "multiples come from its US ADR, whose dividend record on Yahoo "
                      "misses some payments, so its yield is understated.")
        st.dataframe(valuation.style.format({"P/B": "{:.2f}", "P/E": "{:.1f}", "ROE": "{:.1%}",
                                             "Dividend yield": "{:.1%}", "P/B vs peers": "{:+.0%}"},
                                            na_rep="–"))

with perf_tab:
    st.pyplot(charts.performance_chart(a.rebased, label))
    table = a.performance.copy()
    table.insert(0, "Country", [country(n) for n in table.index])
    st.dataframe(table.style.format({**pct(["Total return", "Annualised return"]),
                                     "Annualised volatility": "{:.1%}", "Return / risk": "{:.2f}",
                                     "Max drawdown": "{:.1%}"}))

    st.subheader("Where a dollar investor's return came from")
    st.caption("Always in USD, whatever the currency switch says. Share price change, plus reinvested "
               "dividends, plus the currency, adds up to the total return in US dollars.")
    st.pyplot(charts.attribution_chart(a.attribution))
    attribution = a.attribution.sort_values("Total in USD", ascending=False)
    attribution.attrs = {}  # holds Timestamps, which Streamlit cannot serialise
    st.dataframe(attribution.style.format("{:+.1%}"))

with risk_tab:
    st.pyplot(charts.drawdown_chart(a.drawdowns))
    left, right = st.columns(2)
    left.subheader("Weeks when the VIX spiked")
    left.caption("Average weekly return in the 10% of weeks with the largest VIX rise.")
    left.dataframe(a.stress.style.format("{:+.2%}"))
    right.subheader("Sensitivity to US equities and rates")
    right.caption("Weekly regression on the S&P 500 return and the change in the US 10-year yield. "
                  "Brackets: 95% interval (Newey-West).")
    right.dataframe(with_ci(a.betas, {"Market beta": "{:.2f}", "Rate beta": "{:+.3f}",
                                      "R squared": "{:.2f}", "Weeks": "{:.0f}"}))
    st.pyplot(charts.interval_chart(
        a.betas, "Rate beta", "Rate beta ±",
        "Return for a 1pp rise in the US 10-year yield, S&P 500 held fixed", fmt="{:+.1%}",
        axis_fmt=PercentFormatter(1, decimals=0), colours=charts.BANK_COLOURS))

    st.subheader("Are the sensitivities stable?")
    which = st.radio("Show", ["Market beta", "Rate beta"], horizontal=True)
    if a.rolling[which].empty:
        st.info("The period is too short for a one-year rolling estimate.")
    else:
        title = "S&P 500 beta" if which == "Market beta" else "US 10-year rate beta"
        st.pyplot(charts.rolling_beta_chart(a.rolling[which], a.rolling[f"{which} ±"], a.betas[which],
                                            f"{title}, trailing 52 weeks"))

with drivers_tab:
    st.subheader("Is each bank driven by its home market or by Wall Street?")
    st.caption(f"Weekly returns in {label}. Home market: the country's iShares ETF. Each bank is "
               "itself a large holding of its country ETF, which raises the home-market numbers.")
    st.pyplot(charts.regime_chart(
        a.home[["R² US only", "R² home only"]].rename(
            columns={"R² US only": "S&P 500", "R² home only": "Home market"}),
        "Share of weekly moves explained (R squared)", fmt="{:.0%}"))
    st.pyplot(charts.interval_chart(
        a.home, "Alpha a year", "Alpha a year ±",
        "Return a year not explained by the home or US market (alpha)", colours=charts.BANK_COLOURS))
    st.dataframe(with_ci(a.home, {"R² US only": "{:.2f}", "R² home only": "{:.2f}", "R² both": "{:.2f}",
                                  "Home beta": "{:.2f}", "US beta": "{:.2f}", "Alpha a year": "{:+.1%}",
                                  "Weeks": "{:.0f}"}))

with scenario_tab:
    st.subheader("What if…")
    st.caption("Estimated one-week return in USD for each bank, from a regression on the S&P 500, the "
               "bank's home currency and the US 10-year yield over the selected period. The range "
               "is where 95% of outcomes would land, including each bank's own noise.")
    weekly_spx = metrics.weekly_returns(usd_view.factors[["S&P 500"]])["S&P 500"]
    weekly_fx = metrics.weekly_returns(usd_view.fx).mean(axis=1)
    weekly_rate = usd_view.yields["US 10Y"].ffill().resample("W-FRI").last().diff()
    picks = {
        "Worst S&P 500 week": weekly_spx.idxmin(),
        "Biggest fall in Asian currencies": weekly_fx.idxmin(),
        "Biggest jump in the US 10-year yield": weekly_rate.idxmax(),
    }
    presets = {"Custom": None} | {f"{k} ({v:%d %b %Y})": v for k, v in picks.items()}

    limits = {"spx": (-25.0, 15.0), "fx": (-10.0, 10.0), "rate": (-1.0, 1.0)}
    for key, default in (("spx", -10.0), ("fx", -2.0), ("rate", 0.0)):
        st.session_state.setdefault(key, default)

    def apply_preset() -> None:
        week = presets[st.session_state["preset"]]
        if week is None:
            return
        values = {"spx": weekly_spx.get(week, 0) * 100, "fx": weekly_fx.get(week, 0) * 100,
                  "rate": weekly_rate.get(week, 0)}
        for key, v in values.items():
            lo, hi = limits[key]
            st.session_state[key] = float(min(max(round(v, 2), lo), hi))

    st.selectbox("Start from a real week in the period, or set your own shock", list(presets),
                 key="preset", on_change=apply_preset)
    c1, c2, c3 = st.columns(3)
    spx = c1.slider("S&P 500 move (%)", *limits["spx"], step=0.1, key="spx")
    fx = c2.slider("Asian currencies against USD (%)", *limits["fx"], step=0.1, key="fx",
                   help="Applied to all five currencies. Negative = the dollar strengthens.")
    rate = c3.slider("Change in US 10-year yield (pp)", *limits["rate"], step=0.01, key="rate")
    result = metrics.scenario_predict(a.scenarios, spx / 100, fx / 100, rate)
    if result.empty:
        st.info("Not enough data in the selected period.")
    else:
        result["±"] = (result["High"] - result["Low"]) / 2
        st.pyplot(charts.interval_chart(result, "Expected", "±", "Estimated return for this scenario (USD)",
                                        fmt="{:+.1%}", colours=charts.BANK_COLOURS))
        week = presets[st.session_state.get("preset", "Custom")]
        shown = result[["Expected", "Low", "High"]].copy()
        if week is not None:
            actual = metrics.weekly_returns(usd_view.banks_usd).reindex([week]).T.iloc[:, 0]
            shown["Actual that week"] = actual
            st.caption("For a real week, the actual return is shown next to the estimate. The "
                       "estimate uses the average currency move, not each bank's own. In crash "
                       "weeks the actual loss is often outside the range: in a panic, markets move "
                       "together more than a straight line fitted on normal weeks suggests.")
        st.dataframe(shown.style.format("{:+.1%}", na_rep="–"))
        with st.expander("Model details"):
            st.dataframe(result[["S&P 500 beta", "Currency beta", "Rate beta", "R squared"]]
                         .style.format("{:.2f}"))
            st.caption("Shocks far outside what happened in the period extrapolate a straight line "
                       "and are less reliable.")

with corr_tab:
    st.pyplot(charts.correlation_heatmap(a.correlation, "Weekly return correlations"))
    rolling = metrics.weekly_returns(view.banks_usd).rolling(52).corr(
        metrics.weekly_returns(view.factors[["S&P 500"]])["S&P 500"]).dropna(how="all")
    if not rolling.empty:
        st.subheader("One-year rolling correlation with the S&P 500")
        st.line_chart(rolling, color=[charts.BANK_COLOURS[c] for c in rolling.columns])

with macro_tab:
    st.pyplot(charts.macro_chart(view.yields, view.factors))
    if a.curve.attrs["weeks"]["Curve inverted"] >= 8:
        st.pyplot(charts.regime_chart(a.curve, "Annualised return by yield-curve regime"))
        st.pyplot(charts.interval_chart(a.curve_test, "Difference", "Difference ±",
                                        "Inverted minus normal: difference in annualised return"))
