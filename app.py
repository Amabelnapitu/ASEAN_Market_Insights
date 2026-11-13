"""Interactive dashboard: streamlit run app.py"""

from datetime import date

import pandas as pd
import streamlit as st

from asean_insights import charts, metrics
from asean_insights.config import BANK_NAMES, BANKS
from asean_insights.data import MarketData, load_market_data
from asean_insights.report import analyse

st.set_page_config(page_title="ASEAN Market Insights", page_icon="📊", layout="wide")


@st.cache_data(ttl=6 * 3600, show_spinner="Downloading market data…")
def get_data() -> MarketData:
    return load_market_data(cache_dir=None)


def clip(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    return df.loc[str(start):str(end)]


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

if not chosen or (end - start).days < 90:
    st.info("Pick at least one bank and a period of three months or more.")
    st.stop()

banks = data.banks_usd if currency == "USD" else data.banks_local
view = MarketData(
    banks_local=clip(data.banks_local[chosen], start, end),
    banks_usd=clip(banks[chosen], start, end),  # analyse() reads this field
    factors=clip(data.factors, start, end),
    yields=clip(data.yields, start, end),
)
a = analyse(view)

best = a.performance.loc[chosen, "Total return"].idxmax()
deepest = a.performance.loc[chosen, "Max drawdown"].idxmin()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Best performer", best, f"{a.performance.at[best, 'Total return']:+.0%}")
k2.metric("S&P 500, same period", f"{a.performance.at['S&P 500', 'Total return']:+.0%}")
k3.metric("Deepest drawdown", deepest, f"{a.performance.at[deepest, 'Max drawdown']:.0%}")
k4.metric("US 10Y minus 2Y", f"{view.yields['Spread'].dropna().iloc[-1]:+.2f} pp")

perf_tab, risk_tab, corr_tab, macro_tab = st.tabs(["Performance", "Risk", "Correlations", "Macro"])

with perf_tab:
    st.pyplot(charts.performance_chart(a.rebased, "USD" if currency == "USD" else "local currency"))
    table = a.performance.copy()
    table.insert(0, "Country", [next((b.country for b in BANKS if b.name == n), "US") for n in table.index])
    pct = {c: "{:+.1%}" for c in ("Total return", "Annualised return")}
    st.dataframe(table.style.format({**pct, "Annualised volatility": "{:.1%}", "Return / risk": "{:.2f}",
                                     "Max drawdown": "{:.1%}"}))

with risk_tab:
    st.pyplot(charts.drawdown_chart(a.drawdowns))
    left, right = st.columns(2)
    left.subheader("Weeks when the VIX spiked")
    left.caption("Average weekly return in the 10% of weeks with the largest VIX rise.")
    left.dataframe(a.stress.style.format("{:+.2%}"))
    right.subheader("Sensitivity to US equities and rates")
    right.caption("Weekly regression on the S&P 500 return and the change in the US 10-year yield.")
    right.dataframe(a.betas.style.format({"Market beta": "{:.2f}", "Rate beta": "{:+.3f}",
                                          "R squared": "{:.2f}", "Weeks": "{:.0f}"}))

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
