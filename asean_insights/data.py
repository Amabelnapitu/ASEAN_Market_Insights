"""Download prices from Yahoo Finance and Treasury yields from FRED.

No API key is needed: FRED series come from its public CSV endpoint.
Downloads are cached as CSV files so repeat runs are fast and work offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from .config import BANKS, FACTORS, FRED_SERIES, START_DATE

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ids}"
DEFAULT_CACHE = Path("data")


@dataclass
class MarketData:
    banks_local: pd.DataFrame  # daily close, local currency (dividend-adjusted)
    banks_usd: pd.DataFrame  # the same, converted to US dollars
    factors: pd.DataFrame  # indices, VIX, gold, Brent
    yields: pd.DataFrame  # US 10Y, US 2Y and their spread, in percent


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> pd.DataFrame:
    """Dividend- and split-adjusted daily closes, one column per ticker."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    closes = raw["Close"]
    missing = [t for t in tickers if t not in closes or closes[t].notna().sum() == 0]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no data for: {', '.join(missing)}")
    return closes[tickers].dropna(how="all")


def fetch_fred(series: dict[str, str], start: str, end: str | None = None) -> pd.DataFrame:
    """FRED series as columns named by the keys of `series`."""
    url = FRED_CSV_URL.format(ids=",".join(series.values()))
    raw = pd.read_csv(url, index_col=0, parse_dates=True, na_values=".")
    raw = raw.rename(columns={code: name for name, code in series.items()})
    raw.index.name = "Date"
    return raw.loc[start:end].dropna(how="all")


def to_usd(local: pd.DataFrame, fx: pd.DataFrame, fx_for: dict[str, str | None]) -> pd.DataFrame:
    """Convert each price column to USD.

    `fx_for` maps a column to its FX column (local currency priced in USD), or None when the
    column is already in USD. FX is carried forward onto the price dates, because currency
    markets and local exchanges close on different holidays.
    """
    rates = fx.reindex(local.index.union(fx.index)).ffill().reindex(local.index)
    out = {}
    for col in local.columns:
        pair = fx_for.get(col)
        out[col] = local[col] if pair is None else local[col] * rates[pair]
    return pd.DataFrame(out, index=local.index)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_market_data(
    start: str = START_DATE,
    end: str | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE,
    refresh: bool = False,
) -> MarketData:
    """Everything the analysis needs. Uses today's cached download unless `refresh`."""
    cache = Path(cache_dir) / f"{start}_{end or date.today().isoformat()}" if cache_dir else None
    names = ("banks_local", "banks_usd", "factors", "yields")
    if cache and not refresh and all((cache / f"{n}.csv").exists() for n in names):
        return MarketData(*(_read(cache / f"{n}.csv") for n in names))

    fx_tickers = sorted({b.fx for b in BANKS if b.fx})
    tickers = [b.ticker for b in BANKS] + list(FACTORS.values()) + fx_tickers
    prices = fetch_prices(tickers, start, end)

    banks_local = prices[[b.ticker for b in BANKS]].rename(columns={b.ticker: b.name for b in BANKS})
    banks_local = banks_local.dropna(how="all")
    banks_usd = to_usd(banks_local, prices[fx_tickers], {b.name: b.fx for b in BANKS})
    factors = prices[list(FACTORS.values())].rename(columns={v: k for k, v in FACTORS.items()})
    factors = factors.dropna(how="all")

    yields = fetch_fred(FRED_SERIES, start, end)
    yields["Spread"] = yields["US 10Y"] - yields["US 2Y"]

    data = MarketData(banks_local, banks_usd, factors, yields)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        for n in names:
            getattr(data, n).to_csv(cache / f"{n}.csv")
    return data
