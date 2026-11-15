"""Download prices from Yahoo Finance and Treasury yields from FRED.

No API key is needed: FRED series come from its public CSV endpoint.
Downloads are cached as CSV files so repeat runs are fast and work offline.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import BANKS, FACTORS, FRED_SERIES, HOME_MARKETS, START_DATE

FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ids}"
DEFAULT_CACHE = Path("data")


@dataclass
class MarketData:
    banks_local: pd.DataFrame  # daily close, local currency (dividend-adjusted)
    banks_usd: pd.DataFrame  # the same, converted to US dollars
    factors: pd.DataFrame  # indices, VIX, gold, Brent
    yields: pd.DataFrame  # US 10Y, US 2Y and their spread, in percent
    prices_local: pd.DataFrame  # daily close, local currency, NOT dividend-adjusted
    dividends: pd.DataFrame  # cash dividend per share on its ex-date, local currency (0 otherwise)
    fx: pd.DataFrame  # home currency priced in USD, one column per country
    home: pd.DataFrame  # each country's stock market ETF, in USD, one column per country


@dataclass
class Prices:
    adjusted: pd.DataFrame  # dividend- and split-adjusted closes
    close: pd.DataFrame  # split-adjusted closes, dividends not reinvested
    dividends: pd.DataFrame  # cash dividends on ex-dates


def fetch_prices(tickers: list[str], start: str, end: str | None = None) -> Prices:
    """Daily closes with and without dividends, plus the dividends, one column per ticker."""
    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=False, actions=True, progress=False)
    adjusted = raw["Adj Close"]
    missing = [t for t in tickers if t not in adjusted or adjusted[t].notna().sum() == 0]
    if missing:
        raise RuntimeError(f"Yahoo Finance returned no data for: {', '.join(missing)}")
    keep = adjusted[tickers].dropna(how="all").index
    dividends = raw["Dividends"].reindex(columns=tickers).fillna(0.0) if "Dividends" in raw else None
    return Prices(
        adjusted=adjusted[tickers].loc[keep],
        close=raw["Close"][tickers].loc[keep],
        dividends=(dividends if dividends is not None else adjusted[tickers] * 0.0).loc[keep],
    )


def fetch_fundamentals() -> pd.DataFrame:
    """Today's price-to-book, price-to-earnings and return on equity from Yahoo, per bank.

    Snapshot only: Yahoo gives no history for these. A bank whose data cannot be fetched gets
    a row of NaN rather than an error, because the rest of the analysis does not need it.
    """
    import yfinance as yf

    rows = {}
    for b in BANKS:
        try:
            info = yf.Ticker(b.ticker).info
        except Exception:  # noqa: BLE001 - network errors, rate limits, schema changes
            info = {}
        rows[b.name] = {
            "P/B": info.get("priceToBook"),
            "P/E": info.get("trailingPE"),
            "Reported ROE": info.get("returnOnEquity"),
        }
    return pd.DataFrame.from_dict(rows, orient="index").astype(float)


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


def from_usd(usd: pd.DataFrame, fx: pd.DataFrame, fx_for: dict[str, str]) -> pd.DataFrame:
    """The reverse of `to_usd`: divide each listed column by its FX rate (others unchanged)."""
    inverse = 1 / fx
    rates = inverse.reindex(usd.index.union(fx.index)).ffill().reindex(usd.index)
    out = usd.copy()
    for col, pair in fx_for.items():
        out[col] = usd[col] * rates[pair]
    return out


def _today() -> str:
    """Today's local date as YYYY-MM-DD; names the daily cache folder."""
    return datetime.now().astimezone().date().isoformat()


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, index_col=0, parse_dates=True)


def load_market_data(
    start: str = START_DATE,
    end: str | None = None,
    cache_dir: Path | str | None = DEFAULT_CACHE,
    refresh: bool = False,
) -> MarketData:
    """Everything the analysis needs. Uses today's cached download unless `refresh`."""
    cache = Path(cache_dir) / f"{start}_{end or _today()}" if cache_dir else None
    names = ("banks_local", "banks_usd", "factors", "yields", "prices_local", "dividends", "fx", "home")
    if cache and not refresh and all((cache / f"{n}.csv").exists() for n in names):
        return MarketData(*(_read(cache / f"{n}.csv") for n in names))

    fx_tickers = sorted({b.fx for b in BANKS})
    bank_tickers = [b.ticker for b in BANKS]
    home_tickers = list(HOME_MARKETS.values())
    tickers = bank_tickers + list(FACTORS.values()) + home_tickers + fx_tickers
    p = fetch_prices(tickers, start, end)

    as_names = {b.ticker: b.name for b in BANKS}
    fx_raw = p.adjusted[fx_tickers]
    usd_listed = {b.name: b.fx for b in BANKS if b.adr}  # ADRs: quoted in USD, local is implied
    local_listed = {b.name: (None if b.adr else b.fx) for b in BANKS}

    def local_and_usd(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        quoted = frame[bank_tickers].rename(columns=as_names).dropna(how="all")
        return from_usd(quoted, fx_raw, usd_listed), to_usd(quoted, fx_raw, local_listed)

    banks_local, banks_usd = local_and_usd(p.adjusted)
    prices_local, _ = local_and_usd(p.close)
    dividends = p.dividends[bank_tickers].rename(columns=as_names).loc[banks_local.index]
    dividends = from_usd(dividends, fx_raw, usd_listed).fillna(0.0)

    factors = p.adjusted[list(FACTORS.values())].rename(columns={v: k for k, v in FACTORS.items()})
    factors = factors.dropna(how="all")
    fx = fx_raw.rename(columns={b.fx: b.country for b in BANKS}).dropna(how="all")
    home = p.adjusted[home_tickers].rename(columns={v: k for k, v in HOME_MARKETS.items()})
    home = home.dropna(how="all")

    yields = fetch_fred(FRED_SERIES, start, end)
    yields["Spread"] = yields["US 10Y"] - yields["US 2Y"]

    data = MarketData(banks_local, banks_usd, factors, yields, prices_local, dividends, fx, home)
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
        for n in names:
            getattr(data, n).to_csv(cache / f"{n}.csv")
    return data


def load_fundamentals(cache_dir: Path | str | None = DEFAULT_CACHE, refresh: bool = False) -> pd.DataFrame:
    """Today's valuation snapshot (see `fetch_fundamentals`), cached for the day."""
    path = Path(cache_dir) / f"fundamentals_{_today()}.csv" if cache_dir else None
    if path and path.exists() and not refresh:
        table = pd.read_csv(path, index_col=0)
    else:
        table = fetch_fundamentals()
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)
            table.to_csv(path)
    table.attrs["as_of"] = _today()
    return table
