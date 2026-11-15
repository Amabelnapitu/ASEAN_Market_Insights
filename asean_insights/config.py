"""What the analysis covers: banks, global factors, rates and the sample start."""

from dataclasses import dataclass

START_DATE = "2020-01-01"


@dataclass(frozen=True)
class Bank:
    name: str
    ticker: str  # Yahoo Finance symbol
    country: str
    fx: str  # Yahoo symbol for the home currency priced in USD
    adr: bool = False  # True when the ticker is a US-dollar ADR rather than the home listing


BANKS = [
    Bank("DBS Group", "D05.SI", "Singapore", "SGDUSD=X"),
    Bank("OCBC", "O39.SI", "Singapore", "SGDUSD=X"),
    Bank("Maybank", "1155.KL", "Malaysia", "MYRUSD=X"),
    Bank("Bank Central Asia", "BBCA.JK", "Indonesia", "IDRUSD=X"),
    Bank("Bank Mandiri", "BMRI.JK", "Indonesia", "IDRUSD=X"),
    Bank("Bangkok Bank", "BBL.BK", "Thailand", "THBUSD=X"),
    # Yahoo has no usable history for the Manila listing, so BPI uses its US-dollar ADR. Its
    # peso price is implied by dividing the ADR by the peso rate.
    Bank("BPI", "BPHLY", "Philippines", "PHPUSD=X", adr=True),
]

# Each country's stock market, as the US-listed iShares ETF (in USD). Yahoo has no usable
# history for the Thai and Philippine indices, so ETFs keep all five on the same basis.
# Each bank is itself a sizeable holding of its country's ETF.
HOME_MARKETS = {
    "Singapore": "EWS",
    "Malaysia": "EWM",
    "Indonesia": "EIDO",
    "Thailand": "THD",
    "Philippines": "EPHE",
}

# Global factors the banks are compared against.
FACTORS = {
    "S&P 500": "^GSPC",
    "NASDAQ": "^IXIC",
    "Hang Seng": "^HSI",
    "VIX": "^VIX",
    "Gold": "GC=F",
    "Brent": "BZ=F",
}

# US Treasury constant-maturity yields from FRED (percent).
FRED_SERIES = {"US 10Y": "DGS10", "US 2Y": "DGS2"}

BANK_NAMES = [b.name for b in BANKS]
COUNTRY_OF = {b.name: b.country for b in BANKS}
