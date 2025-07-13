# dashboard.py

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from fredapi import Fred
import os
from dotenv import load_dotenv

# Enviroment Variables
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

if not FRED_API_KEY:
    raise ValueError("❌ FRED_API_KEY not found in environment variables.")

def run_dashboard():
    # TICKERS
    benchmark = {
        'S&P 500': '^GSPC',
        'NASDAQ': '^IXIC',
        'Hang Seng': '^HSI',
        'US 10Y Yield': '^TNX',
        'Gold': 'GC=F',
        'Oil (Brent)': 'BZ=F'
    }

    asean_banks = {
        'DBS Group': 'D05.SI',
        'OCBC': 'O39.SI',
        'Maybank': '1155.KL',
        'Bank Central Asia': 'BBCA.JK',
        'Bank Mandiri': 'BMRI.JK',
        'Bangkok Bank': 'BKKLY',
        'BPI': 'BPHLY'
    }

    vix = {'VIX': '^VIX'}

    all_assets = {**asean_banks, **benchmark, **vix}

    # Data Download
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    tickers = list(all_assets.values())

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    data = data.dropna(how='all', axis=1).ffill()
    data.rename(columns={v: k for k, v in all_assets.items()}, inplace=True)

    # FRED Data
    fred = Fred(api_key=FRED_API_KEY)
    yield_2y = fred.get_series('DGS2', observation_start=start_date, observation_end=end_date)
    yield_10y = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)

    yield_data = pd.DataFrame({'2Y': yield_2y, '10Y': yield_10y})
    yield_data['Spread'] = yield_data['10Y'] - yield_data['2Y']
    yield_data = yield_data.reindex(pd.date_range(start=start_date, end=end_date, freq='B')).ffill()

    # ASEAN Banks Normalized
    asean_names = list(asean_banks.keys())
    asean_normalized = data[asean_names].div(data[asean_names].iloc[0]).multiply(100)

    # Dashboard Plot
    plt.figure(figsize=(14, 10))

    # 1. Indices
    plt.subplot(2, 2, 1)
    for col in ['S&P 500', 'NASDAQ', 'Hang Seng']:
        plt.plot(data[col], label=col)
    plt.title('Equity Market Indices')
    plt.ylabel('Index')
    plt.grid(True)
    plt.legend()

    # 2. ASEAN Banks
    plt.subplot(2, 2, 2)
    for bank in asean_names:
        plt.plot(asean_normalized.index, asean_normalized[bank], label=bank)
    plt.title("ASEAN Bank Stocks (Normalized to 100)")
    plt.grid(True)
    plt.legend()

    # 3. Yield Curve
    plt.subplot(2, 2, 3)
    plt.plot(yield_data.index, yield_data['Spread'], label='10Y - 2Y Spread', color='purple')
    plt.axhline(0, color='red', linestyle='--', label='Inversion Threshold (0)')
    plt.title('US Treasury Yield Curve Spread')
    plt.xlabel('Date')
    plt.ylabel('Spread (%)')
    plt.grid(True)
    plt.legend()

    # 4. Risk Sentiment: VIX + Gold
    plt.subplot(2, 2, 4)
    plt.plot(data.index, data['VIX'], label='VIX', color='purple')
    plt.ylabel('VIX')

    ax2 = plt.gca().twinx()
    gold_normalized = data['Gold'].div(data['Gold'].iloc[0]).multiply(100)
    ax2.plot(gold_normalized, label='Gold', color='gold')
    ax2.set_ylabel('Gold (normalized)')

    plt.title('Risk Sentiment: VIX and Gold')
    lines_1, labels_1 = plt.gca().get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_dashboard()
