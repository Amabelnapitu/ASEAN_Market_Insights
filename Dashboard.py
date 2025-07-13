# dashboard.py

import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from fredapi import Fred  # Optional for yield curve (can be commented if unused)

def run_dashboard():

    # --- Tickers ---
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

    # --- Download Data ---
    start_date = '2020-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    tickers = list(all_assets.values())

    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    data = data.dropna(how='all', axis=1).ffill()
    data.rename(columns={v: k for k, v in all_assets.items()}, inplace=True)

    #FRED Data Extraction
    fred = Fred(api_key='dc83fd159bd76cf5fb3f1306335ff429')

    yield_2y = fred.get_series('DGS2', observation_start=start_date, observation_end=end_date)
    yield_10y = fred.get_series('DGS10', observation_start=start_date, observation_end=end_date)

    #Aligning with business days
    yield_data = pd.DataFrame({'2Y': yield_2y, '10Y': yield_10y})
    business_days = pd.date_range(start=start_date, end=end_date, freq='B')
    yield_data = yield_data.reindex(business_days)
    yield_data.ffill(inplace=True)

    #Spread
    yield_data['Spread'] = yield_data['10Y'] - yield_data['2Y']


    #Normalize ASEAN Banks Data
    asean_banks_names = list ({
        'DBS Group', 'OCBC', 'Maybank', 'Bank Central Asia', 'Bank Mandiri', 'Bangkok Bank', 'BPI'
    })
    asean_banks_names = [col for col in asean_banks_names if col in data.columns]
    asean_normalized = data[asean_banks_names].div(data[asean_banks_names].iloc[0]).multiply(100)

    #Plot Data
    plt.figure(figsize = (16,10))

    #Major Indices
    plt.subplot(2,2,1)
    for col in ['S&P 500', 'NASDAQ', 'Hang Seng']:
        plt.plot(data[col], label=col)
    plt.title('Equity Market Index')
    plt.ylabel('Index')
    plt.grid(True)
    plt.legend()

    #ASEAN Banks
    plt.subplot(2,2,2)
    for bank in asean_banks_names:
        plt.plot(asean_normalized.index, asean_normalized[bank], label=bank)
    plt.title("ASEAN Banks Prices Normalized to 100")
    plt.legend()
    plt.grid(True)

    #Yield Curve
    plt.subplot(2,2,3)
    plt.plot(yield_data.index, yield_data['Spread'], label='10Y-2Y Spread', color='purple')
    plt.axhline(0,color='red',linestyle='--',label='Inversion Threshold (0)')
    plt.title('US Treasury Yield Curve Spread (10Y-2Y)')
    plt.xlabel('Date')
    plt.ylabel('Yield Spread (%)')
    plt.legend()
    plt.grid(True)

    #Risk Sentiment Indicators:
    plt.subplot(2,2,4)
    ax1 = plt.gca()
    ax1.plot(data.index, data['VIX'], label='VIX', color='purple')
    ax1.set_ylabel('VIX')

    #Gold
    ax2 = ax1.twinx()
    gold_normalized = data['Gold'].div(data['Gold'].iloc[0]).multiply(100)
    ax2.plot(gold_normalized, label='Gold', color='gold')
    ax2.set_ylabel('Gold')
    plt.title('Risk Sentiment')
    ax1.grid(True)
    # Labels for the graph
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    plt.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')


    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_dashboard()