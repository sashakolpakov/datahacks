from fredapi import Fred
import pandas as pd
from datetime import datetime, timedelta
import time
import os



def get_fred_data(api_key, start_date, end_date):
    """
    Download all available FRED series from our mapping

    Parameters:
    -----------
    api_key : str
        Your FRED API key
    start_date : str
        Start date in 'YYYY-MM-DD' format
    end_date : str
        End date in 'YYYY-MM-DD' format
    """

    # Initialize FRED API
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))

    # Define FRED series mapping
    fred_series = {
        'SP500': 'SP500',  # S&P 500
        'CHINA_GDP': 'MKTGDPCNA646NWDB',  # China GDP
        'UK_GDP': 'MKTGDPGBA646NWDB',  # UK GDP
        'US_GDP': 'GDP',  # US GDP
        'UK_UNEMPLOYMENT': 'LRUN64TTGBQ156S',  # UK Unemployment Rate
        'CHINA_POP': 'POPCN',  # China Population
        'UK_POP': 'POPGBR',  # UK Population
        'US_POP': 'POPTHM',  # US Population
        'UK_INTEREST_RATE': 'IR3TIB01GBM156N',  # UK Interest Rate
        'US_INTEREST_RATE': 'FEDFUNDS',  # US Federal Funds Rate
        'UK_3M_YIELD': 'IR3TIB01GBM156N',  # UK 3-month Yield
        'US_3M_YIELD': 'TB3MS',  # US 3-month Treasury
        'GERMANY_3M_YIELD': 'IR3TIB01DEM156N',  # German 3-month Yield
        'GERMANY_2Y_YIELD': 'IRLTLT01DEM156N',  # German 2-year Yield
        'UK_2Y_YIELD': 'IRLTLT01GBM156N',  # UK 2-year Yield
        'US_2Y_YIELD': 'DGS2',  # US 2-year Treasury
        'UK_10Y_YIELD': 'IRLTLT01GBM156N',  # UK 10-year Yield
        'US_10Y_YIELD': 'DGS10',  # US 10-year Treasury
        'GERMANY_10Y_YIELD': 'IRLTLT01DEM156N',  # German 10-year Yield
        'CHINA_M2': 'MABMM201CNM189S',  # China M2
        'UK_M2': 'MABMM201GBM189S',  # UK M2
        'US_M2': 'M2SL',  # US M2
        'UK_CPI': 'CPIUKA',  # UK CPI
        'UK_TRADE_BALANCE': 'BOPBCA_GB',  # UK Trade Balance
        'UK_CURRENT_ACCOUNT': 'BOPBCA_GB',  # UK Current Account
        'UK_FOREX_RESERVES': 'TRESEGGBM052N',  # UK Forex Reserves
        'CLI': 'OECDLOCO',  # Composite Leading Indicator
        'NFCI': 'NFCI',  # National Financial Conditions Index
        'VIX': 'VIXCLS',  # VIX
        'BRENT_OIL': 'DCOILBRENTEU',  # Brent Crude Oil
        'GOLD': 'GOLDAMGBD228NLBM',  # Gold Price
        'COPPER': 'PCOPPUSDM'  # Copper Price
    }

    # Initialize dictionary to store results
    data = {}

    # Download each series
    for name, series_id in fred_series.items():
        try:
            print(f"Downloading {name}...")
            series = fred.get_series(series_id, start_date, end_date)
            data[name] = series
            time.sleep(0.5)  # Add delay to avoid hitting rate limits
        except Exception as e:
            print(f"Error downloading {name}: {str(e)}")

    # Convert to DataFrame
    df = pd.DataFrame(data)
    return df

# Now let's check Polygon.io for market indices
def check_polygon_tickers(api_key):
    """
    Check availability of market indices in Polygon.io
    """
    from polygon import RESTClient

    # List of indices we're interested in
    indices = {
        'FTSE 100': ['^FTSE', 'FTSE.L'],
        'ASX 200': ['^AXJO', 'XJO.AX'],
        'DAX': ['^GDAXI', 'DAX.DE'],
        'Euro STOXX 50': ['^STOXX50E', 'STOXX50E.EU'],
        'CSI 300': ['000300.SS', '399300.SZ'],
        # ETFs that track these indices
        'FTSE ETF': ['FTF', 'ISF.L'],
        'ASX ETF': ['STW.AX', 'IOZ.AX'],
        'DAX ETF': ['DAX', 'EXS1.DE'],
        'Euro STOXX ETF': ['FEZ', 'ESTX50.PA'],
        'CSI 300 ETF': ['ASHR', '510300.SS']
    }

    client = RESTClient()
    available_tickers = {}

    for index_name, possible_tickers in indices.items():
        for ticker in possible_tickers:
            try:
                # Try to get details for this ticker
                details = client.get_ticker_details(ticker)
                available_tickers[index_name] = ticker
                print(f"Found {index_name}: {ticker}")
                break  # If we found a working ticker, move to next index
            except Exception as e:
                print(f"Ticker {ticker} not available: {str(e)}")
                continue

    return available_tickers

# Example usage:
if __name__ == "__main__":
    fred_api_key = "YOUR_FRED_API_KEY"
    polygon_api_key = "YOUR_POLYGON_API_KEY"

    # Set date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*5)  # 5 years of data

    # Get FRED data
    fred_data = get_fred_data(
        fred_api_key,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    # Save to CSV
    fred_data.to_csv('fred_data.csv')

    # Check Polygon.io availability
    available_tickers = check_polygon_tickers(polygon_api_key)

    # Print available tickers
    print("\nAvailable Polygon.io tickers:")
    for index, ticker in available_tickers.items():
        print(f"{index}: {ticker}")
