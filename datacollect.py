import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime, timedelta
import time
import io
import zipfile
from dotenv import load_dotenv

# Load environment variables (API keys)
load_dotenv()

try:
    from fredapi import Fred
except ImportError:
    print("Warning: fredapi not installed. FRED data will not be available.")
    print("Install with: pip install fredapi")
    Fred = None

try:
    from polygon import RESTClient
except ImportError:
    print("Warning: polygon-api-client not installed. Polygon.io data will not be available.")
    print("Install with: pip install polygon-api-client")
    RESTClient = None

# Configuration
class Config:
    # Primary currency pair for analysis
    PRIMARY_PAIR = "GBPAUD"  # Format: BASE/QUOTE without the slash
    
    # Major currencies to collect data for
    MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
    
    # Map currencies to their countries
    CURRENCY_COUNTRY_MAP = {
        "USD": "United States",
        "EUR": "Euro Area",
        "GBP": "United Kingdom",
        "JPY": "Japan",
        "CHF": "Switzerland",
        "AUD": "Australia",
        "NZD": "New Zealand",
        "CAD": "Canada"
    }
    
    # Country codes for various APIs
    COUNTRY_CODES = {
        "United States": {"FRED": "US", "WB": "USA", "BIS": "US"},
        "Euro Area": {"FRED": "XM", "WB": "EMU", "BIS": "XM"},
        "United Kingdom": {"FRED": "UK", "WB": "GBR", "BIS": "GB"},
        "Japan": {"FRED": "JP", "WB": "JPN", "BIS": "JP"},
        "Switzerland": {"FRED": "CH", "WB": "CHE", "BIS": "CH"},
        "Australia": {"FRED": "AU", "WB": "AUS", "BIS": "AU"},
        "New Zealand": {"FRED": "NZ", "WB": "NZL", "BIS": "NZ"},
        "Canada": {"FRED": "CA", "WB": "CAN", "BIS": "CA"},
        "China": {"FRED": "CN", "WB": "CHN", "BIS": "CN"}
    }
    
    # List of stock indices to collect
    STOCK_INDICES = {
        "United States": ["S&P 500", "Dow Jones", "NASDAQ"],
        "Euro Area": ["Euro STOXX 50", "CAC 40"],
        "United Kingdom": ["FTSE 100"],
        "Japan": ["Nikkei 225"],
        "Switzerland": ["SMI"],
        "Australia": ["ASX 200"],
        "New Zealand": ["NZX 50"],
        "Canada": ["S&P/TSX"],
        "China": ["CSI 300", "Shanghai Composite"]
    }
    
    # Commodities to collect
    COMMODITIES = ["Copper", "Gold", "Silver", "Crude Oil", "Brent", "Natural Gas", 
                  "Iron Ore", "Coal"]
    
    # Economic indicators to collect for each country
    ECONOMIC_INDICATORS = [
        "GDP", 
        "Unemployment", 
        "CPI", 
        "Interest rates", 
        "Budget Balance", 
        "Trade Balance", 
        "Current Account",
        "Population",
        "Industrial Production"
    ]
    
    # Bond yields to collect for each country
    BOND_YIELDS = [
        "3-month", 
        "2-year", 
        "10-year"
    ]
    
    # Special indicators
    SPECIAL_INDICATORS = [
        "CLI",  # Composite Leading Indicator
        "NFCI",  # National Financial Conditions Index
        "VIX"   # Volatility Index
    ]
    
    # Date range
    START_DATE = datetime.now() - timedelta(days=365*10)  # 10 years ago
    END_DATE = datetime.now()
    DATE_FORMAT = "%Y-%m-%d"

# Initialize API clients based on provided keys
def initialize_clients():
    """Initialize API clients using environment variables"""
    clients = {}
    
    # FRED API
    fred_api_key = os.getenv("FRED_API_KEY")
    if fred_api_key and Fred:
        clients['fred'] = Fred(api_key=fred_api_key)
        print("FRED API client initialized")
    else:
        clients['fred'] = None
        print("FRED API client not initialized (missing key or package)")
    
    # Polygon.io API
    polygon_api_key = os.getenv("POLYGON_API_KEY")
    if polygon_api_key and RESTClient:
        clients['polygon'] = RESTClient(api_key=polygon_api_key)
        print("Polygon.io API client initialized")
    else:
        clients['polygon'] = None
        print("Polygon.io API client not initialized (missing key or package)")
    
    return clients

# Data Collection Functions

def get_fred_series(client, series_id, column_name, start_date, end_date):
    """Get a time series from FRED"""
    if client is None:
        print(f"Cannot retrieve {column_name} - FRED client not initialized")
        return None
    
    try:
        series = client.get_series(series_id, start_date, end_date)
        print(f"Successfully retrieved {column_name} from FRED")
        return series
    except Exception as e:
        print(f"Error retrieving {column_name} from FRED: {str(e)}")
        return None

def get_polygon_forex(client, from_currency, to_currency, column_name, start_date, end_date):
    """Get forex data from Polygon.io"""
    if client is None:
        print(f"Cannot retrieve {column_name} - Polygon client not initialized")
        return None
    
    try:
        ticker = f"C:{from_currency}{to_currency}"
        resp = client.get_aggs(
            ticker=ticker,
            multiplier=1,
            timespan="day",
            from_=start_date,
            to=end_date
        )
        
        # Convert to pandas series
        data = {datetime.fromtimestamp(item.timestamp/1000).date(): item.close 
               for item in resp}
        series = pd.Series(data)
        series.name = column_name
        print(f"Successfully retrieved {column_name} from Polygon.io")
        return series
    except Exception as e:
        print(f"Error retrieving {column_name} from Polygon.io: {str(e)}")
        return None

def get_bis_reer(country_code, column_name, start_date, end_date):
    """Get BIS REER data using their SDMX API"""
    try:
        # Format dates for BIS API
        start_year = start_date.year
        end_year = end_date.year
        
        # BIS REER data URL
        url = f"https://stats.bis.org/api/v1/data/WS_REE_M/M.R.B.{country_code}.A?startPeriod={start_year}&endPeriod={end_year}&format=csv"
        response = requests.get(url)
        
        if response.status_code == 200:
            try:
                # Parse CSV
                df = pd.read_csv(io.StringIO(response.text))
                
                # Process data if available
                if not df.empty:
                    # Assuming the CSV has columns TIME_PERIOD and OBS_VALUE
                    # Adjust column names based on actual response
                    series = pd.Series(
                        df['OBS_VALUE'].values,
                        index=pd.to_datetime(df['TIME_PERIOD'])
                    )
                    series.name = column_name
                    print(f"Successfully retrieved {column_name} from BIS")
                    return series
                else:
                    print(f"Empty data returned for {column_name} from BIS")
            except Exception as e:
                print(f"Error parsing BIS data for {column_name}: {str(e)}")
                return None
        else:
            print(f"Failed to get BIS data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error retrieving {column_name} from BIS: {str(e)}")
        return None

def get_cot_data(currency, column_name_long, column_name_short):
    """Get CFTC COT data"""
    try:
        # CFTC reports URL
        url = "https://www.cftc.gov/files/dea/history/dea_fut_fin_txt_"
        
        # Get recent year data (simplistic approach - would need to be more robust)
        current_year = datetime.now().year
        
        # Try to get current year's data
        response = requests.get(f"{url}{current_year}.zip")
        
        if response.status_code != 200:
            # If current year not available, try previous year
            response = requests.get(f"{url}{current_year-1}.zip")
        
        if response.status_code == 200:
            # Extract data from zip file
            z = zipfile.ZipFile(io.BytesIO(response.content))
            
            # Process the COT data
            long_data = {}
            short_data = {}
            
            # Read each file in the zip
            for filename in z.namelist():
                with z.open(filename) as f:
                    lines = f.read().decode('utf-8').splitlines()
                    
                    for line in lines:
                        # Parse the fixed width format (simplified)
                        # This is a placeholder - actual format needs to be implemented
                        fields = line.split(',')
                        
                        # Check if this is the currency we're looking for
                        # The actual check would depend on how currencies are identified in the report
                        if len(fields) > 5 and currency in fields[3]:
                            # Extract date, long and short positions
                            # Actual indices would need to be adjusted
                            date_str = fields[0]
                            long_pos = float(fields[10]) if fields[10].strip() else 0
                            short_pos = float(fields[11]) if fields[11].strip() else 0
                            
                            # Convert date to datetime
                            try:
                                date = datetime.strptime(date_str, '%Y-%m-%d')
                                long_data[date] = long_pos
                                short_data[date] = short_pos
                            except ValueError:
                                pass
            
            # Convert to Series
            if long_data:
                long_series = pd.Series(long_data)
                long_series.name = column_name_long
                short_series = pd.Series(short_data)
                short_series.name = column_name_short
                
                print(f"Successfully retrieved COT data for {currency}")
                return long_series, short_series
            else:
                print(f"No COT data found for {currency}")
        else:
            print(f"Failed to get COT data: {response.status_code}")
            return None, None
    except Exception as e:
        print(f"Error retrieving COT data for {currency}: {str(e)}")
        return None, None

def get_worldbank_data(indicator_id, column_name, country_code, start_date, end_date):
    """Get data from World Bank API"""
    try:
        # World Bank API URL
        url = f"https://api.worldbank.org/v2/countries/{country_code}/indicators/{indicator_id}?date={start_date.year}:{end_date.year}&format=json&per_page=1000"
        
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # Extract data from JSON response
            if len(data) > 1 and data[1]:  # Check if there's actual data
                records = {f"{item['date']}-01-01": item['value'] for item in data[1] if item['value'] is not None}
                series = pd.Series(records)
                series.index = pd.to_datetime(series.index)
                series.name = column_name
                print(f"Successfully retrieved {column_name} from World Bank")
                return series
            else:
                print(f"No World Bank data found for {indicator_id} in {country_code}")
                return None
        else:
            print(f"Failed to get World Bank data: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error retrieving {column_name} from World Bank: {str(e)}")
        return None

def get_commodity_data(fred_client, commodity, column_name, start_date, end_date):
    """Get commodity price data from FRED or other sources"""
    try:
        # Map commodities to FRED series IDs
        fred_commodity_map = {
            'Copper': 'PCOPPUSDM',        # Global price of Copper
            'Gold': 'GOLDAMGBD228NLBM',   # Gold Fixing Price in London Bullion Market
            'Silver': 'SLVPRUSD',         # Silver Prices
            'Crude Oil': 'DCOILWTICO',    # Crude Oil Prices: WTI
            'Brent': 'DCOILBRENTEU',      # Crude Oil Prices: Brent
            'Natural Gas': 'DNGASEUUSDM', # Natural Gas
            'Iron Ore': 'PIORECRUSDM',    # Global price of Iron Ore
            'Coal': 'PCOALAUUSDM'         # Global price of Coal, Australia
        }
        
        if commodity in fred_commodity_map and fred_client is not None:
            series_id = fred_commodity_map[commodity]
            series = fred_client.get_series(series_id, start_date.strftime(Config.DATE_FORMAT), 
                                          end_date.strftime(Config.DATE_FORMAT))
            print(f"Successfully retrieved {column_name} from FRED")
            return series
        else:
            # Could implement alternative sources here
            print(f"No FRED mapping for {commodity} or FRED client not initialized")
            return None
    except Exception as e:
        print(f"Error retrieving {column_name}: {str(e)}")
        return None

def get_stock_index(fred_client, index_name, column_name, start_date, end_date):
    """Get stock market index data from FRED or other sources"""
    try:
        # Map indices to FRED series IDs
        fred_index_map = {
            'S&P 500': 'SP500',         # S&P 500
            'Dow Jones': 'DJIA',        # Dow Jones Industrial Average
            'NASDAQ': 'NASDAQCOM',      # NASDAQ Composite Index
            'FTSE 100': 'FTSE100',      # FTSE 100
            'Nikkei 225': 'NIKKEI225',  # Nikkei 225
            'DAX': 'DAXIDEV',           # DAX Index
            'VIX': 'VIXCLS'             # CBOE Volatility Index
            # Note: Many international indices may not be directly available in FRED
        }
        
        if index_name in fred_index_map and fred_client is not None:
            series_id = fred_index_map[index_name]
            series = fred_client.get_series(series_id, start_date.strftime(Config.DATE_FORMAT), 
                                          end_date.strftime(Config.DATE_FORMAT))
            print(f"Successfully retrieved {column_name} from FRED")
            return series
        else:
            # Could implement alternative sources like Yahoo Finance here
            print(f"No FRED mapping for {index_name} or FRED client not initialized")
            return None
    except Exception as e:
        print(f"Error retrieving {column_name}: {str(e)}")
        return None

def get_economic_indicator(fred_client, country, indicator, column_name, start_date, end_date):
    """Get economic indicator data from FRED or World Bank"""
    try:
        # Map countries and indicators to FRED series IDs
        indicator_map = {
            # United States
            ('United States', 'GDP'): 'GDP',                     # US Real GDP
            ('United States', 'Unemployment'): 'UNRATE',         # US Unemployment Rate
            ('United States', 'CPI'): 'CPIAUCSL',                # US Consumer Price Index
            ('United States', 'Interest rates'): 'FEDFUNDS',     # US Federal Funds Rate
            ('United States', 'Budget Balance'): 'FYFSD',        # US Federal Surplus or Deficit
            ('United States', 'Trade Balance'): 'BOPGSTB',       # US Trade Balance
            ('United States', 'Current Account'): 'BOPBCA',      # US Current Account Balance
            ('United States', 'Population'): 'POPTHM',           # US Population
            ('United States', 'Industrial Production'): 'INDPRO', # US Industrial Production
            
            # United Kingdom
            ('United Kingdom', 'GDP'): 'UKNGDP',                 # UK GDP
            ('United Kingdom', 'Unemployment'): 'LRUNTTTTGBQ156S', # UK Unemployment Rate
            ('United Kingdom', 'CPI'): 'GBRCPIALLMINMEI',        # UK CPI
            ('United Kingdom', 'Interest rates'): 'BOEBANKRATE', # UK Bank Rate
            ('United Kingdom', 'Budget Balance'): 'GBRNGDP',     # UK Government Net Borrowing
            
            # Euro Area
            ('Euro Area', 'GDP'): 'EUNNGDP',                     # Euro Area GDP
            ('Euro Area', 'Unemployment'): 'LRHUTTTTEZM156S',    # Euro Area Unemployment Rate
            ('Euro Area', 'CPI'): 'CP0000EZ19M086NEST',          # Euro Area CPI
            ('Euro Area', 'Interest rates'): 'ECBDFR',           # ECB Deposit Facility Rate
            
            # Australia
            ('Australia', 'GDP'): 'AUSGDPRQPSMEI',               # Australia GDP
            ('Australia', 'Unemployment'): 'LRUNTTTTAUQ156S',    # Australia Unemployment Rate
            ('Australia', 'CPI'): 'AUSCPIALLQINMEI',             # Australia CPI
            ('Australia', 'Interest rates'): 'IRSTCI01AUM156N',  # Australia Interest Rate
            
            # Add more as needed for other countries and indicators
        }
        
        key = (country, indicator)
        if key in indicator_map and fred_client is not None:
            series_id = indicator_map[key]
            series = fred_client.get_series(series_id, start_date.strftime(Config.DATE_FORMAT), 
                                          end_date.strftime(Config.DATE_FORMAT))
            print(f"Successfully retrieved {column_name} from FRED")
            return series
        else:
            # If not in FRED, try World Bank for some indicators
            print(f"No FRED mapping for {country} {indicator} or FRED client not initialized")
            
            # For some indicators, try World Bank as fallback
            wb_indicator_map = {
                'GDP': 'NY.GDP.MKTP.CD',               # GDP (current US$)
                'Unemployment': 'SL.UEM.TOTL.ZS',       # Unemployment rate
                'Population': 'SP.POP.TOTL',            # Total population
                'CPI': 'FP.CPI.TOTL.ZG',                # Inflation, consumer prices
                'Current Account': 'BN.CAB.XOKA.CD',    # Current account balance
                'Trade Balance': 'NE.RSB.GNFS.CD',      # External balance on goods and services
                'Budget Balance': 'GC.BAL.CASH.GD.ZS',  # Cash surplus/deficit (% of GDP)
                'Industrial Production': 'NV.IND.TOTL.ZS' # Industry, value added (% of GDP)
            }
            
            if indicator in wb_indicator_map:
                country_code = Config.COUNTRY_CODES.get(country, {}).get('WB')
                if country_code:
                    wb_id = wb_indicator_map[indicator]
                    return get_worldbank_data(wb_id, column_name, country_code, start_date, end_date)
            
            return None
    except Exception as e:
        print(f"Error retrieving {column_name}: {str(e)}")
        return None

def get_bond_yield(fred_client, country, term, column_name, start_date, end_date):
    """Get bond yield data from FRED"""
    try:
        # Map countries and terms to FRED series IDs
        bond_map = {
            # United States
            ('United States', '3-month'): 'TB3MS',       # 3-Month Treasury Bill
            ('United States', '2-year'): 'DGS2',         # 2-Year Treasury Constant Maturity
            ('United States', '10-year'): 'DGS10',       # 10-Year Treasury Constant Maturity
            
            # United Kingdom
            ('United Kingdom', '3-month'): 'INTGSTGBM193N', # UK 3-Month Treasury Bill
            ('United Kingdom', '2-year'): 'IRLTLT01GBM156N', # UK 2-Year Government Bond
            ('United Kingdom', '10-year'): 'IRLTLT01GBM156N', # UK 10-Year Government Bond
            
            # Add more as needed
        }
        
        key = (country, term)
        if key in bond_map and fred_client is not None:
            series_id = bond_map[key]
            series = fred_client.get_series(series_id, start_date.strftime(Config.DATE_FORMAT), 
                                          end_date.strftime(Config.DATE_FORMAT))
            print(f"Successfully retrieved {column_name} from FRED")
            return series
        else:
            print(f"No FRED mapping for {country} {term} or FRED client not initialized")
            return None
    except Exception as e:
        print(f"Error retrieving {column_name}: {str(e)}")
        return None

def get_special_indicators(fred_client, start_date, end_date):
    """Get CLI, NFCI, and VIX data from FRED"""
    try:
        indicators = {}
        
        # Series IDs for special indicators
        special_series = {
            'CLI': 'USALOLITONOSTSAM',   # Composite Leading Indicator for US
            'NFCI': 'NFCI',              # National Financial Conditions Index
            'VIX': 'VIXCLS'              # CBOE Volatility Index
        }
        
        if fred_client is not None:
            for name, series_id in special_series.items():
                series = fred_client.get_series(series_id, start_date.strftime(Config.DATE_FORMAT), 
                                              end_date.strftime(Config.DATE_FORMAT))
                series.name = name
                indicators[name] = series
                print(f"Successfully retrieved {name} from FRED")
        else:
            print("FRED client not initialized")
        
        return indicators
    except Exception as e:
        print(f"Error retrieving special indicators: {str(e)}")
        return {}

def add_series_to_df(df, series, column_name):
    """Safely add a series to the dataframe"""
    if series is None or series.empty:
        print(f"Skipping empty series for {column_name}")
        return df
    
    if isinstance(series, pd.Series):
        # Convert to dataframe with date as index
        if not isinstance(series.index, pd.DatetimeIndex):
            try:
                series.index = pd.to_datetime(series.index)
            except:
                print(f"Could not convert index to datetime for {column_name}")
                return df
        
        # Make sure series has a name
        if series.name is None:
            series.name = column_name
        
        # Join with the main dataframe
        df = df.join(series, how='left')
        
        # Rename if necessary
        if series.name != column_name:
            df = df.rename(columns={series.name: column_name})
    
    return df

def main():
    """Main function to collect and organize data"""
    # Initialize API clients
    clients = initialize_clients()
    
    # Date range for data collection
    start_date = Config.START_DATE
    end_date = Config.END_DATE
    
    print(f"Collecting data from {start_date.strftime(Config.DATE_FORMAT)} to {end_date.strftime(Config.DATE_FORMAT)}")
    
    # Parse the primary currency pair
    primary_base = Config.PRIMARY_PAIR[:3]
    primary_quote = Config.PRIMARY_PAIR[3:]
    
    # Initialize the main dataframe with dates as index
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    df = pd.DataFrame(index=date_range)
    df.index.name = 'date'
    
    # 1. Get primary forex pair data
    if clients['polygon']:
        primary_pair_series = get_polygon_forex(
            clients['polygon'], 
            primary_base, 
            primary_quote, 
            Config.PRIMARY_PAIR,
            start_date.strftime(Config.DATE_FORMAT),
            end_date.strftime(Config.DATE_FORMAT)
        )
        df = add_series_to_df(df, primary_pair_series, Config.PRIMARY_PAIR)
    
    # 2. Determine which countries we need data for
    # At minimum, we need data for the countries in the primary pair
    base_country = Config.CURRENCY_COUNTRY_MAP.get(primary_base, "Unknown")
    quote_country = Config.CURRENCY_COUNTRY_MAP.get(primary_quote, "Unknown")
    
    countries_to_process = set()
    
    # Always include the countries in the primary pair
    if base_country != "Unknown":
        countries_to_process.add(base_country)
    if quote_country != "Unknown":
        countries_to_process.add(quote_country)
    
    # Optionally include all countries from major currencies
    for currency in Config.MAJOR_CURRENCIES:
        country = Config.CURRENCY_COUNTRY_MAP.get(currency)
        if country:
            countries_to_process.add(country)
    
    # Add China as it's often important for global markets
    countries_to_process.add("China")
    
    print(f"Collecting data for countries: {', '.join(countries_to_process)}")
    
    # 3. Collect stock market indices
    for country in countries_to_process:
        indices = Config.STOCK_INDICES.get(country, [])
        for index_name in indices:
            series = get_stock_index(
                clients['fred'],
                index_name,
                index_name,
                start_date,
                end_date
            )
            df = add_series_to_df(df, series, index_name)
    
    # 4. Collect commodity data
    for commodity in Config.COMMODITIES:
        series = get_commodity_data(
            clients['fred'],
            commodity,
            commodity,
            start_date,
            end_date
        )
        df = add_series_to_df(df, series, commodity)
    
    # 5. Collect economic indicators for each country
    for country in countries_to_process:
        for indicator in Config.ECONOMIC_INDICATORS:
            column_name = f"{country} {indicator}"
            series = get_economic_indicator(
                clients['fred'],
                country,
                indicator,
                column_name,
                start_date,
                end_date
            )
            df = add_series_to_df(df, series, column_name)
        
        # 6. Collect bond yields for each country
        for term in Config.BOND_YIELDS:
            column_name = f"{country} {term} Gov Y"
            series = get_bond_yield(
                clients['fred'],
                country,
                term,
                column_name,
                start_date,
                end_date
            )
            df = add_series_to_df(df, series, column_name)
    
    # 7. Collect REER data
    for country in countries_to_process:
        country_code = Config.COUNTRY_CODES.get(country, {}).get('BIS')
        if country_code:
            column_name = f"{country} REER"
            series = get_bis_reer(
                country_code,
                column_name,
                start_date,
                end_date
            )
            df = add_series_to_df(df, series, column_name)
    
    # 8. Collect COT data for the primary pair currencies
    for currency in [primary_base, primary_quote]:
        if currency in ["GBP", "AUD", "EUR", "JPY", "CHF", "NZD", "CAD"]:  # Currencies with futures
            long_col = f"{currency} COT_Long"
            short_col = f"{currency} COT_Short"
            
            long_series, short_series = get_cot_data(currency, long_col, short_col)
            df = add_series_to_df(df, long_series, long_col)
            df = add_series_to_df(df, short_series, short_col)
    
    # 9. Collect special indicators
    special_indicators = get_special_indicators(clients['fred'], start_date, end_date)
    for name, series in special_indicators.items():
        df = add_series_to_df(df, series, name)
    
    # 10. Save the data to CSV
    output_filename = f"{Config.PRIMARY_PAIR}_analysis_data.csv"
    df.to_csv(output_filename)
    print(f"Data saved to {output_filename}")
    
    # 11. Display basic statistics
    print("\nData Overview:")
    print(f"Time period: {df.index.min()} to {df.index.max()}")
    print(f"Number of columns: {len(df.columns)}")
    print(f"Data completeness: {df.notna().mean().mean() * 100:.2f}%")
    
    # 12. Display correlation with primary currency pair
    if Config.PRIMARY_PAIR in df.columns:
        print(f"\nTop correlations with {Config.PRIMARY_PAIR}:")
        correlations = df.corr()[Config.PRIMARY_PAIR].sort_values(ascending=False)
        print(correlations.head(10))
        
        # Optionally create a correlation heatmap
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Select top correlated columns
            top_corr_cols = correlations.abs().sort_values(ascending=False).head(15).index
            corr_df = df[top_corr_cols].corr()
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
            plt.title(f'Correlation Matrix for {Config.PRIMARY_PAIR} Analysis')
            plt.tight_layout()
            
            # Save the correlation plot
            plt.savefig(f"{Config.PRIMARY_PAIR}_correlations.png")
            print(f"Correlation heatmap saved to {Config.PRIMARY_PAIR}_correlations.png")
        except Exception as e:
            print(f"Could not create correlation plot: {str(e)}")
    
    return df

if __name__ == "__main__":
    df = main()
    print("\nScript completed successfully")
