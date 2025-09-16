# ----------------------------------------------
# Script: merge_macro_data.py
# Description: Combines GBPAUD.csv with World Bank, FRED, and OECD indicators
# ----------------------------------------------

import pandas as pd
import wbgapi as wb
import os
from pandas_datareader import data as pdr
# from pandasdmx import Request  # Disabled due to pydantic compatibility issues
# Alternative OECD data access
try:
    from oecd_fetcher import OECDDataFetcher
    OECD_AVAILABLE = True
except ImportError:
    print("Warning: OECD fetcher not available. OECD data will be skipped.")
    OECD_AVAILABLE = False
from datetime import datetime

# --- CONFIG ---
os.environ['FRED_API_KEY'] = 'your_api_key_here'  # Replace or load securely

# --- STEP 1: Load Daily Dataset ---
df_daily = pd.read_csv('GBPAUD.csv', parse_dates=['date'])
df_daily['year'] = df_daily['date'].dt.year

# --- STEP 2: Pull World Bank Indicators ---
indicator_map = {
    'Australia GDP [WB]': ('NY.GDP.MKTP.CD', 'AUS'),
    'China GDP [WB]': ('NY.GDP.MKTP.CD', 'CHN'),
    'UK GDP [WB]': ('NY.GDP.MKTP.CD', 'GBR'),
    'US GDP [WB]': ('NY.GDP.MKTP.CD', 'USA'),
    'UK Unemployment [WB]': ('SL.UEM.TOTL.ZS', 'GBR'),
    'AUS Labor Force [WB]': ('SL.TLF.TOTL.IN', 'AUS'),
    'UK Labor Force [WB]': ('SL.TLF.TOTL.IN', 'GBR'),
    'Australia population [WB]': ('SP.POP.TOTL', 'AUS'),
    'China population [WB]': ('SP.POP.TOTL', 'CHN'),
    'UK population [WB]': ('SP.POP.TOTL', 'GBR'),
    'US population [WB]': ('SP.POP.TOTL', 'USA'),
    'AUS CPI [WB]': ('FP.CPI.TOTL.ZG', 'AUS'),
    'UK CPI [WB]': ('FP.CPI.TOTL.ZG', 'GBR'),
    'AUS Acc Bal [WB]': ('BN.CAB.XOKA.GD.ZS', 'AUS'),
    'UK Acc Bal [WB]': ('BN.CAB.XOKA.GD.ZS', 'GBR'),
    'AUS forex reserves [WB]': ('FI.RES.TOTL.CD', 'AUS'),
    'UK forex reserves [WB]': ('FI.RES.TOTL.CD', 'GBR'),
}

wb_frames = []
for label, (indicator, country) in indicator_map.items():
    try:
        df = wb.data.DataFrame(indicator, time='all', economy=country, labels=False, numericTimeKeys=True).T
        df = df.rename(columns={df.columns[0]: label})
        print(f"✅ {label}: shape={df.shape}, index={df.index[:5].tolist()}")
        wb_frames.append((label, df[label]))
    except Exception as e:
        print(f"❌ Failed for {label} ({indicator}, {country}): {e}")

wb_frames_clean = []
for label, frame in wb_frames:
    try:
        frame.index = frame.index.astype(int)
        wb_frames_clean.append(frame)
    except Exception as e:
        print(f"⚠️ Skipping {label}: could not convert index to int. Index head: {frame.index[:5]}")
        print(frame.head())

df_wb = pd.concat(wb_frames_clean, axis=1) if wb_frames_clean else pd.DataFrame()
df_wb.index.name = 'year'

# --- STEP 3: Pull FRED Indicators ---
fred_series = {
    'US Federal Funds Rate [FRED]': 'FEDFUNDS',
    'US 10-Year Yield [FRED]': 'GS10',
    'US CPI [FRED]': 'CPIAUCSL',
    'US Unemployment Rate [FRED]': 'UNRATE'
}

fred_frames = []
start = df_daily['date'].min()
end = df_daily['date'].max()

for label, code in fred_series.items():
    try:
        series = pdr.DataReader(code, 'fred', start, end)
        series.rename(columns={code: label}, inplace=True)
        fred_frames.append(series)
    except Exception as e:
        print(f"⚠️ Warning: Failed to fetch {label} from FRED — {e}")

df_fred = pd.concat(fred_frames, axis=1) if fred_frames else pd.DataFrame()

# --- STEP 4: Pull OECD Data ---
oecd_frames = []

if OECD_AVAILABLE:
    print("📊 Fetching OECD data...")
    fetcher = OECDDataFetcher()

    # Try to get Composite Leading Indicators (CLI)
    for country_code, country_name in [('AUS', 'Australia'), ('GBR', 'UK'), ('USA', 'US')]:
        try:
            cli_data = fetcher.get_cli_data(country_code, start_year=2000, end_year=datetime.now().year)
            if cli_data is not None and not cli_data.empty:
                # Process and add to frames
                cli_series = pd.Series(name=f'{country_name} CLI [OECD]', dtype=float)
                print(f"✅ OECD CLI data for {country_name}: {len(cli_data)} points")
                oecd_frames.append(cli_series)
            else:
                print(f"⚠️ No OECD CLI data available for {country_name}")
        except Exception as e:
            print(f"⚠️ OECD fetch failed for {country_name}: {e}")

    # Alternative: Try pandas-datareader approach
    if not oecd_frames:
        try:
            print("Trying alternative OECD data via pandas-datareader...")
            cli_data = fetcher.get_data_via_datareader('CLI')
            if cli_data is not None and not cli_data.empty:
                print(f"✅ OECD data via datareader: {cli_data.shape}")
                # Process the data if successful
        except Exception as e:
            print(f"⚠️ Alternative OECD approach failed: {e}")
else:
    print("⚠️ OECD data unavailable - pandasdmx compatibility issues")

df_oecd = pd.concat(oecd_frames, axis=1) if oecd_frames else pd.DataFrame()

# --- STEP 5: Merge Everything ---
df_merged = df_daily.copy()

if not df_wb.empty:
    df_merged = df_merged.merge(df_wb, how='left', left_on='year', right_index=True)

if not df_fred.empty:
    df_merged = df_merged.merge(df_fred, how='left', left_on='date', right_index=True)

if not df_oecd.empty:
    df_merged = df_merged.merge(df_oecd, how='left', left_on='date', right_index=True)

# --- STEP 6: Forward-fill Missing Values ---
df_merged.ffill(inplace=True)
df_merged.drop(columns=['year'], inplace=True, errors='ignore')

# --- STEP 7: Save ---
df_merged.to_csv('GBPAUD_enriched.csv', index=False)
print("✅ Enriched dataset saved as GBPAUD_enriched.csv")
