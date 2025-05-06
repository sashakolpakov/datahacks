# ----------------------------------------------
# Script: merge_macro_data.py
# Description: Combines GBPAUD.csv with World Bank, FRED, and OECD indicators
# ----------------------------------------------

import pandas as pd
import wbgapi as wb
import os
from pandas_datareader import data as pdr
from pandasdmx import Request
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

# --- STEP 4: Pull OECD Unemployment via SDMX ---
oecd = Request('OECD')
oecd_frames = []

for code, country in [('LRHUTTTT.STSA.M.AUS', 'Australia'), ('LRHUTTTT.STSA.M.GBR', 'UK')]:
    try:
        resp = oecd.data(resource_id='MEI', key=code, params={'startPeriod': '2000'})
        df = resp.to_pandas()
        df = df.rename(f'{country} Unemployment Rate [OECD]')
        df = df.resample('D').ffill()
        oecd_frames.append(df)
    except Exception as e:
        print(f"⚠️ OECD fetch failed for {country}: {e}")

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
