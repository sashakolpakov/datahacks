import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import base64

# Import the data collection module (from our previous script)
from enhanced_forex_data_script import main, Config

# App title and description
st.title("Forex Data Collection Tool")
st.markdown("""
This tool allows you to collect comprehensive economic and financial data 
for analysis of forex currency pairs.
""")

# Sidebar for configuration
st.sidebar.header("Configuration")

# Currency pair selection
primary_pairs = [
    "EURUSD", "USDJPY", "GBPUSD", "USDCHF", 
    "AUDUSD", "USDCAD", "GBPJPY", "GBPAUD"
]
selected_pair = st.sidebar.selectbox("Select Currency Pair", primary_pairs, index=primary_pairs.index("GBPAUD"))

# Date range selection
default_end_date = datetime.now()
default_start_date = default_end_date - timedelta(days=365*5)  # 5 years

start_date = st.sidebar.date_input("Start Date", default_start_date)
end_date = st.sidebar.date_input("End Date", default_end_date)

# Advanced options collapsible
with st.sidebar.expander("Advanced Options"):
    include_all_countries = st.checkbox("Include All Major Currencies", value=True)
    include_commodities = st.checkbox("Include Commodity Data", value=True)
    include_indices = st.checkbox("Include Stock Indices", value=True)
    include_special = st.checkbox("Include Special Indicators (CLI, NFCI, VIX)", value=True)

# API keys input (optional - use .env file in production)
with st.sidebar.expander("API Keys (Optional)"):
    fred_key = st.text_input("FRED API Key", type="password")
    polygon_key = st.text_input("Polygon.io API Key", type="password")
    
    if fred_key:
        os.environ["FRED_API_KEY"] = fred_key
    if polygon_key:
        os.environ["POLYGON_API_KEY"] = polygon_key

# Main functionality
if st.button("Collect Data"):
    # Show a spinner while working
    with st.spinner(f"Collecting data for {selected_pair}..."):
        # Update Config with the selected options
        Config.PRIMARY_PAIR = selected_pair
        Config.START_DATE = datetime.combine(start_date, datetime.min.time())
        Config.END_DATE = datetime.combine(end_date, datetime.min.time())
        
        # If not including all countries, limit to just the pair countries
        if not include_all_countries:
            base_currency = selected_pair[:3]
            quote_currency = selected_pair[3:]
            Config.MAJOR_CURRENCIES = [base_currency, quote_currency]
        
        # Set other options
        if not include_commodities:
            Config.COMMODITIES = []
        
        if not include_indices:
            Config.STOCK_INDICES = {country: [] for country in Config.STOCK_INDICES}
        
        if not include_special:
            Config.SPECIAL_INDICATORS = []
        
        # Run data collection
        df = main()
        
        # Display success message
        st.success(f"Data collection complete! Found {len(df.columns)} indicators.")
        
        # Show a preview of the data
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # Show basic statistics
        st.subheader("Data Statistics")
        st.write(f"Time Period: {df.index.min()} to {df.index.max()}")
        st.write(f"Data Completeness: {df.notna().mean().mean() * 100:.2f}%")
        
        # Create download link
        csv = df.to_csv()
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="{selected_pair}_data.csv">Download CSV File</a>'
        st.markdown(href, unsafe_allow_html=True)
        
        # Show top correlations
        if selected_pair in df.columns:
            st.subheader(f"Top Correlations with {selected_pair}")
            correlations = df.corr()[selected_pair].sort_values(ascending=False)
            st.dataframe(correlations.head(10))
else:
    # Show instructions when the app first loads
    st.info("Configure your options and click 'Collect Data' to begin.")
    st.write("This tool will gather data from:")
    st.write("- FRED (Federal Reserve Economic Data)")
    st.write("- Polygon.io (for forex data)")
    st.write("- BIS (Bank for International Settlements)")
    st.write("- World Bank (for additional economic indicators)")
    st.write("- CFTC (for Commitments of Traders data)")
    
    # Example image
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/53/GBPAUD_exchange_rate_chart.svg/1200px-GBPAUD_exchange_rate_chart.svg.png", 
             caption="Example GBPAUD Exchange Rate Chart", use_column_width=True)
