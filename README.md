# Configuration Guide for Forex Data Collection

This guide explains how to configure the enhanced forex data collection script to work with different currency pairs and data sources.

## Primary Configuration Options

The script is highly configurable through the `Config` class. Here are the main settings you can adjust:

### 1. Primary Currency Pair

```python
PRIMARY_PAIR = "GBPAUD"  # Format: BASE/QUOTE without the slash
```

Change this to any currency pair you want to analyze. For example:
- `"EURUSD"` for Euro/US Dollar
- `"USDJPY"` for US Dollar/Japanese Yen
- `"GBPUSD"` for British Pound/US Dollar

### 2. Major Currencies

```python
MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
```

This list defines which currencies the script will collect data for. Add or remove currencies as needed. The script will automatically collect more data for countries associated with these currencies.

### 3. Date Range

```python
START_DATE = datetime.now() - timedelta(days=365*10)  # 10 years ago
END_DATE = datetime.now()
```

Adjust the time period for data collection. For example, to collect 5 years of data:
```python
START_DATE = datetime.now() - timedelta(days=365*5)
```

### 4. Economic Indicators

```python
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
```

Add or remove economic indicators as needed. Note that availability may vary by country and data source.

### 5. Stock Indices

```python
STOCK_INDICES = {
    "United States": ["S&P 500", "Dow Jones", "NASDAQ"],
    "Euro Area": ["Euro STOXX 50", "CAC 40"],
    "United Kingdom": ["FTSE 100"],
    # ...
}
```

Configure which stock indices to collect for each country.

### 6. Commodities

```python
COMMODITIES = ["Copper", "Gold", "Silver", "Crude Oil", "Brent", "Natural Gas", 
               "Iron Ore", "Coal"]
```

Add or remove commodities as needed.

## Adding New Data Sources

The script is designed to be extensible with new data sources. Here's how to add a new data source:

1. Create a new function that retrieves data from your source
2. Add appropriate mappings in the appropriate collection function
3. Call your function from the main data collection process

Example for adding a new economic data source:

```python
def get_custom_economic_data(country, indicator, column_name, start_date, end_date):
    """Get economic data from a custom source"""
    try:
        # Implementation for your data source
        # ...
        
        return series
    except Exception as e:
        print(f"Error retrieving {column_name} from custom source: {str(e)}")
        return None
```

Then add it to the main data collection process.

## Country and Currency Mappings

The script includes mappings between currencies, countries, and various country codes used by different APIs:

```python
CURRENCY_COUNTRY_MAP = {
    "USD": "United States",
    "EUR": "Euro Area",
    "GBP": "United Kingdom",
    # ...
}

COUNTRY_CODES = {
    "United States": {"FRED": "US", "WB": "USA", "BIS": "US"},
    "Euro Area": {"FRED": "XM", "WB": "EMU", "BIS": "XM"},
    # ...
}
```

You may need to update these mappings when adding new countries or working with different APIs.

## API Series ID Mappings

Many data sources like FRED require specific series IDs. The script includes mappings for common indicators:

```python
indicator_map = {
    ('United States', 'GDP'): 'GDP',                     # US Real GDP
    ('United States', 'Unemployment'): 'UNRATE',         # US Unemployment Rate
    # ...
}
```

You can find additional series IDs by searching the respective API documentation:

- FRED: https://fred.stlouisfed.org/
- World Bank: https://data.worldbank.org/
- BIS: https://data.bis.org/

## Error Handling

The script includes robust error handling to continue processing even if some data sources fail. Each function returns `None` if data retrieval fails, and the main process safely handles these cases.

## Output Customization

By default, the script:
1. Saves data to a CSV file named after the primary currency pair
2. Displays basic statistics
3. Generates a correlation heatmap of the most important factors

You can customize output by modifying the end of the `main()` function.
