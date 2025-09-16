# DataHacks - Forex Economic Data Analysis Suite

A comprehensive toolkit for collecting, analyzing, and visualizing economic data to understand forex market movements. This suite integrates data from multiple sources including FRED, World Bank, OECD, and Polygon.io to provide deep insights into macroeconomic indicators affecting currency pairs.

## 🚀 Features

- **Multi-Source Data Collection**: Automated fetching from FRED, World Bank, OECD, and Polygon.io APIs
- **Interactive Dashboards**: Both Streamlit and Dash implementations for data exploration
- **Comprehensive Analysis**: Correlation analysis, clustering, anomaly detection, and PCA
- **Automated Data Processing**: Smart data merging and enrichment workflows
- **Export Capabilities**: CSV downloads and data persistence
- **Configurable Currency Pairs**: Support for major forex pairs with extensible configuration

## 📊 Data Sources

| Source | Data Types | Examples |
|--------|------------|----------|
| **FRED** | Economic indicators, interest rates, GDP | US GDP, VIX, Federal funds rate |
| **World Bank** | Country statistics, demographics | Population, unemployment, labor force |
| **OECD** | Leading indicators, composite indices | CLI (Composite Leading Indicator) |
| **Polygon.io** | Real-time market data, forex rates | Currency pair prices, market data |

## 🛠 Installation

### Prerequisites
- Python 3.8+
- API keys for data sources (see Configuration section)

### Setup
```bash
# Clone the repository
git clone <repository-url>
cd datahacks

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Required Dependencies
```bash
pip install pandas numpy requests python-dotenv fredapi polygon-api-client
pip install streamlit dash plotly scikit-learn wbgapi pandas-datareader pandasdmx
```

## ⚙️ Configuration

Create a `.env` file in the root directory:

```env
FRED_API_KEY=your_fred_api_key_here
POLYGON_API_KEY=your_polygon_api_key_here
```

### API Key Setup
1. **FRED API**: Register at [https://fred.stlouisfed.org/docs/api/api_key.html](https://fred.stlouisfed.org/docs/api/api_key.html)
2. **Polygon.io**: Get your key at [https://polygon.io/](https://polygon.io/)

## 📈 Usage Examples

### 1. Basic Data Collection

```python
from data_collect import main, Config

# Collect data for GBP/AUD pair
Config.PRIMARY_PAIR = "GBPAUD"
data = main()
print(f"Collected {len(data)} rows of data")
```

### 2. FRED Data Fetching

```python
from fred_fetcher import get_fred_data
import os
from datetime import datetime, timedelta

# Fetch FRED economic indicators
end_date = datetime.now()
start_date = end_date - timedelta(days=365*5)  # 5 years

fred_data, metadata = get_fred_data(
    api_key=os.getenv("FRED_API_KEY"),
    start_date=start_date.strftime("%Y-%m-%d"),
    end_date=end_date.strftime("%Y-%m-%d")
)

print(f"Downloaded {len(fred_data)} FRED series")
```

### 3. Data Enrichment Pipeline

```python
# Run the complete enrichment pipeline
python merge_macro_data.py
```

This will:
- Load base currency data
- Fetch World Bank indicators
- Merge FRED economic data
- Apply OECD indicators
- Export enriched dataset

### 4. Interactive Analysis

#### Streamlit Dashboard
```bash
streamlit run st.py
```

#### Dash Analytics Dashboard
```bash
python fancy_dash.py
```

## 🔧 Script Overview

### Core Scripts

| Script | Purpose | Key Features |
|--------|---------|-------------|
| `data_collect.py` | Main data collection engine | Multi-API integration, error handling, data validation |
| `fred_fetcher.py` | FRED API specialist | 40+ economic indicators, metadata collection |
| `merge_macro_data.py` | Data enrichment pipeline | World Bank + FRED + OECD integration |
| `st.py` | Streamlit web interface | Interactive data collection, real-time preview |
| `fancy_dash.py` | Advanced analytics dashboard | ML clustering, PCA, anomaly detection |

### Configuration

The `Config` class in `data_collect.py` controls:

```python
class Config:
    PRIMARY_PAIR = "GBPAUD"  # Main currency pair
    MAJOR_CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]

    # Country mappings for different APIs
    CURRENCY_COUNTRY_MAP = {
        "USD": "United States",
        "EUR": "Euro Area",
        # ... more mappings
    }
```

## 📊 Visualization Examples

### Economic Indicator Correlations
```python
import pandas as pd
import plotly.express as px
import numpy as np

# Load enriched data
df = pd.read_csv('GBPAUD_enriched.csv')

# Create correlation heatmap
correlation_matrix = df.select_dtypes(include=[np.number]).corr()
fig = px.imshow(correlation_matrix,
                title="Economic Indicators Correlation Matrix",
                color_continuous_scale='RdBu_r')
fig.show()
```

### Time Series Analysis
```python
import plotly.graph_objects as go

# Plot currency pair with economic indicators
fig = go.Figure()

# Add currency pair price
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['close'],
    name='GBP/AUD',
    yaxis='y1'
))

# Add economic indicator (e.g., GDP)
fig.add_trace(go.Scatter(
    x=df['date'],
    y=df['UK_GDP'],
    name='UK GDP',
    yaxis='y2'
))

# Update layout for dual y-axis
fig.update_layout(
    title="GBP/AUD vs UK GDP",
    yaxis=dict(title="Currency Rate"),
    yaxis2=dict(title="GDP", overlaying='y', side='right')
)
fig.show()
```

### Machine Learning Clustering
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Prepare data for clustering
features = ['close', 'UK_GDP', 'US_GDP', 'VIX']
df_ml = df[features].dropna()

# Standardize features
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_ml)

# Apply K-means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

# Visualize clusters
plt.figure(figsize=(10, 6))
plt.scatter(df_ml['close'], df_ml['VIX'], c=clusters, cmap='viridis')
plt.xlabel('GBP/AUD Close')
plt.ylabel('VIX')
plt.title('Market Regime Clustering')
plt.colorbar(label='Cluster')
plt.show()
```

## 🎯 Advanced Features

### Machine Learning Integration

The Dash dashboard (`fancy_dash.py`) includes:

- **K-Means Clustering**: Identify market regimes
- **PCA Analysis**: Dimensionality reduction for pattern recognition
- **Isolation Forest**: Anomaly detection in market data
- **Interactive Filtering**: Real-time data exploration

### Data Processing Pipeline

```
Raw Currency Data → FRED Economic Indicators → World Bank Country Stats → OECD Leading Indicators → Enriched Dataset → Analysis & Visualization
```

## 🚨 Error Handling

The scripts include comprehensive error handling:

- **API Rate Limiting**: Automatic retry with exponential backoff
- **Data Validation**: Missing data detection and handling
- **Fallback Mechanisms**: Alternative data sources when primary fails
- **Logging**: Detailed execution logs for debugging

## 🔍 Example Outputs

### Sample Data Structure
```
Date        | Close  | UK_GDP | US_GDP | VIX  | Gold_Price
2024-01-01  | 1.8734 | 3.2T   | 27.4T  | 12.4 | 2087.23
2024-01-02  | 1.8756 | 3.2T   | 27.4T  | 13.1 | 2091.45
...
```

### Generated Files
- `GBPAUD_analysis_data.csv` - Raw collected data
- `GBPAUD_enriched.csv` - Processed with all indicators
- `fred_data.csv` - FRED economic indicators
- `fred_metadata.csv` - Series descriptions and metadata

## 🏃‍♂️ Quick Start Guide

1. **Clone and setup**:
   ```bash
   git clone <repo-url>
   cd datahacks
   pip install -r requirements.txt
   ```

2. **Configure API keys**:
   ```bash
   echo "FRED_API_KEY=your_key_here" > .env
   echo "POLYGON_API_KEY=your_key_here" >> .env
   ```

3. **Run data collection**:
   ```bash
   python data_collect.py
   ```

4. **Launch interactive dashboard**:
   ```bash
   streamlit run st.py
   ```

## 📊 Dashboard Screenshots

The Streamlit dashboard provides:
- Currency pair selection
- Date range configuration
- Real-time data collection progress
- Data preview and download
- Basic statistics and visualizations

The Dash dashboard offers:
- Advanced ML analytics
- Interactive clustering
- PCA dimensionality reduction
- Anomaly detection
- Export capabilities

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- Check the logs for detailed error messages
- Ensure all API keys are properly configured
- Verify internet connectivity for data fetching
- Review the `.env` file setup

For issues and feature requests, please open a GitHub issue.

## 🔧 Configuration Details

### Supported Currency Pairs
- EURUSD, USDJPY, GBPUSD, USDCHF
- AUDUSD, USDCAD, GBPJPY, GBPAUD
- Custom pairs (configurable)

### Economic Indicators Collected
- GDP (Gross Domestic Product)
- Unemployment rates
- Inflation (CPI)
- Interest rates and yield curves
- Trade balances
- Current account balances
- Money supply (M2)
- Stock market indices
- Commodity prices (Gold, Oil, Copper)
- Volatility indices (VIX)
- Leading economic indicators

### Data Quality Features
- Automatic data validation
- Missing value handling
- Outlier detection
- Data consistency checks
- Temporal alignment across sources