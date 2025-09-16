"""
OECD Data Fetcher
================

This module provides alternatives to pandasdmx for accessing OECD data.
pandasdmx has compatibility issues with pydantic 2.x.

Three approaches are provided:
1. pandas-datareader (recommended)
2. Direct OECD JSON API requests
3. Simple CSV download approach
"""

import pandas as pd
import pandas_datareader.data as web
import requests
import json
from datetime import datetime, timedelta
import time


class OECDDataFetcher:
    """Alternative OECD data fetcher to replace pandasdmx"""

    OECD_API_BASE = "http://stats.oecd.org/SDMX-JSON/data"
    OECD_CSV_BASE = "https://stats.oecd.org/restsdmx/sdmx.ashx/GetData"

    @staticmethod
    def get_data_via_datareader(dataset_code, start_date=None, end_date=None):
        """
        Get OECD data using pandas-datareader

        Args:
            dataset_code: OECD dataset code (e.g., 'CLI', 'HISTPOP', 'TUD')
            start_date: Start date (datetime or string)
            end_date: End date (datetime or string)

        Returns:
            pandas.DataFrame: OECD data
        """
        try:
            if start_date and end_date:
                data = web.DataReader(dataset_code, 'oecd', start_date, end_date)
            else:
                data = web.DataReader(dataset_code, 'oecd')
            return data
        except Exception as e:
            print(f"Error fetching OECD data via datareader: {e}")
            return None

    @staticmethod
    def get_data_via_json_api(dataset, dimensions, start_period=None, end_period=None, timeout=30):
        """
        Get OECD data via direct JSON API calls

        Args:
            dataset: Dataset name (e.g., 'MEI', 'QNA')
            dimensions: List of dimension lists [countries, indicators, etc.]
            start_period: Start period (e.g., '2020-01')
            end_period: End period (e.g., '2021-12')
            timeout: Request timeout in seconds

        Returns:
            dict: JSON response data
        """
        try:
            # Build dimension string
            dim_args = ['+'.join(d) if d else '' for d in dimensions]
            dim_str = '.'.join(dim_args)

            url = f"{OECDDataFetcher.OECD_API_BASE}/{dataset}/{dim_str}/all"

            params = {}
            if start_period:
                params['startPeriod'] = start_period
            if end_period:
                params['endPeriod'] = end_period

            print(f"Requesting: {url}")
            response = requests.get(url, params=params, timeout=timeout)

            if response.status_code == 200:
                return response.json()
            else:
                print(f"API returned status {response.status_code}")
                return None

        except Exception as e:
            print(f"Error fetching OECD data via JSON API: {e}")
            return None

    @staticmethod
    def get_cli_data(country_code, start_year=2020, end_year=2024):
        """
        Get Composite Leading Indicator (CLI) data for a specific country

        Args:
            country_code: ISO country code (e.g., 'USA', 'GBR', 'AUS')
            start_year: Start year
            end_year: End year

        Returns:
            pandas.DataFrame: CLI data
        """
        try:
            start_date = datetime(start_year, 1, 1)
            end_date = datetime(end_year, 12, 31)

            # Try pandas-datareader first
            data = OECDDataFetcher.get_data_via_datareader('CLI', start_date, end_date)

            if data is not None and not data.empty:
                # Filter for specific country if possible
                if 'LOCATION' in data.columns:
                    data = data[data['LOCATION'] == country_code]
                return data

            # Fallback to JSON API
            dimensions = [[country_code], ['LOLITONOSTSAM'], [], ['M']]
            start_period = f"{start_year}-01"
            end_period = f"{end_year}-12"

            json_data = OECDDataFetcher.get_data_via_json_api(
                'MEI', dimensions, start_period, end_period
            )

            if json_data:
                # Convert JSON to DataFrame (simplified)
                return OECDDataFetcher._json_to_dataframe(json_data)

            return None

        except Exception as e:
            print(f"Error fetching CLI data: {e}")
            return None

    @staticmethod
    def _json_to_dataframe(json_data):
        """Convert OECD JSON response to pandas DataFrame"""
        try:
            # This is a simplified converter - OECD JSON structure is complex
            if 'dataSets' in json_data and len(json_data['dataSets']) > 0:
                observations = json_data['dataSets'][0].get('observations', {})

                data_points = []
                for key, value in observations.items():
                    if isinstance(value, list) and len(value) > 0:
                        data_points.append({
                            'key': key,
                            'value': value[0]
                        })

                return pd.DataFrame(data_points)

            return pd.DataFrame()

        except Exception as e:
            print(f"Error converting JSON to DataFrame: {e}")
            return pd.DataFrame()


# Example usage and testing
if __name__ == "__main__":
    print("=== OECD Data Fetcher Test ===")

    fetcher = OECDDataFetcher()

    # Test 1: Get CLI data for USA
    print("Testing CLI data for USA...")
    cli_data = fetcher.get_cli_data('USA', 2022, 2024)

    if cli_data is not None and not cli_data.empty:
        print(f"✅ CLI data retrieved: {len(cli_data)} rows")
        print(cli_data.head())
    else:
        print("❌ CLI data retrieval failed")

    print("\n=== Available Datasets ===")
    print("Common OECD dataset codes:")
    print("- CLI: Composite Leading Indicators")
    print("- QNA: Quarterly National Accounts")
    print("- MEI: Main Economic Indicators")
    print("- HISTPOP: Historical Population")
    print("- TUD: Trade Union Density")