import ee
# import geemap
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

ee.Initialize()

def analyze_lag_gee(location, start_date, end_date, influenza_csv, max_lag=10):
    """
    Analyzes the lag relationship between GEE temperature data and influenza incidents.

    Args:
        location (ee.Geometry.Point): GEE point geometry for the location.
        start_date (str): Start date (YYYY-MM-DD).
        end_date (str): End date (YYYY-MM-DD).
        influenza_csv (str): Path to the influenza incidents CSV file.
        max_lag (int): Maximum lag to consider.
    """

    # 1. Initialize GEE
    ee.Initialize()

    # 2. GEE Temperature Data Retrieval (ERA5 example)
    temperature_collection = ee.ImageCollection('ECMWF/ERA5/WEEKLY_AGGR') \
        .filterBounds(location) \
        .filterDate(start_date, end_date) \
        .select('mean_2m_air_temperature')

    def get_weekly_mean(image):
        mean_temp = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=location,
            scale=1000  # Adjust scale as needed
        ).get('mean_2m_air_temperature')
        return ee.Feature(None, {'temperature': mean_temp, 'date': image.date().millis()})

    temperature_features = temperature_collection.map(get_weekly_mean).getInfo()['features']

    temperature_data = pd.DataFrame([f['properties'] for f in temperature_features])
    temperature_data['date'] = pd.to_datetime(temperature_data['date'], unit='ms')
    temperature_data.set_index('date', inplace=True)

    # 3. Load Influenza Data
    influenza_df = pd.read_csv(influenza_csv, parse_dates=['date'], index_col='date')

    # 4. Merge DataFrames
    merged_df = pd.merge(temperature_data, influenza_df, left_index=True, right_index=True, how='inner')

    # 5. Lag Analysis (Same as Before)
    ccf = sm.tsa.stattools.ccf(merged_df['influenza'], merged_df['temperature'], nlags=max_lag)
    lags = np.arange(-max_lag, max_lag + 1)

    plt.figure(figsize=(10, 5))
    plt.stem(lags, ccf)
    plt.title('Cross-Correlation Function')
    plt.xlabel('Lag (weeks)')
    plt.ylabel('Correlation')
    plt.show()

    abs_ccf = np.abs(ccf)
    best_lag = lags[np.argmax(abs_ccf)]
    print(f"Best lag based on CCF: {best_lag} weeks")

    r_squared_values = []
    p_values = []
    for lag in range(max_lag + 1):
        merged_df[f'temp_lag_{lag}'] = merged_df['temperature'].shift(lag)
        df_regression = merged_df.dropna()
        if not df_regression.empty:
            X = df_regression[f'temp_lag_{lag}']
            y = df_regression['influenza']
            X = sm.add_constant(X)
            model = sm.OLS(y, X).fit()
            r_squared_values.append(model.rsquared)
            p_values.append(model.pvalues[1])
        else:
            r_squared_values.append(np.nan)
            p_values.append(np.nan)

    plt.figure(figsize=(10, 5))
    plt.plot(range(max_lag + 1), r_squared_values, marker='o')
    plt.title('R-squared vs. Lag')
    plt.xlabel('Lag (weeks)')
    plt.ylabel('R-squared')
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(range(max_lag + 1), p_values, marker='o')
    plt.title('P-values vs. Lag')
    plt.xlabel('Lag (weeks)')
    plt.ylabel('P-values')
    plt.show()

    best_r_squared_lag = np.nanargmax(r_squared_values)
    print(f"Best lag based on R-squared: {best_r_squared_lag} weeks")

    best_p_lag = np.nanargmin(p_values)
    print(f"Best lag based on p-value: {best_p_lag} weeks")

# Example Usage
location = ee.Geometry.Point([-90, 30])  # Replace with your location
start_date = '2020-01-01'
end_date = '2023-12-31'
influenza_csv_path = 'your_influenza_data.csv'  # Replace with your influenza CSV path

analyze_lag_gee(location, start_date, end_date, influenza_csv_path)