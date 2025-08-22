import pandas as pd
import os
import ee
import numpy as np
from statsmodels.tsa.api import AutoReg

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

# Initialize the Earth Engine module.
ee.Initialize()

def get_temperature_data(start_date, end_date):
    point = ee.Geometry.Point([-89.000000, 40.000000])
    dataset = ee.ImageCollection('MODIS/061/MOD11A1')
    temperature = dataset.select('LST_Day_1km')
    temperature = temperature.filterDate(start_date, end_date).mean()
    temperature = temperature.sample(point, 5000).first().get('LST_Day_1km').getInfo()
    return temperature

def compute_ccf(df, temperature_data, max_lag=10):
    df['Temperature'] = temperature_data
    
    if df['percent'].std() == 0 or df['Temperature'].std() == 0 or df['percent'].std() is None or df['Temperature'].std() is None:
        return "NA", []

    ccf_values = [df['percent'].corr(df['Temperature'].shift(lag)) for lag in range(-max_lag, max_lag + 1)]
    best_lag = np.argmax(ccf_values) - max_lag
    return best_lag, ccf_values

def compute_covariance_lag(df, temperature_data, max_lag=10):
    df['Temperature'] = temperature_data
    
    covariance_values = [df['percent'].cov(df['Temperature'].shift(lag)) for lag in range(-max_lag, max_lag + 1)]
    best_lag = np.argmax(covariance_values) - max_lag
    return best_lag, covariance_values

def compute_adl(df, max_lag=10):
    model = AutoReg(df['percent'], lags=max_lag, old_names=False)
    model_fit = model.fit()
    return model_fit.params

def process_csv_file(file_path):
    df = read_csv(file_path)
    # Scale the percent column to match the expected range of 0 to 10
    df['percent'] = df['percent'] * 100

    periods = [
        # ('2018-01-01', '2024-12-31'),
        # ('2018-01-01', '2023-12-31'),
        # ('2018-01-01', '2022-12-31'),
        # ('2019-01-01', '2021-12-31'),
        # ('2019-01-01', '2020-12-31'),
        # ('2019-01-01', '2019-12-31'),
        # ('2020-01-01', '2020-12-31'),
        # ('2020-01-01', '2021-12-31'),
        # ('2020-01-01', '2022-12-31'),
        # ('2020-01-01', '2023-12-31'),
        # ('2020-01-01', '2024-12-31'),
        # ('2021-01-01', '2021-12-31'),
        # ('2021-01-01', '2022-12-31'),
        # ('2021-01-01', '2023-12-31'),
        # ('2021-01-01', '2024-12-31'),
        # ('2022-01-01', '2022-12-31'),
        # ('2022-01-01', '2023-12-31'),
        # ('2022-01-01', '2024-12-31'),
        # ('2023-01-01', '2023-12-31'),
        # ('2023-01-01', '2024-12-31'),
        ('2024-01-01', '2024-12-31')
    ]
    period_labels = [f'{start_date} to {end_date}' for start_date, end_date in periods]

    results = {}
    for (start_date, end_date), period_label in zip(periods, period_labels):
        temperature_data = get_temperature_data(start_date, end_date)
        best_ccf_lag, _ = compute_ccf(df, temperature_data)
        best_covariance_lag, _ = compute_covariance_lag(df, temperature_data)
        adl_params = compute_adl(df)

        results[period_label] = {
            'Best CCF Lag': best_ccf_lag,
            'Best Covariance Lag': best_covariance_lag,
            'ADL Params': adl_params.to_dict()
        }

    results_df = pd.DataFrame.from_dict(results, orient='index')
    output_file = file_path.replace('.csv', '_lag_results.csv')
    results_df.to_csv(output_file)
    print(f'Lag results saved to {output_file}')

def main():
    input_path = input("Enter the file path: ")
    if os.path.isfile(input_path):
        process_csv_file(input_path)
    else:
        print("Invalid path. Please provide a valid file.")

if __name__ == "__main__":
    main()
