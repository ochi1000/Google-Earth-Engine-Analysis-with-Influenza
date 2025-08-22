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
    try:
        point = ee.Geometry.Point([-89.000000, 40.000000])
        dataset = ee.ImageCollection('MODIS/061/MOD11A1')
        temperature = dataset.select('LST_Day_1km')
        temperature = temperature.filterDate(start_date, end_date).mean()
        temperature = temperature.sample(point, 5000).get('LST_Day_1km').getInfo()
        # return temperature
        # print(temperature)
        # return temperature
        if temperature is None:
                return None
        return temperature
    except Exception as e:
        print(e, start_date, end_date, temperature)
        return None

def compute_ccf(df, temperature_data, max_lag=4):
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
        ('2018-01-01', '2024-12-31'),
        ('2018-01-01', '2023-12-31'),
        ('2018-01-01', '2022-12-31'),
        ('2019-01-01', '2021-12-31'),
        ('2019-01-01', '2020-12-31'),
        ('2019-01-01', '2019-12-31'),
        ('2020-01-01', '2020-12-31'),
        ('2020-01-01', '2021-12-31'),
        ('2020-01-01', '2022-12-31'),
        ('2020-01-01', '2023-12-31'),
        ('2020-01-01', '2024-12-31'),
        ('2021-01-01', '2021-12-31'),
        ('2021-01-01', '2022-12-31'),
        ('2021-01-01', '2023-12-31'),
        ('2021-01-01', '2024-12-31'),
        ('2022-01-01', '2022-12-31'),
        ('2022-01-01', '2023-12-31'),
        ('2022-01-01', '2024-12-31'),
        ('2023-01-01', '2023-12-31'),
        ('2023-01-01', '2024-12-31'),
        ('2024-01-01', '2024-12-31')
    ]
    period_labels = [f'{start_date} to {end_date}' for start_date, end_date in periods]

    results = {}
    # Ensure the week_start and week_end columns are in datetime format
    df['week_start'] = pd.to_datetime(df['week_start'], format='%m/%d/%Y')
    df['week_end'] = pd.to_datetime(df['week_end'], format='%m/%d/%Y')

    for (start_date, end_date), period_label in zip(periods, period_labels):
        # Convert start_date and end_date to datetime for comparison
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        period_df = df[(df['week_start'] >= start_date) & (df['week_end'] <= end_date)]
        temperature_data = [
            temp_data for temp_data in [
            get_temperature_data(row['week_start'].strftime('%Y-%m-%d'), row['week_end'].strftime('%Y-%m-%d')) 
            for _, row in period_df.iterrows()
            ] if temp_data is not None
        ]
        best_ccf_lag, _ = compute_ccf(period_df, temperature_data)
        best_covariance_lag, _ = compute_covariance_lag(period_df, temperature_data)
        # adl_params = compute_adl(period_df)

        results[period_label] = {
            'Best CCF Lag': best_ccf_lag,
            'Best Covariance Lag': best_covariance_lag,
            # 'ADL Params': adl_params.to_dict()
        }

    results_df = pd.DataFrame.from_dict(results, orient='index')
    output_file = file_path.replace('.csv', '_lag_results.csv')
    results_df.to_csv(output_file)
    print(f'Lag results saved to {output_file}')

def main():
    # input_path = input("Enter the file path: ")
    input_path = "filtered_file.csv"
    if os.path.isfile(input_path):
        process_csv_file(input_path)
    else:
        print("Invalid path. Please provide a valid file.")

if __name__ == "__main__":
    main()
