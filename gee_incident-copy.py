import pandas as pd
import matplotlib.pyplot as plt
import os
import ee
import numpy as np
import re

def read_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def plot_time_series(df, file_name):
    fig, ax = plt.subplots()

    ax.set_xlabel('Date')
    ax.set_ylabel('ILI Activity Level', color='tab:blue')
    start_date = input("Enter the start date (Y-M-D): ")
    end_date = input("Enter the end date (Y-M-D): ")

    df['Week_Start'] = pd.to_datetime(df['Week_Start'], format='%Y-%m-%d')

    mask = (df['Week_Start'] >= start_date) & (df['Week_Start'] <= end_date)
    df_filtered = df.loc[mask]

    ax.plot(df_filtered['Week_Start'], df_filtered['ILI_Activity_Level'], color='tab:blue', label='ILI Activity Level (Start)')
    ax.scatter(df_filtered['Week_Start'], df_filtered['ILI_Activity_Level'], color='tab:blue')

    ax.set_xticks(df_filtered['Week_Start'])
    ax.set_xticklabels(df_filtered['Week_Start'].dt.strftime('%Y-%m-%d'), rotation=90)

    output_dir = 'time_series'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f"{output_dir}/{file_name}_{start_date}_{end_date}_time_series.png"
    plt.savefig(output_file)
    mask = (df['Week_Start'] >= start_date) & (df['Week_Start'] <= end_date)
    df_filtered = df.loc[mask]

    ax.plot(df_filtered['Week_Start'], df_filtered['ILI_Activity_Level'], color='tab:blue', label='ILI Activity Level (Start)')
    ax.tick_params(axis='y', labelcolor='tab:blue')

    fig.tight_layout()
    plt.legend()
    plt.title(f'Time Series for {file_name}')
    plt.show()

def process_csv_file(file_path):
    df = read_csv(file_path)
    plot_time_series(df, os.path.basename(file_path))

# def main():
#     file_path = input("Enter the CSV file path: ")
#     process_csv_file(file_path)

# if __name__ == "__main__":
#     main()

# Initialize the Earth Engine module.
ee.Initialize()

def get_temperature_data(point, start_date, end_date):
    dataset = ee.ImageCollection('MODIS/061/MOD11A1')
    temperature = dataset.select('LST_Day_1km')
    temperature = temperature.filterDate(start_date, end_date).mean()
    temperature = temperature.sample(point, 5000).first().get('LST_Day_1km').getInfo()
    return temperature

def compute_ccf(df, temperature_data, max_lag=10):
    df['Temperature'] = temperature_data
    ccf_values = [df['ILI_Activity_Level'].corr(df['Temperature'].shift(lag)) for lag in range(-max_lag, max_lag + 1)]
    best_lag = np.argmax(ccf_values) - max_lag
    return best_lag, ccf_values

def extract_coordinates(point_string):
    """Extracts longitude and latitude from a POINT string."""
    match = re.search(r"POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)", point_string)
    if match:
        longitude = float(match.group(1))
        latitude = float(match.group(2))
        return longitude, latitude
    else:
        raise ValueError("Invalid POINT format")

def process_files_in_directory(directory_path):
    results = {}
    periods = [
        ('2019-01-01', '2024-12-31'),
        ('2019-01-01', '2023-12-31'),
        ('2019-01-01', '2022-12-31'),
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

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            df = read_csv(file_path)
            
            point_string = df['ZIP_Code_Location'].iloc[0]
            longitude, latitude = extract_coordinates(point_string)
            point = ee.Geometry.Point([longitude, latitude])
            
            file_results = {}
            for (start_date, end_date), period_label in zip(periods, period_labels):
                temperature_data = get_temperature_data(point, start_date, end_date)
                best_lag, _ = compute_ccf(df, temperature_data)
                file_results[period_label] = best_lag
            
            results[file_name.replace('.csv', '')] = file_results
    
    results_df = pd.DataFrame.from_dict(results, orient='index')
    results_df.to_csv('lag_results.csv')
    print('Lag results saved to lag_results.csv')

def main():
    directory_path = input("Enter the directory path containing CSV files: ")
    process_files_in_directory(directory_path)

if __name__ == "__main__":
    main()
