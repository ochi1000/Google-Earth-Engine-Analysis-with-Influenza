import ee
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Initialize the Earth Engine module.
ee.Initialize()

def extract_coordinates(point_string):
    """Extracts longitude and latitude from a POINT string."""
    match = re.search(r"POINT \((-?\d+\.\d+) (-?\d+\.\d+)\)", point_string)
    if match:
        longitude = float(match.group(1))
        latitude = float(match.group(2))
        return longitude, latitude
    else:
        raise ValueError("Invalid POINT format")

def read_csv(file_path):
    df = pd.read_csv(file_path)
    df['longitude'], df['latitude'] = zip(*df['ZIP_Code_Location'].apply(extract_coordinates))
    return df

def get_gee_data(start_date, end_date, lon, lat, data_type):
    point = ee.Geometry.Point([lon, lat])
    
    if data_type == 'surface_temperature':
        dataset = ee.ImageCollection('MODIS/061/MOD11A1').filterDate(start_date, end_date).select('LST_Day_1km')
    elif data_type == 'wind_speed':
        dataset = ee.ImageCollection('ECMWF/ERA5/DAILY').filterDate(start_date, end_date).select('u10')
    elif data_type == 'humidity':
        dataset = ee.ImageCollection('ECMWF/ERA5/DAILY').filterDate(start_date, end_date).select('q')
    else:
        raise ValueError("Invalid data type")
    
    data = dataset.mean().reduceRegion(ee.Reducer.mean(), point, 1000).getInfo()
    return data

def plot_time_series(df, gee_data, data_type, file_name):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

    ax1.set_ylabel('ILI Activity Level', color='tab:blue')
    ax1.plot(df['Week_Start'], df['ILI_Activity_Level'], color='tab:blue', label='ILI Activity Level')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')
    ax1.set_title(f'Time Series for {file_name}')

    ax2.set_ylabel(f'{data_type.replace("_", " ").title()}', color='tab:red')
    ax2.plot(df['Week_Start'], gee_data, color='tab:red', label=data_type.replace("_", " ").title())
    ax2.tick_params(axis='y', labelcolor='tab:red')
    ax2.legend(loc='upper left')

    # Set the min and max range for plotting
    ax2.set_ylim(min(gee_data), max(gee_data))

    fig.tight_layout()
    plt.xlabel('Date')
    plt.show()

def main():
    directory_path = input("Enter the directory path containing the CSV files: ")
    data_type = input("Enter the data type to plot (surface_temperature, wind_speed, humidity): ")
    start_date = input("Enter the start date (YYYY-MM-DD): ")
    end_date = input("Enter the end date (YYYY-MM-DD): ")
    
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            df = read_csv(file_path)
            
            gee_data = []
            
            for index, row in df.iterrows():
                data = get_gee_data(start_date, end_date, row['longitude'], row['latitude'], data_type)
                if data_type == 'surface_temperature':
                    gee_data.append(data['LST_Day_1km'])
                elif data_type == 'wind_speed':
                    gee_data.append(data['u10'])
                elif data_type == 'humidity':
                    gee_data.append(data['q'])
            
            plot_time_series(df, gee_data, data_type, file_name)

if __name__ == "__main__":
    main()
