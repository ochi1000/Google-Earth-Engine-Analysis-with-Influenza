import pandas as pd
import numpy as np
# from datetime import datetime
# import matplotlib.pyplot as plt
import ee

# Initialize the Earth Engine API
ee.Initialize()

# Load the CSV file
file_path = 'filtered_file.csv'  # Replace with the path to your CSV file
data = pd.read_csv(file_path)

# Convert week_start and week_end to datetime
data['week_start'] = pd.to_datetime(data['week_start'], format='%m/%d/%Y')
data['week_end'] = pd.to_datetime(data['week_end'], format='%m/%d/%Y')

# Extract the year from the week_start column
data['year'] = data['week_start'].dt.year

# Group data by year
grouped = data.groupby('year')

# Function to fetch surface temperature from GEE
# def fetch_surface_temperature(start_date, end_date):
#     collection = ee.ImageCollection("MODIS/061/MOD11A1") \
#         .filterDate(start_date, end_date) \
#         .select('LST_Day_1km')  # Surface temperature (daytime)
#     mean_temp = collection.mean().multiply(0.02).subtract(273.15)  # Convert to Celsius
#     return mean_temp.reduceRegion(
#         reducer=ee.Reducer.mean(),
#         geometry=ee.Geometry.Point([-87.623177, 41.881832]),  # Replace with your location
#         scale=1000
#     ).getInfo().get('LST_Day_1km')

def fetch_surface_temperature(start_date, end_date):
    # Fetch the MODIS image collection for the given date range and select the LST band.
    var_collection = ee.ImageCollection("MODIS/061/MOD11A1") \
        .filterDate(start_date, end_date) \
        .select('LST_Day_1km')
    
    # Check if the collection is empty.
    if int(var_collection.size().getInfo()) == 0:
        print(f"No MODIS images available between {start_date} and {end_date}")
        return None  # or return a default value, e.g., np.nan
    
    # Otherwise, compute the mean image.
    var_mean_img = var_collection.mean()
    
    # Process the image: convert from scale factor 0.02 and Kelvin to Celsius.
    var_processed = var_mean_img.multiply(0.02).subtract(273.15)
    
    # Reduce the image over the specified point (your location).
    var_result = var_processed.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=ee.Geometry.Point([-87.623177, 41.881832]),  # Replace with your location if needed.
        scale=1000
    )
    
    # Try to get the value of LST_Day_1km from the result.
    var_temp = var_result.get('LST_Day_1km')
    if var_temp is None:
        print(f"Temperature value is None for date range {start_date} to {end_date}")
        return None
    
    return var_temp.getInfo()


# Function to calculate cross-correlation and covariance
def calculate_lag(year, df):
    percent = df['percent'].values
    surface_temp = []

    # Fetch surface temperature for each week
    for _, row in df.iterrows():
        temp = fetch_surface_temperature(row['week_start'], row['week_end'])
        surface_temp.append(temp)

    surface_temp = np.array(surface_temp, dtype=np.float64)  # Ensure numeric type

    # Ensure both arrays have the same length
    if len(percent) != len(surface_temp):
        raise ValueError(f"Data mismatch in year {year}: 'percent' and 'surface_temperature' must have the same length.")

    # Handle cases where surface_temp has None or no values
    valid_indices = [i for i, temp in enumerate(surface_temp) if temp is not None]
    if len(valid_indices) == 0:
        return None, None  # Skip if no valid surface temperature data is available

    # Filter percent and surface_temp to only include valid indices
    percent = np.array(percent)[valid_indices]
    surface_temp = np.array(surface_temp)[valid_indices]

    # Cross-correlation
    cross_corr = np.correlate(percent - np.mean(percent), surface_temp - np.mean(surface_temp), mode='full')
    lag_cross_corr = np.argmax(cross_corr) - (len(percent) - 1)

    # Covariance
    covariance = np.cov(np.array(percent).reshape(-1), np.array(surface_temp).reshape(-1))[0, 1]

    return lag_cross_corr, covariance

# Iterate through each year and calculate lag
results = []
for year, group in grouped:
    lag_cross_corr, covariance = calculate_lag(year, group)
    results.append({'year': year, 'cross_correlation_lag': lag_cross_corr, 'covariance_lag': covariance})

# Convert results to a DataFrame
results_df = pd.DataFrame(results)

# Save results to a CSV file
results_df.to_csv('lag_results.csv', index=False)

# Print results
print(results_df)