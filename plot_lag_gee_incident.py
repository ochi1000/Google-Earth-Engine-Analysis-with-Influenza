import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Initialize the Earth Engine API.
ee.Initialize()

# Adjustable analysis time period (change these as needed)
analysis_start = pd.to_datetime('2015-01-01')  # Start of analysis period
analysis_end   = pd.to_datetime('2019-12-31')    # End of analysis period

# Load the CSV file with columns: week_start, week_end, and percent.
file_path = 'filtered_file.csv'  # Update with your CSV file path.
data = pd.read_csv(file_path)

# Convert week_start and week_end columns to datetime objects.
data['week_start'] = pd.to_datetime(data['week_start'], format='%m/%d/%Y')
data['week_end'] = pd.to_datetime(data['week_end'], format='%m/%d/%Y')

# Filter data by the adjustable analysis time period.
data = data[(data['week_start'] >= analysis_start) & (data['week_start'] <= analysis_end)]

# Function to fetch surface temperature from GEE for a given date range.
def fetch_surface_temperature(start_date, end_date):
    # Create an image collection for the specified period.
    collection = ee.ImageCollection("MODIS/061/MOD11A1") \
        .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
        .select('LST_Day_1km')
    
    # Check if the collection is empty.
    if int(collection.size().getInfo()) == 0:
        return None
    
    # Compute the mean image and convert from scaled Kelvin to Celsius.
    mean_temp = collection.mean().multiply(0.02).subtract(273.15)
    
    # Define a point geometry at the location of interest (update coordinates as needed).
    point = ee.Geometry.Point([-87.623177, 41.881832])
    
    # Reduce the image over the point to get a mean temperature value.
    result = mean_temp.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()
    
    return result.get('LST_Day_1km', None)

# For each row in the CSV, fetch the corresponding surface temperature.
surface_temps = []
for idx, row in data.iterrows():
    temp = fetch_surface_temperature(row['week_start'], row['week_end'])
    surface_temps.append(temp)

# Add the temperature data to the DataFrame.
data['Temperature'] = surface_temps

# Drop rows with missing temperature data.
data = data.dropna(subset=['Temperature']).reset_index(drop=True)

# (Optional) Print the data to inspect.
print(data.head())

# Plot the time series with dual y-axes:
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot surface temperature on the left y-axis.
color_temp = 'red'
ax1.set_xlabel("Week Start")
ax1.set_ylabel("Surface Temperature (°C)", color=color_temp)
ax1.plot(data['week_start'], data['Temperature'], marker='o', color=color_temp, label='Surface Temperature')
ax1.tick_params(axis='y', labelcolor=color_temp)

# Create a second y-axis for the incident percent.
ax2 = ax1.twinx()
color_percent = 'blue'
ax2.set_ylabel("Incident %", color=color_percent)
ax2.plot(data['week_start'], data['percent'], marker='x', color=color_percent, label='Incident %')
ax2.tick_params(axis='y', labelcolor=color_percent)

# Title and grid
plt.title(f"Flu Incident % and Surface Temperature\n({analysis_start.date()} to {analysis_end.date()})")
fig.tight_layout()
plt.grid(True)

plt.show()
