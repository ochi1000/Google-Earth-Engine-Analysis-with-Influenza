import ee
import datetime
import folium

# Initialize the Earth Engine module.
ee.Initialize()

# Define the region of interest (example: New Orleans, Louisiana)
region = ee.Geometry.Point([41.886262, -87.622844]).buffer(5000)  # 5 km buffer

# Define the time period as ee.Date objects
start_date = ee.Date('2022-01-01')
end_date = ee.Date('2022-12-31')

# Load the MODIS Land Surface Temperature (Day) product
modis_lst = ee.ImageCollection("MODIS/061/MOD11A1") \
    .filterDate(start_date, end_date) \
    .filterBounds(region) \
    .select('LST_Day_1km')

# Convert LST from Kelvin (scaled by 0.02) to Celsius and rename the band to 'LST_Celsius'
def convert_to_celsius(image):
    lst_c = image.multiply(0.02).subtract(273.15).rename('LST_Celsius')
    return lst_c.copyProperties(image, ['system:time_start'])

lst_in_celsius = modis_lst.map(convert_to_celsius)

# Calculate the number of weeks between start and end dates
n_weeks = end_date.difference(start_date, 'week').round()

# Create a list of week indices (0, 1, 2, ...)
weeks = ee.List.sequence(0, n_weeks.subtract(1))

# Map over the weeks to compute weekly average temperature
def compute_weekly_average(week_index):
    week_start = start_date.advance(week_index, 'week')
    week_end = week_start.advance(1, 'week')
    
    # Filter images for the current week
    week_collection = lst_in_celsius.filterDate(week_start, week_end)
    
    # Compute the count of images in the week
    count = week_collection.size()
    
    # Compute the weekly mean image only if count > 0; otherwise, set to null
    weekly_mean_image = ee.Image(ee.Algorithms.If(count.gt(0), week_collection.mean(), None))
    
    # Reduce the weekly image over the region if it exists
    mean_dict = ee.Dictionary(
        ee.Algorithms.If(
            weekly_mean_image, 
            weekly_mean_image.reduceRegion({
                'reducer': ee.Reducer.mean(),
                'geometry': region,
                'scale': 1000,
                'maxPixels': 1e9
            }), 
            {}
        )
    )
    
    # Get the mean temperature if available; otherwise, return null
    mean_temperature = ee.Algorithms.If(mean_dict.contains('LST_Celsius'), mean_dict.get('LST_Celsius'), None)
    
    return ee.Feature(None, {
        'week_start': week_start.format('YYYY-MM-dd'),
        'mean_temperature': mean_temperature
    })

weekly_time_series = weeks.map(compute_weekly_average)

# Convert the list of features to a FeatureCollection
weekly_time_series_fc = ee.FeatureCollection(weekly_time_series)

# Print the weekly time series data to the Console
print('Weekly Time Series Data:', weekly_time_series_fc.getInfo())

# Create a chart of weekly mean temperature over time
import matplotlib.pyplot as plt

# Extract data for plotting
data = weekly_time_series_fc.getInfo()['features']
dates = [datetime.datetime.strptime(f['properties']['week_start'], '%Y-%m-%d') for f in data]
temperatures = [f['properties']['mean_temperature'] for f in data]

# Plot the data
plt.figure(figsize=(10, 5))
plt.plot(dates, temperatures, marker='o', linestyle='-', linewidth=2, markersize=4)
plt.title('Weekly Mean Land Surface Temperature (°C)')
plt.xlabel('Week Start Date')
plt.ylabel('Temperature (°C)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Display the region on the map

# Create a map centered on the region
map_center = [41.886262, -87.622844]
m = folium.Map(location=map_center, zoom_start=10)

# Add the region to the map
folium.GeoJson(region.getInfo(), name='Region of Interest').add_to(m)

# Display the map
m.save('map.html')
