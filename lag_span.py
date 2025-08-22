import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import ee

# Initialize the Earth Engine API.
ee.Initialize()

# --- Adjustable Analysis Period ---
analysis_start = pd.to_datetime('2015-10-01')  # Start of analysis period
analysis_end   = pd.to_datetime('2020-04-01')    # End of analysis period

# --- 1. Load and Preprocess the CSV Data ---
# CSV should have columns: 'week_start', 'week_end', and 'percent'
file_path = 'incident_data_with_temperature.csv'  # Update with your CSV file path
data = pd.read_csv(file_path)

# Convert week_start and week_end columns to datetime objects.
data['week_start'] = pd.to_datetime(data['week_start'], format='%m/%d/%Y')
data['week_end']   = pd.to_datetime(data['week_end'], format='%m/%d/%Y')

# Filter data based on the user-defined analysis period.
data = data[(data['week_start'] >= analysis_start) & (data['week_start'] <= analysis_end)]
data = data.sort_values('week_start').reset_index(drop=True)

# --- 2. Function to Fetch Surface Temperature from GEE ---
def fetch_surface_temperature(start_date, end_date):
    collection = ee.ImageCollection("MODIS/061/MOD11A1") \
                    .filterDate(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')) \
                    .select('LST_Day_1km')
    
    # Check if the collection is empty.
    if int(collection.size().getInfo()) == 0:
        return None
    
    # Compute the mean image and convert from Kelvin (scale 0.02) to Celsius.
    mean_temp = collection.mean().multiply(0.02).subtract(273.15)
    
    # Define a point geometry (adjust location as needed).
    point = ee.Geometry.Point([-87.623177, 41.881832])
    
    result = mean_temp.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=1000
    ).getInfo()
    
    return result.get('LST_Day_1km', None)

# --- 3. Fetch Temperature Data for Each Week ---
surface_temps = []
for idx, row in data.iterrows():
    temp = fetch_surface_temperature(row['week_start'], row['week_end'])
    surface_temps.append(temp)
data['Temperature'] = surface_temps

# Drop rows with missing temperature data.
data = data.dropna(subset=['Temperature']).reset_index(drop=True)

if data.empty:
    print("No valid temperature data available in the selected period.")
    exit()

# --- 4. Calculate Week-to-Week Differences ---
data['temp_diff'] = data['Temperature'].diff()
data['incident_diff'] = data['percent'].diff()

# Filter to keep only rows with a temperature drop (temp_diff < 0)
# and an increase in incidents (incident_diff > 0)
filtered_data = data[(data['temp_diff'] < 0) & (data['incident_diff'] > 0)].copy()
filtered_data = filtered_data.dropna(subset=['temp_diff', 'incident_diff']).reset_index(drop=True)

if filtered_data.empty:
    print("No data found where temperature drops coincide with incident increases in the selected period.")
    exit()

# --- 5. Cross-Correlation Analysis ---
max_lag = 8  # Maximum lag (in weeks) to consider.
lags = list(range(-max_lag, max_lag + 1))
ccf_values = []

# Calculate cross-correlation for each lag.
for lag in lags:
    if lag > 0:
        x = filtered_data['temp_diff'].shift(lag).dropna()
        y = filtered_data['incident_diff'].iloc[lag:]
    elif lag < 0:
        x = filtered_data['temp_diff'][:lag]
        y = filtered_data['incident_diff'].shift(-lag)[:lag]
    else:
        x = filtered_data['temp_diff']
        y = filtered_data['incident_diff']
    
    # Align lengths.
    min_len = min(len(x), len(y))
    if min_len > 0:
        x = x.iloc[:min_len]
        y = y.iloc[:min_len]
        corr = np.corrcoef(x, y)[0, 1]
    else:
        corr = np.nan
    ccf_values.append(corr)

# Determine the best lag (highest absolute correlation).
best_index = np.nanargmax(np.abs(ccf_values))
best_lag = lags[best_index]
print(f"Best lag (weeks): {best_lag}")
print(f"Cross-correlation at best lag: {ccf_values[best_index]}")

# Plot the cross-correlation function.
plt.figure(figsize=(8,6))
plt.stem(lags, ccf_values, basefmt=" ")
plt.xlabel("Lag (weeks)")
plt.ylabel("Cross-correlation")
plt.title("Cross-correlation between Temperature Drop and Incident Rise")
plt.show()

# --- 6. Regression Analysis with Lagged Variables ---
# Create lagged temperature difference variables (for lags 1 to 4 weeks).
for lag in range(1, 5):
    filtered_data[f'temp_diff_lag{lag}'] = filtered_data['temp_diff'].shift(lag)

# Drop rows with missing lag values.
reg_data = filtered_data.dropna(subset=[f'temp_diff_lag{lag}' for lag in range(1, 5)]).copy()

# Define predictors (e.g., using lag2 and lag3) and outcome.
X = reg_data[['temp_diff_lag2', 'temp_diff_lag3']]
y = reg_data['incident_diff']

# Add a constant for the intercept.
X = sm.add_constant(X)
model = sm.OLS(y, X).fit()
print(model.summary())

# --- 7. Optional: Plot the Time Series for Context ---
plt.figure(figsize=(12,6))
plt.plot(data['week_start'], data['Temperature'], label='Surface Temperature (°C)', marker='o', color='red')
plt.plot(data['week_start'], data['percent'], label='Incident %', marker='x', color='blue')
plt.xlabel("Week Start")
plt.ylabel("Value")
plt.title(f"Flu Incident % and Surface Temperature\n({analysis_start.date()} to {analysis_end.date()})")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
