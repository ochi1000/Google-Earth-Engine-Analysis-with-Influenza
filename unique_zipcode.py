import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'Influenza_Risk_Level_by_ZIP_Code_20250304.csv'
df = pd.read_csv(csv_file_path, dtype={'ZIP_Code': str})  # Ensure ZIP_Code is read as a string

# Convert the Week_Start and Week_End columns to datetime format, handling errors
df['Week_Start'] = pd.to_datetime(df['Week_Start'], format='%m/%d/%Y', errors='coerce')
df['Week_End'] = pd.to_datetime(df['Week_End'], format='%m/%d/%Y', errors='coerce')

# Drop rows where Week_End or Week_Start is NaT (failed parsing)
df = df.dropna(subset=['Week_End', 'Week_Start'])

# Count the number of unique ZIP codes
unique_zipcodes_count = df['ZIP_Code'].nunique()
print(f"Number of unique ZIP codes: {unique_zipcodes_count}")

# # Function to plot ILI_Activity_Level for a given ZIP code and date range
# def plot_ili_activity_level(zip_code, start_date, end_date):
#     # Convert start_date and end_date to datetime
#     start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
#     end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
    
#     # Filter the dataframe for the given ZIP code and date range
#     df_zip = df[(df['ZIP_Code'] == zip_code) & (df['Week_End'] >= start_date) & (df['Week_End'] <= end_date)]
    
#     if df_zip.empty:
#         print(f"No data available for ZIP Code: {zip_code} in the date range {start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}")
#         return
    
#     # Sort the dataframe by Week_Start and Week_End
#     df_zip = df_zip.sort_values(by=['Week_Start', 'Week_End'])
    
#     # Function to plot the selected chart type
#     def plot_chart(chart_type):
#         plt.figure(figsize=(12, 8))
        
#         if chart_type == 'bar':
#             plt.bar(df_zip['Week_End'], df_zip['ILI_Activity_Level'], color='blue', alpha=0.7)
#             plt.title(f'ILI Activity Level for ZIP Code {zip_code} (Bar Chart) {start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}')    
#         elif chart_type == 'scatter':
#             plt.scatter(df_zip['Week_End'], df_zip['ILI_Activity_Level'], color='red', marker='o')
#             plt.title(f'ILI Activity Level for ZIP Code {zip_code} (Scatter Plot) {start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}')
#         elif chart_type == 'line':
#             plt.plot(df_zip['Week_End'], df_zip['ILI_Activity_Level'], color='green', marker='o')
#             plt.title(f'ILI Activity Level for ZIP Code {zip_code} (Line Plot) {start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}')
#         else:
#             print(f"Invalid chart type: {chart_type}")
#             return
        
#         plt.xlabel('Week')
#         plt.ylabel('ILI Activity Level')
#         plt.ylim(0, 12)
#         plt.grid(True)
        
#         # Set x-axis to show weekly intervals
#         plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
#         plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator())
        
#         # Rotate x-axis labels to be vertical
#         plt.xticks(rotation=90)
        
#         plt.tight_layout()
#         plt.show()
    
#     return plot_chart

# # Example usage
# zip_code_to_plot = '60652'  # Replace with the desired ZIP code
# start_date = '01/01/2022'  # Replace with the desired start date
# end_date = '12/31/2022'  # Replace with the desired end date
# plot_chart = plot_ili_activity_level(zip_code_to_plot, start_date, end_date)

# # Choose the chart type: 'bar', 'scatter', or 'line'
# chart_type = 'line'  # Replace with the desired chart type
# plot_chart(chart_type)
