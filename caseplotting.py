import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
csv_file_path = 'filtered_file.csv'
df = pd.read_csv(csv_file_path)  # Ensure ZIP_Code is read as a string

# Convert the week_start and week_end columns to datetime format, handling errors
df['week_start'] = pd.to_datetime(df['week_start'], format='%m/%d/%Y', errors='coerce')
df['week_end'] = pd.to_datetime(df['week_end'], format='%m/%d/%Y', errors='coerce')

# Drop rows where week_end or week_start is NaT (failed parsing)
df = df.dropna(subset=['week_end', 'week_start'])

# Function to plot percent for a given ZIP code and date range
def plot_percent(start_date, end_date):
    # Convert start_date and end_date to datetime
    start_date = pd.to_datetime(start_date, format='%m/%d/%Y')
    end_date = pd.to_datetime(end_date, format='%m/%d/%Y')
    
    # Filter the dataframe for the given ZIP code and date range
    df_zip = df[(df['week_end'] >= start_date) & (df['week_end'] <= end_date)]
    
    if df_zip.empty:
        print(f"No data available for the date range {start_date.strftime('%m/%d/%Y')} - {end_date.strftime('%m/%d/%Y')}")
        return
    
    # Sort the dataframe by week_start and week_end
    df_zip = df_zip.sort_values(by=['week_start', 'week_end'])
    
    # Function to plot the selected chart type
    def plot_chart(chart_type):
        plt.figure(figsize=(12, 8))
        
        if chart_type == 'bar':
            plt.bar(df_zip['week_end'], df_zip['percent']*100, color='blue', alpha=0.7)
            plt.title(f'In_Out Patient (Bar Chart) {start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}')    
        elif chart_type == 'scatter':
            plt.scatter(df_zip['week_end'], df_zip['percent']*100, color='red', marker='o')
            plt.title(f'In_Out Patient (Scatter Plot) {start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}')
        elif chart_type == 'line':
            plt.plot(df_zip['week_start'], df_zip['percent']*100, color='green', marker='o')
            plt.title(f'In_Out Patient (Line Plot) {start_date.strftime("%m/%d/%Y")} - {end_date.strftime("%m/%d/%Y")}')
        else:
            print(f"Invalid chart type: {chart_type}")
            return
        
        plt.xlabel('Week')
        plt.ylabel('In_Out Patient Percent')
        plt.ylim(0, (df_zip['percent']*100).max() + 5)
        plt.grid(True)
        
        # Set x-axis to show weekly intervals
        plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b %d'))
        plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.WeekdayLocator())
        
        # Rotate x-axis labels to be vertical
        plt.xticks(rotation=90)
        
        plt.tight_layout()
        plt.show()
    
    return plot_chart

# Example usage
start_date = '01/01/2023'  # Replace with the desired start date
end_date = '12/31/2023'  # Replace with the desired end date
plot_chart = plot_percent(start_date, end_date)

# Choose the chart type: 'bar', 'scatter', or 'line'
chart_type = 'line'  # Replace with the desired chart type
plot_chart(chart_type)
