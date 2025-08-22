import pandas as pd
import os

# Load the CSV file
input_file = 'Influenza_Risk_Level_by_ZIP_Code_20250304.csv'
df = pd.read_csv(input_file)

# Ensure the Week_Start column is in datetime format
df['Week_Start'] = pd.to_datetime(df['Week_Start'], format='%m/%d/%Y')

# Get unique zip codes
unique_zip_codes = df['ZIP_Code'].unique()

# Create output directory if it doesn't exist
output_dir = 'output_zip_codes'
os.makedirs(output_dir, exist_ok=True)

# Process each unique zip code
for zip_code in unique_zip_codes:
    # Filter records for the current zip code
    zip_code_df = df[df['ZIP_Code'] == zip_code]
    
    # Sort records by Week_Start in ascending order
    zip_code_df = zip_code_df.sort_values(by='Week_Start')
    
    # Save to a new CSV file
    output_file = os.path.join(output_dir, f'{zip_code}.csv')
    zip_code_df.to_csv(output_file, index=False)