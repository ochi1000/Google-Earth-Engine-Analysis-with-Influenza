import pandas as pd

# Load the CSV file
csv_file_path = 'Influenza_Risk_Level_by_ZIP_Code.csv'
df = pd.read_csv(csv_file_path, dtype={'ZIP_Code': str})  # Ensure ZIP_Code is read as a string

# Convert the Week_End column to datetime format, handling errors
df['Week_End'] = pd.to_datetime(df['Week_End'], format='%m/%d/%Y', errors='coerce')

# Drop rows where Week_End is NaT (failed parsing)
df = df.dropna(subset=['Week_End'])

# Extract the year from the Week_End column
df['Year'] = df['Week_End'].dt.year

# Group by ZIP_Code and get unique years
unique_years_per_zip = df.groupby('ZIP_Code')['Year'].unique().apply(list).to_dict()

# Print the result to the console
for zip_code, years in unique_years_per_zip.items():
    print(f"ZIP Code: {zip_code}, Years: {years}")

# Find and display the record where ZIP_Code is '60601' and Week_End year is 2025
zip_code_to_check = '60601'  # Ensure it's a string
record = df[(df['ZIP_Code'] == zip_code_to_check) & (df['Year'] == 2025)]

if not record.empty:
    print("\nRecord where ZIP_Code is 60601 and Week_End year is 2025:")
    print(record)
else:
    print("\nNo record found where ZIP_Code is 60601 and Week_End year is 2025.")
