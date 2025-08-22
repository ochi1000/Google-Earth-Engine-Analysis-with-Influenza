import csv
from datetime import datetime

def process_csv(file_path, output_file):
    filtered_data = []
    
    # Read the CSV file
    with open(file_path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row.get('respiratory_category') == "Influenza" and row.get('demographic_category') == "ALL":
                filtered_data.append(row)
    
    # Sort the filtered data by week_start
    filtered_data.sort(key=lambda x: datetime.strptime(x['week_start'], '%m/%d/%Y'))
    
    # Write the sorted data to a new CSV file
    with open(output_file, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=filtered_data[0].keys())
        writer.writeheader()
        writer.writerows(filtered_data)

if __name__ == "__main__":
    input_file = 'Inpatient__Emergency_Department__and_Outpatient_Visits_for_Respiratory_Illnesses_20250313.csv'  # Replace with your input file path
    output_file = 'output_Inpatient__Emergency_Department__and_Outpatient_Visits_for_Respiratory_Illnesses_20250313.csv'  # Replace with your desired output file path
    process_csv(input_file, output_file)
    print(f"Filtered and sorted data has been written to {output_file}")
