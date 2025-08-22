import pandas as pd

def get_unique_values(file_path, column_name):
    df = pd.read_json(file_path)
    unique_values = df[column_name].value_counts()
    return unique_values

def filter_records(file_path, filters):
    df = pd.read_json(file_path)
    query = ' & '.join([f"{col} == '{val}'" for col, val in filters.items()])
    filtered_df = df.query(query)
    return filtered_df

if __name__ == "__main__":
    file_path = input("Enter the path to the JSON file: ")

    # Task 1
    column_name = input("Enter the column name to get unique values: ")
    unique_values = get_unique_values(file_path, column_name)
    print(f"Unique values and their counts for column '{column_name}':\n{unique_values}")

    # Task 2
    filters = {}
    while True:
        col = input("Enter column name for filtering (or 'done' to finish): ")
        if col.lower() == 'done':
            break
        val = input(f"Enter value for column '{col}': ")
        filters[col] = val

    record = filter_records(file_path, filters)
    print(f"Number of records matching the filters: {len(record)}")
    print(f"{record}")