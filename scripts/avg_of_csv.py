import sys
import pandas as pd

def calculate_column_averages(file_path):
    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(file_path)
        
        # Calculate the averages of numeric columns
        averages = df.mean(numeric_only=True)
        
        print("Column Averages:")
        print(averages)
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_csv>")
    else:
        csv_path = sys.argv[1]
        calculate_column_averages(csv_path)
