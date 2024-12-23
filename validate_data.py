import pandas as pd

def main():
    # Ask the user for the CSV file path
    csv_file = input("Enter the path to the CSV file: ")

    try:
        # Load the CSV file into a DataFrame
        df = pd.read_csv(csv_file, header=None)

        # Get the number of rows and columns
        num_rows, num_cols = df.shape

        # Display the results
        print(f"Shape of {csv_file}:")
        print(f"Number of rows: {num_rows}")
        print(f"Number of columns: {num_cols}")

    except FileNotFoundError:
        print(f"Error: The file '{csv_file}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{csv_file}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: The file '{csv_file}' could not be parsed. Please ensure it's a valid CSV.")

if __name__ == "__main__":
    main()