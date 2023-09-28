import pandas as pd
import os
import argparse
import sys

def main(input_directory, output_file):
    # List all files in the directory with *.tsv or *.txt extension
    files = [f for f in os.listdir(input_directory) if os.path.isfile(os.path.join(input_directory, f)) and (f.endswith('.tsv') or f.endswith('.txt'))]

    # Initialize an empty DataFrame
    merged_df = None

    for file in files:
        file_path = os.path.join(input_directory, file)
        
        # Check if the file uses \t as separator
        with open(file_path, 'r') as f:
            if '\t' not in f.readline():
                print(f"Error: File {file_path} does not use '\\t' as a separator.")
                sys.exit(1)

        # Read the file as TSV
        df = pd.read_csv(file_path, sep='\t')

        # Extract model name from the header
        model_name = df.columns[2]

        # Rename the column with model name
        df.rename(columns={model_name: model_name}, inplace=True)

        if merged_df is None:
            merged_df = df
        else:
            merged_df = pd.merge(merged_df, df, on=['id', 'target'], how='outer')

    # Save merged DataFrame to output_file in TSV format
    merged_df.to_csv(output_file, sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge TSV/TXT files from directory")
    parser.add_argument("input_directory", type=str, help="Input directory containing the files")
    parser.add_argument("output_file", type=str, help="Output file in TSV format")

    args = parser.parse_args()

    main(args.input_directory, args.output_file)

