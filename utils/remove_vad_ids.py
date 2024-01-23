import argparse
import pandas as pd

def filter_json_lines(txt_file, jsonl_file, output_file):
    # Read IDs from the txt file into a set for faster lookup
    with open(txt_file, 'r') as file:
        ids_to_remove = {line.strip() for line in file}

    # Read the json-lines file into a pandas DataFrame
    df = pd.read_json(jsonl_file, lines=True)

    # Filter the DataFrame to exclude rows with IDs in the txt file
    filtered_df = df[~df['id'].isin(ids_to_remove)]

    # Write the filtered data back to a new json-lines file
    filtered_df.to_json(output_file, orient='records', lines=True)

def main():
    parser = argparse.ArgumentParser(description='Filter JSON Lines file based on IDs from a text file.')
    parser.add_argument('--txt_file', required=True, help='Path to the text file containing IDs.')
    parser.add_argument('--jsonl_file', required=True, help='Path to the JSON Lines file to be filtered.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON Lines file.')

    args = parser.parse_args()

    filter_json_lines(args.txt_file, args.jsonl_file, args.output_file)

if __name__ == '__main__':
    main()

