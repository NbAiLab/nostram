import argparse
import pandas as pd
import json
import os

def deduplicate_jsonlines(input_folder, output_file):
    # Initialize an empty DataFrame
    df_combined = pd.DataFrame()

    # List all jsonl files in the input folder
    input_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jsonl')]

    # Process each file
    for file_name in input_files:
        # Read the current json lines file into a DataFrame
        df = pd.read_json(file_name, lines=True,dtype={'id': str, 'group_id': str, 'text': str, 'text_en': str, 'text_nn': str, 'text_language': str, 'task': str, 'timestamped_text_en': str, 'timestamped_text_nn': str, 'wer_no': float, 'wer_nn': float})
        # Append to the combined DataFrame
        df_combined = pd.concat([df_combined, df], ignore_index=True)
    
    # Drop duplicates based on 'id' and 'task'
    df_combined.drop_duplicates(subset=['id', 'task'], inplace=True)
    
    # Write the deduplicated data to a json lines file
    df_combined.to_json(output_file, orient='records', lines=True)

def main():
    parser = argparse.ArgumentParser(description="Deduplicate JSON lines files within a folder based on 'id' and 'task'.")
    parser.add_argument('--input_folder', help='Input folder containing JSON lines files', required=True)
    parser.add_argument('--output_file', help='Output JSON lines file', required=True)
    
    args = parser.parse_args()
    
    deduplicate_jsonlines(args.input_folder, args.output_file)

if __name__ == "__main__":
    main()
