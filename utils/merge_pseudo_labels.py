import argparse
import os
import glob
import pandas as pd
from tqdm import tqdm

# Define the mapping of verbose column names to their simplified versions
mapping = {
    'ModelA-NbAiLab-nb-whisper-medium-fine4-npsc-norm-nohes-Norwegian-transcribe-train-transcription': 'A-no-nohes',
    'ModelB-NbAiLab-nb-whisper-medium-fine3-npsc-norm-raw-Norwegian-transcribe-train-transcription': 'B-no-hes',
    'ModelC-NbAiLab-nb-whisper-medium-beta-Norwegian-transcribe-train-transcription': 'C-no-medium',
    'ModelD-NbAiLab-nb-whisper-medium-fine4-npsc-norm-nohes-Nynorsk-transcribe-train-transcription': 'D-nn-nohes',
    'ModelE-NbAiLab-nb-whisper-medium-fine3-npsc-norm-raw-Nynorsk-transcribe-train-transcription': 'E-nn-hes',
    'ModelF-NbAiLab-nb-whisper-medium-beta-Nynorsk-transcribe-train-transcription': 'F-nn-medium',
    'ModelG-NbAiLab-nb-whisper-medium-beta-Norwegian-transcribe-train-transcription': 'G-no-timestamp',
    'ModelH-openai-whisper-medium-English-translate-train-transcription': 'H-en-timestamp'
}

def simplify_column_names(df):
    """
    Simplify the column names of a DataFrame based on a pre-defined mapping.
    """
    for col in df.columns:
        if col in mapping:
            df.rename(columns={col: mapping[col]}, inplace=True)
    return df

def extract_model_name(filepath):
    """
    Extract model name from the filepath.
    """
    model_name = filepath.split('-', 2)[-1].rsplit('.', 1)[0]
    parent_folder = os.path.basename(os.path.dirname(filepath))
    return f"{parent_folder}-{model_name}"

def main(args):
    """
    Main function to read JSON and TSV files, merge them, and save to a new JSON file.
    """
    # Read the main JSON DataFrame
    df_json = pd.read_json(args.input_json, lines=True, dtype={'id': str})

    # Initialize a list to store added column names
    added_columns = []

    # Loop through all TSV files in the input directory and its subdirectories
    for tsv_file in tqdm(glob.glob(os.path.join(args.input_tsv_dir, "**/*.tsv"), recursive=True)):
        
        # Read the current TSV DataFrame
        df_tsv = pd.read_csv(tsv_file, sep='\t')
        df_tsv['file_id'] = df_tsv['file_id'].astype(str)
        
        # Extract model name from the filepath
        model_name = extract_model_name(tsv_file)
        
        # Rename the second column to match the extracted model_name
        df_tsv.rename(columns={df_tsv.columns[1]: model_name}, inplace=True)
        
        # Add a new column to the JSON DataFrame if it doesn't exist
        if model_name not in df_json.columns:
            df_json[model_name] = pd.Series(dtype="object")
            added_columns.append(model_name)
        
        # Merge JSON and TSV DataFrames
        df_merged = pd.merge(df_json, df_tsv[['file_id', model_name]], left_on='id', right_on='file_id', how='left', suffixes=("", "_new"))
        
        # Update values in the main DataFrame
        df_merged[model_name] = df_merged.apply(lambda row: row[model_name + '_new'] if pd.notna(row[model_name + '_new']) else row[model_name], axis=1)
        
        # Drop redundant columns
        df_merged.drop(columns=[model_name + '_new', 'file_id'], inplace=True)
        
        # Update the main DataFrame
        df_json = df_merged
    
    # Simplify column names if the flag is not set
    if not args.do_not_simplify_columns:
        simplify_column_names(df_json)
    
    # Save the merged DataFrame to a new JSON file
    df_json.to_json(args.output_json, orient='records', lines=True)

    # Print some statistics
    print(f"Total rows in the output file: {len(df_json)}")
    for col in added_columns:
        simplified_col = col if args.do_not_simplify_columns else mapping.get(col, col)
        print(f"{simplified_col}: {df_json[simplified_col].count()} non-NaN values")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON and TSV files")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--input_tsv_dir", type=str, required=True, help="Directory containing input TSV files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--do_not_simplify_columns", action='store_true', help="Flag to prevent column name simplification")

    args = parser.parse_args()
    main(args)
