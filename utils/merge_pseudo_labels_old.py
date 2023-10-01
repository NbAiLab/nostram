import argparse
import os
import glob
import pandas as pd
from tqdm import tqdm

def simplify_column_names(df, should_simplify=True):
    if not should_simplify:
        return df
    
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

    for col in df.columns:
        if col in mapping:
            df.rename(columns={col: mapping[col]}, inplace=True)
            
    return df

def extract_model_name(filepath):
    # Extract the part after the second '-' and until the '.'
    model_name = filepath.split('-', 2)[-1].rsplit('.', 1)[0]
    # Get parent folder name
    parent_folder = os.path.basename(os.path.dirname(filepath))
    # Combine parent folder name and model name
    return f"{parent_folder}-{model_name}"

def main(args):
    print(f"Starting to process {args.input_json}.")
    # Read the main JSON DataFrame
    df_json = pd.read_json(args.input_json, lines=True, dtype={'id': str})

    
    # Initialize stats
    added_columns = []
    
    # Loop through all TSV files in the input directory and its subdirectories
    for tsv_file in tqdm(glob.glob(os.path.join(args.input_tsv_dir, "**/*.tsv"), recursive=True)):
        
        # Read the current TSV DataFrame
        df_tsv = pd.read_csv(tsv_file, sep='\t')
        df_tsv['file_id'] = df_tsv['file_id'].astype(str)  # Convert to string
        second_col = df_tsv.columns[1]
        
        model_name = extract_model_name(tsv_file)
        
        # Rename the column to match the model_name
        df_tsv.rename(columns={second_col: model_name}, inplace=True)
        
        # Check if the column exists, if not create it
        if model_name not in df_json.columns:
            # print(f"Creating column {model_name}")
            df_json[model_name] = pd.Series(dtype="object")
            added_columns.append(model_name)
            
        # Merge DataFrames
        df_merged = pd.merge(df_json, df_tsv[['file_id', model_name]], left_on='id', right_on='file_id', how='left', suffixes=("", "_new"))
        
        # Update the original DataFrame
        df_merged[model_name] = df_merged.apply(lambda row: row[model_name + '_new'] if pd.notna(row[model_name + '_new']) else row[model_name], axis=1)
        
        # Drop the new column and 'file_id'
        df_merged.drop(columns=[model_name + '_new', 'file_id'], inplace=True)
        
        # Update the main DataFrame
        df_json = df_merged
        

    # Simplify column names
    df_json = simplify_column_names(df_json, not args.do_not_simplify_columns)

    # Write the final JSON file
    df_json.to_json(args.output_json, orient='records', lines=True)
    
    # Print statistics
    print(f"Total rows in the output file: {len(df_json)}")
    for col in added_columns:
        print(f"{col}: {df_json[col].count()} filled values")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON and TSV files")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--input_tsv_dir", type=str, required=True, help="Directory containing input TSV files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")
    parser.add_argument("--do_not_simplify_columns", action='store_true', help="Flag to prevent column name simplification")

    
    args = parser.parse_args()
    main(args)