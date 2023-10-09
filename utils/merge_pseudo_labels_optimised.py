import argparse
import os
import glob
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

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

def read_tsv(tsv_file):
    df_tsv = pd.read_csv(tsv_file, sep='\t')
    df_tsv['file_id'] = df_tsv['file_id'].astype(str)
    model_name = extract_model_name(tsv_file)
    df_tsv.rename(columns={df_tsv.columns[1]: model_name}, inplace=True)
    return df_tsv

def extract_model_name(filepath):
    model_name = filepath.split('-', 2)[-1].rsplit('.', 1)[0]
    parent_folder = os.path.basename(os.path.dirname(filepath))
    return f"{parent_folder}-{model_name}"

def simplify_column_names(df):
    for col in df.columns:
        if col in mapping:
            df.rename(columns={col: mapping[col]}, inplace=True)
    return df

def main(args):
    print(f"Starting to process {args.input_json}.")
    df_json = pd.read_json(args.input_json, lines=True, dtype={'id': str})

    # Create a dictionary to store the mapping from file_id to model_name
    update_dict = {}

    for tsv_file in tqdm(glob.glob(os.path.join(args.input_tsv_dir, "**/*.tsv"), recursive=True)):
        df_tsv = pd.read_csv(tsv_file, sep='\t', dtype={'file_id': str})
        model_name = extract_model_name(tsv_file)
        
        for i, row in df_tsv.iterrows():
            file_id = row['file_id']
            if file_id not in update_dict:
                update_dict[file_id] = {}
            update_dict[file_id][model_name] = row.iloc[1]

    # Update df_json using update_dict
    # Create a dictionary to hold the index positions for each file_id
    index_dict = {row['id']: index for index, row in df_json.iterrows()}
    
    # Update df_json using update_dict
    for file_id, updates in update_dict.items():
        index = index_dict.get(file_id)
        if index is not None:
            for model_name, value in updates.items():
                df_json.at[index, model_name] = value

    df_json = simplify_column_names(df_json)

    df_json.to_json(args.output_json, orient='records', lines=True)

    print(f"Total rows in the output file: {len(df_json)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge JSON and TSV files")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file")
    parser.add_argument("--input_tsv_dir", type=str, required=True, help="Directory containing input TSV files")
    parser.add_argument("--output_json", type=str, required=True, help="Path to the output JSON file")
    
    
    args = parser.parse_args()
    main(args)
