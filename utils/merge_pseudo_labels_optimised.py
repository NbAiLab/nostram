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
    df_json = pd.read_json(args.input_json, lines=True, dtype={'id': str})

    added_columns = []

    with Pool() as pool:
        tsv_files = glob.glob(os.path.join(args.input_tsv_dir, "**/*.tsv"), recursive=True)
        tsv_dfs = list(tqdm(pool.imap(read_tsv, tsv_files), total=len(tsv_files)))

    for df_tsv in tsv_dfs:
        model_name = df_tsv.columns[1]
        
        if model_name not in df_json.columns:
            df_json[model_name] = pd.Series(dtype="object")
            added_columns.append(model_name)
        
        df_json.set_index('id', inplace=True)
        df_tsv.set_index('file_id', inplace=True)
        df_json.update(df_tsv)
        df_json.reset_index(inplace=True)

    if not args.do_not_simplify_columns:
        simplify_column_names(df_json)
    
    df_json.to_json(args.output_json, orient='records', lines=True)

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
