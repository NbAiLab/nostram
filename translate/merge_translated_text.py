import pandas as pd
import argparse
import os
import json
from tqdm import tqdm

def process_file(input_json_file_name, input_tsv_file_name, output_file_name, translate_field):
    # Read the original JSON lines file into a dictionary and the translated TSV file
    with open(input_json_file_name, 'r') as f:
        original = {obj['id']: obj for obj in map(json.loads, f)}
    try:
        translated = pd.read_csv(input_tsv_file_name, sep='\t', names=['id', 'original text', 'translated text'])
    except pd.errors.EmptyDataError:
        print(f"The input tsv file '{input_tsv_file_name}' is empty.")
        return

    # Iterate over each row in the translated DataFrame
    for row in tqdm(translated.itertuples(), total=translated.shape[0]):
        ids = row[1].split(',')
        original_texts = row[2].split('<p>')
        translated_texts = row[3].split('<p>')

        # Check if the number of IDs and text snippets match
        if len(ids) != len(original_texts) or len(ids) != len(translated_texts):
            print(f"Error: Number of IDs and text snippets do not match for ID '{row[1]}'")
            continue

        # Update the original DataFrame with the translated text
        for id_, translated_text in zip(ids, translated_texts):
            if id_ == 'id' or id_ not in original:
                print(f"Error: ID '{id_}' from tsv file was not found in the original json lines file.")
                continue
            original[id_][translate_field] = translated_text.strip()

    # Save the original DataFrame to a new JSON lines file
    with open(output_file_name, 'w') as f:
        for obj in original.values():
            f.write(json.dumps(obj))
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_file_name", required=True, help="Name of the original json lines file.")
    parser.add_argument("--input_tsv_file_name", required=True, help="Name of the translated tsv file.")
    parser.add_argument("--output_file_name", required=True, help="Name of the output json lines file.")
    parser.add_argument("--translate_field", default='translated_text_en', help="Name of the translation field in the original json lines file.")
    args = parser.parse_args()

    if not os.path.isfile(args.input_json_file_name):
        print(f"The input json file '{args.input_json_file_name}' does not exist.")
        exit(1)

    if not os.path.isfile(args.input_tsv_file_name):
        print(f"The input tsv file '{args.input_tsv_file_name}' does not exist.")
        exit(1)

    process_file(args.input_json_file_name, args.input_tsv_file_name, args.output_file_name, args.translate_field)
    print(f"Successfully processed '{args.input_json_file_name}' and '{args.input_tsv_file_name}', and wrote results to '{args.output_file_name}'.")
