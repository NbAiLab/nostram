import pandas as pd
import argparse
import os
import json
import html
import re
from tqdm import tqdm

def process_file(input_json_file_name, input_tsv_file_name, output_file_name):
    with open(input_json_file_name, 'r') as f:
        original = {obj['id']: obj for obj in map(json.loads, f)}

    try:
        translated = pd.read_csv(input_tsv_file_name, sep='\t', names=['id', 'original text', 'translated text'])
    except pd.errors.EmptyDataError:
        print(f"The input tsv file '{input_tsv_file_name}' is empty.")
        return

    pattern = re.compile(r'<\|.*?\|>')

    for row in tqdm(translated.itertuples(), total=translated.shape[0]):
        ids = row[1].split(',')
        original_texts = row[2].split('<p>')
        translated_texts = row[3].split('<p>')

        if len(ids) != len(original_texts) or len(ids) != len(translated_texts):
            print(f"Error: Number of IDs and text snippets do not match for ID '{row[1]}'")
            continue

        for id_, translated_text in zip(ids, translated_texts):
            if id_ == 'id':
                continue
            elif id_ not in original:
                print(f"Error: ID '{id_}' from tsv file was not found in the original json lines file.")
                continue

            # Unescape HTML entities
            timestamped_text = html.unescape(translated_text.strip())

            clean_text = ' '.join(re.sub(pattern, '', timestamped_text).split())

            original[id_]['timestamped_text_en'] = timestamped_text
            original[id_]['text_en'] = clean_text

    with open(output_file_name, 'w') as f:
        for obj in original.values():
            f.write(json.dumps(obj))
            f.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json_file_name", required=True, help="Name of the original json lines file.")
    parser.add_argument("--input_tsv_file_name", required=True, help="Name of the translated tsv file.")
    parser.add_argument("--output_file_name", required=True, help="Name of the output json lines file.")
    args = parser.parse_args()

    if not os.path.isfile(args.input_json_file_name):
        print(f"The input json file '{args.input_json_file_name}' does not exist.")
        exit(1)

    if not os.path.isfile(args.input_tsv_file_name):
        print(f"The input tsv file '{args.input_tsv_file_name}' does not exist.")
        exit(1)

    process_file(args.input_json_file_name, args.input_tsv_file_name, args.output_file_name)
    print(f"Successfully processed '{args.input_json_file_name}' and '{args.input_tsv_file_name}', and wrote results to '{args.output_file_name}'.")
