import pandas as pd
import argparse
import os

def process_file(input_json_file_name, input_tsv_file_name, output_file_name, translate_field):
    # Read the original JSON lines file and the translated TSV file
    original = pd.read_json(input_json_file_name, lines=True)
    try:
        translated = pd.read_csv(input_tsv_file_name, sep='\t', names=['id', 'original text', 'translated text'])
    except pd.errors.EmptyDataError:
        print(f"The input tsv file '{input_tsv_file_name}' is empty.")
        return

    # Iterate over each row in the translated DataFrame
    for _, row in translated.iterrows():
        ids = row['id'].split(',')
        original_texts = row['original text'].split('<p>')
        translated_texts = row['translated text'].split('<p>')

        # Check if the number of IDs and text snippets match
        if len(ids) != len(original_texts) or len(ids) != len(translated_texts):
            print(f"Error: Number of IDs and text snippets do not match for ID '{row['id']}'")
            continue

        # Update the original DataFrame with the translated text
        for id_, translated_text in zip(ids, translated_texts):
            if id_ == 'id' or id_ not in original['id'].values:
                print(f"Error: ID '{id_}' from tsv file was not found in the original json lines file.")
                continue
            original.loc[original['id'] == id_, translate_field] = translated_text.strip()

    # Save the original DataFrame to a new JSON lines file
    original.to_json(output_file_name, orient='records', lines=True)

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

