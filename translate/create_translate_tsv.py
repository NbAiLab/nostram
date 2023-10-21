import pandas as pd
import argparse
import os

def process_file(input_file_name, output_file_name, target_field):
    # Read the JSON lines file
    data = pd.read_json(input_file_name, lines=True, dtype={'id': str, 'group_id': str})
    # Ignore lines with "" in the target_field
    data = data[data[target_field] != ""]

    # Replace None in group_id with "xxx"
    data['group_id'] = data['group_id'].fillna('xxx')

    # Convert group_id to string to ensure consistency
    data['group_id'] = data['group_id'].astype(str)

    # Group by group_id and aggregate the id and target_field
    aggregated = data.groupby('group_id').agg({
        'id': lambda ids: ','.join(map(str, ids)),
        target_field: lambda texts: '<p>'.join(texts)
    }).reset_index()

    # Split entries longer than 9000 characters
    result = []
    for _, row in aggregated.iterrows():
        ids = row['id'].split(',')
        texts = row[target_field].split('<p>')
        current_id = []
        current_text = []
        current_length = 0
        for id_, text in zip(ids, texts):
            if current_length + len(text) + len('<p>') <= 9000:  # Account for the length of '<p>' separator
                current_id.append(id_)
                current_text.append(text)
                current_length += len(text) + len('<p>')  # Account for the length of '<p>' separator
            else:
                result.append((','.join(current_id), '<p>'.join(current_text)))
                current_id = [id_]
                current_text = [text]
                current_length = len(text)
        result.append((','.join(current_id), '<p>'.join(current_text)))

    # Convert result to a DataFrame and write it to a TSV file
    result_df = pd.DataFrame(result, columns=['id', target_field])
    result_df.to_csv(output_file_name, sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_name", required=True, help="Name of the input json lines file.")
    parser.add_argument("--output_file_name", required=True, help="Name of the output tsv file.")
    parser.add_argument("--target_field", default="timestamped_text", help="Name of the field in the JSON lines file to process.")
    args = parser.parse_args()

    if not os.path.isfile(args.input_file_name):
        print(f"The input file '{args.input_file_name}' does not exist.")
        exit(1)

    process_file(args.input_file_name, args.output_file_name, args.target_field)
    print(f"Successfully processed '{args.input_file_name}' and wrote results to '{args.output_file_name}'.")
