import argparse
import pandas as pd
import jsonlines
import string
from jiwer import wer

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).strip()

def calculate_wer(df):
    if 'text' not in df or 'wav2vec_text' not in df:
        raise ValueError("Fields 'text' and 'wav2vec_text' must exist in the input file.")

    df['text'] = df['text'].apply(lambda x: remove_punctuation(str(x).lower()))
    df['wav2vec_text'] = df['wav2vec_text'].apply(lambda x: remove_punctuation(str(x).lower().strip()))

    wer_values = []
    for index, row in df.iterrows():
        wer_value = wer(row['text'], row['wav2vec_text'])
        wer_values.append(min(wer_value, 1.0))  # Cap the WER score at 1.0

    df['wav2vec_wer'] = wer_values
    return df

def read_jsonl_file(input_file_name):
    data = []
    with jsonlines.open(input_file_name) as reader:
        for obj in reader:
            data.append(obj)
    df = pd.DataFrame(data)
    return df

def write_jsonl_file(output_file_name, df):
    data = df.to_dict(orient='records')
    with jsonlines.open(output_file_name, mode='w') as writer:
        writer.write_all(data)

def main(input_file_name, output_file_name):
    df = read_jsonl_file(input_file_name)
    df = calculate_wer(df)
    write_jsonl_file(output_file_name, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate Word Error Rate (WER).')
    parser.add_argument('--input_file_name', type=str, required=True, help='The input jsonlines file')
    parser.add_argument('--output_file_name', type=str, required=True, help='The output jsonlines file')

    args = parser.parse_args()
    main(args.input_file_name, args.output_file_name)
