import argparse
import pandas as pd
import jsonlines
from transformers import pipeline

def process_df(df):
    # Define a pipeline for named entity recognition.
    ner_model = pipeline('ner', model='saattrupdan/nbailab-base-ner-scandi', device=0) # device=0 means it uses GPU

    # Function to get all named entities marked with PER in a text
    def get_per_entities(text):
        ner_results = ner_model(text)
        per_entities = []
        current_name = []

        for i, result in enumerate(ner_results):
            word = result['word']
            next_word = ner_results[i+1]['word'] if i < len(ner_results)-1 else ''

            if result['entity'] == 'B-PER':
                if current_name and current_name[-1].endswith(',') and next_word[0].islower():
                    current_name[-1] = current_name[-1] + word.replace('##', '')
                else:
                    if current_name:
                        # If there was a previous name, add it to the list of entities
                        per_entities.append(' '.join(current_name))
                    # Start a new name
                    current_name = [word.replace('##', '')]  
            elif result['entity'] == 'I-PER':
                # If the entity is a continuation of a name, add it to the current name
                if word.startswith('##'):
                    # Remove '##' and concatenate to the last word
                    current_name[-1] = current_name[-1] + word.replace('##', '')
                else:
                    current_name.append(word)
            else:
                if current_name:
                    # If there was a previous name, add it to the list of entities
                    per_entities.append(' '.join(current_name))
                current_name = []

        # Add the last name if it exists
        if current_name:
            per_entities.append(' '.join(current_name))

        return ', '.join(per_entities)

    df['names'] = df['text'].apply(get_per_entities)
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
    df = process_df(df)
    write_jsonl_file(output_file_name, df)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process jsonlines file.')
    parser.add_argument('--input_file_name', type=str, required=True, help='The input jsonlines file')
    parser.add_argument('--output_file_name', type=str, required=True, help='The output jsonlines file')

    args = parser.parse_args()
    main(args.input_file_name, args.output_file_name)
