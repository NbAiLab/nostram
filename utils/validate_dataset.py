import argparse
import json
import jsonlines
import pandas as pd
from jsonschema import validate, ValidationError
from tabulate import tabulate
import numpy as np
from tqdm import tqdm 
import itertools

# Define the dataset schema
schema = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "group_id": {"type": ["string", "null"]},
        "source": {"type": "string", "enum": ["NRK TV", "NRK TV TRANSLATE", "NPSC", "NST", "FLEURS", "AUDIO BOOKS"]},
        "audio_language": {"type": ["string", "null"]},
        "audio": {"type": "string"},
        "audio_duration": {"type": "integer"},
        "previous_text": {"type": ["string", "null"]},
        "text_language": {"type": "string", "enum": ["no", "nn", "en", "es"]},
        "text": {"type": "string"},
        "translated_text_no": {"type": ["string", "null"]},
        "translated_text_nn": {"type": ["string", "null"]},
        "translated_text_en": {"type": ["string", "null"]},
        "translated_text_es": {"type": ["string", "null"]},
        "timestamped_text": {"type": ["string", "null"]},
        "wav2vec_wer": {"type": ["number", "null"]},
        "whisper_wer": {"type": ["number", "null"]},
        "verbosity_level": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, 6, None]}
    },
    "required": ["id", "source", "audio", "audio_duration", "text_language", "text"]
}

def validate_json_format(data):
    try:
        for item in data:
            json.dumps(item)
        print("SUCCESS: No errors according to JSON format specifications")
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON format: {e}")
        return False
    return True

from tqdm import tqdm

def validate_schema(data):
    try:
        with tqdm(total=len(data), desc="Validating dataset") as pbar:
            for item in data:
                validate(item, schema)
                pbar.update(1)
        print("\nSUCCESS: No errors in Scream dataset specifications")
    except ValidationError as e:
        print(f"ERROR: Invalid data according to Scream dataset specifications: {e}")
        return False
    return True

def validate_pandas_import(data):
    try:
        df = pd.DataFrame(data)
        print("SUCCESS: No errors importing data to Pandas DataFrame")
    except Exception as e:
        print(f"ERROR: Failed to load data into a DataFrame: {e}")
        return False, None
    return True, df

def calculate_statistics(df, detailed):
    filled_field_counts = calculate_filled_fields(df, "Total")
    total_stats = pd.concat([calculate_avg_word_counts(df, "Total"),
                             calculate_avg_char_counts(df, "Total"),
                             calculate_unique_value_counts(df, "Total"),
                             calculate_avg_duration(df, "Total")], ignore_index=True)
    total_stats.set_index('Measure', inplace=True)
    total_stats = total_stats[['Value']].rename(columns={'Value': 'Total'})
    stats = total_stats.copy()

    if detailed:
        sources = df['source'].unique()
        
        for source in sources:
            source_df = df[df['source'] == source]
            filled_field_counts = pd.concat([filled_field_counts, calculate_filled_fields(source_df, source)], axis=1)
            
            source_stats = pd.concat([calculate_avg_word_counts(source_df, source),
                                      calculate_avg_char_counts(source_df, source),
                                      calculate_unique_value_counts(source_df, source),
                                      calculate_avg_duration(source_df, source)], ignore_index=True)
            source_stats.set_index('Measure', inplace=True)
            source_stats = source_stats[['Value']].rename(columns={'Value': source})
            stats = pd.concat([stats, source_stats], axis=1)

        # Move 'Total' column to the end
        stats = stats[[col for col in stats if col != 'Total'] + ['Total']]
        filled_field_counts = filled_field_counts[[col for col in filled_field_counts if col != 'Total'] + ['Total']]

    print("\nCounts of filled out fields:")
    print(tabulate(filled_field_counts, headers='keys', tablefmt='psql'))
    print("\nStatistics:")
    print(tabulate(stats, headers='keys', tablefmt='psql', showindex=True))

    total_lines = len(df)
    total_words = df['text'].str.split().str.len().sum()
    total_characters = df['text'].str.len().sum()
    total_duration_ms = df['audio_duration'].sum()
    total_duration = convert_milliseconds(total_duration_ms)

    # Update the summary_df construction to format the numbers with the thousand separator
    summary_df = pd.DataFrame({
        'Measure': ['Total lines', 'Total words in \'text\'', 'Total characters in \'text\'', 'Total audio duration'],
        'Total': [f"{total_lines:,}", f"{total_words:,}", f"{total_characters:,}", total_duration]
    })


    print("\nSummary:")
    print(tabulate(summary_df, headers='keys', tablefmt='psql', showindex=False))


def calculate_filled_fields(data_frame, source):
    filled_counts = data_frame.count()
    filled_counts.name = source
    return filled_counts.to_frame()

def calculate_avg_word_counts(data_frame, source):
    avg_word_count = round(data_frame['text'].apply(lambda s: len(str(s).split())).mean())
    return pd.DataFrame([(source, "Average word count", avg_word_count)], columns=['Source', 'Measure', 'Value'])

def calculate_avg_char_counts(data_frame, source):
    avg_char_count = round(data_frame['text'].apply(lambda s: len(str(s))).mean())
    return pd.DataFrame([(source, "Average character count", avg_char_count)], columns=['Source', 'Measure', 'Value'])

def calculate_unique_value_counts(data_frame, source):
    unique_values_count = len(data_frame['source'].unique())
    return pd.DataFrame([(source, "Unique value count", unique_values_count)], columns=['Source', 'Measure', 'Value'])

def calculate_avg_duration(data_frame, source):
    avg_duration_seconds = round(data_frame['audio_duration'].mean() / 1000, 2) # converting from ms to s
    return pd.DataFrame([(source, "Average duration (s)", avg_duration_seconds)], columns=['Source', 'Measure', 'Value'])

def convert_milliseconds(ms):
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours} hours, {minutes} minutes, {seconds} seconds"

def main(args):
    with open(args.filename, 'r') as f:
        reader = jsonlines.Reader(f)
        data = []
        for line in itertools.islice(reader.iter(), args.n):
            data.append(line)

    if args.statistics_only:
        _, df = validate_pandas_import(data)
        if not df.empty:
            calculate_statistics(df, args.detailed)
    else:
        # General JSON format validation
        if not validate_json_format(data):
            return

        # Scream dataset specifications validation
        if not validate_schema(data):
            return

        # Validation for loading into pandas
        success, df = validate_pandas_import(data)
        if not success:
            return

        # Calculate statistics
        calculate_statistics(df, args.detailed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a Scream dataset.')
    parser.add_argument('filename', type=str, help='the JSONL file to validate')
    parser.add_argument('-n', type=int, help='limit the number of lines to read')
    parser.add_argument('-s', '--statistics_only', action='store_true', help='run only the statistics without validation')
    parser.add_argument('-d', '--detailed', action='store_true', help='detailed statistics for every source')
    args = parser.parse_args()

    main(args)

