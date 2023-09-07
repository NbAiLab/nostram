import argparse
import json
import re
from pandarallel import pandarallel
import pandas as pd

# Initialize pandarallel
pandarallel.initialize()

def clean_text(text):
    stats = {
        "double_spacing": 0,
        "remove_dashes": 0,
        "too_long_ellipses": 0,
        "illegal_ellipses": 0,
        "double_punctuation": 0,
        "unicode_cleaning": 0,
        "remove_line_breaks": 0,
        "remove_tabs": 0,
        "stop_function": 0,
        "fraction_replace": 0
    }

    if "nocaptions" in text:
        return text, stats
    
    original_text = text
    changes_made = False

    text = ' '.join(text.split())
    if text != original_text:
        stats["double_spacing"] += 1
        changes_made = True

    text = re.sub(r"^(?:- |— )", "", text)
    if text != original_text:
        stats["remove_dashes"] += 1
        changes_made = True
    
    # ... (similar blocks for other cleaning functions)

    # Only check for unhandled characters if no changes have been made
    if not changes_made:
        unhandled_char = next((c for c in text if c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—’òè/½¼¾'), None)
        if unhandled_char:
            stats["stop_function"] += 1
            print(f"Unhandled character: {unhandled_char}. Original text: {original_text}")

    return text, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean text in a JSON file.')
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_json(input_file, lines=True)

    total_stats = {
        "double_spacing": 0,
        "remove_dashes": 0,
        "too_long_ellipses": 0,
        "illegal_ellipses": 0,
        "double_punctuation": 0,
        "unicode_cleaning": 0,
        "remove_line_breaks": 0,
        "remove_tabs": 0,
        "stop_function": 0,
        "fraction_replace": 0
    }

    for index, row in df.iterrows():
        cleaned_text, stats = clean_text(row['text'])
        df.at[index, 'text'] = cleaned_text

        for key in stats:
            total_stats[key] += stats[key]

    df.to_json(output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {total_stats}")
