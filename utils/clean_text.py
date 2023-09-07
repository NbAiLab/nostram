import argparse
import json
import re
from pandarallel import pandarallel
import pandas as pd

# Initialize pandarallel
pandarallel.initialize()

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

def clean_text(text):
    global stats
    if "nocaptions" in text:
        return text
    
    original_text = text
    
    text = ' '.join(text.split())  # double_spacing
    if text != original_text:
        stats["double_spacing"] += 1

    text = re.sub(r"^(?:- |— )", "", text)  # remove_dashes
    if text != original_text:
        stats["remove_dashes"] += 1
    
    text = re.sub(r'\.{4,}', '...', text)  # too_long_ellipses
    if text != original_text:
        stats["too_long_ellipses"] += 1

    text = re.sub(r'\.\s*\.\s*\.', '...', text)  # illegal_ellipses
    if text != original_text:
        stats["illegal_ellipses"] += 1

    text = re.sub(r'([!\?])\1+', r'\1', text)  # double_punctuation
    if text != original_text:
        stats["double_punctuation"] += 1
    
    text = text.replace('’', "'").replace('ò', 'o').replace('è', 'e')  # unicode_cleaning
    if text != original_text:
        stats["unicode_cleaning"] += 1

    text = text.replace("\n", " ").replace("\r", " ")  # remove_line_breaks
    if text != original_text:
        stats["remove_line_breaks"] += 1

    text = text.replace("\t", " ")  # remove_tabs
    if text != original_text:
        stats["remove_tabs"] += 1

    if any(c for c in text if c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—’òè/½¼¾'):
        stats["stop_function"] += 1
        print(f"Unhandled character. Original text: {original_text}")

    text = text.replace("1/2", "½").replace("1/4", "¼").replace("3/4", "¾")  # fraction_replace
    if text != original_text:
        stats["fraction_replace"] += 1

    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean text in a JSON file.')
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_json(input_file, lines=True)
    df['text'] = df['text'].parallel_apply(clean_text)
    df.to_json(output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {stats}")
