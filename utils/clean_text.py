import argparse
import re
from pandarallel import pandarallel
import pandas as pd

# Initialize pandarallel
pandarallel.initialize()

def clean_text(text):
    stats = {
        "double_spacing": 0,
        "remove_dashes": 0,
        "illegal_ellipses": 0,
        "double_punctuation": 0,
        "unicode_cleaning": 0,
        "remove_line_breaks": 0,
        "remove_tabs": 0,
        "stop_function": 0
    }

    if "nocaptions" in text:
        return text, stats

    original_text = text
    changes_made = False

    # Double spacing
    text = ' '.join(text.split())
    if text != original_text:
        stats["double_spacing"] += 1
        changes_made = True

    # Remove Dashes
    text = re.sub(r"^(?:- |— )", "", text)
    if text != original_text:
        stats["remove_dashes"] += 1
        changes_made = True

    # Illegal ellipses
    text = re.sub(r'\.\.\.', '…', text)
    if text != original_text:
        stats["illegal_ellipses"] += 1
        changes_made = True

    # Double punctuation
    text = re.sub(r'([!\?])\1+', r'\1', text)
    if text != original_text:
        stats["double_punctuation"] += 1
        changes_made = True

    # Unicode cleaning
    text = text.replace('’', "'").replace('ò', 'o').replace('ê', 'è').replace('Á', 'A').replace("1/2", "½").replace("1/4", "¼").replace("3/4", "¾")
    if text != original_text:
        stats["unicode_cleaning"] += 1
        changes_made = True

    # Remove line breaks
    text = text.replace("\n", " ").replace("\r", " ")
    if text != original_text:
        stats["remove_line_breaks"] += 1
        changes_made = True

    # Remove tabs
    text = text.replace("\t", " ")
    if text != original_text:
        stats["remove_tabs"] += 1
        changes_made = True

    # Stop function
    allowed_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—’òèÁ/½¼¾ÉÒæøåÆØÅ°'
    unhandled_char = next((c for c in text if c not in allowed_chars), None)
    if unhandled_char and not changes_made:
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
        "illegal_ellipses": 0,
        "double_punctuation": 0,
        "unicode_cleaning": 0,
        "remove_line_breaks": 0,
        "remove_tabs": 0,
        "stop_function": 0
    }

    for index, row in df.iterrows():
        cleaned_text, stats = clean_text(row['text'])
        df.at[index, 'text'] = cleaned_text

        for key in stats:
            total_stats[key] += stats[key]

    df.to_json(output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {total_stats}")
