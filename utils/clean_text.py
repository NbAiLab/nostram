import argparse
import re
from pandarallel import pandarallel
import pandas as pd

# Initialize pandarallel
pandarallel.initialize()

def clean_text(text, verbose=False):
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

    # Initialize a list for logging verbose output
    verbose_output = []

    def apply_change(func, text):
        nonlocal changes_made
        cleaned_text = func(text)
        if cleaned_text != text:
            stats[func.__name__] += 1
            changes_made = True
            if verbose:
                verbose_output.append(f"{func.__name__} - {text} - {cleaned_text}")
        return cleaned_text

    text = apply_change(lambda t: ' '.join(t.split()), text)
    text = apply_change(lambda t: re.sub(r"^(?:- |— )", "", t), text)
    text = apply_change(lambda t: re.sub(r'\.\.\.', '…', t), text)
    text = apply_change(lambda t: re.sub(r'([!\?])\1+', r'\1', t), text)
    text = apply_change(lambda t: t.replace('’', "'").replace('ò', 'o').replace('é', 'è').replace('ó', 'ò').replace("1/2", "½").replace("1/4", "¼").replace("3/4", "¾"), text)
    text = apply_change(lambda t: t.replace("\n", " ").replace("\r", " "), text)
    text = apply_change(lambda t: t.replace("\t", " "), text)

    # Stop function
    allowed_chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'*+,-./:;<=>?@[\\]^_`{|}~—’òèÁ/½¼¾ÉÒæøåÆØÅ°'
    unhandled_char = next((c for c in text if c not in allowed_chars), None)
    if unhandled_char and not changes_made:
        stats["stop_function"] += 1
        print(f"Unhandled character: {unhandled_char}. Original text: {original_text}")

    if verbose:
        print("\n".join(verbose_output))

    return text, stats

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean text in a JSON file.')
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose output of changes.')
    
    args = parser.parse_args()

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

    df = pd.read_json(args.input_file, lines=True)

    for index, row in df.iterrows():
        cleaned_text, stats = clean_text(row['text'], args.verbose)
        df.at[index, 'text'] = cleaned_text

        for key in stats:
            total_stats[key] += stats[key]

    df.to_json(args.output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {total_stats}")
