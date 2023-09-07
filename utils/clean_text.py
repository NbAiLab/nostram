import argparse
import pandas as pd
from pandarallel import pandarallel
import ftfy
import re

stats = {
    'double_spacing': 0,
    'too_long_ellipses': 0,
    'illegal_ellipses': 0,
    'double_punctuation': 0,
    'remove_dashes': 0,
    'unicode_cleaning': 0,
    'remove_line_breaks': 0,
    'remove_tabs': 0,
    'stop_function': 0
}

def update_stats(func):
    global stats
    stats[func.__name__] += 1

def clean_text(text):
    if "nocaptions" in text:
        return text

    original_text = text
    if not text.strip():
        return text

    funcs = [
        double_spacing,
        too_long_ellipses,
        illegal_ellipses,
        double_punctuation,
        remove_dashes,
        unicode_cleaning,
        remove_line_breaks,
        remove_tabs,
        stop_function
    ]

    for func in funcs:
        cleaned_text = func(text)
        if cleaned_text != text:
            update_stats(func)
        text = cleaned_text

    return text

# ... The rest of your cleaning functions go here ...

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean up a large JSON lines text file.')
    parser.add_argument('--input_file', required=True, help='Input JSON lines file.')
    parser.add_argument('--output_file', required=True, help='Output JSON lines file.')
    args = parser.parse_args()

    pandarallel.initialize()

    df = pd.read_json(args.input_file, lines=True)
    df['text'] = df['text'].parallel_apply(clean_text)
    df.to_json(args.output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {stats}")
