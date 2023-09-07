import argparse
import pandas as pd
from pandarallel import pandarallel
import ftfy
import re

# Global variable to store statistics
stats = {}

def update_stats_and_print(func, original, cleaned):
    global stats
    if original != cleaned:
        stats[func.__name__] = stats.get(func.__name__, 0) + 1
        print(f"{func.__name__}:\nBefore: {original}\nAfter: {cleaned}")

def double_spacing(text):
    return ' '.join(text.split())

def too_long_ellipses(text):
    return re.sub(r'\.{4,6}', '…', text)

def illegal_ellipses(text):
    return text.replace('...', '…')

def double_punctuation(text):
    return re.sub(r'\.{2,}', '.', text)

def remove_dashes(text):
    return re.sub(r'[-—–]\s*', '', text)

def unicode_cleaning(text):
    return ftfy.fix_text(text)

def remove_line_breaks(text):
    return text.replace('\n', ' ')

def remove_tabs(text):
    return text.replace('\t', ' ')

def stop_function(text):
    allowed_chars = r'[a-zA-ZæøåÆØÅ0-9,.@+?=&%$#§!"]'
    illegal_chars = re.findall(f'[^{allowed_chars}]', text)
    
    if illegal_chars:
        print(f"Illegal character(s) found: {', '.join(set(illegal_chars))}")
        print(f"Raw Text: {text}")
        exit(1)

def clean_text(text):
    original_text = text

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
        update_stats_and_print(func, text, cleaned_text)
        text = cleaned_text

    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean up a large JSON lines text file.')
    parser.add_argument('--input_file', required=True, help='Input JSON lines file.')
    parser.add_argument('--output_file', required=True, help='Output JSON lines file.')
    args = parser.parse_args()

    pandarallel.initialize()

    # Read JSON lines into DataFrame
    df = pd.read_json(args.input_file, lines=True)

    # Apply clean_text function in parallel
    df['text'] = df['text'].parallel_apply(clean_text)

    # Save cleaned DataFrame to JSON lines
    df.to_json(args.output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {stats}")
