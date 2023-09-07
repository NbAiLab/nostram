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
    'stop_function': 0,
    'fraction_replace': 0
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
        remove_dashes,
        too_long_ellipses,
        illegal_ellipses,
        double_punctuation,
        unicode_cleaning,
        remove_line_breaks,
        remove_tabs,
        stop_function,
        fraction_replace
    ]

    for func in funcs:
        cleaned_text = func(text)
        if cleaned_text != text:
            update_stats(func)
        text = cleaned_text

    return text

def double_spacing(text):
    return " ".join(text.split())

def too_long_ellipses(text):
    return re.sub(r'\.{3,}', '...', text)

def illegal_ellipses(text):
    return re.sub(r'(\.{1,2}[^\.])|([^\.]\.{1,2})', '. ', text)

def double_punctuation(text):
    return re.sub(r'([!?,:;"\'\.\-—])\1+', r'\1', text)

def remove_dashes(text):
    return re.sub(r'^[-—]\s*', '', text)

def unicode_cleaning(text):
    return ftfy.fix_text(text)

def remove_line_breaks(text):
    return re.sub(r'[\r\n]+', ' ', text)

def remove_tabs(text):
    return re.sub(r'\t+', ' ', text)

def stop_function(text):
    if re.search(r'[^a-zA-Z0-9\s,.?!:;\'"/\-—èò]', text):
        print(f"Unhandled character. Original text: {text}")
    return text

def fraction_replace(text):
    text = text.replace("1/2", "½").replace("1/4", "¼").replace("3/4", "¾")
    return text

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
