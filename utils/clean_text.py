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

def double_spacing(text):
    return ' '.join(text.split())

def remove_dashes(text):
    if text.startswith("- ") or text.startswith("— "):
        return text[2:]
    return text

def too_long_ellipses(text):
    return re.sub(r'\.{4,}', '...', text)

def illegal_ellipses(text):
    return re.sub(r'\.\s*\.\s*\.', '...', text)

def double_punctuation(text):
    return re.sub(r'([!\?])\1+', r'\1', text)

def unicode_cleaning(text):
    return text.replace('’', "'").replace('ò', 'o').replace('è', 'e')

def remove_line_breaks(text):
    return text.replace("\n", " ").replace("\r", " ")

def remove_tabs(text):
    return text.replace("\t", " ")

def stop_function(text):
    if any(c for c in text if c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~—’òè'):
        print(f"Unhandled character. Original text: {text}")
    return text

def fraction_replace(text):
    return text.replace("1/2", "½").replace("1/4", "¼").replace("3/4", "¾")

def clean_text(text):
    global stats
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
            stats[func.__name__] += 1
        text = cleaned_text

    return text

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean text in a JSON file.')
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    
    args = parser.parse_args()
    input_file = args.input_file
    output_file = args.output_file

    df = pd.read_json(input_file)
    df['text'] = df['text'].parallel_apply(clean_text)
    df.to_json(output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {stats}")
