import argparse
import re
from pandarallel import pandarallel
import pandas as pd
import ftfy

# Initialize pandarallel
pandarallel.initialize()

import ftfy  # Assuming you have imported ftfy

def clean_text(text, verbose=False):
    stats = {
        "double_spacing": 0,
        "remove_dashes": 0,
        "illegal_ellipses": 0,
        "double_punctuation": 0,
        "unicode_cleaning": 0,
        "remove_line_breaks": 0,
        "remove_tabs": 0,
        "special_char_replace": 0,
        "delete_line": False,
        "unhandled": 0
    }

    def special_char_replace(text):
        replacements = {
            "ê": "é",
            "Ê": "É",
            "è": "é",
            "È": "É",
            "ó": "ò",
            "Ó": "Ò",
            "á": "à",
            "Á": "À",
            "â": "à",
            "Â": "À",
            "ô": "ò",
            "«": "",
            "»": "",
            "– ": ""
        }
        new_text = "".join(replacements.get(c, c) for c in text)
        return new_text

    if "nocaptions" in text:
        return text, stats

    original_text = text

    # Delete line if it contains any bracket and a few other weird characters
    if "(" in text or ")" in text or "[" in text or "]" in text or "{" in text or "}" in text or "ã" in text or "ú" in text:
        stats["delete_line"] = True
        if verbose: print(f"Line to be deleted - Original: {original_text}")
        return text, stats
    
    # Unicode cleaning
    text = ftfy.fix_text(text)
    if text != original_text:
        stats["unicode_cleaning"] += 1
        if verbose: print(f"Unicode cleaning - Original: {original_text} - Result: {text}")

    # Double spacing
    new_text = ' '.join(text.split())
    if new_text != text:
        stats["double_spacing"] += 1
        if verbose: print(f"Double spacing - Original: {text} - Result: {new_text}")
        text = new_text
    
    # Special character replacements
    new_text = special_char_replace(text)
    if new_text != text:
        stats["special_char_replace"] += 1
        if verbose: print(f"Special character replacement - Original: {text} - Result: {new_text}")
        text = new_text
        
    # Remove Dashes
    new_text = re.sub(r"^(?:- |— )", "", text)
    if new_text != text:
        stats["remove_dashes"] += 1
        if verbose: print(f"Remove dashes - Original: {text} - Result: {new_text}")
        text = new_text

    # Illegal ellipses
    new_text = re.sub(r'\.\.\.', '…', text)
    if new_text != text:
        stats["illegal_ellipses"] += 1
        if verbose: print(f"Illegal ellipses - Original: {text} - Result: {new_text}")
        text = new_text

    # Double punctuation
    new_text = re.sub(r'([!\?])\1+', r'\1', text)
    if new_text != text:
        stats["double_punctuation"] += 1
        if verbose: print(f"Double punctuation - Original: {text} - Result: {new_text}")
        text = new_text

    # Remove line breaks
    new_text = text.replace("\n", " ").replace("\r", " ")
    if new_text != text:
        stats["remove_line_breaks"] += 1
        if verbose: print(f"Remove line breaks - Original: {text} - Result: {new_text}")
        text = new_text

    # Remove tabs
    new_text = text.replace("\t", " ")
    if new_text != text:
        stats["remove_tabs"] += 1
        if verbose: print(f"Remove tabs - Original: {text} - Result: {new_text}")
        text = new_text

    # Unhandled
    allowed_chars = '²³ëüäö»«abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'()*+,-./:;<=>?@_`—’òÒàÀéÉ½¼¾ÒæøåÆØÅ…°'
    unhandled_char = next((c for c in text if c not in allowed_chars), None)
    if unhandled_char:
        stats["unhandled"] += 1
        stats["delete_line"] = True
        print(f"Unhandled character - Original: {original_text} - Char: {unhandled_char}")
    
    return text, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Clean text in a JSON file.')
    parser.add_argument('--input_file', required=True, help='Path to the input JSON file.')
    parser.add_argument('--output_file', required=True, help='Path to the output JSON file.')
    parser.add_argument('--verbose', action='store_true', help='Verbose output of changes.')
    
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
        "special_char_replace": 0,
        "delete_line": False,
        "unhandled": 0
    }
    
    indices_to_delete = []

    for index, row in df.iterrows():
        cleaned_text, stats = clean_text(row['text'], args.verbose)
        if stats["delete_line"]:
            indices_to_delete.append(index)
            total_stats["delete_line"] += 1
        else:
            df.at[index, 'text'] = cleaned_text

        for key in stats:
            if key != "delete_line":
                total_stats[key] += stats[key]

    # Delete rows
    df.drop(indices_to_delete, inplace=True)

    df.to_json(output_file, orient='records', lines=True)

    print("Cleaning completed.")
    print(f"Statistics: {total_stats}")
