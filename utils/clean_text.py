import argparse
import re
from pandarallel import pandarallel
import pandas as pd
import ftfy
import string
import unicodedata

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
        "special_char_replace": 0,
        "delete_line": False,
        "remove_non_printable": 0,
        "emoticons": 0,
        "delete_uppercase_words": 0,
        "unhandled": 0
    }
    
    def is_printable(char):
        category = unicodedata.category(char)
        return not category.startswith("C")
        
    def contains_emoticon(text):
        for char in text:
            if unicodedata.category(char) == "So":
                return True
        return False
    
    def has_two_uppercase_words(text):
        uppercase_words = re.findall(r'\b[A-Z]{3,}\b', text)
        return len(uppercase_words) >= 2
    
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
            "ô": "ò",
            "е": "e",
            "«": "",
            "»": "",
            "•": "",
            "´": "'",
            "◊": "",
            "⅔" : "2/3",
            "⅛" : "1/8",
            "⅕" : "1/5",
            "‰" : "promille",
            "1 klasse" : "1. klasse",
            "2 klasse" : "2. klasse",
            "3 klasse" : "3. klasse",
            "4 klasse" : "4. klasse",
            "5 klasse" : "5. klasse",
            "6 klasse" : "6. klasse",
            "7 klasse" : "7. klasse",
            "8 klasse" : "8. klasse",
            "9 klasse" : "9. klasse",
            "10 klasse" : "10. klasse"
        }
        new_text = "".join(replacements.get(c, c) for c in text)
        return new_text

        def format_numbers(text):
            formatted_text = ""
            last_index = 0
        
            for match in re.finditer(r'\b\d+\b', text):
                start, end = match.span()
                num = text[start:end]
        
                if ',' in num:  # Don't touch numbers with commas (decimal numbers)
                    formatted_num = num
                elif len(num) == 4:  # Four-digit numbers should remain unchanged
                    formatted_num = num
                elif len(num) > 4:  # For numbers with 5 or more digits, format with ' ' as separator
                    formatted_num = '{: }'.format(int(num)).replace(',', ' ')
        
                formatted_text += text[last_index:start] + formatted_num
                last_index = end
        
            formatted_text += text[last_index:]
            
            return formatted_text
    
    if "nocaptions" in text:
        return text, stats

    original_text = text
    
    # Unicode cleaning
    text = ftfy.fix_text(text)
    if text != original_text:
        stats["unicode_cleaning"] += 1
        if verbose: print(f"Unicode cleaning - Original: {original_text} - Result: {text}")
    
    # Fix trim and double spacing
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
        
    # Remove dashes if they start the line
    new_text = re.sub(r"^(?:[-−–—‐‒‑˗] )", "", text)

    if new_text != text:
        stats["remove_dashes"] += 1
        if verbose: print(f"Remove dashes - Original: {text} - Result: {new_text}")
        text = new_text
        
    new_text = re.sub(r"^(?:- |— )", "", text)
    if new_text != text:
        stats["remove_dashes"] += 1
        if verbose: print(f"Remove dashes - Original: {text} - Result: {new_text}")
        text = new_text

    # Delete line if it contains any emoticon
    if contains_emoticon(text):
        stats["delete_line"] = True
        stats["emoticons"] += 1
        if verbose: print(f"Line to be deleted due to emoticon - Original: {original_text}")
        return text, stats
    
    # Delete line if it contains at least two uppercase words of length at least 3
    if has_two_uppercase_words(text):
        stats["delete_uppercase_words"] += 1
        stats["delete_line"] = True
        if verbose: print(f"Line to be deleted due to uppercase words - Original: {original_text}")
        return text, stats
    
    # Delete line if it contains any bracket and a few other weird characters
    if any(char in text for char in "~()[{}]ã−-–—‐‒íÍ►™�şûłìð❤|↑·þ☠î›˛†Şİćğвïâý√ˇœ¯←˘ı¨Б¸˙˜С‹țș˚⅓_ūēřū¡£ŦŁÕ€^Œ¢ǤŚ‼¤"):
        stats["delete_line"] = True
        if verbose: print(f"Line to be deleted - Original: {original_text}")
        return text, stats

    # Remove non-printable characters
    new_text = ''.join(filter(is_printable, text))
    if new_text != text:
        stats["remove_non_printable"] += 1
        if verbose: print(f"Non-printable characters removed - Original: {text} - Result: {new_text}")
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

    # Repeat removing of double spacing just in case
    new_text = ' '.join(text.split())
    if new_text != text:
        stats["double_spacing"] += 1
        if verbose: print(f"Double spacing - Original: {text} - Result: {new_text}")
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
    
    # Reformat numbers
    new_text = format_numbers(text)
    if new_text != text:
        stats["format_numbers"] += 1
        print(f"Numbers formatted - Original: {text} - Result: {new_text}")
        text = new_text
        
    # Unhandled
    allowed_chars = '½¼¾²³ñëüúäöÖÜÄ»«abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890!"#$%&\'()*+,./:;<=>?¿@`’òÒàÀéÉ½¼¾ÒæøåÆØÅ…°§ÞßÚÎšŠčžŋÇçŽùÑČųµėđÿŧÔţË÷õĐÏŊĐõ'
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
    print(f"Read dataset with: {len(df)} lines.")
    
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
        "remove_non_printable": 0,
        "emoticons": 0,
        "delete_uppercase_words": 0,
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
    print(f"Number of remaining lines: {len(df)}.")
    print(f"Statistics: {total_stats}")
