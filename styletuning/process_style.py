#!/usr/bin/env python3

import argparse
import jsonlines
import os
import sys
import re
import glob
from jiwer import wer

# Text cleaning function
def clean_text(text):
    return re.sub(r"[^a-zA-ZæøåÆØÅ]", " ", str(text).lower()).strip()

# Function to compute WER for different language references
def get_wer(data, language):
    text = clean_text(data.get("text", "").strip())
    if not text:  # Check if the text is empty
        return 1  # Return maximum WER to indicate an invalid comparison

    if language == "no":
        reference_a = clean_text(data.get("A-no-nohes", ""))
        reference_b = clean_text(data.get("B-no-hes", ""))
        if not reference_a or not reference_b:  # Check if either reference is empty
            return 1  # Return maximum WER to indicate an invalid comparison
        return min(wer(reference_a, text), wer(reference_b, text))
    elif language == "nn":
        reference_d = clean_text(data.get("D-nn-nohes", ""))
        reference_e = clean_text(data.get("E-nn-hes", ""))
        if not reference_d or not reference_e:  # Check if either reference is empty
            return 1  # Return maximum WER to indicate an invalid comparison
        return min(wer(reference_d, text), wer(reference_e, text))

def process_line(data, subcorpus):
    if subcorpus == 'nst' and data.get("source") == "nst":
        if get_wer(data, "no") <= 0.1:
            data["task"] = "transcribe"
            return data
    elif subcorpus == 'ellipses':
        text = data.get("text", "")
        if text.startswith("…"):
            return None
        if "…" in text:
            if get_wer(data, "no") <= 0.1 or get_wer(data, "nn") <= 0.1:
                data["task"] = "transcribe"
                return data

    elif subcorpus == 'hesitation':
        hesitation_patterns = [r'\b(ehh?|ehm|mmm?)\b', r'\bEee\b']
        text = data.get("text", "")
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in hesitation_patterns):
            if get_wer(data, "no") <= 0.1 or get_wer(data, "nn") <= 0.1:
                text = re.sub(r'\b(ehh?|Eee)\b', lambda match: 'Eh' if match.group(0).isupper() else 'eh', text, flags=re.IGNORECASE)
                text = re.sub(r'\behm\b', 'ehm', text, flags=re.IGNORECASE)
                text = re.sub(r'\bmmm?\b', 'mm', text, flags=re.IGNORECASE)
                data["text"] = text
                data["task"] = "transcribe"
                return data
        
    elif subcorpus == 'clean_verbatim_no':
        if data.get("text_language") != "no":
            return None
        text = data.get("text", "").strip()
        if not (text and text[0].isupper() and text[-1] in ".?!"):
            return None
        if get_wer(data, "no") == 0:
            data["text"] = text
            data["task"] = "transcribe"
            return data

    elif subcorpus == 'clean_verbatim_nn':
        if data.get("text_language") != "nn":
            return None
        text = data.get("text", "").strip()
        if not (text and text[0].isupper() and text[-1] in ".?!"):
            return None
        if get_wer(data, "nn") == 0:
            data["text"] = text
            data["task"] = "transcribe"
            return data

    return None

def read_files(input_pattern, output_folder, subcorpus):
    line_count = 0
    output_file = os.path.join(output_folder, f"{subcorpus}.jsonl")
    with jsonlines.open(output_file, mode='w') as writer:
        for file_path in glob.glob(input_pattern):
            with jsonlines.open(file_path) as reader:
                for data in reader:
                    processed_data = process_line(data, subcorpus)
                    if processed_data:
                        writer.write(processed_data)
                        line_count += 1
    return line_count, output_file

def main():
    parser = argparse.ArgumentParser(description='Process JSON lines files based on subcorpus routines.')
    parser.add_argument('--input_pattern', type=str, required=True, help='Pattern for input JSON lines files.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder for the processed JSON lines file.')
    parser.add_argument('--subcorpus', type=str, required=True, help='Subcorpus routine to use.',
                        choices=['nst', 'ellipses', 'hesitation', 'clean_verbatim_no', 'clean_verbatim_nn'])

    args = parser.parse_args()

    if not os.path.exists(args.output_folder):
        print(f"Output folder '{args.output_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)

    line_count, output_file = read_files(args.input_pattern, args.output_folder, args.subcorpus)
    print(f"{line_count} lines written to {output_file}")

if __name__ == "__main__":
    main()