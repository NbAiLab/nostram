#!/usr/bin/env python3

import argparse
import jsonlines
import os
import sys
import re

def replace_hesitation(match):
    # Preserve the original casing for the replacement
    text = match.group(0)
    return 'Eh' if text.isupper() else 'eh'

def process_line(data, subcorpus):
    if subcorpus == 'nst' and data.get("source") == "nst":
        data["task"] = "transcribe"
        return data
    elif subcorpus == 'ellipses' and "…" in data.get("text", ""):
        data["task"] = "transcribe"
        return data
    elif subcorpus == 'hesitation':
        # Define hesitation patterns
        hesitation_patterns = [r'\behh?\b', r'\behm\b', r'\bmmm?\b']
        text = data.get("text", "")
        # Replace "Ehh" or "ehh" with "Eh" or "eh" respectively
        text = re.sub(r'\bEhh\b', 'Eh', text, flags=re.IGNORECASE)
        # Check if any hesitation pattern is found in the text
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in hesitation_patterns):
            data["text"] = text
            data["task"] = "transcribe"
            return data
    return None

def read_files(input_folder, output_file, subcorpus):
    line_count = 0
    with jsonlines.open(output_file, mode='w') as writer:
        for filename in filter(lambda f: f.endswith('.jsonl'), os.listdir(input_folder)):
            with jsonlines.open(os.path.join(input_folder, filename)) as reader:
                for data in reader:
                    processed_data = process_line(data, subcorpus)
                    if processed_data:
                        writer.write(processed_data)
                        line_count += 1
    return line_count

def main():
    parser = argparse.ArgumentParser(description='Process JSON lines files based on subcorpus routines.')
    parser.add_argument('--input_folder', type=str, required=True, help='Input folder with JSON lines files.')
    parser.add_argument('--output_file', type=str, required=True, help='Output JSON lines file.')
    parser.add_argument('--subcorpus', type=str, required=True, help='Subcorpus routine to use.',
                        choices=['nst', 'ellipses', 'hesitation'])

    args = parser.parse_args()

    if not os.path.isdir(args.input_folder):
        print(f"Input folder '{args.input_folder}' does not exist.", file=sys.stderr)
        sys.exit(1)

    line_count = read_files(args.input_folder, args.output_file, args.subcorpus)
    print(f"{line_count} lines written to {args.output_file}")

if __name__ == "__main__":
    main()
