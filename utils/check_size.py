import argparse
import pandas as pd
from pandarallel import pandarallel
from transformers import AutoTokenizer
import sys
import json

# Initialize pandarallel
pandarallel.initialize(progress_bar=False)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai/whisper-tiny")

def process_line(line, max_tokens_text, max_tokens_prev, max_tokens_en, max_tokens_timestamped_text, max_tokens_timestamped_text_en, verbose):
    deletion_reasons = {"text": 0, "previous_text": 0, "text_en": 0, "timestamped_text": 0, "timestamped_text_en": 0}

    data = json.loads(line)
    modified_data = data.copy()  # Create a copy of the data to potentially modify

    delete_line = False

    fields_to_check = {
        "text": max_tokens_text,
        "text_en": max_tokens_en,
        "timestamped_text": max_tokens_timestamped_text,
        "timestamped_text_en": max_tokens_timestamped_text_en
    }

    for field, max_tokens in fields_to_check.items():
        current_text = modified_data.get(field, "")
        if current_text is None:
            current_text = ""
        tokens = tokenizer.tokenize(current_text)

        if len(tokens) > max_tokens:
            deletion_reasons[field] += 1
            if verbose:
                print(f"Line exceeds {field} with {len(tokens)} tokens. Content: {line}")
            else:
                print(f"Line exceeds {field} with {len(tokens)} tokens.")
            delete_line = True

    # Handle previous_text separately
    prev_text_tokens = tokenizer.tokenize(modified_data.get("previous_text", "") or "")
    if len(prev_text_tokens) > max_tokens_prev:
        deletion_reasons["previous_text"] += 1
        modified_data["previous_text"] = None
        reason = f"Line exceeds previous_text with {len(prev_text_tokens)} tokens. Setting to None."
        if verbose:
            print(reason + f" Original Content: {line}")
        else:
            print(reason)

    if delete_line:
        return None, deletion_reasons

    return json.dumps(modified_data), deletion_reasons

def main(args):
    # Read the input file
    with open(args.input_file, "r") as f:
        content = f.read()

    # Check if the file is a single JSON object (and not JSON-lines)
    try:
        json.loads(content)
        print(f"Error: {args.input_file} appears to be a JSON file, not a JSON-lines file. Skipping...")
        return
    except json.JSONDecodeError:
        # Expected error if the file is JSON-lines
        pass

    lines = content.splitlines()

    # Convert lines to DataFrame for parallel processing
    df = pd.DataFrame({'lines': lines})

    df['processed'], df['reasons'] = zip(*df['lines'].parallel_apply(process_line, args=(
        args.max_tokens_text, args.max_tokens_prev, args.max_tokens_en, 
        args.max_tokens_timestamped_text, args.max_tokens_timestamped_text_en, args.verbose)))

    # Calculate statistics
    total_lines = len(df)
    deleted_lines = df['processed'].isna().sum()
    retained_lines = total_lines - deleted_lines

    print(f"Total lines: {total_lines}")
    print(f"Deleted lines: {deleted_lines}")
    print(f"Retained lines: {retained_lines}")

    for field in ["text", "text_en", "timestamped_text", "timestamped_text_en"]:
        deletions = df['reasons'].apply(lambda x: x[field]).sum()
        print(f"Deleted due to '{field}': {deletions}")

    modified_prev_text = df['reasons'].apply(lambda x: x["previous_text"]).sum()
    print(f"Modified 'previous_text' by setting to None: {modified_prev_text}")

    processed_df = df['processed'].dropna()
    processed_lines = processed_df.tolist()

    if args.output_file:
        with open(args.output_file, "w") as f:
            for line in processed_lines:
                f.write(line + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tokenize and filter json-lines data.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input json-lines file.")
    parser.add_argument("--output_file", type=str, help="Path to the output json-lines file. If not specified, no file will be written.")
    parser.add_argument("--max_tokens_text", type=int, default=250, help="Maximum tokens allowed for the 'text' field.")
    parser.add_argument("--max_tokens_prev", type=int, default=180, help="Maximum tokens allowed for the 'previous_text' field.")
    parser.add_argument("--max_tokens_en", type=int, default=250, help="Maximum tokens allowed for the 'text_en' field.")
    parser.add_argument("--max_tokens_timestamped_text", type=int, default=250, help="Maximum tokens allowed for the 'timestamped_text' field.")
    parser.add_argument("--max_tokens_timestamped_text_en", type=int, default=250, help="Maximum tokens allowed for the 'timestamped_text_en' field.")
    parser.add_argument("--verbose", action="store_true", help="If set, the entire sentence is listed. If not, only the reason for deletion.")
    args = parser.parse_args()

    main(args)
