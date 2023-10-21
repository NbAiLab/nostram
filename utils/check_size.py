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

    try:
        data = json.loads(line)
        text = data.get("text", "")
        previous_text = data.get("previous_text", "")
        text_en = data.get("text_en", "")
        timestamped_text = data.get("timestamped_text", "")
        timestamped_text_en = data.get("timestamped_text_en", "")

        # Handle None values
        fields = [text, previous_text, text_en, timestamped_text, timestamped_text_en]
        fields = ["" if f is None else f for f in fields]
        text, previous_text, text_en, timestamped_text, timestamped_text_en = fields

        tokens = {
            "text": tokenizer.tokenize(text),
            "previous_text": tokenizer.tokenize(previous_text),
            "text_en": tokenizer.tokenize(text_en),
            "timestamped_text": tokenizer.tokenize(timestamped_text),
            "timestamped_text_en": tokenizer.tokenize(timestamped_text_en)
        }

        max_tokens = {
            "text": max_tokens_text,
            "previous_text": max_tokens_prev,
            "text_en": max_tokens_en,
            "timestamped_text": max_tokens_timestamped_text,
            "timestamped_text_en": max_tokens_timestamped_text_en
        }

        delete_line = False

        for field, token_list in tokens.items():
            if len(token_list) > max_tokens[field]:
                reason = f"Line exceeds {field} with {len(token_list)} tokens."
                deletion_reasons[field] += 1

                if field == "previous_text":
                    data["previous_text"] = None
                    if verbose:
                        print(reason + f" Setting to None. Original Content: {line}")
                    else:
                        print(reason + " Setting to None.")
                else:
                    if verbose:
                        print(reason + f" Content: {line}")
                    else:
                        print(reason)
                    delete_line = True

        if delete_line:
            return None, deletion_reasons

        return json.dumps(data), deletion_reasons
    except Exception as e:
        print(f"Error processing line: {line}. Reason: {str(e)}")
        return None, deletion_reasons

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

    processed_lines = df['processed'].dropna().tolist()

    if args.output_file:
        with open(args.output_file, "w") as f:
            f.writelines(processed_lines)

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
