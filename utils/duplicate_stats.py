import pandas as pd
import argparse

def find_duplicates(input_file, min_duplicates):
    # Load JSONL file into a Pandas DataFrame
    df = pd.read_json(input_file, lines=True)

    # Find duplicates in 'text' column
    duplicate_count = df['text'].value_counts()

    # Filter duplicates based on minimum count
    filtered_duplicates = duplicate_count[duplicate_count >= min_duplicates]

    # Create Markdown table header
    print("| text | number_of_duplicates | id's |")
    print("|------|----------------------|------|")

    # Populate table
    for text, count in filtered_duplicates.items():
        ids = df[df['text'] == text]['id'].tolist()
        print(f"| {text[:20]}... | {count} | {', '.join(ids[:2])}... |")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find duplicate texts in JSONL file.")
    parser.add_argument('--input_file', required=True, help="Path to the input JSONL file.")
    parser.add_argument('--min_duplicates', type=int, default=5, help="Minimum number of duplicates.")
    args = parser.parse_args()

    find_duplicates(args.input_file, args.min_duplicates)

