import pandas as pd
import argparse
import string
from tqdm import tqdm

def find_common_sequence(predictions, target, min_length=3, max_length=50):
    predictions_words = [pred.lower().translate(str.maketrans('', '', string.punctuation)).split() for pred in predictions]
    target_words = target.lower().translate(str.maketrans('', '', string.punctuation)).split()

    for length in range(max_length, min_length - 1, -1):
        for i in range(len(predictions_words[0]) - length + 1):
            ngram = ' '.join(predictions_words[0][i:i+length])
            if all(word not in target_words for word in ngram.split()):
                if all(all(word in pred for word in ngram.split()) for pred in predictions_words[1:]):
                    return ngram, length
    return "", 0

def main(input_filename, min_length, max_length):
    data = pd.read_csv(input_filename, sep='\t', dtype=str)
    data = data.fillna('')

    statistics = {i: 0 for i in range(min_length, max_length + 1)}
    flagged_lines = 0
    unflagged_lines = 0

    with tqdm(total=len(data), position=0, leave=True) as pbar:
        for index, row in data.iterrows():
            target = row['target']
            predictions = row[2:]
            sequence, length = find_common_sequence(predictions, target, min_length, max_length)
            if length > 0:
                tqdm.write(f"{row['id']} - {length} - {sequence}")  # Using tqdm.write instead of print
                flagged_lines += 1
                statistics[length] += 1
            else:
                unflagged_lines += 1
            pbar.update(1)

    print("\nStatistics:")
    print("Flagged lines:", flagged_lines)
    print("Unflagged lines:", unflagged_lines)
    for k, v in statistics.items():
        if v:
            print(f"{k} words: {v}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", required=True, help="Input TSV file")
    parser.add_argument("--min_length", type=int, default=3, help="Minimum length for common sequence")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length for common sequence")
    args = parser.parse_args()

    main(args.input_filename, args.min_length, args.max_length)

