import pandas as pd
import argparse
import string

def find_insertions(target, predictions, min_length=5, max_length=25):
    target_words = target.lower().translate(str.maketrans('', '', string.punctuation)).split()
    best_length = 0
    best_insertion = ""
    for length in range(max_length, min_length - 1, -1):
        for i in range(len(target_words) - length + 1):
            ngram = ' '.join(target_words[i:i+length])
            if all(all(word not in prediction.lower().translate(str.maketrans('', '', string.punctuation)) for word in ngram.split()) for prediction in predictions):
                if length > best_length:
                    best_length = length
                    best_insertion = ngram
    return best_insertion, best_length

def main(input_filename, min_length, max_length):
    data = pd.read_csv(input_filename, sep='\t', dtype=str)
    data = data.fillna('')

    statistics = {i: 0 for i in range(min_length, max_length + 1)}
    flagged_lines = 0
    unflagged_lines = 0

    for index, row in data.iterrows():
        target = row['target']
        predictions = row[2:]
        insertion, length = find_insertions(target, predictions, min_length, max_length)
        if length > 0:
            print(f"{row['id']} - {length} - {insertion}")
            flagged_lines += 1
            statistics[length] += 1
        else:
            unflagged_lines += 1

    print("\nStatistics:")
    print("Flagged lines:", flagged_lines)
    print("Unflagged lines:", unflagged_lines)
    for k, v in statistics.items():
        print(f"{k} words: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", required=True, help="Input TSV file")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length for insertion")
    parser.add_argument("--max_length", type=int, default=25, help="Maximum length for insertion")
    args = parser.parse_args()

    main(args.input_filename, args.min_length, args.max_length)

