import pandas as pd
import argparse
import string
from tqdm import tqdm

def find_insertions(target, predictions, min_length=1, max_length=50):
    target_words = target.lower().translate(str.maketrans('', '', string.punctuation)).split()
    for prediction in predictions:
        if prediction.lower().translate(str.maketrans('', '', string.punctuation)) == target.lower().translate(str.maketrans('', '', string.punctuation)):
            return "", 0

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

def main(input_filename, output_filename, min_length, max_length, verbose):
    data = pd.read_csv(input_filename, sep='\t', dtype=str)
    data = data.fillna('')

    if 'max_ngrams_not_in_pred' not in data.columns:
        target_col_index = data.columns.get_loc('target')
        data.insert(target_col_index, 'max_ngrams_not_in_pred', 0)

    statistics = {i: 0 for i in range(min_length, max_length + 1)}
    flagged_lines = 0
    unflagged_lines = 0

    with tqdm(total=len(data), position=0, leave=True) as pbar:
        for index, row in data.iterrows():
            target = row['target']
            predictions = row[data.columns.get_loc('target')+1:]
            insertion, length = find_insertions(target, predictions, min_length, max_length)
            data.at[index, 'max_ngrams_not_in_pred'] = length
            
            if length > 0:
                if verbose:
                    tqdm.write(f"{row['id']} - {length} - {insertion}")
                flagged_lines += 1
                statistics[length] += 1
            else:
                unflagged_lines += 1

            pbar.update(1)

    data.to_csv(output_filename, sep='\t', index=False)

    print("\nStatistics:")
    print("Flagged lines:", flagged_lines)
    print("Unflagged lines:", unflagged_lines)
    for k, v in statistics.items():
        if v:
            print(f"{k} words: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", required=True, help="Input TSV file")
    parser.add_argument("--output_file", required=True, help="Output TSV file")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length for insertion")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length for insertion")
    parser.add_argument("--verbose", action='store_true', help="Verbose output")
    args = parser.parse_args()

    main(args.input_filename, args.output_file, args.min_length, args.max_length, args.verbose)

