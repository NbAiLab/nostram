import pandas as pd
import argparse
import string
from tqdm import tqdm
from collections import Counter

def find_common_sequence(predictions, target, min_length=3, max_length=50):
    target_words = target.lower().translate(str.maketrans('', '', string.punctuation)).split()

    # Function to get n-grams from a list of words
    def get_ngrams(words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]

    # Find all n-grams for each prediction
    predictions_ngrams = []
    for prediction in predictions:
        prediction_words = prediction.lower().translate(str.maketrans('', '', string.punctuation)).split()
        ngrams = []
        for n in range(min_length, max_length + 1):
            ngrams.extend(get_ngrams(prediction_words, n))
        predictions_ngrams.append(set(ngrams))

    # Find common n-grams across all but one prediction
    common_ngrams = set.intersection(*predictions_ngrams[:2])
    for prediction_ngrams in predictions_ngrams[2:]:
        common_ngrams = common_ngrams.intersection(prediction_ngrams)
        if len(common_ngrams) == 0:
            break

    # Find the longest common n-gram not present in the target
    best_ngram = ""
    best_length = 0
    for ngram in common_ngrams:
        if all(word not in target_words for word in ngram.split()) and len(ngram.split()) > best_length:
            best_ngram = ngram
            best_length = len(ngram.split())

    return best_ngram, best_length


def main(input_filename, output_filename, min_length, max_length, verbose):
    data = pd.read_csv(input_filename, sep='\t', dtype=str)
    data = data.fillna('')

    # Check if 'max_ngrams_not_in_target' exists; if not, insert it
    if 'max_ngrams_not_in_target' not in data.columns:
        data.insert(data.columns.get_loc('target'), 'max_ngrams_not_in_target', 0)
    else:
        data['max_ngrams_not_in_target'] = 0

    statistics = {i: 0 for i in range(min_length, max_length + 1)}
    flagged_lines = 0
    unflagged_lines = 0

    with tqdm(total=len(data), position=0, leave=True) as pbar:
        for index, row in data.iterrows():
            target = row['target']
            predictions = row[3:]  # Adjust index based on your data structure
            sequence, length = find_common_sequence(predictions, target, min_length, max_length)
            data.at[index, 'max_ngrams_not_in_target'] = length  # Update the column

            if length > 0:
                flagged_lines += 1
                statistics[length] += 1
                if verbose:
                    tqdm.write(f"{row['id']} - {length} - {sequence}")
            else:
                unflagged_lines += 1
            pbar.update(1)

    # Save the modified DataFrame to a new TSV file
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
    parser.add_argument("--output_filename", required=True, help="Output TSV file")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length for common sequence")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length for common sequence")
    parser.add_argument("--verbose", action="store_true", help="Display detailed output")
    args = parser.parse_args()

    main(args.input_filename, args.output_filename, args.min_length, args.max_length, args.verbose)

