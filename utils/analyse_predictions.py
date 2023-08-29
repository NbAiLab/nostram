import pandas as pd
import argparse
import string
import re
from collections import Counter
from pandarallel import pandarallel

pandarallel.initialize()

def find_insertions(target, predictions, min_length=1, max_length=50):
    target_words = set(target.lower().split())
    max_ngram = ""
    max_length = 0
    for prediction in predictions:
        prediction_words = prediction.lower().split()
        ngram_counter = Counter()
        for n in range(min_length, max_length + 1):
            ngrams = [' '.join(prediction_words[i:i+n]) for i in range(len(prediction_words) - n + 1)]
            ngram_counter.update(ngrams)
        for ngram, count in ngram_counter.items():
            ngram_words = set(ngram.split())
            if not ngram_words.issubset(target_words):
                length = len(ngram_words)
                if length > max_length:
                    max_ngram = ngram
                    max_length = length
    return max_ngram, max_length

def find_common_sequence(predictions, target, min_length=3, max_length=50):
    target_words = target.lower().translate(str.maketrans('', '', string.punctuation)).split()
    def get_ngrams(words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    predictions_ngrams = []
    for prediction in predictions:
        prediction_words = prediction.lower().translate(str.maketrans('', '', string.punctuation)).split()
        ngrams = []
        for n in range(min_length, max_length + 1):
            ngrams.extend(get_ngrams(prediction_words, n))
        predictions_ngrams.append(set(ngrams))
    common_ngrams = set.intersection(*predictions_ngrams[:2])
    for prediction_ngrams in predictions_ngrams[2:]:
        common_ngrams = common_ngrams.intersection(prediction_ngrams)
        if len(common_ngrams) == 0:
            break
    best_ngram = ""
    best_length = 0
    for ngram in common_ngrams:
        if all(word not in target_words for word in ngram.split()) and len(ngram.split()) > best_length:
            best_ngram = ngram
            best_length = len(ngram.split())
    return best_ngram, best_length

def clean_text(text):
    return re.sub(r'[^a-ø]', ' ', str(text).lower()).strip()

def analyze_row(row, min_length, max_length):
    target = row['target']
    clean_target = clean_text(target)
    predictions = row[9:]
    clean_predictions = [clean_text(p) for p in predictions]

    insertion, length_1 = find_insertions(target, predictions, min_length, max_length)
    sequence, length_2 = find_common_sequence(predictions, target, min_length, max_length)

    first_word = clean_target.split()[0] if clean_target else ""
    last_word = clean_target.split()[-1] if clean_target else ""
    first_word_predicted = int(any(first_word in p.split() for p in clean_predictions))
    last_word_predicted = int(any(last_word in p.split() for p in clean_predictions))

    min_words = min(len(p.split()) for p in clean_predictions) if clean_predictions else 0
    max_words = max(len(p.split()) for p in clean_predictions) if clean_predictions else 0

    return pd.Series([max_words, min_words, last_word_predicted, first_word_predicted, length_1, length_2])

def main(input_filename, output_filename, min_length, max_length):
    data = pd.read_csv(input_filename, sep='\t', dtype=str)
    data = data.fillna('')
    new_cols = ['max_words_predicted', 'min_words_predicted', 'last_word_predicted', 'first_word_predicted', 'max_ngrams_not_in_pred', 'max_ngrams_not_in_target']

    for col in reversed(new_cols):
        if col not in data.columns:
            data.insert(data.columns.get_loc('target'), col, 0)
        else:
            data[col] = 0

    data[new_cols] = data.parallel_apply(analyze_row, axis=1, args=(min_length, max_length))

    data.to_csv(output_filename, sep='\t', index=False)

    print("\nSummary Statistics:")
    for col in new_cols:
        print(f"{col}: {data[col].sum()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", required=True, help="Input TSV file")
    parser.add_argument("--output_filename", required=True, help="Output TSV file")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length for common sequence and insertion")
    parser.add_argument("--max_length", type=int, default=50, help="Maximum length for common sequence and insertion")
    args = parser.parse_args()

    main(args.input_filename, args.output_filename, args.min_length, args.max_length)
