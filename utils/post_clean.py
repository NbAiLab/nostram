# Introduction
# This script performs text analysis on a given TSV file. It identifies various statistics
# related to text prediction such as the minimum and maximum number of words predicted,
# whether the first and last words are correctly predicted, and more. The script is parallelized
# for performance.

import argparse, string,re, os, sys
from collections import Counter
from pandarallel import pandarallel
from tqdm import tqdm
import pandas as pd
import logging
import json
import time
from jiwer import wer
from fuzzywuzzy import fuzz

start_time = time.time()


def exec_time():
    end_time = time.time()
    out = str(round(end_time - start_time, 1)) + " seconds"
    return out

def load_json(jsonline):
    data = pd.read_json(jsonline, lines=True)

    logger.info(f'***  Json parsed. {len(data)} lines. ({exec_time()})')
    print(f'***  Json parsed with {len(data)} lines. ({exec_time()})')

    return data


def read_config(cfile):
    try:
        f = open(cfile, "r")
        config = json.load(f)
    except:
        logger.info(
            "Error. There has to be a valid config-file in the output directory")
        print("Error. There has to be a valid config-file in the output directory")
        exit()

    return config


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        data.to_json(file, orient='records', lines=True, force_ascii=False)
    logger.info(f'Saved jsonl as "{filename}"')

# Function to find insertions that do not exist in the predictions.
def find_insertions(target, predictions, min_length=1, max_length=999):
    target_words = str(target).lower().split()
    max_ngram_not_in_pred = ""
    max_ngram_length = 0

    target_ngrams = set([' '.join(target_words[i:i+n]) for n in range(min_length, max_length + 1) for i in range(len(target_words) - n + 1)])

    for prediction in predictions:
        prediction_words = str(prediction).lower().split()

        # Generate n-grams for prediction
        prediction_ngrams = set([' '.join(prediction_words[i:i+n]) for n in range(min_length, max_length + 1) for i in range(len(prediction_words) - n + 1)])

        # Find n-grams in target but not in prediction
        ngrams_not_in_pred = target_ngrams - prediction_ngrams

        # Update the longest n-gram not in any prediction
        for ngram in ngrams_not_in_pred:
            ngram_length = len(ngram.split())
            if ngram_length > max_ngram_length:
                max_ngram_length = ngram_length
                max_ngram_not_in_pred = ngram

    # If all predictions match the target perfectly, max_ngram_length should be 0
    if max_ngram_length == len(target_words):
        max_ngram_length = 0
        max_ngram_not_in_pred = ""

    return max_ngram_not_in_pred, max_ngram_length

# Function to find common n-grams not in target
def find_common_sequence(predictions, target, min_length=1, max_length=999):
    # Prepare target words
    target_words = str(target).lower().translate(str.maketrans('', '', string.punctuation)).split()
    def get_ngrams(words, n):
        return [' '.join(words[i:i+n]) for i in range(len(words) - n + 1)]
    predictions_ngrams = []
    # Loop through each prediction to find common sequences.
    for prediction in predictions:
        prediction_words = str(prediction).lower().translate(str.maketrans('', '', string.punctuation)).split()
        ngrams = []
        for n in range(min_length, max_length + 1):
            ngrams.extend(get_ngrams(prediction_words, n))
        predictions_ngrams.append(set(ngrams))
    # Intersect to find common n-grams
    common_ngrams = set.intersection(*predictions_ngrams[:2])
    for prediction_ngrams in predictions_ngrams[2:]:
        common_ngrams = common_ngrams.intersection(prediction_ngrams)
        if len(common_ngrams) == 0:
            break
    ngram_not_in_target = ""
    best_length = 0
    # Identify the best n-gram based on length
    for ngram in common_ngrams:
        if all(word not in target_words for word in ngram.split()) and len(ngram.split()) > best_length:
            ngram_not_in_target = ngram
            best_length = len(ngram.split())
    return ngram_not_in_target, best_length

# Text cleaning function
def clean_text(text):
    return re.sub(r'[^a-zA-ZæøåÆØÅ]', ' ', str(text).lower()).strip()

# Function to analyze a single row of data
def analyze_row(row, *config):
    
    #min_length = config[0]['min_seq_length']
    #max_length = config[0]['max_seq_length']

    # Process target and prediction data
   
    clean_target = clean_text(row['text'])
    num_words_target = len(clean_target.split())
    lang_cols = config[0]['predictions'][row["text_language"]]
    
    predictions = row[lang_cols]
    clean_predictions = [clean_text(p) for p in predictions]
    
    # Find insertions and sequences
    ngram_not_in_pred, max_ngrams_not_in_pred = find_insertions(clean_target, predictions)
    ngram_not_in_target, max_ngrams_not_in_target = find_common_sequence(predictions, clean_target)
    
    # Check if first and last words are predicted
    first_word = clean_target.split()[0] if clean_target else ""
    last_word = clean_target.split()[-1] if clean_target else ""
    
    #Fuzzy match
    fuzz_threshold = config[0]['fuzz_threshold']
    first_word_predicted = int(any(fuzz.ratio(first_word, p.split()[0]) >= fuzz_threshold if p.split() else False for p in clean_predictions))
    last_word_predicted = int(any(fuzz.ratio(last_word, p.split()[-1]) >= fuzz_threshold if p.split() else False for p in clean_predictions))

    #fuzz_threshold = 90
    #first_word_predicted_strict = int(any(fuzz.ratio(first_word, p.split()[0]) >= fuzz_threshold if p.split() else False for p in clean_predictions))
    #last_word_predicted_strict = int(any(fuzz.ratio(last_word, p.split()[-1]) >= fuzz_threshold if p.split() else False for p in clean_predictions))

    #if first_word_predicted_strict != first_word_predicted:
    #    print(f"{first_word} - {clean_predictions}")
    
    # Find min and max words in predictions
    min_words = min(len(p.split()) for p in clean_predictions) if clean_predictions else 0
    max_words = max(len(p.split()) for p in clean_predictions) if clean_predictions else 0
    
    whisper_models = lang_cols
    # Calculate the WER-score for the tested models (clean_target vs clean_predictions)
    whisper_wer_scores = [wer(clean_target, p) for p in clean_predictions]
    
    #Check if the word "president" is in at least one of the clean_preditions and not in target
    president = int(any("president" in p.split() and "president" not in clean_target.split() for p in clean_predictions))
    if president == 1:
        print(f"{clean_target} - {clean_predictions}")
    
    
    
    
    # Find the best of the WER scores and the corresponding model
    if whisper_wer_scores:
        whisper_wer = min(whisper_wer_scores)
        
        best_model_index = whisper_wer_scores.index(whisper_wer)
        whisper_best_model = whisper_models[best_model_index]
    else:
        whisper_wer = 1
        whisper_best_model = None

    if whisper_wer > 1:
        whisper_wer = 1
    
    if any([
        first_word_predicted == 0, 
        last_word_predicted == 0, 
        max_ngrams_not_in_pred > 3,
        max_ngrams_not_in_target > 3,
        president == 1
    ]):
        delete = 1
    else:
        delete = 0
    
    # Return results
    return pd.Series([num_words_target, max_words, min_words, last_word_predicted, first_word_predicted, ngram_not_in_target, max_ngrams_not_in_target, ngram_not_in_pred, max_ngrams_not_in_pred, whisper_wer, whisper_models, whisper_wer_scores, whisper_best_model, president, delete])


# Main function to execute the script
def main(args):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    # Invoke logging
    log_name = os.path.basename(args.input_filename).replace(".json", "") + ".log"

    # Create directories if they do not exist
    if not os.path.exists(args.output_folder + "/log"):
        print(args.output_folder)
        os.makedirs(args.output_folder + "/log")

    handler = logging.FileHandler(
        filename=os.path.join(args.output_folder, "log/", log_name),
        mode='w'
    )
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(
        {"DEBUG": logging.DEBUG, "INFO": logging.INFO}[args.log_level])
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)


    config_file_path = os.path.join(
        args.output_folder.replace("/train", "/").replace("/test", "/").replace("/validation", "/"), "config.json")
    config = read_config(config_file_path)

    print(f'*** Starting to process: {args.input_filename}')
    
    # Read data and prepare columns
    data: pd.DataFrame = load_json(args.input_filename)
    data = data.fillna('')
    
    new_cols = ['num_words_target', 'max_words_predicted', 'min_words_predicted', 'last_word_predicted', 'first_word_predicted', 'ngram_not_in_target','max_ngrams_not_in_target', 'ngram_not_in_pred','max_ngrams_not_in_pred', 'whisper_wer', 'whisper_models', 'whisper_wer_scores', 'whisper_best_model', 'president','delete']
    for col in reversed(new_cols):
        if col not in data.columns:
            data.insert(data.columns.get_loc('text'), col, 0)
        else:
            data[col] = 0
    
    # Initialize pandarallel with progress_bar
    pandarallel.initialize(progress_bar=True)
    
    # Perform parallel analysis
    data[new_cols] = data.apply(analyze_row, axis=1, args=(config,))

    #data[new_cols] = data.parallel_apply(analyze_row, axis=1, args=(config,))
    
    # If prune flag is set, keep only approved columns
    if args.prune:
        data = data[data['delete'] != 1]
        approved_cols = ["id", "group_id", "source", "audio_language", "previous_text", "translated_text_no", "translated_text_nn",
                         "translated_text_en", "translated_text_es", "timestamped_text", "wav2vec_wer", "whisper_wer", "text_language", "text", "verbosity_level"]
        data = data[approved_cols]
 
    
    # Save it as jsonl
    output_filename = os.path.join(
        args.output_folder, os.path.basename(args.input_filename))
    save_json(data, output_filename)
    logger.info(
        f'*** Finished processing file.'
        f'\n{len(data)} lines is written to {os.path.join(args.output_folder, os.path.basename(args.input_filename))}. ({exec_time()})')
    
# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", required=True, help="Input JSON lines file")
    parser.add_argument("--output_folder", required=True, help="Output folder")
    parser.add_argument('--log_level', required=False, default="INFO",
                        help='Set logging level to DEBUG to get a report on all decisions')
    parser.add_argument('--prune', action='store_true', default=False,
                        help='Remove all columns not in the approved list and all lines marked for deletion')
    args = parser.parse_args()
    
        # Invoke logger globally
    logger = logging.getLogger()

    if args.log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif args.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        print("Log level not accepted")
        exit()
    
    main(args)
