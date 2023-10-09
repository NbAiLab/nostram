# Introduction
# This script performs text analysis on a given TSV file. It identifies various statistics
# related to text prediction such as the minimum and maximum number of words predicted,
# whether the first and last words are correctly predicted, and more. The script is parallelized
# for performance.
# Import required libraries
import argparse
import string
import re
import os
import sys
from collections import Counter
from pandarallel import pandarallel
from tqdm import tqdm
import pandas as pd
import logging
import json
import time
from jiwer import wer
from fuzzywuzzy import fuzz
from jsonschema import validate, ValidationError


# Initialize the start time
start_time = time.time()

def get_dtypes_from_schema(schema):
    properties = schema["properties"]
    dtypes = {}
    for key, value in properties.items():
        if "type" in value:
            if isinstance(value["type"], list):
                if "null" in value["type"]:
                    dtypes[key] = 'object'
            elif value["type"] == "string":
                dtypes[key] = str
    return dtypes


# Calculate and return the elapsed execution time
def exec_time():
    end_time = time.time()
    return f"{round(end_time - start_time, 1)} seconds"

# Load JSON lines file into a Pandas DataFrame and log the duration
def load_json(jsonline):
    # Define the dataset schema for the input format
    schema = {
	    "type": "object",
	    "properties": {
            "id": {"type": "string"},
            "group_id": {"type": ["string", "null"]},
			"source": {"type": "string", "enum": ["nrk_tv", "nrk_tv_translate", "nrk_tv_silence", "nrk_tv_veryshort","stortinget", "nst", "fleurs", "audio_books"]},
			"audio_language": {"type": ["string", "null"]},
			"audio_duration": {"type": "integer"},
			"previous_text": {"type": ["string", "null"]},
			"text_language": {"type": "string", "enum": ["no", "nn", "en", "es"]},
			"text": {"type": "string"},
			"translated_text_no": {"type": ["string", "null"]},
			"translated_text_nn": {"type": ["string", "null"]},
			"translated_text_en": {"type": ["string", "null"]},
			"translated_text_es": {"type": ["string", "null"]},
			"timestamped_text": {"type": ["string", "null"]},
			"wav2vec_wer": {"type": ["number", "null"]},
			"whisper_wer": {"type": ["number", "null"]},
			"verbosity_level": {"type": ["integer", "null"], "enum": [1, 2, 3, 4, 5, 6, None]}
        },
		"required": ["id", "group_id", "source", "audio_language","previous_text","translated_text_no","translated_text_nn","translated_text_en","translated_text_es","timestamped_text","wav2vec_wer","whisper_wer","text_language", "text","verbosity_level"],
	    "additionalProperties": False
    }
    dtypes = get_dtypes_from_schema(schema)
    data = pd.read_json(jsonline, lines=True, dtype=dtypes)
    # Workaround there group_id is set as int
    string_columns = ['group_id']
    for col in string_columns:
        data[col] = data[col].astype(str)

    logger.info(f"***  Json parsed. {len(data)} lines. ({exec_time()})")
    
    return data

# Read configuration from a JSON file. Exit if file is invalid.
def read_config(cfile):
    try:
        with open(cfile, "r") as f:
            config = json.load(f)
    except:
        logger.info("Error. A valid config-file must exist in the output directory. Place in target, and move one level up when finished.")
        exit()
    return config

# Save Pandas DataFrame as a JSON lines file
def save_json(data, filename):
    # Replace empty strings with None directly
    data.replace("", None, inplace=True) 
    
    with open(filename, "w", encoding="utf-8") as file:
        data.to_json(file, orient="records", lines=True, force_ascii=False)

# Configure logging based on user-defined parameters
def configure_logging(args):
    # Initialize logger
    logger = logging.getLogger()
    
    # Set log level
    if args.log_level == "INFO":
        logger.setLevel(logging.INFO)
    elif args.log_level == "DEBUG":
        logger.setLevel(logging.DEBUG)
    else:
        print("Log level not accepted")
        exit()

    # Create log directory if not exists
    log_name = os.path.basename(args.input_filename).replace(".json", "") + ".log"
    log_dir = os.path.join(args.output_folder, "log/")
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Configure file handler for logs
    file_handler = logging.FileHandler(filename=os.path.join(log_dir, log_name), mode="w")
    formatter = logging.Formatter("%(asctime)s %(message)s")
    file_handler.setFormatter(formatter)
    file_handler.setLevel({"DEBUG": logging.DEBUG, "INFO": logging.INFO}[args.log_level])
    logger.addHandler(file_handler)

    # Configure stdout handler for logs
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.INFO)
    logger.addHandler(stdout_handler)
    
    return logger

def prepare_dataframe_columns(data: pd.DataFrame) -> (pd.DataFrame, list):
    #  Prepares the DataFrame by inserting new columns with default values if they are not already present.

    # Define new columns that are to be inserted
    new_cols = [
        "num_words_target",
        "max_words_predicted",
        "min_words_predicted",
        "last_word_predicted",
        "first_word_predicted",
        "ngram_not_in_target",
        "max_ngrams_not_in_target",
        "ngram_not_in_pred",
        "max_ngrams_not_in_pred",
        "whisper_wer",
        "whisper_models",
        "whisper_wer_scores",
        "whisper_best_model",
        "president",
        "delete",
    ]

    # Loop through the new columns in reverse order
    for col in reversed(new_cols):
        # If column does not exist, insert it at the position of 'text' column with default value 0
        if col not in data.columns:
            data.insert(data.columns.get_loc("text"), col, 0)
        # If column already exists, reset its value to 0
        else:
            data[col] = 0
    
    return data, new_cols

def prune_dataframe(data: pd.DataFrame) -> pd.DataFrame:
    # Removes rows and columns from the DataFrame based on predefined conditions.

    # Define the approved columns to keep
    approved_cols = [
        "id",
        "group_id",
        "source",
        "audio_duration",
        "audio_language",
        "previous_text",
        "translated_text_no",
        "translated_text_nn",
        "translated_text_en",
        "translated_text_es",
        "timestamped_text",
        "wav2vec_wer",
        "whisper_wer",
        "text_language",
        "text",
        "verbosity_level",
    ]

    # Remove rows where 'delete' column is 1
    data = data[data["delete"] != 1]

    # Keep only approved columns
    data = data[approved_cols]
    
    return data


def calculate_stats(df, config):
    # Calculate total row count
    total_count = len(df)

    # Compute condition counts
    counts = {
        'first_word': sum(df['first_word_predicted'] == config["first_word_predicted_threshold"]),
        'last_word': sum(df['last_word_predicted'] == config["last_word_predicted_threshold"]),
        'whisper_wer': sum(df['whisper_wer'] > config["max_wer_threshold"]),
        'max_ngrams_pred': sum(df['max_ngrams_not_in_pred'] > config["max_ngrams_not_in_pred_threshold"]),
        'max_ngrams_target': sum(df['max_ngrams_not_in_target'] > config["max_ngrams_not_in_target_threshold"]),
        'president': sum(df['president'] == config["president_value"]),
        'delete': sum(df['delete'] == 1)
    }

    # Compute percentages
    percents = {key: (val / total_count) * 100 for key, val in counts.items()}

    # Log percentages based on config flags
    config_map = {
        'first_word': 'check_first_word_predicted',
        'last_word': 'check_last_word_predicted',
        'whisper_wer': 'check_wer',
        'max_ngrams_pred': 'check_max_ngrams_not_in_pred',
        'max_ngrams_target': 'check_max_ngrams_not_in_target',
        'president': 'check_president'
    }
    logger.info("\n\n*** Percentages:")
    for key, cfg_key in config_map.items():
        if config[cfg_key]:
            logger.info(f"Percentage of {key} meeting condition: {percents[key]:.1f}%")

    logger.info(f"Percentage for deletion: {percents['delete']:.1f}%")


# Function to find insertions that do not exist in the predictions.
def find_insertions(target, predictions, min_length=1, max_length=999):
    target_words = str(target).lower().split()
    max_ngram_not_in_pred = ""
    max_ngram_length = 0

    target_ngrams = set(
        [
            " ".join(target_words[i : i + n])
            for n in range(min_length, max_length + 1)
            for i in range(len(target_words) - n + 1)
        ]
    )

    for prediction in predictions:
        prediction_words = str(prediction).lower().split()

        # Generate n-grams for prediction
        prediction_ngrams = set(
            [
                " ".join(prediction_words[i : i + n])
                for n in range(min_length, max_length + 1)
                for i in range(len(prediction_words) - n + 1)
            ]
        )

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
    target_words = (
        str(target).lower().translate(str.maketrans("", "", string.punctuation)).split()
    )

    def get_ngrams(words, n):
        return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]

    predictions_ngrams = []
    # Loop through each prediction to find common sequences.
    for prediction in predictions:
        prediction_words = (
            str(prediction)
            .lower()
            .translate(str.maketrans("", "", string.punctuation))
            .split()
        )
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
        if (
            all(word not in target_words for word in ngram.split())
            and len(ngram.split()) > best_length
        ):
            ngram_not_in_target = ngram
            best_length = len(ngram.split())
    return ngram_not_in_target, best_length


# Text cleaning function
def clean_text(text):
    return re.sub(r"[^a-zA-ZæøåÆØÅ]", " ", str(text).lower()).strip()


# Function to analyze a single row of data
def analyze_row(row, *config):
    config = config[0]
    # min_length = config[0]['min_seq_length']
    # max_length = config[0]['max_seq_length']

    # Process target and prediction data

    clean_target = clean_text(row["text"])
    num_words_target = len(clean_target.split())
    lang_cols = config["predictions"][row["text_language"]]

    predictions = row[lang_cols]
    clean_predictions = [clean_text(p) for p in predictions]

    # Find insertions and sequences
    ngram_not_in_pred, max_ngrams_not_in_pred = find_insertions(
        clean_target, predictions
    )
    ngram_not_in_target, max_ngrams_not_in_target = find_common_sequence(
        predictions, clean_target
    )

    # Check if first and last words are predicted
    first_word = clean_target.split()[0] if clean_target else ""
    last_word = clean_target.split()[-1] if clean_target else ""

    # Fuzzy match
    fuzz_threshold = config["fuzz_threshold"]
    first_word_predicted = int(
        any(
            fuzz.ratio(first_word, p.split()[0]) >= fuzz_threshold
            if p.split()
            else False
            for p in clean_predictions
        )
    )
    last_word_predicted = int(
        any(
            fuzz.ratio(last_word, p.split()[-1]) >= fuzz_threshold
            if p.split()
            else False
            for p in clean_predictions
        )
    )


    # Find min and max words in predictions
    min_words = (
        min(len(p.split()) for p in clean_predictions) if clean_predictions else 0
    )
    max_words = (
        max(len(p.split()) for p in clean_predictions) if clean_predictions else 0
    )

    whisper_models = lang_cols
    # Calculate the WER-score for the tested models (clean_target vs clean_predictions)
    if clean_target:
        whisper_wer_scores = [wer(clean_target, p) for p in clean_predictions]
    else:
        whisper_wer_scores = [1]


    # Check if the word "president" is in at least one of the clean_preditions and not in target
    president = int(
        any(
            "president" in p.split() and "president" not in clean_target.split()
            for p in clean_predictions
        )
    )

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

    # Delete conditions
    conditions = []
    if config["check_first_word_predicted"]:
        conditions.append(first_word_predicted == config["first_word_predicted_threshold"])

    if config["check_last_word_predicted"]:
        conditions.append(last_word_predicted == config["last_word_predicted_threshold"])

    if config["check_max_ngrams_not_in_pred"]:
        conditions.append(max_ngrams_not_in_pred > config["max_ngrams_not_in_pred_threshold"])

    if config["check_max_ngrams_not_in_target"]:
        conditions.append(max_ngrams_not_in_target > config["max_ngrams_not_in_target_threshold"])

    if config["check_president"]:
        conditions.append(president == config["president_value"])

    if config["check_wer"]:
        conditions.append(whisper_wer > config["max_wer_threshold"])
    
    delete = 1 if any(conditions) else 0
    
    # Return results
    return pd.Series(
        [
            num_words_target,
            max_words,
            min_words,
            last_word_predicted,
            first_word_predicted,
            ngram_not_in_target,
            max_ngrams_not_in_target,
            ngram_not_in_pred,
            max_ngrams_not_in_pred,
            whisper_wer,
            whisper_models,
            whisper_wer_scores,
            whisper_best_model,
            president,
            delete,
        ]
    )


# Main function to execute the script
def main(args):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)

    # Read config
    config = read_config(os.path.join(args.output_folder, "config.json"))

    logger.info(f"*** Starting to process: {args.input_filename}")

    # Read data and prepare columns
    data: pd.DataFrame = load_json(args.input_filename)
    data = data.fillna("")

    # Prepare dataframe columns
    data, new_cols = prepare_dataframe_columns(data)

    # Initialize pandarallel with progress_bar
    pandarallel.initialize(progress_bar=True)

    # Perform parallel analysis
    data[new_cols] = data.parallel_apply(analyze_row, axis=1, args=(config,))

    # Statistics
    calculate_stats(data, config)
    
    # If prune flag is set, keep only approved columns
    if args.prune:
        data = prune_dataframe(data)

    # Save it as jsonl
    output_filename = os.path.join(
        args.output_folder, os.path.basename(args.input_filename)
    )
    save_json(data, output_filename)
    logger.info(
        f"\n*** Finished processing file."
        f"\n{len(data)} lines is written to {os.path.join(args.output_folder, os.path.basename(args.input_filename))}. ({exec_time()})"
    )


# Entry point of the script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_filename", required=True, help="Input JSON lines file")
    parser.add_argument("--output_folder", required=True, help="Output folder")
    parser.add_argument(
        "--log_level",
        required=False,
        default="INFO",
        help="Set logging level to DEBUG to get a report on all decisions",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        default=False,
        help="Remove all columns not in the approved list and all lines marked for deletion",
    )
    args = parser.parse_args()

    logger = configure_logging(args)

    main(args)
