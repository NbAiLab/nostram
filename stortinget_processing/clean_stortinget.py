import argparse
import pandas as pd
import jsonlines
import string

WORD_COUNT_THRESHOLD = 4
MAX_WAV2VEC_WER = 0.6
MATCH_ALL_SUBNAMES = True
DROP_TEMP_FIELD = True
MIN_VERBOSITY_LEVEL = 4
LAST_SEQUENCE_LENGTH = 5
LAST_SEQUENCE_WORD_RATE = 0.1

def remove_punctuation(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator).strip()

def filter_lines(input_file_name, output_file_name):
    print(f"Starting to process {input_file_name}")

    with jsonlines.open(input_file_name) as reader:
        data = [line for line in reader]

    df = pd.DataFrame(data)

    if 'names' not in df:
        raise ValueError("Field 'names' must exist in the input file.")
    if 'wav2vec_wer' not in df:
        raise ValueError("Field 'wav2vec_wer' must exist in the input file.")
    if 'wav2vec_text' not in df:
        raise ValueError("Field 'wav2vec_text' must exist in the input file.")
    if 'verbosity_level' not in df:
        raise ValueError("Field 'verbosity_level' must exist in the input file.")

    filtered_df, dropped_counts = check_lines(df)

    print("Total lines:", len(df))
    print("Lines saved:", len(filtered_df))
    print("Lines dropped:", sum(dropped_counts.values()))
    print_summary(dropped_counts)

    if DROP_TEMP_FIELD:
        filtered_df = filtered_df.drop(["wav2vec_text", "names"], axis=1)

    with jsonlines.open(output_file_name, mode='w') as writer:
        writer.write_all(filtered_df.to_dict(orient='records'))

def check_lines(df):
    filtered_rows = []
    dropped_counts = {
        "Word Count": 0,
        "Names": 0,
        "WER Score": 0,
        "Verbosity Level": 0,
        "Last Sequence Word Rate": 0
    }

    for _, row in df.iterrows():
        if pd.isna(row['names']):
            valid_word_count = check_word_count(row)
            valid_wer_score = check_wer(row)
            valid_verbosity_level = check_verbosity_level(row)
            valid_last_sequence_overlap = check_last_sequence_overlap(row)

            if valid_word_count and valid_wer_score and valid_verbosity_level and valid_last_sequence_overlap:
                filtered_rows.append(row)
            else:
                if not valid_word_count:
                    dropped_counts["Word Count"] += 1
                if not valid_wer_score:
                    dropped_counts["WER Score"] += 1
                if not valid_verbosity_level:
                    dropped_counts["Verbosity Level"] += 1
                if not valid_last_sequence_overlap:
                    dropped_counts["Last Sequence Word Rate"] += 1
        else:
            valid_names = check_names(row)
            valid_word_count = check_word_count(row)
            valid_wer_score = check_wer(row)
            valid_verbosity_level = check_verbosity_level(row)
            valid_last_sequence_overlap = check_last_sequence_overlap(row)

            if valid_names and valid_word_count and valid_wer_score and valid_verbosity_level and valid_last_sequence_overlap:
                filtered_rows.append(row)
            else:
                if not valid_names:
                    dropped_counts["Names"] += 1
                if not valid_word_count:
                    dropped_counts["Word Count"] += 1
                if not valid_wer_score:
                    dropped_counts["WER Score"] += 1
                if not valid_verbosity_level:
                    dropped_counts["Verbosity Level"] += 1
                if not valid_last_sequence_overlap:
                    dropped_counts["Last Sequence Word Rate"] += 1

    return pd.DataFrame(filtered_rows), dropped_counts

def check_names(row):
    if pd.isna(row['names']):
        return True

    sub_names = [subname.lower() for name in row['names'].split(',') for subname in name.split()]
    wav2vec_text = row['wav2vec_text'].lower()

    if MATCH_ALL_SUBNAMES:
        if all(subname in wav2vec_text for subname in sub_names):
            return True
    else:
        if any(subname in wav2vec_text for subname in sub_names):
            return True

    print("Line dropped because it does not contain necessary names.")
    print("Text:", row['text'])
    print("Wav2Vec Text:", row['wav2vec_text'])
    print("Names:", row['names'])
    print("Missing Names:", ", ".join(sub_name for sub_name in sub_names if sub_name not in wav2vec_text))
    print("--------------------------------------------------")
    return False

def check_word_count(row):
    word_count = len(row['text'].split())
    if word_count < WORD_COUNT_THRESHOLD:
        print("Line dropped because the length of text is less than", WORD_COUNT_THRESHOLD)
        print("Text:", row['text'])
        print("Word Count:", word_count)
        print("--------------------------------------------------")
        return False
    return True

def check_wer(row):
    if pd.notna(row['wav2vec_wer']) and row['wav2vec_wer'] <= MAX_WAV2VEC_WER:
        return True

    print("Line dropped because the wav2vec_wer score is below the minimum threshold.")
    print("Text:", row['text'])
    print("Wav2Vec Text:", row['wav2vec_text'])
    print("WER Score:", row['wav2vec_wer'])
    print("--------------------------------------------------")
    return False

def check_verbosity_level(row):
    if pd.notna(row['verbosity_level']) and row['verbosity_level'] >= MIN_VERBOSITY_LEVEL:
        return True

    print("Line dropped because the verbosity_level is below the minimum level threshold.")
    print("Text:", row['text'])
    print("Wav2Vec Text:", row['wav2vec_text'])
    print("Verbosity Level:", row['verbosity_level'])
    print("--------------------------------------------------")
    return False



def check_last_sequence_overlap(row):
    text = row['text'].lower()
    wav2vec_text = row['wav2vec_text'].lower()

    if len(text.split()) < LAST_SEQUENCE_LENGTH or len(wav2vec_text.split()) < LAST_SEQUENCE_LENGTH:
        return False

    last_sequence_text = text.split()[-(LAST_SEQUENCE_LENGTH + 3):]
    last_sequence_wav2vec_text = wav2vec_text.split()[-LAST_SEQUENCE_LENGTH:]

    translator = str.maketrans('', '', string.punctuation)
    last_sequence_text = [word.translate(translator) for word in last_sequence_text]
    last_sequence_wav2vec_text = [word.translate(translator) for word in last_sequence_wav2vec_text]

    overlap_found = any(word in last_sequence_wav2vec_text for word in last_sequence_text)

    if overlap_found:
        return True

    print("Line dropped because the last sequence word overlap is below the minimum rate.")
    print("Text:", row['text'])
    print("Wav2Vec Text:", row['wav2vec_text'])
    print("Last Sequence (Text):", " ".join(last_sequence_text))
    print("Last Sequence (Wav2Vec Text):", " ".join(last_sequence_wav2vec_text))
    print("Last Sequence Word Overlap: 0")
    print("Total Words in Last Sequence:", len(last_sequence_text))
    print("--------------------------------------------------")
    return False




def print_summary(dropped_counts):
    print("\nSummary:")
    for reason, count in dropped_counts.items():
        if count > 0:
            print(f"{count} dropped because of {reason}")
    print("--------------------------------------------------")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Filter lines based on name matching, word count, WER score, verbosity level, and last sequence word rate.')
    parser.add_argument('--input_file_name', type=str, required=True, help='The input jsonlines file')
    parser.add_argument('--output_file_name', type=str, required=True, help='The output jsonlines file')

    args = parser.parse_args()
    filter_lines(args.input_file_name, args.output_file_name)
