import functools
import math
import re
from collections import Counter
import os
import sys
import numpy as np
import pandas as pd
from pandarallel import pandarallel
from jiwer import wer
from sentence_transformers import SentenceTransformer, util
import argparse
import logging
import json

from extractor.detect_voice import VoiceDetector

pandarallel.initialize(use_memory_fs=False)
model = SentenceTransformer('NbAiLab/nb-sbert-base')


def _find_abs(a, b):
    if isinstance(a, str) and isinstance(b, str):
        a, b = text_to_counter(a), text_to_counter(b)
    intersect = a & b
    abs_a = sum(a.values()) if isinstance(a, Counter) else len(a)
    abs_b = sum(b.values()) if isinstance(b, Counter) else len(b)
    abs_i = sum(intersect.values()) if isinstance(
        intersect, Counter) else len(intersect)
    abs_ab = sum(a[k] * b[k] for k in intersect)
    return abs_a, abs_b, abs_i, abs_ab


def bert_similarity(st1, st2):
    embeddings = model.encode([st1, st2])
    cosine_scores = util.cos_sim(embeddings[0], embeddings[1])
    return cosine_scores[0][0].item()


def text_to_counter(text, multiset=True):
    init = Counter if multiset else set
    clean = re.sub(r'[^\w\s]', '', text).lower()
    return init(clean.split())


def match_percentage(source, target):
    source_set = text_to_counter(source)
    target_set = text_to_counter(target)
    common = source_set & target_set
    if not source_set:
        return 0.0
    return len(common) / len(source_set)


def last_isin(source, target):
    source_word_list = re.sub(r'[^\w\s]', '', source).lower().split()
    target_word_list = re.sub(r'[^\w\s]', '', target).lower().split()

    if source_word_list[-1] in target_word_list:
        output = 1
    else:
        output = 0

    return output


def last_islast(source, target):
    source_word_list = re.sub(r'[^\w\s]', '', source).lower().split()
    target_word_list = re.sub(r'[^\w\s]', '', target).lower().split()

    if len(target_word_list) == 0:
        return 0

    if source_word_list[-1] == target_word_list[-1]:
        output = 1
    else:
        output = 0

    return output


def cosine_similarity(a, b):
    abs_a, abs_b, abs_i, abs_ab = _find_abs(a, b)
    if 0 == abs_a or 0 == abs_b:
        return 1.
    return abs_ab / (math.sqrt(abs_a * abs_b))


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if token1[t1 - 1] == token2[t2 - 1]:
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if a <= b and a <= c:
                    distances[t1][t2] = a + 1
                elif b <= a and b <= c:
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def relative_levenshtein(a, b):
    return levenshtein_distance(a, b) / max(len(a), len(b))


def jaro_winkler_distance(st1, st2):
    """
    Compute Jaro-Winkler distance between two strings.
    https://rosettacode.org/wiki/Jaro-Winkler_distance#Python
    """
    if len(st1) < len(st2):
        st1, st2 = st2, st1
    len1, len2 = len(st1), len(st2)
    if len2 == 0:
        return 0.0
    delta = max(0, len2 // 2 - 1)
    flag = [False for _ in range(len2)]  # flags for possible transpositions
    ch1_match = []
    for idx1, ch1 in enumerate(st1):
        for idx2, ch2 in enumerate(st2):
            if idx1 + delta >= idx2 >= idx1 - delta and ch1 == ch2 and not flag[idx2]:
                flag[idx2] = True
                ch1_match.append(ch1)
                break

    matches = len(ch1_match)
    if matches == 0:
        return 1.0
    transpositions, idx1 = 0, 0
    for idx2, ch2 in enumerate(st2):
        if flag[idx2]:
            transpositions += (ch2 != ch1_match[idx1])
            idx1 += 1

    jaro = (matches / len1 + matches / len2 +
            (matches - transpositions / 2) / matches) / 3.0
    commonprefix = 0
    for i in range(min(4, len2)):
        commonprefix += (st1[i] == st2[i])

    return 1.0 - (jaro + commonprefix * 0.1 * (1 - jaro))


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


def xis_valid_mat_per_ber_sim(data: pd.DataFrame, min_mat_per, min_ber_sim):
    """
    Checks is mat_per and ber_sim is high enough
    """
    w2v_mat_per = data.w2v_mat_per
    w2v_ber_sim = data.w2v_ber_sim
    cond = (w2v_ber_sim.empty | w2v_mat_per.empty) | (pd.Series(
        w2v_ber_sim >= 0.3).astype(int) + pd.Series(w2v_mat_per >= 0.4).astype(int))
    cond = cond.map(lambda x: True if x >= 1 else False)
    cond = cond | (pd.isna(w2v_ber_sim) | pd.isna(w2v_mat_per))

    return cond


def is_valid_mat_per_ber_sim(data: pd.DataFrame, min_mat_per, min_ber_sim):
    """
    Checks if mat_per and ber_sim is high enough
    """
    w2v_mat_per = data["w2v_mat_per"]
    w2v_ber_sim = data["w2v_ber_sim"]
    cond1 = (w2v_ber_sim >= min_ber_sim) & (w2v_mat_per >= min_mat_per)
    cond2 = (pd.isna(w2v_ber_sim) | pd.isna(w2v_mat_per))
    cond = cond1 | cond2
    return cond


def run_vad(row, voice_detector, audio_folder):
    voice_detector.select_sourcefile(os.path.join(audio_folder, row.id.split("_")[0][-2:], row.id + ".mp3"))
    voice_segments = voice_detector.analyze()
    if voice_detector.is_tmp:
        os.remove(voice_detector.sourcefile)
    return len(voice_segments) <= 0 and isinstance(voice_segments, list)


def main(args):
    if not os.path.isfile(args.input_file):
        print(f"{args.input_file} is not a valid input file")
        sys.exit(1)

    if not os.path.isfile(args.transcript_file):
        print(f"{args.transcript_file} is not a valid transcript file")
        sys.exit(1)

    if args.audio_folder is not None and not os.path.isdir(args.audio_folder):
        print(f"{args.audio_folder} is not a valid audio folder")
        sys.exit(1)

    if not os.path.isdir(args.output_folder):
        print(f"{args.output_folder} is not a valid output folder")
        sys.exit(1)

    # df = pd.read_json(input_file, lines=True, nrows=1_000)
    df = pd.read_json(open(args.input_file), lines=True)

    is_silence = df.source == "NRK TV SILENCE"
    is_translate = df.source == "NRK TV TRANSLATE"
    is_transcribe = df.source == "NRK TV"
    silence_df = df[is_silence]
    translate_df = df[is_translate]
    df = df[is_transcribe]

    assert (is_silence | is_translate | is_transcribe).all()

    if args.audio_folder is not None and len(silence_df) > 0:
        # silent_indices = run_vad(df, args.audio_folder)
        # print(f"Silent samples in transcribe: {len(silent_indices)}/{len(df)}")
        # silent_indices = run_vad(translate_df, args.audio_folder)
        # print(f"Silent samples in translate: {len(silent_indices)}/{len(translate_df)}")
        voice_detector = VoiceDetector(method="silero")
        is_silent = silence_df.apply(functools.partial(run_vad,
                                                       voice_detector=voice_detector,
                                                       audio_folder=args.audio_folder),
                                     axis=1)
        print(f"Silent samples in silence: {is_silent.sum()}/{len(silence_df)}")
        is_silent.to_frame().to_csv("is_silent.csv", header=False)
        silence_df = silence_df[is_silent]

    config_file_path = os.path.join(
        args.output_folder.replace("/train", "/").replace("/test", "/").replace("/validation", "/"), "config.json")

    config = read_config(config_file_path)

    # transcriptions = pd.read_json(transcript_file, lines=True, nrows=100_000)
    transcriptions = pd.read_json(args.transcript_file, lines=True)

    print(
        f"Starting to merge. Length={len(df)}. Length of transcripts={len(transcriptions)}")
    df = df.merge(transcriptions, left_on="id", right_on="file",
                  how="left", suffixes=("", "_transcription"))
    df = df.fillna("")
    print(f"Finished merging. Length={len(df)}")

    print("Starting mat_per")
    df['w2v_mat_per'] = df.apply(lambda row: match_percentage(
        row["text"], row["text_transcription"]) if row["text_transcription"] != "" else None, axis=1)

    print("Starting ber_sim")
    df['w2v_ber_sim'] = df.apply(lambda row: bert_similarity(
        row["text"], row["text_transcription"]) if row["text_transcription"] != "" else None, axis=1)

    # Delete the entire program if the average mat_per is below 0.5
    grouped = df.groupby("program_id")
    cond = grouped["w2v_mat_per"].mean()[grouped["w2v_mat_per"].mean() < 0.5].index
    cond = df["id"].isin(cond)
    logger.debug(
        f'\n\n*** The following text was deleted because the mat_per average for the group was not high enough:' f'\n {df[cond][["text", "text_transcription", "w2v_mat_per", "w2v_ber_sim"]]}')
    df = df[~cond].reset_index(drop=True)

    # Delete stuff if the distance between text and transcripts is too large
    print("Evaluating mat_per and bert_sim")
    cond = df.apply(lambda row: is_valid_mat_per_ber_sim(
        row, config['min_mat_per'], config['min_ber_sim']), axis=1)

    logger.debug(
        f'\n\n*** The following text was deleted because the ber_sim and the mat_per was not high enough:' f'\n {df[~cond][["text", "text_transcription", "w2v_mat_per", "w2v_ber_sim"]]}')
    df = df[cond]

    # Calculate the number of words
    df["word_count_subtitles"] = df["text"].str.split().apply(len)
    df["word_count_transcription"] = df["text_transcription"].str.split().apply(len)
    df["verbosity_score"] = np.where(
        df["word_count_transcription"] != 0, df["word_count_subtitles"] / df["word_count_transcription"], 0)

    # Calculate Verbosity
    df["verbosity"] = np.choose(np.searchsorted(np.array(
        [0.50, 0.60, 0.70, 0.80, 0.90]), df["verbosity_score"]), [1, 2, 3, 4, 5, 6])

    # These are not used at the moment but keeping them in the code
    # print("Starting last_isin")
    # df['w2v_last_isin'] = df.apply(lambda row: last_isin(row["text"], row["text_transcription"]), axis=1)
    # print("Starting last_islast")
    # df['w2v_last_islast'] = df.apply(lambda row: last_islast(row["text"], row["text_transcription"]), axis=1)
    # print("Starting cos_sim")
    # df['w2v_cos_sim'] = df.apply(lambda row: cosine_similarity(row["text"], row["text_transcription"]), axis=1)
    # print("Starting jar_win")
    # df['w2v_jar_win'] = df.apply(lambda row: 1 - jaro_winkler_distance(row["text"], row["text_transcription"]), axis=1)
    # print("Starting war_sco")
    # df['w2v_wer_sco'] = df.apply(lambda row: wer(row["text"], row["text_transcription"]), axis=1)

    # Remove the chunk times to save space
    df = df.drop(
        columns=['chunks', 'file', 'model', 'text_transcription', 'w2v_mat_per', 'w2v_ber_sim', 'word_count_subtitles',
                 'word_count_transcription'])

    df = pd.concat([df, translate_df, silence_df])

    output_file = os.path.join(
        args.output_folder, os.path.basename(args.input_file))

    with open(output_file, 'w', encoding='utf-8') as file:
        df.to_json(file, orient='records', lines=True, force_ascii=False)

    print(f"Finished writing {len(df)} lines to {output_file}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True,
                        help='Path to input file.')
    parser.add_argument('--transcript_file', required=True,
                        help='Path to transcript file.')
    parser.add_argument('--audio_folder',
                        help="Path to audio folder (for VAD).")
    parser.add_argument('--output_folder', required=True,
                        help='Path to output folder.')
    parser.add_argument('--log_level', required=False, default="INFO",
                        help='Set logging level to DEBUG to get a report on all decisions')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

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
