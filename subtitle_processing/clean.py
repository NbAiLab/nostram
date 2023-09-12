######################################################################
# Cleans up subtitle jsonl files
#####################################################################
import functools
import os
import sys
import json

import librosa
import numpy as np
import pandas as pd
import re
from slugify import slugify
import hashlib
from datetime import datetime
import logging
import time
import argparse
import ftfy
# from pydub import AudioSegment
from pandarallel import pandarallel
from util import detect_lang
from utils.clean_text import clean_text

REMOVE_COL = "**REMOVE**"
pandarallel.initialize(use_memory_fs=False)
start_time = time.time()


def exec_time():
    end_time = time.time()
    out = str(round(end_time - start_time, 1)) + " seconds"
    return out


# compile regexes
username_regex = re.compile(r'(^|[^@\w])@(\w{1,15})\b')
url_regex = re.compile(r'((www\.[^\s]+)|(https?://[^\s]+)|(http?://[^\s]+))')
email_regex = re.compile(r'[\w\.-]+@[\w\.-]+')
control_char_regex = re.compile(r'[\r\n\t]+')


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


def count_alphawords(text):
    # Counts the number of pure alphawords (at least two characters long) in a text string
    # Adds spaces before some characters, if not . and , would lead to non-alpha words
    pat = re.compile(r"([.,;()!:/])")
    text = pat.sub(" \\1 ", text)
    num = sum((w.isalpha() and len(w) >= 2) for w in text.split())
    return num


def count_words(text):
    # Counts the number of words (at least two characters long) in a text string
    # Adds spaces before some characters, if not . and , would lead to non-alpha words
    pat = re.compile(r"([.,;()!:/])")
    text = pat.sub(" \\1 ", text)
    num = len([word for word in text.split() if len(word) >= 2])
    return num


def max_word_length(text):
    # Return the max word length in a text snippet
    wordlist = text.split()
    # Remove all links from the wordlist
    wordlist = [
        x for x in wordlist if 'http' not in x and '/' not in x and '-' not in x]

    if len(wordlist):
        max_length = len(max(wordlist, key=len))
    else:
        max_length = 0

    return max_length


def get_hash(text):
    slug = slugify(text)
    result = hashlib.md5(slug.encode())
    return result.hexdigest()


def normalise_unicode(text):
    input_text = text
    text = text.strip()
    text = " ".join(text.split())
    text = ftfy.fix_text(text)

    if input_text != text:
        logger.debug(f'Changed "{input_text}" -> "{text}"')

    return text


def add_task_field(data: pd.DataFrame):
    """
    Typically TTV is a superset of NOR, so anything in NOR is a spoken foreign language translated to Norwegian text,
     while TTV is Norwegian OR foreign language to Norwegian text.
    This method subtracts any duplicates from TTV, leaving only the Norwegian->Norwegian parts,
     and returns two separate dataframes.
    Note that NOR will sometimes include Norwegian->Sami as well
    """
    # Some programs have overlapping subtitles in different languages, remove those
    dup_sub_start_end = data.duplicated(
        subset=["program_id", "vtt_folder", "start_time", "end_time"], keep=False)
    data.drop(data[dup_sub_start_end].index, inplace=True)

    # Mark duplicates by program+time+text
    dup_start = data.duplicated(
        subset=["program_id", "start_time", "text"], keep=False)
    dup_end = data.duplicated(
        subset=["program_id", "end_time", "text"], keep=False)

    # Find duplicates only by program+text (both transcribe and transcribe_translate)
    dup_text_type = data.duplicated(
        subset=["program_id", "vtt_folder", "text"], keep=False)
    dup_text = data.duplicated(subset=["program_id", "text"], keep=False)

    # Duplicates by time, duplicates by text only (unless vtt_folder is also duplicate)
    mask = ~(dup_start | dup_end) & ~(dup_text & ~dup_text_type)

    data.loc[mask & (data.vtt_folder == "vtt_transcribe_translate"), "task"] = "transcribe"
    data.loc[~mask & (data.vtt_folder ==
                      "vtt_transcribe_translate"), "task"] = "UNK"
    data.loc[data.vtt_folder == "vtt_translate", "task"] = "translate"


def remove_inaudible(data: pd.DataFrame):
    """
    Some special cases of patterns in subtitles that are not spoken out loud.
    """
    data.text = data.text.str.replace(
        r"(Nord|Sør)-Sápmi:"
        r"|(Norske? )?tekst(er|ar|ing|):.*"
        r"|(English )?subtitles:.*"
        r"|(Ådåsdåjmadiddje|Ođasredaktevra|Ođashoavda)/nyhetsredaktør:.*"
        # Remove parentheses without age/political party
        r"|\((?!(\d|\d\d|1\d\d|Ap|H|Sp|Frp|SV|R|V|MdG|KrF)\))[^)]*\)"
        r"|Opptak av simultanteksting",
        # r"|I forrige episode:",
        "", regex=True, flags=re.IGNORECASE)
    new_text = data.text.str.replace(
        r"^\W+$", "", regex=True)  # Only non-word characters
    modified = data.text[new_text != data.text]
    data.text = new_text
    removed = data[data.text.str.len() == 0]
    data.drop(removed.index, inplace=True)

    return modified


def is_invalid_duration(data: pd.DataFrame):
    """
    Some durations will clearly not make sense, either by being too short (some are even negative) or being too long.
    This filters out some of the more extreme durations based on text length.
    """
    # "The most widely known rule on the speed of interlingual subtitles–“the six-seconds rule”–stipulates that a full
    # two-line subtitle should be displayed for six seconds in order for an average viewer to be able to read it.
    # The six-seconds rule is equivalent to approximately 140–150 wpm or 12 cps."
    durations = data.duration / 1000
    lengths = data.text.str.len()
    too_fast = durations <= lengths / 24
    too_slow = durations > lengths / 6 + 10
    cond = too_slow | too_fast

    return cond


def merge_subtitles(data: pd.DataFrame, drop_multiple_speakers=False, combine_continued_sentences=False):
    """
    This method goes through and concatenates any lines that belong together,
     e.g. sentences over two lines, sentences over several consecutive lines in different timestamps
     (continuation is denoted by ending a line and starting the next line with -).
    If two lines have dashes that denotes multiple speakers speaking simultaneously or in rapid succession,
     these can optionally be filtered out using `drop_multiple_speakers`
    (note: generation of dataframe replaces newlines with a pipe for parsing)
    """
    data = data.copy().sort_values(
        ["program_id", "start_time", "end_time"][int("program_id" not in data.columns):])
    # Inconsistent usage, just stick to the normal dash
    data.text = data.text.str.replace("—", "-")
    # old_text = data.text

    if combine_continued_sentences:
        # is_overlapping = data.text.str.fullmatch(r"-.+(<br>-.+)+")
        expects_continuation = data.text.str.strip().str.endswith("-")

        def cat(df):
            first = df.iloc[0].copy()
            last = df.iloc[-1]
            first.text = df.text.str.strip().str.cat(sep="<p>")
            first.end_time = last.end_time
            first.duration = first.end_time - first.start_time
            first.id = "_".join(first.id.split("_")[:-1] + [str(first.end_time)])
            first[REMOVE_COL] = df[REMOVE_COL].any()

            return first

        groups = (~expects_continuation.shift(1, fill_value=False)).cumsum()
        data = data.groupby(groups).apply(cat)

    if drop_multiple_speakers:
        # Find overlapping speakers, e.g. `-Vi svarte frimerker.<br>-Det var ikke alle som visste det.`
        is_overlapping = data.text.str.match(r".*(^|<p>)-[^<>]*(?<!-)<br>-", )
        # is_overlapping = data[is_overlapping].text
        # data.drop(is_overlapping.index, inplace=True)
        data.loc[is_overlapping, "**is_overlapping**"] = data[is_overlapping].text

    # To be dropped later
    has_triple_lines = data.text.str.contains("[^<>]+<br>[^<>]+<br>[^<>]+")
    data.loc[has_triple_lines, "**has_triple_lines**"] = data[has_triple_lines].text.copy()

    data["**before_merge**"] = data.text.copy()
    # Continued word, e.g. `Det er viktig å treffe folk som har mer for-<br>nuftige interesser enn de gamle kompisene.`
    data.text = data.text.str.replace(r"-<br>(?!-)", "", regex=True)

    # Continued sentence
    if combine_continued_sentences:
        data.text = data.text.str.replace(r"-<p>-?", " ", regex=True)
    else:
        data.text = data.text.str.replace("(^\s*-|-\s*$)", " ", regex=True)

    # Now remove any remainders
    data.text = data.text.str.replace("<(p|br)>-?|^-", " ", regex=True)

    data.text = data.text.str.strip().replace("  +", " ", regex=True)
    data.drop(data[data.text.isna() | (data.text.str.len() == 0)].index, inplace=True)

    # modified = pd.DataFrame({"old_text": old_text[data.index], "new_text": data.text})
    # deleted = old_text.drop(data.index)

    return data


def offset_timestamp_text(text, delta_seconds):
    prev_stamp = -float("inf")

    def transform(match):
        nonlocal prev_stamp
        stamp = float(match[1])
        stamp += delta_seconds
        stamp = 2 * round(stamp / 2, 2)
        if stamp < 0:
            # If we get negative values, it means the first subtitle will not show yet due to the previous subtitle
            # overlapping with the current one. It should still represent the start of the audio clip,
            # and new positive values should still be respected
            stamp = 0

        # Handle some potential rounding errors
        if -0.02 <= stamp < 0:
            stamp = 0.0
        elif 30 < stamp <= 30.02:
            stamp = 30.0

        # Debugging
        if stamp < prev_stamp:
            # Same argument as above (stamp < 0)
            stamp = prev_stamp
        elif not 0 <= stamp <= 30:
            ...
            # print(f"{stamp} outside 0-30 range")
        prev_stamp = stamp
        return f"<|{stamp:.2f}|>"

    return re.sub(r"<\|(\d+\.\d+)\|>", transform, text)


def make_timestamp_text(row):
    end_stamp = (row.end_time - row.start_time) / 1000
    # assert end_stamp > 0
    end_stamp = 2 * round(end_stamp / 2, 2)
    if "Oldsmobilen" in row.text:
        ...
    return f"<|0.00|> {row.text.strip()}<|{end_stamp:.2f}|>"


def adjust_overlapping_subtitles(data):
    data = data.sort_values(["start_time", "end_time"])
    data.iloc[1:, data.columns.get_loc("start_time")] = np.maximum(data["start_time"].values[1:],
                                                                   data["end_time"].values[:-1])
    data["duration"] = data["end_time"] - data["start_time"]
    data = data[data["duration"] > 0]
    data["id"] = data.apply(lambda r: f"{r.program_id}_{r.start_time}_{r.end_time}", axis=1)
    return data


def combine_to_size(data, target_duration_seconds=26, max_separation_seconds=5):
    groups = []
    group = [data.iloc[0]]
    for i in range(1, len(data) + 1):
        # Need an extra iteration to include the last group
        row = data.iloc[i] if i < len(data) else None
        if (row is not None
                and 0 < row.end_time - group[0].start_time <= target_duration_seconds * 1000
                and row.start_time - group[-1].end_time <= max_separation_seconds * 1000
                and not row[REMOVE_COL] and not group[-1][REMOVE_COL]
                and detect_lang(row.text) == detect_lang(group[-1].text)):
            group.append(row)
        else:
            first = group[0].copy()

            ts_text = "".join([offset_timestamp_text(r.timestamp_text, (r.start_time - first.start_time) / 1000)
                               for r in group])
            ts_text = offset_timestamp_text(ts_text, 0)  # Clean up any decreasing timestamps
            text = " ".join([r.text for r in group])
            first.timestamp_text = re.sub(r"\s+", " ", ts_text)
            first.text = re.sub(r"\s+", " ", text)

            if "Oldsmobilen" in ts_text:
                ...

            first.end_time = group[-1].end_time
            if row is not None and row.start_time - group[-1].end_time > max_separation_seconds * 1000:
                first.end_time += 1000
            first.duration = first.end_time - first.start_time
            first.id = f"{first.program_id}_{first.start_time}_{first.end_time}"
            first[REMOVE_COL] = any(r[REMOVE_COL] for r in group)
            groups.append(first)
            group = [row]
    assert group[0] is None
    new_data = pd.concat(groups, axis=1, ignore_index=True).T
    new_data[REMOVE_COL] = new_data[REMOVE_COL].astype(bool)
    return new_data


def pad_with_silence(data: pd.DataFrame, max_length_seconds: float):
    max_length_ms = round(max_length_seconds * 1000)
    out_data = []
    for i in range(len(data)):
        row = data.iloc[i].copy()
        current_length = row.end_time - row.start_time
        leftover_space = max_length_ms - current_length
        if row[REMOVE_COL]:
            out_data.append(row)
            continue
        assert leftover_space >= 0

        # Find midpoint between this subtitle and the previous/next so there's no overlap
        # NRK shows subtitle until it ends, so we trust leftover_before even if negative, but not leftover_after
        if i == 0:
            leftover_before = row.start_time
        else:
            leftover_before = (row.start_time - data.iloc[i - 1].end_time) // 2
            # leftover_before = max(leftover_before, 0)
        if i == len(data) - 1:
            leftover_after = 0  # Unless we can find time until program end?
        else:
            leftover_after = (data.iloc[i + 1].start_time - row.end_time) // 2
            leftover_after = max(leftover_after, 0)

        # Try to place subtitle in the middle
        pad_before = min(leftover_before, leftover_space // 2)
        pad_after = min(leftover_after, leftover_space // 2)

        # If there's less space on either of the sides we can give more space to the opposite side
        if pad_before < leftover_space // 2:
            pad_after = min(leftover_after, leftover_space - pad_before)
        if pad_after < leftover_space // 2:
            pad_before = min(leftover_before, leftover_space - pad_after)

        valid_length = 0 < current_length + pad_before + pad_after <= max_length_ms

        row.timestamp_text = offset_timestamp_text(row.timestamp_text, pad_before / 1000)
        row.start_time -= pad_before
        row.end_time += pad_after
        row.duration = row.end_time - row.start_time
        row.id = f"{row.program_id}_{row.start_time}_{row.end_time}"
        row[REMOVE_COL] |= row.end_time <= row.start_time | ~valid_length
        out_data.append(row)
    out_data = pd.concat(out_data, axis=1, ignore_index=True).T
    out_data[REMOVE_COL] = out_data[REMOVE_COL].astype(bool)
    return out_data


def add_empty_captions(data: pd.DataFrame, max_duration=30, min_duration=1, min_padding=2):
    max_duration = round(1000 * max_duration)
    min_duration = round(1000 * min_duration)
    min_padding = round(1000 * min_padding)
    data = data.copy().sort_values(["start_time", "end_time"])
    new_data = []
    for i in range(len(data) - 1):
        prev_ts = data.iloc[i].end_time
        next_ts = data.iloc[i + 1].start_time

        prev_ts += min_padding
        next_ts -= min_padding

        while next_ts - prev_ts > min_duration:
            row = data.iloc[i].copy()

            start_time = prev_ts
            if next_ts - prev_ts > max_duration:
                end_time = prev_ts + min(max_duration, (next_ts - prev_ts) // 2)
            else:
                end_time = next_ts
            assert min_duration <= end_time - start_time <= max_duration

            row.id = f"{row.program_id}_{start_time}_{end_time}"
            row.text = "<|nocaptions|>"
            row.timestamp_text = "<|nocaptions|>"
            row.start_time = start_time
            row.end_time = end_time
            row.duration = end_time - start_time

            new_data.append(row)
            prev_ts = end_time
    if len(new_data) > 0:
        new_data = pd.concat(new_data, axis=1, ignore_index=True).T
        new_data[REMOVE_COL] = new_data[REMOVE_COL].astype(bool)
    else:
        new_data = data.iloc[:0].copy()
    return new_data


def remove_italics(data: pd.DataFrame):
    """
    Italics are used liberally to denote things like emphasis, narrators, voices from phones etc.
    Generally they are spoken, and should thus be included.
    One special case is parallel translations from tertiary language to Norwegian and Sami,
     with one language italicized and the other not, on separate lines.
    """
    # cond = data.text.str.fullmatch(r"((^|<br>)[^<>]+)+(<br><i>[^<>]+<\/i>)+")
    # dropped = data.text[cond]
    # data.drop(data[cond].index, inplace=True)
    old_text = data.text
    data.text = data.text.str.replace("</?i>", "", regex=True).str.strip()
    cond = old_text != data.text
    modified = pd.DataFrame(
        {"old_text": old_text[cond], "new_text": data.text[cond]})
    return modified


def find_simultaneous(data: pd.DataFrame):
    simultaneous = data[data.text.str.lower().str.contains("opptak av simultanteksting")]
    program_ids = simultaneous.program_id.unique()

    return program_ids


def create_histogram(data: pd.DataFrame):
    hist_string = ""
    histogram = np.histogram(data["duration"], bins=30, range=(0, 30000))[0]
    for i in histogram:
        hist_string += str(i) + ", "

    hist_string = "[" + hist_string[:-2].strip() + "]"

    return hist_string


def create_audio_segments_command(id, audio, start_time, duration, right_padding=0):
    if (start_time >= 0) and duration:
        corename = audio.split(".")[0]
        subfolder = corename.split("_")[0][-2:] + "/"

        # Create this directory if it does not exists (ffmpeg can not do this)
        if not os.path.exists(os.path.join(args.audio_output_folder, subfolder)):
            os.makedirs(os.path.join(args.audio_output_folder, subfolder))

        command = f"ffmpeg -ac 1 -n -ss {start_time / 1000} -t {(duration / 1000) + int(right_padding)} -i {os.path.join(args.audio_input_folder, audio)} -acodec libmp3lame -ar 16000 {os.path.join(args.audio_output_folder, subfolder + id + '.mp3')}"
    else:
        command = f"cp {os.path.join(args.audio_input_folder, audio)} {args.audio_output_folder}"
        print("This should not happen! Please debug this. Most likely reason is that we are running this on old files.")
        breakpoint()
    return command


def left_align(text):
    if len(text) == 0:
        return text
    elif isinstance(text, pd.DataFrame):
        text = text.copy()
        for col in text.columns:
            if text[col].dtype == object:  # Assume string
                text[col] = left_align(text[col])
        return text
    else:
        # Utility for debug output
        return text.str.ljust(int(text.str.len().max()))


def update_mp3_name(id, audio):
    corename = audio.split(".")[0]
    subfolder = corename.split("_")[0][-2:] + "/"
    final_name = subfolder + id + '.mp3'
    return final_name


def show_length(data):
    valid_data = data[~data[REMOVE_COL]] if REMOVE_COL in data.columns else data
    if "duration" in data.columns:
        duration_seconds = valid_data["duration"].sum() // 1000
    else:
        duration_seconds = valid_data["audio_duration"].sum() // 1000
    hours, seconds = divmod(duration_seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f"\n\tSamples: {len(valid_data)}" \
           f"\n\tDuration: {hours:d}h {minutes:02d}m {seconds:02d}s"


def select_random_samples(sub_df, n):
    if n == "log":
        n = max(1, round(np.log2(len(sub_df))))
    return sub_df.sample(n=min(n, len(sub_df)), replace=False)


def remove_text_program_duplicates(data: pd.DataFrame, max_duplicates):
    data['program_id_4chars'] = data['program_id'].str[:4]

    random_samples = (data.groupby(['program_id_4chars', 'vtt_folder', 'text'])
                      .parallel_apply(select_random_samples, n=max_duplicates))
    indices = random_samples.index.get_level_values(None)
    data = data.drop("program_id_4chars", axis=1)
    data.loc[~data.index.isin(indices), REMOVE_COL] = True
    return data


def sanitize(row):
    new_text, stats = clean_text(row.text)
    stats["text"] = new_text
    return pd.Series(stats)


def main(args):
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_colwidth", None)
    ocr_doc = 1

    # Invoke logging
    log_name = os.path.basename(args.input_file).replace(".json", "")
    log_name = log_name + ".log"

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

    print(f'*** Starting to process: {args.input_file}')
    data: pd.DataFrame = load_json(args.input_file)
    data[REMOVE_COL] = False
    data[REMOVE_COL] = data[REMOVE_COL].astype(bool)

    logger.info(f'***  Data loaded. {len(data)} subtitles. ({exec_time()})')
    print(
        f'Log written to {os.path.join(args.output_folder, "log/", log_name)}. ({exec_time()})')

    # Fix unicode
    if config['normalise_unicode']:
        data['text'] = data['text'].parallel_apply(normalise_unicode)
        logger.info(
            f'***  Normalised unicode. Removed double spaces. Trimmed string. ({exec_time()})')

    # Minimum length of subtitle
    if config['min_length_subtitle']:
        cond = data.text.str.len() >= config['min_length_subtitle']
        logger.debug(f'\n\n*** The following text was deleted because the article minimum length was too small:'
                     f'\n {data[~cond]["text"]}')
        # data = data[cond]
        data.loc[~cond, REMOVE_COL] = True
        logger.info(
            f'***  Completed filtering min length article. ({exec_time()}) {show_length(data)}')

    # Remove paragraphs with curly brackets
    if config['drop_subtitles_with_curly_brackets']:
        cond = data['text'].str.contains('\\{')
        logger.debug(f'\n\n*** The following text was deleted because it contained left curly brackets:'
                     f'\n {data[cond]["text"]}')
        data = data[~cond]
        cond = data['text'].str.contains('\\}')
        logger.debug(f'\n\n*** The following text was deleted because it contained right curly brackets:'
                     f'\n {data[cond]["text"]}')
        # data = data[~cond]
        data.loc[cond, REMOVE_COL] = True
        logger.info(f'***  Completed filtering out subtitles with curly brackets. ({exec_time()}) {show_length(data)}')

    # Filter out paragraphs with encoding errors
    if config['drop_subtitles_with_encoding_errors']:
        cond = data['text'].str.contains('�')
        # data = data[~cond]
        data.loc[cond, REMOVE_COL] = True
        logger.info(
            f'***  Filtered out encoding errors. ({exec_time()}) {show_length(data)}')

    simultaneous = find_simultaneous(data)
    if config["simultaneous_subtitles"] == "detect":
        data["is_simultaneous"] = data.program_id.isin(simultaneous)
        logger.info(
            f'***  {data.is_simultaneous.sum()} rows were marked as simultaneous texting. ({exec_time()})')
    elif config["simultaneous_subtitles"] == "delete":
        cond = data.program_id.isin(simultaneous)
        logger.debug(f'\n\n*** The following program ids were deleted because they had simultaneous texting:'
                     f'\n {simultaneous}')
        data = data[~cond]
        logger.info(
            f'***  Filtered out simultaneous texting. ({exec_time()}) {show_length(data)}')

    if 'vtt_folder' in data.columns:
        add_task_field(data)
    else:
        data = data.assign(task='transcribe')

    if 'start_time' not in data.columns:
        data = data.assign(start_time='')
    if 'end_time' not in data.columns:
        data = data.assign(end_time='')

    logger.info(
        f"***  Added task field, counts: {data.task.value_counts().to_dict()}")
    if config['task']:
        tasks = config["task"]
        if isinstance(tasks, str):
            tasks = [tasks]
        cond = data['task'].isin(tasks)
        logger.debug(f'\n\n*** The following text was deleted because it was not the correct task:'
                     f'\n {data[~cond][["text", "task"]]}')
        data = data[cond]
        logger.info(
            f'***  Filtered out tasks. ({exec_time()}) {show_length(data)}')

    if config['drop_italics']:
        modified = remove_italics(data)
        # logger.debug(f'\n\n*** The following text was deleted because it is suspected to be bilingual:'
        #              f'\n {dropped}')
        logger.debug(f'\n\n*** The following text was modified because it contained italics text:'
                     f'\n {modified}')
        logger.info(
            f'***  Filtered out italics. ({exec_time()}) {show_length(data)}')

    if config['drop_inaudible']:
        modified = remove_inaudible(data)
        logger.debug(f'\n\n*** The following text was modified because it contained inaudible text:'
                     f'\n {modified}')
        logger.info(
            f'***  Filtered out inaudible. ({exec_time()}) {show_length(data)}')

    if config['drop_invalid_durations']:
        cond = is_invalid_duration(data)
        logger.debug(f'\n\n*** The following text was deleted because the speaking rate is too fast or too slow:'
                     f'\n {data[cond][["text", "duration"]]}')
        # data = data[~cond]
        data.loc[cond, REMOVE_COL] = True
        logger.info(f'***  Filtered out too fast and too slow speaking rates. '
                    f'({exec_time()}) {show_length(data)}')

    # Merge subtitles only if the origin is from subtitles
    if config['merge_subtitles'] and 'vtt_folder' in data.columns:
        logger.info(
            f'***  Histogram before merging subtitles: {create_histogram(data)}. '
            f"{show_length(data)}")
        data = data.groupby(["program_id", "vtt_folder"]).parallel_apply(
            functools.partial(merge_subtitles,
                              drop_multiple_speakers=config['drop_multiple_speakers'],
                              combine_continued_sentences=config['combine_continued_sentences']))

        data = data.reset_index(drop=True)

        if "**is_overlapping**" in data.columns:
            cond = data["**is_overlapping**"]
            logger.debug(f"\n\n*** The following lines were removed for containing overlapping speakers:"
                         f"\n{left_align(cond[cond.notna()])}")
            # data = data[cond.isna()]
            data.loc[cond.notna(), REMOVE_COL] = True
            data = data.drop("**is_overlapping**", axis=1)

        cond = data["**has_triple_lines**"]
        logger.debug(f"\n\n*** The following text was removed because it contained more than two lines in one subtitle:"
                     f"\n{left_align(cond[cond.notna()])}")
        # data = data[cond.isna()]
        data.loc[cond.notna(), REMOVE_COL] = True
        data = data.drop("**has_triple_lines**", axis=1)

        before_merge = data["**before_merge**"]
        cond = before_merge.str.replace(r"\s*<br>\s*", " ", regex=True) != data.text
        modified = pd.DataFrame({"before": before_merge[cond], "after": data.text[cond]})
        logger.debug(f"\n\n*** The following text was modified during merging of subtitles:"
                     f"\n{left_align(modified)}")
        data = data.drop("**before_merge**", axis=1)

        # data = data.reset_index().drop("level_1", axis=1)
        # modified, deleted = rmerge_subtitles(data, drop_multiple_speakers=config['drop_multiple_speakers'])
        # logger.debug(f'\n\n*** The following text was modified because of text continuation or speaker overlap:'
        #              f'\n {modified}')
        # logger.debug(f'\n\n*** The following text was deleted because of text continuation or speaker overlap:'
        #              f'\n {deleted}')
        logger.info(f'***  Filtered out text continuation and/or speaker overlap. '
                    f'({exec_time()}) {show_length(data)}')
        logger.info(f'***  Histogram after merging subtitles: {create_histogram(data)}')

    if config['remove_cpossible']:
        cond = data.text.str.contains("CPossible")
        logger.debug(f'\n\n*** The following text was deleted because it contained "CPossible":'
                     f'\n {data[cond]["text"]}')
        # data = data[~cond]
        data.loc[cond, REMOVE_COL] = True
        logger.info(f'***  Filtered out "CPossible". ({exec_time()}) {show_length(data)}')

    print("test")
    stats = data.parallel_apply(sanitize, axis=1)
    data[REMOVE_COL] = stats["delete_line"]
    #logger.debug(f"\n\n*** Full stats for sanitizing: \n{left_align(stats)}")
    logger.info(f"*** Sanitized text ({exec_time()}) {show_length(data)}")
    logger.info(f"*** Total stats: {stats.drop('text', axis=1).sum().to_dict()}")
    print(test2)
    
    data = data.groupby(["program_id", "vtt_folder"]).parallel_apply(adjust_overlapping_subtitles)
    data = data.reset_index(drop=True)
    data["timestamp_text"] = data.parallel_apply(make_timestamp_text, axis=1)

    max_duplicates = config.get("max_duplicates_text_program", None)
    if max_duplicates is not None:
        data = remove_text_program_duplicates(data, max_duplicates)

        logger.info(f'***  Removed duplicate text/program combinations. '
                    f'({exec_time()}) {show_length(data)}')
        logger.info(f'***  Histogram after removing duplicates: {create_histogram(data)}')

    # Make bigger segments only if the origin is from subtitles
    if config['make_bigger_segments'] and 'vtt_folder' in data.columns:
        data = data.groupby(["program_id", "vtt_folder"]).parallel_apply(
            functools.partial(combine_to_size,
                              target_duration_seconds=config["target_duration_seconds"],
                              max_separation_seconds=config["max_separation_seconds"])
        )
        data = data.reset_index(drop=True)

        logger.info(f'***  Combined texts to fill out context length. '
                    f'({exec_time()}) {show_length(data)}')
        logger.info(f'***  Histogram after merging subtitles: {create_histogram(data)}')

    # Filter out too long posts
    if config['max_duration_seconds']:
        cond = data['duration'].between(0, config['max_duration_seconds'] * 1000, inclusive="right")
        # data = data[cond]
        data.loc[~cond, REMOVE_COL] = True
        logger.info(f'***  Filtered out too long segments. '
                    f'({exec_time()}) {show_length(data)}')

    if config["pad_with_silence"]:
        data = data.groupby(["program_id", "vtt_folder"]).parallel_apply(
            functools.partial(pad_with_silence,
                              max_length_seconds=config["max_duration_seconds"])
        )
        data = data.reset_index(drop=True)

        logger.info(f'***  Histogram after padding with silence: {create_histogram(data)} '
                    f"{show_length(data)}")

    if config['detect_lang_text']:
        do_lang_detect = True
        if "lang_text" in data.columns:
            mask = data["lang_text"].isna()
            data.loc[~mask, "lang_text_confidence"] = 1.
            if mask.sum() == 0:
                do_lang_detect = False
        else:
            mask = slice(None)

        if do_lang_detect:
            program_language = data[mask].groupby(["program_id", "vtt_folder"]) \
                .parallel_apply(lambda x: detect_lang(x.text.str.cat(sep=" "), return_proba=True))
            program_language = program_language.to_dict()
            program_language = data[mask].parallel_apply(lambda x:
                                                         pd.Series(program_language[(x.program_id, x.vtt_folder)],
                                                                   index=["language", "confidence"]),
                                                         axis=1)

            languages = data[mask].text.parallel_apply(lambda x: pd.Series(detect_lang(x, return_proba=True),
                                                                           index=["language", "confidence"]))

            data.loc[mask, "lang_text"] = languages["language"]
            data.loc[mask, "lang_text_confidence"] = languages["confidence"]

            allowed_langs = config.get("allow_lang_text", None)
            if allowed_langs:
                if isinstance(allowed_langs, str):
                    allowed_langs = [allowed_langs]

                potential_norwegian = ~data[mask]["lang_text"].isin(["eng", "sme", "nob", "nno"])
                is_short = data[mask].text.str.split().str.len() <= 5
                is_prog_norwegian = program_language["language"].isin(["nob", "nno"])
                inherit_prog_lang = (potential_norwegian | is_short) & is_prog_norwegian

                data.loc[inherit_prog_lang, "lang_text"] = program_language[inherit_prog_lang]["language"]
                data.loc[inherit_prog_lang, "lang_text_confidence"] = program_language[inherit_prog_lang]["confidence"]

                invalid_prog_lang = ~program_language["language"].isin(allowed_langs)
                data.loc[invalid_prog_lang, REMOVE_COL] = True

                invalid_lang = ~data["lang_text"].isin(allowed_langs)
                data.loc[invalid_lang, REMOVE_COL] = True

            three_to_two = {"nob": "no", "nno": "nn", "eng": "en", "sme": "se"}
            data["lang_text"] = data["lang_text"].apply(lambda x: three_to_two.get(x, x))

            logger.info(f'***  Removed invalid languages. '
                        f'({exec_time()}) {show_length(data)}')

    if config["add_empty_captions"]:
        new_data = data.groupby("program_id").parallel_apply(
            functools.partial(add_empty_captions,
                              max_duration=config["max_duration_seconds"])
        )
        new_data = new_data.reset_index(drop=True)

        # languages = data["lang_text"].value_counts()
        # new_data["lang_text"] = np.random.choice(languages.index.to_list(),
        #                                          size=len(new_data),
        #                                          p=(languages / languages.sum()).to_list())
        # new_data["lang_text_confidence"] = 0.
        new_data["task"] = "silence"

        data = pd.concat([data, new_data])

        logger.info(f'***  Added empty captions. '
                    f'({exec_time()}) {show_length(data)}')

    if args.audio_input_folder:
        def calculate_duration(fname):
            fpath = os.path.join(args.audio_input_folder, fname)
            duration = int(librosa.get_duration(filename=fpath) * 1000)
            return duration

        unique_files = data["audio"].drop_duplicates()
        durations = unique_files.parallel_apply(calculate_duration)
        file_duration = dict(zip(unique_files, durations))
        durations = data["audio"].map(file_duration)
        data = data[data["end_time"] <= durations]

        logger.info(f'***  Removed subtitles that end after the audio file.'
                    f'({exec_time()}) {show_length(data)}')

    if args.audio_output_folder:
        if not args.audio_input_folder:
            print("You also need to provide an input folder")
            os._exit(1)
        # data.groupby(["program_id"]).parallel_apply(create_audio_segments)
        if args.right_padding:
            right_padding = args.right_padding
        else:
            right_padding = 0

        data['command'] = data.apply(lambda row: create_audio_segments_command(row['id'], row['audio'],
                                                                               row['start_time'],
                                                                               row['duration'], right_padding), axis=1)

        filename = os.path.basename(args.input_file).replace(".json", "")

        with open(os.path.join(args.audio_output_folder, filename + '_process_list.sh'), 'w') as f:
            for text in data['command'].tolist():
                f.write(text + '\n')

        logger.info(
            f'Audio processing list written to {os.path.join(args.audio_output_folder, filename + "_process_list.sh")}\n '
            f'({exec_time()}) {show_length(data)}')
        logger.info(f'***  Histogram after writing audio files: {create_histogram(data)} '
                    f"{show_length(data)}")

    # Update the audio file path even if the audio is not generated

    data['audio'] = data.apply(lambda row: update_mp3_name(row['id'], row['audio']), axis=1)

    data = data[~data[REMOVE_COL]]
    data = data.drop(REMOVE_COL, axis=1)

    # Do some general cleaning

    # Leave just a few columns for the online dataset
    # final_table_columns = ["id", "text", "timestamp_text", "start_time", "end_time", "duration", "program_id", "medium",
    #                        "source",
    #                        "category", "title", "subtitle", "audio", "lang_text", "lang_text_confidence", "lang_voice",
    #                        "lang_voice_confidence", "task"]
    is_nrk = "program_id" in data.columns
    data = data.rename({"program_id": "group_id", "duration": "audio_duration", "lang_text": "text_language",
                        "timestamp_text": "timestamped_text"}, axis=1)

    data["previous_text"] = np.nan
    data["previous_text"] = data.text.shift()
    new_group = data["group_id"].ne(data['group_id'].shift())
    data.loc[new_group, "previous_text"] = np.nan
    if is_nrk:
        data["source"] = data.task.map({"translate": "nrk_tv_translate",
                                        "transcribe": "nrk_tv",
                                        "silence": "nrk_tv_silence"})
        new_source = data["source"].ne(data['source'].shift())
        data.loc[new_source, "previous_text"] = np.nan
        data["previous_text"] = data["previous_text"].replace("<|nocaptions|>", np.nan)
    else:
        raise NotImplementedError("Source unknown")

    final_table_columns = ["id", "group_id", "source", "audio_language", "audio_duration", "previous_text",
                           "text_language", "text", "translated_text_no", "translated_text_nn", "translated_text_en",
                           "translated_text_es", "timestamped_text", "wav2vec_wer", "whisper_wer", "verbosity_level"]
    data = data[data.columns.intersection(final_table_columns)]

    # Add final table columns if they do not exist
    for col in final_table_columns:
        if col not in data.columns:
            print(f"{col} did not exist, adding it...")
            data[col] = np.nan

    # Change the orders of the columns so that the json looks nicer
    data = data[final_table_columns]

    # Replace NaNs with empty strings
    # data = data.replace(np.nan, '', regex=True)

    data = data.drop_duplicates(["id", "text", "group_id", "source"])
    logger.info(f'***  Removed duplicate IDs. ({exec_time()}) {show_length(data)}')

    # Save it as jsonl
    output_filename = os.path.join(
        args.output_folder, os.path.basename(args.input_file))
    save_json(data, output_filename)
    logger.info(
        f'*** Finished processing file. {show_length(data)}.'
        f'\nResult is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}. ({exec_time()})')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True,
                        help='Path to input file.')
    parser.add_argument('--output_folder', required=True,
                        help='Path to output folder.')
    parser.add_argument('--audio_output_folder', required=False,
                        help='Path where audio segments should be placed. If not specified, audio segments are not renerated.')
    parser.add_argument('--right_padding', required=False,
                        help='This adds n seconds padding to the sound file. The name of the file is not changing.')
    parser.add_argument('--audio_input_folder', required=False,
                        help='Path where audio segments should be read. If not specified, audio segments are not renerated.')
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
