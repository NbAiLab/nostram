######################################################################
### Cleans up subtitle jsonl files
#####################################################################

import os, sys
import json

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
from pandarallel import pandarallel

pandarallel.initialize(use_memory_fs=True)
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
        logger.info("Error. There has to be a valid config-file in the output directory")
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
    wordlist = [x for x in wordlist if 'http' not in x and '/' not in x and '-' not in x]

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
    dup_sub_start_end = data.duplicated(subset=["program_id", "vtt_folder", "start_time", "end_time"], keep=False)
    data.drop(data[dup_sub_start_end].index, inplace=True)

    # Mark duplicates by program+time+text, mark anything except first (NOR since alphabetical) as duplicate
    dup_start = data.duplicated(subset=["program_id", "start_time", "text"], keep="first")
    dup_end = data.duplicated(subset=["program_id", "end_time", "text"], keep="first")

    # Find duplicates only by program+text (both NOR and TTV)
    dup_text_type = data.duplicated(subset=["program_id", "vtt_folder", "text"], keep=False)
    dup_text = data.duplicated(subset=["program_id", "text"], keep=False)

    # Duplicates by time, duplicates by text only (unless vtt_folder is also duplicate)
    mask = ~(dup_start | dup_end) & ~(dup_text & ~dup_text_type)

    data.loc[mask & (data.vtt_folder == "vtt_transcribe_translate"), "task"] = "transcribe"
    # data.loc[~mask & (data.vtt_folder == "vtt_transcribe_translate"), "task"] = "UNK"
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
        r"|\((?!(\d|\d\d|1\d\d|Ap|H|Sp|Frp|SV|R|V|MdG|KrF)\))[^)]*\)"  # Remove parentheses without age/political party
        r"|Opptak av simultanteksting",
        # r"|I forrige episode:",
        "", regex=True, flags=re.IGNORECASE)
    new_text = data.text.str.replace(r"^\W+$", "", regex=True)  # Only non-word characters
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


def remove_splits(data: pd.DataFrame, drop_overlapping=False):
    """
    This method goes through and concatenates any lines that belong together,
     e.g. sentences over two lines, sentences over several consecutive lines in different timestamps
     (continuation is denoted by ending a line and starting the next line with -).
    If two lines have dashes that denotes multiple speakers speaking simultaneously or in rapid succession,
     these can optionally be filtered out using `drop_overlapping`
    (note: generation of dataframe replaces newlines with a pipe for parsing)
    """
    data = data.sort_values(["program_id", "start_time", "end_time"])
    data.text = data.text.str.replace("—", "-")  # Inconsistent usage, just stick to the normal dash
    old_text = data.text

    expects_continuation = False
    overlapping = False
    start_index = None
    string = ""
    delete_mask = np.zeros(len(data), dtype=bool)

    for i, text in enumerate(data["text"]):
        text = text.strip()
        is_continuation = text.startswith("-")
        if is_continuation:
            is_continuation = expects_continuation
            overlapping = overlapping or bool(re.fullmatch(r"-.+(<br>-.+)+", text))
            if drop_overlapping and overlapping:
                if expects_continuation:  # Invalidate the whole sequence
                    delete_mask[start_index:i + 1] = True
                elif text.endswith("-"):  # Propagate invalidation forward
                    start_index = i
                    expects_continuation = True
                    delete_mask[i] = True
                else:
                    delete_mask[i] = True
                continue
        overlapping = False  # Stop propagation

        expects_continuation = text.endswith("-")

        if is_continuation:
            string += "<br>" + text
            if not expects_continuation:
                data.iloc[start_index, 5] = string
                data.iloc[start_index, 4] = data.iloc[i, 4]
                delete_mask[start_index + 1: i + 1] = True
                string = None
        elif expects_continuation:
            start_index = i
            string = text
        else:
            start_index = None
            string = None

    data.drop(data[delete_mask].index, inplace=True)

    # Continued word, e.g. `Det er viktig å treffe folk som har mer for-<br>nuftige interesser enn de gamle kompisene.`
    data.text = data.text.str.replace(r"-<br>(?!-)", "", regex=True)

    # This has very inconsistent usage (either two people talking or word continuation)
    data.drop(data[data.text.str.contains("-<br>-")].index)

    data.text = data.text.str.replace(r"^-|\-?<br>-?", " ", regex=True)
    data.text = data.text.str.strip().replace("  +", " ", regex=True)
    data.drop(data[data.text.isna() | (data.text.str.len() == 0)].index, inplace=True)

    modified = pd.DataFrame({"old_text": old_text[data.index], "new_text": data.text})
    deleted = old_text.drop(data.index)

    return modified, deleted


def remove_italics(data: pd.DataFrame):
    """
    Italics are used liberally to denote things like emphasis, narrators, voices from phones etc.
    Generally they are spoken, and should thus be included.
    One special case is parallel translations from tertiary language to Norwegian and Sami,
     with one language italicized and the other not, on separate lines.
    """
    cond = data.text.str.fullmatch(r"((^|<br>)[^<>]+)+(<br><i>[^<>]+<\/i>)+")
    dropped = data.text[cond]
    data.drop(data[cond].index, inplace=True)
    old_text = data.text
    data.text = data.text.str.replace("</?i>", " ", regex=True).str.strip()
    cond = old_text != data.text
    modified = pd.DataFrame({"old_text": old_text[cond], "new_text": data.text[cond]})
    return dropped, modified


def find_simultaneous(data: pd.DataFrame):
    simultaneous = data[data.text.str.lower().str.contains("opptak av simultanteksting")]
    program_ids = simultaneous.program_id.unique()

    return program_ids


def main(args):
    pd.set_option("display.max_rows", None)
    ocr_doc = 1

    # Invoke logging
    log_name = os.path.basename(args.input_file).replace(".json", "")
    log_name = log_name + ".log"

    # Create directories if they do not exist
    if not os.path.exists(args.output_folder + "/log"):
        os.makedirs(args.output_folder + "/log")

    handler = logging.FileHandler(
        filename=os.path.join(args.output_folder, "log/", log_name),
        mode='w'
    )
    formatter = logging.Formatter('%(asctime)s %(message)s')
    handler.setFormatter(formatter)
    handler.setLevel({"DEBUG": logging.DEBUG, "INFO": logging.INFO}[args.log_level])
    logger.addHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    config = read_config(args.output_folder + "/config.json")

    print(f'*** Starting to process: {args.input_file}')
    data: pd.DataFrame = load_json(args.input_file)

    logger.info(f'***  Data loaded. {len(data)} subtitles. ({exec_time()})')
    print(f'Log written to {os.path.join(args.output_folder, "log/", log_name)}. ({exec_time()})')

    # Set number of characters in an subtitle
    # Add this to the frame since we will use it later for sorting
    if len(data) > 0:
        data['doc_length'] = data["text"].parallel_apply(len).groupby(data['id']).transform(sum)

    # Create columns if they do not exist
    # Probably not needed for subtitles but I leave the structure just in case
    # if 'publish_date' not in data:
    #    data['publish_date'] = publish_date

    # Fix possible NaN is mixing datasets
    # data['document_word_confidence'] = data['document_word_confidence'].fillna(1.0)
    # data['confidence'] = data['confidence'].fillna(1.0)
    # data['publish_date'] = data['publish_date'].fillna(publish_date)

    # Fix unicode
    if config['normalise_unicode']:
        data['text'] = data['text'].parallel_apply(normalise_unicode)
        logger.info(f'***  Normalised unicode. Removed double spaces. Trimmed string. ({exec_time()})')

    # Add hash
    #
    # data['hash'] = data['text'].parallel_apply(lambda x: get_hash(x))
    # logger.info(f'***  Added hash. ({exec_time()})')
    # print(f'***  Added hash. ({exec_time()})')

    # Example filter
    # Filter for paragraph confidence
    # if ocr_doc:
    #     cond = data['confidence'].astype(float) >= config['min_confidence_paragraph']
    #     logger.debug(f'\n\n*** The following text was deleted because paragraph confidence was too low:\n {data[~cond]["text"]}')
    #     data = data[cond]
    #     logger.info(f'***  Completed filtering paragraph confidence. Valid posts = {len(data)}. ({exec_time()})')
    #     print(f'***  Completed filtering paragraph confidence. Valid posts = {len(data)}. ({exec_time()})')
    # else:
    #     print(f'***  Skipped paragraph confidence evaluation since this is not an ocr document. ({exec_time()})')

    # #Number of alpha words in subtitle
    # if len(data)>0:
    #     cond = data['text'].parallel_apply(lambda x: count_alphawords(x)) >= config['min_alphawords_paragraph']
    #     logger.debug(f'\n\n*** The following text was deleted because too few alpha words: \n{data[~cond]["text"]}')
    #     data = data[cond]
    # logger.info(f'***  Completed filtering min alpha words. Valid posts = {len(data)}. ({exec_time()})')
    # print(f'***  Completed filtering min alpha words. Valid posts = {len(data)}. ({exec_time()})')

    # Numbers of words in subtitle
    # if len(data)>0:
    #     cond = data['text'].parallel_apply(lambda x: count_words(x)) >= config['min_words_paragraph']
    #     logger.debug(f'\n\n*** The following text was deleted because it had too few words: \n{data[~cond]["text"]}')
    #     data = data[cond]
    # logger.info(f'***  Completed filtering min words. Valid posts = {len(data)}. ({exec_time()})')
    # print(f'***  Completed filtering min words. Valid posts = {len(data)}. ({exec_time()})')

    # Minimum length of subtitle
    if config['min_length_subtitle']:
        cond = data['doc_length'] >= config['min_length_subtitle']
        logger.debug(f'\n\n*** The following text was deleted because the article minimum length was too small:'
                     f'\n {data[~cond]["text"]}')
        data = data[cond]
        logger.info(f'***  Completed filtering min length article. Valid posts = {len(data)}. ({exec_time()})')

    # Remove paragraphs with curly brackets
    if config['drop_subtitles_with_curly_brackets']:
        cond = data['text'].str.contains('\\{')
        logger.debug(f'\n\n*** The following text was deleted because it contained left curly brackets:'
                     f'\n {data[cond]["text"]}')
        data = data[~cond]
        cond = data['text'].str.contains('\\}')
        logger.debug(f'\n\n*** The following text was deleted because it contained right curly brackets:'
                     f'\n {data[cond]["text"]}')
        data = data[~cond]
        logger.info(f'***  Completed filtering out subtitles with curly brackets. '
                    f'Valid subtitles = {len(data)}. ({exec_time()})')

    # Filter out paragraphs with encoding errors
    if config['drop_subtitles_with_encoding_errors']:
        cond = data['text'].str.contains('�')
        data = data[~cond]
        logger.info(f'***  Filtered out encoding errors. The length is now {len(data)}. ({exec_time()})')

    simultaneous = find_simultaneous(data)
    if config["simultaneous_subtitles"] == "detect":
        data["is_simultaneous"] = data.program_id.isin(simultaneous)
        logger.info(f'***  {data.is_simultaneous.sum()} rows were marked as simultaneous texting. ({exec_time()})')
    elif config["simultaneous_subtitles"] == "delete":
        cond = data.program_id.isin(simultaneous)
        logger.debug(f'\n\n*** The following program ids were deleted because they had simultaneous texting:'
                     f'\n {simultaneous}')
        data = data[~cond]
        logger.info(f'***  Filtered out simultaneous texting. The length is now {len(data)}. ({exec_time()})')

    add_task_field(data)
    if config['task']:
        cond = data['task'] == config['task']
        logger.debug(f'\n\n*** The following text was deleted because it was not the correct task:'
                     f'\n {data[~cond][["text", "task"]]}')
        data = data[cond]
        logger.info(f'***  Filtered out tasks. The length is now {len(data)}. ({exec_time()})')

    if config['drop_italics']:
        dropped, modified = remove_italics(data)
        logger.debug(f'\n\n*** The following text was deleted because it is suspected to be bilingual:'
                     f'\n {dropped}')
        logger.debug(f'\n\n*** The following text was modified because it contained italics text:'
                     f'\n {modified}')
        logger.info(f'***  Filtered out italics. The length is now {len(data)}. ({exec_time()})')

    if config['drop_inaudible']:
        modified = remove_inaudible(data)
        logger.debug(f'\n\n*** The following text was modified because it contained inaudible text:'
                     f'\n {modified}')
        logger.info(f'***  Filtered out encoding errors. The length is now {len(data)}. ({exec_time()})')

    if config['drop_invalid_durations']:
        cond = is_invalid_duration(data)
        logger.debug(f'\n\n*** The following text was modified because the speaking rate is too fast or too slow:'
                     f'\n {data[cond]}')
        data = data[~cond]
        logger.info(f'***  Filtered out too fast and too slow speaking rates. '
                    f'The length is now {len(data)}. ({exec_time()})')

    if config['remove_splits']:
        modified, deleted = remove_splits(data, drop_overlapping=config['drop_overlapping'])
        logger.debug(f'\n\n*** The following text was modified because of text continuation or speaker overlap:'
                     f'\n {modified}')
        logger.debug(f'\n\n*** The following text was deleted because of text continuation or speaker overlap:'
                     f'\n {deleted}')
        logger.info(f'***  Filtered out text continuation and/or speaker overlap. '
                    f'The length is now {len(data)}. ({exec_time()})')

    # TODO filter out `CPossible`

    # Remove duplicates
    # if len(data)>0:
    #     data.sort_values(by=['doc_length','paragraph_id'], inplace=True, ascending=[False,True])
    #     data.drop_duplicates(subset="hash",inplace=True,keep='first')
    # logger.info(f'***  Finished deduplicating. Final valid posts: {len(data)}. ({exec_time()})')
    # print(f'***  Finished deduplicating. Final valid posts: {len(data)}. ({exec_time()})')

    # Minimise the size of the jsonl
    # if config['minimise_jsonl'] and len(data)>0:
    #     valid_columns = ['id','doc_type','publish_year','doc_length','paragraph_id','hash','text']
    #     data.drop(columns=[col for col in data if col not in valid_columns], inplace=True)
    #     logger.info(f'***  Minimised the dataframe. ({exec_time()})')
    #     print(f'***  Minimised the dataframe. ({exec_time()})')

    # Tidy up the file and sort it
    # data['publish_year'] = data['publish_year'].astype(int)
    # data['paragraph_id'] = data['paragraph_id'].astype(int)
    # if len(data)>0:
    #     data.sort_values(['doc_length', 'paragraph_id'], ascending=[False, True], inplace=True)
    # logger.info(f'***  Fixed data type and sorted the dataframe. ({exec_time()})')
    # print(f'***  Fixed data type and sorted the dataframe. ({exec_time()})')

    # Save it as jsonl
    output_filename = os.path.join(args.output_folder, os.path.basename(args.input_file))
    save_json(data, output_filename)
    logger.info(
        f'*** Finished processing file. Result has {len(data)} posts. Total length is {round(data["duration"].sum()/1000/60/60,1)} hours. Result is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}. ({exec_time()})')
    print(
        f'*** Finished processing file. Result has {len(data)} posts. \nTotal length is {round(data["duration"].sum()/1000/60/60,1)} hours. \nResult is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}. ({exec_time()})')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to input file.')
    parser.add_argument('--output_folder', required=True, help='Path to output folder.')
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
