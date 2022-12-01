######################################################################
### Cleans up subtitle jsonl files
#####################################################################

import os, sys
import json
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


def main(args):
    pd.set_option("display.max_rows", None)
    ocr_doc = 1

    # Invoke logging
    log_name = os.path.basename(args.input_file).replace(".json", "")
    log_name = log_name + ".log"

    # Create directories if they do not exist
    if not os.path.exists(args.output_folder + "/log"):
        os.makedirs(args.output_folder + "/log")

    logging.basicConfig(filename=os.path.join(args.output_folder, "log/", log_name), format='%(asctime)s %(message)s',
                        filemode='w')
    config = read_config(args.output_folder + "/config.json")

    print(f'*** Starting to process: {args.input_file}')
    data = load_json(args.input_file)

    logger.info(f'***  Data loaded. {len(data)} subtitles. ({exec_time()})')
    print(
        f'*** Data loaded with {len(data)} subtitles. Log written to {os.path.join(args.output_folder, "log/", log_name)}. ({exec_time()})')

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
        print(f'***  Normalised unicode. Removed double spaces. Trimmed string.({exec_time()})')

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
        logger.debug(
            f'\n\n*** The following text was deleted because the article minimum lenght was too small:\n {data[~cond]["text"]}')
        data = data[cond]
        logger.info(f'***  Completed filtering min length article. Valid posts = {len(data)}. ({exec_time()})')
        print(f'***  Completed filtering min length article. Valid posts = {len(data)}. ({exec_time()})')

    # Remove paragraphs with curly brackets
    if config['drop_subtitles_with_curly_brackets']:
        cond = data['text'].str.contains('\\{')
        logger.debug(
            f'\n\n*** The following text was deleted because it contained left curly brackets:\n {data[cond]["text"]}')
        data = data[~cond]
        cond = data['text'].str.contains('\\}')
        logger.debug(
            f'\n\n*** The following text was deleted because it contained right curly brackets:\n {data[cond]["text"]}')
        data = data[~cond]
        print(
            f'***  Completed filtering out subtitles with curly brackets. Valid subtitles = {len(data)}. ({exec_time()})')

    # Filter out paragraphs with encoding errors
    if config['drop_subtitles_with_encoding_errors']:
        cond = data['text'].str.contains('�')
        data = data[~cond]
        logger.info(f'***  Filtered out encoding errors. The length is now {len(data)}. ({exec_time()})')
        print(f'***  Filtered out encoding errors. The length is now {len(data)}. ({exec_time()})')

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
        f'*** Finished processing file. Result has {len(data)} subtitles. Result is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}. ({exec_time()})')
    print(
        f'*** Finished processing file. Result has {len(data)} posts. Result is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}. ({exec_time()})')


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
