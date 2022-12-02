######################################################################
### Creates the NPSC Bokmål dataset file
#####################################################################

import os
import pandas as pd
from slugify import slugify
from datetime import datetime
import argparse

def load_json(jsonline):
    data = pd.read_json(jsonline, lines=True)
    print(f'***  Json parsed with {len(data)} lines.')

    return data


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        data.to_json(file, orient='records', lines=True, force_ascii=False)
    print(f'Saved jsonl as "{filename}"')


def main(args):
    pd.set_option("display.max_rows", None)

    print(f'*** Starting to process: {args.input_file}')
    data = load_json(args.input_file)

    data["id"] = "NPSC"+data["meeting_date"].astype(str)+"_"+data["start_time"].astype(str)+"_"+data["end_time"].astype(str)
    data["duration"] = data["end_time"]-data["start_time"]
    data["program_id"] = "NPSC"+data["meeting_date"].astype(str)
    data["start_time"] = data["start_time"]
    data["end_time"] = data["end_time"]
    data["medium"] = "Stortinget"
    data["source"] = "NPSC"
    data["category"] = "politikk"
    data["title"] = "NPSC "
    data["subtitle"] = data["meeting_date"].astype(str)
    data["audio"] = data["meeting_date"].astype(str)+"-"+data["sentence_id"].astype(str)+".mp3"
    data["text"] = data["sentence_nob"]
    data["lang_text"] = "nob"
    data["lang_voice"] = "nor"
    
    #Drop some stuff we dont need any more
    data = data.drop(['data_split', 'sentence_id','sentence_order','speaker_id','speaker_name','sentence_text','sentence_language_code','normsentence_text','transsentence_text','translated','transcriber_id','reviewer_id','total_duration','path','sentence_nob','sentence_nno'], axis=1)
    
    #Save it as jsonl
    output_filename = os.path.join(args.output_folder, os.path.basename(args.input_file).replace(".json","_nob.json"))
    save_json(data, output_filename)
    print(f'*** Finished processing file. \nResult has {len(data)} posts. \nTotal length is {round(data["duration"].sum()/1000/60/60,1)} hours. \nResult is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to input file.')
    parser.add_argument('--output_folder', required=True, help='Path to output folder.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)



