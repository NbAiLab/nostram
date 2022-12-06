######################################################################
### Creates the NST dataset file
#####################################################################

import os
import pandas as pd
from slugify import slugify
from datetime import datetime
import argparse
from pandarallel import pandarallel
pandarallel.initialize(use_memory_fs=True)

def load_json(jsonline):
    data = pd.read_json(jsonline, lines=True)
    print(f'***  Json parsed with {len(data)} lines.')

    return data


def save_json(data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        data.to_json(file, orient='records', lines=True, force_ascii=False)
    print(f'Saved jsonl as "{filename}"')


def main(args):
    import librosa
    pd.set_option("display.max_rows", None)

    print(f'*** Starting to process: {args.input_file}')
    def calculate_duration(mp3):
        duration = int(librosa.get_duration(filename=mp3)*1000)
        return duration
    
    data = load_json(args.input_file)
    # Keep only rows that are sentences
    print(f"Initial length is {len(data)}.")
    data = data[data["type"].str.contains("IS")==True]
    print(f"After checking if it is a sentence: {len(data)}")
    data = data[data["text"].str.contains("\\", regex=False)==False]
    print(f"After checking for escape: {len(data)}")
    data = data[data["text"].str.contains(")", regex=False)==False]
    print(f"After checking for paranthesis: {len(data)}")
    data = data[(data["text"].str.contains(".", regex=False)==True) | (data["text"].str.contains("?", regex=False)==True)]
    print(f"After checking that the sentence has punctation or question mark: {len(data)}")
    
    data["id"] = "NST_"+data["pid"].astype(str)+"_"+data["file"].str.replace(".wav","", regex=False)
    data["program_id"] = "NST"+data["pid"].astype(str)
    data["end_time"] = ""
    data["end_time"] = ""
    data["medium"] = "NST"
    data["source"] = "NST"
    data["category"] = "stemmeprøver"
    data["title"] = "NST"
    data["subtitle"] = data["pid"].astype(str)
    data["audio"] = data["pid"].astype(str)+"_"+data["file"].str.replace(".wav",".mp3", regex=False)
    data["duration"] = args.mp3_folder+data["audio"]
    data["duration"] = data["duration"].parallel_apply(calculate_duration)
    data["text"] = data["text"]
    data["lang_text"] = "nob"
    data["lang_text_confidence"] = 1
    data["lang_voice"] = "nor"
    data["lang_voice_confidence"] = 1
    data["region_of_birth"] = data["Region_of_Birth"]
    data["nst_type"] = data["type"]
    
    
    #Drop some stuff we dont need any more
    data = data.drop(['pid', 'Age','Region_of_Birth','Region_of_Youth','Remarks','Sex','Speaker_ID','Directory','Imported_sheet_file','Number_of_recordings','RecDate','RecTime','Record_duration','Record_session','Sheet_number','ANSI_Codepage','Board','ByteFormat','Channels','CharacterSet','Coding','DOS_Codepage','Delimiter','Frequency','Memo','Script','Version','DST','NOI','QUA','SND','SPC','UTT','file','t0','t1','t2'], axis=1)
    
    
    
    #Save it as jsonl
    output_filename = os.path.join(args.output_folder, os.path.basename(args.input_file))
    save_json(data, output_filename)
    print(f'*** Finished processing file. Result has {len(data)} posts. \nTotal length is {round(data["duration"].sum()/1000/60/60,1)} hours. \nResult is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to input file.')
    parser.add_argument('--mp3_folder', required=True, help='Path to mp3 folder (for calculating duration).')
    parser.add_argument('--output_folder', required=True, help='Path to output folder.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)



