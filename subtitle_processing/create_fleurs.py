######################################################################
### Creates the Fleurs dataset file based on the downloaded set
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
    data["id"] = "fleurs_"+data['path'].apply(os.path.basename).str.replace(".wav","",regex=False)
    data["group_id"] = None
    data["source"] = "fleurs"
    data["audio"] = "fleurs_"+data['path'].apply(os.path.basename).str.replace(".wav",".mp3",regex=False)
    data["audio_duration"] = args.mp3_folder+"/"+data["audio"]
    data["audio_duration"] = data["audio_duration"].parallel_apply(calculate_duration)
    data["text"] = data["raw_transcription"]
    data["text_language"] = "no"
    data["audio_language"] = "no"
    data["previous_text"] = None
    data["timestamped_text"] = None
    data["translated_text_en"] = None
    data["translated_text_es"] = None
    data["translated_text_nn"] = None
    data["translated_text_no"] = None
    data["verbosity_level"] = None
    data["wav2vec_wer"] = None
    data["whisper_wer"] = None
    
    #Drop some stuff we dont need any more
    data = data.drop(['num_samples','path','audio','transcription','raw_transcription','gender','lang_id','language','lang_group_id'], axis=1)
    
    
    
    #Save it as jsonl
    output_filename = os.path.join(args.output_folder, os.path.basename(args.input_file))
    save_json(data, output_filename)
    print(output_filename)
    print(f'*** Finished processing file. Result has {len(data)} posts. \nTotal length is {round(data["audio_duration"].sum()/1000/60/60,1)} hours. \nResult is written to {os.path.join(args.output_folder, os.path.basename(args.input_file))}')


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




