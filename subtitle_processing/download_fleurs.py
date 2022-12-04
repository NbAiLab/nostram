from datasets import load_dataset
import os
import argparse
from pydub import AudioSegment

def read_audio_file(example):
    with open(example["audio"]["path"], "rb") as f:
        return {"audio": {"bytes": f.read()}}

def main(args):
    if not os.path.exists(args.output_folder+"/audio"):
            os.makedirs(args.output_folder+"/audio") 
    
    norwegian_fleurs  = load_dataset("google/fleurs", "nb_no")


    #norwegian_fleurs = norwegian_fleurs.map(read_audio_file)
    #ds.save_to_disk(args.output_folder+"/audio")
    
    #Extract and save all files as mp3
    for split in norwegian_fleurs:
        for item in norwegian_fleurs[split]:
            AudioSegment.from_wav(item['path']).export(args.output_folder+"/audio/fleurs_"+os.path.basename(item['path']).replace(".wav",".mp3"), format="mp3")
    
    # Delete the audio column wsince it is causing errors
    norwegian_fleurs = norwegian_fleurs.map(remove_columns=["audio"])
    
    for split,dataset in norwegian_fleurs.items():
            dataset.to_json(f"{args.output_folder}/norwegian_fleurs-{split}.json")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_folder', required=True, help='Path to output folder.')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    main(args)
