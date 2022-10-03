
import json
import sys
import os
import jsonlines
from argparse import ArgumentParser
import pandas as pd
import glob
import urllib.request

###################################################
# Debug script made for detecting missing subtitles
###################################################

def main(args):
    target_file = os.path.join(args.directory,'process_list/','tv.json')
    
    data = pd.read_json(target_file, lines=True) # read data frame from json file
    
    ## Lets just work with part of the data
    data = data.sample(n=100)

    for index, row in data.iterrows():
        print(f"\n{row['episode_id']} - {row['serie_title']} - {row['title']}")
        print(f"https://tv.nrk.no/se?v={row['episode_id']}")
        audio = os.path.join(args.directory,'audio/',row['episode_id']+".mp4")
        vtt_ttv = os.path.join(args.directory,'vtt_ttv/',row['episode_id']+ ".vtt")
        vtt_nor = os.path.join(args.directory,'vtt_nor/',row['episode_id']+ ".vtt")
        manifest = os.path.join(args.directory,'manifest/',row['episode_id']+ "_manifest.json") 
        breakpoint()

        manifest_json = json.loads(manifest)
        
        for n,f in zip(["audio","manifest","vtt_ttv","vtt_nor"],[audio,manifest,vtt_ttv,vtt_nor]):
            if os.path.isfile(f):
                print(f"{n} is OK")
            else:
                print(f"No {n}")
        



def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
