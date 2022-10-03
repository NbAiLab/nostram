
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
    result = pd.DataFrame()

    for index, row in data.iterrows():
        #print(f"\n{row['episode_id']} - {row['serie_title']} - {row['title']}")
        #print(f"https://tv.nrk.no/se?v={row['episode_id']}")
        audio = os.path.join(args.directory,'audio/',row['episode_id']+".mp4")
        vtt_ttv = os.path.join(args.directory,'vtt_ttv/',row['episode_id']+ ".vtt")
        vtt_nor = os.path.join(args.directory,'vtt_nor/',row['episode_id']+ ".vtt")
        manifest = os.path.join(args.directory,'manifest/',row['episode_id']+ "_manifest.json") 
        
        i = 0
        status = {}
        for name,f in zip(["audio","vtt_ttv","vtt_nor"],[audio,vtt_ttv,vtt_nor, manifest]):
            i += 1
            
            if os.path.isfile(f):
                #print(f"{name} is OK")
                status[name] = "OK"

            else:
                #print(f"No {name}")
                status[name] = "-"
        
        #Check Manifest file and count references
        
        try:
            f = open(manifest, encoding = 'utf-8')
            manifest_json = json.load(f)
            vtt_ref = manifest_json['playable'].get('subtitles','')

            #print(f"Manifest subtitles: {vtt_ref}")
            status['manifest'] = str(len(vtt_ref))
        except:
            #print(f"No manifest")
            status['manifest'] = "-"
        finally:
            f.close()

        # Add the row to the dataset
        this_result = pd.DataFrame({'SerieTitle': [row['serie_title']],'Title' : [row['title']],'Play' : [f"[{row['episode_id']}](https://tv.nrk.no/se?v={row['episode_id']})"],'Audio': [status['audio']],'VTT-TTV' : [status['vtt_ttv']],'VTT-NOR' : [status['vtt_nor']],'Manifest': [status['manifest']]})
        result = pd.concat([result, this_result], ignore_index = True, axis = 0)
    

    with open('debug_stats.md', 'w') as f:
        f.write(result.to_markdown(index=False))
    
    print(result[0:10])
    print("The complete file is written to debug_stats.md")

def parse_args():
    # Parse commandline
    parser = ArgumentParser()
    parser.add_argument("-d", "--directory", dest="directory", help="Directory to Json-lines file to analyse", required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
